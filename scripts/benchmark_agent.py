"""Agent benchmark: run the conversations in benchmarks/agent_cases.yaml through the agent with each chat model.

Every case is a new conversation played turn by turn through the same streaming path as the chat UI,
with the search tool on an in-memory index built with the chosen embedding model, and the vehicle
lookup answering from the pages captured in benchmarks/vehicle_pages.yaml. Each turn is
scored with deterministic checks against the catalog: intent, retrieval hit@3, filters applied,
constraint correctness, groundedness and answer checks (see src/tiredai/benchmarks/conversations.py).

With LANGFUSE_* keys set, the cases are synced to the Langfuse dataset tiredai-agent and each model
(and repeat) becomes one experiment run there, every conversation a trace with its scores. The
summary is printed and saved to benchmarks/results/ either way. Other LLM_* and AGENT_* settings
come from .env and are the same for every model.

Usage:
    uv run python scripts/benchmark_agent.py [--llm-model MODEL ...] [--embedding-model [PROVIDER:]MODEL]
        [--repeat N] [--case ID ...] [--concurrency N] [--local] [--check]
"""

import argparse
import os
import json
import sys
import time
from collections import defaultdict

from tiredai import tracing
from tiredai.agent import build_agent, load_system_prompt, trace_metadata
from tiredai.benchmarks import experiments as ex
from tiredai.benchmarks.cases import DATASET, check_cases, load_cases
from tiredai.benchmarks.catalog import Catalog
from tiredai.benchmarks.conversations import SCORES, ConversationTask, conversation_evaluator, turn_seconds
from tiredai.benchmarks.metrics import mean, percentile
from tiredai.config import Settings
from tiredai.embeddings import EmbeddingError, is_free_model
from tiredai.search import CatalogSearch, make_search_tool
from tiredai.vehicles import FrozenPages, VehicleLookup, make_vehicle_tool

LLM_PARAMETERS = ("temperature", "top_p", "max_tokens", "seed", "frequency_penalty", "presence_penalty", "reasoning_effort")


def run_details(result) -> dict:
    outputs = [r.output for r in result.item_results if isinstance(r.output, dict)]
    seconds = [s for o in outputs for s in turn_seconds(o)]
    return {
        "turn_errors": sum(bool(t["error"]) for o in outputs for t in o["turns"]),
        "seconds_per_turn": mean(seconds),
        "p95_seconds": percentile(seconds, 0.95),
        "model_calls": sum(o["model_calls"] for o in outputs),
        "input_tokens": sum(o["input_tokens"] for o in outputs),
        "output_tokens": sum(o["output_tokens"] for o in outputs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm-model", action="append", metavar="MODEL", help="OpenRouter model id; repeat to compare models")
    parser.add_argument("--embedding-model", metavar="[PROVIDER:]MODEL", help="default: the .env embedding model")
    parser.add_argument("--repeat", type=int, default=1, help="runs per model, to see how consistent it is (default: 1)")
    parser.add_argument("--case", action="append", metavar="ID", help="run only these cases")
    parser.add_argument("--concurrency", type=int, default=4, help="conversations at once (default: 4; :free models run one)")
    parser.add_argument("--local", action="store_true", help="send nothing to Langfuse, even with keys set")
    parser.add_argument("--check", action="store_true", help="only validate the cases against the catalog")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up in logs, not only at the end
    if args.local:
        os.environ.update(LANGFUSE_PUBLIC_KEY="", LANGFUSE_SECRET_KEY="")  # .env doesn't override set variables
    settings = ex.with_embedding(Settings.load(), args.embedding_model)

    catalog = Catalog.load(settings.processed_data_path)
    cases = load_cases(ex.BENCHMARKS_DIR / "agent_cases.yaml")
    vehicle_pages = FrozenPages.load(ex.BENCHMARKS_DIR / "vehicle_pages.yaml")
    if problems := check_cases(cases, catalog, vehicle_pages):
        sys.exit("The cases don't fit the catalog:\n" + "\n".join(f"  {p}" for p in problems))
    print(f"{len(cases)} cases ({sum(len(c.turns) for c in cases)} turns) fit the catalog.")
    if args.check:
        return
    if args.case:
        if unknown := set(args.case) - {c.id for c in cases}:
            sys.exit(f"Unknown case ids: {', '.join(sorted(unknown))}")
        cases = [c for c in cases if c.id in args.case]
    if not settings.openrouter_api_key:
        sys.exit("The agent benchmark calls the chat model: set OPENROUTER_API_KEY in .env")

    langfuse = tracing.client()
    all_cases = [c.case() for c in load_cases(ex.BENCHMARKS_DIR / "agent_cases.yaml")]
    if tracing.enabled():
        items = ex.sync_dataset(langfuse, DATASET, "Shopper conversations with what each turn must do (benchmarks/agent_cases.yaml)", all_cases)
        wanted = {c.id for c in cases}
        data = [item for item in items if item.metadata["case"] in wanted]
        print(f"Synced to the Langfuse dataset {DATASET}.")
    else:
        data = ex.local_items([c.case() for c in cases])
        print("Running locally (--local or no Langfuse keys): nothing is sent to Langfuse.")

    embedding = ex.embedding_label(settings)
    print(f"Indexing the catalog with {embedding} ...")
    started = time.perf_counter()
    try:
        client, encoder = ex.memory_index(settings, lambda done, total: print(f"  {done:>6,} / {total:,}", end="\r", flush=True))
    except (EmbeddingError, ValueError) as exc:
        sys.exit(f"Indexing failed: {exc}")
    print(f"  indexed in {time.perf_counter() - started:.0f}s")
    tools = [make_search_tool(CatalogSearch(client, ex.COLLECTION, encoder, max_results=settings.agent.max_search_results)),
             make_vehicle_tool(VehicleLookup(vehicle_pages))]  # fmt: skip

    stamp, revision = ex.timestamp(), ex.git_revision()
    prompt = ex.fingerprint(load_system_prompt(settings.llm.system_prompt_path))
    summaries, passes = [], defaultdict(lambda: defaultdict(list))
    for model in args.llm_model or [settings.llm.model]:
        model_settings = ex.with_llm(settings, model)
        parameters = {p: getattr(model_settings.llm, p) for p in LLM_PARAMETERS if getattr(model_settings.llm, p) is not None}
        concurrency = 1 if is_free_model(model) else args.concurrency
        for repeat in range(1, args.repeat + 1):
            label = model + (f" #{repeat}" if args.repeat > 1 else "")
            agent = build_agent(model_settings, tools=tools)
            print(f"\nRunning {label} on {len(data)} conversations ({concurrency} at a time) ...")
            result = langfuse.run_experiment(
                name=f"Agent: {model}",
                run_name=f"{label} · {embedding} · {stamp}",
                description=f"LLM {model}, embeddings {embedding}, git {revision}",
                data=data,
                task=ConversationTask(agent, trace_metadata(model_settings)),
                evaluators=[conversation_evaluator(catalog)],
                max_concurrency=concurrency,
                metadata={
                    "llm_model": model,
                    "embedding_model": embedding,
                    "git": revision,
                    "system_prompt": prompt,
                    "llm_parameters": json.dumps(parameters),
                    "history_messages": str(settings.agent.history_messages),
                    "max_tool_calls": str(settings.agent.max_tool_calls),
                    "max_search_results": str(settings.agent.max_search_results),
                    "repeat": f"{repeat}/{args.repeat}",
                },
            )
            setup = {"llm_model": model, "embedding_model": embedding, "repeat": repeat}
            summary = ex.summarize(label, result, len(data), run_details(result), setup)
            summaries.append(summary)
            for case_result in summary.results:
                passes[model][case_result["case"]].append(case_result["scores"].get("passed", 0))
    client.close()
    tracing.flush()

    order = [*SCORES, "passed"]
    print("\n" + ex.markdown_table(summaries, order))
    notes = [f"{stamp}, git {revision}, system prompt {prompt}, embeddings {embedding}",
             f"{len(data)} conversations; LLM parameters from .env: {json.dumps(parameters) if parameters else 'provider defaults'}"]  # fmt: skip
    if args.repeat > 1:
        for model, by_case in passes.items():
            always = sum(len(p) == args.repeat and all(p) for p in by_case.values())
            notes.append(f"{model}: {always}/{len(data)} conversations passed in all {args.repeat} runs")
            print(notes[-1])
    md_path, _ = ex.write_report("agent", "Agent benchmark", summaries, order, notes)
    print(f"\nSaved {md_path.relative_to(ex.BENCHMARKS_DIR.parent)} (and .json with every conversation).")
    for s in summaries:
        if s.url:
            print(f"  {s.label}: {s.url}")


if __name__ == "__main__":
    main()
