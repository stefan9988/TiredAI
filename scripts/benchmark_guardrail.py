"""Guardrail benchmark: how well Jev tells the shopper messages the agent should handle from the ones to stop.

It measures the guardrail the app runs (src/tiredai/guardrail.py: the same policy, questions and block
score), with GUARDRAIL_MODEL and GUARDRAIL_THRESHOLD from .env unless --model and --threshold say otherwise.

Every case of benchmarks/guardrail_cases.yaml (a message, optionally after a captured conversation from
benchmarks/guardrail_conversations.yaml) is sent to Jev's Decisions API with the policy and three yes/no
questions, once per variant of how much conversation Jev sees: message (none), recent (the agent's
AGENT_HISTORY_MESSAGES window) and full. See src/tiredai/benchmarks/guardrail.py for the questions and
scores. The agent is not involved: this only measures the guardrail.

With LANGFUSE_* keys set, the cases are synced to the Langfuse dataset tiredai-guardrail and each model
and variant becomes one experiment run there. The summary, the accuracy per tag and the misjudged cases
are printed and saved to benchmarks/results/ either way.

Usage:
    uv run python scripts/benchmark_guardrail.py [--model MODEL ...] [--variant message|recent|full ...]
        [--threshold T] [--case ID ...] [--concurrency N] [--local] [--check]
"""

import argparse
import os
import sys

from tiredai import tracing
from tiredai.benchmarks import experiments as ex
from tiredai.benchmarks.guardrail import (
    DATASET,
    SCORES,
    SETUP_VERSION,
    VARIANTS,
    GuardTask,
    check_cases,
    dataset_cases,
    guard_evaluator,
    load_captured,
    load_cases,
    misjudged,
    run_details,
    tag_table,
)
from tiredai.config import Settings
from tiredai.guardrail import JevClient


def main() -> None:
    settings = Settings.load()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", metavar="MODEL",
                        help=f"Decisions model; repeat to compare (default: GUARDRAIL_MODEL, {settings.guardrail.model})")
    parser.add_argument("--variant", action="append", choices=VARIANTS, help="default: all three")
    parser.add_argument("--threshold", type=float, default=settings.guardrail.threshold,
                        help=f"block at this block score or above (default: GUARDRAIL_THRESHOLD, {settings.guardrail.threshold})")
    parser.add_argument("--case", action="append", metavar="ID", help="run only these cases")
    parser.add_argument("--concurrency", type=int, default=8, help="requests at once (default: 8)")
    parser.add_argument("--local", action="store_true", help="send nothing to Langfuse, even with keys set")
    parser.add_argument("--check", action="store_true", help="only validate the cases and the captured conversations")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    if args.local:
        os.environ.update(LANGFUSE_PUBLIC_KEY="", LANGFUSE_SECRET_KEY="")  # .env doesn't override set variables
    if not 0 < args.threshold <= 1:
        sys.exit("--threshold must be above 0 and at most 1")

    cases = load_cases(ex.BENCHMARKS_DIR / "guardrail_cases.yaml")
    captured = load_captured(ex.BENCHMARKS_DIR / "guardrail_conversations.yaml")
    if problems := check_cases(cases, captured):
        sys.exit("The cases can't run:\n" + "\n".join(f"  {p}" for p in problems))
    allow = sum(c.expect == "allow" for c in cases.cases)
    print(f"{len(cases.cases)} cases ({allow} allow, {len(cases.cases) - allow} block; "
          f"{sum(bool(c.conversation) for c in cases.cases)} after a conversation) are ready.")  # fmt: skip
    if args.check:
        return
    selected = cases.cases
    if args.case:
        if unknown := set(args.case) - {c.id for c in cases.cases}:
            sys.exit(f"Unknown case ids: {', '.join(sorted(unknown))}")
        selected = [c for c in cases.cases if c.id in args.case]
    if not settings.openrouter_api_key:
        sys.exit("The guardrail benchmark calls Jev on OpenRouter: set OPENROUTER_API_KEY in .env")

    langfuse = tracing.client()
    all_cases = dataset_cases(cases, captured)
    if tracing.enabled():
        items = ex.sync_dataset(langfuse, DATASET, "Shopper messages to allow or block (benchmarks/guardrail_cases.yaml)", all_cases)
        wanted = {c.id for c in selected}
        data = [item for item in items if item.metadata["case"] in wanted]
        print(f"Synced to the Langfuse dataset {DATASET}.")
    else:
        wanted = {c.id for c in selected}
        data = ex.local_items([c for c in all_cases if c.id in wanted])
        print("Running locally (--local or no Langfuse keys): nothing is sent to Langfuse.")

    stamp, revision = ex.timestamp(), ex.git_revision()
    history = settings.agent.history_messages
    summaries = []
    for model in args.model or [settings.guardrail.model]:
        jev = JevClient(settings.openrouter_api_key, model)
        for variant in args.variant or VARIANTS:
            label = f"{model} · {variant}"
            print(f"\nRunning {label} on {len(data)} cases ({args.concurrency} at a time) ...")
            result = langfuse.run_experiment(
                name=f"Guardrail: {model}",
                run_name=f"{label} · {stamp}",
                description=f"Decisions model {model}, conversation shown: {variant}, git {revision}",
                data=data,
                task=GuardTask(jev, variant, history),
                evaluators=[guard_evaluator(args.threshold)],
                max_concurrency=args.concurrency,
                metadata={
                    "guard_model": model,
                    "variant": variant,
                    "threshold": str(args.threshold),
                    "history_messages": str(history),
                    "policy": SETUP_VERSION,
                    "git": revision,
                },
            )
            setup = {"guard_model": model, "variant": variant, "threshold": args.threshold}
            summaries.append(ex.summarize(label, result, len(data), run_details(result), setup))
    tracing.flush()

    by_tag, wrong = tag_table(selected, summaries), misjudged(selected, summaries)
    print("\n" + ex.markdown_table(summaries, SCORES) + "\n\nAccuracy per tag:\n" + by_tag + "\n\n" + wrong)
    notes = [
        f"{stamp}, git {revision}, policy and questions {SETUP_VERSION}",
        f"{len(data)} cases; blocked at a block score of {args.threshold} or more; recent = the last {history} chat messages",
        "correct: accuracy; false_block: allow cases that were blocked (lower is better); caught: block cases that "
        "were blocked; auc: ROC AUC of the block score; best_threshold: the one with the best balanced accuracy",
    ]  # fmt: skip
    sections = ["## Accuracy per tag\n\n" + by_tag, "## Misjudged cases\n\n" + wrong]
    md_path, _ = ex.write_report("guardrail", "Guardrail benchmark", summaries, SCORES, notes, sections=sections)
    print(f"\nSaved {md_path.relative_to(ex.BENCHMARKS_DIR.parent)} (and .json with every case).")
    for s in summaries:
        if s.url:
            print(f"  {s.label}: {s.url}")


if __name__ == "__main__":
    main()
