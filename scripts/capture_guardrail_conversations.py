"""Play the conversations of benchmarks/guardrail_cases.yaml through the agent and freeze its answers.

The guardrail benchmark shows Jev the conversation before a case's message, so follow-ups like "what
about the second one?" come after a real answer listing real tires. Each conversation's shopper
messages run through the agent (the .env chat model, on an in-memory index with the .env embedding
model), and the transcript is saved to benchmarks/guardrail_conversations.yaml, which the benchmark
reads as it is. By default only conversations that were never captured, or whose turns changed, are
played; read the new answers before committing them.

Usage:
    uv run python scripts/capture_guardrail_conversations.py [--conversation ID ...] [--all]
"""

import argparse
import asyncio
import sys
import uuid

from tiredai import tracing
from tiredai.agent import build_agent, trace_metadata
from tiredai.benchmarks import experiments as ex
from tiredai.benchmarks.guardrail import capture_conversation, load_captured, load_cases, stale, write_captured
from tiredai.config import Settings
from tiredai.embeddings import EmbeddingError
from tiredai.search import CatalogSearch, make_search_tool

CASES = ex.BENCHMARKS_DIR / "guardrail_cases.yaml"
CAPTURED = ex.BENCHMARKS_DIR / "guardrail_conversations.yaml"
HEADER = ("The conversations of guardrail_cases.yaml as the agent answered them, written by\n"
          "# scripts/capture_guardrail_conversations.py. Don't edit by hand; capture again after changing a conversation.")  # fmt: skip


async def play(conversations: list, agent, settings: Settings) -> dict[str, dict]:
    """Each conversation in a new thread, one after another (in one event loop, which the chat model's client keeps)."""
    played = {}
    for conversation in conversations:
        print(f"Playing {conversation.id} ({len(conversation.turns)} turns) with {settings.llm.model} ...")
        messages = await capture_conversation(agent, conversation.turns, f"benchmark-{uuid.uuid4()}", trace_metadata(settings))
        played[conversation.id] = {"llm_model": settings.llm.model, "captured_at": ex.timestamp(), "transcript": messages}
    return played


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conversation", action="append", metavar="ID", help="capture these again (default: the stale ones)")
    parser.add_argument("--all", action="store_true", help="capture every conversation again")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    settings = Settings.load()

    cases = load_cases(CASES)
    captured = load_captured(CAPTURED)
    known = {c.id: c for c in cases.conversations}
    if unknown := set(args.conversation or []) - set(known):
        sys.exit(f"Unknown conversation ids: {', '.join(sorted(unknown))}")
    wanted = list(known) if args.all else args.conversation or stale(cases.conversations, captured)
    removed = set(captured) - set(known)
    if not wanted and not removed:
        print("Every conversation is captured with its current turns.")
        return
    if wanted and not settings.openrouter_api_key:
        sys.exit("Capturing calls the chat model: set OPENROUTER_API_KEY in .env")

    if wanted:
        print(f"Indexing the catalog with {ex.embedding_label(settings)} ...")
        try:
            client, encoder = ex.memory_index(settings)
        except (EmbeddingError, ValueError) as exc:
            sys.exit(f"Indexing failed: {exc}")
        tools = [make_search_tool(CatalogSearch(client, ex.COLLECTION, encoder, max_results=settings.agent.max_search_results))]
        captured |= asyncio.run(play([known[key] for key in wanted], build_agent(settings, tools=tools), settings))
        client.close()
        tracing.flush()

    ordered = {key: captured[key] for key in known if key in captured}  # the order of the cases file, removed ones dropped
    write_captured(CAPTURED, ordered, HEADER)
    print(f"Saved {CAPTURED.relative_to(ex.BENCHMARKS_DIR.parent)}: {len(wanted)} captured, {len(removed)} removed.")


if __name__ == "__main__":
    main()
