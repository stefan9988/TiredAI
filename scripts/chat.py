"""Chat with the tire assistant in the terminal; answers are streamed as they are generated.

The conversation is remembered until you exit. Type /new to start a new conversation, or press
Ctrl+D (or type /exit) to quit. Pass a message as an argument to ask a single question.

Usage:
    uv run python scripts/chat.py ["What does UTQG mean?"]
"""

import argparse
import sys
import uuid

from tiredai.agent import build_agent, stream_reply
from tiredai.config import Settings
from tiredai.embeddings import build_encoder
from tiredai.search import catalog_tools
from tiredai.vectorstore import connect


def ask(agent, message: str, thread_id: str) -> None:
    print("assistant> ", end="", flush=True)
    try:
        for text in stream_reply(agent, message, thread_id):
            print(text, end="", flush=True)
    except Exception as exc:  # show provider errors (rate limits, auth) without a traceback
        print(f"\n[error] {type(exc).__name__}: {exc}", file=sys.stderr)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("message", nargs="?", help="ask one question and exit")
    args = parser.parse_args()

    settings = Settings.load()
    try:
        client = connect(settings)
    except Exception as exc:  # e.g. the API server has the local store open
        client = None
        print(f"[warning] Catalog unavailable, answering without product search: {exc}", file=sys.stderr)
    try:
        tools = catalog_tools(
            client,
            settings.qdrant_collection,
            lambda: build_encoder(settings),
            max_results=settings.agent.max_search_results,
        )
        if client is not None and not tools:
            print("[warning] No index found; run scripts/build_index.py for product search.", file=sys.stderr)
        agent = build_agent(settings, tools=tools)
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(str(exc))

    thread_id = str(uuid.uuid4())
    if args.message:
        ask(agent, args.message, thread_id)
        return

    print(f"TiredAI ({settings.llm.model}). /new starts a new conversation, Ctrl+D quits.")
    while True:
        try:
            message = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if message in ("/exit", "/quit"):
            return
        if message == "/new":
            thread_id = str(uuid.uuid4())
            print("Started a new conversation.")
        elif message:
            ask(agent, message, thread_id)


if __name__ == "__main__":
    main()
