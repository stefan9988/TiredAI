"""Chat with the tire assistant in the terminal; answers are streamed as they are generated.

The conversation is remembered until you exit. Type /new to start a new conversation, or press
Ctrl+D (or type /exit) to quit. Pass a message as an argument to ask a single question. The guardrail
checks every message first, like on the chat page; --no-guardrail sends them straight to the chat model.
--no-web-search takes the vehicle lookup (a web search for a vehicle's tire sizes) away.

Usage:
    uv run python scripts/chat.py [--no-guardrail] [--no-web-search] ["What does UTQG mean?"]
"""

import argparse
import sys
import uuid

from tiredai import tracing
from tiredai.agent import build_agent, has_guardrail, stream_reply, tool_names, trace_metadata
from tiredai.config import Settings
from tiredai.embeddings import build_encoder
from tiredai.search import catalog_tools
from tiredai.vectorstore import connect
from tiredai.vehicles import TOOL_NAME as VEHICLE_TOOL
from tiredai.vehicles import vehicle_tools


def ask(agent, message: str, thread_id: str, metadata: dict, guardrail: bool, web_search: bool = True) -> None:
    print("assistant> ", end="", flush=True)
    try:
        for text in stream_reply(agent, message, thread_id, source="cli", metadata=metadata, guardrail=guardrail,
                                 web_search=web_search):  # fmt: skip
            print(text, end="", flush=True)
    except Exception as exc:  # show provider errors (rate limits, auth) without a traceback
        print(f"\n[error] {type(exc).__name__}: {exc}", file=sys.stderr)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("message", nargs="?", help="ask one question and exit")
    parser.add_argument("--no-guardrail", action="store_true", help="don't check messages with the guardrail")
    parser.add_argument("--no-web-search", action="store_true", help="don't let the agent look vehicles up on the web")
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
        agent = build_agent(settings, tools=tools + vehicle_tools(settings))
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(str(exc))

    tracing.start()
    try:
        guardrail = not args.no_guardrail and has_guardrail(agent)
        web_search = not args.no_web_search and VEHICLE_TOOL in tool_names(agent)
        converse(agent, args.message, trace_metadata(settings), settings.llm.model, guardrail, web_search)
    finally:
        tracing.flush()


def converse(agent, message: str | None, metadata: dict, model: str, guardrail: bool = False, web_search: bool = True) -> None:
    thread_id = str(uuid.uuid4())
    if message:
        ask(agent, message, thread_id, metadata, guardrail, web_search)
        return

    switches = f"guardrail {'on' if guardrail else 'off'}, web search {'on' if web_search else 'off'}"
    print(f"TiredAI ({model}, {switches}). /new starts a new conversation, Ctrl+D quits.")
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
            ask(agent, message, thread_id, metadata, guardrail, web_search)


if __name__ == "__main__":
    main()
