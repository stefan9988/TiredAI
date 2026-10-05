"""Langfuse tracing (https://langfuse.com/docs/observability/best-practices).

Each shopper message is one trace, `answer-shopper-message`, and the traces of a conversation are
grouped into a Langfuse session (the conversation id). The trace's input is the shopper's message and
its output the answer. Inside it, the LangChain integration records every model call as a generation
and every search_tires call as a tool; search.py adds the catalog lookup as a retriever, with the
query embedding inside it.

Tracing is on when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set (LANGFUSE_BASE_URL picks the
region or a self-hosted server, LANGFUSE_TRACING_ENVIRONMENT the environment). Without them nothing
is recorded or sent.
"""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langfuse import Langfuse, LangfuseOtelSpanAttributes, is_default_export_span, propagate_attributes
from langfuse.langchain import CallbackHandler
from opentelemetry.sdk.trace import ReadableSpan

TURN = "answer-shopper-message"
SERVICE = "tiredai"
# The client used while tracing is off: it records nothing, and unlike a client without keys it
# doesn't log a warning on every use.
OFF_KEY = "tracing-off"

logger = logging.getLogger(__name__)
_client: Langfuse | None = None
_public_key = OFF_KEY  # the key of _client, which the LangChain handler needs to find it


def worth_exporting(span: ReadableSpan) -> bool:
    """Langfuse's default span filter, without two middleware steps that only repeat what the trace shows.

    The tool-call limit's step runs after every model call and only updates counters, unless calls
    were over the limit: then its output holds the error results, which stay in the trace. The
    guardrail's step (agent.GUARDRAIL_NODE) runs at the start of every turn, also with the guardrail
    off; its check is the check-message observation, and the "agent" in its name would make Langfuse
    type it as an agent.
    """
    if not is_default_export_span(span):
        return False
    if span.name == "Guardrail.before_agent":
        return False
    if span.name.startswith("ToolCallLimitMiddleware"):
        return '"messages"' in str(span.attributes.get(LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT, ""))
    return True


def new_client(**options: Any) -> Langfuse:
    """A Langfuse client with the app's span filter; options go to Langfuse() (tests pass span_exporter)."""
    # OpenTelemetry reads the service name from here; without it traces say "unknown_service:python".
    os.environ.setdefault("OTEL_SERVICE_NAME", SERVICE)
    return Langfuse(should_export_span=worth_exporting, **options)


def configured() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def client() -> Langfuse:
    """The app's Langfuse client, created on first use from the environment (so after .env is loaded)."""
    global _client, _public_key
    if _client is None:
        if configured():
            _client, _public_key = new_client(), os.environ["LANGFUSE_PUBLIC_KEY"]
        else:
            _client, _public_key = Langfuse(tracing_enabled=False, public_key=OFF_KEY, secret_key=OFF_KEY), OFF_KEY
    return _client


def use_client(langfuse: Langfuse | None, public_key: str = OFF_KEY) -> None:
    """Replace the client (tests pass one that exports to memory); None goes back to the environment."""
    global _client, _public_key
    _client, _public_key = langfuse, public_key


def enabled() -> bool:
    client()
    return _public_key != OFF_KEY


def start() -> None:
    """Set tracing up at startup and say whether it is on; a wrong key or URL is reported here."""
    if not enabled():
        logger.info("Langfuse tracing is off (set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to turn it on)")
        return
    try:
        authenticated = client().auth_check()
    except Exception as exc:  # unreachable server, timeout
        logger.warning("Langfuse tracing is on, but the server can't be reached: %s", exc)
        return
    if authenticated:
        logger.info("Langfuse tracing is on")
    else:
        logger.warning("Langfuse tracing is on, but the keys were rejected; check LANGFUSE_* in .env")


def flush() -> None:
    """Send whatever is still queued (the SDK also shuts down at exit); call when the app stops."""
    if _client is not None:
        _client.flush()


class Turn:
    """The trace of one shopper message; pass `callbacks` to the agent and `finish` the answer."""

    def __init__(self, root, callbacks: list[BaseCallbackHandler]):
        self._root = root
        self.callbacks = callbacks

    def finish(self, answer: str) -> None:
        self._root.update(output=answer)


@contextmanager
def trace_turn(message: str, conversation_id: str, *, source: str, metadata: dict[str, Any]) -> Iterator[Turn]:
    """Trace one shopper message. `source` (chat-stream, chat or cli) becomes a tag."""
    langfuse = client()
    with langfuse.start_as_current_observation(as_type="span", name=TURN, input=message, metadata=metadata) as root:
        with propagate_attributes(session_id=conversation_id, tags=[source]):
            callbacks = [CallbackHandler(public_key=_public_key)] if enabled() else []
            try:
                yield Turn(root, callbacks)
            except Exception as exc:
                root.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
                raise
