import json
import os
import zlib
from collections import Counter

import httpx
import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langfuse import LangfuseOtelSpanAttributes as Attr
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import Field
from qdrant_client import models

from tiredai import tracing
from tiredai.guardrail import Guard, JevClient

# Tests never send traces: empty keys keep tracing off, and load_dotenv doesn't override them with .env.
# Nor do they call OpenRouter: without its key the agent has no guardrail unless a test passes a fake one.
os.environ.update(LANGFUSE_PUBLIC_KEY="", LANGFUSE_SECRET_KEY="", OPENROUTER_API_KEY="")

# One valid raw CSV row, as text exactly like the catalog stores it.
RAW_ROW = {
    "sku": "N889368-99",
    "name": "Accelera Phi-R 205/55R15 92V XL",
    "brand": "Accelera",
    "model": "Phi-R",
    "size": "205/55R15",
    "season": "All Season",
    "carType": "Passenger",
    "performance": "Performance",
    "speedRating": "V",
    "loadIndex": "92",
    "loadRange": "XL",
    "runFlat": "false",
    "sidewall": "BSW: Black Side Wall",
    "utqg": "400AA",
    "treadDepth": "9/32",
    "mileageWarranty": "50,000 miles",
    "overallDiameter": "23.90",
    "rimDiameter": "15",
    "price": "59.930000",
}


def raw_frame(*overrides: dict) -> pd.DataFrame:
    """Raw catalog rows: one per overrides dict, each based on RAW_ROW with a unique SKU."""
    rows = [{**RAW_ROW, "sku": f"SKU-{i}", **o} for i, o in enumerate(overrides or [{}])]
    return pd.DataFrame(rows, dtype=str)


@pytest.fixture
def write_csv(tmp_path):
    def write(*overrides: dict):
        path = tmp_path / "catalog.csv"
        raw_frame(*overrides).to_csv(path, index=False)
        return path

    return write


class RecordingModel(GenericFakeChatModel):
    """Fake chat model that streams scripted replies word by word and records every prompt it gets."""

    prompts: list = Field(default_factory=list)

    def _stream(self, messages, *args, **kwargs):
        self.prompts.append(messages)
        yield from super()._stream(messages, *args, **kwargs)


class FailingModel(GenericFakeChatModel):
    """Fake chat model whose provider is down."""

    def _stream(self, messages, *args, **kwargs):
        raise RuntimeError("provider unavailable")
        yield  # makes this a generator, like the real _stream


def fake_model(*replies: str) -> RecordingModel:
    return RecordingModel(messages=iter(AIMessage(content=r) for r in replies))


class ToolCallingModel(RecordingModel):
    """Scripted model that can also call tools, like a real chat model bound to tools."""

    def bind_tools(self, tools, **kwargs):
        return self

    def _stream(self, messages, *args, **kwargs):
        self.prompts.append(messages)
        message = next(self.messages)
        # Unparsable calls are streamed with their raw argument text, like a provider would send them.
        calls = [(c, json.dumps(c["args"])) for c in message.tool_calls] + [(c, c["args"]) for c in message.invalid_tool_calls]
        chunks = [
            {"name": c["name"], "args": args, "id": c["id"], "index": i, "type": "tool_call_chunk"}
            for i, (c, args) in enumerate(calls)
        ]
        chunk = AIMessageChunk(content=message.content, additional_kwargs=message.additional_kwargs, tool_call_chunks=chunks)
        yield ChatGenerationChunk(message=chunk)


def tool_call(name: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call-{name}"}])


def unparsable_call(raw_args: str, call_id: str = "bad-call", name: str = "search_tires") -> AIMessage:
    """A tool call whose argument text is not JSON."""
    return AIMessage(
        content="", invalid_tool_calls=[{"name": name, "args": raw_args, "id": call_id, "error": None, "type": "invalid_tool_call"}]
    )


class FakeEncoder:
    """Deterministic, offline stand-in for the embedding models: vectors come from hashed lowercase tokens."""

    dense_dim = 8

    def encode_documents(self, texts):
        return [self.dense(t) for t in texts], [self.sparse(t) for t in texts]

    def encode_query(self, text):
        return self.dense(text), self.sparse(text)

    def dense(self, text):
        vector = [0.0] * self.dense_dim
        for token in text.lower().split():
            vector[zlib.crc32(token.encode()) % self.dense_dim] += 1.0
        return vector

    def sparse(self, text):
        counts = Counter(zlib.crc32(token.encode()) for token in text.lower().split())
        return models.SparseVector(indices=list(counts), values=[float(c) for c in counts.values()])


KEY = "pk-lf-test"


@pytest.fixture(scope="session")
def langfuse_in_memory():
    # One client for the session: Langfuse keeps one client per public key. It exports to memory,
    # and its server address is never contacted.
    exporter = InMemorySpanExporter()
    client = tracing.new_client(public_key=KEY, secret_key="sk-lf-test", base_url="http://127.0.0.1:9", span_exporter=exporter)
    return client, exporter


class Traces:
    def __init__(self, client, exporter):
        self.client = client
        self.exporter = exporter

    def spans(self):
        self.client.flush()
        return sorted(self.exporter.get_finished_spans(), key=lambda s: s.start_time)

    def named(self, name):
        return [s for s in self.spans() if s.name == name]

    def tree(self):
        """(name, type) of each observation, nested as [(name, type), [children...]]."""
        spans = self.spans()
        children = {}
        for span in spans:
            children.setdefault(span.parent.span_id if span.parent else None, []).append(span)

        def node(span):
            return [(span.name, span.attributes.get(Attr.OBSERVATION_TYPE)), [node(c) for c in children.get(span.context.span_id, [])]]

        return [node(root) for root in children.get(None, [])]


@pytest.fixture
def traces(langfuse_in_memory):
    client, exporter = langfuse_in_memory
    exporter.clear()
    tracing.use_client(client, KEY)
    yield Traces(client, exporter)
    tracing.use_client(None)


def decisions(**probabilities) -> dict:
    """A Decisions API answer with these probabilities of yes."""
    return {"model": "typesafe/jev-1.13-20260917", "answers": {name: {"type": "noul", "noul": p} for name, p in probabilities.items()},
            "usage": {"input_tokens": 600, "output_tokens": 30, "cost": 0.0000252}}  # fmt: skip


class FakeDecisions:
    """The Decisions API: replies in order, the last one again after that (a dict is a 200 body, an int an
    error status, an exception is raised, e.g. httpx.ReadTimeout), recording each request."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def states(self) -> list[dict]:
        return [json.loads(r.content)["state"] for r in self.requests]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, int):
            return httpx.Response(reply, text="provider says no")
        return httpx.Response(200, json=reply)


def jev_client(api: FakeDecisions, slept: list | None = None, max_retries: int = 3) -> JevClient:
    return JevClient("test-key", "typesafe/jev-1.13", client=httpx.Client(transport=httpx.MockTransport(api)),
                     max_retries=max_retries, sleep=(slept if slept is not None else []).append)  # fmt: skip


def fake_guard(api: FakeDecisions, threshold: float = 0.7) -> Guard:
    """The app's guard, on the fake Decisions API and without retries."""
    return Guard(jev_client(api, max_retries=0), threshold)


def sse_events(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        events.append((fields["event"], json.loads(fields["data"])))
    return events
