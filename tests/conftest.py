import json
import zlib
from collections import Counter

import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from pydantic import Field
from qdrant_client import models

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
        yield ChatGenerationChunk(message=AIMessageChunk(content=message.content, tool_call_chunks=chunks))


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
