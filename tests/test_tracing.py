import asyncio
import dataclasses
import json

import pytest
from conftest import FailingModel, FakeEncoder, ToolCallingModel, raw_frame
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langfuse import LangfuseOtelSpanAttributes as Attr
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from qdrant_client import QdrantClient

from tiredai import tracing
from tiredai.agent import astream_turn, build_agent, stream_reply
from tiredai.api import create_app
from tiredai.config import AgentSettings, Settings
from tiredai.documents import products
from tiredai.embeddings import EmbeddingError, Encoder
from tiredai.preprocessing import normalize
from tiredai.search import CatalogSearch, make_search_tool
from tiredai.vectorstore import index_products

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


class NamedModel(ToolCallingModel):
    """A scripted model that reports a model name, like ChatOpenRouter does."""

    model: str = "test/chat-model"


class Dense:
    model = "test/embedding-model"

    def embed_query(self, text):
        return FakeEncoder().dense(text)


class Sparse:
    def embed_query(self, text):
        return FakeEncoder().sparse(text)


def search_tool():
    client = QdrantClient(":memory:")
    index_products(client, "tires", products(normalize(raw_frame({}, {"size": "215/60R16"}))), FakeEncoder())
    return make_search_tool(CatalogSearch(client, "tires", Encoder(Dense(), Sparse()), max_results=20))


def search(call_id: str, **args) -> dict:
    return {"name": "search_tires", "args": args, "id": call_id}


@pytest.fixture
def settings(tmp_path):
    prompt = tmp_path / "system.md"
    prompt.write_text("You are a test assistant.")
    loaded = Settings.load()
    return dataclasses.replace(
        loaded, llm=dataclasses.replace(loaded.llm, system_prompt_path=prompt), agent=AgentSettings.from_env({})
    )


def run_turn(agent, message: str, conversation_id: str, **kwargs) -> list[dict]:
    async def collect():
        return [event async for event in astream_turn(agent, message, conversation_id, **kwargs)]

    return asyncio.run(collect())


def test_a_turn_is_one_trace_with_the_message_and_the_answer(traces, settings):
    model = NamedModel(messages=iter([AIMessage(content="", tool_calls=[search("c1", size="205/55R15", query="accelera")]),
                                      AIMessage(content="The Accelera Phi-R is $59.93.")]))  # fmt: skip
    agent = build_agent(settings, model=model, tools=[search_tool()])

    run_turn(agent, "205/55R15 tires?", "conversation-1", metadata={"llm_model": "test/chat-model"})

    spans = traces.spans()
    assert len({s.context.trace_id for s in spans}) == 1
    [root] = [s for s in spans if s.parent is None]
    assert root.name == tracing.TURN
    assert root.attributes[Attr.OBSERVATION_INPUT] == "205/55R15 tires?"
    assert root.attributes[Attr.OBSERVATION_OUTPUT] == "The Accelera Phi-R is $59.93."
    assert root.attributes[f"{Attr.OBSERVATION_METADATA}.llm_model"] == "test/chat-model"
    # Every observation carries the conversation as its session and where the message came from.
    assert {s.attributes[Attr.TRACE_SESSION_ID] for s in spans} == {"conversation-1"}
    assert all(s.attributes[Attr.TRACE_TAGS] == ("chat-stream",) for s in spans)
    assert root.resource.attributes["service.name"] == tracing.SERVICE


def test_model_calls_tools_and_retrieval_are_nested_with_their_types(traces, settings):
    model = NamedModel(messages=iter([AIMessage(content="", tool_calls=[search("c1", size="205/55R15", query="accelera")]),
                                      AIMessage(content="Found it.")]))  # fmt: skip
    agent = build_agent(settings, model=model, tools=[search_tool()])

    run_turn(agent, "205/55R15 tires?", "conversation-1")

    generation = ("NamedModel", "generation")
    assert traces.tree() == [
        [(tracing.TURN, "span"), [
            [("tire_agent", "agent"), [
                [("model", "chain"), [[generation, []]]],
                [("tools", "chain"), [
                    [("search_tires", "tool"), [
                        [("retrieve-products", "retriever"), [[("embed-query", "embedding"), []]]],
                    ]],
                ]],
                [("model", "chain"), [[generation, []]]],
            ]],
        ]],
    ]  # fmt: skip
    # The tool-call limit's bookkeeping after each model call blocked nothing, so it isn't exported.
    assert not traces.named("ToolCallLimitMiddleware.after_model")


def test_generations_record_the_model_prompt_and_reasoning(traces, settings):
    answer = AIMessage(content="No tires match.", additional_kwargs={"reasoning_content": "The size is not listed."})
    agent = build_agent(settings, model=NamedModel(messages=iter([answer])))

    run_turn(agent, "Tires in 999/99R99?", "conversation-1")

    [generation] = traces.named("NamedModel")
    assert generation.attributes[Attr.OBSERVATION_MODEL] == "test/chat-model"
    prompt = json.loads(generation.attributes[Attr.OBSERVATION_INPUT])
    assert [m["role"] for m in prompt] == ["system", "user"]
    output = json.loads(generation.attributes[Attr.OBSERVATION_OUTPUT])
    assert output["content"] == "No tires match."
    assert output["additional_kwargs"]["reasoning_content"] == "The size is not listed."


def test_retrieval_records_the_applied_filters_ranking_and_embedding(traces, settings):
    model = NamedModel(messages=iter([AIMessage(content="", tool_calls=[search("c1", size="205 55 15", query="accelera")]),
                                      AIMessage(content="Found it.")]))  # fmt: skip
    agent = build_agent(settings, model=model, tools=[search_tool()])

    run_turn(agent, "205 55 15 accelera?", "conversation-1")

    [retrieval] = traces.named("retrieve-products")
    # The size as the search applied it (normalized), not as the model wrote it.
    assert json.loads(retrieval.attributes[Attr.OBSERVATION_INPUT]) == {
        "query": "accelera", "filters": {"size": "205/55R15"}, "sort": "relevance"
    }  # fmt: skip
    output = json.loads(retrieval.attributes[Attr.OBSERVATION_OUTPUT])
    assert output["total_matching"] == 1 and output["order"] == "relevance"
    [product] = output["products"]
    assert product["sku"] == "SKU-0" and product["score"] > 0
    [embedding] = traces.named("embed-query")
    assert embedding.attributes[Attr.OBSERVATION_MODEL] == "test/embedding-model"
    assert embedding.attributes[Attr.OBSERVATION_INPUT] == "accelera"


def test_a_failed_embedding_marks_the_retrieval_as_an_error(traces, settings):
    class DownDense(Dense):
        def embed_query(self, text):
            raise EmbeddingError("OpenRouter daily free-model limit reached")

    client = QdrantClient(":memory:")
    index_products(client, "tires", products(normalize(raw_frame({}))), FakeEncoder())
    catalog = CatalogSearch(client, "tires", Encoder(DownDense(), Sparse()), max_results=20)

    with tracing.client().start_as_current_observation(name="test-search"):
        assert "temporarily unavailable" in catalog.search(query="quiet tires")["error"]

    [retrieval] = traces.named("retrieve-products")
    [embedding] = traces.named("embed-query")
    assert retrieval.attributes[Attr.OBSERVATION_LEVEL] == embedding.attributes[Attr.OBSERVATION_LEVEL] == "ERROR"
    assert "daily free-model limit" in retrieval.attributes[Attr.OBSERVATION_STATUS_MESSAGE]


def test_a_failed_turn_is_marked_as_an_error(traces, settings):
    agent = build_agent(settings, model=FailingModel(messages=iter([])))

    with pytest.raises(RuntimeError):
        run_turn(agent, "Hi", "conversation-1")

    [root] = traces.named(tracing.TURN)
    assert root.attributes[Attr.OBSERVATION_LEVEL] == "ERROR"
    assert root.attributes[Attr.OBSERVATION_STATUS_MESSAGE] == "RuntimeError: provider unavailable"


def test_searches_over_the_limit_stay_in_the_trace(traces, settings):
    limited = dataclasses.replace(settings, agent=AgentSettings.from_env({"AGENT_MAX_TOOL_CALLS": "1"}))
    calls = [search("c1", size="205/55R15"), search("c2", size="215/60R16")]
    model = NamedModel(messages=iter([AIMessage(content="", tool_calls=calls), AIMessage(content="One search ran.")]))
    agent = build_agent(limited, model=model, tools=[search_tool()])

    run_turn(agent, "Both sizes?", "conversation-1")

    # The step that blocked the second call is kept: its output holds the error result.
    [blocked] = traces.named("ToolCallLimitMiddleware.after_model")
    assert "limit" in blocked.attributes[Attr.OBSERVATION_OUTPUT]
    assert len(traces.named("search_tires")) == 1


def test_terminal_turns_are_tagged_cli(traces, settings):
    agent = build_agent(settings, model=NamedModel(messages=iter([AIMessage(content="Hello.")])))

    assert "".join(stream_reply(agent, "Hi", "conversation-1")) == "Hello."

    [root] = traces.named(tracing.TURN)
    assert root.attributes[Attr.TRACE_TAGS] == ("cli",)
    assert root.attributes[Attr.OBSERVATION_OUTPUT] == "Hello."


def test_api_turns_are_tagged_by_endpoint_and_grouped_by_conversation(traces, settings, tmp_path):
    api_settings = dataclasses.replace(
        settings, conversations_path=tmp_path / "conversations.sqlite", qdrant_path=tmp_path / "vectorstore", qdrant_url=None
    )
    model = NamedModel(messages=iter([AIMessage(content="Streamed."), AIMessage(content="As JSON.")]))

    with TestClient(create_app(api_settings, model=model)) as client:
        conversation_id = client.post("/chat", json={"message": "First"}).json()["conversation_id"]
        client.post("/chat/stream", json={"message": "Second", "conversation_id": conversation_id})

    turns = traces.named(tracing.TURN)
    assert [t.attributes[Attr.TRACE_TAGS] for t in turns] == [("chat",), ("chat-stream",)]
    assert {t.attributes[Attr.TRACE_SESSION_ID] for t in turns} == {conversation_id}
    assert turns[0].attributes[f"{Attr.OBSERVATION_METADATA}.llm_model"] == settings.llm.model


def test_without_keys_nothing_is_traced(langfuse_in_memory, settings):
    _, exporter = langfuse_in_memory
    exporter.clear()
    tracing.use_client(None)  # back to the environment, which has no keys in tests
    agent = build_agent(settings, model=NamedModel(messages=iter([AIMessage(content="Hello.")])))

    with tracing.trace_turn("Hi", "conversation-1", source="cli", metadata={}) as turn:
        assert turn.callbacks == []  # the LangChain handler isn't even attached
    assert "".join(stream_reply(agent, "Hi", "conversation-1")) == "Hello."

    assert not tracing.enabled()
    assert exporter.get_finished_spans() == ()
