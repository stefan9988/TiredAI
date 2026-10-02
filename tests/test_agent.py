import dataclasses
import json

import pytest
from conftest import FakeEncoder, ToolCallingModel, fake_model, raw_frame, tool_call
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from qdrant_client import QdrantClient

from tiredai.agent import (
    SEGMENT_SEPARATOR,
    build_agent,
    build_chat_model,
    load_system_prompt,
    stream_reply,
    thread_config,
    transcript,
)
from tiredai.config import LLMSettings, Settings
from tiredai.documents import products
from tiredai.preprocessing import normalize
from tiredai.search import CatalogSearch, make_search_tool
from tiredai.vectorstore import index_products


@pytest.fixture
def settings(tmp_path):
    prompt = tmp_path / "system.md"
    prompt.write_text("You are a test assistant.\n")
    loaded = Settings.load()
    return dataclasses.replace(loaded, llm=dataclasses.replace(loaded.llm, system_prompt_path=prompt))


def test_reply_is_streamed_in_chunks(settings):
    agent = build_agent(settings, model=fake_model("All season tires work year round."))

    chunks = list(stream_reply(agent, "What are all season tires?", "thread-1"))

    assert len(chunks) > 1
    assert "".join(chunks) == "All season tires work year round."


def test_system_prompt_comes_from_the_prompt_file(settings):
    model = fake_model("Hi.")
    agent = build_agent(settings, model=model)

    list(stream_reply(agent, "Hello", "thread-1"))

    first = model.prompts[0][0]
    assert isinstance(first, SystemMessage) and first.content == "You are a test assistant."


def test_conversation_history_is_kept_per_thread(settings):
    model = fake_model("First answer.", "Second answer.", "Other thread.")
    agent = build_agent(settings, model=model)

    list(stream_reply(agent, "I need 205/55R16 tires", "thread-1"))
    list(stream_reply(agent, "Something cheaper?", "thread-1"))
    list(stream_reply(agent, "Hello", "thread-2"))

    second_turn = [m.content for m in model.prompts[1]]
    assert second_turn == ["You are a test assistant.", "I need 205/55R16 tires", "First answer.", "Something cheaper?"]
    assert [m.content for m in model.prompts[2]] == ["You are a test assistant.", "Hello"]


def test_project_system_prompt_exists():
    prompt = load_system_prompt(LLMSettings.from_env({}).system_prompt_path)

    assert "Size search" in prompt and "Product inquiry" in prompt and "Education" in prompt


def test_missing_system_prompt_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="SYSTEM_PROMPT_PATH"):
        load_system_prompt(tmp_path / "missing.md")


def test_empty_system_prompt_is_rejected(tmp_path):
    empty = tmp_path / "empty.md"
    empty.write_text("  \n")

    with pytest.raises(ValueError, match="empty"):
        load_system_prompt(empty)


def test_chat_model_gets_every_configured_parameter():
    llm = LLMSettings.from_env(
        {
            "LLM_MODEL": "nvidia/nemotron-3-ultra-550b-a55b:free",
            "LLM_TEMPERATURE": "0.2",
            "LLM_TOP_P": "0.9",
            "LLM_MAX_TOKENS": "800",
            "LLM_SEED": "7",
            "LLM_REASONING_EFFORT": "low",
            "LLM_EXTRA_PARAMS": '{"top_k": 40}',
            "LLM_TIMEOUT_SECONDS": "30",
            "LLM_MAX_RETRIES": "1",
        }
    )

    model = build_chat_model(llm, "test-key")

    assert model._default_params == {
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "stream": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 800,
        "seed": 7,
        "reasoning": {"effort": "low"},
        "top_k": 40,
    }
    assert model.request_timeout == 30_000  # milliseconds
    assert model.max_retries == 1


def test_unset_parameters_are_not_sent():
    model = build_chat_model(LLMSettings.from_env({}), "test-key")

    assert model._default_params == {"model": "nvidia/nemotron-3-ultra-550b-a55b:free", "stream": True}


def test_chat_model_requires_an_api_key():
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        build_chat_model(LLMSettings.from_env({}), None)


def test_agent_answers_from_search_results(settings):
    client = QdrantClient(":memory:")
    index_products(client, "tires", products(normalize(raw_frame({}))), FakeEncoder())
    tool = make_search_tool(CatalogSearch(client, "tires", FakeEncoder()))
    model = ToolCallingModel(
        messages=iter([tool_call("search_tires", size="205/55R15"), AIMessage(content="The Accelera Phi-R is $59.93.")])
    )
    agent = build_agent(settings, model=model, tools=[tool])

    reply = "".join(stream_reply(agent, "Tires in 205/55R15?", "thread-1"))

    assert reply == "The Accelera Phi-R is $59.93."
    [result] = [m for m in model.prompts[1] if isinstance(m, ToolMessage)]
    found = json.loads(result.content)
    assert found["total_matching"] == 1 and found["products"][0]["price"] == 59.93
    client.close()


def test_text_before_and_after_a_tool_call_is_separated(settings):
    client = QdrantClient(":memory:")
    index_products(client, "tires", products(normalize(raw_frame({}))), FakeEncoder())
    tool = make_search_tool(CatalogSearch(client, "tires", FakeEncoder()))
    first_search = {"name": "search_tires", "args": {"size": "205/55R15"}, "id": "call-1"}
    model = ToolCallingModel(
        messages=iter(
            [
                AIMessage(content="Let me check.", tool_calls=[first_search]),
                tool_call("search_tires", size="205/55R16"),  # a step without text adds no separator
                AIMessage(content="One tire fits."),
                # Some models write only whitespace before a tool call: no separator for that.
                AIMessage(content="\n\n", tool_calls=[{**first_search, "id": "call-2"}]),
                AIMessage(content="You're welcome."),
            ]
        )
    )
    agent = build_agent(settings, model=model, tools=[tool])

    reply = "".join(stream_reply(agent, "Tires in 205/55R15?", "thread-1"))
    next_reply = "".join(stream_reply(agent, "Thanks", "thread-1"))

    assert reply == "Let me check." + SEGMENT_SEPARATOR + "One tire fits."
    assert next_reply == "\n\nYou're welcome."
    saved = transcript(agent.get_state(thread_config("thread-1")).values["messages"])
    assert [m["content"] for m in saved if m["role"] == "assistant"] == [reply, "You're welcome."]
    client.close()


def test_transcript_shows_what_the_shopper_saw():
    messages = [
        HumanMessage("205/55R15 under $60?"),
        AIMessage(content="Let me check. ", tool_calls=[{"name": "search_tires", "args": {}, "id": "call-1"}]),
        ToolMessage(content='{"total_matching": 1}', tool_call_id="call-1", name="search_tires"),
        AIMessage(content="One tire fits."),
        HumanMessage("Thanks"),
        AIMessage(content="\n\n", tool_calls=[{"name": "search_tires", "args": {}, "id": "call-2"}]),
        ToolMessage(content="{}", tool_call_id="call-2", name="search_tires"),
        AIMessage(content=[{"type": "reasoning", "reasoning": "hidden"}, {"type": "text", "text": "You're welcome."}]),
    ]

    assert transcript(messages) == [
        {"role": "user", "content": "205/55R15 under $60?"},
        {"role": "assistant", "content": "Let me check. " + SEGMENT_SEPARATOR + "One tire fits."},
        {"role": "user", "content": "Thanks"},
        {"role": "assistant", "content": "You're welcome."},
    ]


def test_transcript_keeps_a_turn_that_got_no_answer():
    assert transcript([HumanMessage("Hi"), HumanMessage("Hello?"), AIMessage(content="Hello.")]) == [
        {"role": "user", "content": "Hi"},
        {"role": "user", "content": "Hello?"},
        {"role": "assistant", "content": "Hello."},
    ]
