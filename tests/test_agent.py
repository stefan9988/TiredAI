import dataclasses

import pytest
from conftest import fake_model
from langchain_core.messages import SystemMessage

from tiredai.agent import build_agent, build_chat_model, load_system_prompt, stream_reply
from tiredai.config import LLMSettings, Settings


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
