import pytest

from tiredai.config import ROOT, AgentSettings, LLMSettings, Settings


def test_llm_defaults_leave_model_parameters_unset():
    llm = LLMSettings.from_env({})

    assert llm.model == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert llm.temperature is None and llm.top_p is None and llm.max_tokens is None and llm.seed is None
    assert llm.reasoning_effort is None and llm.extra_params == {}
    assert llm.max_retries == 2
    assert llm.streaming is True
    assert llm.system_prompt_path == ROOT / "prompts" / "system.md"


def test_llm_parameters_are_parsed_from_env():
    llm = LLMSettings.from_env(
        {
            "LLM_MODEL": "some/model",
            "LLM_TEMPERATURE": "0.2",
            "LLM_TOP_P": "0.9",
            "LLM_MAX_TOKENS": "800",
            "LLM_SEED": "7",
            "LLM_FREQUENCY_PENALTY": "0.1",
            "LLM_PRESENCE_PENALTY": "-0.1",
            "LLM_REASONING_EFFORT": "low",
            "LLM_EXTRA_PARAMS": '{"top_k": 40}',
            "LLM_TIMEOUT_SECONDS": "30",
            "LLM_MAX_RETRIES": "0",
            "LLM_STREAMING": "false",
            "SYSTEM_PROMPT_PATH": "/tmp/prompt.md",
        }
    )

    assert llm.model == "some/model"
    assert (llm.temperature, llm.top_p, llm.max_tokens, llm.seed) == (0.2, 0.9, 800, 7)
    assert (llm.frequency_penalty, llm.presence_penalty) == (0.1, -0.1)
    assert llm.reasoning_effort == "low"
    assert llm.extra_params == {"top_k": 40}
    assert llm.timeout_seconds == 30.0
    assert llm.max_retries == 0
    assert llm.streaming is False
    assert str(llm.system_prompt_path) == "/tmp/prompt.md"


def test_empty_values_mean_unset():
    llm = LLMSettings.from_env({"LLM_TEMPERATURE": "", "LLM_MAX_TOKENS": "  ", "LLM_STREAMING": ""})

    assert llm.temperature is None and llm.max_tokens is None and llm.streaming is True


@pytest.mark.parametrize(
    "name, value",
    [
        ("LLM_TEMPERATURE", "warm"),
        ("LLM_MAX_TOKENS", "1.5"),
        ("LLM_STREAMING", "maybe"),
        ("LLM_EXTRA_PARAMS", "{not json"),
        ("LLM_EXTRA_PARAMS", "[1, 2]"),
    ],
)
def test_invalid_values_name_the_variable(name, value):
    with pytest.raises(ValueError, match=name):
        LLMSettings.from_env({name: value})


def test_agent_limit_defaults():
    assert AgentSettings.from_env({}) == AgentSettings(history_messages=10, max_tool_calls=5, max_search_results=20)
    assert AgentSettings.from_env({"AGENT_MAX_TOOL_CALLS": " "}).max_tool_calls == 5


def test_agent_limits_are_parsed_from_env():
    limits = AgentSettings.from_env(
        {"AGENT_HISTORY_MESSAGES": "4", "AGENT_MAX_TOOL_CALLS": "1", "AGENT_MAX_SEARCH_RESULTS": "50"}
    )

    assert limits == AgentSettings(history_messages=4, max_tool_calls=1, max_search_results=50)


@pytest.mark.parametrize("name", ["AGENT_HISTORY_MESSAGES", "AGENT_MAX_TOOL_CALLS", "AGENT_MAX_SEARCH_RESULTS"])
@pytest.mark.parametrize("value", ["0", "-3", "1.5", "ten"])
def test_agent_limits_must_be_positive_integers(name, value):
    with pytest.raises(ValueError, match=name):
        AgentSettings.from_env({name: value})


def test_embedding_concurrency_defaults_to_8_and_must_be_positive(monkeypatch):
    monkeypatch.delenv("EMBEDDING_CONCURRENCY", raising=False)
    monkeypatch.setattr("tiredai.config.load_dotenv", lambda path: None)
    assert Settings.load().embedding_concurrency == 8

    monkeypatch.setenv("EMBEDDING_CONCURRENCY", "3")
    assert Settings.load().embedding_concurrency == 3

    monkeypatch.setenv("EMBEDDING_CONCURRENCY", "0")
    with pytest.raises(ValueError, match="EMBEDDING_CONCURRENCY"):
        Settings.load()
