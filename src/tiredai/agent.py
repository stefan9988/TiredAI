"""The shopping assistant: a LangChain agent on an OpenRouter chat model, with per-thread memory."""

from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from tiredai.config import LLMSettings, Settings

APP_TITLE = "TiredAI"


def load_system_prompt(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"System prompt not found: {path} (set SYSTEM_PROMPT_PATH)")
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"System prompt is empty: {path}")
    return prompt


def build_chat_model(llm: LLMSettings, api_key: str | None) -> ChatOpenRouter:
    if not api_key:
        raise ValueError("The chat model requires OPENROUTER_API_KEY")
    return ChatOpenRouter(
        model=llm.model,
        api_key=api_key,
        base_url=llm.base_url,
        temperature=llm.temperature,
        top_p=llm.top_p,
        max_tokens=llm.max_tokens,
        seed=llm.seed,
        frequency_penalty=llm.frequency_penalty,
        presence_penalty=llm.presence_penalty,
        reasoning={"effort": llm.reasoning_effort} if llm.reasoning_effort else None,
        model_kwargs=llm.extra_params,
        # ChatOpenRouter takes the timeout in milliseconds.
        timeout=round(llm.timeout_seconds * 1000) if llm.timeout_seconds is not None else None,
        max_retries=llm.max_retries,
        streaming=llm.streaming,
        app_title=APP_TITLE,
    )


def build_agent(
    settings: Settings,
    *,
    model: BaseChatModel | None = None,
    tools: Sequence[BaseTool] = (),
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Agent with the system prompt from SYSTEM_PROMPT_PATH; history is kept per thread id."""
    return create_agent(
        model or build_chat_model(settings.llm, settings.openrouter_api_key),
        tools=list(tools),
        system_prompt=load_system_prompt(settings.llm.system_prompt_path),
        checkpointer=checkpointer or InMemorySaver(),
        name="tire_assistant",
    )


def thread_config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def _user_input(message: str) -> dict:
    return {"messages": [{"role": "user", "content": message}]}


def _answer_text(chunk: object, metadata: dict) -> str:
    # Only answer text from the model: tool results and reasoning blocks are not shown.
    if isinstance(chunk, AIMessageChunk) and metadata.get("langgraph_node") == "model":
        return chunk.text
    return ""


def stream_reply(agent: CompiledStateGraph, message: str, thread_id: str) -> Iterator[str]:
    """Send one shopper message and yield the assistant's answer text as it is generated."""
    for chunk, metadata in agent.stream(_user_input(message), thread_config(thread_id), stream_mode="messages"):
        if text := _answer_text(chunk, metadata):
            yield text


async def astream_reply(agent: CompiledStateGraph, message: str, thread_id: str) -> AsyncIterator[str]:
    """Async version of stream_reply, for the API."""
    async for chunk, metadata in agent.astream(
        _user_input(message), thread_config(thread_id), stream_mode="messages"
    ):
        if text := _answer_text(chunk, metadata):
            yield text
