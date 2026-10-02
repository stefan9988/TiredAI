"""The shopping assistant: a LangChain agent on an OpenRouter chat model, with per-thread memory."""

from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from tiredai.config import LLMSettings, Settings
from tiredai.search import describe_results, describe_search

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


# Text the model writes before and after a tool call is kept apart by a Markdown rule.
SEGMENT_SEPARATOR = "\n\n---\n\n"


class _AnswerText:
    """Picks the answer text out of one turn's streamed chunks.

    Only text from the model is shown: tool results and reasoning blocks are not. Text from a later
    model step (after a tool call) starts with SEGMENT_SEPARATOR, unless the earlier steps wrote only
    whitespace.
    """

    def __init__(self):
        self.step = None  # the model step that last wrote visible text

    def __call__(self, chunk: object, metadata: dict) -> str:
        if not (isinstance(chunk, AIMessageChunk) and metadata.get("langgraph_node") == "model"):
            return ""
        text = chunk.text
        if not text.strip():
            return text
        previous, self.step = self.step, metadata.get("langgraph_step")
        return SEGMENT_SEPARATOR + text if previous not in (None, self.step) else text


def stream_reply(agent: CompiledStateGraph, message: str, thread_id: str) -> Iterator[str]:
    """Send one shopper message and yield the assistant's answer text as it is generated."""
    answer_text = _AnswerText()
    for chunk, metadata in agent.stream(_user_input(message), thread_config(thread_id), stream_mode="messages"):
        if text := answer_text(chunk, metadata):
            yield text


def _status(stage: str, text: str) -> dict:
    return {"type": "status", "stage": stage, "text": text}


def _describe_call(call: dict) -> str:
    return describe_search(call["args"]) if call["name"] == "search_tires" else f"Running {call['name']}"


def _describe_result(message: ToolMessage) -> str:
    return describe_results(message.content) if message.name == "search_tires" else f"{message.name} finished"


async def astream_turn(agent: CompiledStateGraph, message: str, thread_id: str) -> AsyncIterator[dict]:
    """One shopper message as a stream of events for the UI.

    Yields {"type": "status", "stage": "thinking" | "searching" | "results", "text": ...} as the
    agent works, and {"type": "token", "text": ...} for each chunk of the answer.
    """
    yield _status("thinking", "Thinking…")
    answer_text = _AnswerText()
    async for mode, chunk in agent.astream(
        _user_input(message), thread_config(thread_id), stream_mode=["messages", "updates"]
    ):
        if mode == "messages":
            if text := answer_text(*chunk):
                yield {"type": "token", "text": text}
            continue
        # "updates" carry each finished step: model steps with complete tool calls, then tool results.
        for update in chunk.values():
            messages = (update or {}).get("messages", [])
            for item in messages:
                if isinstance(item, AIMessage):
                    for call in item.tool_calls:
                        yield _status("searching", _describe_call(call))
                elif isinstance(item, ToolMessage):
                    yield _status("results", _describe_result(item))
            if any(isinstance(item, ToolMessage) for item in messages):
                yield _status("thinking", "Thinking…")  # the model runs again with the results


async def astream_reply(agent: CompiledStateGraph, message: str, thread_id: str) -> AsyncIterator[str]:
    """Only the answer text of astream_turn."""
    async for event in astream_turn(agent, message, thread_id):
        if event["type"] == "token":
            yield event["text"]


def transcript(messages: Sequence[BaseMessage]) -> list[dict]:
    """The conversation as the shopper saw it: {"role": "user" | "assistant", "content": ...} per message.

    Tool calls and results are left out, and a turn's answer text is joined like the stream sent it.
    """
    turns = []
    for message in messages:
        if isinstance(message, HumanMessage):
            turns.append({"role": "user", "content": message.text})
        elif isinstance(message, AIMessage) and message.text.strip():
            if turns and turns[-1]["role"] == "assistant":
                turns[-1]["content"] += SEGMENT_SEPARATOR + message.text
            else:
                turns.append({"role": "assistant", "content": message.text})
    return turns


async def aget_transcript(agent: CompiledStateGraph, thread_id: str) -> list[dict]:
    state = await agent.aget_state(thread_config(thread_id))
    return transcript(state.values.get("messages", []))
