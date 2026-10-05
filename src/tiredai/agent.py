"""The shopping assistant: a LangChain agent on an OpenRouter chat model, with per-thread memory."""

import dataclasses
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallLimitMiddleware,
    ToolCallRequest,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from tiredai.config import AgentSettings, LLMSettings, Settings
from tiredai.search import describe_results, describe_search
from tiredai.tracing import trace_turn

APP_TITLE = "TiredAI"
# How many times the model is asked again when none of its tool calls could be parsed.
UNPARSABLE_CALL_RETRIES = 2

logger = logging.getLogger(__name__)


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


class RecentHistory(AgentMiddleware):
    """Sends the model only the latest turns (see recent_turns); the saved conversation keeps all of them."""

    def __init__(self, max_messages: int):
        super().__init__()
        self.max_messages = max_messages

    def _trimmed(self, request: ModelRequest) -> ModelRequest:
        return request.override(messages=recent_turns(request.messages, self.max_messages))

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        return handler(self._trimmed(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        return await handler(self._trimmed(request))


class UnparsableToolCallsError(RuntimeError):
    pass


def _unparsable_call_errors(message: AIMessage) -> list[ToolMessage]:
    return [
        ToolMessage(
            content=f"Error: the arguments of this {call.get('name') or 'tool'} call could not be parsed "
            f"({call.get('error') or 'malformed or truncated JSON'}). Send the call again with valid JSON arguments.",
            name=call.get("name") or "unknown",
            tool_call_id=call["id"],
            status="error",
        )
        for call in message.invalid_tool_calls
        if call.get("id")
    ]


class UnparsableToolCalls(AgentMiddleware):
    """Tool calls whose arguments can't be parsed get an error result, so the model can send them again.

    If those were the model's only calls, it is asked again right away (otherwise the turn would end
    without an answer), at most UNPARSABLE_CALL_RETRIES times before the turn fails. If it made valid
    calls too, they run and the model sees the errors along with their results. Failed attempts stay
    in the history, each with its error result.
    """

    def _review(self, request: ModelRequest, response: ModelResponse, trace: list[BaseMessage]) -> ModelResponse | ModelRequest:
        """The response to keep, or the request to try again with."""
        message = response.result[-1] if response.result else None
        if not (isinstance(message, AIMessage) and message.invalid_tool_calls):
            return dataclasses.replace(response, result=[*trace, *response.result]) if trace else response
        trace.extend([*response.result, *_unparsable_call_errors(message)])
        if message.tool_calls:
            return dataclasses.replace(response, result=list(trace))
        attempts = sum(isinstance(m, AIMessage) for m in trace)
        if attempts > UNPARSABLE_CALL_RETRIES:
            raise UnparsableToolCallsError(f"The model sent tool calls that could not be parsed {attempts} times in a row")
        return request.override(messages=[*request.messages, *trace])

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        trace: list[BaseMessage] = []
        outcome = self._review(request, handler(request), trace)
        while isinstance(outcome, ModelRequest):
            outcome = self._review(request, handler(outcome), trace)
        return outcome

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        trace: list[BaseMessage] = []
        outcome = self._review(request, await handler(request), trace)
        while isinstance(outcome, ModelRequest):
            outcome = self._review(request, await handler(outcome), trace)
        return outcome


def _tool_failure(request: ToolCallRequest, exc: Exception) -> ToolMessage:
    call = request.tool_call
    logger.exception("Tool %s failed", call["name"])  # called from an except block, so the traceback is logged
    return ToolMessage(
        content=f"Error: {call['name']} failed with an internal error ({type(exc).__name__}: {exc}).",
        name=call["name"],
        tool_call_id=call["id"],
        status="error",
    )


class ToolErrors(AgentMiddleware):
    """A tool that raises gives the model an error result instead of failing the whole turn.

    The model can then tell the shopper it can't look products up right now, and every tool call in
    the history keeps a result, which providers require.
    """

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]
    ) -> ToolMessage | Command:
        try:
            return handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            return _tool_failure(request, exc)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]
    ) -> ToolMessage | Command:
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            return _tool_failure(request, exc)


def agent_middleware(limits: AgentSettings) -> list[AgentMiddleware]:
    return [
        RecentHistory(limits.history_messages),
        UnparsableToolCalls(),
        ToolErrors(),
        # Calls over the limit get an error result instead of running, and the model answers with what it has.
        ToolCallLimitMiddleware(run_limit=limits.max_tool_calls, exit_behavior="continue"),
    ]


def build_agent(
    settings: Settings,
    *,
    model: BaseChatModel | None = None,
    tools: Sequence[BaseTool] = (),
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Agent with the system prompt from SYSTEM_PROMPT_PATH and the AGENT_* limits; history is kept per thread id."""
    return create_agent(
        model or build_chat_model(settings.llm, settings.openrouter_api_key),
        tools=list(tools),
        system_prompt=load_system_prompt(settings.llm.system_prompt_path),
        middleware=agent_middleware(settings.agent),
        checkpointer=checkpointer or InMemorySaver(),
        name="tire_agent",  # "agent" in the name makes Langfuse type the agent loop as an agent
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


def trace_metadata(settings: Settings) -> dict:
    """What a trace records about the setup that answered it, to compare models and limits."""
    return {
        "llm_model": settings.llm.model,
        "embedding_model": f"{settings.embedding_provider}:{settings.embedding_model}",
        "history_messages": settings.agent.history_messages,
        "max_tool_calls": settings.agent.max_tool_calls,
        "max_search_results": settings.agent.max_search_results,
    }


def _traced_config(thread_id: str, callbacks: list) -> RunnableConfig:
    return {**thread_config(thread_id), "callbacks": callbacks}


def stream_reply(
    agent: CompiledStateGraph, message: str, thread_id: str, *, source: str = "cli", metadata: dict | None = None
) -> Iterator[str]:
    """Send one shopper message and yield the assistant's answer text as it is generated."""
    with trace_turn(message, thread_id, source=source, metadata=metadata or {}) as turn:
        answer_text, answer = _AnswerText(), []
        config = _traced_config(thread_id, turn.callbacks)
        for chunk, chunk_metadata in agent.stream(_user_input(message), config, stream_mode="messages"):
            if text := answer_text(chunk, chunk_metadata):
                answer.append(text)
                yield text
        turn.finish("".join(answer))


def _status(stage: str, text: str) -> dict:
    return {"type": "status", "stage": stage, "text": text}


def _describe_call(call: dict) -> str:
    return describe_search(call["args"]) if call["name"] == "search_tires" else f"Running {call['name']}"


def _describe_result(message: ToolMessage) -> str:
    return describe_results(message.content) if message.name == "search_tires" else f"{message.name} finished"


def tool_call_details(call: dict, result: ToolMessage | None) -> dict:
    """A tool call as the model made it and what it got back, so its answer can be checked against the data.

    {"id", "name", "args": the arguments as sent (their raw text if they could not be parsed),
     "error": the error the model got instead of a result, or None,
     "result": the tool's output, parsed if it is JSON (search_tires: the filters as applied, the
               counts and the products), or None after an error}
    """
    args = "" if call["args"] is None else call["args"]
    details = {"id": call["id"], "name": call["name"] or "unknown", "args": args, "error": None, "result": None}
    if result is None:
        details["error"] = "This call has no result."
        return details
    try:
        output = json.loads(result.text)
    except ValueError:
        output = result.text
    if result.status == "error":
        details["error"] = result.text
    elif isinstance(output, dict) and "error" in output:
        details["error"] = str(output["error"])
    else:
        details["result"] = output
    return details


def _tool_calls(message: AIMessage) -> list[dict]:
    """Every call the model made, including those whose arguments could not be parsed."""
    return [*message.tool_calls, *message.invalid_tool_calls]


async def astream_turn(
    agent: CompiledStateGraph,
    message: str,
    thread_id: str,
    *,
    source: str = "chat-stream",
    metadata: dict | None = None,
) -> AsyncIterator[dict]:
    """One shopper message as a stream of events for the UI, traced as one Langfuse trace.

    Yields {"type": "status", "stage": "thinking" | "searching" | "results", "text": ...} as the
    agent works, {"type": "tool_call", **tool_call_details} when a tool call has its result, and
    {"type": "token", "text": ...} for each chunk of the answer.
    """
    with trace_turn(message, thread_id, source=source, metadata=metadata or {}) as turn:
        answer = []
        async for event in _astream_events(agent, message, _traced_config(thread_id, turn.callbacks)):
            if event["type"] == "token":
                answer.append(event["text"])
            yield event
        turn.finish("".join(answer))


async def _astream_events(agent: CompiledStateGraph, message: str, config: RunnableConfig) -> AsyncIterator[dict]:
    yield _status("thinking", "Thinking…")
    answer_text = _AnswerText()
    calls = {}  # this turn's tool calls by id, to pair with their results
    async for mode, chunk in agent.astream(_user_input(message), config, stream_mode=["messages", "updates"]):
        if mode == "messages":
            if text := answer_text(*chunk):
                yield {"type": "token", "text": text}
            continue
        # "updates" carry each finished step: model steps with complete tool calls, then tool results.
        for node, update in chunk.items():
            messages = (update or {}).get("messages", [])
            for item in messages:
                if isinstance(item, AIMessage):
                    for call in item.tool_calls:
                        yield _status("searching", _describe_call(call))
                    calls.update((call["id"], call) for call in _tool_calls(item))
                elif isinstance(item, ToolMessage):
                    over_limit = node.startswith(ToolCallLimitMiddleware.__name__)
                    yield _status("results", "Search limit reached" if over_limit else _describe_result(item))
                    call = calls.get(item.tool_call_id) or {"id": item.tool_call_id, "name": item.name, "args": {}}
                    yield {"type": "tool_call", **tool_call_details(call, item)}
            if messages and isinstance(messages[-1], ToolMessage):
                yield _status("thinking", "Thinking…")  # the model runs again with the results


async def astream_reply(
    agent: CompiledStateGraph, message: str, thread_id: str, *, source: str = "chat", metadata: dict | None = None
) -> AsyncIterator[str]:
    """Only the answer text of astream_turn."""
    async for event in astream_turn(agent, message, thread_id, source=source, metadata=metadata):
        if event["type"] == "token":
            yield event["text"]


def transcript(messages: Sequence[BaseMessage]) -> list[dict]:
    """The conversation as the shopper saw it: {"role": "user" | "assistant", "content": ...} per message.

    A turn's answer text is joined like the stream sent it, and the answer lists the turn's tool calls
    under "tool_calls" (see tool_call_details). A turn whose model wrote no text has no answer.
    """
    results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    turns = []
    for message in messages:
        if isinstance(message, HumanMessage):
            turns.append({"role": "user", "content": message.text})
        elif isinstance(message, AIMessage):
            if not turns or turns[-1]["role"] != "assistant":
                turns.append({"role": "assistant", "content": "", "tool_calls": []})
            answer = turns[-1]
            if message.text.strip():
                answer["content"] += (SEGMENT_SEPARATOR if answer["content"] else "") + message.text
            answer["tool_calls"] += [tool_call_details(c, results.get(c["id"])) for c in _tool_calls(message)]
    return [turn for turn in turns if turn["role"] == "user" or turn["content"]]


def recent_turns(messages: Sequence[BaseMessage], max_messages: int) -> list[BaseMessage]:
    """The latest whole turns that hold at most `max_messages` chat messages, counted like transcript().

    A turn starts at a shopper message and keeps its tool calls and results. The current (last) turn
    is always kept, whatever its size.
    """
    starts = [i for i, message in enumerate(messages) if isinstance(message, HumanMessage)]
    if not starts:
        return list(messages)
    keep = starts[-1]
    count = len(transcript(messages[keep:]))
    for start in reversed(starts[:-1]):
        count += len(transcript(messages[start:keep]))
        if count > max_messages:
            return list(messages[keep:])
        keep = start
    return list(messages)


async def aget_transcript(agent: CompiledStateGraph, thread_id: str) -> list[dict]:
    state = await agent.aget_state(thread_config(thread_id))
    return transcript(state.values.get("messages", []))
