"""The shopping assistant: a LangChain agent on an OpenRouter chat model, with per-thread memory."""

import asyncio
import dataclasses
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallLimitMiddleware,
    ToolCallRequest,
    hook_config,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

from tiredai.config import AgentSettings, LLMSettings, Settings
from tiredai.guardrail import REPLIES, Guard, build_guard
from tiredai.search import describe_results, describe_search
from tiredai.tracing import trace_turn
from tiredai.vehicles import TOOL_NAME as VEHICLE_TOOL
from tiredai.vehicles import describe_lookup, describe_lookup_results

APP_TITLE = "TiredAI"
# How many times the model is asked again when none of its tool calls could be parsed.
UNPARSABLE_CALL_RETRIES = 2
# The graph node of the Guardrail middleware, which writes the guardrail's reply when it blocks a message.
GUARDRAIL_NODE = "Guardrail.before_agent"

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


# What the model reads instead of a message the guardrail blocked, by the guardrail's reason.
BLOCKED_PLACEHOLDER = "[Message withheld: the guardrail blocked it as {reason}.]"
BLOCKED_REASONS = {"off_topic": "off-topic", "manipulation": "an attempt to change the assistant's rules",
                   "harmful": "a harmful request"}  # fmt: skip


def withhold_blocked(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """The messages with the text of each shopper message the guardrail blocked replaced by a placeholder.

    The model knows something was asked and refused (the guardrail's reply stays), but never reads an
    injection or an off-topic request, not even in a later turn's history.
    """
    shown = list(messages)
    for i, message in enumerate(shown):
        decision = message.response_metadata.get("guardrail") if isinstance(message, AIMessage) else None
        if decision and i and isinstance(shown[i - 1], HumanMessage):
            reason = BLOCKED_REASONS.get(decision.get("reason"), "off-limits")
            shown[i - 1] = HumanMessage(content=BLOCKED_PLACEHOLDER.format(reason=reason), id=shown[i - 1].id)
    return shown


class RecentHistory(AgentMiddleware):
    """Sends the model only the latest turns (see recent_turns), without the text of messages the guardrail
    blocked (see withhold_blocked); the saved conversation keeps all of them as they were."""

    def __init__(self, max_messages: int):
        super().__init__()
        self.max_messages = max_messages

    def _trimmed(self, request: ModelRequest) -> ModelRequest:
        return request.override(messages=withhold_blocked(recent_turns(request.messages, self.max_messages)))

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        return handler(self._trimmed(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        return await handler(self._trimmed(request))


@dataclass(frozen=True)
class TurnContext:
    """Settings of one shopper message, passed to the agent run as its context."""

    guardrail: bool = False  # check the message with the guardrail before the model sees it
    web_search: bool = True  # the model may look vehicles up on the web (the vehicle lookup tool)


# Added to the system prompt in place of the vehicle lookup when a turn has web search off.
WEB_SEARCH_OFF = ("Web search is off for this message, so you can't look up a vehicle's tire sizes. If the shopper "
                  "doesn't know their size, tell them it is on the sticker inside the driver's door and on the tire's "
                  "sidewall.")  # fmt: skip


def _web_search_off(runtime: Runtime) -> bool:
    return bool(runtime.context) and not runtime.context.web_search


def _without_web_search(request: ModelRequest) -> ModelRequest:
    tools = [tool for tool in request.tools if getattr(tool, "name", None) != VEHICLE_TOOL]
    if len(tools) == len(request.tools):
        return request  # the agent has no vehicle lookup
    return request.override(tools=tools, system_message=SystemMessage(content=f"{request.system_prompt}\n\n{WEB_SEARCH_OFF}"))


def _web_search_refused(request: ToolCallRequest) -> ToolMessage | None:
    call = request.tool_call
    if call["name"] != VEHICLE_TOOL or not _web_search_off(request.runtime):
        return None
    return ToolMessage(content="Error: web search is off for this message, so vehicles can't be looked up.",
                       name=call["name"], tool_call_id=call["id"], status="error")  # fmt: skip


class WebSearchSwitch(AgentMiddleware):
    """Takes the vehicle lookup (a web search) away for a turn whose TurnContext has web_search off: the model
    doesn't get the tool and its system prompt says so (WEB_SEARCH_OFF), and a call to it anyway, e.g. copied
    from an earlier turn, gets an error result."""

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        return handler(_without_web_search(request) if _web_search_off(request.runtime) else request)

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        return await handler(_without_web_search(request) if _web_search_off(request.runtime) else request)

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]
    ) -> ToolMessage | Command:
        return _web_search_refused(request) or handler(request)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]
    ) -> ToolMessage | Command:
        return _web_search_refused(request) or await handler(request)


def guard_input(messages: Sequence[BaseMessage], history_messages: int) -> tuple[list[dict], str]:
    """What the guardrail reads about the latest shopper message: the chat messages before it that the model
    would also see (the same whole turns as RecentHistory), as {"role", "content"}, and the message itself."""
    shown = transcript(recent_turns(messages, history_messages))
    return [{"role": m["role"], "content": m["content"]} for m in shown[:-1]], shown[-1]["content"]


class Guardrail(AgentMiddleware):
    """Before the agent runs, asks the guard about the shopper's message when the turn has the guardrail on
    (TurnContext). A blocked message gets the guardrail's reply for its reason, with the decision in its
    response_metadata["guardrail"], and the turn ends without calling the model."""

    def __init__(self, guard: Guard, history_messages: int):
        super().__init__()
        self.guard = guard
        self.history_messages = history_messages

    @staticmethod
    def _on(runtime: Runtime) -> bool:
        return bool(runtime.context and runtime.context.guardrail)

    def _outcome(self, decision: dict) -> dict | None:
        if not decision["blocked"]:
            return None
        saved = {k: v for k, v in decision.items() if k != "error"}
        return {"messages": [AIMessage(content=REPLIES[decision["reason"]], response_metadata={"guardrail": saved})],
                "jump_to": "end"}  # fmt: skip

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state, runtime: Runtime) -> dict | None:
        if not self._on(runtime):
            return None
        return self._outcome(self.guard.check(*guard_input(state["messages"], self.history_messages)))

    @hook_config(can_jump_to=["end"])
    async def abefore_agent(self, state, runtime: Runtime) -> dict | None:
        if not self._on(runtime):
            return None
        conversation, message = guard_input(state["messages"], self.history_messages)
        return self._outcome(await asyncio.to_thread(self.guard.check, conversation, message))


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


def agent_middleware(limits: AgentSettings, guard: Guard | None = None) -> list[AgentMiddleware]:
    return [
        *([Guardrail(guard, limits.history_messages)] if guard else []),
        RecentHistory(limits.history_messages),
        WebSearchSwitch(),
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
    guard: Guard | None = None,
) -> CompiledStateGraph:
    """Agent with the system prompt from SYSTEM_PROMPT_PATH and the AGENT_* limits; history is kept per thread id.

    With an OpenRouter key (or `guard`), it has the guardrail, which each turn turns on with TurnContext.
    """
    return create_agent(
        model or build_chat_model(settings.llm, settings.openrouter_api_key),
        tools=list(tools),
        system_prompt=load_system_prompt(settings.llm.system_prompt_path),
        middleware=agent_middleware(settings.agent, guard or build_guard(settings.guardrail, settings.openrouter_api_key)),
        checkpointer=checkpointer or InMemorySaver(),
        context_schema=TurnContext,
        name="tire_agent",  # "agent" in the name makes Langfuse type the agent loop as an agent
    )


def has_guardrail(agent: CompiledStateGraph) -> bool:
    return GUARDRAIL_NODE in agent.nodes


def tool_names(agent: CompiledStateGraph) -> set[str]:
    node = agent.nodes.get("tools")
    return set(getattr(node.bound, "tools_by_name", {})) if node else set()


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
    whitespace. A message the guardrail blocked gets its reply, which comes whole.
    """

    def __init__(self):
        self.step = None  # the model step that last wrote visible text

    def __call__(self, chunk: object, metadata: dict) -> str:
        if metadata.get("langgraph_node") == GUARDRAIL_NODE and isinstance(chunk, AIMessage):
            return chunk.text
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


def _turn_metadata(agent: CompiledStateGraph, metadata: dict | None, guardrail: bool, web_search: bool) -> dict:
    return {**(metadata or {}), "guardrail": guardrail and has_guardrail(agent),
            "web_search": web_search and VEHICLE_TOOL in tool_names(agent)}  # fmt: skip


def stream_reply(
    agent: CompiledStateGraph,
    message: str,
    thread_id: str,
    *,
    source: str = "cli",
    metadata: dict | None = None,
    guardrail: bool = False,
    web_search: bool = True,
) -> Iterator[str]:
    """Send one shopper message and yield the assistant's answer text as it is generated."""
    metadata = _turn_metadata(agent, metadata, guardrail, web_search)
    with trace_turn(message, thread_id, source=source, metadata=metadata) as turn:
        answer_text, answer = _AnswerText(), []
        config = _traced_config(thread_id, turn.callbacks)
        context = TurnContext(guardrail=guardrail, web_search=web_search)
        for chunk, chunk_metadata in agent.stream(_user_input(message), config, stream_mode="messages", context=context):
            if text := answer_text(chunk, chunk_metadata):
                answer.append(text)
                yield text
        turn.finish("".join(answer))


def _status(stage: str, text: str) -> dict:
    return {"type": "status", "stage": stage, "text": text}


# Status lines of each tool: (for its call's arguments, for its result).
DESCRIPTIONS = {"search_tires": (describe_search, describe_results), VEHICLE_TOOL: (describe_lookup, describe_lookup_results)}


def _describe_call(call: dict) -> str:
    describe = DESCRIPTIONS.get(call["name"])
    return describe[0](call["args"]) if describe else f"Running {call['name']}"


def _describe_result(message: ToolMessage) -> str:
    describe = DESCRIPTIONS.get(message.name)
    return describe[1](message.content) if describe else f"{message.name} finished"


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
    guardrail: bool = False,
    web_search: bool = True,
) -> AsyncIterator[dict]:
    """One shopper message as a stream of events for the UI, traced as one Langfuse trace.

    Yields {"type": "status", "stage": "checking" | "thinking" | "searching" | "results", "text": ...}
    as the agent works (checking: the guardrail looks at the message, when `guardrail` is on and the
    agent has one), {"type": "tool_call", **tool_call_details} when a tool call has its result, and
    {"type": "token", "text": ...} for each chunk of the answer. When the guardrail blocks the message,
    its reply is the only token, followed by {"type": "guardrail", **decision}. With `web_search` off the
    model can't look vehicles up (WebSearchSwitch).
    """
    metadata = _turn_metadata(agent, metadata, guardrail, web_search)
    with trace_turn(message, thread_id, source=source, metadata=metadata) as turn:
        answer = []
        context = TurnContext(guardrail=metadata["guardrail"], web_search=web_search)
        events = _astream_events(agent, message, _traced_config(thread_id, turn.callbacks), context)
        async for event in events:
            if event["type"] == "token":
                answer.append(event["text"])
            yield event
        turn.finish("".join(answer))


async def _astream_events(
    agent: CompiledStateGraph, message: str, config: RunnableConfig, context: TurnContext
) -> AsyncIterator[dict]:
    guardrail = context.guardrail
    yield _status("checking", "Checking the message…") if guardrail else _status("thinking", "Thinking…")
    answer_text = _AnswerText()
    calls = {}  # this turn's tool calls by id, to pair with their results
    stream = agent.astream(_user_input(message), config, stream_mode=["messages", "updates"], context=context)
    async for mode, chunk in stream:
        if mode == "messages":
            if text := answer_text(*chunk):
                yield {"type": "token", "text": text}
            continue
        # "updates" carry each finished step: model steps with complete tool calls, then tool results.
        for node, update in chunk.items():
            messages = (update or {}).get("messages", [])
            if node == GUARDRAIL_NODE:
                if messages:  # blocked: its reply was the answer
                    yield {"type": "guardrail", **messages[-1].response_metadata["guardrail"]}
                elif guardrail:
                    yield _status("thinking", "Thinking…")
                continue
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


def transcript(messages: Sequence[BaseMessage]) -> list[dict]:
    """The conversation as the shopper saw it: {"role": "user" | "assistant", "content": ...} per message.

    A turn's answer text is joined like the stream sent it, and the answer lists the turn's tool calls
    under "tool_calls" (see tool_call_details). A turn whose model wrote no text has no answer. An
    answer from the guardrail has its decision under "guardrail".
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
            if "guardrail" in message.response_metadata:
                answer["guardrail"] = message.response_metadata["guardrail"]
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
