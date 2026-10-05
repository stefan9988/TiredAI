"""HTTP API for the tire assistant. Conversation history is stored in SQLite and survives restarts.

    POST /chat                          JSON reply
    POST /chat/stream                   the same reply streamed as Server-Sent Events
    GET  /conversations                 every chat, most recently active first
    GET  /conversations/{id}/messages   a chat's messages, to reopen it
    GET  /benchmarks                    the latest benchmark results per model (benchmarks/results/)
    GET  /health                        service, model and vector store status
    GET  /              chat page (static files in src/tiredai/static)

Run with scripts/serve.py, or: uvicorn tiredai.api:create_app --factory
"""

import asyncio
import logging
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field, field_validator

from tiredai import tracing
from tiredai.agent import aget_transcript, astream_reply, astream_turn, build_agent, trace_metadata
from tiredai.benchmarks import reports
from tiredai.benchmarks.experiments import RESULTS_DIR
from tiredai.config import Settings
from tiredai.conversations import Conversation, ConversationStore
from tiredai.embeddings import build_encoder
from tiredai.search import QueryEncoder, catalog_tools
from tiredai.vectorstore import connect

logger = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 4000
STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    conversation_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Omit to start a new conversation; send the id from an earlier reply to continue it.",
    )

    @field_validator("message")
    @classmethod
    def not_blank(cls, message: str) -> str:
        if not message.strip():
            raise ValueError("message must not be blank")
        return message


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str


class ToolCall(BaseModel):
    """A tool call behind an answer and what the model got back, to check the answer against."""

    id: str | None
    name: str
    args: dict[str, Any] | str = Field(description="The arguments the model sent; their raw text if they could not be parsed.")
    error: str | None = Field(description="The error the model got instead of a result.")
    result: Any = Field(
        description="The tool's output. For search_tires: total_matching, returned, order, the filters as applied and the products."
    )


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list, description="An answer's tool calls, in order.")


class VectorStoreHealth(BaseModel):
    status: Literal["ok", "unavailable"]
    collection: str
    points: int | None = None
    detail: str | None = None


class BenchmarkMetric(BaseModel):
    name: str
    label: str
    description: str
    better: Literal["higher", "lower"] | None = Field(description="Which way the metric is better; None for neither.")


class BenchmarkRow(BaseModel):
    model: str = Field(description="The chat model (agent), embedding model (retrieval; BM25 for sparse ranking) or guard model.")
    embedding_model: str | None = Field(description="Agent: the embedding model of the index it searched.")
    ranking: str | None = Field(description="Retrieval: hybrid, dense or sparse.")
    variant: str | None = Field(description="Guardrail: how much of the conversation it saw: message, recent or full.")
    threshold: float | None = Field(description="Guardrail: messages were blocked at this block score or above.")
    finished_at: str
    runs: int = Field(description="Runs averaged into this row (--repeat).")
    items: int
    failed: int = Field(description="Cases whose run failed, so they have no scores.")
    scores: dict[str, float | None]
    details: dict[str, float | None]
    urls: list[str] = Field(description="The runs in Langfuse.")


class BenchmarkResults(BaseModel):
    scores: list[BenchmarkMetric]
    details: list[BenchmarkMetric]
    rows: list[BenchmarkRow]


class BenchmarksResponse(BaseModel):
    agent: BenchmarkResults
    retrieval: BenchmarkResults
    guardrail: BenchmarkResults


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model: str
    vector_store: VectorStoreHealth


def create_app(
    settings: Settings | None = None,
    *,
    model: BaseChatModel | None = None,
    encoder: QueryEncoder | None = None,
    benchmark_results: Path = RESULTS_DIR,
) -> FastAPI:
    """Build the app; `model` and `encoder` replace the configured models, `benchmark_results` the
    reports folder (used by tests)."""
    settings = settings or Settings.load()
    metadata = trace_metadata(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        tracing.start()
        settings.conversations_path.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(str(settings.conversations_path)) as checkpointer:
            # Create the history tables now, so an unusable database fails at startup, not on the first chat.
            await checkpointer.setup()
            app.state.conversations = ConversationStore(checkpointer.conn, checkpointer.lock)
            await app.state.conversations.setup()
            app.state.vector_store, app.state.vector_store_error = None, None
            try:
                app.state.vector_store = connect(settings)
            except Exception as exc:  # e.g. the local store is locked by a running build_index.py
                app.state.vector_store_error = str(exc)
            # Without an index the agent has no search tool; /health reports why.
            tools = catalog_tools(
                app.state.vector_store,
                settings.qdrant_collection,
                lambda: encoder or build_encoder(settings),
                max_results=settings.agent.max_search_results,
            )
            app.state.agent = build_agent(settings, model=model, tools=tools, checkpointer=checkpointer)
            if added := await app.state.conversations.backfill(checkpointer, app.state.agent):
                logger.info("Added %d earlier conversations to the chat list", added)
            # One turn at a time per conversation, so concurrent requests can't interleave its history.
            app.state.locks = defaultdict(asyncio.Lock)
            try:
                yield
            finally:
                if app.state.vector_store is not None:
                    app.state.vector_store.close()
                tracing.flush()

    app = FastAPI(title="TiredAI", summary="Tire shopping assistant", lifespan=lifespan)

    async def resolve_conversation(body: ChatRequest, request: Request) -> str:
        if body.conversation_id is None:
            return str(uuid.uuid4())
        if await request.app.state.conversations.get(body.conversation_id) is None:
            raise HTTPException(
                404, f"Unknown conversation {body.conversation_id!r}. Omit conversation_id to start a new one."
            )
        return body.conversation_id

    @app.post("/chat")
    async def chat(body: ChatRequest, request: Request, conversation_id: str = Depends(resolve_conversation)) -> ChatResponse:
        """Send a message and get the whole reply at once."""
        await request.app.state.conversations.record_turn(conversation_id, body.message)
        async with request.app.state.locks[conversation_id]:
            try:
                parts = [
                    text
                    async for text in astream_reply(
                        request.app.state.agent, body.message, conversation_id, source="chat", metadata=metadata
                    )
                ]
            except Exception as exc:
                logger.exception("Model request failed")
                raise HTTPException(502, f"Model request failed: {exc}") from exc
        return ChatResponse(conversation_id=conversation_id, reply="".join(parts))

    @app.post("/chat/stream", response_class=EventSourceResponse)
    async def chat_stream(body: ChatRequest, request: Request, conversation_id: str = Depends(resolve_conversation)):
        """Send a message and receive the reply as Server-Sent Events.

        Events: `start` {conversation_id}; then, as the agent works, `status` {stage, text} (stage is
        thinking, searching or results), `tool_call` {id, name, args, error, result} when a tool call
        has its result (like ToolCall), and `token` {text} for each chunk of the answer; then `end`
        {conversation_id, reply}. If the model fails, `error` {message} replaces `end`. The chat is
        listed by /conversations from `start` on.
        """
        await request.app.state.conversations.record_turn(conversation_id, body.message)
        yield ServerSentEvent(event="start", data={"conversation_id": conversation_id})
        parts = []
        async with request.app.state.locks[conversation_id]:
            try:
                async for event in astream_turn(
                    request.app.state.agent, body.message, conversation_id, source="chat-stream", metadata=metadata
                ):
                    kind = event.pop("type")
                    if kind == "token":
                        parts.append(event["text"])
                    yield ServerSentEvent(event=kind, data=event)
            except Exception as exc:
                logger.exception("Model request failed")
                yield ServerSentEvent(event="error", data={"message": f"Model request failed: {exc}"})
                return
        yield ServerSentEvent(event="end", data={"conversation_id": conversation_id, "reply": "".join(parts)})

    @app.get("/conversations")
    async def conversations(request: Request) -> list[Conversation]:
        """Every chat, most recently active first. A chat's title is its first message."""
        return await request.app.state.conversations.list()

    @app.get("/conversations/{conversation_id}/messages", responses={404: {"description": "Unknown conversation"}})
    async def conversation_messages(conversation_id: str, request: Request) -> list[ChatMessage]:
        """A chat's messages in order: the shopper's and the assistant's answers, each with its tool calls."""
        if await request.app.state.conversations.get(conversation_id) is None:
            raise HTTPException(404, f"Unknown conversation {conversation_id!r}")
        return [ChatMessage(**m) for m in await aget_transcript(request.app.state.agent, conversation_id)]

    @app.get("/benchmarks")
    async def benchmarks() -> BenchmarksResponse:
        """The latest result of each benchmarked model, from the reports in benchmarks/results/."""
        rows = await asyncio.to_thread(reports.latest_results, benchmark_results)

        def results(kind: str) -> BenchmarkResults:
            def metrics(group, info):
                return [
                    BenchmarkMetric(name=n, label=label, description=d, better=reports.better(group, n))
                    for n, label, d in info[kind]
                ]

            return BenchmarkResults(
                scores=metrics("scores", reports.SCORES), details=metrics("details", reports.DETAILS), rows=rows[kind]
            )

        return BenchmarksResponse(**{kind: results(kind) for kind in reports.SCORES})

    @app.get("/health")
    async def health(request: Request) -> HealthResponse:
        vector_store = vector_store_health(request.app.state.vector_store, request.app.state.vector_store_error)
        return HealthResponse(
            status="ok" if vector_store.status == "ok" else "degraded",
            model=settings.llm.model,
            vector_store=vector_store,
        )

    @app.get("/", include_in_schema=False)
    async def chat_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def vector_store_health(client, error: str | None) -> VectorStoreHealth:
        collection = settings.qdrant_collection
        if client is None:
            return VectorStoreHealth(status="unavailable", collection=collection, detail=error)
        try:
            if not client.collection_exists(collection):
                return VectorStoreHealth(
                    status="unavailable",
                    collection=collection,
                    detail=f"Collection {collection!r} not found; run scripts/build_index.py",
                )
            return VectorStoreHealth(status="ok", collection=collection, points=client.count(collection).count)
        except Exception as exc:
            return VectorStoreHealth(status="unavailable", collection=collection, detail=str(exc))

    return app
