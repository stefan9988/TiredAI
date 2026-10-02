"""HTTP API for the tire assistant. Conversation history is stored in SQLite and survives restarts.

    POST /chat          JSON reply
    POST /chat/stream   the same reply streamed as Server-Sent Events
    GET  /health        service, model and vector store status

Run with scripts/serve.py, or: uvicorn tiredai.api:create_app --factory
"""

import asyncio
import logging
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field, field_validator

from tiredai.agent import astream_reply, build_agent, thread_config
from tiredai.config import Settings
from tiredai.embeddings import build_encoder
from tiredai.search import QueryEncoder, catalog_tools
from tiredai.vectorstore import connect

logger = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 4000


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


class VectorStoreHealth(BaseModel):
    status: Literal["ok", "unavailable"]
    collection: str
    points: int | None = None
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model: str
    vector_store: VectorStoreHealth


def create_app(
    settings: Settings | None = None, *, model: BaseChatModel | None = None, encoder: QueryEncoder | None = None
) -> FastAPI:
    """Build the app; `model` and `encoder` replace the configured models (used by tests)."""
    settings = settings or Settings.load()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.conversations_path.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(str(settings.conversations_path)) as checkpointer:
            # Create the history tables now, so an unusable database fails at startup, not on the first chat.
            await checkpointer.setup()
            app.state.checkpointer = checkpointer
            app.state.vector_store, app.state.vector_store_error = None, None
            try:
                app.state.vector_store = connect(settings)
            except Exception as exc:  # e.g. the local store is locked by a running build_index.py
                app.state.vector_store_error = str(exc)
            # Without an index the agent has no search tool; /health reports why.
            tools = catalog_tools(
                app.state.vector_store, settings.qdrant_collection, lambda: encoder or build_encoder(settings)
            )
            app.state.agent = build_agent(settings, model=model, tools=tools, checkpointer=checkpointer)
            # One turn at a time per conversation, so concurrent requests can't interleave its history.
            app.state.locks = defaultdict(asyncio.Lock)
            try:
                yield
            finally:
                if app.state.vector_store is not None:
                    app.state.vector_store.close()

    app = FastAPI(title="TiredAI", summary="Tire shopping assistant", lifespan=lifespan)

    async def resolve_conversation(body: ChatRequest, request: Request) -> str:
        if body.conversation_id is None:
            return str(uuid.uuid4())
        if await request.app.state.checkpointer.aget_tuple(thread_config(body.conversation_id)) is None:
            raise HTTPException(
                404, f"Unknown conversation {body.conversation_id!r}. Omit conversation_id to start a new one."
            )
        return body.conversation_id

    @app.post("/chat")
    async def chat(body: ChatRequest, request: Request, conversation_id: str = Depends(resolve_conversation)) -> ChatResponse:
        """Send a message and get the whole reply at once."""
        async with request.app.state.locks[conversation_id]:
            try:
                parts = [text async for text in astream_reply(request.app.state.agent, body.message, conversation_id)]
            except Exception as exc:
                logger.exception("Model request failed")
                raise HTTPException(502, f"Model request failed: {exc}") from exc
        return ChatResponse(conversation_id=conversation_id, reply="".join(parts))

    @app.post("/chat/stream", response_class=EventSourceResponse)
    async def chat_stream(body: ChatRequest, request: Request, conversation_id: str = Depends(resolve_conversation)):
        """Send a message and receive the reply as Server-Sent Events.

        Events: `start` {conversation_id}, then `token` {text} for each chunk, then `end`
        {conversation_id, reply}. If the model fails, `error` {message} replaces `end`.
        """
        yield ServerSentEvent(event="start", data={"conversation_id": conversation_id})
        parts = []
        async with request.app.state.locks[conversation_id]:
            try:
                async for text in astream_reply(request.app.state.agent, body.message, conversation_id):
                    parts.append(text)
                    yield ServerSentEvent(event="token", data={"text": text})
            except Exception as exc:
                logger.exception("Model request failed")
                yield ServerSentEvent(event="error", data={"message": f"Model request failed: {exc}"})
                return
        yield ServerSentEvent(event="end", data={"conversation_id": conversation_id, "reply": "".join(parts)})

    @app.get("/health")
    async def health(request: Request) -> HealthResponse:
        vector_store = vector_store_health(request.app.state.vector_store, request.app.state.vector_store_error)
        return HealthResponse(
            status="ok" if vector_store.status == "ok" else "degraded",
            model=settings.llm.model,
            vector_store=vector_store,
        )

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
