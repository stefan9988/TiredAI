"""The list of chats: a title and timestamps per conversation. The messages live in LangGraph's checkpoints."""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

import aiosqlite
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from tiredai.agent import aget_transcript, thread_config

TITLE_CHARS = 100


@dataclass(frozen=True)
class Conversation:
    id: str
    title: str
    created_at: str
    updated_at: str


def title_from(message: str) -> str:
    """A chat is named after its first message, on one line and cut to TITLE_CHARS."""
    title = " ".join(message.split())
    return title if len(title) <= TITLE_CHARS else title[: TITLE_CHARS - 1].rstrip() + "…"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ConversationStore:
    """Kept in the history database, on the checkpointer's connection and lock so writes never contend."""

    def __init__(self, conn: aiosqlite.Connection, lock: asyncio.Lock):
        self.conn = conn
        self.lock = lock

    async def setup(self) -> None:
        async with self.lock:
            await self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS conversations_updated_at ON conversations (updated_at);
                """
            )
            await self.conn.commit()

    async def record_turn(self, conversation_id: str, message: str) -> None:
        """Called when the shopper sends a message: the first one names a new chat, every one bumps it to the top."""
        await self._upsert(conversation_id, title_from(message), _now(), _now())

    async def _upsert(self, conversation_id: str, title: str, created_at: str, updated_at: str) -> None:
        async with self.lock:
            await self.conn.execute(
                """
                INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (conversation_id, title, created_at, updated_at),
            )
            await self.conn.commit()

    async def get(self, conversation_id: str) -> Conversation | None:
        async with self.lock, self.conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?", (conversation_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return Conversation(*row) if row else None

    async def list(self) -> list[Conversation]:
        """Most recently active first."""
        async with self.lock, self.conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC, rowid DESC"
        ) as cursor:
            return [Conversation(*row) async for row in cursor]

    async def backfill(self, checkpointer: BaseCheckpointSaver, agent: CompiledStateGraph) -> int:
        """Lists conversations saved before chats had titles. Returns how many were added."""
        async with self.lock, self.conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id NOT IN (SELECT id FROM conversations)"
        ) as cursor:
            missing = [thread_id async for (thread_id,) in cursor]
        added = 0
        for thread_id in missing:
            messages = await aget_transcript(agent, thread_id)
            first_user = next((m["content"] for m in messages if m["role"] == "user"), None)
            if first_user is None:
                continue
            # Checkpoints are listed newest first.
            stamps = [c.checkpoint["ts"] async for c in checkpointer.alist(thread_config(thread_id))]
            await self._upsert(thread_id, title_from(first_user), stamps[-1], stamps[0])
            added += 1
        return added
