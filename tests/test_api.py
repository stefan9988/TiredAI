import dataclasses
import json
import sqlite3
import uuid

import pytest
from conftest import FailingModel, FakeEncoder, fake_model, raw_frame
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from tiredai.api import MAX_MESSAGE_CHARS, create_app
from tiredai.config import Settings
from tiredai.documents import products
from tiredai.preprocessing import normalize
from tiredai.vectorstore import index_products

PROMPT = "You are a test assistant."


@pytest.fixture
def settings(tmp_path):
    prompt = tmp_path / "system.md"
    prompt.write_text(PROMPT)
    loaded = Settings.load()
    return dataclasses.replace(
        loaded,
        conversations_path=tmp_path / "conversations.sqlite",
        qdrant_path=tmp_path / "vectorstore",
        qdrant_url=None,
        qdrant_collection="tires",
        llm=dataclasses.replace(loaded.llm, system_prompt_path=prompt),
    )


def serve(settings, model):
    return TestClient(create_app(settings, model=model))


def sse_events(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_first_message_starts_a_conversation(settings):
    with serve(settings, fake_model("Hello shopper.")) as client:
        response = client.post("/chat", json={"message": "Hi"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Hello shopper."
    uuid.UUID(body["conversation_id"])


def test_follow_up_messages_see_the_history(settings):
    model = fake_model("Which size?", "Here are cheaper ones.")
    with serve(settings, model) as client:
        conversation_id = client.post("/chat", json={"message": "I need tires"}).json()["conversation_id"]
        response = client.post("/chat", json={"message": "Cheaper please", "conversation_id": conversation_id})

    assert response.json()["conversation_id"] == conversation_id
    assert [m.content for m in model.prompts[1]] == [PROMPT, "I need tires", "Which size?", "Cheaper please"]


def test_history_survives_a_restart(settings):
    with serve(settings, fake_model("First answer.")) as client:
        conversation_id = client.post("/chat", json={"message": "I need 205/55R16"}).json()["conversation_id"]

    model = fake_model("Second answer.")
    with serve(settings, model) as client:
        response = client.post("/chat", json={"message": "Cheaper?", "conversation_id": conversation_id})

    assert response.status_code == 200
    assert [m.content for m in model.prompts[0]] == [PROMPT, "I need 205/55R16", "First answer.", "Cheaper?"]


def test_history_tables_are_created_at_startup(settings):
    with serve(settings, fake_model("unused")):
        with sqlite3.connect(settings.conversations_path) as db:
            tables = {name for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}

    assert {"checkpoints", "writes"} <= tables


def test_unusable_history_database_fails_at_startup(settings, tmp_path):
    not_a_database = tmp_path / "conversations.sqlite"
    not_a_database.write_text("this is not a SQLite file")

    with pytest.raises(sqlite3.DatabaseError):
        with serve(dataclasses.replace(settings, conversations_path=not_a_database), fake_model("unused")):
            pass


def test_conversations_are_kept_apart(settings):
    model = fake_model("One.", "Two.")
    with serve(settings, model) as client:
        first = client.post("/chat", json={"message": "First chat"}).json()["conversation_id"]
        second = client.post("/chat", json={"message": "Second chat"}).json()["conversation_id"]

    assert first != second
    assert [m.content for m in model.prompts[1]] == [PROMPT, "Second chat"]


@pytest.mark.parametrize("path", ["/chat", "/chat/stream"])
def test_unknown_conversation_is_rejected(settings, path):
    with serve(settings, fake_model("unused")) as client:
        response = client.post(path, json={"message": "Hi", "conversation_id": "does-not-exist"})

    assert response.status_code == 404
    assert "Unknown conversation" in response.json()["detail"]


@pytest.mark.parametrize("message", ["", "   ", "x" * (MAX_MESSAGE_CHARS + 1)])
def test_invalid_messages_are_rejected(settings, message):
    with serve(settings, fake_model("unused")) as client:
        assert client.post("/chat", json={"message": message}).status_code == 422


def test_stream_sends_start_tokens_and_end(settings):
    with serve(settings, fake_model("All season tires work year round.")) as client:
        response = client.post("/chat/stream", json={"message": "What are all season tires?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response.text)
    names = [name for name, _ in events]
    assert names[0] == "start" and names[-1] == "end" and set(names[1:-1]) == {"token"}
    assert len(names) > 3
    conversation_id = events[0][1]["conversation_id"]
    reply = "".join(data["text"] for name, data in events if name == "token")
    assert reply == "All season tires work year round."
    assert events[-1][1] == {"conversation_id": conversation_id, "reply": reply}


def test_stream_and_json_share_the_conversation(settings):
    model = fake_model("Streamed answer.", "JSON answer.")
    with serve(settings, model) as client:
        conversation_id = sse_events(client.post("/chat/stream", json={"message": "Hi"}).text)[0][1]["conversation_id"]
        client.post("/chat", json={"message": "Again", "conversation_id": conversation_id})

    assert [m.content for m in model.prompts[1]] == [PROMPT, "Hi", "Streamed answer.", "Again"]


def test_model_failure_returns_502(settings):
    with serve(settings, FailingModel(messages=iter([]))) as client:
        response = client.post("/chat", json={"message": "Hi"})

    assert response.status_code == 502
    assert "provider unavailable" in response.json()["detail"]


def test_model_failure_while_streaming_sends_an_error_event(settings):
    with serve(settings, FailingModel(messages=iter([]))) as client:
        events = sse_events(client.post("/chat/stream", json={"message": "Hi"}).text)

    assert [name for name, _ in events] == ["start", "error"]
    assert "provider unavailable" in events[1][1]["message"]


def test_health_reports_a_missing_index(settings):
    with serve(settings, fake_model("unused")) as client:
        body = client.get("/health").json()

    assert body["status"] == "degraded"
    assert body["model"] == settings.llm.model
    assert body["vector_store"]["status"] == "unavailable"
    assert "build_index.py" in body["vector_store"]["detail"]


def test_health_reports_the_indexed_products(settings):
    store = QdrantClient(path=str(settings.qdrant_path))
    index_products(store, "tires", products(normalize(raw_frame({}, {}))), FakeEncoder())
    store.close()

    with serve(settings, fake_model("unused")) as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["vector_store"] == {"status": "ok", "collection": "tires", "points": 2, "detail": None}
