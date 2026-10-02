import dataclasses
import json
import sqlite3
import uuid

import pytest
from conftest import FailingModel, FakeEncoder, ToolCallingModel, fake_model, raw_frame, tool_call, unparsable_call
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from qdrant_client import QdrantClient

from tiredai.agent import SEGMENT_SEPARATOR, build_agent, stream_reply
from tiredai.api import MAX_MESSAGE_CHARS, create_app
from tiredai.config import AgentSettings, Settings
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
        agent=AgentSettings.from_env({}),
    )


def with_limits(settings, **limits):
    return dataclasses.replace(settings, agent=dataclasses.replace(settings.agent, **limits))


def index_catalog(settings, *rows: dict):
    store = QdrantClient(path=str(settings.qdrant_path))
    index_products(store, "tires", products(normalize(raw_frame(*rows))), FakeEncoder())
    store.close()


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
    assert names[:2] == ["start", "status"] and names[-1] == "end" and set(names[2:-1]) == {"token"}
    assert len(names) > 4
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

    assert [name for name, _ in events] == ["start", "status", "error"]
    assert "provider unavailable" in events[-1][1]["message"]


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


def test_agent_gets_the_search_tool_when_the_index_exists(settings):
    store = QdrantClient(path=str(settings.qdrant_path))
    index_products(store, "tires", products(normalize(raw_frame({}))), FakeEncoder())
    store.close()
    model = ToolCallingModel(messages=iter([tool_call("search_tires", size="205/55R15"), AIMessage(content="Found one.")]))

    with TestClient(create_app(settings, model=model, encoder=FakeEncoder())) as client:
        response = client.post("/chat", json={"message": "205/55R15 tires?"})

    assert response.json()["reply"] == "Found one."
    [result] = [m for m in model.prompts[1] if isinstance(m, ToolMessage)]
    assert json.loads(result.content)["total_matching"] == 1


def test_stream_reports_status_while_searching(settings):
    store = QdrantClient(path=str(settings.qdrant_path))
    index_products(store, "tires", products(normalize(raw_frame({}))), FakeEncoder())
    store.close()
    model = ToolCallingModel(
        messages=iter([tool_call("search_tires", size="205/55R15", max_price=60), AIMessage(content="One tire fits.")])
    )

    with TestClient(create_app(settings, model=model, encoder=FakeEncoder())) as client:
        events = sse_events(client.post("/chat/stream", json={"message": "205/55R15 under $60?"}).text)

    statuses = [data["text"] for name, data in events if name == "status"]
    assert statuses == ["Thinking…", "Searching the catalog: 205/55R15 · up to $60", "Found 1 tire", "Thinking…"]
    assert [name for name, _ in events if name != "status"] == ["start", "tool_call", "token", "end"]
    assert events[-1][1]["reply"] == "One tire fits."


def test_stream_sends_each_search_with_the_data_the_model_got(settings):
    index_catalog(settings, {}, {"size": "215/60R16"})
    model = ToolCallingModel(
        messages=iter([tool_call("search_tires", size="205 55 15", max_price=60), AIMessage(content="One tire fits.")])
    )

    with TestClient(create_app(settings, model=model, encoder=FakeEncoder())) as client:
        events = sse_events(client.post("/chat/stream", json={"message": "205/55R15 under $60?"}).text)

    [search] = [data for name, data in events if name == "tool_call"]
    [sent] = [m for m in model.prompts[1] if isinstance(m, ToolMessage)]
    assert search["args"] == {"size": "205 55 15", "max_price": 60}  # as the model asked
    assert search["result"]["filters"] == {"size": "205/55R15", "price": {"min": None, "max": 60}}  # as applied
    assert search["result"] == json.loads(sent.content) and search["error"] is None
    assert search["result"]["products"][0]["name"] == "Accelera Phi-R 205/55R15 92V XL"
    names = [name for name, _ in events]
    assert names.index("tool_call") < names.index("token")  # shown before the answer is written


def test_failed_searches_are_sent_with_their_error(settings):
    index_catalog(settings, {})
    model = ToolCallingModel(
        messages=iter(
            [
                unparsable_call("size: 205/55R15"),
                tool_call("search_tires", size="999/99R99"),
                AIMessage(content="That size isn't sold here."),
            ]
        )
    )

    with TestClient(create_app(settings, model=model, encoder=FakeEncoder())) as client:
        events = sse_events(client.post("/chat/stream", json={"message": "999/99R99?"}).text)

    unparsable, unknown_size = [data for name, data in events if name == "tool_call"]
    assert unparsable["args"] == "size: 205/55R15" and "could not be parsed" in unparsable["error"]
    assert unknown_size["args"] == {"size": "999/99R99"} and "not in the catalog" in unknown_size["error"]
    assert unparsable["result"] is None and unknown_size["result"] is None


def test_answer_text_around_a_search_is_separated(settings):
    store = QdrantClient(path=str(settings.qdrant_path))
    index_products(store, "tires", products(normalize(raw_frame({}))), FakeEncoder())
    store.close()
    search = {"name": "search_tires", "args": {"size": "205/55R15"}, "id": "call-1"}
    model = ToolCallingModel(
        messages=iter([AIMessage(content="Let me check.", tool_calls=[search]), AIMessage(content="One tire fits.")])
    )

    with TestClient(create_app(settings, model=model, encoder=FakeEncoder())) as client:
        events = sse_events(client.post("/chat/stream", json={"message": "205/55R15?"}).text)
        conversation_id = events[0][1]["conversation_id"]
        saved = client.get(f"/conversations/{conversation_id}/messages").json()

    expected = "Let me check." + SEGMENT_SEPARATOR + "One tire fits."
    assert "".join(data["text"] for name, data in events if name == "token") == expected
    assert events[-1][1]["reply"] == expected
    assert saved[1]["content"] == expected


def test_stream_without_tools_only_reports_thinking(settings):
    with serve(settings, fake_model("UTQG is a grading system.")) as client:
        events = sse_events(client.post("/chat/stream", json={"message": "What is UTQG?"}).text)

    assert [data for name, data in events if name == "status"] == [{"stage": "thinking", "text": "Thinking…"}]


def test_chats_are_listed_by_their_first_message(settings):
    with serve(settings, fake_model("unused")) as client:
        assert client.get("/conversations").json() == []

    with serve(settings, fake_model("Which size?", "Here are cheaper ones.", "UTQG is a grade.")) as client:
        first = client.post("/chat", json={"message": "I need\n  winter tires"}).json()["conversation_id"]
        client.post("/chat", json={"message": "Cheaper please", "conversation_id": first})
        second = sse_events(client.post("/chat/stream", json={"message": "What is UTQG?"}).text)[0][1]["conversation_id"]
        listed = client.get("/conversations").json()

    assert [(c["id"], c["title"]) for c in listed] == [(second, "What is UTQG?"), (first, "I need winter tires")]
    assert listed[1]["created_at"] < listed[1]["updated_at"]


def test_continuing_a_chat_moves_it_to_the_top(settings):
    with serve(settings, fake_model("One.", "Two.", "Three.")) as client:
        first = client.post("/chat", json={"message": "First"}).json()["conversation_id"]
        client.post("/chat", json={"message": "Second"})
        client.post("/chat/stream", json={"message": "Back to the first", "conversation_id": first})
        listed = client.get("/conversations").json()

    assert [c["title"] for c in listed] == ["First", "Second"]


def test_a_chat_can_be_reopened_after_a_restart(settings):
    with serve(settings, fake_model("Which size?", "Here are cheaper ones.")) as client:
        conversation_id = client.post("/chat", json={"message": "I need tires"}).json()["conversation_id"]
        client.post("/chat", json={"message": "Cheaper please", "conversation_id": conversation_id})

    with serve(settings, fake_model("unused")) as client:
        listed = client.get("/conversations").json()
        response = client.get(f"/conversations/{conversation_id}/messages")

    assert [c["id"] for c in listed] == [conversation_id]
    assert response.json() == [
        {"role": "user", "content": "I need tires", "tool_calls": []},
        {"role": "assistant", "content": "Which size?", "tool_calls": []},
        {"role": "user", "content": "Cheaper please", "tool_calls": []},
        {"role": "assistant", "content": "Here are cheaper ones.", "tool_calls": []},
    ]


def test_messages_of_an_unknown_chat_are_404(settings):
    with serve(settings, fake_model("unused")) as client:
        response = client.get("/conversations/does-not-exist/messages")

    assert response.status_code == 404
    assert "Unknown conversation" in response.json()["detail"]


def test_reopened_chats_show_the_searches_behind_each_answer(settings):
    index_catalog(settings, {})
    model = ToolCallingModel(messages=iter([tool_call("search_tires", size="205/55R15"), AIMessage(content="Found one.")]))

    with TestClient(create_app(settings, model=model, encoder=FakeEncoder())) as client:
        events = sse_events(client.post("/chat/stream", json={"message": "205/55R15 tires?"}).text)
        conversation_id = events[0][1]["conversation_id"]

    with serve(settings, fake_model("unused")) as client:  # after a restart
        question, answer = client.get(f"/conversations/{conversation_id}/messages").json()

    assert question == {"role": "user", "content": "205/55R15 tires?", "tool_calls": []}
    assert answer["content"] == "Found one."
    [streamed] = [data for name, data in events if name == "tool_call"]
    assert answer["tool_calls"] == [streamed]
    assert answer["tool_calls"][0]["result"]["total_matching"] == 1


def test_a_chat_whose_first_reply_failed_is_listed_and_can_continue(settings):
    with serve(settings, FailingModel(messages=iter([]))) as client:
        conversation_id = sse_events(client.post("/chat/stream", json={"message": "Hi"}).text)[0][1]["conversation_id"]

    with serve(settings, fake_model("Hello.")) as client:
        assert [c["title"] for c in client.get("/conversations").json()] == ["Hi"]
        response = client.post("/chat", json={"message": "Hello?", "conversation_id": conversation_id})

    assert response.status_code == 200
    assert response.json()["reply"] == "Hello."


def test_chats_from_before_titles_existed_are_listed(settings):
    # History written by the agent alone, like the API did before it kept a chat list.
    with SqliteSaver.from_conn_string(str(settings.conversations_path)) as checkpointer:
        agent = build_agent(settings, model=fake_model("Which size?", "Noted."), checkpointer=checkpointer)
        list(stream_reply(agent, "I need tires", "old-chat"))
        list(stream_reply(agent, "205/55R16", "old-chat"))

    with serve(settings, fake_model("unused")) as client:
        [listed] = client.get("/conversations").json()
        messages = client.get("/conversations/old-chat/messages").json()

    assert (listed["id"], listed["title"]) == ("old-chat", "I need tires")
    assert listed["created_at"] < listed["updated_at"]
    assert [m["content"] for m in messages] == ["I need tires", "Which size?", "205/55R16", "Noted."]


def test_streamed_turns_send_the_model_only_recent_history(settings):
    model = fake_model("One.", "Two.", "Three.")
    with serve(with_limits(settings, history_messages=3), model) as client:
        conversation_id = None
        for message in ("First", "Second", "Third"):
            events = sse_events(
                client.post("/chat/stream", json={"message": message, "conversation_id": conversation_id}).text
            )
            conversation_id = events[0][1]["conversation_id"]
        saved = client.get(f"/conversations/{conversation_id}/messages").json()

    assert [m.content for m in model.prompts[-1]] == [PROMPT, "Second", "Two.", "Third"]
    assert [m["content"] for m in saved] == ["First", "One.", "Second", "Two.", "Third", "Three."]


def test_searches_over_the_limit_are_not_run(settings):
    index_catalog(settings, {})
    search = lambda i: {"name": "search_tires", "args": {"size": "205/55R15"}, "id": f"call-{i}"}  # noqa: E731
    model = ToolCallingModel(
        messages=iter(
            [
                AIMessage(content="", tool_calls=[search(1)]),
                AIMessage(content="", tool_calls=[search(2), search(3)]),
                AIMessage(content="Here is what I found."),
                AIMessage(content="", tool_calls=[search(4)]),  # the next message may search again
                AIMessage(content="Still there."),
            ]
        )
    )

    with TestClient(create_app(with_limits(settings, max_tool_calls=2), model=model, encoder=FakeEncoder())) as client:
        events = sse_events(client.post("/chat/stream", json={"message": "205/55R15?"}).text)
        conversation_id = events[0][1]["conversation_id"]
        client.post("/chat", json={"message": "Again?", "conversation_id": conversation_id})

    statuses = [data["text"] for name, data in events if name == "status"]
    assert statuses.count("Found 1 tire") == 2 and "Search limit reached" in statuses
    assert events[-1][1]["reply"] == "Here is what I found."
    results = {m.tool_call_id: m for m in model.prompts[2] if isinstance(m, ToolMessage)}
    assert json.loads(results["call-1"].content)["total_matching"] == 1
    assert json.loads(results["call-2"].content)["total_matching"] == 1
    assert "limit exceeded" in results["call-3"].content and results["call-3"].status == "error"
    next_turn = {m.tool_call_id: m for m in model.prompts[4] if isinstance(m, ToolMessage)}
    assert json.loads(next_turn["call-4"].content)["total_matching"] == 1


def test_searches_return_the_configured_number_of_products(settings):
    index_catalog(settings, {}, {}, {})
    model = ToolCallingModel(messages=iter([tool_call("search_tires", size="205/55R15"), AIMessage(content="Two shown.")]))

    with TestClient(create_app(with_limits(settings, max_search_results=2), model=model, encoder=FakeEncoder())) as client:
        client.post("/chat", json={"message": "205/55R15?"})

    [result] = [json.loads(m.content) for m in model.prompts[1] if isinstance(m, ToolMessage)]
    assert (result["total_matching"], result["returned"]) == (3, 2)


def test_a_crashing_search_is_reported_to_the_model_while_streaming(settings):
    index_catalog(settings, {})

    class BrokenEncoder(FakeEncoder):
        def encode_query(self, text):
            raise RuntimeError("encoder crashed")

    model = ToolCallingModel(
        messages=iter([tool_call("search_tires", query="quiet tires"), AIMessage(content="I can't search right now.")])
    )

    with TestClient(create_app(settings, model=model, encoder=BrokenEncoder())) as client:
        events = sse_events(client.post("/chat/stream", json={"message": "Quiet tires?"}).text)

    assert [data["text"] for name, data in events if name == "status"][-2:] == ["The search failed", "Thinking…"]
    assert events[-1] == ("end", {"conversation_id": events[0][1]["conversation_id"], "reply": "I can't search right now."})
    [result] = [m for m in model.prompts[1] if isinstance(m, ToolMessage)]
    assert "RuntimeError: encoder crashed" in result.content


def test_an_unparsable_call_is_sent_again_while_streaming(settings):
    index_catalog(settings, {})
    model = ToolCallingModel(
        messages=iter([unparsable_call("size: 205/55R15"), tool_call("search_tires", size="205/55R15"), AIMessage(content="Found one.")])
    )

    with TestClient(create_app(settings, model=model, encoder=FakeEncoder())) as client:
        events = sse_events(client.post("/chat/stream", json={"message": "205/55R15?"}).text)

    statuses = [data["text"] for name, data in events if name == "status"]
    assert statuses == ["Thinking…", "The search failed", "Searching the catalog: 205/55R15", "Found 1 tire", "Thinking…"]
    assert events[-1][1]["reply"] == "Found one."


def test_calls_that_stay_unparsable_end_the_stream_with_an_error(settings):
    index_catalog(settings, {})
    model = ToolCallingModel(messages=iter([unparsable_call("size: 205/55R15", f"bad-{i}") for i in range(3)]))

    with TestClient(create_app(settings, model=model, encoder=FakeEncoder())) as client:
        events = sse_events(client.post("/chat/stream", json={"message": "205/55R15?"}).text)

    assert events[-1][0] == "error" and "could not be parsed 3 times" in events[-1][1]["message"]
