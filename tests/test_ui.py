import dataclasses
import shutil
import subprocess

import pytest
from conftest import fake_model
from fastapi.testclient import TestClient

from tiredai.api import create_app
from tiredai.config import ROOT, Settings


@pytest.fixture
def client(tmp_path):
    prompt = tmp_path / "system.md"
    prompt.write_text("You are a test assistant.")
    loaded = Settings.load()
    settings = dataclasses.replace(
        loaded,
        conversations_path=tmp_path / "conversations.sqlite",
        qdrant_path=tmp_path / "vectorstore",
        qdrant_url=None,
        llm=dataclasses.replace(loaded.llm, system_prompt_path=prompt),
    )
    with TestClient(create_app(settings, model=fake_model("unused"))) as client:
        yield client


def test_chat_page_is_served_at_the_root(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>TiredAI</title>" in response.text
    assert '<script type="module" src="/static/app.js">' in response.text


@pytest.mark.parametrize("path, media_type", [("app.js", "text/javascript"), ("lib.mjs", "text/javascript"), ("style.css", "text/css")])
def test_page_assets_are_served(client, path, media_type):
    response = client.get(f"/static/{path}")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(media_type)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_page_helpers():
    # Markdown rendering and SSE parsing run in the browser; node runs their tests.
    result = subprocess.run(
        ["node", "--test", "tests/ui/lib.test.mjs"], cwd=ROOT, capture_output=True, text=True, timeout=60
    )

    assert result.returncode == 0, result.stdout + result.stderr
