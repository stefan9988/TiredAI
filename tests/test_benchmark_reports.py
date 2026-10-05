import dataclasses
import json

import pytest
from conftest import fake_model
from fastapi.testclient import TestClient

from tiredai.api import create_app
from tiredai.benchmarks.experiments import RunSummary
from tiredai.benchmarks.reports import SCORES, latest_results
from tiredai.config import AgentSettings, Settings


def agent_run(model: str, passed: float, repeat: int = 1, embedding: str = "openrouter:qwen", **details) -> RunSummary:
    return RunSummary(
        label=model, run_name=f"{model} · now", items=27, failed=0, scores={"passed": passed, "groundedness": 1.0},
        details={"seconds_per_turn": 4.0, **details}, setup={"llm_model": model, "embedding_model": embedding, "repeat": repeat},
        url=f"https://langfuse.example/{model}/{repeat}",
    )  # fmt: skip


def retrieval_run(ranking: str, model: str | None, mrr: float) -> RunSummary:
    setup = {"ranking": ranking} | ({"embedding_model": model} if model else {})
    return RunSummary(label=f"{ranking} · {model}", run_name="r", items=188, failed=0, scores={"reciprocal_rank": mrr}, setup=setup)


def guardrail_run(variant: str, correct: float, threshold: float = 0.5, model: str = "typesafe/jev-1.13") -> RunSummary:
    return RunSummary(
        label=f"{model} · {variant}", run_name="g", items=116, failed=0, scores={"correct": correct, "false_block": 0.0, "caught": correct},
        details={"auc": 1.0, "best_threshold": 0.41, "cost_usd": 0.004}, setup={"guard_model": model, "variant": variant, "threshold": threshold},
    )  # fmt: skip


def write(directory, stem: str, runs: list[RunSummary]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.json").write_text(json.dumps([dataclasses.asdict(r) for r in runs]))


def test_each_model_shows_its_newest_result(tmp_path):
    write(tmp_path, "20261005T090000Z-agent", [agent_run("ling", 0.5), agent_run("qwen", 0.7)])
    write(tmp_path, "20261005T100000Z-agent", [agent_run("ling", 0.9)])

    rows = latest_results(tmp_path)["agent"]

    assert [(r["model"], r["scores"]["passed"], r["finished_at"]) for r in rows] == [
        ("ling", 0.9, "2026-10-05T10:00:00Z"),  # benchmarked again: its newer row, sorted by passed
        ("qwen", 0.7, "2026-10-05T09:00:00Z"),
    ]
    assert rows[0]["embedding_model"] == "openrouter:qwen" and rows[0]["urls"] == ["https://langfuse.example/ling/1"]


def test_a_chat_model_has_a_row_per_embedding_model(tmp_path):
    write(tmp_path, "20261005T090000Z-agent", [agent_run("ling", 0.5, embedding="openrouter:qwen"), agent_run("qwen", 0.7)])
    write(tmp_path, "20261005T100000Z-agent", [agent_run("ling", 0.9, embedding="openrouter:openai/small")])
    write(tmp_path, "20261005T110000Z-agent", [agent_run("ling", 0.6, embedding="openrouter:qwen")])

    rows = latest_results(tmp_path)["agent"]

    assert [(r["model"], r["embedding_model"], r["scores"]["passed"]) for r in rows] == [
        ("ling", "openrouter:openai/small", 0.9),
        ("qwen", "openrouter:qwen", 0.7),
        ("ling", "openrouter:qwen", 0.6),  # benchmarked again with the same embeddings: only this row changed
    ]


def test_repeats_of_a_model_are_averaged(tmp_path):
    write(tmp_path, "20261005T100000Z-agent", [agent_run("ling", 0.8, 1, input_tokens=100), agent_run("ling", 0.6, 2, input_tokens=200)])

    [row] = latest_results(tmp_path)["agent"]

    assert row["runs"] == 2 and row["scores"]["passed"] == pytest.approx(0.7) and row["details"]["input_tokens"] == 150
    assert row["urls"] == ["https://langfuse.example/ling/1", "https://langfuse.example/ling/2"]


def test_retrieval_rows_are_per_model_and_ranking_with_the_apps_ranking_first(tmp_path):
    write(tmp_path, "20261005T100000Z-retrieval", [
        retrieval_run("dense", "qwen", 0.95), retrieval_run("sparse", None, 0.99), retrieval_run("hybrid", "bge", 0.94),
        retrieval_run("hybrid", "qwen", 0.97),
    ])  # fmt: skip

    rows = latest_results(tmp_path)["retrieval"]

    assert [(r["model"], r["ranking"]) for r in rows] == [("qwen", "hybrid"), ("bge", "hybrid"), ("qwen", "dense"), ("BM25", "sparse")]


def test_guardrail_rows_are_per_model_variant_and_threshold_with_the_most_correct_first(tmp_path):
    write(tmp_path, "20261005T090000Z-guardrail", [guardrail_run("message", 0.9), guardrail_run("full", 0.95)])
    write(tmp_path, "20261005T100000Z-guardrail", [
        guardrail_run("message", 0.99), guardrail_run("recent", 1.0), guardrail_run("full", 1.0), guardrail_run("full", 0.97, threshold=0.75),
    ])  # fmt: skip

    rows = latest_results(tmp_path)["guardrail"]

    assert [(r["model"], r["variant"], r["threshold"], r["scores"]["correct"]) for r in rows] == [
        ("typesafe/jev-1.13", "recent", 0.5, 1.0),  # ties: the least conversation first
        ("typesafe/jev-1.13", "full", 0.5, 1.0),  # the newer run replaced the 0.95 one
        ("typesafe/jev-1.13", "message", 0.5, 0.99),
        ("typesafe/jev-1.13", "full", 0.75, 0.97),  # another threshold: its own row
    ]
    assert rows[0]["details"] == {"auc": 1.0, "best_threshold": 0.41, "cost_usd": 0.004} and rows[0]["items"] == 116


def test_unreadable_and_other_files_are_skipped(tmp_path):
    write(tmp_path, "20261005T100000Z-agent", [agent_run("ling", 0.9)])
    (tmp_path / "20261005T110000Z-agent.json").write_text("{not json")
    (tmp_path / "notes.json").write_text("[]")

    assert [r["model"] for r in latest_results(tmp_path)["agent"]] == ["ling"]
    assert latest_results(tmp_path / "missing") == {"agent": [], "retrieval": [], "guardrail": []}


def test_the_api_serves_the_results_with_what_each_score_means(tmp_path):
    results = tmp_path / "results"
    write(results, "20261005T100000Z-agent", [agent_run("ling", 0.9)])
    write(results, "20261005T100000Z-guardrail", [guardrail_run("recent", 1.0)])
    prompt = tmp_path / "system.md"
    prompt.write_text("You are a test assistant.")
    loaded = Settings.load()
    settings = dataclasses.replace(
        loaded, conversations_path=tmp_path / "conversations.sqlite", qdrant_path=tmp_path / "vectorstore", qdrant_url=None,
        llm=dataclasses.replace(loaded.llm, system_prompt_path=prompt), agent=AgentSettings.from_env({}),
    )  # fmt: skip

    with TestClient(create_app(settings, model=fake_model("unused"), benchmark_results=results)) as client:
        body = client.get("/benchmarks").json()

    assert [s["name"] for s in body["agent"]["scores"]] == [name for name, _, _ in SCORES["agent"]]
    assert body["agent"]["rows"][0]["model"] == "ling" and body["agent"]["rows"][0]["scores"]["passed"] == 0.9
    assert body["retrieval"]["rows"] == [] and body["retrieval"]["scores"][0]["label"] == "Hit@1"
    guardrail = body["guardrail"]
    assert [(r["model"], r["variant"], r["threshold"]) for r in guardrail["rows"]] == [("typesafe/jev-1.13", "recent", 0.5)]
    # Which way each metric is better, for highlighting and the first sort: false blocks and times lower, AUC higher.
    better = {m["name"]: m["better"] for m in guardrail["scores"] + guardrail["details"]}
    assert better["correct"] == "higher" and better["false_block"] == "lower" and better["auc"] == "higher"
    assert better["best_threshold"] is None and better["seconds"] == "lower"
    assert {m["name"]: m["better"] for m in body["agent"]["details"]}["seconds_per_turn"] == "lower"
