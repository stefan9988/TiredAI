"""Running a benchmark as a Langfuse experiment, and reporting it.

The benchmark files in benchmarks/ are the source of truth for the cases. With Langfuse keys set
they are synced to a Langfuse dataset first (new and changed cases are upserted, removed ones
archived), and every run becomes a dataset run: one trace per case, one score per metric, and the
runs of different models side by side in the dataset's Experiments view. Without keys the same
experiment runs locally. Either way a summary is printed and saved to benchmarks/results/.
"""

import dataclasses
import hashlib
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from langfuse import Langfuse
from langfuse.api import NotFoundError
from qdrant_client import QdrantClient

from tiredai.benchmarks.metrics import mean
from tiredai.config import ROOT, Settings
from tiredai.documents import products
from tiredai.embeddings import Encoder, build_encoder
from tiredai.vectorstore import index_products

BENCHMARKS_DIR = ROOT / "benchmarks"
RESULTS_DIR = BENCHMARKS_DIR / "results"
PROVIDERS = ("fastembed", "openrouter")
ACTIVE, ARCHIVED = "ACTIVE", "ARCHIVED"
COLLECTION = "benchmark"


def parse_embedding(spec: str, default_provider: str) -> tuple[str, str]:
    """'openrouter:qwen/qwen3-embedding-8b' -> (provider, model); without a known provider, `default_provider`'s."""
    provider, separator, model = spec.partition(":")
    if separator and provider in PROVIDERS:
        return provider, model
    return default_provider, spec


def with_embedding(settings: Settings, spec: str | None) -> Settings:
    if not spec:
        return settings
    provider, model = parse_embedding(spec, settings.embedding_provider)
    return dataclasses.replace(settings, embedding_provider=provider, embedding_model=model)


def with_llm(settings: Settings, model: str | None) -> Settings:
    return dataclasses.replace(settings, llm=dataclasses.replace(settings.llm, model=model)) if model else settings


def embedding_label(settings: Settings) -> str:
    return f"{settings.embedding_provider}:{settings.embedding_model}"


def memory_index(settings: Settings, progress=None) -> tuple[QdrantClient, Encoder]:
    """The processed catalog indexed in memory with the settings' embedding model.

    The app's index in data/vectorstore/ is left alone (and may stay open in a running server). Vectors
    of remote models come from the embedding cache, so only a model's first benchmark calls its API.
    """
    encoder = build_encoder(settings)
    client = QdrantClient(":memory:")
    index_products(client, COLLECTION, products(pd.read_parquet(settings.processed_data_path)), encoder, on_batch=progress)
    return client, encoder


def git_revision(root: Path = ROOT) -> str:
    """The commit the benchmark ran on, marked -dirty when tracked files were changed since.

    Untracked files don't count: earlier result files in benchmarks/results/ aren't code.
    """
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True,
                                text=True, check=True)  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit.stdout.strip() + ("-dirty" if status.stdout.strip() else "")


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")


@dataclass(frozen=True)
class Case:
    """One benchmark case, as it becomes a dataset item."""

    id: str
    input: Any
    expected_output: Any
    metadata: dict


def local_items(cases: list[Case]) -> list[dict]:
    return [{"input": c.input, "expected_output": c.expected_output, "metadata": c.metadata} for c in cases]


def item_id(dataset: str, case_id: str) -> str:
    # Langfuse item ids are unique across the whole project, so they carry the dataset name.
    return f"{dataset}--{case_id}"


def sync_dataset(langfuse: Langfuse, name: str, description: str, cases: list[Case]) -> list:
    """Make the Langfuse dataset `name` hold exactly `cases`; returns its items in the order of `cases`."""
    try:
        langfuse.api.datasets.get(dataset_name=name)
        existing = {item.id: item for item in langfuse.get_dataset(name).items}
    except NotFoundError:
        langfuse.create_dataset(name=name, description=description)
        existing = {}

    wanted = {item_id(name, c.id): c for c in cases}
    for key, case in wanted.items():
        current = existing.get(key)
        content = (case.input, case.expected_output, case.metadata)
        if current is None or current.status != ACTIVE or (current.input, current.expected_output, current.metadata) != content:
            langfuse.create_dataset_item(
                dataset_name=name, id=key, input=case.input, expected_output=case.expected_output, metadata=case.metadata, status=ACTIVE
            )
    for key, item in existing.items():
        if key not in wanted and item.status != ARCHIVED:
            langfuse.create_dataset_item(
                dataset_name=name, id=key, input=item.input, expected_output=item.expected_output, metadata=item.metadata, status=ARCHIVED
            )

    items = {item.id: item for item in langfuse.get_dataset(name).items}
    return [items[key] for key in wanted]


def item_value(item, name: str):
    """A field of an experiment item, which is a dict for local data and a DatasetItem for Langfuse data."""
    return item.get(name) if isinstance(item, dict) else getattr(item, name)


@dataclass
class RunSummary:
    label: str  # what this run measured, e.g. "hybrid · openrouter:qwen/qwen3-embedding-8b"
    run_name: str
    items: int
    failed: int  # cases whose task raised: they have no output and no scores
    scores: dict[str, float]  # mean of each score over the cases that have it
    details: dict[str, Any] = field(default_factory=dict)  # other figures, e.g. latency and tokens
    setup: dict[str, Any] = field(default_factory=dict)  # what was benchmarked: llm_model, embedding_model, ranking, repeat
    url: str | None = None
    results: list[dict] = field(default_factory=list)  # every case with its output and scores


def summarize(label: str, result, items: int, details: dict | None = None, setup: dict | None = None) -> RunSummary:
    """The scores of a langfuse ExperimentResult averaged per name."""
    values: dict[str, list[float]] = defaultdict(list)
    results = []
    for item_result in result.item_results:
        scores = {}
        for evaluation in item_result.evaluations:
            if isinstance(evaluation.value, bool | int | float):
                values[evaluation.name].append(float(evaluation.value))
                scores[evaluation.name] = evaluation.value
        metadata = item_value(item_result.item, "metadata") or {}
        results.append({"case": metadata.get("case"), "output": item_result.output, "scores": scores,
                        "comments": {e.name: e.comment for e in item_result.evaluations if e.comment}})  # fmt: skip
    return RunSummary(
        label=label,
        run_name=result.run_name,
        items=items,
        failed=items - len(result.item_results),
        scores={name: mean(v) for name, v in values.items()},
        details=details or {},
        setup=setup or {},
        url=result.dataset_run_url,
        results=results,
    )


def _cell(value) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 10 else f"{value:,.0f}"
    return str(value)


def markdown_table(summaries: list[RunSummary], score_order: list[str]) -> str:
    """One row per run: its scores in `score_order` (then any others), failures and details."""
    names = [n for n in score_order if any(n in s.scores for s in summaries)]
    names += sorted({n for s in summaries for n in s.scores} - set(names))
    detail_names = list(dict.fromkeys(n for s in summaries for n in s.details))
    header = ["run", *names, "failed", *detail_names]
    rows = [
        [s.label, *(_cell(s.scores.get(n)) for n in names), f"{s.failed}/{s.items}", *(_cell(s.details.get(n)) for n in detail_names)]
        for s in summaries
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join([*lines, *("| " + " | ".join(row) + " |" for row in rows)])


def write_report(kind: str, title: str, summaries: list[RunSummary], score_order: list[str], notes: list[str],
                 directory: Path = RESULTS_DIR) -> tuple[Path, Path]:  # fmt: skip
    """benchmarks/results/<time>-<kind>.md (the summary) and .json (every case with its output and scores)."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{kind}"
    links = [f"- {s.label}: {s.url}" for s in summaries if s.url]
    markdown = "\n\n".join(
        part
        for part in (
            f"# {title}",
            "\n".join(f"- {note}" for note in notes),
            markdown_table(summaries, score_order),
            "Langfuse runs:\n" + "\n".join(links) if links else "",
        )
        if part
    )
    md_path, json_path = directory / f"{stem}.md", directory / f"{stem}.json"
    md_path.write_text(markdown + "\n")
    json_path.write_text(json.dumps([dataclasses.asdict(s) for s in summaries], indent=2, ensure_ascii=False, default=str) + "\n")
    return md_path, json_path
