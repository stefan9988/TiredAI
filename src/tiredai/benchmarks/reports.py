"""The latest benchmark results per model, read from the reports in benchmarks/results/, for the chat page.

Every benchmark run writes <time>-<kind>.json (see experiments.write_report). A model's row comes
from the newest report that has it, so benchmarking one model again replaces only its row. Repeats
of a model in one report (--repeat) are averaged into one row.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from tiredai.benchmarks.metrics import mean

logger = logging.getLogger(__name__)

# (name, column label, what it measures), in the order the page shows them.
SCORES = {
    "agent": [
        ("passed", "Passed", "Conversations where every check of every turn passed"),
        ("intent_accuracy", "Intent", "Turns that took their flow's route: no search for education and off-topic, a "
                                      "search for the named product, a search with the size (or a question first)"),
        ("retrieval_hit@3", "Hit@3", "The requested product, or a tire meeting the constraints and in stock, is in "
                                     "the top 3 of a search"),
        ("filters_applied", "Filters", "The shopper's hard constraints the searches applied as filters"),
        ("constraint_correctness", "Constraints", "Recommended tires that meet the constraints and are in stock"),
        ("groundedness", "Grounded", "Prices, SKUs and specs in the answers that match the products the search returned"),
        ("answer_checks", "Answers", "Required facts present, forbidden phrases absent, missing products called missing"),
    ],
    "retrieval": [
        ("hit@1", "Hit@1", "Product queries whose product ranks first"),
        ("hit@3", "Hit@3", "Product queries whose product is in the top 3"),
        ("hit@10", "Hit@10", "Product queries whose product is in the top 10"),
        ("reciprocal_rank", "MRR", "Mean of 1 / rank of the first relevant product (product queries)"),
        ("precision@10", "P@10", "Share of the top 10 matching every attribute of a descriptive query"),
        ("ndcg@10", "nDCG@10", "Ranking quality of the top 10 for descriptive queries, with partial matches counting partly"),
    ],
}  # fmt: skip
DETAILS = {
    "agent": [
        ("seconds_per_turn", "s / turn", "Mean time to answer a message"),
        ("p95_seconds", "p95 s", "95% of messages were answered within this time"),
        ("input_tokens", "Tokens in", "Prompt tokens over the whole run"),
        ("output_tokens", "Tokens out", "Completion tokens over the whole run"),
    ],
    "retrieval": [("seconds_per_query", "s / query", "Mean time per query, embedding included")],
}
RANKINGS = ("hybrid", "dense", "sparse")


def _finished_at(stem: str) -> str | None:
    try:
        return datetime.strptime(stem.partition("-")[0], "%Y%m%dT%H%M%SZ").strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None


def _key(kind: str, run: dict):
    setup = run.get("setup") or {}
    if kind == "agent":
        return setup.get("llm_model") or run["label"]
    return (setup.get("embedding_model"), setup.get("ranking")) if setup else run["label"]


def _row(kind: str, finished_at: str, runs: list[dict]) -> dict:
    setup = runs[0].get("setup") or {}
    names = dict.fromkeys(name for run in runs for name in run["scores"])
    details = dict.fromkeys(name for run in runs for name in run.get("details", {}))
    if kind == "agent":
        model = setup.get("llm_model") or runs[0]["label"]
    else:
        model = setup.get("embedding_model") or ("BM25" if setup.get("ranking") == "sparse" else runs[0]["label"])
    return {
        "model": model,
        "embedding_model": setup.get("embedding_model") if kind == "agent" else None,
        "ranking": setup.get("ranking"),
        "finished_at": finished_at,
        "runs": len(runs),
        "items": runs[0]["items"],
        "failed": sum(run["failed"] for run in runs),
        "scores": {name: mean(run["scores"].get(name) for run in runs) for name in names},
        "details": {name: mean(run.get("details", {}).get(name) for run in runs) for name in details},
        "urls": [run["url"] for run in runs if run.get("url")],
    }


def latest_results(directory: Path) -> dict[str, list[dict]]:
    """Per kind ('agent', 'retrieval'), one row per model (and ranking) from the newest report that has it.

    Agent rows are sorted by the share of conversations passed, retrieval rows by ranking (hybrid
    first, as the app uses it) and then MRR.
    """
    newest: dict[tuple, tuple[str, list[dict]]] = {}
    for path in sorted(directory.glob("*.json")):  # names start with the time, so later reports come last
        kind = path.stem.partition("-")[2]
        finished_at = _finished_at(path.stem)
        if kind not in SCORES or finished_at is None:
            continue
        try:
            runs = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("Skipping benchmark report %s: %s", path.name, exc)
            continue
        groups = defaultdict(list)
        for run in runs:
            groups[_key(kind, run)].append(run)
        for key, group in groups.items():
            newest[(kind, key)] = (finished_at, group)

    rows = {kind: [] for kind in SCORES}
    for (kind, _), (finished_at, group) in newest.items():
        rows[kind].append(_row(kind, finished_at, group))
    rows["agent"].sort(key=lambda r: -(r["scores"].get("passed") or 0))
    rank = {name: i for i, name in enumerate(RANKINGS)}
    rows["retrieval"].sort(key=lambda r: (rank.get(r["ranking"], len(rank)), -(r["scores"].get("reciprocal_rank") or 0)))
    return rows
