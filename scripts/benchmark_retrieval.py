"""Retrieval benchmark: compare embedding models, and dense, BM25 and hybrid ranking, on shopper queries.

Each embedding model gets its own in-memory index of the processed catalog, and every query of
benchmarks/retrieval_products.yaml and benchmarks/retrieval_descriptive.yaml runs through the app's
catalog search (see src/tiredai/benchmarks/retrieval.py for the queries and scores). Vectors of
remote models are cached, so only a model's first run calls its API; queries are embedded in one
batch. BM25 alone doesn't depend on the embedding model, so --ranking sparse runs once.

With LANGFUSE_* keys set, the queries are synced to the Langfuse dataset tiredai-retrieval and each
model and ranking becomes one experiment run there. The summary is printed and saved to
benchmarks/results/ either way.

Usage:
    uv run python scripts/benchmark_retrieval.py [--embedding-model [PROVIDER:]MODEL ...]
        [--ranking hybrid|dense|sparse ...] [--local] [--check]

    PROVIDER is fastembed or openrouter; without it, EMBEDDING_PROVIDER's. Default: the .env model.
"""

import argparse
import os
import sys
import time

from tiredai import tracing
from tiredai.benchmarks import experiments as ex
from tiredai.benchmarks.catalog import Catalog
from tiredai.benchmarks.retrieval import (
    DATASET,
    DESCRIPTIVE_SCORES,
    PRODUCT_SCORES,
    TOP_K,
    RankingTask,
    check_queries,
    load_queries,
    ranking_evaluator,
)
from tiredai.config import Settings
from tiredai.embeddings import EmbeddingError
from tiredai.search import CatalogSearch

RANKINGS = ("hybrid", "dense", "sparse")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedding-model", action="append", metavar="[PROVIDER:]MODEL", help="repeat to compare models")
    parser.add_argument("--ranking", action="append", choices=RANKINGS, help="default: hybrid and dense")
    parser.add_argument("--local", action="store_true", help="send nothing to Langfuse, even with keys set")
    parser.add_argument("--check", action="store_true", help="only validate the queries against the catalog")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up in logs, not only at the end
    if args.local:
        os.environ.update(LANGFUSE_PUBLIC_KEY="", LANGFUSE_SECRET_KEY="")  # .env doesn't override set variables
    settings = Settings.load()
    rankings = list(dict.fromkeys(args.ranking or ["hybrid", "dense"]))

    catalog = Catalog.load(settings.processed_data_path)
    queries = load_queries(ex.BENCHMARKS_DIR / "retrieval_products.yaml", ex.BENCHMARKS_DIR / "retrieval_descriptive.yaml")
    if problems := check_queries(queries, catalog):
        sys.exit("The queries don't fit the catalog:\n" + "\n".join(f"  {p}" for p in problems))
    kinds = {kind: sum(q.kind == kind for q in queries) for kind in ("product", "descriptive")}
    print(f"{len(queries)} queries ({kinds['product']} product, {kinds['descriptive']} descriptive) fit the catalog.")
    if args.check:
        return

    cases = [q.case() for q in queries]
    langfuse = tracing.client()
    if tracing.enabled():
        data = ex.sync_dataset(langfuse, DATASET, "Shopper queries with their relevant products (benchmarks/retrieval_*.yaml)", cases)
        print(f"Synced to the Langfuse dataset {DATASET}.")
    else:
        data = ex.local_items(cases)
        print("Running locally (--local or no Langfuse keys): nothing is sent to Langfuse.")

    stamp, revision = ex.timestamp(), ex.git_revision()
    summaries, failed_models = [], []
    for spec in args.embedding_model or [None]:
        model_settings = ex.with_embedding(settings, spec)
        label = ex.embedding_label(model_settings)
        print(f"\nIndexing the catalog with {label} ...")
        started = time.perf_counter()
        try:
            client, encoder = ex.memory_index(model_settings, lambda done, total: print(f"  {done:>6,} / {total:,}", end="\r", flush=True))
            encoder.dense.embed_queries([q.query for q in queries])
        except (EmbeddingError, ValueError) as exc:
            print(f"  skipped: {exc}")
            failed_models.append(label)
            continue
        print(f"  indexed in {time.perf_counter() - started:.0f}s")

        for ranking in rankings:
            run_label = "sparse (BM25)" if ranking == "sparse" else f"{ranking} · {label}"
            if any(s.label == run_label for s in summaries):
                continue  # BM25 ranks the same whatever the embedding model
            search = CatalogSearch(client, ex.COLLECTION, encoder, max_results=max(TOP_K, settings.agent.max_search_results), ranking=ranking)
            print(f"Running {run_label} ...")
            started = time.perf_counter()
            result = langfuse.run_experiment(
                name=f"Retrieval: {run_label}",
                run_name=f"{run_label} · {stamp}",
                description=f"{ranking} ranking with {label}, git {revision}",
                data=data,
                task=RankingTask(search),
                evaluators=[ranking_evaluator(catalog)],
                max_concurrency=1,  # local Qdrant runs queries one at a time anyway
                metadata={"embedding_model": label, "ranking": ranking, "git": revision},
            )
            seconds = (time.perf_counter() - started) / max(len(data), 1)
            setup = {"ranking": ranking} if ranking == "sparse" else {"embedding_model": label, "ranking": ranking}
            summaries.append(ex.summarize(run_label, result, len(data), {"seconds_per_query": seconds}, setup))
        client.close()
    tracing.flush()

    if not summaries:
        sys.exit("No model could be benchmarked.")
    order = PRODUCT_SCORES + DESCRIPTIVE_SCORES
    print("\n" + ex.markdown_table(summaries, order))
    notes = [f"{stamp}, git {revision}", f"{kinds['product']} product queries (hit@k, reciprocal_rank), "
             f"{kinds['descriptive']} descriptive queries (precision@10, ndcg@10)"]  # fmt: skip
    if failed_models:
        notes.append(f"skipped (embedding failed): {', '.join(failed_models)}")
    md_path, _ = ex.write_report("retrieval", "Retrieval benchmark", summaries, order, notes)
    print(f"\nSaved {md_path.relative_to(ex.BENCHMARKS_DIR.parent)} (and .json with every query).")
    for s in summaries:
        if s.url:
            print(f"  {s.label}: {s.url}")


if __name__ == "__main__":
    main()
