"""Build the Qdrant index: preprocess the raw CSV, embed every product and load it into Qdrant.

Each run rebuilds the collection from scratch and then checks every stored payload against the
processed data. Settings come from .env (see .env.example); by default the index lives in
data/vectorstore/.

Usage:
    uv run python scripts/build_index.py [--skip-preprocess] [--batch-size N]
"""

import argparse
import sys
import time

import pandas as pd

from tiredai.config import Settings
from tiredai.documents import products
from tiredai.embeddings import EmbeddingError, build_encoder
from tiredai.preprocessing import PreprocessingError, preprocess
from tiredai.vectorstore import DENSE, connect, index_products, verify_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-preprocess", action="store_true", help="reuse the existing processed Parquet file")
    parser.add_argument("--batch-size", type=int, default=256, help="products embedded per batch (default: 256)")
    args = parser.parse_args()
    settings = Settings.load()

    if args.skip_preprocess:
        print(f"Skipping preprocessing, using {settings.processed_data_path}")
    else:
        print(f"Preprocessing {settings.raw_data_path} ...")
        try:
            preprocess(settings.raw_data_path, settings.processed_data_path)
        except PreprocessingError as exc:
            sys.exit(str(exc))

    items = products(pd.read_parquet(settings.processed_data_path))

    print(f"Loading embedding model {settings.embedding_model} ({settings.embedding_provider}) and sparse model "
          f"{settings.sparse_model} (local models are downloaded on first run) ...")
    try:
        encoder = build_encoder(settings)
    except ValueError as exc:
        sys.exit(str(exc))

    client = connect(settings)
    location = settings.qdrant_url or settings.qdrant_path
    print(f"Indexing {len(items):,} products into '{settings.qdrant_collection}' at {location} ...")
    started = time.perf_counter()

    def progress(done: int, total: int) -> None:
        print(f"  {done:>6,} / {total:,}", end="\r", flush=True)

    try:
        count = index_products(client, settings.qdrant_collection, items, encoder, args.batch_size, progress)
    except EmbeddingError as exc:
        client.close()
        sys.exit(f"\nEmbedding failed: {exc}\nVectors fetched so far are cached; rerun to continue.")
    problems = verify_index(client, settings.qdrant_collection, items)
    dense_dim = client.get_collection(settings.qdrant_collection).config.params.vectors[DENSE].size
    client.close()
    if problems:
        details = "\n".join(f"  {p}" for p in problems[:20])
        sys.exit(f"\nIndex verification failed with {len(problems):,} problems:\n{details}")

    print(f"\nIndexed {count:,} points in {time.perf_counter() - started:.0f}s "
          f"(dense: {dense_dim} dims, sparse: BM25)")
    print("Verified: every stored payload matches the processed data.")


if __name__ == "__main__":
    main()
