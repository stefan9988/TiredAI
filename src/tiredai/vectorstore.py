"""Qdrant collection holding one point per product, with a dense vector and a BM25 sparse vector."""

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from qdrant_client import QdrantClient, models

from tiredai.config import Settings
from tiredai.documents import document_text, point_id

DENSE = "dense"
SPARSE = "sparse"
DOCUMENT_KEY = "document"


class Encoder(Protocol):
    def encode_documents(self, texts: list[str]) -> tuple[list[list[float]], list[models.SparseVector]]: ...


def connect(settings: Settings) -> QdrantClient:
    """A Qdrant server when QDRANT_URL is set, otherwise the local on-disk store."""
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(settings.qdrant_path))


def recreate_collection(client: QdrantClient, collection: str, dense_dim: int) -> None:
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection,
        vectors_config={DENSE: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)},
        # Qdrant/bm25 stores term frequencies only; Qdrant applies IDF at query time.
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )


def index_products(
    client: QdrantClient,
    collection: str,
    products: list[dict],
    encoder: Encoder,
    batch_size: int = 256,
    on_batch: Callable[[int, int], None] | None = None,
) -> int:
    """Rebuild the collection from scratch with one point per product and return the point count."""
    if not products:
        raise ValueError("No products to index")
    skus = [p["sku"] for p in products]
    if len(set(skus)) != len(skus):
        raise ValueError("SKUs must be unique: each one becomes a point id")

    for start in range(0, len(products), batch_size):
        batch = products[start : start + batch_size]
        texts = [document_text(p) for p in batch]
        dense, sparse = encoder.encode_documents(texts)
        if start == 0:
            # The vector size comes from the model's output, so any dense model works unchanged.
            recreate_collection(client, collection, len(dense[0]))
        client.upsert(
            collection,
            points=[
                models.PointStruct(
                    id=point_id(p["sku"]),
                    vector={DENSE: d, SPARSE: s},
                    payload={**p, DOCUMENT_KEY: text},
                )
                for p, text, d, s in zip(batch, texts, dense, sparse)
            ],
        )
        if on_batch:
            on_batch(start + len(batch), len(products))

    count = client.count(collection, exact=True).count
    if count != len(products):
        raise RuntimeError(f"Collection has {count:,} points, expected {len(products):,}")
    return count


def verify_index(client: QdrantClient, collection: str, products: list[dict]) -> list[str]:
    """Compare every stored payload with its source product; returns a list of mismatches."""
    stored = {}
    offset = None
    while True:
        points, offset = client.scroll(collection, limit=1000, offset=offset, with_payload=True)
        stored.update({str(p.id): p.payload for p in points})
        if offset is None:
            break

    problems = []
    for product in products:
        payload = stored.get(point_id(product["sku"]))
        if payload is None:
            problems.append(f"{product['sku']}: missing from the collection")
        elif {k: v for k, v in payload.items() if k != DOCUMENT_KEY} != product:
            problems.append(f"{product['sku']}: stored payload differs from the source row")
    if len(stored) != len(products):
        problems.append(f"collection has {len(stored):,} points, expected {len(products):,}")
    return problems


def index_fingerprint(products: list[dict], settings: Settings) -> str:
    """Changes whenever a rebuild would give a different index: other products or embedded texts, or other models."""
    digest = hashlib.sha256()
    models_used = [settings.embedding_provider, settings.embedding_model, settings.sparse_model, settings.qdrant_collection]
    digest.update(json.dumps(models_used).encode())
    for product in products:
        digest.update(json.dumps(product, sort_keys=True, default=str).encode())
        digest.update(document_text(product).encode())
    return digest.hexdigest()


def index_state_path(settings: Settings) -> Path:
    """Where build_index.py records what the index was built from."""
    return settings.processed_data_path.with_name("index-state.json")


def save_index_state(path: Path, fingerprint: str, points: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"fingerprint": fingerprint, "points": points, "built_at": datetime.now(UTC).isoformat()}
    path.write_text(json.dumps(state, indent=2) + "\n")


def index_is_current(client: QdrantClient, collection: str, state_path: Path, fingerprint: str, points: int) -> bool:
    """True when the last build recorded this fingerprint and the collection still holds all its points."""
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError):
        return False
    return (
        state.get("fingerprint") == fingerprint
        and state.get("points") == points
        and client.collection_exists(collection)
        and client.count(collection, exact=True).count == points
    )
