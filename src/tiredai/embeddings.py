"""Dense and sparse text encoders used to index products and to embed queries."""

import hashlib
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import batched
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import models

from tiredai import tracing
from tiredai.config import Settings

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"


class EmbeddingError(Exception):
    pass


class DenseEmbedder(Protocol):
    model: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedDense:
    """Local ONNX model, runs on CPU."""

    def __init__(self, model: str, cache_dir: Path):
        self.model = model
        self._model = TextEmbedding(model, cache_dir=str(cache_dir))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


class EmbeddingCache:
    """Vectors already fetched from a remote model, kept on disk so rebuilds don't spend API requests.

    Stored as float32, the precision Qdrant keeps anyway. Safe to use from several threads: the agent
    runs tools in worker threads.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\0{text}".encode()).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        found = {}
        with self._lock:
            for chunk in batched(keys, 500):  # stay below SQLite's limit on query parameters
                rows = self._db.execute(
                    f"SELECT key, vector FROM embeddings WHERE key IN ({','.join('?' * len(chunk))})", chunk
                ).fetchall()
                found.update({key: np.frombuffer(blob, dtype=np.float32).tolist() for key, blob in rows})
        return found

    def put_many(self, vectors: dict[str, list[float]]) -> None:
        with self._lock, self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
                [(key, np.asarray(vector, dtype=np.float32).tobytes()) for key, vector in vectors.items()],
            )

    def close(self) -> None:
        self._db.close()


def is_free_model(model: str) -> bool:
    return model.endswith(":free")


class OpenRouterDense:
    """Embeddings from the OpenRouter API, cached on disk.

    Free models allow 20 requests per minute and 50 per day (1000 with 10+ purchased credits), shared
    across all free models, so their requests are paced and sent one at a time (see build_dense).
    Paid models aren't held to those limits, so up to `concurrency` batches are requested at once.
    Every vector is cached either way.
    """

    BATCH_SIZE = 256  # provider maximum per request

    def __init__(
        self,
        model: str,
        api_key: str,
        cache: EmbeddingCache,
        *,
        query_prefix: str = "query: ",
        document_prefix: str = "passage: ",
        client: httpx.Client | None = None,
        concurrency: int = 1,
        min_interval: float = 3.1,
        max_retries: int = 5,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.model = model
        self.cache = cache
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.client = client or httpx.Client(timeout=120)
        self.headers = {"Authorization": f"Bearer {api_key}", "X-Title": "TiredAI"}
        self.concurrency = concurrency
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.sleep = sleep
        self._pace = threading.Lock()
        self._last_request = float("-inf")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.query_prefix + text])[0]

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        keys = [EmbeddingCache.key(self.model, text) for text in inputs]
        vectors = self.cache.get_many(keys)
        missing = [(key, text) for key, text in dict(zip(keys, inputs)).items() if key not in vectors]
        for chunk, fetched in self._fetch(list(batched(missing, self.BATCH_SIZE))):
            # Round to float32 like the cache does, so fresh and cached vectors are identical.
            new = {key: np.asarray(v, dtype=np.float32).tolist() for (key, _), v in zip(chunk, fetched)}
            self.cache.put_many(new)
            vectors.update(new)
        return [vectors[key] for key in keys]

    def _fetch(self, chunks: list[tuple]) -> Iterator[tuple[tuple, list[list[float]]]]:
        """Each batch with its vectors as its request finishes, at most `concurrency` requests at a time.

        After a failure no new requests start; batches already fetched are still yielded (so the
        caller caches them) before the error is raised.
        """
        if self.concurrency == 1 or len(chunks) <= 1:
            for chunk in chunks:
                yield chunk, self._request([text for _, text in chunk])
            return
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(chunks))) as pool:
            requests = {pool.submit(self._request, [text for _, text in chunk]): chunk for chunk in chunks}
            error = None
            for future in as_completed(requests):
                if future.cancelled():
                    continue
                try:
                    fetched = future.result()
                except Exception as exc:
                    if error is None:
                        error = exc
                        for waiting in requests:
                            waiting.cancel()  # only those not started yet
                    continue
                yield requests[future], fetched
            if error is not None:
                raise error

    def _request(self, inputs: list[str]) -> list[list[float]]:
        for attempt in range(self.max_retries + 1):
            self._wait_for_rate_limit()
            response = self.client.post(
                OPENROUTER_EMBEDDINGS_URL, headers=self.headers, json={"model": self.model, "input": inputs}
            )
            if response.status_code == 200 and "data" in (body := response.json()):
                data = sorted(body["data"], key=lambda item: item["index"])
                if len(data) != len(inputs):
                    raise EmbeddingError(f"OpenRouter returned {len(data)} embeddings for {len(inputs)} inputs")
                return [item["embedding"] for item in data]
            if response.status_code == 429 and "per-day" in response.text:
                raise EmbeddingError(f"OpenRouter daily free-model limit reached: {response.text[:300]}")
            if response.status_code in (429, 500, 502, 503, 529) and attempt < self.max_retries:
                self.sleep(float(response.headers.get("Retry-After", 2 ** (attempt + 1))))
                continue
            raise EmbeddingError(f"OpenRouter returned {response.status_code}: {response.text[:300]}")
        raise AssertionError("unreachable")

    def _wait_for_rate_limit(self) -> None:
        with self._pace:  # requests start at least min_interval apart, whichever thread sends them
            wait = self._last_request + self.min_interval - time.monotonic()
            if wait > 0:
                self.sleep(wait)
            self._last_request = time.monotonic()


class BM25Sparse:
    """Keyword vectors; Qdrant applies IDF at query time (see vectorstore.recreate_collection)."""

    def __init__(self, model: str, cache_dir: Path):
        self._model = SparseTextEmbedding(model, cache_dir=str(cache_dir))

    def embed_documents(self, texts: list[str]) -> list[models.SparseVector]:
        return [self._vector(e) for e in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> models.SparseVector:
        return self._vector(next(iter(self._model.query_embed(text))))

    @staticmethod
    def _vector(embedding) -> models.SparseVector:
        return models.SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())


class Encoder:
    """Pairs a dense embedder with BM25 so every text gets both vectors."""

    def __init__(self, dense: DenseEmbedder, sparse: BM25Sparse):
        self.dense = dense
        self.sparse = sparse

    def encode_documents(self, texts: list[str]) -> tuple[list[list[float]], list[models.SparseVector]]:
        return self.dense.embed_documents(texts), self.sparse.embed_documents(texts)

    def encode_query(self, text: str) -> tuple[list[float], models.SparseVector]:
        with tracing.client().start_as_current_observation(
            as_type="embedding", name="embed-query", input=text, model=self.dense.model
        ) as embedding:
            try:
                dense, sparse = self.dense.embed_query(text), self.sparse.embed_query(text)
            except Exception as exc:
                embedding.update(level="ERROR", status_message=str(exc))
                raise
            embedding.update(output={"dense_dimensions": len(dense), "sparse_terms": len(sparse.indices)})
        return dense, sparse


def build_dense(settings: Settings) -> DenseEmbedder:
    if settings.embedding_provider == "fastembed":
        return FastEmbedDense(settings.embedding_model, settings.model_cache_dir)
    if settings.embedding_provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError("EMBEDDING_PROVIDER=openrouter requires OPENROUTER_API_KEY")
        cache = EmbeddingCache(settings.embedding_cache_path)
        if is_free_model(settings.embedding_model):
            return OpenRouterDense(settings.embedding_model, settings.openrouter_api_key, cache)
        return OpenRouterDense(
            settings.embedding_model,
            settings.openrouter_api_key,
            cache,
            concurrency=settings.embedding_concurrency,
            min_interval=0,
        )
    raise ValueError(f"Unknown EMBEDDING_PROVIDER {settings.embedding_provider!r}, expected 'fastembed' or 'openrouter'")


def build_encoder(settings: Settings) -> Encoder:
    return Encoder(build_dense(settings), BM25Sparse(settings.sparse_model, settings.model_cache_dir))
