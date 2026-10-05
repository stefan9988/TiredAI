import dataclasses
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from tiredai.config import Settings
from tiredai.embeddings import EmbeddingCache, EmbeddingError, OpenRouterDense, build_dense

MODEL = "nvidia/nemotron-3-embed-1b:free"


class FakeOpenRouter:
    """Records requests and answers like the embeddings endpoint; vectors encode the input text length."""

    def __init__(self, *responses: httpx.Response):
        self.queued = list(responses)
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if self.queued:
            return self.queued.pop(0)
        data = [{"index": i, "embedding": [float(len(text)), 1.0]} for i, text in enumerate(body["input"])]
        return httpx.Response(200, json={"data": list(reversed(data))})  # order must come from "index"


@pytest.fixture
def api():
    return FakeOpenRouter()


@pytest.fixture
def cache(tmp_path):
    cache = EmbeddingCache(tmp_path / "embeddings.sqlite")
    yield cache
    cache.close()


def embedder(api, cache, concurrency: int = 1) -> OpenRouterDense:
    return OpenRouterDense(
        MODEL,
        "test-key",
        cache,
        client=httpx.Client(transport=httpx.MockTransport(api)),
        concurrency=concurrency,
        min_interval=0,
        sleep=lambda s: None,
    )


class SlowOpenRouter(FakeOpenRouter):
    """Takes a moment per request and records how many were in flight at once."""

    def __init__(self, fail_on: str | None = None):
        super().__init__()
        self.fail_on = fail_on
        self.lock = threading.Lock()
        self.in_flight = self.most_in_flight = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        with self.lock:
            self.in_flight += 1
            self.most_in_flight = max(self.most_in_flight, self.in_flight)
        try:
            time.sleep(0.05)
            if self.fail_on and self.fail_on in request.content.decode():
                return httpx.Response(401, text="invalid key")
            return super().__call__(request)
        finally:
            with self.lock:
                self.in_flight -= 1


def test_documents_and_queries_get_the_model_prefixes(api, cache):
    dense = embedder(api, cache)

    dense.embed_documents(["Michelin Pilot Sport 4S."])
    dense.embed_query("summer tires")

    assert api.requests[0] == {"model": MODEL, "input": ["passage: Michelin Pilot Sport 4S."]}
    assert api.requests[1]["input"] == ["query: summer tires"]


def test_vectors_follow_input_order(api, cache):
    vectors = embedder(api, cache).embed_documents(["a", "bbb", "cc"])

    assert [v[0] for v in vectors] == [len("passage: a"), len("passage: bbb"), len("passage: cc")]


def test_requests_are_batched_at_the_provider_maximum(api, cache):
    embedder(api, cache).embed_documents([f"tire {i}" for i in range(600)])

    assert [len(r["input"]) for r in api.requests] == [256, 256, 88]


def test_cached_vectors_are_not_requested_again(api, cache, tmp_path):
    first = embedder(api, cache).embed_documents(["a", "b"])
    reopened = EmbeddingCache(tmp_path / "embeddings.sqlite")
    second = embedder(api, reopened).embed_documents(["b", "a", "c"])
    reopened.close()

    assert len(api.requests) == 2
    assert api.requests[1]["input"] == ["passage: c"]
    assert second[:2] == [first[1], first[0]]


def test_duplicate_texts_are_requested_once(api, cache):
    vectors = embedder(api, cache).embed_documents(["same", "same"])

    assert api.requests[0]["input"] == ["passage: same"]
    assert vectors[0] == vectors[1]


def test_rate_limited_request_is_retried(cache):
    api = FakeOpenRouter(httpx.Response(429, headers={"Retry-After": "0"}, text="rate limited"))

    assert embedder(api, cache).embed_query("winter tires")
    assert len(api.requests) == 2


def test_daily_limit_fails_fast(cache):
    api = FakeOpenRouter(httpx.Response(429, text='{"error":{"message":"Rate limit exceeded: free-models-per-day"}}'))

    with pytest.raises(EmbeddingError, match="daily"):
        embedder(api, cache).embed_query("winter tires")
    assert len(api.requests) == 1


def test_error_response_raises(cache):
    api = FakeOpenRouter(httpx.Response(401, text="invalid key"))

    with pytest.raises(EmbeddingError, match="401"):
        embedder(api, cache).embed_query("winter tires")


def test_short_response_raises(cache):
    api = FakeOpenRouter(httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]}))

    with pytest.raises(EmbeddingError, match="1 embeddings for 2 inputs"):
        embedder(api, cache).embed_documents(["a", "b"])


def test_openrouter_provider_requires_an_api_key():
    settings = dataclasses.replace(Settings.load(), embedding_provider="openrouter", openrouter_api_key=None)

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        build_dense(settings)


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
        build_dense(dataclasses.replace(Settings.load(), embedding_provider="pinecone"))


def test_cache_can_be_used_from_other_threads(api, cache):
    # The agent runs tools in worker threads, so query embeddings are cached from those threads.
    dense = embedder(api, cache)
    with ThreadPoolExecutor(max_workers=4) as pool:
        vectors = list(pool.map(dense.embed_query, ["a", "b", "a", "c"]))

    assert vectors[0] == vectors[2]
    assert len(api.requests) <= 4


def texts(batches: int) -> list[str]:
    return [f"tire {i}" for i in range(256 * batches)]


def test_batches_are_requested_in_parallel_up_to_the_concurrency(cache):
    api = SlowOpenRouter()

    vectors = embedder(api, cache, concurrency=3).embed_documents(texts(7))

    assert api.most_in_flight == 3
    assert len(api.requests) == 7
    assert [v[0] for v in vectors] == [float(len("passage: " + t)) for t in texts(7)]  # still in input order


def test_one_request_at_a_time_without_concurrency(cache):
    api = SlowOpenRouter()

    embedder(api, cache).embed_documents(texts(3))

    assert api.most_in_flight == 1 and len(api.requests) == 3


def test_batches_fetched_before_a_failure_are_cached(cache):
    api = SlowOpenRouter(fail_on="tire 600")  # the third batch fails
    dense = embedder(api, cache, concurrency=3)

    with pytest.raises(EmbeddingError, match="401"):
        dense.embed_documents(texts(3))

    cached = cache.get_many([EmbeddingCache.key(MODEL, "passage: " + t) for t in texts(3)])
    assert len(cached) == 512  # the other two batches


def test_free_models_are_paced_and_paid_models_run_in_parallel(tmp_path):
    settings = dataclasses.replace(
        Settings.load(),
        embedding_provider="openrouter",
        openrouter_api_key="test-key",
        embedding_cache_path=tmp_path / "embeddings.sqlite",
        embedding_concurrency=6,
    )

    free = build_dense(dataclasses.replace(settings, embedding_model="nvidia/nemotron-3-embed-1b:free"))
    paid = build_dense(dataclasses.replace(settings, embedding_model="qwen/qwen3-embedding-8b"))

    assert (free.concurrency, free.min_interval) == (1, 3.1)
    assert (paid.concurrency, paid.min_interval) == (6, 0)
