import dataclasses
import json

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


def embedder(api, cache) -> OpenRouterDense:
    return OpenRouterDense(
        MODEL, "test-key", cache, client=httpx.Client(transport=httpx.MockTransport(api)), min_interval=0, sleep=lambda s: None
    )


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
