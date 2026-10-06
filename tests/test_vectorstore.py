import dataclasses

import pytest
from conftest import FakeEncoder, raw_frame
from qdrant_client import QdrantClient, models

from tiredai.config import Settings
from tiredai.documents import point_id, products
from tiredai.preprocessing import normalize
from tiredai.vectorstore import (
    DENSE,
    DOCUMENT_KEY,
    SPARSE,
    index_fingerprint,
    index_is_current,
    index_products,
    save_index_state,
    verify_index,
)

COLLECTION = "tires"


CATALOG = products(
    normalize(
        raw_frame(
            {"sku": "CHEAP", "price": "59.930000"},
            {"sku": "PRICEY", "name": "Michelin Pilot Sport 4S 205/55R15 94Y XL", "brand": "Michelin",
             "model": "Pilot Sport 4S", "season": "Summer", "performance": "High Performance", "price": "189.990000"},
            {"sku": "OTHER-SIZE", "name": "Kumho Crugen HT51 255/70R15 108T", "brand": "Kumho", "model": "55",
             "size": "255/70R15", "carType": "Truck/SUV", "performance": "N/A", "price": "129.990000"},
        )
    )
)


@pytest.fixture
def client():
    client = QdrantClient(":memory:")
    index_products(client, COLLECTION, CATALOG, FakeEncoder())
    yield client
    client.close()


def skus(points) -> set[str]:
    return {p.payload["sku"] for p in points}


def scroll(client, *conditions) -> set[str]:
    points, _ = client.scroll(COLLECTION, scroll_filter=models.Filter(must=list(conditions)), limit=100)
    return skus(points)


def test_every_product_is_stored_with_its_full_payload(client):
    assert client.count(COLLECTION).count == len(CATALOG)
    assert verify_index(client, COLLECTION, CATALOG) == []


def test_payload_includes_the_embedded_document(client):
    [point] = client.retrieve(COLLECTION, [point_id("CHEAP")])

    assert point.payload[DOCUMENT_KEY] == "Accelera Phi-R 205/55R15 92V XL. All Season Performance tire for Passenger."


def test_lookup_by_sku(client):
    [point] = client.retrieve(COLLECTION, [point_id("PRICEY")])

    assert point.payload["name"] == "Michelin Pilot Sport 4S 205/55R15 94Y XL"
    assert point.payload["price"] == 189.99


def test_exact_size_filter(client):
    assert scroll(client, models.FieldCondition(key="size", match=models.MatchValue(value="205/55R15"))) == {
        "CHEAP",
        "PRICEY",
    }


def test_size_and_price_range_filter(client):
    assert scroll(
        client,
        models.FieldCondition(key="size", match=models.MatchValue(value="205/55R15")),
        models.FieldCondition(key="price", range=models.Range(lt=100)),
    ) == {"CHEAP"}


def test_model_filter(client):
    assert scroll(client, models.FieldCondition(key="model", match=models.MatchValue(value="Pilot Sport 4S"))) == {
        "PRICEY"
    }


def test_any_of_filter(client):
    condition = models.FieldCondition(key="season", match=models.MatchAny(any=["Summer", "Winter"]))

    assert scroll(client, condition) == {"PRICEY"}


def test_boolean_filter(client):
    assert scroll(client, models.FieldCondition(key="runFlat", match=models.MatchValue(value=False))) == {
        "CHEAP",
        "PRICEY",
        "OTHER-SIZE",
    }


def test_missing_field_filter(client):
    assert scroll(client, models.IsEmptyCondition(is_empty=models.PayloadField(key="performance"))) == {"OTHER-SIZE"}


def test_hybrid_query_with_filter_ranks_the_named_product_first(client):
    encoder = FakeEncoder()
    query = "michelin pilot sport 4s"
    response = client.query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=encoder.dense(query), using=DENSE, limit=10),
            models.Prefetch(query=encoder.sparse(query), using=SPARSE, limit=10),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=models.Filter(must=[models.FieldCondition(key="size", match=models.MatchValue(value="205/55R15"))]),
        limit=3,
    )

    assert response.points[0].payload["sku"] == "PRICEY"
    assert skus(response.points) <= {"CHEAP", "PRICEY"}


def test_rebuild_replaces_the_previous_collection(client):
    index_products(client, COLLECTION, CATALOG[:1], FakeEncoder())

    assert client.count(COLLECTION).count == 1


def test_collection_uses_the_vector_size_of_the_model(client):
    assert client.get_collection(COLLECTION).config.params.vectors[DENSE].size == FakeEncoder.dense_dim


def test_empty_catalog_is_rejected(client):
    with pytest.raises(ValueError, match="No products"):
        index_products(client, COLLECTION, [], FakeEncoder())


def test_duplicate_skus_are_rejected(client):
    with pytest.raises(ValueError, match="unique"):
        index_products(client, COLLECTION, [CATALOG[0], CATALOG[0]], FakeEncoder())


def test_verify_index_reports_changed_payload(client):
    client.set_payload(COLLECTION, payload={"price": 1.0}, points=[point_id("CHEAP")])

    assert verify_index(client, COLLECTION, CATALOG) == ["CHEAP: stored payload differs from the source row"]


def test_index_persists_on_disk(tmp_path):
    client = QdrantClient(path=str(tmp_path))
    index_products(client, COLLECTION, CATALOG, FakeEncoder())
    client.close()

    reopened = QdrantClient(path=str(tmp_path))
    assert reopened.count(COLLECTION).count == len(CATALOG)
    reopened.close()


def test_fingerprint_changes_with_the_products_or_the_embedding_models():
    settings = Settings.load()
    changed_price = [{**CATALOG[0], "price": 61.0}, *CATALOG[1:]]
    other_model = dataclasses.replace(settings, embedding_model="another/model")

    assert index_fingerprint(CATALOG, settings) == index_fingerprint([dict(p) for p in CATALOG], settings)
    assert index_fingerprint(changed_price, settings) != index_fingerprint(CATALOG, settings)
    assert index_fingerprint(CATALOG[:2], settings) != index_fingerprint(CATALOG, settings)
    assert index_fingerprint(CATALOG, other_model) != index_fingerprint(CATALOG, settings)


def test_index_is_current_only_after_a_recorded_build_of_the_same_data(client, tmp_path):
    state = tmp_path / "processed" / "index-state.json"

    assert not index_is_current(client, COLLECTION, state, "abc", len(CATALOG))  # nothing recorded yet
    save_index_state(state, "abc", len(CATALOG))
    assert index_is_current(client, COLLECTION, state, "abc", len(CATALOG))
    assert not index_is_current(client, COLLECTION, state, "other", len(CATALOG))  # the data changed

    client.delete(COLLECTION, points_selector=models.PointIdsList(points=[point_id("CHEAP")]))
    assert not index_is_current(client, COLLECTION, state, "abc", len(CATALOG))  # a build was cut short
    client.delete_collection(COLLECTION)
    assert not index_is_current(client, COLLECTION, state, "abc", len(CATALOG))  # the store was deleted


def test_an_unreadable_state_means_rebuild(client, tmp_path):
    state = tmp_path / "index-state.json"
    state.write_text("not json")

    assert not index_is_current(client, COLLECTION, state, "abc", len(CATALOG))
