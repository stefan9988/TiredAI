import pytest
from conftest import FakeEncoder, raw_frame
from qdrant_client import QdrantClient, models

from tiredai.documents import point_id, products
from tiredai.preprocessing import normalize
from tiredai.vectorstore import DENSE, DOCUMENT_KEY, SPARSE, index_products, verify_index

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
