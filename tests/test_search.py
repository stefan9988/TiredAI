import json

import pytest
from conftest import FakeEncoder, raw_frame
from qdrant_client import QdrantClient

from tiredai.documents import products
from tiredai.embeddings import EmbeddingError
from tiredai.preprocessing import normalize
from tiredai.search import (
    CatalogSearch,
    catalog_tools,
    describe_results,
    describe_search,
    make_search_tool,
    parse_size,
    speed_rank,
)
from tiredai.vectorstore import index_products

PASSENGER = {"carType": "Passenger", "season": "All Season", "performance": "Touring", "runFlat": "false"}

CATALOG = [
    {**PASSENGER, "sku": "CHEAP", "name": "Accelera Phi-R 205/55R16 91V", "brand": "Accelera", "model": "Phi-R",
     "size": "205/55R16", "performance": "Performance", "speedRating": "V", "price": "59.930000"},
    {**PASSENGER, "sku": "PILOT", "name": "Michelin Pilot Sport 4S 205/55R16 94Y XL", "brand": "Michelin",
     "model": "Accelera", "size": "205/55R16", "season": "Summer", "performance": "High Performance",
     "speedRating": "Y", "price": "189.990000"},
    {**PASSENGER, "sku": "RUNFLAT", "name": "Bridgestone Turanza RFT 205/55R16 91H Run Flat", "brand": "Bridgestone",
     "model": "Turanza RFT", "size": "205/55R16", "speedRating": "H", "runFlat": "true", "price": "149.990000"},
    {"sku": "LT-KO2", "name": "BFGoodrich All-Terrain T/A KO2 LT 265/70R17 121S E (10 Ply)", "brand": "BFGoodrich",
     "model": "All-Terrain T/A KO2", "size": "265/70R17", "season": "All Season", "carType": "Light Truck",
     "performance": "All Terrain", "speedRating": "S", "loadIndex": "121/118", "runFlat": "false", "price": "239.990000"},
    {"sku": "SUV-HT", "name": "Kumho Crugen HT51 265/70R17 113T", "brand": "Kumho", "model": "Crugen HT51",
     "size": "265/70R17", "season": "All Season", "carType": "Truck/SUV", "performance": "Highway",
     "speedRating": "T", "runFlat": "false", "price": "129.990000"},
    {**PASSENGER, "sku": "VINTAGE", "name": "Firestone Deluxe Champion 5.20-13 (WWW)", "brand": "Firestone",
     "model": "Deluxe Champion", "size": "5.2-13", "performance": "N/A", "speedRating": "N/A", "price": "99.990000"},
    {"sku": "TRACTOR", "name": "BKT TR-135 8-16 6 Ply (TT)", "brand": "BKT", "model": "TR-135", "size": "8-16",
     "season": "All Season", "carType": "Tractor", "performance": "N/A", "speedRating": "A6/A8", "runFlat": "false",
     "price": "115.000000"},
]  # fmt: skip


@pytest.fixture(scope="module")
def catalog():
    client = QdrantClient(":memory:")
    index_products(client, "tires", products(normalize(raw_frame(*CATALOG))), FakeEncoder())
    yield CatalogSearch(client, "tires", FakeEncoder())
    client.close()


def skus(result: dict) -> list[str]:
    return [p["sku"] for p in result["products"]]


@pytest.mark.parametrize(
    "text, key, light_truck",
    [
        ("205/55R16", "205/55R16", False),
        ("205/55r16", "205/55R16", False),
        ("205 55 16", "205/55R16", False),
        ("205/55/16", "205/55R16", False),
        ("205/55 R 16", "205/55R16", False),
        ("P205/55R16", "205/55R16", False),
        ("205/55ZR16", "205/55R16", False),
        ("205/55 ZR 16", "205/55R16", False),
        ("LT265/70R17", "265/70R17", True),
        ("lt 265/70r17", "265/70R17", True),
        ("5.20-13", "5.2-13", False),
        ("38x13.50R26", "38X13.5R26", False),
        ("195R14C", "195R14C", False),
    ],
)
def test_size_formatting_is_normalized(text, key, light_truck):
    parsed = parse_size(text)
    assert (parsed.key, parsed.light_truck) == (key, light_truck)


def test_speed_ratings_follow_the_rating_order_not_the_alphabet():
    assert speed_rank("H") < speed_rank("V") < speed_rank("Z") < speed_rank("W") < speed_rank("Y")
    assert speed_rank("T") < speed_rank("H")
    assert speed_rank("A6/A8") == speed_rank("A6")
    assert speed_rank("D/A") is None


@pytest.mark.parametrize("size", ["205/55R16", "205 55 16", "205/55r16", "P205/55ZR16"])
def test_size_filter_returns_only_that_size(catalog, size):
    result = catalog.search(size=size)

    assert result["total_matching"] == 3
    assert {p["size"] for p in result["products"]} == {"205/55R16"}


def test_lt_size_returns_only_light_truck_tires(catalog):
    assert skus(catalog.search(size="LT265/70R17")) == ["LT-KO2"]
    assert set(skus(catalog.search(size="265/70R17"))) == {"LT-KO2", "SUV-HT"}


def test_size_with_different_number_formatting_is_found(catalog):
    assert skus(catalog.search(size="5.20-13")) == ["VINTAGE"]


def test_unknown_size_is_reported_not_substituted(catalog):
    result = catalog.search(size="215/60R16")

    assert "not in the catalog" in result["error"]


def test_price_range_and_price_order(catalog):
    assert skus(catalog.search(size="205/55R16", max_price=150)) == ["CHEAP", "RUNFLAT"]
    assert skus(catalog.search(size="205/55R16", min_price=100, max_price=150)) == ["RUNFLAT"]
    assert skus(catalog.search(size="205/55R16", sort="price_desc")) == ["PILOT", "RUNFLAT", "CHEAP"]


@pytest.mark.parametrize("min_price, max_price", [(-1, None), (200, 100)])
def test_invalid_price_range_is_rejected(catalog, min_price, max_price):
    assert "error" in catalog.search(min_price=min_price, max_price=max_price)


def test_brand_matching_ignores_case(catalog):
    assert skus(catalog.search(brand="michelin")) == ["PILOT"]


def test_unknown_brand_suggests_close_matches(catalog):
    assert "Did you mean: Michelin?" in catalog.search(brand="Michelen")["error"]


def test_season_car_type_and_performance_filters(catalog):
    assert skus(catalog.search(season="summer")) == ["PILOT"]
    assert skus(catalog.search(car_type="truck/suv")) == ["SUV-HT"]
    assert skus(catalog.search(performance="All Terrain")) == ["LT-KO2"]


def test_unknown_filter_value_lists_the_allowed_ones(catalog):
    error = catalog.search(season="monsoon")["error"]

    assert "Unknown season 'monsoon'" in error and "All Season, Summer" in error


def test_run_flat_filter(catalog):
    assert skus(catalog.search(size="205/55R16", run_flat=True)) == ["RUNFLAT"]
    assert skus(catalog.search(size="205/55R16", run_flat=False)) == ["CHEAP", "PILOT"]


def test_minimum_speed_rating(catalog):
    assert skus(catalog.search(size="205/55R16", min_speed_rating="V")) == ["CHEAP", "PILOT"]
    assert skus(catalog.search(size="205/55R16", min_speed_rating="w")) == ["PILOT"]
    assert "TRACTOR" in skus(catalog.search(min_speed_rating="A6", limit=10))  # 'A6/A8' counts as A6
    assert "TRACTOR" not in skus(catalog.search(min_speed_rating="A8", limit=10))


def test_unknown_speed_rating_is_rejected(catalog):
    assert "Unknown speed rating" in catalog.search(min_speed_rating="X")["error"]


def test_query_ranks_the_named_product_first(catalog):
    result = catalog.search(query="michelin pilot sport 4s", size="205/55R16")

    assert result["order"] == "relevance"
    assert skus(result)[0] == "PILOT"


def test_query_with_price_sort_orders_by_price(catalog):
    result = catalog.search(query="tire", size="205/55R16", sort="price_asc")

    assert skus(result) == ["CHEAP", "RUNFLAT", "PILOT"]


def test_results_hide_model_and_embedded_text(catalog):
    [product] = catalog.search(brand="Michelin")["products"]

    assert "model" not in product and "document" not in product
    assert product["name"] == "Michelin Pilot Sport 4S 205/55R16 94Y XL"
    assert product["price"] == 189.99


def test_no_match_says_so(catalog):
    result = catalog.search(size="205/55R16", brand="Kumho")

    assert result["total_matching"] == 0 and result["products"] == []
    assert "No products" in result["note"]


def test_limit_is_capped(catalog):
    assert catalog.search(limit=100)["returned"] == len(CATALOG)
    assert catalog.search(limit=2)["returned"] == 2
    assert catalog.search(limit=0)["returned"] == 1


def test_embedding_failure_is_returned_as_an_error(catalog):
    class DownEncoder:
        def encode_query(self, text):
            raise EmbeddingError("OpenRouter daily free-model limit reached")

    broken = CatalogSearch(catalog.client, "tires", DownEncoder())

    assert "temporarily unavailable" in broken.search(query="winter tires")["error"]
    assert broken.search(size="205/55R16")["total_matching"] == 3  # filters alone need no embedding


def test_tool_returns_json_and_lists_allowed_values(catalog):
    tool = make_search_tool(catalog)

    result = json.loads(tool.invoke({"size": "205/55R16", "sort": "price_asc", "limit": 1}))

    assert result["products"][0]["sku"] == "CHEAP"
    assert "Light Truck, Passenger, Tractor, Truck/SUV" in tool.description
    assert "A1" in tool.description and "Y" in tool.description


def test_no_tool_without_an_index():
    client = QdrantClient(":memory:")

    def factory():
        raise AssertionError("the encoder must not be built without an index")

    assert catalog_tools(client, "tires", factory) == []
    assert catalog_tools(None, "tires", factory) == []


@pytest.mark.parametrize(
    "args, text",
    [
        ({}, "Searching the catalog"),
        ({"size": "205/60R15", "season": "All Season", "max_price": 60}, "Searching the catalog: 205/60R15 · All Season · up to $60"),
        ({"query": "Eagle F1", "brand": "Goodyear", "size": "255/50R19"}, 'Searching the catalog: 255/50R19 · "Eagle F1" · Goodyear'),
        ({"min_price": "50", "max_price": 99.5}, "Searching the catalog: $50–$99.50"),
        ({"min_price": 100}, "Searching the catalog: from $100"),
        ({"run_flat": True, "min_speed_rating": "H", "sort": "price_asc"}, "Searching the catalog: run-flat · speed rating H or higher · cheapest first"),
        ({"run_flat": "false", "car_type": "Truck/SUV", "sort": "price_desc"}, "Searching the catalog: Truck/SUV · no run-flat · most expensive first"),
    ],
)
def test_search_is_described_for_the_status_line(args, text):
    assert describe_search(args) == text


@pytest.mark.parametrize(
    "content, text",
    [
        ('{"total_matching": 6, "products": []}', "Found 6 tires"),
        ('{"total_matching": 1, "products": []}', "Found 1 tire"),
        ('{"total_matching": 1250, "products": []}', "Found 1,250 tires"),
        ('{"total_matching": 0, "products": []}', "No tires match"),
        ('{"error": "Size \'1/2R3\' is not in the catalog."}', "Search problem: Size '1/2R3' is not in the catalog."),
        ("Error invoking tool: invalid arguments", "The search failed"),
    ],
)
def test_search_result_is_described_for_the_status_line(content, text):
    assert describe_results(content) == text
