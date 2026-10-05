import json

import pandas as pd
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
     "treadDepth": "N/A", "mileageWarranty": "", "price": "115.000000"},
]  # fmt: skip

# The generated columns, set per product instead of from the SKU: (available, recommendations).
STOCK = {"CHEAP": (True, 3), "PILOT": (True, 5), "RUNFLAT": (False, 4), "LT-KO2": (True, 4), "SUV-HT": (True, 2),
         "VINTAGE": (False, 1), "TRACTOR": (True, 3)}  # fmt: skip


@pytest.fixture(scope="module")
def catalog():
    frame = normalize(raw_frame(*CATALOG))
    frame["available"] = pd.array([STOCK[sku][0] for sku in frame["sku"]], dtype="boolean")
    frame["recommendations"] = pd.array([STOCK[sku][1] for sku in frame["sku"]], dtype="Int64")
    client = QdrantClient(":memory:")
    index_products(client, "tires", products(frame), FakeEncoder())
    yield CatalogSearch(client, "tires", FakeEncoder(), max_results=20)
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
    assert "TRACTOR" in skus(catalog.search(min_speed_rating="A6"))  # 'A6/A8' counts as A6
    assert "TRACTOR" not in skus(catalog.search(min_speed_rating="A8"))


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


def test_units_are_sent_as_text_and_stock_as_a_flag(catalog):
    [pilot] = catalog.search(brand="Michelin")["products"]
    [tractor] = catalog.search(brand="BKT")["products"]

    assert (pilot["treadDepth"], pilot["mileageWarranty"], pilot["recommendations"]) == ("9/32 in", "50,000 miles", "5/5")
    assert pilot["available"] is True
    assert "treadDepth32nds" not in pilot and "mileageWarrantyMiles" not in pilot
    assert "treadDepth" not in tractor and "mileageWarranty" not in tractor  # missing values stay missing
    stored = catalog.client.scroll("tires", with_payload=True, limit=10)[0]
    assert {p.payload["sku"]: p.payload["treadDepth32nds"] for p in stored if "treadDepth32nds" in p.payload}["PILOT"] == 9


def test_out_of_stock_products_are_returned_flagged(catalog):
    result = catalog.search(size="205/55R16")

    assert {p["sku"]: p["available"] for p in result["products"]} == {"CHEAP": True, "RUNFLAT": False, "PILOT": True}


def test_minimum_recommendation_level(catalog):
    result = catalog.search(size="205/55R16", min_recommendations=4)

    assert skus(result) == ["RUNFLAT", "PILOT"]
    assert result["filters"]["recommendations"] == {"min": 4}
    assert skus(catalog.search(min_recommendations=5)) == ["PILOT"]


@pytest.mark.parametrize("level", [0, 6])
def test_recommendation_level_out_of_range_is_rejected(catalog, level):
    assert "min_recommendations must be 1 to 5" in catalog.search(min_recommendations=level)["error"]


def test_most_recommended_first(catalog):
    without_query = catalog.search(size="205/55R16", sort="recommendations_desc")
    with_query = catalog.search(query="tire", size="205/55R16", sort="recommendations_desc")

    assert skus(without_query) == skus(with_query) == ["PILOT", "RUNFLAT", "CHEAP"]
    assert without_query["order"] == "recommendations_desc"
    assert with_query["order"] == "recommendations_desc among the 50 most relevant"


def test_no_match_says_so(catalog):
    result = catalog.search(size="205/55R16", brand="Kumho")

    assert result["total_matching"] == 0 and result["products"] == []
    assert "No products" in result["note"]


def test_searches_return_up_to_max_results_products(catalog):
    assert catalog.search()["returned"] == len(CATALOG)

    small = CatalogSearch(catalog.client, "tires", FakeEncoder(), max_results=2)
    by_price, by_relevance, query_by_price = small.search(), small.search(query="tire"), small.search(query="tire", sort="price_desc")

    assert skus(by_price) == ["CHEAP", "VINTAGE"]
    assert by_relevance["returned"] == 2
    assert skus(query_by_price) == ["LT-KO2", "PILOT"]
    assert all(r["total_matching"] == len(CATALOG) for r in (by_price, by_relevance, query_by_price))


def test_ranking_can_use_dense_or_bm25_vectors_alone(catalog):
    class SplitEncoder(FakeEncoder):
        """The dense vector points at the Michelin, the BM25 vector at the Bridgestone."""

        def encode_query(self, text):
            return self.dense("michelin pilot sport 4s"), self.sparse("bridgestone turanza rft")

    def top(ranking: str) -> list[str]:
        search = CatalogSearch(catalog.client, "tires", SplitEncoder(), max_results=20, ranking=ranking)
        return skus(search.search(query="any words", size="205/55R16"))[:2]

    assert top("dense")[0] == "PILOT"
    assert top("sparse")[0] == "RUNFLAT"
    assert set(top("hybrid")) == {"PILOT", "RUNFLAT"}


def test_embedding_failure_is_returned_as_an_error(catalog):
    class DownEncoder:
        def encode_query(self, text):
            raise EmbeddingError("OpenRouter daily free-model limit reached")

    broken = CatalogSearch(catalog.client, "tires", DownEncoder(), max_results=20)

    assert "temporarily unavailable" in broken.search(query="winter tires")["error"]
    assert broken.search(size="205/55R16")["total_matching"] == 3  # filters alone need no embedding


def test_tool_returns_json_and_lists_allowed_values(catalog):
    tool = make_search_tool(catalog)

    result = json.loads(tool.invoke({"size": "205/55R16", "sort": "price_asc"}))

    assert skus(result) == ["CHEAP", "RUNFLAT", "PILOT"]
    assert "limit" not in tool.args  # the result count is a setting, not the model's choice
    assert "up to 20 products" in tool.description
    assert "Light Truck, Passenger, Tractor, Truck/SUV" in tool.description
    assert "A1" in tool.description and "Y" in tool.description
    assert "available (false: out of stock" in tool.description and "recommendations_desc" in str(tool.args["sort"])


def test_tool_rejects_unknown_arguments(catalog):
    tool = make_search_tool(catalog)

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        tool.invoke({"size": "205/55R16", "speed_rating": "V"})


def test_no_tool_without_an_index():
    client = QdrantClient(":memory:")

    def factory():
        raise AssertionError("the encoder must not be built without an index")

    assert catalog_tools(client, "tires", factory, max_results=20) == []
    assert catalog_tools(None, "tires", factory, max_results=20) == []


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
        ({"min_recommendations": 4, "sort": "recommendations_desc"}, "Searching the catalog: recommended 4/5 or higher · most recommended first"),
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
