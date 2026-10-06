import random

import pytest
from conftest import FakeEncoder
from qdrant_client import QdrantClient

from tiredai.benchmarks.catalog import Catalog, gain, product_line
from tiredai.benchmarks.experiments import BENCHMARKS_DIR
from tiredai.benchmarks.metrics import hit_at, ndcg_at, percentile, precision_at, reciprocal_rank
from tiredai.benchmarks.retrieval import (
    RankingTask,
    RetrievalQuery,
    check_queries,
    load_queries,
    loose_size,
    product_queries,
    ranking_evaluator,
    with_typo,
    write_product_queries,
)
from tiredai.search import CatalogSearch
from tiredai.vectorstore import index_products


def product(sku: str, name: str, size: str, **fields) -> dict:
    return {"sku": sku, "name": name, "size": size, "carType": "Passenger", "season": "All Season", "price": 100.0,
            "available": True, **fields}  # fmt: skip


PRODUCTS = [
    product("PHI-92", "Accelera Phi-R 205/55R15 92V XL", "205/55R15"),
    product("PHI-88", "Accelera Phi-R 205/55R15 88V", "205/55R15"),
    product("PHI-16", "Accelera Phi-R 205/55R16 91V", "205/55R16"),
    product("BLADE", "Atturo Trail Blade H/T LT 265/70R17 123/120S E (10 Ply)", "265/70R17", carType="Light Truck"),
    product("WINTER", "Hankook Winter i*Pike RS2 225/45R17 94T XL", "225/45R17", season="Winter"),
    product("VINTAGE", "Firestone Deluxe Champion 5.20-13 (WWW)", "5.2-13"),
    product("TRACTOR", "BKT TR-135 8-16 6 Ply (TT)", "8-16", carType="Tractor", season="All Season"),
]


@pytest.fixture
def catalog():
    return Catalog(PRODUCTS)


def test_ranking_metrics():
    ranking = [False, True, False, True]

    assert (hit_at(ranking, 1), hit_at(ranking, 3)) == (0.0, 1.0)
    assert reciprocal_rank(ranking) == 0.5 and reciprocal_rank([False, False]) == 0.0
    assert precision_at(ranking, 4) == 0.5 and precision_at([True], 10) == 0.1
    assert ndcg_at([1, 1], [1, 1, 0.5], 2) == 1.0
    assert 0 < ndcg_at([0, 1], [1, 1], 2) < ndcg_at([1, 0], [1, 1], 2) < 1
    assert ndcg_at([0.5], [0, 0], 1) == 0.0
    assert percentile([1, 2, 3, 4], 0.5) == 2 and percentile([3, 1, 2], 0.95) == 3 and percentile([], 0.5) is None


@pytest.mark.parametrize(
    "sku, line",
    [("PHI-92", "Accelera Phi-R"), ("BLADE", "Atturo Trail Blade H/T"), ("TRACTOR", "BKT TR-135"),
     # the size isn't written like the size field, so the name is its own line
     ("VINTAGE", "Firestone Deluxe Champion 5.20-13 (WWW)")],
)  # fmt: skip
def test_product_line_is_the_name_before_the_size(catalog, sku, line):
    assert product_line(catalog[sku]) == line == catalog[sku]["line"]


def test_gain_is_the_share_of_matching_fields(catalog):
    relevant = {"line": ["Accelera Phi-R"], "size": ["205/55R15"]}

    assert gain(catalog["PHI-88"], relevant) == 1.0
    assert gain(catalog["PHI-16"], relevant) == 0.5
    assert gain(catalog["WINTER"], {"season": ["Winter", "All Weather"]}) == 1.0


def test_sizes_are_loosened_like_shoppers_type_them():
    assert loose_size("205/55R15") == "205 55 15"
    assert loose_size("245/40ZR18") == "245 40 18"
    assert loose_size("215/75R17.5") == "215 75 17.5"
    assert loose_size("31X10.00R14") == "31x10.00r14"


def test_a_typo_drops_one_inner_letter_of_a_word():
    for seed in range(20):
        typo = with_typo("Accelera Phi-R", random.Random(seed))
        assert typo.endswith(" Phi-R")  # not a letters-only word
        word = typo.split(" ")[0]
        assert len(word) == len("Accelera") - 1 and word[0] == "A" and word[-1] == "a"
    assert with_typo("BKT TR-135", random.Random(0)) is None


def test_product_queries_cover_every_style_and_are_reproducible(catalog):
    queries = product_queries(catalog, products=3, seed=1)

    assert product_queries(catalog, products=3, seed=1) == queries
    by_style = {}
    for q in queries:
        by_style.setdefault(q.source_sku, {})[q.style] = q
    # Car types are sampled in proportion, and every type gets at least one product (here 2 + 1 + 1).
    assert sorted(catalog[sku]["carType"] for sku in by_style) == ["Light Truck", "Passenger", "Passenger", "Tractor"]
    for sku, styles in by_style.items():
        p = catalog[sku]
        assert styles["full_name"].query == p["name"] and styles["full_name"].relevant == {"sku": [sku]}
        assert styles["line_only"].relevant == {"line": [p["line"]]}
        assert styles["shopper"].relevant == {"line": [p["line"]], "size": [p["size"]]}
        assert styles["shopper"].query == styles["shopper"].query.lower()
    assert "VINTAGE" not in by_style  # its name doesn't hold the size field


def test_queries_round_trip_through_their_files(catalog, tmp_path):
    queries = product_queries(catalog, products=2, seed=3)
    write_product_queries(tmp_path / "products.yaml", queries, "generated")
    (tmp_path / "descriptive.yaml").write_text("- {id: winter, query: snow tires, relevant: {season: [Winter]}}\n")

    loaded = load_queries(tmp_path / "products.yaml", tmp_path / "descriptive.yaml")

    assert loaded[:-1] == queries
    assert loaded[-1].kind == "descriptive" and loaded[-1].relevant == {"season": ["Winter"]}


def test_queries_that_cannot_be_scored_are_reported(catalog):
    queries = [
        RetrievalQuery(id="typo-field", query="x", relevant={"seasons": ["Winter"]}),
        RetrievalQuery(id="too-few", query="snow tires", relevant={"season": ["Winter"]}),
        RetrievalQuery(id="gone", kind="product", query="x", relevant={"sku": ["NOPE"]}, source_sku="NOPE"),
        RetrievalQuery(id="fine", kind="product", query="x", relevant={"sku": ["PHI-92"]}, source_sku="PHI-92"),
    ]

    problems = check_queries(queries, catalog)

    assert [p.split(":")[0] for p in problems] == ["typo-field", "too-few", "gone", "gone"]


def test_committed_query_files_are_valid():
    queries = load_queries(BENCHMARKS_DIR / "retrieval_products.yaml", BENCHMARKS_DIR / "retrieval_descriptive.yaml")

    assert {q.kind for q in queries} == {"product", "descriptive"}
    assert all(q.style in ("full_name", "shopper", "typo", "line_only") for q in queries if q.kind == "product")


@pytest.fixture
def search(catalog):
    client = QdrantClient(":memory:")
    index_products(client, "benchmark", PRODUCTS, FakeEncoder())
    yield CatalogSearch(client, "benchmark", FakeEncoder(), max_results=10)
    client.close()


def test_a_named_product_query_is_scored_by_where_its_product_ranks(catalog, search):
    item = RetrievalQuery(id="q", kind="product", query="hankook winter i*pike rs2", relevant={"sku": ["WINTER"]}).case()

    output = RankingTask(search)(item={"input": item.input})
    scores = {e.name: e.value for e in ranking_evaluator(catalog)(output=output, expected_output=item.expected_output, metadata=item.metadata)}

    assert output["ranking"][0] == "WINTER" and output["top3"][0] == "Hankook Winter i*Pike RS2 225/45R17 94T XL"
    assert scores == {"hit@1": 1.0, "hit@3": 1.0, "hit@10": 1.0, "reciprocal_rank": 1.0}


def test_a_descriptive_query_is_scored_by_precision_and_ndcg(catalog):
    evaluate = ranking_evaluator(catalog)
    expected = {"relevant": {"season": ["Winter"], "carType": ["Passenger"]}}

    def scores(ranking):
        return {e.name: e.value for e in evaluate(output={"ranking": ranking}, expected_output=expected, metadata={"kind": "descriptive"})}

    # The winter tire matches both fields; the other passenger tires match one.
    best = scores(["WINTER", "PHI-92", "PHI-88", "PHI-16", "VINTAGE"])
    worse = scores(["PHI-92", "WINTER", "PHI-88", "PHI-16", "VINTAGE"])

    assert best["precision@10"] == worse["precision@10"] == 0.1  # one fully relevant product in the catalog
    assert best["ndcg@10"] == 1.0 > worse["ndcg@10"]


def test_a_failed_search_fails_the_item(search):
    class Down:
        def encode_query(self, text):
            from tiredai.embeddings import EmbeddingError

            raise EmbeddingError("quota")

    broken = CatalogSearch(search.client, "benchmark", Down(), max_results=10)

    with pytest.raises(RuntimeError, match="temporarily unavailable"):
        RankingTask(broken)(item={"input": {"query": "tires"}})
