import pytest

from tiredai.benchmarks.answers import check_facts, meets, mentions, words
from tiredai.benchmarks.catalog import Catalog


def tire(sku, name, size, price, **fields) -> dict:
    return {"sku": sku, "name": name, "size": size, "price": price, "available": True, "season": "All Season",
            "carType": "Passenger", **fields}  # fmt: skip


SEEN = list(
    Catalog(
        [
            tire("NEXEN", "Nexen Classe Premiere CP672 205/55R16 91V", "205/55R16", 84.64, treadDepth32nds=10,
                 mileageWarrantyMiles=70000, utqg="500AA", recommendations=4),
            tire("GEN-91", "General Altimax RT45 205/55R16 91V", "205/55R16", 124.99, treadDepth32nds=11, recommendations=3),
            tire("GEN-94", "General Altimax RT45 205/55R16 94V XL", "205/55R16", 131.50),
            tire("GEN-15", "General Altimax RT45 205/60R15 91H", "205/60R15", 99.99),
            tire("SUV", "Goodyear Eagle F1 Asymmetric SUV 255/50R19 107W", "255/50R19", 300.00),
            tire("SUV4X4", "Goodyear Eagle F1 Asymmetric SUV-4X4 (MO) 255/50R19 103W", "255/50R19", 241.99),
            tire("KELLY", "Kelly Edge Touring Plus 205/55R16 91V", "205/55R16", 95.68, available=False),
        ]  # fmt: skip
    )
)


def named(answer: str) -> list[tuple[str, ...]]:
    return [m.skus for m in mentions(answer, SEEN)]


def test_words_ignore_case_markdown_and_sentence_ends():
    assert words("**General Altimax RT45.** It's $124.99, H/T 5.20-13!") == "general altimax rt45 it s 124.99 h/t 5.20 13"


def test_products_are_found_by_their_line():
    assert named("I'd go with the **Nexen Classe Premiere CP672**.") == [("NEXEN",)]
    assert named("Nothing from Nexen here.") == []


def test_the_longest_matching_line_wins():
    assert named("The Goodyear Eagle F1 Asymmetric SUV-4X4 (MO) is in stock.") == [("SUV4X4",)]


def test_products_sharing_a_line_are_told_apart_by_size_service_and_price():
    assert named("General Altimax RT45 in 205/60R15") == [("GEN-15",)]
    assert named("General Altimax RT45 94V XL") == [("GEN-94",)]
    assert named("General Altimax RT45 for $124.99") == [("GEN-91",)]
    assert named("General Altimax RT45") == [("GEN-91", "GEN-94", "GEN-15")]


def test_a_sku_names_its_product_once():
    assert named("General Altimax RT45 (SKU N1530531-99)") == [("GEN-91", "GEN-94", "GEN-15")]
    seen = [*SEEN, *Catalog([tire("N1530531-99", "Michelin Defender 2 205/55R16 91H", "205/55R16", 181.86)])]
    assert [m.skus for m in mentions("Michelin Defender 2, SKU N1530531-99", seen)] == [("N1530531-99",)]


def test_products_called_out_of_stock_are_marked():
    [kelly] = mentions("- Kelly Edge Touring Plus – $95.68 (out of stock)", SEEN)
    assert kelly.unavailable


def test_a_table_column_saying_no_stock_marks_its_rows():
    answer = """| Tire | Price | Available |
|---|---|---|
| **Nexen Classe Premiere CP672** | $84.64 | ✅ Yes |
| **Kelly Edge Touring Plus** | $95.68 | ❌ No |

General Altimax RT45 is a good pick."""

    assert [(m.skus, m.unavailable) for m in mentions(answer, SEEN)] == [
        (("NEXEN",), False), (("KELLY",), True), (("GEN-91", "GEN-94", "GEN-15"), False)
    ]  # fmt: skip


def facts(answer: str, shopper=frozenset()) -> list[tuple[str, str, bool]]:
    return [(f.kind, f.text, f.supported) for f in check_facts(answer, SEEN, set(shopper))]


def test_a_price_next_to_a_product_must_be_its_price():
    assert facts("Nexen Classe Premiere CP672: $84.64") == [("price", "$84.64", True)]
    assert facts("Nexen Classe Premiere CP672: $124.99") == [("price", "$124.99", False)]  # another tire's price


def test_prices_for_sets_rounded_prices_and_budgets_are_supported():
    assert facts("Nexen Classe Premiere CP672: $338.56 for four") == [("price", "$338.56", True)]
    assert facts("Nexen Classe Premiere CP672 at about $85") == [("price", "$85", True)]
    assert facts("Nexen Classe Premiere CP672, under your $100 budget", shopper={100.0}) == [("price", "$100", True)]


def test_a_summary_line_may_quote_any_price_it_was_shown():
    assert facts("Prices run from $84.64 to $131.50, not $75.") == [
        ("price", "$84.64", True), ("price", "$131.50", True), ("price", "$75", False)
    ]  # fmt: skip


def test_specs_are_checked_against_the_product_the_lines_are_about():
    answer = """1. **Nexen Classe Premiere CP672** – $84.64
   - Tread depth: 10/32", warranty 70,000 miles, UTQG 500AA, recommended 4/5
2. **General Altimax RT45 205/55R16 91V** – $124.99
   - Tread depth: 10/32", 3/5"""

    assert facts(answer) == [
        ("price", "$84.64", True), ("tread_depth", "10/32", True), ("warranty", "70,000 miles", True),
        ("utqg", "500AA", True), ("recommendations", "4/5", True),
        ("price", "$124.99", True), ("tread_depth", "10/32", False), ("recommendations", "3/5", True),
    ]  # fmt: skip


def test_specs_without_a_product_are_general_knowledge_and_unchecked():
    assert facts("New tires have about 10/32 of tread; replace them at 2/32.") == []
    assert facts("Most tires carry a 50k mile warranty.\n\nNexen Classe Premiere CP672 has 70k miles.") == [("warranty", "70k miles", True)]


def test_skus_must_be_among_the_products_shown():
    assert facts("SKU NEXEN") == []  # not a SKU pattern
    assert facts("SKU N1234567-99") == [("sku", "N1234567-99", False)]


@pytest.mark.parametrize(
    "constraints, failed",
    [
        ({"size": "205/55R16", "season": "all season"}, []),
        ({"size": "205 55 16", "max_price": 90}, []),
        ({"size": "LT205/55R16"}, ["size"]),  # an LT size needs a light-truck tire
        ({"max_price": 80, "min_price": 90}, ["max_price", "min_price"]),
        ({"brand": "Michelin", "car_type": "Passenger", "run_flat": False}, ["brand"]),
    ],
)
def test_constraints_are_checked_like_the_search_filters(constraints, failed):
    nexen = SEEN[0] | {"brand": "Nexen", "runFlat": False}
    assert meets(nexen, constraints) == failed


def test_phrases_match_whole_words():
    from tiredai.benchmarks.answers import has_phrase

    assert has_phrase("I only help with **tires**.", "tire")
    assert not has_phrase("I'm TiredAI.", "tire")
    assert has_phrase("It costs $181.86.", "$181.86") and has_phrase("We don’t carry it", "don't carry")
    assert not has_phrase("discount codes: none", "discount code:")
