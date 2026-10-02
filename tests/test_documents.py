import uuid

from conftest import raw_frame

from tiredai.documents import document_text, point_id, products
from tiredai.preprocessing import normalize


def product(**overrides) -> dict:
    return products(normalize(raw_frame(overrides)))[0]


def test_document_text_describes_the_tire_in_words():
    assert document_text(product()) == "Accelera Phi-R 205/55R15 92V XL. All Season Performance tire for Passenger."


def test_missing_performance_is_left_out():
    assert document_text(product(performance="N/A")) == "Accelera Phi-R 205/55R15 92V XL. All Season tire for Passenger."


def test_run_flat_is_only_mentioned_when_true():
    assert "Run-flat." in document_text(product(runFlat="true"))
    assert "flat" not in document_text(product(runFlat="false")).lower()


def test_sidewall_is_only_mentioned_when_not_black():
    assert document_text(product(sidewall="RWL: Raised White Letters")).endswith("Raised White Letters sidewall.")
    assert "sidewall" not in document_text(product()).lower()


def test_numbers_outside_the_name_are_not_embedded():
    text = document_text(product())

    for value in ("59.93", "50000", "50,000", "400AA", "23.9"):
        assert value not in text


def test_products_drop_nulls_and_use_plain_python_types():
    item = product(performance="N/A", treadDepth="", mileageWarranty="")

    assert "performance" not in item and "treadDepth32nds" not in item and "mileageWarrantyMiles" not in item
    assert type(item["price"]) is float
    assert type(item["runFlat"]) is bool
    assert type(item["sku"]) is str
    assert type(product()["treadDepth32nds"]) is int


def test_point_id_is_a_stable_uuid_per_sku():
    assert point_id("N889368-99") == point_id("N889368-99")
    assert point_id("N889368-99") != point_id("N889368-98")
    uuid.UUID(point_id("N889368-99"))
