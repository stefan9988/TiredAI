from collections import Counter

import pandas as pd
import pytest
from conftest import raw_frame

from tiredai.preprocessing import PreprocessingError, is_available, normalize, preprocess, recommendation_level, verify


def test_converts_typed_fields_and_renames_unit_columns():
    row = normalize(raw_frame()).iloc[0]

    assert row["price"] == 59.93
    assert row["overallDiameter"] == 23.9
    assert row["rimDiameter"] == 15.0
    assert not row["runFlat"]
    assert row["treadDepth32nds"] == 9
    assert row["mileageWarrantyMiles"] == 50000
    assert "treadDepth" not in row.index and "mileageWarranty" not in row.index


def test_missing_markers_become_null():
    row = normalize(raw_frame({"performance": "N/A", "treadDepth": "", "mileageWarranty": ""})).iloc[0]

    assert pd.isna(row["performance"])
    assert pd.isna(row["treadDepth32nds"])
    assert pd.isna(row["mileageWarrantyMiles"])


def test_size_separator_is_uppercased():
    assert normalize(raw_frame({"size": "38x13.50R26"})).iloc[0]["size"] == "38X13.50R26"


def test_other_text_columns_are_unchanged():
    row = normalize(raw_frame({"loadIndex": "123/120", "utqg": "300AAA", "model": "117/116"})).iloc[0]

    assert (row["loadIndex"], row["utqg"], row["model"]) == ("123/120", "300AAA", "117/116")


@pytest.mark.parametrize(
    "column, value",
    [
        ("treadDepth", "10/16"),
        ("runFlat", "yes"),
        ("price", "12,50"),
        ("mileageWarranty", "50000"),
        ("rimDiameter", "15 in"),
    ],
)
def test_unexpected_format_is_rejected(column, value):
    with pytest.raises(PreprocessingError, match=column):
        normalize(raw_frame({column: value}))


def test_generated_columns_are_fixed_by_the_sku():
    skus = ["N889368-99", "N891706-99", "N952097-99"]
    first = normalize(raw_frame(*({"sku": s} for s in skus)))
    reordered = normalize(raw_frame(*({"sku": s} for s in reversed(skus))))

    for column in ("available", "recommendations"):
        assert list(first[column]) == list(reordered[column])[::-1]
    assert list(first["available"]) == [is_available(s) for s in skus]
    assert list(first["recommendations"]) == [recommendation_level(s) for s in skus]
    assert str(first["available"].dtype) == "boolean" and str(first["recommendations"].dtype) == "Int64"


def test_generated_columns_follow_the_requested_distribution():
    skus = [f"N{i}-99" for i in range(10_000)]
    available = sum(is_available(s) for s in skus) / len(skus)
    levels = Counter(recommendation_level(s) for s in skus)

    assert 0.78 < available < 0.82
    assert set(levels) <= {1, 2, 3, 4, 5}
    assert 3.95 < sum(level * n for level, n in levels.items()) / len(skus) < 4.05
    # Normal around 4 with spread 0.6: about 20% / 60% / 20% at 3 / 4 / 5.
    shares = {level: n / len(skus) for level, n in levels.items()}
    assert 0.17 < shares[3] < 0.23 and 0.56 < shares[4] < 0.64 and 0.17 < shares[5] < 0.23


def test_verify_detects_a_changed_generated_value():
    raw = raw_frame()
    stored = normalize(raw)
    stored.loc[0, "recommendations"] = 6

    [problem] = verify(raw, stored)
    assert problem.startswith("recommendations row 0: stored") and "for sku 'SKU-0'" in problem


def test_verify_detects_float_precision_loss():
    raw = raw_frame({"price": "12.34567890123456789"})

    problems = verify(raw, normalize(raw))

    assert len(problems) == 1 and problems[0].startswith("price row 0")


def test_verify_detects_changed_value():
    raw = raw_frame()
    stored = normalize(raw)
    stored.loc[0, "treadDepth32nds"] = 10

    [problem] = verify(raw, stored)
    assert problem.startswith("treadDepth row 0: raw '9/32' -> stored")


def test_preprocess_writes_verified_parquet(write_csv, tmp_path):
    output = tmp_path / "out" / "tires.parquet"

    preprocess(write_csv({}, {"performance": "N/A"}), output)

    stored = pd.read_parquet(output)
    assert len(stored) == 2
    assert str(stored["price"].dtype) == "Float64"
    assert str(stored["treadDepth32nds"].dtype) == "Int64"
    assert str(stored["runFlat"].dtype) == "boolean"
    assert str(stored["available"].dtype) == "boolean" and str(stored["recommendations"].dtype) == "Int64"
    assert not output.with_name(output.name + ".tmp").exists()


def test_preprocess_failure_leaves_no_output(write_csv, tmp_path):
    output = tmp_path / "tires.parquet"

    with pytest.raises(PreprocessingError, match="Verification failed"):
        preprocess(write_csv({"price": "12.34567890123456789"}), output)

    assert list(tmp_path.glob("tires.parquet*")) == []
