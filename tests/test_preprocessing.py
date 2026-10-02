import pandas as pd
import pytest
from conftest import raw_frame

from tiredai.preprocessing import PreprocessingError, normalize, preprocess, verify


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
    assert not output.with_name(output.name + ".tmp").exists()


def test_preprocess_failure_leaves_no_output(write_csv, tmp_path):
    output = tmp_path / "tires.parquet"

    with pytest.raises(PreprocessingError, match="Verification failed"):
        preprocess(write_csv({"price": "12.34567890123456789"}), output)

    assert list(tmp_path.glob("tires.parquet*")) == []
