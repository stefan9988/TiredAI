import pandas as pd
import pytest

# One valid raw CSV row, as text exactly like the catalog stores it.
RAW_ROW = {
    "sku": "N889368-99",
    "name": "Accelera Phi-R 205/55R15 92V XL",
    "brand": "Accelera",
    "model": "Phi-R",
    "size": "205/55R15",
    "season": "All Season",
    "carType": "Passenger",
    "performance": "Performance",
    "speedRating": "V",
    "loadIndex": "92",
    "loadRange": "XL",
    "runFlat": "false",
    "sidewall": "BSW: Black Side Wall",
    "utqg": "400AA",
    "treadDepth": "9/32",
    "mileageWarranty": "50,000 miles",
    "overallDiameter": "23.90",
    "rimDiameter": "15",
    "price": "59.930000",
}


def raw_frame(*overrides: dict) -> pd.DataFrame:
    """Raw catalog rows: one per overrides dict, each based on RAW_ROW with a unique SKU."""
    rows = [{**RAW_ROW, "sku": f"SKU-{i}", **o} for i, o in enumerate(overrides or [{}])]
    return pd.DataFrame(rows, dtype=str)


@pytest.fixture
def write_csv(tmp_path):
    def write(*overrides: dict):
        path = tmp_path / "catalog.csv"
        raw_frame(*overrides).to_csv(path, index=False)
        return path

    return write
