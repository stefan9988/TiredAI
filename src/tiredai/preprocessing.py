"""Normalize the tire catalog without losing information.

Only conversions confirmed to be lossless are applied:
  * every column:    'N/A' and empty cells become null
  * size:            lowercase 'x' separator becomes 'X' ('38x13.50R26' -> '38X13.50R26')
  * price, overallDiameter, rimDiameter: text -> float
  * runFlat:         'true' / 'false' -> bool
  * treadDepth:      '10/32' -> treadDepth32nds = 10
  * mileageWarranty: '50,000 miles' -> mileageWarrantyMiles = 50000

Two columns the raw CSV lacks are generated from each product's SKU, so every run gives the same values:
  * available:       true for about 80% of products (false: out of stock)
  * recommendations: the store's recommendation level, a whole number 1-5 drawn from a normal
                     distribution around 4 (spread 0.6)

The Parquet file is read back after writing and every value is checked against the raw CSV (and
the generated columns against their SKU). If anything does not match, the output is discarded and
PreprocessingError is raised.
"""

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from statistics import NormalDist

import pandas as pd

MISSING = {"", "N/A"}

DECIMAL = re.compile(r"\d+(?:\.\d+)?")
TREAD_32NDS = re.compile(r"(\d+)/32")
MILES = re.compile(r"(\d{1,3}(?:,\d{3})*) miles")
BOOLEANS = {"true": True, "false": False}

AVAILABLE_SHARE = 0.8
RECOMMENDATION_LEVELS = NormalDist(mu=4, sigma=0.6)


class PreprocessingError(Exception):
    pass


def fullmatch(pattern: re.Pattern, text: str) -> re.Match:
    match = pattern.fullmatch(text)
    if match is None:
        raise ValueError(f"{text!r} does not match {pattern.pattern!r}")
    return match


def parse_decimal(text: str) -> float:
    return float(fullmatch(DECIMAL, text).group())


def same_decimal(raw: str, value: object) -> bool:
    # Compared as exact decimals, so '29.30' == 29.3 but any float rounding would be caught.
    return Decimal(raw) == Decimal(str(float(value)))


def parse_bool(text: str) -> bool:
    if text not in BOOLEANS:
        raise ValueError(f"{text!r} is not one of {sorted(BOOLEANS)}")
    return BOOLEANS[text]


@dataclass(frozen=True)
class Conversion:
    target: str
    dtype: str
    parse: Callable[[str], object]
    # True when the stored value still carries everything the raw text did.
    matches: Callable[[str, object], bool]


CONVERSIONS = {
    "size": Conversion(
        "size", "string", lambda s: s.replace("x", "X"), lambda raw, v: v == raw.replace("x", "X")
    ),
    "price": Conversion("price", "Float64", parse_decimal, same_decimal),
    "overallDiameter": Conversion("overallDiameter", "Float64", parse_decimal, same_decimal),
    "rimDiameter": Conversion("rimDiameter", "Float64", parse_decimal, same_decimal),
    "runFlat": Conversion("runFlat", "boolean", parse_bool, lambda raw, v: raw == ("true" if v else "false")),
    "treadDepth": Conversion(
        "treadDepth32nds",
        "Int64",
        lambda s: int(fullmatch(TREAD_32NDS, s).group(1)),
        lambda raw, v: raw == f"{int(v)}/32",
    ),
    "mileageWarranty": Conversion(
        "mileageWarrantyMiles",
        "Int64",
        lambda s: int(fullmatch(MILES, s).group(1).replace(",", "")),
        lambda raw, v: raw == f"{int(v):,} miles",
    ),
}


def sku_fraction(sku: str, column: str) -> float:
    """A number in (0, 1) fixed by the SKU and column, so generated values never change between runs."""
    digest = hashlib.sha256(f"{column}:{sku}".encode()).digest()
    return ((int.from_bytes(digest[:8], "big") >> 11) + 0.5) / 2**53  # 53 bits: exact as a float


def is_available(sku: str) -> bool:
    return sku_fraction(sku, "available") < AVAILABLE_SHARE


def recommendation_level(sku: str) -> int:
    return min(5, max(1, round(RECOMMENDATION_LEVELS.inv_cdf(sku_fraction(sku, "recommendations")))))


@dataclass(frozen=True)
class Generated:
    dtype: str
    generate: Callable[[str], object]  # the value for a SKU


GENERATED = {
    "available": Generated("boolean", is_available),
    "recommendations": Generated("Int64", recommendation_level),
}


def conversion_for(column: str) -> Conversion:
    return CONVERSIONS.get(column, Conversion(column, "string", str, lambda raw, v: raw == v))


def read_raw(path: Path) -> pd.DataFrame:
    # Read everything as text so no value is coerced before the conversions above.
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    columns = {}
    for col in raw.columns:
        conv = conversion_for(col)
        values = []
        for row, text in raw[col].items():
            if text in MISSING:
                values.append(None)
                continue
            try:
                values.append(conv.parse(text))
            except ValueError as exc:
                raise PreprocessingError(
                    f"Cannot normalize {col} at row {row} (sku {raw.at[row, 'sku']}): {exc}"
                ) from exc
        columns[conv.target] = pd.array(values, dtype=conv.dtype)
    for column, gen in GENERATED.items():
        columns[column] = pd.array([gen.generate(sku) for sku in raw["sku"]], dtype=gen.dtype)
    return pd.DataFrame(columns, index=raw.index)


def verify(raw: pd.DataFrame, stored: pd.DataFrame) -> list[str]:
    expected = [conversion_for(c).target for c in raw.columns] + list(GENERATED)
    if len(stored) != len(raw):
        return [f"row count {len(stored):,} != {len(raw):,}"]
    if list(stored.columns) != expected:
        return [f"columns {list(stored.columns)} != {expected}"]

    problems = []
    for col in raw.columns:
        conv = conversion_for(col)
        if str(stored[conv.target].dtype) != conv.dtype:
            problems.append(f"{conv.target}: dtype {stored[conv.target].dtype} != {conv.dtype}")
        for row, (text, value) in enumerate(zip(raw[col], stored[conv.target])):
            if text in MISSING:
                ok = pd.isna(value)
            else:
                ok = not pd.isna(value) and conv.matches(text, value)
            if not ok:
                problems.append(f"{col} row {row}: raw {text!r} -> stored {value!r}")
    for column, gen in GENERATED.items():
        if str(stored[column].dtype) != gen.dtype:
            problems.append(f"{column}: dtype {stored[column].dtype} != {gen.dtype}")
        for row, (sku, value) in enumerate(zip(raw["sku"], stored[column])):
            if pd.isna(value) or value != gen.generate(sku):
                problems.append(f"{column} row {row}: stored {value!r}, expected {gen.generate(sku)!r} for sku {sku!r}")
    return problems


def preprocess(input_path: Path, output_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize the raw CSV into a verified Parquet file and return (raw, normalized)."""
    raw = read_raw(input_path)
    normalized = normalize(raw)

    # Write to a temporary file and only keep it once the reloaded data has been verified.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".tmp")
    normalized.to_parquet(tmp, index=False)
    problems = verify(raw, pd.read_parquet(tmp))
    if problems:
        tmp.unlink()
        details = "\n".join(f"  {p}" for p in problems[:20])
        raise PreprocessingError(f"Verification failed with {len(problems):,} mismatches, nothing written:\n{details}")
    tmp.replace(output_path)
    return raw, normalized


def summary(raw: pd.DataFrame, normalized: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in raw.columns:
        conv = conversion_for(col)
        out = normalized[conv.target]
        edited = ""
        if conv.dtype == "string":
            edited = sum(1 for t, v in zip(raw[col], out) if t not in MISSING and v != t)
        rows.append(
            {
                "column": col,
                "output": conv.target,
                "dtype": conv.dtype,
                "N/A -> null": int((raw[col] == "N/A").sum()),
                "empty -> null": int((raw[col] == "").sum()),
                "text edited": edited,
            }
        )
    return pd.DataFrame(rows)
