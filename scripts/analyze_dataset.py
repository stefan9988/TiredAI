"""Profile the tire catalog CSV: missing data, unique values, distributions and consistency checks.

Usage:
    uv run python scripts/analyze_dataset.py [path/to/catalog.csv] [--examples N]
"""

import argparse
import re
from pathlib import Path

import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data" / "tires_sample_10k_sku.csv"

# Values the catalog uses to mean "no data" besides an empty cell.
PLACEHOLDERS = {"N/A", "NA", "n/a", "None", "null", "-"}

# Columns with at most this many distinct values get their full value counts printed.
LOW_CARDINALITY = 40
TOP_N = 10

NUMBER = re.compile(r"\d+(?:\.\d+)?")


def section(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def is_missing(series: pd.Series) -> pd.Series:
    stripped = series.str.strip()
    return (stripped == "") | stripped.isin(PLACEHOLDERS)


def canonical_numbers(text: str) -> str:
    """Uppercase and rewrite every number in canonical form, so '5.20-13' and '5.2-13' compare equal."""
    return NUMBER.sub(lambda m: f"{float(m.group()):g}", text.upper())


def size_pattern(size: str) -> str:
    return NUMBER.sub("#", size) if size else "<empty>"


def rim_from_size(size: str) -> float | None:
    # Vintage sizes such as '28X3' are overall diameter x width, so the rim is not encoded.
    if re.fullmatch(r"[\d.]+X[\d.]+", size, re.IGNORECASE):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)C?$", size)
    return float(match.group(1)) if match else None


def overview(df: pd.DataFrame, path: Path) -> None:
    section("OVERVIEW")
    print(f"File:     {path}")
    print(f"Size:     {path.stat().st_size / 1024:,.0f} KiB")
    print(f"Rows:     {len(df):,}")
    print(f"Columns:  {len(df.columns)}")
    print(f"Names:    {', '.join(df.columns)}")


def missing_data(df: pd.DataFrame) -> None:
    section("MISSING DATA  (empty cells and placeholder values such as 'N/A')")
    rows = []
    for col in df.columns:
        stripped = df[col].str.strip()
        empty = int((stripped == "").sum())
        placeholder = int(stripped.isin(PLACEHOLDERS).sum())
        rows.append(
            {
                "column": col,
                "empty": empty,
                "placeholder": placeholder,
                "missing": empty + placeholder,
                "missing %": round(100 * (empty + placeholder) / len(df), 2),
            }
        )
    report = pd.DataFrame(rows).set_index("column").sort_values("missing", ascending=False)
    print(report.to_string())

    missing = df.apply(is_missing)
    print(f"\nRows with no missing values:        {int((~missing.any(axis=1)).sum()):,}")
    print(f"Rows with at least one missing:     {int(missing.any(axis=1).sum()):,}")

    mixed = report[(report["empty"] > 0) & (report["placeholder"] > 0)].index.tolist()
    if mixed:
        print(f"Columns mixing empty AND placeholder markers: {', '.join(mixed)}")

    # Missingness is often structural (e.g. no UTQG for tractor tires), so break it down by vehicle type.
    cols = [c for c in report.index if report.loc[c, "missing"] > 0]
    if cols and "carType" in df.columns:
        print("\nMissing % by carType (shows whether gaps are structural or random):")
        by_type = missing[cols].groupby(df["carType"]).mean().mul(100).round(1)
        by_type.insert(0, "rows", df["carType"].value_counts())
        print(by_type.to_string())


def unique_values(df: pd.DataFrame) -> None:
    section("UNIQUE VALUES")
    summary = pd.DataFrame(
        {
            "unique": df.nunique(),
            "unique %": (100 * df.nunique() / len(df)).round(2),
            "top value": df.apply(lambda s: s.replace("", "<empty>").value_counts().index[0]),
            "top count": df.apply(lambda s: s.value_counts().iloc[0]),
        }
    )
    print(summary.to_string())

    for col in df.columns:
        counts = df[col].replace("", "<empty>").value_counts()
        if len(counts) == len(df):
            continue  # identifier-like column, every value is distinct
        full = len(counts) <= LOW_CARDINALITY
        label = "all values" if full else f"top {TOP_N} of {len(counts):,}"
        print(f"\n-- {col} ({label})")
        shown = counts if full else counts.head(TOP_N)
        print(shown.to_frame("count").assign(pct=(100 * shown / len(df)).round(2)).to_string(header=False))


def parsed_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric views of columns that are stored as text with units or composite values."""
    return pd.DataFrame(
        {
            "price": pd.to_numeric(df["price"], errors="coerce"),
            "overallDiameter": pd.to_numeric(df["overallDiameter"], errors="coerce"),
            "rimDiameter": pd.to_numeric(df["rimDiameter"], errors="coerce"),
            "loadIndex (first)": pd.to_numeric(df["loadIndex"].str.split("/").str[0], errors="coerce"),
            "treadDepth (/32 in)": pd.to_numeric(df["treadDepth"].str.extract(r"^(\d+)/32$")[0], errors="coerce"),
            "mileageWarranty (mi)": pd.to_numeric(
                df["mileageWarranty"].str.extract(r"^([\d,]+) miles$")[0].str.replace(",", ""), errors="coerce"
            ),
            "utqg treadwear": pd.to_numeric(df["utqg"].str.extract(r"^(\d+)")[0], errors="coerce"),
        }
    )


def numeric_stats(df: pd.DataFrame, numeric: pd.DataFrame, examples: int) -> None:
    section("NUMERIC DISTRIBUTIONS")
    stats = numeric.describe().T
    source = {"loadIndex (first)": "loadIndex", "treadDepth (/32 in)": "treadDepth",
              "mileageWarranty (mi)": "mileageWarranty", "utqg treadwear": "utqg"}
    # A value that is present in the CSV but fails to parse points to an unexpected format.
    stats["unparseable"] = [
        int((numeric[c].isna() & ~is_missing(df[source.get(c, c)])).sum()) for c in numeric.columns
    ]
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    iqr = q3 - q1
    stats["extreme outliers (3xIQR)"] = ((numeric < q1 - 3 * iqr) | (numeric > q3 + 3 * iqr)).sum()
    print(stats.round(2).to_string())

    price = numeric["price"]
    cols = ["sku", "name", "carType", "price"]
    print(f"\nMost expensive {examples}:")
    print(df.loc[price.nlargest(examples).index, cols].to_string())
    print(f"\nCheapest {examples}:")
    print(df.loc[price.nsmallest(examples).index, cols].to_string())

    print("\nPrice by carType:")
    print(price.groupby(df["carType"]).describe()[["count", "min", "25%", "50%", "75%", "max"]].round(2).to_string())
    print("\nPrice by season:")
    print(price.groupby(df["season"]).describe()[["count", "min", "50%", "max"]].round(2).to_string())


def duplicates(df: pd.DataFrame) -> None:
    section("DUPLICATES")
    checks = {
        "duplicate sku": df.duplicated("sku", keep=False),
        "duplicate name": df.duplicated("name", keep=False),
        "duplicate name (case-insensitive)": df["name"].str.lower().str.strip().duplicated(keep=False),
        "identical rows": df.duplicated(keep=False),
        "identical apart from sku": df.drop(columns="sku").duplicated(keep=False),
        "same brand + model + size (load/speed variants)": df.duplicated(["brand", "model", "size"], keep=False),
    }
    for label, mask in checks.items():
        print(f"{label:<50} {int(mask.sum()):>6,} rows")


def consistency(df: pd.DataFrame, numeric: pd.DataFrame, examples: int) -> None:
    section("CONSISTENCY CHECKS")
    name_canon = df["name"].map(canonical_numbers)
    size_canon = df["size"].map(canonical_numbers)
    rim_parsed = df["size"].map(rim_from_size).astype(float)
    load_speed = df["loadIndex"] + df["speedRating"]
    has_load_speed = ~is_missing(df["loadIndex"]) & ~is_missing(df["speedRating"])

    checks = [
        (
            "name does not start with brand",
            ~df.apply(lambda r: r["name"].startswith(r["brand"]), axis=1),
            ["name", "brand"],
        ),
        (
            "model text not found in name (model column unreliable)",
            ~df.apply(lambda r: r["model"] in r["name"], axis=1),
            ["name", "brand", "model"],
        ),
        (
            "model looks like a number/size, not a model name",
            df["model"].str.fullmatch(r"[\d\s./xX-]+"),
            ["name", "model"],
        ),
        (
            "model is another brand's name",
            df["model"].isin(set(df["brand"])) & (df["model"] != df["brand"]),
            ["name", "brand", "model"],
        ),
        (
            "size missing",
            is_missing(df["size"]),
            ["sku", "name", "size", "rimDiameter"],
        ),
        (
            "size not found in name (after normalizing numbers)",
            ~is_missing(df["size"]) & ~pd.Series([s in n for s, n in zip(size_canon, name_canon)], index=df.index),
            ["name", "size"],
        ),
        (
            "rimDiameter differs from rim in size",
            rim_parsed.notna() & ((rim_parsed - numeric["rimDiameter"]).abs() > 0.01),
            ["name", "size", "rimDiameter"],
        ),
        (
            "loadIndex+speedRating not found in name",
            has_load_speed & ~pd.Series([ls in n for ls, n in zip(load_speed, df["name"])], index=df.index),
            ["name", "loadIndex", "speedRating"],
        ),
        (
            "rimDiameter > 40 (likely millimetres, metric sizing)",
            numeric["rimDiameter"] > 40,
            ["name", "size", "rimDiameter", "carType"],
        ),
        (
            "overallDiameter <= rimDiameter (impossible for inch rims)",
            (numeric["rimDiameter"] <= 40) & (numeric["overallDiameter"] <= numeric["rimDiameter"]),
            ["name", "size", "overallDiameter", "rimDiameter"],
        ),
        (
            "overallDiameter > 100 (likely unit or decimal error)",
            numeric["overallDiameter"] > 100,
            ["name", "size", "overallDiameter"],
        ),
        (
            "price <= 0",
            numeric["price"] <= 0,
            ["sku", "name", "price"],
        ),
        (
            "runFlat not true/false",
            ~df["runFlat"].isin(["true", "false"]),
            ["name", "runFlat"],
        ),
        (
            "cells with leading/trailing whitespace",
            df.apply(lambda s: s != s.str.strip()).any(axis=1),
            ["sku", "name"],
        ),
    ]

    for label, mask, cols in checks:
        mask = mask.fillna(False).astype(bool)
        count = int(mask.sum())
        print(f"\n[{'FAIL' if count else ' OK '}] {label}: {count:,} rows ({100 * count / len(df):.2f}%)")
        if count:
            print(df.loc[mask, cols].head(examples).to_string())


def size_formats(df: pd.DataFrame) -> None:
    section("SIZE FORMATS  (digits replaced by #)")
    patterns = df["size"].map(size_pattern)
    report = (
        df.assign(pattern=patterns)
        .groupby("pattern")
        .agg(count=("size", "size"), example=("size", "first"), carTypes=("carType", lambda s: ", ".join(s.unique()[:3])))
        .sort_values("count", ascending=False)
    )
    print(report.to_string())


def requirements_coverage(df: pd.DataFrame) -> None:
    section("COVERAGE OF FIELDS NEEDED BY THE CHATBOT")
    wanted = {
        "availability / stock": r"avail|stock|inventory|qty|quantity",
        "price": r"price",
        "size": r"^size$",
        "season": r"season",
        "performance category": r"perform",
        "SKU": r"sku",
    }
    for label, pattern in wanted.items():
        found = [c for c in df.columns if re.search(pattern, c, re.IGNORECASE)]
        print(f"{label:<25} {'-> ' + ', '.join(found) if found else 'NOT FOUND'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV, help="catalog CSV (default: %(default)s)")
    parser.add_argument("--examples", type=int, default=5, help="example rows shown per finding (default: 5)")
    args = parser.parse_args()

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 70)
    pd.set_option("display.max_rows", 500)

    # Read everything as text so empty cells and 'N/A' placeholders stay distinguishable.
    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    numeric = parsed_numeric(df)

    overview(df, args.csv)
    missing_data(df)
    unique_values(df)
    numeric_stats(df, numeric, args.examples)
    duplicates(df)
    consistency(df, numeric, args.examples)
    size_formats(df)
    requirements_coverage(df)


if __name__ == "__main__":
    main()
