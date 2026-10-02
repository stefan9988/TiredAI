"""Normalize the raw tire CSV into a verified Parquet file (see tiredai.preprocessing for the rules).

Usage:
    uv run python scripts/preprocess_dataset.py [input.csv] [--output path.parquet]
"""

import argparse
import sys
from pathlib import Path

from tiredai.config import Settings
from tiredai.preprocessing import PreprocessingError, preprocess, summary


def main() -> None:
    settings = Settings.load()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", type=Path, default=settings.raw_data_path, help="raw CSV (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=settings.processed_data_path, help="Parquet file (default: %(default)s)")
    args = parser.parse_args()

    try:
        raw, normalized = preprocess(args.input, args.output)
    except PreprocessingError as exc:
        sys.exit(str(exc))

    print(summary(raw, normalized).to_string(index=False))
    print(f"\nWrote {len(normalized):,} rows x {len(normalized.columns)} columns to {args.output}")
    print("Verified: every stored value round-trips to the raw CSV ('N/A' and empty both map to null).")


if __name__ == "__main__":
    main()
