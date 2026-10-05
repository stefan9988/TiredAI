"""The processed catalog as the benchmarks read it, and how a product's relevance to a query is decided."""

import re
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from tiredai.documents import products

# The LT or P of a size can be written apart from it in the name: "Atturo Trail Blade H/T LT 265/70R17 ...".
SIZE_PREFIX = re.compile(r"\s+(LT|P)$")


def product_line(product: dict) -> str:
    """The name before the size: brand and tread line, e.g. 'Accelera Phi-R' for 'Accelera Phi-R 205/55R15 92V XL'.

    A name without its size in it (1% of them) is its own line.
    """
    name, size = product["name"], product.get("size")
    if not size or size not in name:
        return name
    return SIZE_PREFIX.sub("", name[: name.index(size)].strip()) or name


class Catalog:
    """Products by SKU, each with its derived 'line'."""

    def __init__(self, items: list[dict]):
        self.products = {p["sku"]: {**p, "line": product_line(p)} for p in items}

    @classmethod
    def load(cls, path: Path) -> "Catalog":
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; run scripts/preprocess_dataset.py first")
        return cls(products(pd.read_parquet(path)))

    def __getitem__(self, sku: str) -> dict:
        return self.products[sku]

    def __contains__(self, sku: str) -> bool:
        return sku in self.products

    def __iter__(self) -> Iterator[dict]:
        return iter(self.products.values())

    def __len__(self) -> int:
        return len(self.products)

    def fields(self) -> set[str]:
        return {field for product in self for field in product}


def gain(product: dict, relevant: dict[str, list]) -> float:
    """The share of `relevant`'s fields whose allowed values include the product's value (1.0: fully relevant)."""
    return sum(product.get(field) in allowed for field, allowed in relevant.items()) / len(relevant)
