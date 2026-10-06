"""Turn normalized catalog rows into the text that gets embedded and the payload stored with it."""

import uuid

import pandas as pd

BLACK_SIDEWALL = "BSW:"


def point_id(sku: str) -> str:
    """Qdrant ids must be integers or UUIDs, so each SKU maps to a stable UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"tiredai:sku:{sku}"))


def products(df: pd.DataFrame) -> list[dict]:
    """Rows as plain Python values with nulls dropped, ready to be stored as a Qdrant payload."""
    return [{k: v for k, v in row.items() if not pd.isna(v)} for row in df.to_dict(orient="records")]


def document_text(product: dict) -> str:
    """Words a shopper would use to describe the tire.

    Numbers and codes (price, warranty, UTQG, ...) stay in the payload for filtering, apart from
    those already in the product name.
    """
    kind = " ".join(v for v in (product.get("season"), product.get("performance"), "tire") if v)
    if car_type := product.get("carType"):
        kind += f" for {car_type}"
    parts = [f"{product['name']}.", f"{kind}."]

    # Only stated when true: embeddings handle negation poorly.
    if product.get("runFlat"):
        parts.append("Run-flat.")

    # Black sidewall is the default for 92% of tires, so only other styles carry information.
    sidewall = product.get("sidewall")
    if sidewall and not sidewall.startswith(BLACK_SIDEWALL):
        parts.append(f"{sidewall.split(': ', 1)[-1]} sidewall.")

    return " ".join(parts)
