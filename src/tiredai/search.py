"""Catalog search behind the agent's search_tires tool: exact filters plus hybrid (dense + BM25) ranking.

Hard constraints (size, price, season, ...) are Qdrant filters, so every result satisfies them;
the optional query text only ranks products within those filters.
"""

import difflib
import json
import re
from dataclasses import dataclass
from typing import Literal, Protocol

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient, models

from tiredai import tracing
from tiredai.embeddings import EmbeddingError
from tiredai.vectorstore import DENSE, DOCUMENT_KEY, SPARSE

# Speed ratings from slowest to fastest; the letters are not alphabetical.
# Z ("over 149 mph") guarantees more than V but not W's 168 mph, so it sits between them.
SPEED_ORDER = [
    "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
    "B", "C", "D", "E", "F", "G", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "H", "V", "Z", "W", "Y",
]  # fmt: skip

LIGHT_TRUCK = "Light Truck"
CANDIDATES = 50  # at least this many results are taken from each of the dense and BM25 searches before fusion
# 'model' is wrong in 21% of rows, so it is kept out of what the agent sees.
HIDDEN_FIELDS = {"model", DOCUMENT_KEY}
# Numbers sent to the agent as text with their unit: as bare numbers the model misread them (a
# tread depth of 8/32" became "8,000"). The stored values stay numbers.
AS_TEXT = {
    "treadDepth32nds": ("treadDepth", lambda n: f"{n}/32 in"),
    "mileageWarrantyMiles": ("mileageWarranty", lambda n: f"{n:,} miles"),
    "recommendations": ("recommendations", lambda n: f"{n}/5"),
}

Sort = Literal["relevance", "price_asc", "price_desc", "recommendations_desc"]
# Payload field and descending flag of each sort besides relevance.
SORT_KEYS = {"price_asc": ("price", False), "price_desc": ("price", True), "recommendations_desc": ("recommendations", True)}

NUMBER = re.compile(r"\d+(?:\.\d+)?")
PREFIX = re.compile(r"(LT|P)\s*(?=\d)")
ZR = re.compile(r"(?<=\d)\s*Z\s*R\s*(?=\d)")
# '205 55 16', '205/55/16', '205/55 R16' -> 205/55R16
LOOSE_METRIC = re.compile(r"(\d{3})[\s/]+(\d{2})[\s/]*R?\s*(\d{2}(?:\.\d)?)")


class QueryEncoder(Protocol):
    def encode_query(self, text: str) -> tuple[list[float], models.SparseVector]: ...


def size_key(size: str) -> str:
    """Sizes that differ only in number formatting share a key ('5.20-13' and '5.2-13')."""
    return NUMBER.sub(lambda m: f"{float(m.group()):g}", size.upper())


@dataclass(frozen=True)
class SizeQuery:
    key: str
    light_truck: bool


def parse_size(text: str) -> SizeQuery:
    """Normalize how a shopper wrote a size; an LT prefix marks a light-truck size."""
    size = text.strip().upper()
    light_truck = False
    if match := PREFIX.match(size):
        light_truck = match.group(1) == "LT"
        size = size[match.end() :]
    size = ZR.sub("R", size)
    if match := LOOSE_METRIC.fullmatch(size):
        size = f"{match[1]}/{match[2]}R{match[3]}"
    return SizeQuery(size_key(re.sub(r"\s+", "", size)), light_truck)


def product_view(payload: dict) -> dict:
    """A stored product as the agent sees it: hidden fields left out, AS_TEXT fields as text."""
    product = {}
    for key, value in payload.items():
        if key in HIDDEN_FIELDS:
            continue
        name, as_text = AS_TEXT.get(key, (key, None))
        product[name] = as_text(value) if as_text else value
    return product


def speed_rank(rating: str) -> int | None:
    """Rank of a catalog speed rating; dual ratings like 'A6/A8' count as the lower one."""
    parts = rating.upper().split("/")
    if not all(part in SPEED_ORDER for part in parts):
        return None
    return min(SPEED_ORDER.index(part) for part in parts)


class CatalogSearch:
    """Every search returns up to `max_results` products (AGENT_MAX_SEARCH_RESULTS)."""

    def __init__(self, client: QdrantClient, collection: str, encoder: QueryEncoder, *, max_results: int):
        self.client = client
        self.collection = collection
        self.encoder = encoder
        self.max_results = max_results

        values: dict[str, set[str]] = {key: set() for key in ("size", "brand", "season", "carType", "performance", "speedRating")}
        offset = None
        while True:
            points, offset = self.client.scroll(collection, limit=1000, offset=offset, with_payload=list(values))
            for point in points:
                for key, seen in values.items():
                    if key in point.payload:
                        seen.add(point.payload[key])
            if offset is None:
                break

        self.sizes: dict[str, list[str]] = {}
        for size in sorted(values["size"]):
            self.sizes.setdefault(size_key(size), []).append(size)
        self.brands = {b.lower(): b for b in values["brand"]}
        self.seasons = sorted(values["season"])
        self.car_types = sorted(values["carType"])
        self.performances = sorted(values["performance"])
        self.speed_ratings = sorted(values["speedRating"])

    def search(
        self,
        query: str | None = None,
        size: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        season: str | None = None,
        brand: str | None = None,
        car_type: str | None = None,
        performance: str | None = None,
        run_flat: bool | None = None,
        min_speed_rating: str | None = None,
        min_recommendations: int | None = None,
        sort: Sort = "relevance",
    ) -> dict:
        must: list[models.Condition] = []
        filters: dict = {}
        errors: list[str] = []

        def match(key: str, values: list):
            must.append(models.FieldCondition(key=key, match=models.MatchAny(any=values)))
            filters[key] = values[0] if len(values) == 1 else values

        if size:
            parsed = parse_size(size)
            if parsed.key not in self.sizes:
                errors.append(f"Size {size!r} is not in the catalog. Check the size, or ask the shopper for it.")
            else:
                match("size", self.sizes[parsed.key])
                if parsed.light_truck:
                    match("carType", [LIGHT_TRUCK])

        if (min_price is not None and min_price < 0) or (max_price is not None and max_price < 0):
            errors.append("Prices must not be negative.")
        elif min_price is not None and max_price is not None and min_price > max_price:
            errors.append(f"min_price {min_price} is above max_price {max_price}.")
        elif min_price is not None or max_price is not None:
            must.append(models.FieldCondition(key="price", range=models.Range(gte=min_price, lte=max_price)))
            filters["price"] = {"min": min_price, "max": max_price}

        for key, value, allowed in (
            ("season", season, self.seasons),
            ("carType", car_type, self.car_types),
            ("performance", performance, self.performances),
        ):
            if value:
                resolved = next((a for a in allowed if a.lower() == value.strip().lower()), None)
                if resolved is None:
                    errors.append(f"Unknown {key} {value!r}. Use one of: {', '.join(allowed)}.")
                else:
                    match(key, [resolved])

        if brand:
            resolved = self.brands.get(brand.strip().lower())
            if resolved is None:
                close = difflib.get_close_matches(brand.lower(), self.brands, n=3, cutoff=0.6)
                hint = f" Did you mean: {', '.join(self.brands[c] for c in close)}?" if close else ""
                errors.append(f"Brand {brand!r} is not in the catalog.{hint}")
            else:
                match("brand", [resolved])

        if run_flat is not None:
            must.append(models.FieldCondition(key="runFlat", match=models.MatchValue(value=run_flat)))
            filters["runFlat"] = run_flat

        if min_speed_rating:
            minimum = min_speed_rating.strip().upper()
            if minimum not in SPEED_ORDER:
                errors.append(f"Unknown speed rating {min_speed_rating!r}. From slowest to fastest: {', '.join(SPEED_ORDER)}.")
            else:
                ratings = [r for r in self.speed_ratings if (rank := speed_rank(r)) is not None and rank >= SPEED_ORDER.index(minimum)]
                match("speedRating", ratings)  # an empty list matches nothing

        if min_recommendations is not None:
            if not 1 <= min_recommendations <= 5:
                errors.append(f"min_recommendations must be 1 to 5, not {min_recommendations}.")
            else:
                must.append(models.FieldCondition(key="recommendations", range=models.Range(gte=min_recommendations)))
                filters["recommendations"] = {"min": min_recommendations}

        if errors:
            return {"error": " ".join(errors)}

        query_filter = models.Filter(must=must) if must else None
        query = query.strip() if query and query.strip() else None
        with tracing.client().start_as_current_observation(
            as_type="retriever",
            name="retrieve-products",
            input={"query": query, "filters": filters, "sort": sort},
            metadata={"collection": self.collection, "max_results": self.max_results},
        ) as retrieval:
            try:
                total, points, order = self._retrieve(query, query_filter, sort)
            except EmbeddingError as exc:
                retrieval.update(level="ERROR", status_message=str(exc))
                return {"error": f"Search is temporarily unavailable: {exc}"}
            # The ranking as retrieved; the tool's output holds every field the model got.
            ranked = [{"sku": p.payload["sku"], "name": p.payload["name"], "score": getattr(p, "score", None)} for p in points]
            retrieval.update(output={"total_matching": total, "order": order, "products": ranked})

        result = {
            "total_matching": total,
            "returned": len(points),
            "order": order,
            "filters": filters,
            "products": [product_view(p.payload) for p in points],
        }
        if total == 0:
            result["note"] = "No products in the catalog match these filters."
        return result

    def _retrieve(self, query: str | None, query_filter: models.Filter | None, sort: Sort) -> tuple[int, list, str]:
        """(products matching the filter, the points to return, their order)."""
        limit = self.max_results
        candidates = max(CANDIDATES, limit)
        total = self.client.count(self.collection, count_filter=query_filter, exact=True).count

        if query:
            dense, sparse = self.encoder.encode_query(query)
            points = self.client.query_points(
                self.collection,
                prefetch=[
                    models.Prefetch(query=dense, using=DENSE, limit=candidates),
                    models.Prefetch(query=sparse, using=SPARSE, limit=candidates),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=query_filter,
                limit=limit if sort == "relevance" else candidates,
                with_payload=True,
            ).points
            if sort != "relevance":
                # Ordered among the most relevant products for the query; ties keep their relevance order.
                key, descending = SORT_KEYS[sort]
                points = sorted(points, key=lambda p: p.payload[key], reverse=descending)[:limit]
            order = sort if sort == "relevance" else f"{sort} among the {candidates} most relevant"
        else:
            # Without a query there is nothing to rank by, so relevance means cheapest first.
            order = "price_asc" if sort == "relevance" else sort
            key, descending = SORT_KEYS[order]
            points, _ = self.client.scroll(
                self.collection,
                scroll_filter=query_filter,
                order_by=models.OrderBy(key=key, direction=models.Direction.DESC if descending else models.Direction.ASC),
                limit=limit,
                with_payload=True,
            )
        return total, points, order


class SearchArgs(BaseModel):
    # An unknown argument (e.g. speed_rating instead of min_speed_rating) is an error the model sees and
    # can fix, instead of a filter that silently isn't applied.
    model_config = ConfigDict(extra="forbid")

    query: str | None = Field(
        None,
        description="Words describing the tire, or the product name the shopper asked about. Ranks results by relevance.",
    )
    size: str | None = Field(None, description="Tire size as the shopper gave it, e.g. '205/55R16' or 'LT265/70R17'.")
    min_price: float | None = Field(None, description="Lowest price per tire.")
    max_price: float | None = Field(None, description="Highest price per tire.")
    season: str | None = None
    brand: str | None = None
    car_type: str | None = None
    performance: str | None = None
    run_flat: bool | None = Field(None, description="true for only run-flat tires, false to exclude them.")
    min_speed_rating: str | None = Field(None, description="Lowest acceptable speed rating, e.g. 'H'.")
    min_recommendations: int | None = Field(None, description="Lowest acceptable store recommendation level, 1 to 5.")
    sort: Sort = Field(
        "relevance",
        description="'relevance' (needs a query), 'price_asc', 'price_desc' or 'recommendations_desc' (most recommended first).",
    )


def make_search_tool(catalog: CatalogSearch) -> BaseTool:
    description = f"""Search the tire catalog. Use it for every question about products, prices or specifications, and answer only from what it returns.

Every filter is a hard constraint: all returned products satisfy all of them. The size is matched exactly after normalizing its formatting, and an LT size returns only light-truck tires. Without a query, results are sorted by price (cheapest first) unless another sort is given.

Allowed values (case-insensitive):
- season: {', '.join(catalog.seasons)}
- car_type: {', '.join(catalog.car_types)}
- performance: {', '.join(catalog.performances)}
- min_speed_rating, slowest to fastest: {', '.join(SPEED_ORDER)}

Returns JSON with total_matching (products matching all filters) and up to {catalog.max_results} products. If total_matching is 0, nothing in the catalog matches.

Product fields include available (false: out of stock, never recommend it) and recommendations (the store's recommendation level, from 1/5 to 5/5)."""

    def search_tires(**kwargs) -> str:
        return json.dumps(catalog.search(**kwargs), ensure_ascii=False)

    return StructuredTool.from_function(
        func=search_tires, name="search_tires", description=description, args_schema=SearchArgs
    )


def catalog_tools(
    client: QdrantClient | None, collection: str, encoder_factory, *, max_results: int
) -> list[BaseTool]:
    """The search tool when the index exists, otherwise no tools; the encoder is only built if needed."""
    if client is None or not client.collection_exists(collection):
        return []
    return [make_search_tool(CatalogSearch(client, collection, encoder_factory(), max_results=max_results))]


def _money(value) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"${amount:,.0f}" if amount.is_integer() else f"${amount:,.2f}"


SORT_LABELS = {"price_asc": "cheapest first", "price_desc": "most expensive first", "recommendations_desc": "most recommended first"}


def describe_search(args: dict) -> str:
    """Status line for a search_tires call, e.g. 'Searching the catalog: 205/60R15 · All Season · up to $60'."""
    parts = [str(args["size"])] if args.get("size") else []
    if args.get("query"):
        parts.append(f'"{args["query"]}"')
    parts += [str(args[key]) for key in ("season", "brand", "car_type", "performance") if args.get(key)]
    if args.get("run_flat") in (True, "true"):
        parts.append("run-flat")
    elif args.get("run_flat") in (False, "false"):
        parts.append("no run-flat")
    if args.get("min_speed_rating"):
        parts.append(f"speed rating {args['min_speed_rating']} or higher")
    if args.get("min_recommendations") is not None:
        parts.append(f"recommended {args['min_recommendations']}/5 or higher")
    low, high = args.get("min_price"), args.get("max_price")
    if low is not None and high is not None:
        parts.append(f"{_money(low)}–{_money(high)}")
    elif high is not None:
        parts.append(f"up to {_money(high)}")
    elif low is not None:
        parts.append(f"from {_money(low)}")
    if args.get("sort") in SORT_LABELS:
        parts.append(SORT_LABELS[args["sort"]])
    return "Searching the catalog" + (f": {' · '.join(parts)}" if parts else "")


def describe_results(content: str) -> str:
    """Status line for what search_tires returned."""
    try:
        result = json.loads(content)
    except (TypeError, ValueError):
        return "The search failed"  # e.g. invalid arguments; the agent gets the error text
    if "error" in result:
        return f"Search problem: {result['error']}"
    total = result.get("total_matching", 0)
    return "No tires match" if total == 0 else f"Found {total:,} tire{'' if total == 1 else 's'}"
