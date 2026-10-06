"""Agent benchmark cases (benchmarks/agent_cases.yaml): conversations, and what each turn must do."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tiredai.benchmarks.answers import meets, tire_sizes, words
from tiredai.benchmarks.catalog import Catalog
from tiredai.benchmarks.experiments import Case
from tiredai.search import parse_size
from tiredai.vehicles import FrozenPages, Vehicle, VehicleLookupError

DATASET = "tiredai-agent"
Intent = Literal["size_search", "product_inquiry", "education", "off_topic", "vehicle_lookup"]


class Constraints(BaseModel):
    """A turn's hard constraints, named like the search_tires arguments: searches must apply them as
    filters, and every product the answer recommends must meet them (and be in stock)."""

    model_config = ConfigDict(extra="forbid")

    size: str | None = None
    season: str | None = None
    brand: str | None = None
    car_type: str | None = None
    performance: str | None = None
    run_flat: bool | None = None
    min_price: float | None = None
    max_price: float | None = None

    def given(self) -> dict:
        return self.model_dump(exclude_none=True)


class VehicleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    year: int
    make: str
    model: str

    def vehicle(self) -> Vehicle:
        return Vehicle(self.year, self.make, self.model)


class Expect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Intent
    # Size search: asking about preferences before searching is fine.
    may_ask: bool = False
    # The size is incomplete: the agent must ask for it and not search with a guessed size.
    asks_for_size: bool = False
    constraints: Constraints = Field(default_factory=Constraints)
    # Product inquiry: the product asked about, or that it isn't in the catalog.
    product_sku: str | None = None
    not_in_catalog: bool = False
    # Words the search query of a product lookup must contain, e.g. ["eagle f1", "asymmetric"].
    product_terms: list[str] = Field(default_factory=list)
    # Every recommended product costs less than the cheapest one recommended in the previous turn.
    cheaper_than_previous: bool = False
    # Each group: at least one of its phrases appears in the answer (case-insensitive).
    must_mention: list[list[str]] = Field(default_factory=list)
    # None of these phrases appear in the answer.
    must_not_mention: list[str] = Field(default_factory=list)
    # Vehicle lookup: the vehicle the agent must have looked up, in this turn or an earlier one, and what then:
    vehicle: VehicleSpec | None = None
    # it has one fitment for the shopper's trim: search these sizes (one, or the front and the rear size);
    sizes: list[str] = Field(default_factory=list)
    # it has several: list them and ask which one, no search;
    asks_which_size: bool = False
    # no page is about it: no search with a guessed size;
    vehicle_not_found: bool = False
    # or the year, make or model is missing: ask for it, no lookup of a guessed vehicle.
    asks_for_vehicle: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> "Expect":
        outcomes = [bool(self.sizes), self.asks_which_size, self.vehicle_not_found, self.asks_for_vehicle]
        if self.intent == "vehicle_lookup":
            if sum(outcomes) != 1:
                raise ValueError("a vehicle lookup needs one of sizes, asks_which_size, vehicle_not_found or asks_for_vehicle")
            if not self.asks_for_vehicle and self.vehicle is None:
                raise ValueError("a vehicle lookup needs the vehicle")
        elif any(outcomes) or self.vehicle:
            raise ValueError("vehicle, sizes, asks_which_size, vehicle_not_found and asks_for_vehicle are for vehicle lookups")
        if self.intent == "product_inquiry":
            if not self.product_terms:
                raise ValueError("a product inquiry needs product_terms")
            if bool(self.product_sku) == self.not_in_catalog:
                raise ValueError("a product inquiry needs either product_sku or not_in_catalog: true")
        elif self.product_sku or self.not_in_catalog:
            raise ValueError("product_sku and not_in_catalog are for product inquiries")
        if (self.may_ask or self.asks_for_size) and self.intent != "size_search":
            raise ValueError("may_ask and asks_for_size are for size searches")
        return self


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user: str
    expect: Expect


class AgentCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    tags: list[str] = Field(default_factory=list)
    turns: list[Turn] = Field(min_length=1)

    def case(self) -> Case:
        return Case(
            id=self.id,
            input={"turns": [t.user for t in self.turns]},
            expected_output={"turns": [t.expect.model_dump(exclude_defaults=True) for t in self.turns]},
            metadata={"case": self.id, "description": self.description, "intents": [t.expect.intent for t in self.turns],
                      "tags": self.tags},  # fmt: skip
        )


def load_cases(path: Path) -> list[AgentCase]:
    cases = [AgentCase(**row) for row in yaml.safe_load(path.read_text()) or []]
    ids = [c.id for c in cases]
    if duplicates := sorted({i for i in ids if ids.count(i) > 1}):
        raise ValueError(f"Duplicate case ids: {', '.join(duplicates)}")
    return cases


def vehicles(cases: list[AgentCase]) -> list[Vehicle]:
    """The vehicles the cases look up, each once, in the order of the cases."""
    found = {}
    for case in cases:
        for turn in case.turns:
            if turn.expect.vehicle:
                vehicle = turn.expect.vehicle.vehicle()
                found.setdefault(str(vehicle).lower(), vehicle)
    return list(found.values())


def check_vehicle_turn(where: str, expect: Expect, catalog: Catalog, pages: FrozenPages) -> list[str]:
    """A vehicle lookup's vehicle has captured pages, and its sizes are on them and in stock in the catalog."""
    if expect.vehicle is None:
        return []
    try:
        found = pages.pages(expect.vehicle.vehicle()).pages
    except VehicleLookupError as exc:
        return [f"{where}: {exc}"]
    on_pages = set().union(*(tire_sizes(p["excerpt"]) for p in found))
    problems = []
    for size in expect.sizes:
        key = parse_size(size).key
        if key not in on_pages:
            problems.append(f"{where}: size {size} is not on the pages captured for the {expect.vehicle.vehicle()}")
        if not any(p.get("available") and p.get("size") and parse_size(p["size"]).key == key for p in catalog):
            problems.append(f"{where}: no product in stock has size {size}")
    return problems


def check_cases(cases: list[AgentCase], catalog: Catalog, vehicle_pages: FrozenPages | None = None) -> list[str]:
    """What makes a case wrong or impossible to pass given the catalog (and the pages captured for vehicle lookups)."""
    problems = []
    sizes = {parse_size(p["size"]).key for p in catalog if p.get("size")}
    values = {field: {str(p.get(field)).lower() for p in catalog} for field in ("season", "brand", "carType", "performance")}
    for case in cases:
        for number, turn in enumerate(case.turns, start=1):
            where, expect = f"{case.id} turn {number}", turn.expect
            constraints = expect.constraints.given()
            if "size" in constraints and parse_size(constraints["size"]).key not in sizes:
                problems.append(f"{where}: size {constraints['size']} is not in the catalog")
            for key, field in (("season", "season"), ("brand", "brand"), ("car_type", "carType"), ("performance", "performance")):
                if key in constraints and str(constraints[key]).lower() not in values[field]:
                    problems.append(f"{where}: {key} {constraints[key]!r} is not in the catalog")
            if constraints and not any(p.get("available") and not meets(p, constraints) for p in catalog):
                problems.append(f"{where}: no product in stock meets {constraints}")
            if vehicle_pages is not None:
                problems += check_vehicle_turn(where, expect, catalog, vehicle_pages)
            terms = [words(t) for t in expect.product_terms]
            if expect.product_sku:
                if expect.product_sku not in catalog:
                    problems.append(f"{where}: SKU {expect.product_sku} is not in the catalog")
                    continue
                product = catalog[expect.product_sku]
                if not product.get("available"):
                    problems.append(f"{where}: {product['name']} is out of stock")
                if missing := [t for t in terms if t not in words(product["name"])]:
                    problems.append(f"{where}: product_terms {missing} are not in {product['name']!r}")
            if expect.not_in_catalog:
                if found := [p["name"] for p in catalog if all(t in words(p["name"]) for t in terms)]:
                    problems.append(f"{where}: not_in_catalog, but {len(found)} products match {terms}, e.g. {found[0]!r}")
    return problems
