"""Agent benchmark cases (benchmarks/agent_cases.yaml): conversations, and what each turn must do."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tiredai.benchmarks.answers import meets, words
from tiredai.benchmarks.catalog import Catalog
from tiredai.benchmarks.experiments import Case
from tiredai.search import parse_size

DATASET = "tiredai-agent"
Intent = Literal["size_search", "product_inquiry", "education", "off_topic"]


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

    @model_validator(mode="after")
    def _consistent(self) -> "Expect":
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


def check_cases(cases: list[AgentCase], catalog: Catalog) -> list[str]:
    """What makes a case wrong or impossible to pass given the catalog."""
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
            terms = [words(t) for t in expect.product_terms]
            if expect.product_sku:
                if expect.product_sku not in catalog:
                    problems.append(f"{where}: SKU {expect.product_sku} is not in the catalog")
                    continue
                product = catalog[expect.product_sku]
                if not product.get("available"):
                    problems.append(f"{where}: {product['name']} is out of stock")
                if missing := [t for t in terms if f" {t} " not in f" {words(product['name'])} "]:
                    problems.append(f"{where}: product_terms {missing} are not in {product['name']!r}")
            if expect.not_in_catalog:
                if found := [p["name"] for p in catalog if all(f" {t} " in f" {words(p['name'])} " for t in terms)]:
                    problems.append(f"{where}: not_in_catalog, but {len(found)} products match {terms}, e.g. {found[0]!r}")
    return problems
