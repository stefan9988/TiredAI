"""Vehicle tire size lookup: the find_vehicle_tire_sizes tool, for shoppers who don't know their tire size.

The tool searches a few tire-size sites (VEHICLE_LOOKUP_SITES) for the vehicle's year, make and model
with OpenRouter's web search (the web plugin, Exa engine) and returns the pages found: site, url, title
and the excerpt the search engine read. The chat model reads the sizes per trim from the excerpts
itself; nothing here interprets them. The trim isn't searched for: the pages list every trim.

OpenRouter only searches as part of a chat completion, so a lookup is a minimal completion (a few
output tokens, VEHICLE_LOOKUP_MODEL or the chat model) whose text is thrown away; the pages come from
its url_citation annotations. A search costs about $0.007. Lookups that found pages are cached in
SQLite (VEHICLE_LOOKUP_CACHE_PATH) without expiry, since factory sizes don't change. The agent
benchmark reads captured pages instead (FrozenPages, benchmarks/vehicle_pages.yaml).
"""

import datetime
import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

import httpx
import yaml
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from tiredai import tracing
from tiredai.config import Settings

TOOL_NAME = "find_vehicle_tire_sizes"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
ENGINE = "exa"
CARRIER_MAX_TOKENS = 16  # the completion's text is thrown away
MAX_EXCERPT_CHARS = 4000
FIRST_YEAR = 1950


class VehicleLookupError(Exception):
    pass


@dataclass(frozen=True)
class Vehicle:
    year: int
    make: str
    model: str

    def __str__(self) -> str:
        return f"{self.year} {self.make} {self.model}"

    def query(self) -> str:
        return f"{self} tire size"


def _model_words(vehicle: Vehicle) -> set[str]:
    return set(vehicle.model.lower().replace("-", " ").split())


def same_vehicle(a: Vehicle, b: Vehicle) -> bool:
    """Same year and make, and one model's words contain the other's ('3 Series 330i' and '330i')."""
    if a.year != b.year or a.make.lower() != b.make.lower():
        return False
    words_a, words_b = _model_words(a), _model_words(b)
    return words_a <= words_b or words_b <= words_a


def clean_excerpt(text: str) -> str:
    """The excerpt without blank lines and the '...' lines the search engine puts between passages."""
    lines = [line.rstrip() for line in text.splitlines() if line.strip() and line.strip() != "..."]
    excerpt = "\n".join(lines)
    return excerpt if len(excerpt) <= MAX_EXCERPT_CHARS else excerpt[:MAX_EXCERPT_CHARS].rstrip() + " …"


def site_of(url: str, sites: tuple[str, ...]) -> str | None:
    """The site of `sites` the url belongs to ('vehicle.firestonecompleteautocare.com' is firestonecompleteautocare.com)."""
    host = (urlsplit(url).hostname or "").lower()
    return next((site for site in sites if host == site or host.endswith("." + site)), None)


@dataclass(frozen=True)
class Found:
    pages: list[dict]  # {"site", "url", "title", "excerpt"}
    cached: bool = False
    cost: float | None = None  # what the search cost in USD, when it was made now


class PageSource(Protocol):
    sites: tuple[str, ...]

    def pages(self, vehicle: Vehicle) -> Found: ...


class PageCache:
    """Pages of earlier lookups that found some, on disk. Safe to use from several threads: the agent runs
    tools in worker threads."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS pages (key TEXT PRIMARY KEY, query TEXT NOT NULL, pages TEXT NOT NULL, saved_at TEXT NOT NULL)"
        )

    @staticmethod
    def key(query: str, sites: tuple[str, ...], max_results: int) -> str:
        # Other sites or result counts find other pages, so they are another lookup.
        spec = [ENGINE, sorted(sites), max_results, " ".join(query.lower().split())]
        return hashlib.sha256(json.dumps(spec).encode()).hexdigest()

    def get(self, key: str) -> list[dict] | None:
        with self._lock:
            row = self._db.execute("SELECT pages FROM pages WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, query: str, pages: list[dict]) -> None:
        saved_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO pages VALUES (?, ?, ?, ?)", (key, query, json.dumps(pages), saved_at))

    def close(self) -> None:
        self._db.close()


class SiteSearch:
    """OpenRouter's web search limited to `sites`. No retries: a shopper is waiting, and the agent can ask again."""

    def __init__(self, api_key: str, model: str, sites: tuple[str, ...], *, max_results: int = 5,
                 client: httpx.Client | None = None, cache: PageCache | None = None):  # fmt: skip
        self.model = model
        self.sites = sites
        self.max_results = max_results
        self.client = client or httpx.Client(timeout=20)
        self.cache = cache
        self.headers = {"Authorization": f"Bearer {api_key}", "X-Title": "TiredAI"}

    def pages(self, vehicle: Vehicle) -> Found:
        query = vehicle.query()
        key = PageCache.key(query, self.sites, self.max_results)
        if self.cache and (cached := self.cache.get(key)) is not None:
            return Found(cached, cached=True)
        body = {
            "model": self.model,
            "max_tokens": CARRIER_MAX_TOKENS,
            "plugins": [{"id": "web", "engine": ENGINE, "max_results": self.max_results, "include_domains": list(self.sites)}],
            "messages": [{"role": "user", "content": query}],
        }
        try:
            response = self.client.post(CHAT_URL, headers=self.headers, json=body)
        except httpx.HTTPError as exc:
            raise VehicleLookupError(f"the web search failed ({type(exc).__name__}: {exc})") from exc
        if response.status_code != 200:
            raise VehicleLookupError(f"OpenRouter returned {response.status_code}: {response.text[:300]}")
        data = response.json()
        if "error" in data:
            raise VehicleLookupError(f"OpenRouter returned an error: {json.dumps(data['error'])[:300]}")
        pages = self._pages(data)
        if pages and self.cache:
            self.cache.put(key, query, pages)
        return Found(pages, cost=(data.get("usage") or {}).get("cost"))

    def _pages(self, data: dict) -> list[dict]:
        """The pages of the completion's url_citation annotations: on the sites, each url once, with an excerpt."""
        message = ((data.get("choices") or [{}])[0]).get("message") or {}
        pages, urls = [], set()
        for annotation in message.get("annotations") or []:
            citation = annotation.get("url_citation") or {}
            url = citation.get("url") or ""
            site, excerpt = site_of(url, self.sites), clean_excerpt(citation.get("content") or "")
            if site and excerpt and url not in urls:
                urls.add(url)
                pages.append({"site": site, "url": url, "title": citation.get("title") or "", "excerpt": excerpt})
        return pages


class _Dumper(yaml.SafeDumper):
    pass


# Excerpts are written as blocks, so the captured file reads like the pages.
_Dumper.add_representer(str, lambda dumper, text: dumper.represent_scalar("tag:yaml.org,2002:str", text,
                                                                          style="|" if "\n" in text else None))  # fmt: skip


class FrozenPages:
    """Pages captured for the agent benchmark's vehicles (benchmarks/vehicle_pages.yaml), so it doesn't search.

    A lookup gets the pages of the captured vehicle it matches (same_vehicle); any other vehicle is an error.
    """

    def __init__(self, sites: tuple[str, ...], max_results: int | None, captured: list[dict]):
        self.sites = sites
        self.max_results = max_results
        self.rows = captured  # {"vehicle": {"year", "make", "model"}, "captured_at", "pages"}
        self.captured = [(Vehicle(**row["vehicle"]), row["pages"]) for row in captured]

    @classmethod
    def load(cls, path: Path) -> "FrozenPages":
        data = (yaml.safe_load(path.read_text()) if path.exists() else None) or {}
        return cls(tuple(data.get("sites") or ()), data.get("max_results"), data.get("vehicles") or [])

    def write(self, path: Path, header: str) -> None:
        data = {"sites": list(self.sites), "max_results": self.max_results, "vehicles": self.rows}
        path.write_text(f"# {header}\n" + yaml.dump(data, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=120))

    def row(self, vehicle: Vehicle) -> dict | None:
        """What was captured for exactly this vehicle (ignoring case)."""
        key = str(vehicle).lower()
        return next((row for row in self.rows if str(Vehicle(**row["vehicle"])).lower() == key), None)

    def pages(self, vehicle: Vehicle) -> Found:
        for captured, pages in self.captured:
            if same_vehicle(captured, vehicle):
                return Found(pages, cached=True)
        raise VehicleLookupError(f"no pages were captured for the {vehicle} (run scripts/capture_vehicle_pages.py)")


def _name(text: str) -> str:
    return " ".join(text.split())


class VehicleLookup:
    def __init__(self, source: PageSource):
        self.source = source

    @property
    def sites(self) -> tuple[str, ...]:
        return self.source.sites

    def find(self, year: int, make: str, model: str) -> dict:
        """{"vehicle", "sites", "pages": [{"site", "url", "title", "excerpt"}], "note" when none}, or {"error"}."""
        last_year = datetime.date.today().year + 1
        if not FIRST_YEAR <= year <= last_year:
            return {"error": f"year must be between {FIRST_YEAR} and {last_year}, not {year}. Ask the shopper for the model year."}
        if not _name(make) or not _name(model):
            return {"error": "make and model must not be empty. Ask the shopper for them."}
        vehicle = Vehicle(year, _name(make), _name(model))
        with tracing.client().start_as_current_observation(
            as_type="retriever",
            name="look-up-vehicle-sizes",
            input={"vehicle": str(vehicle), "query": vehicle.query(), "sites": list(self.sites)},
        ) as observation:
            try:
                found = self.source.pages(vehicle)
            except VehicleLookupError as exc:
                observation.update(level="ERROR", status_message=str(exc))
                return {"error": f"The tire size lookup is unavailable: {exc}"}
            observation.update(output={"pages": [p["url"] for p in found.pages]}, metadata={"cached": found.cached, "cost": found.cost})
        result = {"vehicle": str(vehicle), "sites": list(self.sites), "pages": found.pages}
        if not found.pages:
            result["note"] = "No page about this vehicle was found on these sites."
        return result


class VehicleArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    year: int = Field(description="The model year, e.g. 2016.")
    make: str = Field(description="The manufacturer, e.g. 'Ford'.")
    model: str = Field(description="The model, e.g. 'Focus'.")


def make_vehicle_tool(lookup: VehicleLookup) -> BaseTool:
    description = f"""Look up the factory tire sizes of a vehicle on tire-size websites ({', '.join(lookup.sites)}). Use it only when the shopper doesn't know their tire size and tells you their vehicle. It needs the year, make and model: ask the shopper for any that is missing, never guess them. The trim isn't needed: the pages list every trim.

Returns JSON with the pages found, each with its site, url, title and excerpt. Read the sizes from the excerpts yourself: they list the sizes per trim and option, sometimes different front and rear sizes, and sometimes the factory speed rating. Use only pages about this exact year, make and model. If pages is empty, nothing was found."""

    def find_vehicle_tire_sizes(**kwargs) -> str:
        return json.dumps(lookup.find(**kwargs), ensure_ascii=False)

    return StructuredTool.from_function(
        func=find_vehicle_tire_sizes, name=TOOL_NAME, description=description, args_schema=VehicleArgs
    )


def vehicle_tools(settings: Settings, *, client: httpx.Client | None = None) -> list[BaseTool]:
    """The lookup tool, searching live with a cache; no tools without an OpenRouter key."""
    if not settings.openrouter_api_key:
        return []
    config = settings.vehicle_lookup
    search = SiteSearch(
        settings.openrouter_api_key,
        config.model or settings.llm.model,
        config.sites,
        max_results=config.max_results,
        client=client or httpx.Client(timeout=config.timeout_seconds),
        cache=PageCache(config.cache_path),
    )
    return [make_vehicle_tool(VehicleLookup(search))]


def describe_lookup(args: dict) -> str:
    """Status line for a lookup, e.g. 'Looking up tire sizes: 2016 Ford Focus'."""
    vehicle = " ".join(str(args[key]) for key in ("year", "make", "model") if args.get(key))
    return "Looking up tire sizes" + (f": {vehicle}" if vehicle else "")


def describe_lookup_results(content: str) -> str:
    try:
        result = json.loads(content)
    except (TypeError, ValueError):
        return "The lookup failed"  # e.g. invalid arguments; the agent gets the error text
    if "error" in result:
        return f"Lookup problem: {result['error']}"
    count = len(result.get("pages") or [])
    return "No tire size pages found" if count == 0 else f"Found {count} tire size page{'' if count == 1 else 's'}"
