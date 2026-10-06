import asyncio
import dataclasses
import json

import httpx
import pytest
from conftest import ToolCallingModel, ToolRecordingModel, tool_call
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from pydantic import ValidationError

from tiredai.agent import WEB_SEARCH_OFF, astream_turn, build_agent, transcript
from tiredai.config import DEFAULT_VEHICLE_SITES, AgentSettings, Settings, VehicleLookupSettings
from tiredai.vehicles import (
    CHAT_URL,
    TOOL_NAME,
    FrozenPages,
    PageCache,
    SiteSearch,
    Vehicle,
    VehicleLookup,
    VehicleLookupError,
    clean_excerpt,
    describe_lookup,
    describe_lookup_results,
    make_vehicle_tool,
    same_vehicle,
    vehicle_tools,
)

SITES = ("tiresize.com", "firestonecompleteautocare.com")
FOCUS = Vehicle(2016, "Ford", "Focus")


def citation(url: str, content: str, title: str = "2016 Ford Focus Tire Sizes") -> dict:
    return {"type": "url_citation", "url_citation": {"url": url, "title": title, "content": content, "start_index": 0, "end_index": 0}}


def completion(*annotations: dict, cost: float = 0.0071) -> dict:
    """A chat completion with the web plugin's results as annotations, like OpenRouter returns it."""
    message = {"role": "assistant", "content": "The 2016", "annotations": list(annotations)}
    return {"choices": [{"message": message}], "usage": {"prompt_tokens": 2100, "completion_tokens": 16, "cost": cost}}


FOCUS_PAGE = citation("https://tiresize.com/tires/Ford/Focus/2016/", "2016 Ford Focus Sedan S\n...\n195/65R15\n...\n\n2016 Ford Focus RS\n...\n235/35R19")


class FakeCompletions:
    """OpenRouter's chat completions: replies in order, the last one again after that (a dict is a 200 body,
    an int an error status, an exception is raised), recording each request."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, int):
            return httpx.Response(reply, text="provider says no")
        return httpx.Response(200, json=reply)


def site_search(api: FakeCompletions, cache: PageCache | None = None, sites=SITES, max_results=5) -> SiteSearch:
    client = httpx.Client(transport=httpx.MockTransport(api))
    return SiteSearch("test-key", "test/carrier", sites, max_results=max_results, client=client, cache=cache)


# --- The web search ----------------------------------------------------------------------------------


def test_a_lookup_searches_the_sites_for_the_vehicle_in_a_minimal_completion():
    api = FakeCompletions(completion(FOCUS_PAGE))

    found = site_search(api).pages(FOCUS)

    [request] = api.requests
    assert str(request.url) == CHAT_URL and request.headers["Authorization"] == "Bearer test-key"
    assert api.bodies() == [{
        "model": "test/carrier",
        "max_tokens": 16,
        "plugins": [{"id": "web", "engine": "exa", "max_results": 5, "include_domains": list(SITES)}],
        "messages": [{"role": "user", "content": "2016 Ford Focus tire size"}],
    }]  # fmt: skip
    assert found.pages == [{"site": "tiresize.com", "url": "https://tiresize.com/tires/Ford/Focus/2016/", "title": "2016 Ford Focus Tire Sizes",
                            "excerpt": "2016 Ford Focus Sedan S\n195/65R15\n2016 Ford Focus RS\n235/35R19"}]  # fmt: skip
    assert found.cost == 0.0071 and not found.cached


def test_pages_off_the_sites_repeated_or_without_text_are_left_out():
    api = FakeCompletions(completion(
        FOCUS_PAGE,
        citation("https://vehicle.firestonecompleteautocare.com/ford/focus/2016/tires/", "| S | 195/65R15 | H |"),
        citation("https://tiresize.com/tires/Ford/Focus/2016/", "the same page again"),
        citation("https://www.reddit.com/r/FordFocus/", "my focus runs 205/55R16"),
        citation("https://evil-tiresize.com/x", "a lookalike domain"),
        citation("https://tiresize.com/empty", "\n...\n"),
    ))  # fmt: skip

    pages = site_search(api).pages(FOCUS).pages

    assert [(p["site"], p["url"]) for p in pages] == [
        ("tiresize.com", "https://tiresize.com/tires/Ford/Focus/2016/"),
        ("firestonecompleteautocare.com", "https://vehicle.firestonecompleteautocare.com/ford/focus/2016/tires/"),
    ]


def test_long_excerpts_are_cut():
    excerpt = clean_excerpt("x" * 5000)

    assert len(excerpt) == 4002 and excerpt.endswith(" …")


@pytest.mark.parametrize(
    "reply, message",
    [
        (402, "OpenRouter returned 402: provider says no"),
        ({"error": {"code": 400, "message": "bad plugin"}}, "OpenRouter returned an error"),
        (httpx.ReadTimeout("timed out"), "the web search failed (ReadTimeout: timed out)"),
    ],
)
def test_a_failed_search_raises_a_lookup_error(reply, message):
    with pytest.raises(VehicleLookupError, match=message.replace("(", r"\(").replace(")", r"\)")):
        site_search(FakeCompletions(reply)).pages(FOCUS)


# --- The cache ---------------------------------------------------------------------------------------


def test_found_pages_are_cached_so_the_vehicle_is_searched_once(tmp_path):
    api = FakeCompletions(completion(FOCUS_PAGE))
    first = site_search(api, PageCache(tmp_path / "pages.sqlite")).pages(FOCUS)

    # Another process with the same file, and the vehicle written another way.
    again = site_search(api, PageCache(tmp_path / "pages.sqlite")).pages(Vehicle(2016, "ford", "FOCUS"))

    assert len(api.requests) == 1
    assert again.pages == first.pages and again.cached and again.cost is None


def test_lookups_that_found_nothing_are_not_cached(tmp_path):
    api = FakeCompletions(completion(), completion(FOCUS_PAGE))
    search = site_search(api, PageCache(tmp_path / "pages.sqlite"))

    assert search.pages(FOCUS).pages == []
    assert len(search.pages(FOCUS).pages) == 1
    assert len(api.requests) == 2


def test_other_sites_or_result_counts_are_another_lookup():
    key = PageCache.key("2016 Ford Focus tire size", SITES, 5)

    assert key == PageCache.key("2016  ford focus TIRE size", tuple(reversed(SITES)), 5)
    assert key != PageCache.key("2016 Ford Focus tire size", SITES[:1], 5)
    assert key != PageCache.key("2016 Ford Focus tire size", SITES, 3)


# --- The tool ----------------------------------------------------------------------------------------


def lookup(api: FakeCompletions) -> VehicleLookup:
    return VehicleLookup(site_search(api))


def test_the_tool_returns_the_vehicle_the_sites_and_the_pages():
    tool = make_vehicle_tool(lookup(FakeCompletions(completion(FOCUS_PAGE))))

    result = json.loads(tool.invoke({"year": 2016, "make": " Ford ", "model": "Focus"}))

    assert result["vehicle"] == "2016 Ford Focus" and result["sites"] == list(SITES)
    assert [p["url"] for p in result["pages"]] == ["https://tiresize.com/tires/Ford/Focus/2016/"]
    assert "note" not in result


def test_a_lookup_that_found_nothing_says_so():
    result = lookup(FakeCompletions(completion())).find(2021, "Ford", "Focus")

    assert result["pages"] == [] and result["note"] == "No page about this vehicle was found on these sites."


def test_a_failed_search_is_an_error_result_for_the_model():
    result = lookup(FakeCompletions(503)).find(2016, "Ford", "Focus")

    assert result == {"error": "The tire size lookup is unavailable: OpenRouter returned 503: provider says no"}


@pytest.mark.parametrize(
    "args, message",
    [
        ({"year": 16, "make": "Ford", "model": "Focus"}, "year must be between 1950 and"),
        ({"year": 2016, "make": " ", "model": "Focus"}, "make and model must not be empty"),
    ],
)
def test_impossible_vehicles_are_rejected_without_searching(args, message):
    api = FakeCompletions(completion(FOCUS_PAGE))

    assert message in lookup(api).find(**args)["error"]
    assert api.requests == []


def test_the_tool_takes_only_year_make_and_model():
    tool = make_vehicle_tool(lookup(FakeCompletions(completion(FOCUS_PAGE))))

    assert tool.name == TOOL_NAME
    assert "tiresize.com, firestonecompleteautocare.com" in tool.description
    with pytest.raises(ValidationError):
        tool.invoke({"year": 2016, "make": "Ford", "model": "Focus", "trim": "S"})
    with pytest.raises(ValidationError):
        tool.invoke({"make": "Ford", "model": "Focus"})


def test_the_app_gets_the_tool_only_with_an_openrouter_key(tmp_path):
    loaded = Settings.load()
    lookup_settings = dataclasses.replace(loaded.vehicle_lookup, cache_path=tmp_path / "pages.sqlite")
    settings = dataclasses.replace(loaded, vehicle_lookup=lookup_settings)

    assert vehicle_tools(settings) == []
    [tool] = vehicle_tools(dataclasses.replace(settings, openrouter_api_key="test-key"))
    assert tool.name == TOOL_NAME and (tmp_path / "pages.sqlite").exists()


def test_status_lines():
    assert describe_lookup({"year": 2016, "make": "Ford", "model": "Focus"}) == "Looking up tire sizes: 2016 Ford Focus"
    assert describe_lookup({}) == "Looking up tire sizes"
    assert describe_lookup_results(json.dumps({"pages": [{}, {}]})) == "Found 2 tire size pages"
    assert describe_lookup_results(json.dumps({"pages": [{}]})) == "Found 1 tire size page"
    assert describe_lookup_results(json.dumps({"pages": []})) == "No tire size pages found"
    assert describe_lookup_results(json.dumps({"error": "down"})) == "Lookup problem: down"
    assert describe_lookup_results("Error: bad arguments") == "The lookup failed"


# --- Frozen pages for the benchmark ------------------------------------------------------------------


def test_vehicles_match_when_one_model_name_contains_the_other():
    assert same_vehicle(FOCUS, Vehicle(2016, "FORD", "focus"))
    assert same_vehicle(Vehicle(2018, "BMW", "330i"), Vehicle(2018, "BMW", "3 Series 330i"))
    assert same_vehicle(Vehicle(2016, "Chevrolet", "Corvette Stingray"), Vehicle(2016, "Chevrolet", "Corvette"))
    assert not same_vehicle(FOCUS, Vehicle(2017, "Ford", "Focus"))
    assert not same_vehicle(FOCUS, Vehicle(2016, "Ford", "Fiesta"))
    assert not same_vehicle(FOCUS, Vehicle(2016, "Chevrolet", "Focus"))


def test_frozen_pages_answer_the_captured_vehicles_and_nothing_else(tmp_path):
    pages = [{"site": "tiresize.com", "url": "https://tiresize.com/x", "title": "t", "excerpt": "S\n195/65R15"}]
    row = {"vehicle": {"year": 2016, "make": "Ford", "model": "Focus"}, "captured_at": "2026-10-05 12:00 UTC", "pages": pages}
    path = tmp_path / "vehicle_pages.yaml"
    FrozenPages(SITES, 5, [row]).write(path, "Captured for a test.")

    frozen = FrozenPages.load(path)

    assert path.read_text().startswith("# Captured for a test.\n") and "excerpt: |-\n" in path.read_text()
    assert frozen.sites == SITES and frozen.max_results == 5 and frozen.row(Vehicle(2016, "ford", "focus")) == row
    assert frozen.pages(Vehicle(2016, "Ford", "Focus S")).pages == pages
    with pytest.raises(VehicleLookupError, match="no pages were captured for the 2017 Ford Focus"):
        frozen.pages(Vehicle(2017, "Ford", "Focus"))
    assert FrozenPages.load(tmp_path / "missing.yaml").rows == []


# --- Settings ----------------------------------------------------------------------------------------


def test_lookup_settings_defaults_and_overrides(tmp_path):
    cache = tmp_path / "pages.sqlite"

    assert VehicleLookupSettings.from_env({}, cache) == VehicleLookupSettings(
        sites=DEFAULT_VEHICLE_SITES, max_results=5, model=None, timeout_seconds=20.0, cache_path=cache
    )
    custom = VehicleLookupSettings.from_env({"VEHICLE_LOOKUP_SITES": " TireSize.com, mavis.com ,", "VEHICLE_LOOKUP_MAX_RESULTS": "3",
                                             "VEHICLE_LOOKUP_MODEL": "test/cheap", "VEHICLE_LOOKUP_TIMEOUT_SECONDS": "7.5"}, cache)  # fmt: skip
    assert custom.sites == ("tiresize.com", "mavis.com") and custom.max_results == 3
    assert custom.model == "test/cheap" and custom.timeout_seconds == 7.5


@pytest.mark.parametrize(
    "name, value",
    [
        ("VEHICLE_LOOKUP_SITES", "https://tiresize.com/tires"),
        ("VEHICLE_LOOKUP_SITES", "tiresize"),
        ("VEHICLE_LOOKUP_MAX_RESULTS", "0"),
        ("VEHICLE_LOOKUP_TIMEOUT_SECONDS", "-1"),
    ],
)
def test_invalid_lookup_settings_name_the_variable(tmp_path, name, value):
    with pytest.raises(ValueError, match=name):
        VehicleLookupSettings.from_env({name: value}, tmp_path / "pages.sqlite")


# --- In the agent ------------------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path):
    prompt = tmp_path / "system.md"
    prompt.write_text("You are a test assistant.")
    loaded = Settings.load()
    return dataclasses.replace(loaded, llm=dataclasses.replace(loaded.llm, system_prompt_path=prompt), agent=AgentSettings.from_env({}))


def test_the_agent_streams_the_lookup_status_and_the_pages_it_got(settings):
    model = ToolCallingModel(messages=iter([tool_call(TOOL_NAME, year=2016, make="Ford", model="Focus"),
                                            AIMessage(content="The S takes 195/65R15.")]))  # fmt: skip
    tool = make_vehicle_tool(lookup(FakeCompletions(completion(FOCUS_PAGE))))
    agent = build_agent(settings, model=model, tools=[tool])

    async def collect():
        return [event async for event in astream_turn(agent, "Tires for my 2016 Focus?", "thread-1")]

    events = asyncio.run(collect())

    statuses = [e["text"] for e in events if e["type"] == "status"]
    assert statuses == ["Thinking…", "Looking up tire sizes: 2016 Ford Focus", "Found 1 tire size page", "Thinking…"]
    [call] = [e for e in events if e["type"] == "tool_call"]
    assert call["name"] == TOOL_NAME and call["args"] == {"year": 2016, "make": "Ford", "model": "Focus"}
    assert call["error"] is None and call["result"]["pages"][0]["site"] == "tiresize.com"
    # The model read the pages, and a reopened chat shows them under the answer too.
    assert "195/65R15" in model.prompts[1][-1].content
    [_, answer] = transcript(agent.get_state({"configurable": {"thread_id": "thread-1"}}).values["messages"])
    assert answer["tool_calls"][0]["result"]["vehicle"] == "2016 Ford Focus"


def run(agent, message: str, **kwargs) -> list[dict]:
    async def collect():
        return [event async for event in astream_turn(agent, message, "thread-1", **kwargs)]

    return asyncio.run(collect())


@tool
def search_tires(size: str) -> str:
    """Search the catalog."""
    return "[]"


def test_with_web_search_off_the_model_gets_no_vehicle_lookup_and_is_told_why(settings):
    vehicle_tool = make_vehicle_tool(lookup(FakeCompletions(completion(FOCUS_PAGE))))
    model = ToolRecordingModel(messages=iter([AIMessage(content="Check the door sticker."), AIMessage(content="Which trim?")]))
    agent = build_agent(settings, model=model, tools=[search_tires, vehicle_tool])

    run(agent, "Tires for my 2016 Focus?", web_search=False)
    run(agent, "And now?")  # on by default

    assert model.bound == [["search_tires"], ["search_tires", TOOL_NAME]]
    assert model.prompts[0][0].content.endswith(WEB_SEARCH_OFF)
    assert WEB_SEARCH_OFF not in model.prompts[1][0].content


def test_a_vehicle_lookup_called_anyway_with_web_search_off_gets_an_error(settings):
    api = FakeCompletions(completion(FOCUS_PAGE))
    model = ToolRecordingModel(messages=iter([tool_call(TOOL_NAME, year=2016, make="Ford", model="Focus"), AIMessage(content="Sorry.")]))
    agent = build_agent(settings, model=model, tools=[make_vehicle_tool(lookup(api))])

    events = run(agent, "Tires for my 2016 Focus?", web_search=False)

    [call] = [e for e in events if e["type"] == "tool_call"]
    assert call["error"] == "Error: web search is off for this message, so vehicles can't be looked up."
    assert api.requests == []


def test_web_search_off_changes_nothing_for_an_agent_without_the_lookup(settings):
    model = ToolRecordingModel(messages=iter([AIMessage(content="Hi.")]))
    agent = build_agent(settings, model=model, tools=[search_tires])

    run(agent, "Hello", web_search=False)

    assert model.bound == [["search_tires"]] and WEB_SEARCH_OFF not in model.prompts[0][0].content
