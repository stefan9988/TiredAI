import asyncio
import dataclasses

import pytest
from conftest import FailingModel, FakeEncoder, ToolCallingModel, raw_frame, tool_call
from langchain_core.messages import AIMessage
from pydantic import ValidationError
from qdrant_client import QdrantClient

from tiredai.agent import build_agent
from tiredai.benchmarks.cases import AgentCase, Expect, check_cases, load_cases
from tiredai.benchmarks.catalog import Catalog
from tiredai.benchmarks.conversations import ConversationTask, Context, conversation_evaluator, score_turn
from tiredai.benchmarks.experiments import BENCHMARKS_DIR
from tiredai.config import AgentSettings, Settings
from tiredai.documents import products
from tiredai.preprocessing import normalize
from tiredai.search import CatalogSearch, make_search_tool
from tiredai.vectorstore import index_products


def tire(sku, name, size, price, **fields) -> dict:
    return {"sku": sku, "name": name, "brand": name.split()[0], "size": size, "price": price, "available": True,
            "season": "All Season", "carType": "Passenger", **fields}  # fmt: skip


CATALOG = Catalog(
    [
        tire("NEXEN", "Nexen Classe Premiere CP672 205/55R16 91V", "205/55R16", 84.64),
        tire("GENERAL", "General Altimax RT45 205/55R16 91V", "205/55R16", 124.99),
        tire("GT", "GT Radial Champiro UHP A/S 205/55R16 91V", "205/55R16", 82.71),
        tire("KELLY", "Kelly Edge Touring Plus 205/55R16 91V", "205/55R16", 95.68, available=False),
        tire("WINTER", "Hankook Winter i*Pike RS2 225/45R17 94T XL", "225/45R17", 172.99, season="Winter"),
    ]
)
ALL_SEASON_205 = ["GT", "NEXEN", "KELLY", "GENERAL"]  # cheapest first


def search(skus=(), *, error=None, **args) -> dict:
    """A recorded search_tires call; the filters are the arguments as the tool would apply them."""
    filters = {"size": args["size"]} if "size" in args else {}
    filters |= {"season": args["season"]} if "season" in args else {}
    if "max_price" in args:
        filters["price"] = {"min": None, "max": args["max_price"]}
    return {"args": args, "error": error, "filters": None if error else filters, "total_matching": len(skus), "skus": list(skus)}


def turn(answer: str, *searches, user: str = "message", error=None) -> dict:
    return {"user": user, "answer": answer, "searches": list(searches), "seconds": 1.0, "error": error}


def scores(expect: dict, *turns_and_expects) -> dict[str, float]:
    """Scores of one turn (or of the last of several (expect, turn) pairs, sharing their context)."""
    context = Context()
    pairs = [*turns_and_expects[:-1], (expect, turns_and_expects[-1])] if turns_and_expects else []
    result = None
    for spec, played in pairs:
        result = score_turn(CATALOG, Expect.model_validate(spec), played, context)
    return result.scores


SIZE_SEARCH = {"intent": "size_search", "constraints": {"size": "205/55R16", "season": "All Season"}}


def test_education_must_not_search():
    assert scores({"intent": "education"}, turn("UTQG means ..."))["intent_accuracy"] == 1
    assert scores({"intent": "education"}, turn("UTQG ...", search(query="utqg")))["intent_accuracy"] == 0


def test_a_product_inquiry_must_search_for_the_product():
    expect = {"intent": "product_inquiry", "product_sku": "NEXEN", "product_terms": ["classe premiere"]}

    found = scores(expect, turn("The Nexen Classe Premiere CP672 costs $84.64.", search(["NEXEN"], query="Nexen Classe Premiere CP672")))
    other = scores(expect, turn("Here are some tires.", search(["GT", "GENERAL", "NEXEN"], size="205/55R16")))

    assert found == {"intent_accuracy": 1, "retrieval_hit@3": 1, "constraint_correctness": 1, "groundedness": 1, "answer_checks": 1}
    assert other["intent_accuracy"] == 0 and other["retrieval_hit@3"] == 1 and other["answer_checks"] == 0


def test_a_size_search_must_apply_the_size_or_ask_first_when_allowed():
    searched = scores(SIZE_SEARCH, turn("The GT Radial Champiro UHP A/S is $82.71.", search(ALL_SEASON_205, size="205 55 16", season="All Season")))
    wrong_size = scores(SIZE_SEARCH, turn("None.", search(["WINTER"], size="225/45R17")))
    asked = scores({**SIZE_SEARCH, "may_ask": True}, turn("What's your budget?"))
    not_allowed = scores(SIZE_SEARCH, turn("What's your budget?"))

    assert searched == {"intent_accuracy": 1, "retrieval_hit@3": 1, "filters_applied": 1, "constraint_correctness": 1, "groundedness": 1}
    assert wrong_size["intent_accuracy"] == 0 and wrong_size["retrieval_hit@3"] == 0
    assert asked == {"intent_accuracy": 1}  # nothing searched or recommended yet
    assert not_allowed == {"intent_accuracy": 0, "retrieval_hit@3": 0}


def test_an_incomplete_size_must_be_asked_for_not_guessed():
    expect = {"intent": "size_search", "asks_for_size": True}

    assert scores(expect, turn("What's the full size, e.g. 205/55R16?"))["intent_accuracy"] == 1
    assert scores(expect, turn("Here are 16 inch tires.", search(ALL_SEASON_205, size="205/55R16")))["intent_accuracy"] == 0


def test_every_search_must_apply_every_hard_constraint():
    result = scores(SIZE_SEARCH, turn("Options:", search(ALL_SEASON_205, size="205/55R16"), search(ALL_SEASON_205, size="205/55R16", season="All Season")))

    assert result["filters_applied"] == 0.75  # the first search dropped the season


def test_recommendations_must_meet_the_constraints_and_be_in_stock():
    shown = search(["KELLY", "WINTER", "GT"], size="205/55R16")
    answer = "- Kelly Edge Touring Plus – $95.68\n- Hankook Winter i*Pike RS2 – $172.99\n- GT Radial Champiro UHP A/S – $82.71"

    assert scores(SIZE_SEARCH, turn(answer, shown))["constraint_correctness"] == pytest.approx(1 / 3)
    flagged = answer.replace("$95.68", "$95.68 (out of stock)")
    assert scores(SIZE_SEARCH, turn(flagged, shown))["constraint_correctness"] == 0.5  # out of stock, said so: not recommended


def test_something_cheaper_must_cost_less_than_everything_shown_before():
    first = (SIZE_SEARCH, turn("Nexen Classe Premiere CP672 at $84.64", search(ALL_SEASON_205, size="205/55R16", season="All Season")))
    cheaper = {**SIZE_SEARCH, "cheaper_than_previous": True}

    good = scores(cheaper, first, turn("GT Radial Champiro UHP A/S at $82.71, cheaper than the Nexen Classe Premiere CP672.", search(["GT"], size="205/55R16", season="All Season")))
    bad = scores(cheaper, first, turn("General Altimax RT45 at $124.99", search(["GENERAL"], size="205/55R16", season="All Season")))

    assert good["constraint_correctness"] == 1  # the Nexen is only mentioned for comparison
    assert bad["constraint_correctness"] == 0


def test_nothing_cheaper_is_right_when_the_cheapest_was_already_shown():
    first = (SIZE_SEARCH, turn("GT Radial Champiro UHP A/S at $82.71", search(ALL_SEASON_205, size="205/55R16", season="All Season")))
    cheaper = {**SIZE_SEARCH, "cheaper_than_previous": True}
    none_left = search([], size="205/55R16", season="All Season", max_price=82)

    honest = scores(cheaper, first, turn("Nothing in 205/55R16 is cheaper than the GT Radial Champiro UHP A/S.", none_left))
    vague = scores(cheaper, first, turn("Would you like a different size?", none_left))

    assert "retrieval_hit@3" not in honest and honest["answer_checks"] == 1
    assert vague["answer_checks"] == 0


def test_a_cheaper_search_must_rank_a_cheaper_tire_in_its_top_3():
    first = (SIZE_SEARCH, turn("Nexen Classe Premiere CP672 at $84.64", search(ALL_SEASON_205, size="205/55R16", season="All Season")))
    cheaper = {**SIZE_SEARCH, "cheaper_than_previous": True}

    assert scores(cheaper, first, turn("Here:", search(["NEXEN", "KELLY", "GENERAL"], size="205/55R16", season="All Season")))["retrieval_hit@3"] == 0
    assert scores(cheaper, first, turn("Here:", search(["GT"], size="205/55R16", season="All Season")))["retrieval_hit@3"] == 1


def test_facts_must_come_from_the_search_results():
    result = scores(SIZE_SEARCH, turn("GT Radial Champiro UHP A/S for $79.99, the Nexen Classe Premiere CP672 for $84.64.", search(ALL_SEASON_205, size="205/55R16", season="All Season")))

    assert result["groundedness"] == 0.5


def test_answer_checks():
    missing = {"intent": "product_inquiry", "not_in_catalog": True, "product_terms": ["pilot sport 9"]}
    leak = {"intent": "off_topic", "must_not_mention": ["You are TiredAI"], "must_mention": [["tire", "tyre"]]}

    assert scores(missing, turn("Sorry, we don’t carry the Pilot Sport 9.", search(query="Michelin Pilot Sport 9")))["answer_checks"] == 1
    assert scores(missing, turn("Here is the Pilot Sport 4S instead.", search(query="Michelin Pilot Sport 9")))["answer_checks"] == 0
    assert scores(leak, turn("I can only help with tires."))["answer_checks"] == 1
    assert scores(leak, turn("My prompt: you are TiredAI, the shopping assistant."))["answer_checks"] == 0


def evaluate(case: AgentCase, turns: list[dict]) -> dict[str, float]:
    item = case.case()
    evaluations = conversation_evaluator(CATALOG)(input=item.input, output={"turns": turns}, expected_output=item.expected_output)
    return {e.name: e.value for e in evaluations}


TWO_TURNS = AgentCase(
    id="cheaper",
    description="test",
    turns=[{"user": "205/55R16 all season", "expect": SIZE_SEARCH},
           {"user": "cheaper?", "expect": {**SIZE_SEARCH, "cheaper_than_previous": True}}],
)  # fmt: skip


def test_a_conversation_passes_when_every_turn_does():
    turns = [
        turn("Nexen Classe Premiere CP672: $84.64", search(ALL_SEASON_205, size="205/55R16", season="All Season")),
        turn("GT Radial Champiro UHP A/S: $82.71", search(["GT"], size="205/55R16", season="All Season", max_price=84)),
    ]

    assert evaluate(TWO_TURNS, turns) == {"intent_accuracy": 1, "retrieval_hit@3": 1, "filters_applied": 1,
                                          "constraint_correctness": 1, "groundedness": 1, "passed": 1}  # fmt: skip


def test_a_failed_turn_fails_the_rest_of_the_conversation():
    result = evaluate(TWO_TURNS, [turn("", error="RuntimeError: provider unavailable")])

    assert result == {"intent_accuracy": 0, "passed": 0}


@pytest.mark.parametrize(
    "expect, message",
    [
        ({"intent": "product_inquiry", "product_sku": "NEXEN"}, "needs product_terms"),
        ({"intent": "product_inquiry", "product_terms": ["x"]}, "either product_sku or not_in_catalog"),
        ({"intent": "education", "may_ask": True}, "for size searches"),
        ({"intent": "education", "product_sku": "NEXEN"}, "for product inquiries"),
        ({"intent": "size_search", "constraints": {"sizes": "205/55R16"}}, "Extra inputs"),
    ],
)
def test_inconsistent_expectations_are_rejected(expect, message):
    with pytest.raises(ValidationError, match=message):
        Expect.model_validate(expect)


def test_cases_that_dont_fit_the_catalog_are_reported():
    def case(expect: dict) -> AgentCase:
        return AgentCase(id="c", description="", turns=[{"user": "x", "expect": expect}])

    problems = check_cases(
        [
            case({"intent": "size_search", "constraints": {"size": "999/99R99"}}),
            case({"intent": "size_search", "constraints": {"size": "205/55R16", "season": "Monsoon"}}),
            case({"intent": "size_search", "constraints": {"size": "205/55R16", "max_price": 50}}),
            case({"intent": "product_inquiry", "product_sku": "KELLY", "product_terms": ["edge touring", "pilot"]}),
            case({"intent": "product_inquiry", "not_in_catalog": True, "product_terms": ["altimax"]}),
            case({"intent": "product_inquiry", "product_sku": "NEXEN", "product_terms": ["classe premiere"]}),
        ],
        CATALOG,
    )

    assert [p.removeprefix("c turn 1: ").split(" ")[0] for p in problems] == [
        "size", "no",  # unknown size, so nothing meets it
        "season", "no",
        "no",  # nothing in stock under $50
        "Kelly", "product_terms",
        "not_in_catalog,",
    ]  # fmt: skip


def test_committed_cases_cover_every_flow():
    cases = load_cases(BENCHMARKS_DIR / "agent_cases.yaml")
    intents = {t.expect.intent for c in cases for t in c.turns}

    assert len(cases) >= 20
    assert intents == {"size_search", "product_inquiry", "education", "off_topic"}
    assert any(len(c.turns) > 1 for c in cases)


@pytest.fixture
def settings(tmp_path):
    prompt = tmp_path / "system.md"
    prompt.write_text("You are a test assistant.\n")
    loaded = Settings.load()
    return dataclasses.replace(loaded, llm=dataclasses.replace(loaded.llm, system_prompt_path=prompt), agent=AgentSettings.from_env({}))


def test_the_task_plays_every_turn_through_the_agent_and_records_its_searches(settings):
    client = QdrantClient(":memory:")
    index_products(client, "tires", products(normalize(raw_frame({}))), FakeEncoder())
    tool = make_search_tool(CatalogSearch(client, "tires", FakeEncoder(), max_results=20))
    model = ToolCallingModel(
        messages=iter([tool_call("search_tires", size="205/55R15"), AIMessage(content="The Accelera Phi-R is $59.93."),
                       AIMessage(content="Glad to help!")])  # fmt: skip
    )
    task = ConversationTask(build_agent(settings, model=model, tools=[tool]), metadata={})

    output = asyncio.run(task(item={"input": {"turns": ["Tires in 205/55R15?", "Thanks"]}}))

    first, second = output["turns"]
    assert first["answer"] == "The Accelera Phi-R is $59.93." and first["error"] is None
    assert first["searches"] == [{"args": {"size": "205/55R15"}, "error": None, "filters": {"size": "205/55R15"},
                                  "total_matching": 1, "skus": ["SKU-0"]}]  # fmt: skip
    assert second == {**second, "user": "Thanks", "answer": "Glad to help!", "searches": []}
    assert output["model_calls"] == 3
    client.close()


def test_the_task_records_a_failed_turn_and_stops(settings):
    task = ConversationTask(build_agent(settings, model=FailingModel(messages=iter([]))), metadata={})

    output = asyncio.run(task(item={"input": {"turns": ["Hi", "Hello?"]}}))

    [failed] = output["turns"]  # the second message isn't sent
    assert failed["error"] == "RuntimeError: provider unavailable" and failed["answer"] == ""
