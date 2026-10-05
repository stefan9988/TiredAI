"""Agent benchmark: whole conversations through the real agent, every turn scored with deterministic checks.

The task sends each shopper message through astream_turn, exactly like the chat UI, and records the
answer, every search_tires call (arguments, filters as applied, returned SKUs) and the time taken.
The evaluator reads that against the turn's expectations (cases.Expect) and the catalog.

Scores of a conversation, each the mean over the turns it applies to:
- intent_accuracy: the turn took its flow's route. Education and off-topic: no search. Product
  inquiry: a search whose query names the product (product_terms). Size search: a search with the
  size filter, or, where asking is allowed (may_ask), a question without a search; with an incomplete
  size (asks_for_size): a question and no search with a guessed size.
- retrieval_hit@3: the requested product is in the top 3 of a search; for a size search, a product
  in the top 3 meets the constraints and is in stock (and with cheaper_than_previous costs less than
  the cheapest product recommended before; when the catalog has nothing cheaper, this doesn't apply).
- filters_applied: the share of the turn's hard constraints each search applied as filters.
- constraint_correctness: the share of recommended products (named, and not called out of stock)
  that meet the constraints, are in stock, and with cheaper_than_previous cost less than the cheapest
  product recommended in the previous turn.
- groundedness: the share of checkable facts in the answer (prices, SKUs, specs; see answers.py)
  that the products the agent was shown support.
- answer_checks: must_mention and must_not_mention phrases, "not in the catalog" for products that
  aren't, naming the requested product, and saying so when nothing cheaper is left.
- passed: 1 when every score of every turn is 1.
"""

import asyncio
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage
from langfuse import Evaluation

from tiredai.agent import astream_turn, thread_config
from tiredai.benchmarks.answers import check_facts, has_phrase, meets, mentions, numbers, words
from tiredai.benchmarks.cases import Expect
from tiredai.benchmarks.catalog import Catalog
from tiredai.benchmarks.experiments import item_value
from tiredai.benchmarks.metrics import mean
from tiredai.search import parse_size

SCORES = ["intent_accuracy", "retrieval_hit@3", "filters_applied", "constraint_correctness", "groundedness", "answer_checks"]
NOT_IN_CATALOG = (
    "not in our catalog", "not in the catalog", "isn't in our catalog", "isn't in the catalog", "not in our inventory",
    "couldn't find", "could not find", "can't find", "cannot find", "didn't find", "did not find", "unable to find",
    "don't carry", "do not carry", "don't have", "do not have", "doesn't exist", "does not exist", "no exact match",
    "not found", "no match", "doesn't appear", "does not appear", "not listed", "isn't listed", "don't see", "do not see",
    "don't stock", "do not stock", "no results", "not available in our catalog", "not available in the catalog",
    "not part of our catalog", "don't offer", "do not offer",
)  # fmt: skip
# Saying that nothing cheaper is left, when nothing is.
NOTHING_CHEAPER = ("nothing", "none", "no cheaper", "no other", "the cheapest", "cheapest available", "lowest price",
                   "lowest-priced", "least expensive", *NOT_IN_CATALOG)  # fmt: skip
# Provider errors that pass, so the conversation is played again: overloads, rate limits, timeouts. A
# daily quota that ran out ("free-models-per-day") is not one of them.
TRANSIENT = re.compile(r"\b(429|500|502|503|504|529)\b|overloaded|temporarily|rate.?limit|timed? ?out", re.IGNORECASE)
RETRY_WAITS = (20, 60)  # seconds before the second and third attempt
FILTER_KEYS = {"season": "season", "brand": "brand", "car_type": "carType", "performance": "performance"}


def search_record(event: dict) -> dict:
    """A search_tires call from the stream's tool_call event, without the product payloads (the catalog has them)."""
    result = event["result"] if isinstance(event["result"], dict) else {}
    return {
        "args": event["args"] if isinstance(event["args"], dict) else {"raw": event["args"]},
        "error": event["error"],
        "filters": result.get("filters") if not event["error"] else None,
        "total_matching": result.get("total_matching"),
        "skus": [p["sku"] for p in result.get("products", [])],
    }


def transient(error: str | None) -> bool:
    return bool(error) and "per-day" not in error and bool(TRANSIENT.search(error))


class ConversationTask:
    """Plays an item's shopper messages to the agent, one turn after another, in a new conversation.

    When a turn fails with a provider error that passes (TRANSIENT), the whole conversation is played
    again in a new thread, up to len(RETRY_WAITS) times, so the scores measure the model rather than
    its provider's load. `attempts` in the output says how many plays it took.
    """

    def __init__(self, agent, metadata: dict, *, waits: tuple[float, ...] = RETRY_WAITS,
                 sleep: Callable[[float], Awaitable] = asyncio.sleep):  # fmt: skip
        self.agent = agent
        self.metadata = metadata  # recorded on every turn's trace, like the app does
        self.waits = waits
        self.sleep = sleep

    async def __call__(self, *, item, **kwargs) -> dict:
        messages = item_value(item, "input")["turns"]
        for attempt, wait in enumerate((*self.waits, None), start=1):
            output = await self._play(messages)
            error = next((t["error"] for t in output["turns"] if t["error"]), None)
            if wait is None or not transient(error):
                return {**output, "attempts": attempt}
            await self.sleep(wait)
        raise AssertionError("unreachable")

    async def _play(self, messages: list[str]) -> dict:
        thread_id = f"benchmark-{uuid.uuid4()}"
        turns = []
        for message in messages:
            turns.append(await self._turn(message, thread_id))
            if turns[-1]["error"]:
                break  # the conversation can't go on as written
        return {"thread_id": thread_id, "turns": turns, **await self._usage(thread_id)}

    async def _turn(self, message: str, thread_id: str) -> dict:
        started = time.perf_counter()
        answer, searches, error = [], [], None
        try:
            async for event in astream_turn(self.agent, message, thread_id, source="benchmark", metadata=self.metadata):
                if event["type"] == "token":
                    answer.append(event["text"])
                elif event["type"] == "tool_call" and event["name"] == "search_tires":
                    searches.append(search_record(event))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        seconds = round(time.perf_counter() - started, 2)
        return {"user": message, "answer": "".join(answer), "searches": searches, "seconds": seconds, "error": error}

    async def _usage(self, thread_id: str) -> dict:
        state = await self.agent.aget_state(thread_config(thread_id))
        calls = [m for m in state.values.get("messages", []) if isinstance(m, AIMessage)]
        usage = [m.usage_metadata or {} for m in calls]
        return {
            "model_calls": len(calls),
            "input_tokens": sum(u.get("input_tokens", 0) for u in usage),
            "output_tokens": sum(u.get("output_tokens", 0) for u in usage),
        }


@dataclass
class TurnScore:
    scores: dict[str, float] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)

    def set(self, name: str, value: float | bool, note: str) -> None:
        self.scores[name] = float(value)
        self.notes[name] = note


@dataclass
class Context:
    """What the conversation has shown so far."""

    seen: dict[str, dict] = field(default_factory=dict)  # products returned by any search, by SKU
    previous_recommended: list[str] = field(default_factory=list)
    shopper_numbers: set[float] = field(default_factory=set)


def _applied(filters: dict, key: str, value) -> bool:
    if key == "size":
        applied = filters.get("size") or []
        applied = [applied] if isinstance(applied, str) else applied
        return any(parse_size(v).key == parse_size(value).key for v in applied)
    if key in ("min_price", "max_price"):
        bound = (filters.get("price") or {}).get("max" if key == "max_price" else "min")
        return bound is not None and (bound <= value + 0.005 if key == "max_price" else bound >= value - 0.005)
    if key == "run_flat":
        return filters.get("runFlat") == value
    applied = filters.get(FILTER_KEYS[key]) or []
    applied = [applied] if isinstance(applied, str) else applied
    return any(str(v).lower() == str(value).lower() for v in applied)


def _query_names(search: dict, terms: list[str]) -> bool:
    query = search["args"].get("query") or ""
    # As a substring of the words, so a brand's spelling counts too: "ContiCrossContact" names "crosscontact".
    return bool(query) and all(words(term) in words(query) for term in terms)


def _describe(searches: list[dict]) -> str:
    if not searches:
        return "no search"
    parts = []
    for s in searches:
        args = ", ".join(f"{k}={v!r}" for k, v in s["args"].items())
        parts.append(f"search({args})" + (f" -> error: {s['error'][:80]}" if s["error"] else f" -> {s['total_matching']} matching"))
    return "; ".join(parts)


def score_turn(catalog: Catalog, expect: Expect, turn: dict, context: Context) -> TurnScore:
    """The turn's scores; `context` is updated with what the turn showed."""
    score = TurnScore()
    searches = turn["searches"]
    ok = [s for s in searches if s["filters"] is not None]
    answer, asked = turn["answer"], "?" in turn["answer"]
    constraints = expect.constraints.given()
    previous = [catalog[sku]["price"] for sku in context.previous_recommended]
    # "Cheaper" is relative to what the previous answer recommended, so whether it can be met depends on that.
    below = min(previous) if expect.cheaper_than_previous and previous else None

    def fits(product: dict) -> bool:
        return bool(product.get("available")) and not meets(product, constraints) and (below is None or product["price"] < below)

    nothing_cheaper = below is not None and not any(fits(p) for p in catalog)
    context.shopper_numbers |= numbers(turn["user"])
    for s in ok:
        context.seen.update((sku, catalog[sku]) for sku in s["skus"] if sku in catalog)

    # intent_accuracy
    what = _describe(searches)
    if expect.intent in ("education", "off_topic"):
        score.set("intent_accuracy", not searches, what)
    elif expect.intent == "product_inquiry":
        score.set("intent_accuracy", any(_query_names(s, expect.product_terms) for s in searches),
                  f"{what}; the query must contain {expect.product_terms}")  # fmt: skip
    elif expect.asks_for_size:
        guessed = any(s["filters"].get("size") for s in ok)
        score.set("intent_accuracy", asked and not guessed, f"{what}; {'asked' if asked else 'did not ask'} a question")
    else:
        sized = any(_applied(s["filters"], "size", constraints["size"]) for s in ok) if "size" in constraints else bool(ok)
        asked_first = expect.may_ask and not searches and asked
        score.set("intent_accuracy", sized or asked_first, what + ("; asked a question first" if asked_first else ""))

    # retrieval_hit@3
    if expect.product_sku:
        hit = any(expect.product_sku in s["skus"][:3] for s in ok)
        score.set("retrieval_hit@3", hit, f"{expect.product_sku} {'in' if hit else 'not in'} the top 3 of a search")
    elif expect.intent == "size_search" and not expect.asks_for_size and not (expect.may_ask and not searches) and not nothing_cheaper:
        good = {sku for s in ok for sku in s["skus"][:3] if sku in catalog and fits(catalog[sku])}
        cheaper = f", cheaper than ${below:.2f}" if below is not None else ""
        score.set("retrieval_hit@3", bool(good), f"{len(good)} in-stock products meeting the constraints{cheaper} in the top 3 of the searches")

    # filters_applied
    if constraints and ok and not expect.asks_for_size:
        shares, missing = [], set()
        for s in ok:
            applied = [key for key in constraints if _applied(s["filters"], key, constraints[key])]
            shares.append(len(applied) / len(constraints))
            missing |= set(constraints) - set(applied)
        note = f"missing {sorted(missing)}" if missing else f"all of {sorted(constraints)} applied"
        score.set("filters_applied", mean(shares), note)

    # constraint_correctness
    named = mentions(answer, list(context.seen.values()))
    recommended = [m for m in named if not m.unavailable]
    if recommended:
        failures = []
        for mention in recommended:
            reasons = []
            for sku in mention.skus:  # a mention that fits several products passes if one of them does
                product = catalog[sku]
                failed = meets(product, constraints) + ([] if product.get("available") else ["available"])
                # Products recommended before may come up again for comparison.
                if below is not None and sku not in context.previous_recommended and product["price"] >= below:
                    failed.append(f"cheaper than ${below:.2f}")
                if not failed:
                    break
                reasons.append(f"{product['name']}: {', '.join(failed)}")
            else:
                failures.append("; ".join(reasons))
        share = 1 - len(failures) / len(recommended)
        score.set("constraint_correctness", share, "fails " + " | ".join(failures) if failures else f"{len(recommended)} recommended, all fine")
    context.previous_recommended = [sku for m in recommended for sku in m.skus]

    # groundedness
    facts = check_facts(answer, list(context.seen.values()), context.shopper_numbers)
    if facts:
        unsupported = [f"{f.kind} {f.text!r}" for f in facts if not f.supported]
        note = "unsupported: " + ", ".join(unsupported) if unsupported else f"{len(facts)} facts, all supported"
        score.set("groundedness", 1 - len(unsupported) / len(facts), note)

    # answer_checks
    checks = [(any(has_phrase(answer, p) for p in group), f"mentions {group[0]!r}") for group in expect.must_mention]
    checks += [(not has_phrase(answer, p), f"doesn't mention {p!r}") for p in expect.must_not_mention]
    if expect.not_in_catalog:
        checks.append((any(has_phrase(answer, p) for p in NOT_IN_CATALOG), "says the product isn't in the catalog"))
    if expect.product_sku:
        checks.append((any(expect.product_sku in m.skus for m in named), "names the requested product"))
    if nothing_cheaper:
        checks.append((any(has_phrase(answer, p) for p in NOTHING_CHEAPER), f"says nothing below ${below:.2f} meets the constraints"))
    if checks:
        failed = [label for passed, label in checks if not passed]
        score.set("answer_checks", 1 - len(failed) / len(checks), "failed: " + "; ".join(failed) if failed else "all passed")
    return score


def score_conversation(catalog: Catalog, expects: list[Expect], turns: list[dict]) -> list[TurnScore]:
    context, scores = Context(), []
    for number, expect in enumerate(expects):
        turn = turns[number] if number < len(turns) else None
        if turn is None or turn["error"]:
            failed = TurnScore()
            failed.set("intent_accuracy", 0, f"turn failed: {turn['error']}" if turn else "not run: an earlier turn failed")
            scores.append(failed)
            continue
        scores.append(score_turn(catalog, expect, turn, context))
    return scores


def conversation_evaluator(catalog: Catalog):
    """The item evaluator: each score is the mean over the turns it applies to, its comment says why per turn."""

    def evaluate_conversation(*, output, expected_output, **kwargs) -> list[Evaluation]:
        expects = [Expect.model_validate(e) for e in expected_output["turns"]]
        turns = score_conversation(catalog, expects, output["turns"])
        evaluations = []
        for name in SCORES:
            applied = [(n, t) for n, t in enumerate(turns, start=1) if name in t.scores]
            if applied:
                comment = "\n".join(f"turn {n}: {t.scores[name]:.2f} – {t.notes[name]}" for n, t in applied)
                evaluations.append(Evaluation(name=name, value=mean(t.scores[name] for _, t in applied), comment=comment))
        failing = [f"turn {n} {name}" for n, t in enumerate(turns, start=1) for name, v in t.scores.items() if v < 1]
        evaluations.append(Evaluation(name="passed", value=float(not failing), comment="failing: " + ", ".join(failing) if failing else None))
        return evaluations

    return evaluate_conversation


def turn_seconds(output: dict) -> list[float]:
    return [t["seconds"] for t in output["turns"] if not t["error"]]
