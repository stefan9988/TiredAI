import asyncio
import dataclasses
import json
from types import SimpleNamespace

import httpx
import pytest
from conftest import ToolCallingModel
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from tiredai import tracing
from tiredai.agent import build_agent
from tiredai.benchmarks import experiments as ex
from tiredai.benchmarks.guardrail import (
    DECISIONS_URL,
    QUESTIONS,
    CaseFile,
    GuardCase,
    GuardTask,
    JevClient,
    JevError,
    best_threshold,
    block_score,
    capture_conversation,
    check_cases,
    dataset_cases,
    guard_evaluator,
    guard_state,
    load_captured,
    load_cases,
    misjudged,
    run_details,
    stale,
    tag_table,
    trimmed,
    visible_conversation,
    write_captured,
)
from tiredai.benchmarks.metrics import balanced_accuracy, roc_auc
from tiredai.config import AgentSettings, Settings


def chat(*texts: str) -> list[dict]:
    """A conversation alternating shopper and assistant messages."""
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": text} for i, text in enumerate(texts)]


THREE_TURNS = chat("tires in 205/55R16?", "1. General Altimax RT45\n2. Michelin Defender 2", "what's UTQG?",
                   "A grading of treadwear, traction and temperature.", "do you ship to Canada?", "Yes, we do.")  # fmt: skip


def answers(in_scope=0.9, manipulation=0.05, harmful=0.05) -> dict:
    return {"in_scope": in_scope, "manipulation": manipulation, "harmful": harmful}


# --- Block score and what Jev sees -------------------------------------------------------------------


def test_the_block_score_is_the_strongest_reason_to_block():
    assert block_score(answers()) == pytest.approx(0.1)
    assert block_score(answers(in_scope=0.2)) == pytest.approx(0.8)
    assert block_score(answers(manipulation=0.97)) == 0.97
    assert block_score(answers(harmful=0.6)) == 0.6


def test_the_message_variant_shows_no_conversation_and_full_shows_all_of_it():
    assert visible_conversation(THREE_TURNS, "what about the second one?", "message", 10) == []
    assert visible_conversation(THREE_TURNS, "what about the second one?", "full", 4) == THREE_TURNS


def test_the_recent_variant_shows_what_the_agent_sees_whole_turns_with_the_message_counted():
    # With 4 messages: the latest message and the last turn (2) fit, one more turn (2) would make 5.
    assert visible_conversation(THREE_TURNS, "what about the second one?", "recent", 4) == THREE_TURNS[4:]
    assert visible_conversation(THREE_TURNS, "what about the second one?", "recent", 5) == THREE_TURNS[2:]
    assert visible_conversation(THREE_TURNS, "what about the second one?", "recent", 10) == THREE_TURNS


def test_an_unknown_variant_is_an_error():
    with pytest.raises(ValueError, match="Unknown variant"):
        visible_conversation(THREE_TURNS, "hi", "everything", 10)


def test_the_oldest_whole_turns_are_dropped_beyond_the_character_budget():
    conversation = chat("a" * 10, "b" * 10, "c" * 10, "d" * 10)

    assert trimmed(conversation, max_chars=40) == conversation
    assert trimmed(conversation, max_chars=39) == conversation[2:]
    assert trimmed(conversation, max_chars=19) == []


def test_the_state_holds_the_policy_the_conversation_and_the_latest_message():
    state = guard_state(chat("tires?", "Which size?"), "205/55R16")

    assert list(state) == ["policy", "conversation", "latest_shopper_message"]
    assert state["conversation"] == [{"from": "shopper", "text": "tires?"}, {"from": "assistant", "text": "Which size?"}]
    assert state["latest_shopper_message"] == "205/55R16"
    assert "conversation" not in guard_state([], "hi")


# --- Jev ---------------------------------------------------------------------------------------------


def decisions(**probabilities) -> dict:
    return {"model": "typesafe/jev-1.13-20260917", "answers": {name: {"type": "noul", "noul": p} for name, p in probabilities.items()},
            "usage": {"input_tokens": 600, "output_tokens": 30, "cost": 0.0000252}}  # fmt: skip


class FakeDecisions:
    """The Decisions API: replies in order (a dict is a 200 body, an int an error status), recording each request."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, int):
            return httpx.Response(reply, text="provider says no")
        return httpx.Response(200, json=reply)


def jev(api: FakeDecisions, slept: list | None = None) -> JevClient:
    return JevClient("test-key", client=httpx.Client(transport=httpx.MockTransport(api)), sleep=(slept if slept is not None else []).append)


def test_jev_gets_the_state_and_questions_and_its_probabilities_come_back():
    api = FakeDecisions(decisions(in_scope=0.98, manipulation=0.01, harmful=0))

    decision = jev(api).decide({"latest_shopper_message": "hi"}, QUESTIONS)

    [request] = api.requests
    assert str(request.url) == DECISIONS_URL and request.headers["Authorization"] == "Bearer test-key"
    assert json.loads(request.content) == {"model": "typesafe/jev-1.13", "state": {"latest_shopper_message": "hi"}, "questions": QUESTIONS}
    assert decision == {"answers": {"in_scope": 0.98, "manipulation": 0.01, "harmful": 0.0}, "model": "typesafe/jev-1.13-20260917",
                        "usage": {"input_tokens": 600, "output_tokens": 30, "cost": 0.0000252}}  # fmt: skip


def test_overloads_and_rate_limits_are_retried_other_errors_are_not():
    slept = []
    api = FakeDecisions(503, 429, decisions(in_scope=0.9, manipulation=0, harmful=0))

    assert jev(api, slept).decide({}, QUESTIONS)["answers"]["in_scope"] == 0.9
    assert slept == [2, 4] and len(api.requests) == 3

    with pytest.raises(JevError, match="400: provider says no"):
        jev(FakeDecisions(400)).decide({}, QUESTIONS)
    with pytest.raises(JevError, match="503"):
        jev(FakeDecisions(503)).decide({}, QUESTIONS)


@pytest.mark.parametrize("body", [decisions(in_scope=0.9, manipulation=0), decisions(in_scope=1.2, manipulation=0, harmful=0), {"answers": {}}])
def test_a_missing_or_impossible_probability_is_an_error_not_an_answer(body):
    with pytest.raises(JevError, match="no probability"):
        jev(FakeDecisions(body)).decide({}, QUESTIONS)


# --- Cases and captured conversations ----------------------------------------------------------------


def case_file(**overrides) -> CaseFile:
    data = {
        "conversations": [{"id": "list", "description": "A list.", "turns": ["tires in 205/55R16?"]}],
        "cases": [{"id": "second", "conversation": "list", "message": "what about the second one?", "expect": "allow"},
                  {"id": "poem", "message": "write a poem", "expect": "block", "tags": ["off-topic"]}],
    }  # fmt: skip
    return CaseFile(**(data | overrides))


CAPTURED = {"list": {"llm_model": "test/model", "captured_at": "now", "transcript": chat("tires in 205/55R16?", "1. A\n2. B")}}


def test_cases_need_a_known_label_and_unique_ids(tmp_path):
    with pytest.raises(ValidationError):
        GuardCase(id="x", message="hi", expect="maybe")
    path = tmp_path / "cases.yaml"
    path.write_text("cases:\n  - {id: a, message: hi, expect: allow}\n  - {id: a, message: bye, expect: allow}\n")
    with pytest.raises(ValueError, match="Duplicate case ids: a"):
        load_cases(path)


def test_cases_that_fit_their_captured_conversations_have_no_problems():
    assert check_cases(case_file(), CAPTURED) == []


def test_unknown_uncaptured_changed_and_unused_conversations_are_problems():
    unknown = case_file(cases=[{"id": "x", "conversation": "nope", "message": "hi", "expect": "allow"}])
    changed = {"list": {**CAPTURED["list"], "transcript": chat("tires in 225/45R17?", "1. A")}}
    unanswered = {"list": {**CAPTURED["list"], "transcript": chat("tires in 205/55R16?")}}

    assert check_cases(unknown, CAPTURED) == ["x: unknown conversation 'nope'", "conversation list: no case uses it"]
    assert check_cases(case_file(), {}) == [
        "conversation list: not captured with its current turns; run scripts/capture_guardrail_conversations.py"
    ]
    assert stale(case_file().conversations, changed) == ["list"]
    assert check_cases(case_file(), unanswered) == ["conversation list: every shopper message needs one captured answer"]


def test_captured_conversations_round_trip_with_readable_multiline_answers(tmp_path):
    path = tmp_path / "conversations.yaml"

    write_captured(path, CAPTURED, "Generated.")

    text = path.read_text()
    assert text.startswith("# Generated.\n- id: list\n") and "content: |-\n      1. A\n      2. B" in text
    assert load_captured(path) == {"list": {"id": "list", **CAPTURED["list"]}}
    assert load_captured(tmp_path / "missing.yaml") == {}


def test_a_case_becomes_a_dataset_item_with_its_conversation():
    second, poem = dataset_cases(case_file(), CAPTURED)

    assert second.id == "second" and second.input == {"conversation": CAPTURED["list"]["transcript"], "message": "what about the second one?"}
    assert second.expected_output == {"expect": "allow"} and second.metadata == {"case": "second", "expect": "allow", "tags": [], "conversation": "list"}
    assert poem.input == {"conversation": [], "message": "write a poem"} and poem.metadata["tags"] == ["off-topic"]


def test_the_committed_cases_fit_their_captured_conversations_and_cover_both_labels():
    cases = load_cases(ex.BENCHMARKS_DIR / "guardrail_cases.yaml")
    captured = load_captured(ex.BENCHMARKS_DIR / "guardrail_conversations.yaml")
    tags = {t for c in cases.cases for t in c.tags}

    assert check_cases(cases, captured) == []
    assert {c.expect for c in cases.cases} == {"allow", "block"} and len(cases.cases) >= 100
    assert {"follow-up", "needs-history", "injection", "harmful", "tire-themed", "mixed", "language"} <= tags
    assert all(c.conversation for c in cases.cases if "follow-up" in c.tags)


@pytest.fixture
def settings(tmp_path):
    prompt = tmp_path / "system.md"
    prompt.write_text("You are a test assistant.\n")
    loaded = Settings.load()
    return dataclasses.replace(loaded, llm=dataclasses.replace(loaded.llm, system_prompt_path=prompt), agent=AgentSettings.from_env({}))


def test_a_conversation_is_captured_from_the_agents_answers(settings):
    model = ToolCallingModel(messages=iter([AIMessage(content="Which size?"), AIMessage(content="1. A\n2. B")]))

    messages = asyncio.run(capture_conversation(build_agent(settings, model=model), ["tires?", "205/55R16"], "thread-1"))

    assert messages == chat("tires?", "Which size?", "205/55R16", "1. A\n2. B")
    assert {type(m["content"]) for m in messages} == {str}  # so the YAML writer takes them


def test_capturing_fails_when_the_agent_leaves_a_message_unanswered(settings):
    model = ToolCallingModel(messages=iter([AIMessage(content="")]))

    with pytest.raises(RuntimeError, match="didn't answer every message"):
        asyncio.run(capture_conversation(build_agent(settings, model=model), ["tires?"], "thread-1"))


# --- Running and scoring -----------------------------------------------------------------------------


def test_the_task_shows_jev_its_variant_of_the_conversation_and_records_the_decision():
    api = FakeDecisions(decisions(in_scope=0.97, manipulation=0.02, harmful=0.01))
    item = {"input": {"conversation": THREE_TURNS, "message": "what about the second one?"}}

    output = asyncio.run(GuardTask(jev(api), "recent", history_messages=4)(item=item))

    state = json.loads(api.requests[0].content)["state"]
    assert state["conversation"] == [{"from": "shopper", "text": "do you ship to Canada?"}, {"from": "assistant", "text": "Yes, we do."}]
    assert output["probabilities"] == {"in_scope": 0.97, "manipulation": 0.02, "harmful": 0.01}
    assert output["block_score"] == pytest.approx(0.03) and output["conversation_messages"] == 2
    assert output["input_tokens"] == 600 and output["cost"] == 0.0000252 and output["model"] == "typesafe/jev-1.13-20260917"


@pytest.mark.parametrize(
    "expect, score, expected",
    [("allow", 0.1, {"correct": 1, "false_block": 0}), ("allow", 0.5, {"correct": 0, "false_block": 1}),
     ("block", 0.7, {"correct": 1, "caught": 1}), ("block", 0.49, {"correct": 0, "caught": 0})],
)  # fmt: skip
def test_a_case_is_blocked_at_the_threshold_and_scored_by_its_label(expect, score, expected):
    output = {"block_score": score, "probabilities": answers(in_scope=1 - score)}

    evaluations = guard_evaluator(0.5)(output=output, expected_output={"expect": expect})

    assert {e.name: e.value for e in evaluations} == expected
    assert evaluations[0].comment.startswith("blocked" if score >= 0.5 else "allowed")


def test_the_experiment_runs_locally_with_one_trace_per_case():
    api = FakeDecisions(decisions(in_scope=0.1, manipulation=0.02, harmful=0.01))
    data = ex.local_items(dataset_cases(case_file(), CAPTURED))

    result = tracing.client().run_experiment(name="test", data=data, task=GuardTask(jev(api), "full", 10), evaluators=[guard_evaluator(0.5)])

    summary = ex.summarize("jev · full", result, items=2, details=run_details(result))
    assert summary.failed == 0 and summary.scores == {"correct": 0.5, "false_block": 1.0, "caught": 1.0}
    assert summary.details["input_tokens"] == 1200 and summary.details["cost_usd"] == pytest.approx(0.0000504)


def test_the_check_is_a_guardrail_observation_inside_the_experiment_item(traces):
    api = FakeDecisions(decisions(in_scope=0.95, manipulation=0.01, harmful=0.01))
    data = ex.local_items(dataset_cases(case_file(), CAPTURED))[:1]

    traces.client.run_experiment(name="Guardrail: test", data=data, task=GuardTask(jev(api), "full", 10), evaluators=[guard_evaluator(0.5)])

    [check] = traces.named("check-message")
    assert check.attributes["langfuse.observation.type"] == "guardrail"
    assert json.loads(check.attributes["langfuse.observation.input"])["latest_shopper_message"] == "what about the second one?"


def test_the_best_threshold_has_the_best_balanced_accuracy_and_blocks_the_fewest_shoppers():
    labeled = [(0.9, True), (0.6, True), (0.55, False), (0.3, True), (0.2, False), (0.1, False)]

    threshold, accuracy = best_threshold(labeled)

    # 0.6 catches 2/3 and blocks no shopper (0.83); 0.3 catches all and blocks 1/3 (also 0.83): the higher wins.
    assert threshold == 0.6 and accuracy == pytest.approx(5 / 6)
    assert best_threshold([(0.4, True)]) == (None, None)


def test_balanced_accuracy_and_auc_need_both_classes():
    assert balanced_accuracy([(True, True), (False, True), (False, False), (False, False)]) == 0.75
    assert balanced_accuracy([(True, True)]) is None
    assert roc_auc([0.9, 0.5], [0.5, 0.1]) == pytest.approx(0.875)  # one tie of four pairs
    assert roc_auc([0.9], []) is None


def summary(label: str, correct: dict[str, int]) -> ex.RunSummary:
    results = [{"case": case, "scores": {"correct": value}, "comments": {"correct": f"note on {case}"}} for case, value in correct.items()]
    return ex.RunSummary(label=label, run_name=label, items=len(results), failed=0, scores={}, results=results)


def test_the_accuracy_per_tag_and_the_misjudged_cases_are_reported_per_run():
    cases = case_file().cases
    runs = [summary("jev · message", {"second": 0, "poem": 1}), summary("jev · full", {"second": 1, "poem": 1})]

    table = tag_table(cases, runs)
    wrong = misjudged(cases, runs)

    assert "| tag | cases | jev · message | jev · full |" in table
    assert "| expect: allow | 1 | 0.00 | 1.00 |" in table and "| off-topic | 1 | 1.00 | 1.00 |" in table
    assert "**jev · message**: 1 wrong:\n- `second` (expect allow): 'what about the second one?' → note on second" in wrong
    assert "**jev · full**: 0 wrong" in wrong


def test_reports_can_add_sections_after_the_table(tmp_path):
    md_path, _ = ex.write_report("guardrail", "Guardrail benchmark", [summary("jev", {"poem": 1})], ["correct"], ["note"],
                                 directory=tmp_path, sections=["## Accuracy per tag\n\n| tag |"])  # fmt: skip

    text = md_path.read_text()
    assert text.index("| run |") < text.index("## Accuracy per tag")
