"""Guardrail benchmark: how well Jev, TypeSafe's decision model on OpenRouter, tells which shopper messages the
tire assistant should handle, with the policy, questions and block score the app uses (tiredai.guardrail).
A message is blocked when its block score, max(1 - in_scope, manipulation, harmful), reaches the threshold.

A case (benchmarks/guardrail_cases.yaml) is a latest message and its label, allow or block, optionally
after a conversation. Conversations are played through the real agent once and frozen with its answers
(benchmarks/guardrail_conversations.yaml, written by scripts/capture_guardrail_conversations.py). Every
case runs in each VARIANT of how much of the conversation Jev sees:
- message: the latest message alone
- recent: the chat messages the agent itself sees (AGENT_HISTORY_MESSAGES, whole turns), as the app sends them
- full: the whole conversation; Jev reads 32k tokens, so the oldest turns beyond MAX_CONVERSATION_CHARS are dropped

Scores per case, at the threshold: correct; false_block (allow cases: 1 when blocked); caught (block
cases: 1 when blocked). Their means are the accuracy, the false-block rate and the share of bad
messages caught. Per run (run_details): the ROC AUC of the block score, which doesn't depend on the
threshold, and the threshold with the best balanced accuracy.
"""

import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal

import yaml
from langchain_core.messages import AIMessage, HumanMessage
from langfuse import Evaluation
from pydantic import BaseModel, ConfigDict, Field

from tiredai import tracing
from tiredai.agent import aget_transcript, astream_turn, guard_input
from tiredai.benchmarks.experiments import Case, RunSummary, fingerprint, item_value
from tiredai.benchmarks.metrics import balanced_accuracy, mean, percentile, roc_auc
from tiredai.guardrail import QUESTIONS, POLICY, JevClient, JevError, block_score, guard_state, trimmed

DATASET = "tiredai-guardrail"
VARIANTS = ("message", "recent", "full")
SCORES = ["correct", "false_block", "caught"]
SETUP_VERSION = fingerprint(json.dumps([POLICY, QUESTIONS], sort_keys=True))  # recorded with every run


def visible_conversation(conversation: list[dict], message: str, variant: str, history_messages: int) -> list[dict]:
    """The part of the conversation before `message` that Jev sees in `variant`."""
    if variant == "message":
        return []
    if variant == "recent":
        messages = [HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]) for m in conversation]
        return trimmed(guard_input([*messages, HumanMessage(message)], history_messages)[0])
    if variant == "full":
        return trimmed(conversation)
    raise ValueError(f"Unknown variant {variant!r}; one of {', '.join(VARIANTS)}")


# --- Cases -------------------------------------------------------------------------------------------


class Conversation(BaseModel):
    """Shopper messages played through the agent to make the conversation a case's message follows."""

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    turns: list[str] = Field(min_length=1)


class GuardCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    message: str
    expect: Literal["allow", "block"]
    conversation: str | None = None  # the id of the conversation before `message`
    tags: list[str] = Field(default_factory=list)
    description: str | None = None

    def case(self, conversation: list[dict]) -> Case:
        metadata = {"case": self.id, "expect": self.expect, "tags": self.tags, "conversation": self.conversation,
                    "description": self.description}  # fmt: skip
        return Case(
            id=self.id,
            input={"conversation": conversation, "message": self.message},
            expected_output={"expect": self.expect},
            metadata={k: v for k, v in metadata.items() if v is not None},
        )


class CaseFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversations: list[Conversation] = Field(default_factory=list)
    cases: list[GuardCase] = Field(min_length=1)


def _duplicates(ids: list[str]) -> list[str]:
    return sorted({i for i in ids if ids.count(i) > 1})


def load_cases(path: Path) -> CaseFile:
    cases = CaseFile(**(yaml.safe_load(path.read_text()) or {}))
    for kind, ids in (("case", [c.id for c in cases.cases]), ("conversation", [c.id for c in cases.conversations])):
        if duplicates := _duplicates(ids):
            raise ValueError(f"Duplicate {kind} ids: {', '.join(duplicates)}")
    return cases


def load_captured(path: Path) -> dict[str, dict]:
    """The captured conversations by id: {"llm_model", "captured_at", "transcript": [{"role", "content"}, ...]}."""
    if not path.exists():
        return {}
    return {row["id"]: row for row in yaml.safe_load(path.read_text()) or []}


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(str, lambda dumper, text: dumper.represent_scalar("tag:yaml.org,2002:str", text,
                                                                          style="|" if "\n" in text else None))  # fmt: skip


def write_captured(path: Path, captured: dict[str, dict], header: str) -> None:
    rows = [{"id": key, **{k: v for k, v in row.items() if k != "id"}} for key, row in captured.items()]
    path.write_text(f"# {header}\n" + yaml.dump(rows, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=120))


def shopper_turns(row: dict) -> list[str]:
    return [m["content"] for m in row["transcript"] if m["role"] == "user"]


def stale(conversations: list[Conversation], captured: dict[str, dict]) -> list[str]:
    """Ids of the conversations that need capturing: never captured, or whose shopper messages changed since."""
    return [c.id for c in conversations if c.id not in captured or shopper_turns(captured[c.id]) != c.turns]


def check_cases(cases: CaseFile, captured: dict[str, dict]) -> list[str]:
    """What makes the cases unusable: unknown or uncaptured conversations, and conversations no case uses."""
    problems = []
    known = {c.id for c in cases.conversations}
    for case in cases.cases:
        if case.conversation and case.conversation not in known:
            problems.append(f"{case.id}: unknown conversation {case.conversation!r}")
    for conversation_id in stale(cases.conversations, captured):
        problems.append(f"conversation {conversation_id}: not captured with its current turns; run "
                        "scripts/capture_guardrail_conversations.py")  # fmt: skip
    for conversation in cases.conversations:
        row = captured.get(conversation.id)
        if row and [m["role"] for m in row["transcript"]] != ["user", "assistant"] * len(conversation.turns):
            problems.append(f"conversation {conversation.id}: every shopper message needs one captured answer")
    used = {c.conversation for c in cases.cases}
    problems += [f"conversation {c.id}: no case uses it" for c in cases.conversations if c.id not in used]
    return problems


def dataset_cases(cases: CaseFile, captured: dict[str, dict]) -> list[Case]:
    return [c.case(captured[c.conversation]["transcript"] if c.conversation else []) for c in cases.cases]


async def capture_conversation(agent, turns: list[str], thread_id: str, metadata: dict | None = None) -> list[dict]:
    """The shopper messages played through the agent in a new conversation, as {"role", "content"} messages."""
    for message in turns:
        async for _ in astream_turn(agent, message, thread_id, source="benchmark", metadata=metadata):
            pass
    # str(): message text is a str subclass, which the YAML writer doesn't take.
    messages = [{"role": m["role"], "content": str(m["content"])} for m in await aget_transcript(agent, thread_id)]
    if [m["role"] for m in messages] != ["user", "assistant"] * len(turns):
        raise RuntimeError(f"The agent didn't answer every message: {messages}")
    return messages


# --- Running and scoring -----------------------------------------------------------------------------


class GuardTask:
    """Asks Jev about an item's latest message, showing it the conversation as `variant` says."""

    def __init__(self, jev: JevClient, variant: str, history_messages: int):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; one of {', '.join(VARIANTS)}")
        self.jev = jev
        self.variant = variant
        self.history_messages = history_messages

    async def __call__(self, *, item, **kwargs) -> dict:
        data = item_value(item, "input")
        conversation = visible_conversation(data["conversation"], data["message"], self.variant, self.history_messages)
        state = guard_state(conversation, data["message"])
        with tracing.client().start_as_current_observation(
            as_type="guardrail", name="check-message", input=state, metadata={"model": self.jev.model, "variant": self.variant}
        ) as check:
            started = time.perf_counter()
            try:
                decision = await asyncio.to_thread(self.jev.decide, state, QUESTIONS)
            except JevError as exc:
                check.update(level="ERROR", status_message=str(exc))
                raise
            seconds = round(time.perf_counter() - started, 3)
            score = block_score(decision["answers"])
            check.update(output={**decision["answers"], "block_score": score}, metadata={"usage": decision["usage"]})
        return {
            "probabilities": decision["answers"],
            "block_score": round(score, 4),
            "conversation_messages": len(conversation),
            "seconds": seconds,
            "model": decision["model"],
            "input_tokens": decision["usage"].get("input_tokens", 0),
            "cost": decision["usage"].get("cost"),
        }


def guard_evaluator(threshold: float):
    """Blocked means a block score at or above `threshold`."""

    def evaluate_guard(*, output, expected_output, **kwargs) -> list[Evaluation]:
        blocked, should_block = output["block_score"] >= threshold, expected_output["expect"] == "block"
        answers = ", ".join(f"{name} {p:.2f}" for name, p in output["probabilities"].items())
        comment = f"{'blocked' if blocked else 'allowed'}, block score {output['block_score']:.2f} ({answers})"
        return [
            Evaluation(name="correct", value=float(blocked == should_block), comment=comment),
            Evaluation(name="caught" if should_block else "false_block", value=float(blocked), comment=comment),
        ]

    return evaluate_guard


def best_threshold(labeled: list[tuple[float, bool]]) -> tuple[float | None, float | None]:
    """The block-score threshold with the best balanced accuracy on (block score, should block) pairs, and that
    accuracy. Of equally good thresholds, the highest, so the fewest shoppers are blocked."""
    best = (None, None)
    for threshold in sorted({score for score, _ in labeled}, reverse=True):
        accuracy = balanced_accuracy([(score >= threshold, should_block) for score, should_block in labeled])
        if accuracy is not None and (best[1] is None or accuracy > best[1]):
            best = (threshold, accuracy)
    return best


def run_details(result) -> dict:
    labeled, seconds, tokens, cost = [], [], 0, 0.0
    for item_result in result.item_results:
        output = item_result.output
        if not isinstance(output, dict):
            continue
        labeled.append((output["block_score"], item_value(item_result.item, "expected_output")["expect"] == "block"))
        seconds.append(output["seconds"])
        tokens += output["input_tokens"]
        cost += output["cost"] or 0
    threshold, accuracy = best_threshold(labeled)
    return {
        "auc": roc_auc([s for s, block in labeled if block], [s for s, block in labeled if not block]),
        "best_threshold": threshold,
        "best_balanced_accuracy": accuracy,
        "seconds": mean(seconds),
        "p95_seconds": percentile(seconds, 0.95),
        "input_tokens": tokens,
        "cost_usd": round(cost, 8),
    }


def tag_table(cases: list[GuardCase], summaries: list[RunSummary]) -> str:
    """The accuracy of each run (columns) on the cases with each tag (rows), and how many cases have it."""
    tagged = defaultdict(set)
    for case in cases:
        for tag in [f"expect: {case.expect}", *case.tags]:
            tagged[tag].add(case.id)
    correct = [{r["case"]: r["scores"].get("correct") for r in s.results} for s in summaries]
    header = ["tag", "cases", *(s.label for s in summaries)]
    rows = []
    for tag, ids in sorted(tagged.items(), key=lambda kv: (not kv[0].startswith("expect"), kv[0])):
        cells = [mean(scores.get(i) for i in ids) for scores in correct]
        rows.append([tag, str(len(ids)), *("–" if c is None else f"{c:.2f}" for c in cells)])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join([*lines, *("| " + " | ".join(row) + " |" for row in rows)])


def misjudged(cases: list[GuardCase], summaries: list[RunSummary]) -> str:
    """The cases each run got wrong, with Jev's answers, to read what went wrong."""
    by_id = {c.id: c for c in cases}
    lines = []
    for summary in summaries:
        wrong = [r for r in summary.results if r["scores"].get("correct") == 0]
        lines.append(f"**{summary.label}**: {len(wrong)} wrong" + (":" if wrong else ""))
        for r in wrong:
            case = by_id.get(r["case"])
            if case:
                lines.append(f"- `{case.id}` (expect {case.expect}): {case.message!r} → {r['comments'].get('correct', '')}")
    return "\n".join(lines)
