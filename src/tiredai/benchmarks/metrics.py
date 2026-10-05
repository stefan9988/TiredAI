"""Ranking metrics (`relevant` is a ranking as booleans, `gains` as graded relevance from 0 to 1), and the
classification metrics of the guardrail benchmark."""

import math
import statistics
from collections.abc import Iterable


def hit_at(relevant: list[bool], k: int) -> float:
    """1 when a relevant result is in the top k."""
    return float(any(relevant[:k]))


def reciprocal_rank(relevant: list[bool]) -> float:
    """1 / rank of the first relevant result, 0 when there is none; its mean over queries is the MRR."""
    return next((1 / rank for rank, hit in enumerate(relevant, start=1) if hit), 0.0)


def precision_at(relevant: list[bool], k: int) -> float:
    """The share of the top k that is relevant; a ranking shorter than k counts the missing ones as misses."""
    return sum(relevant[:k]) / k


def dcg(gains: list[float], k: int) -> float:
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains[:k], start=1))


def ndcg_at(gains: list[float], best_gains: Iterable[float], k: int) -> float:
    """DCG of the top k over the best possible DCG, from the gains of every product that could be returned."""
    best = dcg(sorted(best_gains, reverse=True), k)
    return dcg(gains, k) / best if best else 0.0


def balanced_accuracy(decisions: Iterable[tuple[bool, bool]]) -> float | None:
    """The mean of the share of positives called positive and the share of negatives called negative, from
    (predicted, actual) pairs; None unless both classes are present."""
    pairs = list(decisions)
    positives = [predicted for predicted, actual in pairs if actual]
    negatives = [not predicted for predicted, actual in pairs if not actual]
    if not positives or not negatives:
        return None
    return (sum(positives) / len(positives) + sum(negatives) / len(negatives)) / 2


def roc_auc(positive_scores: list[float], negative_scores: list[float]) -> float | None:
    """The chance that a positive scores higher than a negative (ties count half), at any threshold; None
    unless both are given."""
    if not positive_scores or not negative_scores:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positive_scores for n in negative_scores)
    return wins / (len(positive_scores) * len(negative_scores))


def mean(values: Iterable[float | None]) -> float | None:
    """Mean of the values that are set; None when none are."""
    present = [v for v in values if v is not None]
    return statistics.fmean(present) if present else None


def percentile(values: list[float], share: float) -> float | None:
    """The value below which `share` of the values lie (nearest rank)."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(share * len(ordered)) - 1)]
