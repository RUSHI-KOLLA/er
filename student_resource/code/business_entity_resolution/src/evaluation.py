from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Mapping


IdentifierSet = set[str]
Labels = Mapping[str, IdentifierSet]
Predictions = Mapping[str, Iterable[str]]


@dataclass(frozen=True)
class EntityScore:
    precision: float
    recall: float
    f_beta: float
    true_count: int
    predicted_count: int
    intersection_count: int


@dataclass(frozen=True)
class MetricSummary:
    entity_count: int
    precision: float
    recall: float
    f_beta: float
    true_singletons: int
    predicted_singletons: int
    correct_singletons: int
    missing_prediction_rows: int

    def as_dict(self, beta: float = 0.5) -> dict[str, float | int]:
        return {
            "entity_count": self.entity_count,
            "precision": self.precision,
            "recall": self.recall,
            f"f_{beta}": self.f_beta,
            "true_singletons": self.true_singletons,
            "predicted_singletons": self.predicted_singletons,
            "correct_singletons": self.correct_singletons,
            "missing_prediction_rows": self.missing_prediction_rows,
        }


def _finite_or_zero(value: float) -> float:
    return value if isfinite(value) else 0.0


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    if denominator <= 0.0:
        return 0.0
    return (1.0 + beta_squared) * precision * recall / denominator


def score_entity(predicted: Iterable[str], truth: Iterable[str], beta: float = 0.5) -> EntityScore:
    predicted_set = set(predicted)
    truth_set = set(truth)
    intersection_count = len(predicted_set & truth_set)
    predicted_count = len(predicted_set)
    true_count = len(truth_set)
    if predicted_count == 0 and true_count == 0:
        return EntityScore(1.0, 1.0, 1.0, 0, 0, 0)
    precision = intersection_count / predicted_count if predicted_count else 1.0
    recall = intersection_count / true_count if true_count else 0.0
    return EntityScore(
        precision=_finite_or_zero(precision),
        recall=_finite_or_zero(recall),
        f_beta=f_beta(precision, recall, beta),
        true_count=true_count,
        predicted_count=predicted_count,
        intersection_count=intersection_count,
    )


def macro_f_beta(
    predictions: Predictions,
    truth: Labels,
    beta: float = 0.5,
) -> MetricSummary:
    if not truth:
        return MetricSummary(0, 0.0, 0.0, 0.0, 0, 0, 0, 0)
    scores: list[EntityScore] = []
    true_singletons = 0
    predicted_singletons = 0
    correct_singletons = 0
    missing_prediction_rows = 0
    for source1_id, true_ids in truth.items():
        if source1_id not in predictions:
            missing_prediction_rows += 1
        predicted_ids = predictions.get(source1_id, ())
        predicted_set = set(predicted_ids)
        score = score_entity(predicted_set, true_ids, beta)
        scores.append(score)
        true_empty = not true_ids
        predicted_empty = not predicted_set
        true_singletons += int(true_empty)
        predicted_singletons += int(predicted_empty)
        correct_singletons += int(true_empty and predicted_empty)
    entity_count = len(scores)
    precision = sum(score.precision for score in scores) / entity_count
    recall = sum(score.recall for score in scores) / entity_count
    macro_score = sum(score.f_beta for score in scores) / entity_count
    return MetricSummary(
        entity_count=entity_count,
        precision=_finite_or_zero(precision),
        recall=_finite_or_zero(recall),
        f_beta=_finite_or_zero(macro_score),
        true_singletons=true_singletons,
        predicted_singletons=predicted_singletons,
        correct_singletons=correct_singletons,
        missing_prediction_rows=missing_prediction_rows,
    )


def blocking_recall_at_k(
    candidates: Mapping[str, Iterable[str]],
    truth: Labels,
    k: int | None = None,
) -> dict[str, float | int]:
    if k is not None and k < 0:
        raise ValueError("k must be non-negative")
    total_truth = 0
    recovered = 0
    total_candidates = 0
    truth_entities = 0
    covered_truth_entities = 0
    for source1_id, true_ids in truth.items():
        if not true_ids:
            continue
        truth_entities += 1
        total_truth += len(true_ids)
        candidate_values = list(candidates.get(source1_id, ()))
        if k is not None:
            candidate_values = candidate_values[:k]
        candidate_ids = set(candidate_values)
        total_candidates += len(candidate_ids)
        found = len(candidate_ids & set(true_ids))
        recovered += found
        covered_truth_entities += int(found > 0)
    return {
        "truth_link_count": total_truth,
        "recovered_link_count": recovered,
        "blocking_recall": recovered / total_truth if total_truth else 1.0,
        "truth_entity_count": truth_entities,
        "covered_truth_entity_count": covered_truth_entities,
        "mean_candidate_count": total_candidates / truth_entities if truth_entities else 0.0,
    }


def entity_level_split(
    entity_ids: Iterable[str],
    validation_fraction: float = 0.2,
    seed: int = 2026,
) -> tuple[set[str], set[str]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    training: set[str] = set()
    validation: set[str] = set()
    for entity_id in entity_ids:
        digest = hashlib.blake2b(
            f"{seed}:{entity_id}".encode("utf-8"), digest_size=8
        ).digest()
        bucket = int.from_bytes(digest, "big") / float(1 << 64)
        if bucket < validation_fraction:
            validation.add(entity_id)
        else:
            training.add(entity_id)
    return training, validation


def threshold_predictions(
    scored_pairs: Mapping[str, Mapping[str, float]],
    threshold: float,
) -> dict[str, set[str]]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    return {
        source1_id: {
            candidate_id
            for candidate_id, score in candidate_scores.items()
            if score >= threshold
        }
        for source1_id, candidate_scores in scored_pairs.items()
    }


def iter_confusion(
    predictions: Predictions,
    truth: Labels,
) -> Iterable[tuple[str, str, set[str], set[str]]]:
    for source1_id, true_ids in truth.items():
        predicted_ids = set(predictions.get(source1_id, ()))
        if predicted_ids & set(true_ids):
            kind = "true_positive"
        elif predicted_ids and not true_ids:
            kind = "false_positive"
        elif not predicted_ids and true_ids:
            kind = "false_negative"
        else:
            kind = "correct"
        yield source1_id, kind, predicted_ids, set(true_ids)
