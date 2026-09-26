from __future__ import annotations

import math
import pickle
import random
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

from .evaluation import (
    MetricSummary,
    blocking_recall_at_k,
    macro_f_beta,
    threshold_predictions,
)
from .block import CandidateEvidence
from .config import DEFAULT_SETTINGS
from .features import extract_pair_features, feature_names
from .io import iter_ground_truth, iter_records
from .match import decide_matches
from .normalize import NormalizedRecord, normalize_record


@dataclass(frozen=True)
class TrainingExample:
    source1_entity_id: str
    candidate_entity_id: str
    label: int
    features: dict[str, float]


TargetLookup = Mapping[str, NormalizedRecord] | Callable[[str], NormalizedRecord | None]


@dataclass
class ModelBundle:
    model: object | None
    feature_names: tuple[str, ...]
    kind: str
    threshold: float = 0.72
    constant: float | None = None

    def predict_proba(self, feature_rows: Iterable[Mapping[str, float]]) -> list[float]:
        rows = [
            [float(features.get(name, 0.0)) for name in self.feature_names]
            for features in feature_rows
        ]
        if not rows:
            return []
        if self.model is not None:
            probabilities = self.model.predict_proba(rows)
            return [float(row[1] if len(row) > 1 else row[0]) for row in probabilities]
        if self.constant is not None:
            return [self.constant] * len(rows)
        return [_heuristic_probability(row, self.feature_names) for row in rows]


def _heuristic_probability(values: list[float], names: tuple[str, ...]) -> float:
    features = dict(zip(names, values))
    raw_score = (
        -5.0
        + 4.5 * features.get("name_core_exact", 0.0)
        + 2.5 * features.get("name_exact", 0.0)
        + 3.5 * features.get("name_token_jaccard", 0.0)
        + 1.2 * features.get("name_token_sort_exact", 0.0)
        + 2.0 * features.get("address_exact", 0.0)
        + 3.0 * features.get("address_token_jaccard", 0.0)
        + 1.2 * features.get("address_postal_exact", 0.0)
        + 1.5 * features.get("address_house_exact", 0.0)
        + 0.8 * features.get("address_city_exact", 0.0)
        + 0.8 * features.get("country_exact", 0.0)
        - 12.0 * features.get("address_conflict", 0.0)
        - 18.0 * features.get("city_conflict", 0.0)
        - 14.0 * features.get("name_address_name_conflict", 0.0)
        - 20.0 * features.get("country_conflict", 0.0)
        - 6.0 * features.get("name_address_disagreement", 0.0)
        + 0.6 * features.get("address_sequence_ratio", 0.0)
        + 0.2 * features.get("name_sequence_ratio", 0.0)
        + 0.2 * features.get("retrieval_score", 0.0)
        - 0.08 * features.get("candidate_count", 0.0)
    )
    probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, raw_score))))
    return float(min(1.0, max(0.0, probability)))


def _lookup_target(target_lookup: TargetLookup, entity_id: str) -> NormalizedRecord | None:
    if isinstance(target_lookup, Mapping):
        return target_lookup.get(entity_id)
    return target_lookup(entity_id)


def build_training_examples(
    reference_records: Iterable[NormalizedRecord],
    candidates: Mapping[str, Iterable[object]],
    target_lookup: TargetLookup,
    truth: Mapping[str, set[str]],
    max_pairs: int = 500_000,
    seed: int = 2026,
    name_frequency: Mapping[str, int] | None = None,
) -> list[TrainingExample]:
    if max_pairs < 1:
        raise ValueError("max_pairs must be positive")
    positive_limit = max(1, max_pairs // 2)
    negative_limit = max(1, max_pairs - positive_limit)
    limits = {1: positive_limit, 0: negative_limit}
    reservoirs: dict[int, list[TrainingExample]] = {0: [], 1: []}
    seen = {0: 0, 1: 0}
    generator = random.Random(seed)
    for reference in reference_records:
        source1_id = reference.raw.entity_id
        evidence_rows = tuple(candidates.get(source1_id, ()))
        ordered = sorted(
            evidence_rows,
            key=lambda item: (-float(item.score), item.candidate_entity_id),
        )
        runner_up_score = float(ordered[1].score) if len(ordered) > 1 else 0.0
        for evidence in ordered:
            target = _lookup_target(target_lookup, evidence.candidate_entity_id)
            if target is None:
                continue
            label = int(evidence.candidate_entity_id in truth.get(source1_id, set()))
            features = extract_pair_features(
                reference,
                target,
                evidence=evidence,
                candidate_count=len(ordered),
                runner_up_score=runner_up_score,
                name_frequency=name_frequency,
            )
            example = TrainingExample(
                source1_entity_id=source1_id,
                candidate_entity_id=evidence.candidate_entity_id,
                label=label,
                features=features,
            )
            reservoir = reservoirs[label]
            if len(reservoir) < limits[label]:
                reservoir.append(example)
            else:
                replacement = generator.randrange(seen[label] + 1)
                if replacement < limits[label]:
                    reservoir[replacement] = example
            seen[label] += 1
    examples = reservoirs[0] + reservoirs[1]
    examples.sort(key=lambda row: (row.source1_entity_id, row.candidate_entity_id))
    return examples


def fit_model(
    examples: Iterable[TrainingExample],
    seed: int = 2026,
    max_iterations: int = 120,
) -> ModelBundle:
    rows = list(examples)
    if not rows:
        raise ValueError("at least one training example is required")
    names = feature_names()
    labels = {row.label for row in rows}
    if len(labels) == 1:
        constant = float(next(iter(labels)))
        return ModelBundle(
            model=None,
            feature_names=names,
            kind="constant",
            constant=constant,
        )
    try:
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingClassifier

        matrix = np.asarray(
            [
                [row.features.get(name, 0.0) for name in names]
                for row in rows
            ],
            dtype="float32",
        )
        target = np.asarray([row.label for row in rows], dtype="int8")
        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=max_iterations,
            max_leaf_nodes=31,
            min_samples_leaf=2,
            l2_regularization=1.0,
            random_state=seed,
        )
        model.fit(matrix, target)
        return ModelBundle(model=model, feature_names=names, kind="hist_gradient_boosting")
    except ImportError:
        return ModelBundle(model=None, feature_names=names, kind="heuristic")


def select_threshold(
    scored_pairs: Mapping[str, Mapping[str, float]],
    truth: Mapping[str, set[str]],
    beta: float = 0.5,
    candidates: Mapping[str, set[str]] | None = None,
    enforce_exclusivity: bool = True,
) -> tuple[float, MetricSummary]:
    score_count = sum(len(values) for values in scored_pairs.values())
    thresholds = {index / 100.0 for index in range(101)}
    if score_count <= 10_000:
        for candidate_scores in scored_pairs.values():
            thresholds.update(
                score for score in candidate_scores.values() if 0.0 <= score <= 1.0
            )
    if candidates is None:
        candidates = {
            source1_id: set(candidate_scores)
            for source1_id, candidate_scores in scored_pairs.items()
        }
    best_threshold = 1.0
    best_summary = macro_f_beta({}, truth, beta=beta)
    for threshold in sorted(thresholds):
        if enforce_exclusivity:
            decision = decide_matches(
                scored_pairs,
                candidates=candidates,
                threshold=threshold,
                enforce_exclusivity=True,
            ).predictions
        else:
            decision = threshold_predictions(scored_pairs, threshold)
        summary = macro_f_beta(decision, truth, beta=beta)
        if (summary.f_beta, threshold) > (best_summary.f_beta, best_threshold):
            best_threshold = threshold
            best_summary = summary
    return best_threshold, best_summary


def save_model(model: ModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: Path) -> ModelBundle:
    with Path(path).open("rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, ModelBundle):
        raise ValueError(f"unexpected model artifact: {path}")
    if model.feature_names != feature_names():
        raise ValueError(f"model feature schema does not match this pipeline: {path}")
    return model


@dataclass(frozen=True)
class TrainingSummary:
    model_kind: str
    threshold: float
    training_entity_count: int
    validation_entity_count: int
    positive_training_examples: int
    negative_training_examples: int
    validation_precision: float
    validation_recall: float
    validation_f_beta: float
    blocking_recall_at_20: float
    blocking_recall_at_50: float


def _truth_for_ids(truth_path: Path, entity_ids: set[str]) -> dict[str, set[str]]:
    return {
        row.source1_entity_id: set(row.matched_entity_ids)
        for row in iter_ground_truth(truth_path)
        if row.source1_entity_id in entity_ids
    }


def _normalized_for_ids(reference_path: Path, entity_ids: set[str]):
    for record in iter_records(reference_path, source=1):
        if record.entity_id in entity_ids:
            yield normalize_record(record)


def _index_lookup(target_index):
    if hasattr(target_index, "get"):
        @lru_cache(maxsize=100_000)
        def lookup(entity_id: str):
            return target_index.get(entity_id)
        return lookup
    return target_index.targets.get


def train_from_index(
    reference_path: Path,
    truth_path: Path,
    target_index,
    model_path: Path,
    training_ids: set[str],
    validation_ids: set[str],
    max_pairs: int = 500_000,
    candidate_cap: int = 50,
    country_fallback_limit: int = 10,
    stage1: int | None = None,
    seed: int = 2026,
    max_training_entities: int = DEFAULT_SETTINGS.max_training_entities,
) -> TrainingSummary:
    if not training_ids:
        raise ValueError("training_ids must not be empty")
    truth = _truth_for_ids(truth_path, training_ids | validation_ids)
    target_lookup = _index_lookup(target_index)
    training_records = tuple(_normalized_for_ids(reference_path, training_ids))
    if len(training_records) > max_training_entities:
        raise MemoryError(
            "training candidate map exceeded its safety limit; lower the entity limit"
        )
    query_batch_size = DEFAULT_SETTINGS.query_batch_size
    evidence_chunks: list[tuple[CandidateEvidence, ...]] = []
    for start in range(0, len(training_records), query_batch_size):
        evidence_chunks.extend(
            target_index.query_many(
                training_records[start : start + query_batch_size],
                cap=candidate_cap,
                country_fallback_limit=country_fallback_limit,
                stage1=stage1,
            )
        )
    training_evidence = tuple(evidence_chunks)
    training_candidates = {
        reference.raw.entity_id: evidence
        for reference, evidence in zip(training_records, training_evidence)
    }
    training_examples = build_training_examples(
        training_records,
        training_candidates,
        target_lookup,
        truth,
        max_pairs=max_pairs,
        seed=seed,
    )
    if not training_examples:
        raise ValueError("blocking produced no usable training examples")
    model = fit_model(training_examples, seed=seed)
    validation_scores: dict[str, dict[str, float]] = {}
    validation_candidates: dict[str, tuple[str, ...]] = {}
    for reference in _normalized_for_ids(reference_path, validation_ids):
        evidence_rows = target_index.query(
            reference,
            cap=candidate_cap,
            country_fallback_limit=country_fallback_limit,
            stage1=stage1,
        )
        validation_candidates[reference.raw.entity_id] = tuple(
            evidence.candidate_entity_id for evidence in evidence_rows
        )
        if not evidence_rows:
            validation_scores[reference.raw.entity_id] = {}
            continue
        features = []
        runner_up_score = float(evidence_rows[1].score) if len(evidence_rows) > 1 else 0.0
        for evidence in evidence_rows:
            target = target_lookup(evidence.candidate_entity_id)
            if target is None:
                continue
            features.append(
                extract_pair_features(
                    reference,
                    target,
                    evidence=evidence,
                    candidate_count=len(evidence_rows),
                    runner_up_score=runner_up_score,
                )
            )
        probabilities = model.predict_proba(features)
        validation_scores[reference.raw.entity_id] = {
            evidence.candidate_entity_id: float(probability)
            for evidence, probability in zip(evidence_rows, probabilities)
        }
    validation_truth = {
        source1_id: truth.get(source1_id, set()) for source1_id in validation_ids
    }
    threshold, validation_summary = select_threshold(
        validation_scores,
        validation_truth,
        candidates=validation_candidates,
        enforce_exclusivity=True,
    )
    blocking_20 = blocking_recall_at_k(validation_candidates, validation_truth, k=20)
    blocking_50 = blocking_recall_at_k(validation_candidates, validation_truth, k=50)
    model.threshold = threshold
    save_model(model, model_path)
    return TrainingSummary(
        model_kind=model.kind,
        threshold=threshold,
        training_entity_count=len(training_ids),
        validation_entity_count=len(validation_ids),
        positive_training_examples=sum(row.label == 1 for row in training_examples),
        negative_training_examples=sum(row.label == 0 for row in training_examples),
        validation_precision=validation_summary.precision,
        validation_recall=validation_summary.recall,
        validation_f_beta=validation_summary.f_beta,
        blocking_recall_at_20=float(blocking_20["blocking_recall"]),
        blocking_recall_at_50=float(blocking_50["blocking_recall"]),
    )
