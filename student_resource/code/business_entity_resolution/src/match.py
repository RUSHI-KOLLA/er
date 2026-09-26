from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


ScoredPairs = Mapping[str, Mapping[str, float]]
CandidateSets = Mapping[str, set[str]]


@dataclass(frozen=True)
class MatchResult:
    predictions: dict[str, set[str]]
    selected_edges: tuple[tuple[str, str, float], ...]
    collision_count: int


def _validate_threshold(threshold: float) -> None:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")


def independent_threshold(
    scored_pairs: ScoredPairs,
    threshold: float,
) -> dict[str, set[str]]:
    _validate_threshold(threshold)
    return {
        source1_id: {
            candidate_id
            for candidate_id, score in candidate_scores.items()
            if score >= threshold
        }
        for source1_id, candidate_scores in scored_pairs.items()
    }


def decide_matches(
    scored_pairs: ScoredPairs,
    candidates: CandidateSets | None = None,
    threshold: float = 0.72,
    enforce_exclusivity: bool = True,
) -> MatchResult:
    _validate_threshold(threshold)
    source1_ids = set(scored_pairs)
    if candidates is not None:
        source1_ids.update(candidates)
    eligible: dict[str, list[tuple[str, float]]] = {}
    for source1_id in sorted(source1_ids):
        allowed = candidates.get(source1_id) if candidates is not None else None
        rows: list[tuple[str, float]] = []
        for candidate_id, score in scored_pairs.get(source1_id, {}).items():
            if score < threshold:
                continue
            if allowed is not None and candidate_id not in allowed:
                continue
            rows.append((candidate_id, float(score)))
        eligible[source1_id] = rows
    predictions = {source1_id: set() for source1_id in sorted(source1_ids)}
    if not enforce_exclusivity:
        for source1_id, rows in eligible.items():
            predictions[source1_id].update(candidate_id for candidate_id, _ in rows)
        selected = tuple(
            (source1_id, candidate_id, score)
            for source1_id in sorted(eligible)
            for candidate_id, score in sorted(eligible[source1_id])
        )
        return MatchResult(predictions, selected, 0)
    claims: dict[str, tuple[str, float]] = {}
    collision_count = 0
    for source1_id in sorted(eligible):
        for candidate_id, score in sorted(eligible[source1_id], key=lambda item: (-item[1], item[0])):
            current = claims.get(candidate_id)
            if current is None:
                claims[candidate_id] = (source1_id, score)
                continue
            collision_count += 1
            current_source1, current_score = current
            if (score, current_source1) > (current_score, source1_id):
                claims[candidate_id] = (source1_id, score)
    for candidate_id, (source1_id, score) in claims.items():
        predictions[source1_id].add(candidate_id)
    selected = tuple(
        sorted(
            (source1_id, candidate_id, score)
            for candidate_id, (source1_id, score) in claims.items()
        )
    )
    return MatchResult(predictions, selected, collision_count)
