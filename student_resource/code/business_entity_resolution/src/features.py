from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Mapping

from .block import CandidateEvidence
from .normalize import NormalizedRecord


_NUMBER_RE = re.compile(r"\d+")
_FEATURE_NAMES = (
    "name_exact",
    "name_core_exact",
    "name_token_sort_exact",
    "name_token_jaccard",
    "name_levenshtein",
    "name_sequence_ratio",
    "name_char_ngram_cosine",
    "shared_rare_token_count",
    "rare_name_token_score",
    "address_exact",
    "address_postal_exact",
    "address_house_exact",
    "address_city_exact",
    "address_token_jaccard",
    "address_levenshtein",
    "address_sequence_ratio",
    "address_char_ngram_cosine",
    "numeric_overlap",
    "country_exact",
    "country_conflict",
    "address_conflict",
    "city_conflict",
    "name_address_name_conflict",
    "name_address_disagreement",
    "cross_postal_name_similarity",
    "field_agreement_count",
    "missing_name",
    "missing_address",
    "missing_country",
    "candidate_count",
    "retrieval_score",
    "retrieval_rank",
    "retrieval_method_count",
    "score_gap",
)


def feature_names() -> tuple[str, ...]:
    return _FEATURE_NAMES


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    value = numerator / denominator
    return value if math.isfinite(value) else 0.0


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(value.split()) if value else ()


def _jaccard(left: str, right: str) -> float:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    if not left_tokens and not right_tokens:
        return 0.0
    return _safe_ratio(len(left_tokens & right_tokens), len(left_tokens | right_tokens))


def _levenshtein_distance(left: str, right: str, maximum_length: int = 256) -> int:
    left = left[:maximum_length]
    right = right[:maximum_length]
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    start = 0
    shared_limit = min(len(left), len(right))
    while start < shared_limit and left[start] == right[start]:
        start += 1
    left_end = len(left)
    right_end = len(right)
    while (
        left_end > start
        and right_end > start
        and left[left_end - 1] == right[right_end - 1]
    ):
        left_end -= 1
        right_end -= 1
    left = left[start:left_end]
    right = right[start:right_end]
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        row_append = current.append
        previous_row = previous
        above = left_index
        right_index = 1
        for right_character in right:
            insert = above + 1
            delete = previous_row[right_index] + 1
            substitute = previous_row[right_index - 1] + (
                left_character != right_character
            )
            value = insert if insert < delete else delete
            if substitute < value:
                value = substitute
            row_append(value)
            above = value
            right_index += 1
        previous = current
    return previous[-1]


def _edit_similarity(left: str, right: str) -> float:
    if not left and not right:
        return 0.0
    if not left or not right:
        return 0.0
    return 1.0 - _levenshtein_distance(left, right) / max(len(left), len(right))


def _sequence_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def _ngrams(value: str, size: int = 3) -> Counter[str]:
    if not value:
        return Counter()
    if len(value) < size:
        return Counter((value,))
    return Counter(value[index : index + size] for index in range(len(value) - size + 1))


def _cosine(left: str, right: str) -> float:
    left_counts = _ngrams(left)
    right_counts = _ngrams(right)
    if not left_counts or not right_counts:
        return 0.0
    numerator = sum(value * right_counts.get(key, 0) for key, value in left_counts.items())
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    return _safe_ratio(numerator, left_norm * right_norm)


def _numbers(value: str) -> set[str]:
    return set(_NUMBER_RE.findall(value))


def _number_jaccard(left: str, right: str) -> float:
    left_numbers = _numbers(left)
    right_numbers = _numbers(right)
    if not left_numbers and not right_numbers:
        return 0.0
    return _safe_ratio(len(left_numbers & right_numbers), len(left_numbers | right_numbers))


def _exact(left: str | None, right: str | None) -> float:
    return float(bool(left) and bool(right) and left == right)


def extract_pair_features(
    reference: NormalizedRecord,
    candidate: NormalizedRecord,
    evidence: CandidateEvidence | None = None,
    candidate_count: int = 1,
    runner_up_score: float = 0.0,
    name_frequency: Mapping[str, int] | None = None,
) -> dict[str, float]:
    name_frequency = name_frequency or {}
    reference_name = reference.name.core_folded or reference.name.folded
    candidate_name = candidate.name.core_folded or candidate.name.folded
    reference_address = reference.address.folded
    candidate_address = candidate.address.folded
    reference_tokens = set(_tokens(reference_name))
    candidate_tokens = set(_tokens(candidate_name))
    shared_tokens = reference_tokens & candidate_tokens
    rare_shared = sum(
        1 for token in shared_tokens if name_frequency.get(token, 1) <= 5
    )
    rare_score = sum(
        1.0 / max(name_frequency.get(token, 1), 1) for token in shared_tokens
    )
    rare_score /= max(1, min(len(reference_tokens), len(candidate_tokens)))
    name_core_exact = _exact(reference.name.core_folded, candidate.name.core_folded)
    address_exact = _exact(reference_address, candidate_address)
    address_token_jaccard = _jaccard(reference_address, candidate_address)
    postal_exact = _exact(reference.address.postal_code, candidate.address.postal_code)
    house_exact = _exact(reference.address.house_number, candidate.address.house_number)
    city_exact = _exact(reference.address.city, candidate.address.city)
    country_exact = _exact(reference.country, candidate.country)
    address_present = bool(reference.address.clean and candidate.address.clean)
    address_conflict = float(
        address_present and not address_exact and address_token_jaccard < 0.25
    )
    city_conflict = float(
        bool(reference.address.city and candidate.address.city)
        and reference.address.city != candidate.address.city
        and address_token_jaccard < 0.75
    )
    name_address_name_conflict = float(
        address_exact
        and bool(reference_name and candidate_name)
        and _jaccard(reference_name, candidate_name) < 0.2
    )
    country_conflict = float(
        bool(reference.country and candidate.country)
        and reference.country != candidate.country
    )
    name_address_disagreement = float(
        name_core_exact and address_present and address_token_jaccard < 0.5
    )
    features = {
        "name_exact": _exact(reference.name.folded, candidate.name.folded),
        "name_core_exact": name_core_exact,
        "name_token_sort_exact": _exact(reference.name.token_sort, candidate.name.token_sort),
        "name_token_jaccard": _jaccard(reference_name, candidate_name),
        "name_levenshtein": _edit_similarity(reference_name, candidate_name),
        "name_sequence_ratio": _sequence_ratio(reference_name, candidate_name),
        "name_char_ngram_cosine": _cosine(reference_name, candidate_name),
        "shared_rare_token_count": float(rare_shared),
        "rare_name_token_score": rare_score,
        "address_exact": address_exact,
        "address_postal_exact": postal_exact,
        "address_house_exact": house_exact,
        "address_city_exact": city_exact,
        "address_token_jaccard": address_token_jaccard,
        "address_levenshtein": _edit_similarity(reference_address, candidate_address),
        "address_sequence_ratio": _sequence_ratio(reference_address, candidate_address),
        "address_char_ngram_cosine": _cosine(reference_address, candidate_address),
        "numeric_overlap": _number_jaccard(reference_address, candidate_address),
        "country_exact": country_exact,
        "country_conflict": country_conflict,
        "address_conflict": address_conflict,
        "city_conflict": city_conflict,
        "name_address_name_conflict": name_address_name_conflict,
        "name_address_disagreement": name_address_disagreement,
        "cross_postal_name_similarity": (
            _jaccard(reference_name, candidate_name) if postal_exact else 0.0
        ),
        "field_agreement_count": float(
            name_core_exact + address_exact + postal_exact + house_exact + country_exact
        ),
        "missing_name": float(not reference.name.clean or not candidate.name.clean),
        "missing_address": float(
            not reference.address.clean or not candidate.address.clean
        ),
        "missing_country": float(not reference.country or not candidate.country),
        "candidate_count": float(max(candidate_count, 0)),
        "retrieval_score": float(evidence.score if evidence else 0.0),
        "retrieval_rank": float(evidence.best_rank if evidence else 0.0),
        "retrieval_method_count": float(len(evidence.methods) if evidence else 0.0),
        "score_gap": float((evidence.score if evidence else 0.0) - runner_up_score),
    }
    return {name: float(value) for name, value in features.items()}
