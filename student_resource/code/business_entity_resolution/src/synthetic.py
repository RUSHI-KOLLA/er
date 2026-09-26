from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .block import BlockingIndex
from .features import extract_pair_features
from .io import Record
from .normalize import normalize_record
from .train import ModelBundle
from .features import feature_names


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    reference_name: str
    reference_address: str
    reference_country: str
    candidate_name: str
    candidate_address: str
    candidate_country: str
    should_match: bool


@dataclass(frozen=True)
class SyntheticResult:
    name: str
    passed: bool
    retrieved: bool
    scored: bool
    detail: str


def _case(
    name: str,
    reference_name: str,
    reference_address: str,
    candidate_name: str,
    candidate_address: str,
    should_match: bool,
    reference_country: str = "US",
    candidate_country: str = "US",
) -> SyntheticCase:
    return SyntheticCase(
        name=name,
        reference_name=reference_name,
        reference_address=reference_address,
        reference_country=reference_country,
        candidate_name=candidate_name,
        candidate_address=candidate_address,
        candidate_country=candidate_country,
        should_match=should_match,
    )


SYNTHETIC_CASES = (
    _case("legal_suffix_swap", "Acme Pvt Ltd", "12 Park Rd", "Acme Private Limited", "12 Park Road", True),
    _case("legal_suffix_leading", "LLC Acme", "12 Park Rd", "Acme LLC", "12 Park Road", True),
    _case("dba_variant", "Acme Trading", "12 Park Rd", "Acme DBA", "12 Park Road", True),
    _case("word_order", "Learning Center Moncada", "1 Main Road", "Moncada Learning Center", "1 Main Road", True),
    _case("street_abbreviation", "Acme", "12 Park Road", "Acme", "12 Park Rd", True),
    _case("avenue_abbreviation", "Acme", "12 Park Avenue", "Acme", "12 Park Ave", True),
    _case("missing_postal", "Acme", "12 Park Road, 90210", "Acme", "12 Park Road", True),
    _case("landmark_address", "Acme", "Near SBI ATM, 12 Park Road", "Acme", "12 Park Road, Near SBI ATM", True),
    _case("accent_variant", "Café École", "10 Rue de Paris", "Cafe Ecole", "10 Rue de Paris", True),
    _case(
        "component_reordering",
        "Acme Services",
        "12 Park Road, Suite 4",
        "Suite 4 Acme Services",
        "12 Park Road",
        True,
    ),
    _case("house_number_variant", "Acme", "B-402 Park Road", "Acme", "B-00402 Park Road", True),
    _case("same_name_same_address", "Corner Cafe", "5 Main Road", "Corner Cafe", "5 Main Road", True),
    _case("same_name_different_address", "Corner Cafe", "5 Main Road", "Corner Cafe", "500 Market Street", False),
    _case("same_address_different_name", "Corner Cafe", "5 Main Road", "Unrelated Bakery", "5 Main Road", False),
    _case(
        "near_duplicate_different_business",
        "Acme Dental",
        "12 Park Road",
        "Acme Dental Care",
        "14 Park Road",
        False,
    ),
    _case("different_city", "Acme Dental", "12 Park Road, Boston", "Acme Dental", "12 Park Road, Chicago", False),
    _case("country_mismatch_same_name", "Acme", "12 Park Road", "Acme", "12 Park Road", False, "France", "India"),
    _case(
        "french_street",
        "Ecole Team",
        "175 Boulevard du President Franklin Roosevelt",
        "Ecole Team",
        "175 Boulevard President Franklin Roosevelt",
        True,
        "France",
        "France",
    ),
    _case("missing_address", "Acme", "", "Acme", "", True),
    _case("non_match_missing_address", "Acme", "", "Different Business", "", False),
)


def default_synthetic_model() -> ModelBundle:
    return ModelBundle(model=None, feature_names=feature_names(), kind="heuristic")


def run_synthetic_suite(
    cases: Iterable[SyntheticCase] = SYNTHETIC_CASES,
    threshold: float = 0.72,
) -> tuple[SyntheticResult, ...]:
    model = default_synthetic_model()
    results: list[SyntheticResult] = []
    for case in cases:
        reference = normalize_record(
            Record(
                "S1-1",
                case.reference_name,
                case.reference_address,
                case.reference_country,
            )
        )
        candidate = normalize_record(
            Record(
                "S2-1",
                case.candidate_name,
                case.candidate_address,
                case.candidate_country,
            )
        )
        index = BlockingIndex.from_records((candidate,))
        evidence = index.query(reference, cap=5, country_fallback_limit=2)
        retrieved = bool(evidence)
        score = 0.0
        if evidence:
            score = model.predict_proba(
                [
                    extract_pair_features(
                        reference,
                        candidate,
                        evidence=evidence[0],
                        candidate_count=1,
                    )
                ]
            )[0]
        scored = score >= threshold
        passed = scored == case.should_match
        results.append(
            SyntheticResult(
                name=case.name,
                passed=passed,
                retrieved=retrieved,
                scored=scored,
                detail=f"score={score:.4f}; retrieved={retrieved}",
            )
        )
    return tuple(results)
