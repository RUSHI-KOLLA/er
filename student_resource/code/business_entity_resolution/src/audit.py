from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Iterable

from .block import SqliteBlockingIndex
from .config import DEFAULT_SETTINGS
from .evaluation import blocking_recall_at_k
from .io import iter_ground_truth, iter_records
from .normalize import normalize_record
from .submit import open_or_build_index


def audit_file(path: Path, source: int, sample_limit: int | None = None) -> dict[str, object]:
    country_counts: Counter[str] = Counter()
    anomaly_counts: Counter[str] = Counter()
    anomaly_examples: list[dict[str, str]] = []
    row_count = 0
    blank_name_count = 0
    blank_address_count = 0
    blank_country_count = 0
    records = iter_records(path, source=source)
    if sample_limit is not None:
        records = islice(records, sample_limit)
    for record in records:
        row_count += 1
        normalized = normalize_record(record)
        country_counts[normalized.country or "<blank>"] += 1
        blank_name_count += int(not normalized.name.clean)
        blank_address_count += int(not normalized.address.clean)
        blank_country_count += int(not normalized.country)
        for flag in normalized.flags:
            anomaly_counts[flag] += 1
        if normalized.flags and len(anomaly_examples) < 20:
            anomaly_examples.append(
                {
                    "entity_id": record.entity_id,
                    "business_name": record.business_name,
                    "flags": ",".join(normalized.flags),
                }
            )
    return {
        "path": str(path),
        "source": source,
        "row_count": row_count,
        "country_counts": dict(sorted(country_counts.items())),
        "blank_name_count": blank_name_count,
        "blank_address_count": blank_address_count,
        "blank_country_count": blank_country_count,
        "anomaly_counts": dict(sorted(anomaly_counts.items())),
        "anomaly_examples": anomaly_examples,
    }


def _bucket(entity_id: str, seed: int) -> float:
    digest = hashlib.blake2b(
        f"{seed}:{entity_id}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def audit_ground_truth(truth_path: Path) -> dict[str, object]:
    row_count = 0
    singleton_count = 0
    link_count = 0
    duplicate_count = 0
    max_links = 0
    seen: set[str] = set()
    link_histogram: Counter[int] = Counter()
    for row in iter_ground_truth(truth_path):
        row_count += 1
        if row.source1_entity_id in seen:
            duplicate_count += 1
        seen.add(row.source1_entity_id)
        links = len(row.matched_entity_ids)
        link_count += links
        max_links = max(max_links, links)
        singleton_count += int(links == 0)
        link_histogram[links] += 1
    return {
        "path": str(truth_path),
        "row_count": row_count,
        "duplicate_source1_rows": duplicate_count,
        "singleton_count": singleton_count,
        "singleton_ratio": singleton_count / row_count if row_count else 0.0,
        "link_count": link_count,
        "mean_links_per_entity": link_count / row_count if row_count else 0.0,
        "max_links_per_entity": max_links,
        "link_histogram": {
            str(bucket): link_histogram[bucket] for bucket in sorted(link_histogram)
        },
    }


def audit_country_sanity(
    source1_path: Path,
    target_paths: Iterable[Path],
    truth_path: Path,
    sample_links: int = 100_000,
    seed: int = 2026,
) -> dict[str, object]:
    gt_stats = audit_ground_truth(truth_path)
    total_links = int(gt_stats["link_count"])
    fraction = min(1.0, sample_links / total_links) if total_links else 1.0
    sampled_pairs: dict[str, str] = {}
    scanned_links = 0
    for row in iter_ground_truth(truth_path):
        for target_id in row.matched_entity_ids:
            scanned_links += 1
            if len(sampled_pairs) >= sample_links:
                break
            if _bucket(target_id, seed) < fraction:
                sampled_pairs.setdefault(target_id, row.source1_entity_id)
        if len(sampled_pairs) >= sample_links:
            break
    sampled_source1 = set(sampled_pairs.values())
    source1_countries: dict[str, str] = {}
    for record in iter_records(source1_path, source=1):
        if record.entity_id in sampled_source1:
            source1_countries[record.entity_id] = record.country.strip().lower()
    target_countries: dict[str, str] = {}
    for target_path in target_paths:
        for record in iter_records(target_path):
            if record.entity_id in sampled_pairs:
                target_countries[record.entity_id] = record.country.strip().lower()
    same_country = 0
    different_country = 0
    blank_target_country = 0
    missing_target = 0
    for target_id, source1_id in sampled_pairs.items():
        target_country = target_countries.get(target_id)
        if target_country is None:
            missing_target += 1
            continue
        if not target_country:
            blank_target_country += 1
            continue
        if target_country == source1_countries.get(source1_id):
            same_country += 1
        else:
            different_country += 1
    return {
        "scanned_link_count": scanned_links,
        "sampled_link_count": len(sampled_pairs),
        "found_target_count": len(target_countries),
        "missing_target_count": missing_target,
        "blank_target_country_count": blank_target_country,
        "same_country_link_count": same_country,
        "different_country_link_count": different_country,
        "same_country_ratio": (
            same_country / (same_country + different_country)
            if (same_country + different_country)
            else 0.0
        ),
        "country_is_rank_signal_not_gate": True,
    }


def _select_validation_ids(
    entity_ids: Iterable[str],
    validation_fraction: float,
    limit: int | None,
    seed: int,
) -> set[str]:
    if not 0.0 < validation_fraction <= 1.0:
        raise ValueError("validation_fraction must be in (0, 1]")
    selected: list[tuple[float, str]] = []
    for entity_id in entity_ids:
        bucket = _bucket(entity_id, seed)
        if bucket >= validation_fraction:
            continue
        selected.append((bucket, entity_id))
        if limit is not None and validation_fraction == 1.0 and len(selected) >= limit:
            break
    if limit is not None and len(selected) > limit:
        selected.sort()
        selected = selected[:limit]
    return {entity_id for _, entity_id in selected}


def measure_blocking_recall(
    source1_path: Path,
    target_paths: Iterable[Path],
    truth_path: Path,
    source1_limit: int | None = None,
    target_limit: int | None = None,
    validation_fraction: float = 0.2,
    candidate_cap: int = 50,
    country_fallback_limit: int = 10,
    stage1: int | None = None,
    seed: int = 2026,
    index_path: Path | None = None,
    country: str | None = None,
    bucket_limit: int = 5000,
) -> dict[str, object]:
    wanted_country = country.strip().lower() if country else None
    scan_ids = (
        record.entity_id
        for record in iter_records(source1_path, source=1)
        if wanted_country is None
        or record.country.strip().lower() == wanted_country
    )
    if source1_limit is not None:
        scan_ids = islice(scan_ids, source1_limit)
    selected_ids = _select_validation_ids(
        scan_ids,
        validation_fraction,
        source1_limit,
        seed,
    )
    truth_subset = {
        row.source1_entity_id: set(row.matched_entity_ids)
        for row in iter_ground_truth(truth_path)
        if row.source1_entity_id in selected_ids
    }
    index = open_or_build_index(
        target_paths,
        index_path=index_path,
        target_limit=target_limit,
        bucket_limit=bucket_limit,
        country=country,
    )
    candidate_map: dict[str, tuple[str, ...]] = {}
    try:
        batch: list[object] = []

        def flush_batch() -> None:
            if not batch:
                return
            evidence_batch = index.query_many(
                batch,
                cap=candidate_cap,
                country_fallback_limit=country_fallback_limit,
                stage1=stage1,
            )
            for reference, candidates in zip(batch, evidence_batch):
                candidate_map[reference.raw.entity_id] = tuple(
                    item.candidate_entity_id for item in candidates
                )
            batch.clear()

        for record in iter_records(source1_path, source=1):
            if record.entity_id not in selected_ids:
                continue
            batch.append(normalize_record(record))
            if len(batch) >= DEFAULT_SETTINGS.query_batch_size:
                flush_batch()
        flush_batch()
    finally:
        index_metadata = (
            index.metadata() if isinstance(index, SqliteBlockingIndex) else {}
        )
        if isinstance(index, SqliteBlockingIndex):
            index.close()
    truth_for_metrics = {
        source1_id: truth_subset.get(source1_id, set()) for source1_id in selected_ids
    }
    recall_20 = blocking_recall_at_k(candidate_map, truth_for_metrics, k=20)
    recall_50 = blocking_recall_at_k(candidate_map, truth_for_metrics, k=50)
    return {
        "selected_entity_count": len(selected_ids),
        "candidate_cap": candidate_cap,
        "country_fallback_limit": country_fallback_limit,
        "stage1": stage1,
        "bucket_limit": bucket_limit,
        "target_limit_per_source": target_limit,
        "country": (country or "all").strip().lower(),
        "truth_link_count": recall_50["truth_link_count"],
        "recall_at_20": recall_20["blocking_recall"],
        "recall_at_50": recall_50["blocking_recall"],
        "mean_candidate_count_at_50": recall_50["mean_candidate_count"],
        "recovered_at_20": recall_20["recovered_link_count"],
        "recovered_at_50": recall_50["recovered_link_count"],
        "index_metadata": index_metadata,
    }


def write_audit_report(
    output_path: Path,
    train_reports: Iterable[dict[str, object]],
    test_reports: Iterable[dict[str, object]],
    extra: dict[str, object] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"train": list(train_reports), "test": list(test_reports)}
    if extra:
        report.update(extra)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _run_audit_command(arguments: argparse.Namespace) -> int:
    from .config import Settings

    settings = Settings(data_root=Path(arguments.data_root))
    split = arguments.split
    train_reports = (
        [
            audit_file(settings.source_path("train", source), source, arguments.sample_limit)
            for source in (1, 2, 3)
        ]
        if split in {"train", "both"}
        else []
    )
    test_reports = (
        [
            audit_file(settings.source_path("test", source), source, arguments.sample_limit)
            for source in (1, 2, 3)
        ]
        if split in {"test", "both"}
        else []
    )
    extra: dict[str, object] = {}
    if split in {"train", "both"}:
        extra["ground_truth"] = audit_ground_truth(settings.train_ground_truth_path)
    report_path = Path(arguments.output)
    write_audit_report(report_path, train_reports, test_reports, extra)
    print(
        json.dumps(
            {"train": train_reports, "test": test_reports, **extra},
            indent=2,
        )
    )
    return 0


def _run_sanity_command(arguments: argparse.Namespace) -> int:
    from .config import Settings

    settings = Settings(data_root=Path(arguments.data_root))
    report = {
        "ground_truth": audit_ground_truth(settings.train_ground_truth_path),
        "country_sanity": audit_country_sanity(
            source1_path=settings.train_source1_path,
            target_paths=[settings.source_path("train", 2), settings.source_path("train", 3)],
            truth_path=settings.train_ground_truth_path,
            sample_links=arguments.sample_links,
            seed=settings.seed,
        ),
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _run_blocking_command(arguments: argparse.Namespace) -> int:
    from .config import Settings

    settings = Settings(data_root=Path(arguments.data_root))
    report = measure_blocking_recall(
        source1_path=settings.train_source1_path,
        target_paths=[settings.source_path("train", 2), settings.source_path("train", 3)],
        truth_path=settings.train_ground_truth_path,
        source1_limit=arguments.source1_limit,
        target_limit=arguments.target_limit,
        validation_fraction=arguments.validation_fraction,
        candidate_cap=arguments.candidate_cap,
        country_fallback_limit=arguments.country_fallback_limit,
        stage1=arguments.stage1 if arguments.stage1 > 0 else None,
        bucket_limit=arguments.bucket_limit,
        index_path=(
            Path(arguments.index_db)
            if arguments.index_db
            else Path(arguments.output).parent / "train_blocking_index.sqlite"
        ),
        country=arguments.country,
    )
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit data and blocking recall")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("dataset")
    audit_parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[3] / "dataset"))
    audit_parser.add_argument(
        "--split", choices=("train", "test", "both"), default="both"
    )
    audit_parser.add_argument("--sample-limit", type=int, default=None)
    audit_parser.add_argument("--output", default="artifacts/eda.json")
    audit_parser.set_defaults(handler=_run_audit_command)
    sanity_parser = subparsers.add_parser("sanity")
    sanity_parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[3] / "dataset"))
    sanity_parser.add_argument("--sample-links", type=int, default=100_000)
    sanity_parser.add_argument("--output", default="artifacts/sanity.json")
    sanity_parser.set_defaults(handler=_run_sanity_command)
    blocking_parser = subparsers.add_parser("blocking")
    blocking_parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[3] / "dataset"))
    blocking_parser.add_argument("--source1-limit", type=int, default=20_000)
    blocking_parser.add_argument("--target-limit", type=int, default=None)
    blocking_parser.add_argument("--validation-fraction", type=float, default=0.2)
    blocking_parser.add_argument("--candidate-cap", type=int, default=50)
    blocking_parser.add_argument("--country-fallback-limit", type=int, default=10)
    blocking_parser.add_argument("--stage1", type=int, default=DEFAULT_SETTINGS.blocking_stage1)
    blocking_parser.add_argument("--bucket-limit", type=int, default=DEFAULT_SETTINGS.bucket_limit)
    blocking_parser.add_argument("--index-db", default=None)
    blocking_parser.add_argument("--country", default=None, help="restrict index and queries to one country")
    blocking_parser.add_argument("--output", default="artifacts/blocking_recall.json")
    blocking_parser.set_defaults(handler=_run_blocking_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
