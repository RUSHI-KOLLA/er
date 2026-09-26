from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import asdict, dataclass
import json
from itertools import chain, islice
import multiprocessing
from pathlib import Path
import shutil
import sqlite3
from typing import Iterable
import zlib

from .block import BlockingIndex, CandidateEvidence, SqliteBlockingIndex
from .config import DEFAULT_SETTINGS, Settings
from .evaluation import entity_level_split
from .features import extract_pair_features, feature_names
from .io import (
    CANDIDATE_HEADER,
    MATCHING_HEADER,
    iter_records,
    write_candidate_pairs,
    write_matching_results,
)
from .match import MatchResult, decide_matches
from .normalize import NormalizedRecord, normalize_record
from .train import ModelBundle, load_model, train_from_index


@dataclass(frozen=True)
class PipelineSummary:
    source1_count: int
    candidate_edge_count: int
    matched_edge_count: int
    collision_count: int
    complete: bool = True


TargetIndex = BlockingIndex | SqliteBlockingIndex


def _iter_normalized(
    path: Path,
    source: int,
    limit: int | None = None,
    country: str | None = None,
    shard: tuple[int, int] | None = None,
) -> Iterable[NormalizedRecord]:
    records = iter_records(path, source=source)
    if limit is not None:
        records = islice(records, limit)
    if shard is not None:
        shard_index, shard_count = shard
        records = (
            record
            for record in records
            if zlib.crc32(record.entity_id.encode("utf-8")) % shard_count == shard_index
        )
    if country is None:
        return (normalize_record(record) for record in records)
    wanted = country.strip().lower()
    return (
        normalized
        for record in records
        if (normalized := normalize_record(record)).country == wanted
    )


def _batched(rows: Iterable[object], batch_size: int) -> Iterable[tuple[object, ...]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    batch: list[object] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield tuple(batch)
            batch = []
    if batch:
        yield tuple(batch)


def _stage1_argument(value: int) -> int | None:
    return value if value > 0 else None


def _iter_queries(
    index: TargetIndex,
    records: Iterable[NormalizedRecord],
    cap: int,
    country_fallback_limit: int,
    batch_size: int = DEFAULT_SETTINGS.query_batch_size,
    stage1: int | None = None,
) -> Iterable[
    tuple[
        int,
        NormalizedRecord,
        tuple[CandidateEvidence, ...],
        dict[str, NormalizedRecord],
    ]
]:
    ordinal = 0
    for batch in _batched(records, batch_size):
        evidence_batch, target_map = index.query_many_with_targets(
            batch,
            cap=cap,
            country_fallback_limit=country_fallback_limit,
            stage1=stage1,
        )
        for reference, evidence_rows in zip(batch, evidence_batch):
            yield ordinal, reference, evidence_rows, target_map
            ordinal += 1


def _target_record(
    index: TargetIndex,
    entity_id: str,
) -> NormalizedRecord | None:
    if isinstance(index, BlockingIndex):
        return index.targets.get(entity_id)
    return index.get(entity_id)


def _target_records(
    index: TargetIndex,
    entity_ids: Iterable[str],
) -> dict[str, NormalizedRecord]:
    ids = tuple(dict.fromkeys(entity_ids))
    if isinstance(index, BlockingIndex):
        return {entity_id: index.targets[entity_id] for entity_id in ids if entity_id in index.targets}
    return index.get_many(ids)


def build_target_index(
    target_paths: Iterable[Path],
    index_path: Path | None = None,
    limit_per_source: int | None = None,
    bucket_limit: int = 5000,
    max_in_memory_targets: int = DEFAULT_SETTINGS.max_in_memory_targets,
    country: str | None = None,
) -> TargetIndex:
    paths = tuple(target_paths)
    records = chain.from_iterable(
        _iter_normalized(path, source=index + 2, limit=limit_per_source, country=country)
        for index, path in enumerate(paths)
    )
    if index_path is not None:
        return SqliteBlockingIndex.build(
            records,
            index_path,
            bucket_limit=bucket_limit,
            overwrite=True,
            metadata={
                "target_paths": json.dumps([str(path) for path in paths]),
                "target_limit": "all" if limit_per_source is None else limit_per_source,
                "bucket_limit": bucket_limit,
                "country": country.strip().lower() if country else "all",
            },
        )

    def bounded_records():
        count = 0
        for record in records:
            count += 1
            if count > max_in_memory_targets:
                raise MemoryError(
                    "in-memory target index exceeded its safety limit; "
                    "provide index_path or lower max_in_memory_targets"
                )
            yield record

    return BlockingIndex.from_records(
        bounded_records(), bucket_limit=bucket_limit
    )


def _default_model() -> ModelBundle:
    return ModelBundle(model=None, feature_names=feature_names(), kind="heuristic")


def run_inference(
    source1_path: Path,
    target_index: TargetIndex,
    output_dir: Path,
    model: ModelBundle | None = None,
    threshold: float = 0.72,
    candidate_cap: int = 50,
    country_fallback_limit: int = 10,
    source1_limit: int | None = None,
    enforce_exclusivity: bool = True,
    max_in_memory_edges: int = DEFAULT_SETTINGS.max_in_memory_edges,
) -> PipelineSummary:
    if candidate_cap < 1:
        raise ValueError("candidate_cap must be positive")
    if max_in_memory_edges < 1:
        raise ValueError("max_in_memory_edges must be positive")
    model = model or _default_model()
    all_candidates: dict[str, set[str]] = {}
    all_scores: dict[str, dict[str, float]] = {}
    source1_count = 0
    candidate_edge_count = 0
    for reference in _iter_normalized(source1_path, source=1, limit=source1_limit):
        source1_id = reference.raw.entity_id
        evidence_rows = target_index.query(
            reference,
            cap=candidate_cap,
            country_fallback_limit=country_fallback_limit,
        )
        candidate_ids = {item.candidate_entity_id for item in evidence_rows}
        all_candidates[source1_id] = candidate_ids
        candidate_edge_count += len(candidate_ids)
        if candidate_edge_count > max_in_memory_edges:
            raise MemoryError(
                "in-memory inference exceeded its safety limit; use the streaming path"
            )
        if not evidence_rows:
            all_scores[source1_id] = {}
            source1_count += 1
            continue
        feature_rows: list[dict[str, float]] = []
        valid_evidence = []
        runner_up_score = float(evidence_rows[1].score) if len(evidence_rows) > 1 else 0.0
        for evidence in evidence_rows:
            candidate = _target_record(target_index, evidence.candidate_entity_id)
            if candidate is None:
                raise RuntimeError(
                    f"candidate {evidence.candidate_entity_id} is absent from the target index"
                )
            valid_evidence.append(evidence)
            feature_rows.append(
                extract_pair_features(
                    reference,
                    candidate,
                    evidence=evidence,
                    candidate_count=len(evidence_rows),
                    runner_up_score=runner_up_score,
                )
            )
        probabilities = model.predict_proba(feature_rows)
        all_scores[source1_id] = {
            evidence.candidate_entity_id: float(probability)
            for evidence, probability in zip(valid_evidence, probabilities)
        }
        source1_count += 1
    match_result: MatchResult = decide_matches(
        all_scores,
        candidates=all_candidates,
        threshold=threshold,
        enforce_exclusivity=enforce_exclusivity,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_candidate_pairs(
        output_dir / "candidate_pairs.tsv",
        ((source1_id, all_candidates.get(source1_id, set())) for source1_id in all_candidates),
    )
    write_matching_results(
        output_dir / "matching_results.tsv",
        (
            (source1_id, match_result.predictions.get(source1_id, set()))
            for source1_id in all_candidates
        ),
    )
    return PipelineSummary(
        source1_count=source1_count,
        candidate_edge_count=candidate_edge_count,
        matched_edge_count=sum(len(ids) for ids in match_result.predictions.values()),
        collision_count=match_result.collision_count,
        complete=source1_limit is None,
    )


def _write_streaming_matches(
    connection: sqlite3.Connection,
    path: Path,
    enforce_exclusivity: bool,
) -> None:
    table = "claims" if enforce_exclusivity else "qualified_pairs"
    edge_rows = iter(
        connection.execute(
            f"""
            SELECT o.ordinal, e.source1_id, e.candidate_id
            FROM {table} AS e
            JOIN source1_order AS o ON o.source1_id = e.source1_id
            ORDER BY o.ordinal, e.candidate_id
            """
        )
    )
    next_edge = next(edge_rows, None)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("source1_entity_id", "matched_entity_ids"))
        for ordinal, source1_id in connection.execute(
            "SELECT ordinal, source1_id FROM source1_order ORDER BY ordinal"
        ):
            matched_ids: list[str] = []
            while next_edge is not None and next_edge[0] == ordinal:
                matched_ids.append(next_edge[2])
                next_edge = next(edge_rows, None)
            writer.writerow((source1_id, ",".join(matched_ids)))


def _flush_feature_batch(
    connection: sqlite3.Connection,
    pending: list[tuple[str, list[CandidateEvidence], list[dict[str, float]]]],
    model: ModelBundle,
    threshold: float,
) -> int:
    if not pending:
        return 0
    flat_features = [features for _, _, rows in pending for features in rows]
    probabilities = model.predict_proba(flat_features)
    expected_count = sum(len(rows) for _, _, rows in pending)
    if len(probabilities) != expected_count:
        raise RuntimeError("model returned an invalid probability count")
    qualified_count = 0
    offset = 0
    for source1_id, evidence_rows, _ in pending:
        for evidence in evidence_rows:
            score = float(probabilities[offset])
            offset += 1
            if score >= threshold:
                connection.execute(
                    "INSERT INTO qualified_pairs VALUES (?, ?, ?)",
                    (source1_id, evidence.candidate_entity_id, score),
                )
                qualified_count += 1
    pending.clear()
    return qualified_count


def run_inference_streaming(
    source1_path: Path,
    target_index: TargetIndex,
    output_dir: Path,
    model: ModelBundle | None = None,
    threshold: float = 0.72,
    candidate_cap: int = 50,
    country_fallback_limit: int = 10,
    source1_limit: int | None = None,
    enforce_exclusivity: bool = True,
    keep_work: bool = False,
    country: str | None = None,
    stage1: int | None = None,
    shard: tuple[int, int] | None = None,
) -> PipelineSummary:
    if candidate_cap < 1:
        raise ValueError("candidate_cap must be positive")
    model = model or _default_model()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate_pairs.tsv"
    matching_path = output_dir / "matching_results.tsv"
    candidate_temp_path = output_dir / "candidate_pairs.tsv.tmp"
    matching_temp_path = output_dir / "matching_results.tsv.tmp"
    incomplete_marker = output_dir / ".incomplete"
    work_path = output_dir / "scoring.sqlite"
    for path in (
        candidate_path,
        matching_path,
        candidate_temp_path,
        matching_temp_path,
        work_path,
        Path(f"{work_path}-wal"),
        Path(f"{work_path}-shm"),
    ):
        path.unlink(missing_ok=True)
    incomplete_marker.write_text("incomplete\n", encoding="utf-8")
    connection = sqlite3.connect(work_path)
    connection.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        CREATE TABLE source1_order (
            ordinal INTEGER PRIMARY KEY,
            source1_id TEXT NOT NULL UNIQUE
        );
        CREATE TABLE qualified_pairs (
            source1_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            score REAL NOT NULL,
            PRIMARY KEY (source1_id, candidate_id)
        );
        CREATE TABLE claims (
            candidate_id TEXT PRIMARY KEY,
            source1_id TEXT NOT NULL,
            score REAL NOT NULL
        );
        CREATE INDEX claims_source1 ON claims(source1_id);
        """
    )
    source1_count = 0
    candidate_edge_count = 0
    qualified_edge_count = 0
    pending_features: list[
        tuple[str, list[CandidateEvidence], list[dict[str, float]]]
    ] = []
    pending_pair_count = 0
    feature_batch_pairs = DEFAULT_SETTINGS.feature_batch_pairs
    try:
        with candidate_temp_path.open("w", encoding="utf-8", newline="") as candidate_handle:
            candidate_writer = csv.writer(
                candidate_handle, delimiter="\t", lineterminator="\n"
            )
            candidate_writer.writerow(("source1_entity_id", "candidate_entity_ids"))
            for ordinal, reference, evidence_rows, target_map in _iter_queries(
                target_index,
                _iter_normalized(
                    source1_path,
                    source=1,
                    limit=source1_limit,
                    country=country,
                    shard=shard,
                ),
                candidate_cap,
                country_fallback_limit,
                stage1=stage1,
            ):
                source1_id = reference.raw.entity_id
                connection.execute(
                    "INSERT INTO source1_order VALUES (?, ?)", (ordinal, source1_id)
                )
                candidate_ids = [
                    evidence.candidate_entity_id for evidence in evidence_rows
                ]
                candidate_writer.writerow((source1_id, ",".join(candidate_ids)))
                candidate_edge_count += len(candidate_ids)
                if evidence_rows:
                    target_records = target_map
                    feature_rows: list[dict[str, float]] = []
                    for evidence in evidence_rows:
                        candidate = target_records.get(evidence.candidate_entity_id)
                        if candidate is None:
                            raise RuntimeError(
                                f"candidate {evidence.candidate_entity_id} is absent from the target index"
                            )
                        runner_up_score = (
                            float(evidence_rows[1].score)
                            if len(evidence_rows) > 1
                            else 0.0
                        )
                        feature_rows.append(
                            extract_pair_features(
                                reference,
                                candidate,
                                evidence=evidence,
                                candidate_count=len(evidence_rows),
                                runner_up_score=runner_up_score,
                            )
                        )
                    pending_features.append((source1_id, list(evidence_rows), feature_rows))
                    pending_pair_count += len(feature_rows)
                    if pending_pair_count >= feature_batch_pairs:
                        qualified_edge_count += _flush_feature_batch(
                            connection,
                            pending_features,
                            model,
                            threshold,
                        )
                        pending_pair_count = 0
                        connection.commit()
                source1_count += 1
                if source1_count % 10_000 == 0:
                    connection.commit()
            qualified_edge_count += _flush_feature_batch(
                connection,
                pending_features,
                model,
                threshold,
            )
            candidate_handle.flush()
        connection.commit()
        if enforce_exclusivity:
            connection.execute(
                """
                INSERT INTO claims (candidate_id, source1_id, score)
                SELECT candidate_id, source1_id, score
                FROM qualified_pairs
                WHERE 1
                ON CONFLICT(candidate_id) DO UPDATE SET
                    source1_id = excluded.source1_id,
                    score = excluded.score
                WHERE excluded.score > claims.score
                   OR (excluded.score = claims.score
                       AND excluded.source1_id < claims.source1_id)
                """
            )
            connection.commit()
            matched_edge_count = connection.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        else:
            matched_edge_count = qualified_edge_count
        collision_count = qualified_edge_count - matched_edge_count
        _write_streaming_matches(
            connection,
            matching_temp_path,
            enforce_exclusivity,
        )
        candidate_temp_path.replace(candidate_path)
        matching_temp_path.replace(matching_path)
        incomplete_marker.unlink(missing_ok=True)
    finally:
        connection.close()
    if not keep_work:
        for path in (work_path, Path(f"{work_path}-wal"), Path(f"{work_path}-shm")):
            path.unlink(missing_ok=True)
    return PipelineSummary(
        source1_count=source1_count,
        candidate_edge_count=candidate_edge_count,
        matched_edge_count=matched_edge_count,
        collision_count=collision_count,
        complete=source1_limit is None,
    )


def run_fixture(
    source1_path: Path,
    target_paths: Iterable[Path],
    output_dir: Path,
    threshold: float = 0.72,
    candidate_cap: int = 50,
) -> PipelineSummary:
    index = build_target_index(target_paths)
    return run_inference(
        source1_path=source1_path,
        target_index=index,
        output_dir=output_dir,
        threshold=threshold,
        candidate_cap=candidate_cap,
    )


def _sample_entity_ids(
    path: Path, limit: int | None, country: str | None = None
) -> set[str]:
    wanted = country.strip().lower() if country else None
    selected: set[str] = set()
    for record in iter_records(path, source=1):
        if wanted is not None and record.country.strip().lower() != wanted:
            continue
        selected.add(record.entity_id)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_or_build_index(
    target_paths: Iterable[Path],
    index_path: Path | None,
    target_limit: int | None,
    bucket_limit: int,
    country: str | None = None,
) -> TargetIndex:
    paths = tuple(target_paths)
    expected_metadata = {
        "schema_version": "4",
        "target_paths": json.dumps([str(path) for path in paths]),
        "target_limit": "all" if target_limit is None else target_limit,
        "country": country.strip().lower() if country else "all",
    }
    if index_path is not None and index_path.exists():
        try:
            index = SqliteBlockingIndex(index_path, bucket_limit=bucket_limit)
        except sqlite3.DatabaseError:
            for stale in (
                index_path,
                Path(f"{index_path}-wal"),
                Path(f"{index_path}-shm"),
            ):
                stale.unlink(missing_ok=True)
            index = None
        if index is not None:
            metadata = index.metadata()
            matches = index.is_complete() and all(
                metadata.get(key) == str(value)
                for key, value in expected_metadata.items()
            )
            if matches:
                if isinstance(index, SqliteBlockingIndex):
                    index.ensure_postings()
                return index
            index.close()
    built = build_target_index(
        paths,
        index_path=index_path,
        limit_per_source=target_limit,
        bucket_limit=bucket_limit,
        country=country,
    )
    if isinstance(built, SqliteBlockingIndex):
        built.ensure_postings()
    return built


def open_or_build_index(
    target_paths: Iterable[Path],
    index_path: Path | None,
    target_limit: int | None,
    bucket_limit: int = 5000,
    country: str | None = None,
) -> TargetIndex:
    return _open_or_build_index(
        target_paths,
        index_path,
        target_limit,
        bucket_limit,
        country,
    )


def _run_train_command(arguments: argparse.Namespace) -> int:
    from .config import Settings

    settings = Settings(
        data_root=Path(arguments.data_root),
        output_dir=Path(arguments.output_dir),
        model_path=Path(arguments.model),
    )
    settings.ensure_output_dir()
    selected_ids = _sample_entity_ids(
        settings.train_source1_path,
        arguments.training_entity_limit,
        arguments.country,
    )
    if not selected_ids:
        raise ValueError("training entity selection is empty; check --country and --training-entity-limit")
    training_ids, validation_ids = entity_level_split(
        selected_ids,
        validation_fraction=arguments.validation_fraction,
        seed=settings.seed,
    )
    if not training_ids and selected_ids:
        training_ids = set(selected_ids)
    index_path = (
        Path(arguments.index_db)
        if arguments.index_db
        else settings.model_path.parent / "train_blocking_index.sqlite"
    )
    target_paths = [settings.source_path("train", 2), settings.source_path("train", 3)]
    index = _open_or_build_index(
        target_paths,
        index_path,
        arguments.target_limit,
        arguments.bucket_limit,
        arguments.country,
    )
    try:
        summary = train_from_index(
            reference_path=settings.train_source1_path,
            truth_path=settings.train_ground_truth_path,
            target_index=index,
            model_path=settings.model_path,
            training_ids=training_ids,
            validation_ids=validation_ids,
            max_pairs=settings.training_pair_cap if arguments.max_pairs is None else arguments.max_pairs,
            candidate_cap=arguments.candidate_cap,
            country_fallback_limit=arguments.country_fallback_limit,
            stage1=_stage1_argument(arguments.stage1),
            seed=settings.seed,
        )
        index_metadata = index.metadata() if isinstance(index, SqliteBlockingIndex) else {}
    finally:
        if isinstance(index, SqliteBlockingIndex):
            index.close()
    report = asdict(summary)
    report.update(
        {
            "country": (arguments.country or "all").strip().lower(),
            "model_sha256": _file_sha256(settings.model_path),
            "index_metadata": index_metadata,
            "training_entity_count": len(training_ids),
            "validation_entity_count": len(validation_ids),
        }
    )
    _write_json(settings.output_dir / "training_summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _discover_countries(path: Path) -> list[str]:
    countries: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader, None)
        for row in reader:
            if row and len(row) >= 4:
                countries.add(row[3].strip().lower())
    return sorted(countries)


def _concatenate_outputs(
    part_dirs: Iterable[tuple[str, Path]],
    final_dir: Path,
    delete_parts: bool = False,
) -> dict[str, int]:
    final_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    for file_name, header in (
        ("candidate_pairs.tsv", CANDIDATE_HEADER),
        ("matching_results.tsv", MATCHING_HEADER),
    ):
        target_path = final_dir / file_name
        target_path.unlink(missing_ok=True)
        row_count = 0
        with target_path.open("w", encoding="utf-8", newline="") as target_handle:
            writer = csv.writer(target_handle, delimiter="\t", lineterminator="\n")
            writer.writerow(header)
            for _, part_dir in part_dirs:
                part_path = part_dir / file_name
                if not part_path.exists():
                    raise FileNotFoundError(
                        f"missing partition output: {part_path}; "
                        "an earlier partition did not complete"
                    )
                with part_path.open("r", encoding="utf-8", newline="") as part_handle:
                    reader = csv.reader(part_handle, delimiter="\t")
                    part_header = tuple(next(reader, ()))
                    if part_header != tuple(header):
                        raise ValueError(
                            f"unexpected header in {part_path}: {part_header}"
                        )
                    for row in reader:
                        writer.writerow(row)
                        row_count += 1
                if delete_parts:
                    part_path.unlink(missing_ok=True)
        written[file_name] = row_count
    return written


def _run_shard_worker(
    source1_path: Path,
    index_path: Path,
    output_dir: Path,
    country: str,
    shard: tuple[int, int],
    model: ModelBundle | None,
    threshold: float,
    candidate_cap: int,
    country_fallback_limit: int,
    stage1: int | None,
    source1_limit: int | None,
    enforce_exclusivity: bool,
    bucket_limit: int,
) -> None:
    index = SqliteBlockingIndex(index_path, bucket_limit=bucket_limit)
    try:
        summary = run_inference_streaming(
            source1_path=source1_path,
            target_index=index,
            output_dir=output_dir,
            model=model,
            threshold=threshold,
            candidate_cap=candidate_cap,
            country_fallback_limit=country_fallback_limit,
            stage1=stage1,
            source1_limit=source1_limit,
            enforce_exclusivity=enforce_exclusivity,
            country=country,
            shard=shard,
        )
    finally:
        index.close()
    _write_json(output_dir / "shard_summary.json", asdict(summary))


def _run_sharded_country(
    country: str,
    source1_path: Path,
    target_paths: list[Path],
    part_dir: Path,
    index_path: Path | None,
    arguments: argparse.Namespace,
    model: ModelBundle | None,
    threshold: float,
    shard_count: int,
) -> dict[str, object]:
    if index_path is None:
        raise ValueError("--in-memory cannot be combined with --shard-count")
    built = _open_or_build_index(
        target_paths,
        index_path,
        arguments.target_limit,
        arguments.bucket_limit,
        country,
    )
    if isinstance(built, SqliteBlockingIndex):
        built.close()
    context = multiprocessing.get_context("fork")
    shard_dirs: list[Path] = []
    processes: list[multiprocessing.Process] = []
    for shard_index in range(shard_count):
        shard_dir = part_dir / f"shard{shard_index}"
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
        shard_dir.mkdir(parents=True)
        shard_dirs.append(shard_dir)
        processes.append(
            context.Process(
                target=_run_shard_worker,
                kwargs={
                    "source1_path": source1_path,
                    "index_path": index_path,
                    "output_dir": shard_dir,
                    "country": country,
                    "shard": (shard_index, shard_count),
                    "model": model,
                    "threshold": threshold,
                    "candidate_cap": arguments.candidate_cap,
                    "country_fallback_limit": arguments.country_fallback_limit,
                    "stage1": _stage1_argument(arguments.stage1),
                    "source1_limit": arguments.source1_limit,
                    "enforce_exclusivity": not arguments.no_exclusivity,
                    "bucket_limit": arguments.bucket_limit,
                },
            )
        )
    for process in processes:
        process.start()
    exitcodes = []
    for process in processes:
        process.join()
        exitcodes.append(process.exitcode)
    shard_summaries: list[dict[str, object]] = []
    for shard_dir in shard_dirs:
        summary_path = shard_dir / "shard_summary.json"
        if summary_path.exists():
            shard_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    if any(code != 0 for code in exitcodes) or len(shard_summaries) != shard_count:
        raise RuntimeError(
            f"shard workers failed for {country}: exitcodes={exitcodes} "
            f"summaries={len(shard_summaries)}/{shard_count}"
        )
    written = _concatenate_outputs(
        [(f"shard{index}", shard_dir) for index, shard_dir in enumerate(shard_dirs)],
        part_dir,
        delete_parts=True,
    )
    totals: dict[str, object] = {
        "source1_count": sum(int(value["source1_count"]) for value in shard_summaries),
        "candidate_edge_count": sum(
            int(value["candidate_edge_count"]) for value in shard_summaries
        ),
        "matched_edge_count": sum(
            int(value["matched_edge_count"]) for value in shard_summaries
        ),
        "collision_count": sum(
            int(value["collision_count"]) for value in shard_summaries
        ),
        "complete": all(bool(value["complete"]) for value in shard_summaries),
    }
    totals.update({"shards": shard_count, "written_rows": written})
    for shard_dir in shard_dirs:
        shutil.rmtree(shard_dir, ignore_errors=True)
    if not arguments.keep_work:
        for stale in (
            index_path,
            Path(f"{index_path}-wal"),
            Path(f"{index_path}-shm"),
        ):
            stale.unlink(missing_ok=True)
        shutil.rmtree(Path(f"{index_path}.postings"), ignore_errors=True)
    print(
        json.dumps(
            {
                "country": country,
                "source1_count": totals["source1_count"],
                "shards": shard_count,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return totals


def _run_partitioned_inference(
    settings: Settings,
    arguments: argparse.Namespace,
    source1_path: Path,
    target_paths: list[Path],
    model: ModelBundle | None,
    model_kind: str,
    model_hash: str | None,
    threshold: float,
    countries: list[str],
) -> int:
    output_dir = Path(settings.output_dir)
    parts_root = output_dir / "parts"
    part_entries: list[tuple[str, Path]] = []
    summaries: dict[str, dict[str, object]] = {}
    shard_count = max(1, int(arguments.shard_count))
    for country in countries:
        part_dir = parts_root / country
        part_dir.mkdir(parents=True, exist_ok=True)
        index_path = None if arguments.in_memory else part_dir / "blocking_index.sqlite"
        if shard_count > 1:
            summaries[country] = _run_sharded_country(
                country=country,
                source1_path=source1_path,
                target_paths=target_paths,
                part_dir=part_dir,
                index_path=index_path,
                arguments=arguments,
                model=model,
                threshold=threshold,
                shard_count=shard_count,
            )
            part_entries.append((country, part_dir))
            continue
        index = _open_or_build_index(
            target_paths,
            index_path,
            arguments.target_limit,
            arguments.bucket_limit,
            country,
        )
        try:
            summary = run_inference_streaming(
                source1_path=source1_path,
                target_index=index,
                output_dir=part_dir,
                model=model,
                threshold=threshold,
                candidate_cap=arguments.candidate_cap,
                country_fallback_limit=arguments.country_fallback_limit,
                stage1=_stage1_argument(arguments.stage1),
                source1_limit=arguments.source1_limit,
                enforce_exclusivity=not arguments.no_exclusivity,
                keep_work=arguments.keep_work,
                country=country,
            )
        finally:
            if isinstance(index, SqliteBlockingIndex):
                index.close()
        if index_path is not None and not arguments.keep_work:
            for stale in (index_path, Path(f"{index_path}-wal"), Path(f"{index_path}-shm")):
                stale.unlink(missing_ok=True)
            shutil.rmtree(Path(f"{index_path}.postings"), ignore_errors=True)
        summaries[country] = asdict(summary)
        part_entries.append((country, part_dir))
        print(
            json.dumps(
                {"country": country, "source1_count": summary.source1_count},
                sort_keys=True,
            ),
            flush=True,
        )
    written = _concatenate_outputs(part_entries, output_dir)
    totals = {
        "source1_count": sum(int(value["source1_count"]) for value in summaries.values()),
        "candidate_edge_count": sum(
            int(value["candidate_edge_count"]) for value in summaries.values()
        ),
        "matched_edge_count": sum(
            int(value["matched_edge_count"]) for value in summaries.values()
        ),
        "collision_count": sum(
            int(value["collision_count"]) for value in summaries.values()
        ),
        "complete": all(bool(value["complete"]) for value in summaries.values()),
    }
    report: dict[str, object] = dict(totals)
    report.update(
        {
            "partitioned_by_country": True,
            "shards": shard_count,
            "countries": countries,
            "parts": summaries,
            "written_rows": written,
            "model_kind": model_kind,
            "model_sha256": model_hash,
            "threshold": threshold,
            "target_limited": arguments.target_limit is not None,
            "source1_limited": arguments.source1_limit is not None,
        }
    )
    _write_json(output_dir / "inference_summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _run_infer_command(arguments: argparse.Namespace) -> int:
    from .config import Settings

    settings = Settings(
        data_root=Path(arguments.data_root),
        output_dir=Path(arguments.output_dir),
    )
    settings.ensure_output_dir()
    if arguments.split == "train":
        source1_path = settings.train_source1_path
        target_paths = [settings.source_path("train", 2), settings.source_path("train", 3)]
    else:
        source1_path = settings.test_source1_path
        target_paths = [settings.source_path("test", 2), settings.source_path("test", 3)]
    model = None
    model_kind = "heuristic"
    model_hash = None
    if arguments.model and arguments.heuristic:
        raise ValueError("choose either --model or --heuristic, not both")
    if arguments.model:
        model_path = Path(arguments.model)
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model = load_model(model_path)
        model_kind = model.kind
        model_hash = _file_sha256(model_path)
    elif not arguments.heuristic:
        raise ValueError("infer requires --model or the explicit --heuristic flag")
    threshold = arguments.threshold
    if threshold is None:
        threshold = model.threshold if model is not None else DEFAULT_SETTINGS.match_threshold
    if arguments.country and arguments.partition_by_country:
        raise ValueError("choose either --country or --partition-by-country, not both")
    if arguments.shard_count < 1:
        raise ValueError("--shard-count must be positive")
    if arguments.shard_count > 1 and not arguments.partition_by_country:
        raise ValueError("--shard-count requires --partition-by-country")
    if arguments.partition_by_country:
        if arguments.index_db:
            raise ValueError(
                "--index-db is not supported with --partition-by-country; "
                "each partition writes its own index beside its part output"
            )
        countries = _discover_countries(source1_path)
        if not countries:
            raise ValueError(f"no countries found in {source1_path}")
        return _run_partitioned_inference(
            settings,
            arguments,
            source1_path,
            target_paths,
            model,
            model_kind,
            model_hash,
            threshold,
            countries,
        )
    country = arguments.country.strip().lower() if arguments.country else None
    index_path = (
        None
        if arguments.in_memory
        else Path(arguments.index_db)
        if arguments.index_db
        else settings.output_dir / "blocking_index.sqlite"
    )
    index = _open_or_build_index(
        target_paths,
        index_path,
        arguments.target_limit,
        arguments.bucket_limit,
        country,
    )
    try:
        summary = run_inference_streaming(
            source1_path=source1_path,
            target_index=index,
            output_dir=settings.output_dir,
            model=model,
            threshold=threshold,
            candidate_cap=arguments.candidate_cap,
            country_fallback_limit=arguments.country_fallback_limit,
            stage1=_stage1_argument(arguments.stage1),
            source1_limit=arguments.source1_limit,
            enforce_exclusivity=not arguments.no_exclusivity,
            keep_work=arguments.keep_work,
            country=country,
        )
    finally:
        if isinstance(index, SqliteBlockingIndex):
            index.close()
    report = asdict(summary)
    report.update(
        {
            "model_kind": model_kind,
            "model_sha256": model_hash,
            "threshold": threshold,
            "country": country or "all",
            "index_path": str(index_path) if index_path is not None else None,
            "target_limited": arguments.target_limit is not None,
            "source1_limited": arguments.source1_limit is not None,
        }
    )
    _write_json(settings.output_dir / "inference_summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Business entity resolution pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train the candidate classifier")
    train_parser.add_argument("--data-root", default=str(DEFAULT_SETTINGS.data_root))
    train_parser.add_argument("--output-dir", default=str(DEFAULT_SETTINGS.output_dir))
    train_parser.add_argument("--model", default=str(DEFAULT_SETTINGS.model_path))
    train_parser.add_argument("--index-db", default=None)
    train_parser.add_argument("--training-entity-limit", type=int, default=50_000)
    train_parser.add_argument("--validation-fraction", type=float, default=0.2)
    train_parser.add_argument("--target-limit", type=int, default=None)
    train_parser.add_argument("--candidate-cap", type=int, default=DEFAULT_SETTINGS.candidate_cap)
    train_parser.add_argument("--country-fallback-limit", type=int, default=DEFAULT_SETTINGS.country_fallback_limit)
    train_parser.add_argument("--bucket-limit", type=int, default=DEFAULT_SETTINGS.bucket_limit)
    train_parser.add_argument(
        "--stage1",
        type=int,
        default=DEFAULT_SETTINGS.blocking_stage1,
        help="stage-1 pool size for the similarity rerank; <= candidate-cap disables it",
    )
    train_parser.add_argument("--max-pairs", type=int, default=None)
    train_parser.add_argument("--country", default=None, help="restrict training and index to one country")
    train_parser.set_defaults(handler=_run_train_command)

    infer_parser = subparsers.add_parser("infer", help="write matching and candidate TSVs")
    infer_parser.add_argument("--data-root", default=str(DEFAULT_SETTINGS.data_root))
    infer_parser.add_argument("--output-dir", default=str(DEFAULT_SETTINGS.output_dir))
    infer_parser.add_argument("--split", choices=("train", "test"), default="test")
    infer_parser.add_argument("--index-db", default=None)
    infer_parser.add_argument("--in-memory", action="store_true")
    infer_parser.add_argument("--model", default=None)
    infer_parser.add_argument("--heuristic", action="store_true")
    infer_parser.add_argument("--threshold", type=float, default=None)
    infer_parser.add_argument("--source1-limit", type=int, default=None)
    infer_parser.add_argument("--target-limit", type=int, default=None)
    infer_parser.add_argument("--candidate-cap", type=int, default=DEFAULT_SETTINGS.candidate_cap)
    infer_parser.add_argument("--country-fallback-limit", type=int, default=DEFAULT_SETTINGS.country_fallback_limit)
    infer_parser.add_argument("--bucket-limit", type=int, default=DEFAULT_SETTINGS.bucket_limit)
    infer_parser.add_argument(
        "--stage1",
        type=int,
        default=DEFAULT_SETTINGS.blocking_stage1,
        help="stage-1 pool size for the similarity rerank; <= candidate-cap disables it",
    )
    infer_parser.add_argument("--no-exclusivity", action="store_true")
    infer_parser.add_argument("--keep-work", action="store_true")
    infer_parser.add_argument("--country", default=None, help="restrict index and queries to one country")
    infer_parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="parallel workers per country partition; requires --partition-by-country",
    )
    infer_parser.add_argument(
        "--partition-by-country",
        action="store_true",
        help="build, infer, and free one country index at a time, then merge outputs",
    )
    infer_parser.set_defaults(handler=_run_infer_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
