from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")
MATCHING_HEADER = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")


class DataFormatError(ValueError):
    pass


@dataclass(frozen=True)
class Record:
    entity_id: str
    business_name: str
    business_address: str
    country: str

    @property
    def source(self) -> int:
        return int(self.entity_id[1])


@dataclass(frozen=True)
class GroundTruthRow:
    source1_entity_id: str
    matched_entity_ids: tuple[str, ...]


def _validate_prefix(entity_id: str, source: int | None = None) -> None:
    if (
        len(entity_id) < 3
        or entity_id[0] != "S"
        or entity_id[1] not in {"1", "2", "3"}
        or entity_id[2] != "-"
    ):
        raise DataFormatError(f"invalid entity ID: {entity_id!r}")
    actual = int(entity_id[1])
    if source is not None and actual != source:
        raise DataFormatError(f"expected S{source}- ID, received {entity_id!r}")


def _iter_tsv(path: Path, expected_columns: Sequence[str]) -> Iterator[list[str]]:
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise DataFormatError(f"could not open {path}: {exc}") from exc
    with handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as exc:
            raise DataFormatError(f"empty TSV file: {path}") from exc
        normalized_header = tuple(value.strip().lower() for value in header)
        if normalized_header != tuple(expected_columns):
            raise DataFormatError(
                f"unexpected header in {path}: {normalized_header}; "
                f"expected {tuple(expected_columns)}"
            )
        for line_number, row in enumerate(reader, start=2):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != len(expected_columns):
                raise DataFormatError(
                    f"malformed row {line_number} in {path}: expected "
                    f"{len(expected_columns)} columns, received {len(row)}"
                )
            yield row


def iter_records(path: Path, source: int | None = None) -> Iterator[Record]:
    for row in _iter_tsv(path, SOURCE_COLUMNS):
        entity_id = row[0].strip()
        _validate_prefix(entity_id, source)
        yield Record(
            entity_id=entity_id,
            business_name=row[1],
            business_address=row[2],
            country=row[3],
        )


def load_records(path: Path, source: int | None = None) -> list[Record]:
    return list(iter_records(path, source))


def iter_ground_truth(path: Path) -> Iterator[GroundTruthRow]:
    for row in _iter_tsv(path, GROUND_TRUTH_COLUMNS):
        source1_id = row[0].strip()
        _validate_prefix(source1_id, 1)
        raw_ids = row[1].strip()
        matched = tuple(dict.fromkeys(item.strip() for item in raw_ids.split(",") if item.strip()))
        for entity_id in matched:
            _validate_prefix(entity_id)
            if entity_id.startswith("S1-"):
                raise DataFormatError(f"ground truth contains a Source-1 match: {entity_id}")
        yield GroundTruthRow(source1_id, matched)


def load_ground_truth(path: Path) -> dict[str, set[str]]:
    return {row.source1_entity_id: set(row.matched_entity_ids) for row in iter_ground_truth(path)}


def iter_source1_ids(path: Path) -> Iterator[str]:
    for record in iter_records(path, source=1):
        yield record.entity_id


def parse_id_list(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))


def _write_rows(
    path: Path,
    header: Sequence[str],
    rows: Iterable[tuple[str, Iterable[str]]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for source1_id, ids in rows:
            unique_ids = list(dict.fromkeys(entity_id.strip() for entity_id in ids if entity_id.strip()))
            writer.writerow((source1_id, ",".join(unique_ids)))
            count += 1
    return count


def write_matching_results(path: Path, rows: Iterable[tuple[str, Iterable[str]]]) -> int:
    return _write_rows(path, MATCHING_HEADER, rows)


def write_candidate_pairs(path: Path, rows: Iterable[tuple[str, Iterable[str]]]) -> int:
    return _write_rows(path, CANDIDATE_HEADER, rows)


def read_output_rows(path: Path, expected_header: Sequence[str]) -> dict[str, tuple[str, ...]]:
    rows: dict[str, tuple[str, ...]] = {}
    for row in _iter_tsv(path, expected_header):
        source1_id = row[0].strip()
        if source1_id in rows:
            raise DataFormatError(f"duplicate output row: {source1_id}")
        rows[source1_id] = parse_id_list(row[1])
    return rows
