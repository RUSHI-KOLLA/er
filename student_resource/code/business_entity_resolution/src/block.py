from __future__ import annotations

from array import array
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
import shutil
import sqlite3
from typing import Iterable, Mapping

import numpy as np

from .io import Record
from .normalize import NormalizedRecord, normalize_record


@dataclass(frozen=True)
class CandidateEvidence:
    candidate_entity_id: str
    score: float
    methods: frozenset[str]
    best_rank: int
    best_rank_score: float


_METHOD_WEIGHTS = {
    "name_core_exact": 1.0,
    "name_exact": 0.96,
    "name_token_sort": 0.9,
    "name_prefix": 0.55,
    "name_token": 0.35,
    "address_postal": 0.82,
    "address_house": 0.72,
    "address_number": 0.62,
    "address_city": 0.78,
    "address_token": 0.3,
}
_METHOD_ORDER = tuple(_METHOD_WEIGHTS)

_RERANK_WEIGHTS = (2.0, 1.0, 1.0, 2.0)
_SIMILARITY_CACHE_LIMIT = 20_000
_LIGHT_RE = re.compile(r"[^a-z0-9]+")


def _folded_sets(text: str) -> tuple[set[int], set[str]]:
    folded = _LIGHT_RE.sub(" ", text.lower()).strip()
    padded = folded.replace(" ", "_")
    raw = padded.encode("ascii")
    span = len(raw) - 2
    grams = (
        {
            (raw[index] << 16) | (raw[index + 1] << 8) | raw[index + 2]
            for index in range(span)
        }
        if span > 0
        else set()
    )
    tokens = set(folded.split())
    return grams, tokens


def _jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if not overlap:
        return 0.0
    return overlap / (len(left) + len(right) - overlap)


def _similarity_sets(
    raw_name: str,
    raw_address: str,
    cache: dict[str, tuple[set[int], set[str]]],
) -> tuple[set[int], set[str], set[int], set[str]]:
    name_grams, name_tokens = _cached_sets(raw_name, cache)
    address_grams, address_tokens = _cached_sets(raw_address, cache)
    return name_grams, name_tokens, address_grams, address_tokens


def _cached_sets(
    raw_text: str,
    cache: dict[str, tuple[set[int], set[str]]],
) -> tuple[set[int], set[str]]:
    sets = cache.get(raw_text)
    if sets is None:
        sets = _folded_sets(raw_text or "")
        if len(cache) >= _SIMILARITY_CACHE_LIMIT:
            for index, existing in enumerate(list(cache)):
                if index & 1:
                    del cache[existing]
        cache[raw_text] = sets
    return sets


def _rerank_bonus(
    query_sets: tuple[set[int], set[str], set[int], set[str]],
    candidate_sets: tuple[set[int], set[str], set[int], set[str]],
) -> float:
    query_name_grams, query_name_tokens, query_address_grams, query_address_tokens = query_sets
    name_grams, name_tokens, address_grams, address_tokens = candidate_sets
    return (
        _RERANK_WEIGHTS[0] * _jaccard(query_name_grams, name_grams)
        + _RERANK_WEIGHTS[1] * _jaccard(query_name_tokens, name_tokens)
        + _RERANK_WEIGHTS[2] * _jaccard(query_address_grams, address_grams)
        + _RERANK_WEIGHTS[3] * _jaccard(query_address_tokens, address_tokens)
    )


_POSTINGS_SCHEMA = "postings-v1"
_POSTINGS_DIR_SUFFIX = ".postings"
_POSTING_WEIGHTS = np.array(
    [_METHOD_WEIGHTS[name] for name in _METHOD_ORDER], dtype=np.float64
)
_METHOD_IDS = {name: index for index, name in enumerate(_METHOD_ORDER)}
_METHOD_MASKS = {name: 1 << index for index, name in enumerate(_METHOD_ORDER)}
_METHOD_BIT_WEIGHTS = 1 << np.arange(len(_METHOD_ORDER), dtype=np.int64)
_MASK_TO_METHODS = {
    mask: frozenset(
        name for index, name in enumerate(_METHOD_ORDER) if mask & (1 << index)
    )
    for mask in range(1 << len(_METHOD_ORDER))
}
_FINGERPRINT_KEYS = (
    "schema_version",
    "normalization_version",
    "row_count",
    "country",
    "target_paths",
    "target_limit",
    "complete",
)


def _index_fingerprint(metadata: Mapping[str, str]) -> str:
    payload = "|".join(str(metadata.get(key, "")) for key in _FINGERPRINT_KEYS)
    payload += "|" + "|".join(_METHOD_ORDER)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _key_hash(method: str, key: str) -> int:
    digest = hashlib.blake2b(f"{method}\x00{key}".encode("utf-8"), digest_size=8)
    return int.from_bytes(digest.digest(), "big")


@dataclass
class _Postings:
    hashes: np.ndarray
    starts: np.ndarray
    counts: np.ndarray
    targets: np.ndarray
    methods: np.ndarray
    countries: np.ndarray
    country_ids: dict[str, int]
    row_count: int


def _load_postings(path: Path, metadata: Mapping[str, str]) -> _Postings | None:
    directory = Path(f"{path}{_POSTINGS_DIR_SUFFIX}")
    meta_path = directory / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("schema") != _POSTINGS_SCHEMA:
            return None
        if meta.get("fingerprint") != _index_fingerprint(metadata):
            return None
        n_groups = int(meta["n_groups"])
        n_postings = int(meta["n_postings"])
        row_count = int(meta["row_count"])
        countries = {
            str(key): int(value) for key, value in meta["country_ids"].items()
        }
        return _Postings(
            hashes=np.memmap(
                directory / "hashes.bin", dtype=np.uint64, mode="r",
                shape=(n_groups,),
            ),
            starts=np.memmap(
                directory / "starts.bin", dtype=np.uint64, mode="r",
                shape=(n_groups,),
            ),
            counts=np.memmap(
                directory / "counts.bin", dtype=np.uint32, mode="r",
                shape=(n_groups,),
            ),
            targets=np.memmap(
                directory / "targets.bin", dtype=np.int32, mode="r",
                shape=(n_postings,),
            ),
            methods=np.memmap(
                directory / "methods.bin", dtype=np.uint8, mode="r",
                shape=(n_postings,),
            ),
            countries=np.memmap(
                directory / "countries.bin", dtype=np.uint8, mode="r",
                shape=(row_count,),
            ),
            country_ids=countries,
            row_count=row_count,
        )
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None


def _rank_evidence(
    record: NormalizedRecord,
    evidence: dict[str, dict[str, object]],
    targets: Mapping[str, NormalizedRecord],
    cap: int,
    country_fallback_limit: int,
) -> tuple[CandidateEvidence, ...]:
    ranked = sorted(
        evidence.items(),
        key=lambda item: (
            -float(item[1]["score"]),
            targets[item[0]].country != record.country,
            item[0],
        ),
    )
    selected: list[tuple[str, float, frozenset[str]]] = []
    cross_country_count = 0
    for candidate_id, entry in ranked:
        same_country = targets[candidate_id].country == record.country
        if not same_country:
            if cross_country_count >= country_fallback_limit:
                continue
            cross_country_count += 1
        selected.append(
            (
                candidate_id,
                float(entry["score"]),
                frozenset(entry["methods"]),
            )
        )
        if len(selected) >= cap:
            break
    return tuple(
        CandidateEvidence(
            candidate_entity_id=candidate_id,
            score=score,
            methods=methods,
            best_rank=rank,
            best_rank_score=score,
        )
        for rank, (candidate_id, score, methods) in enumerate(selected, start=1)
    )


class BlockingIndex:
    def __init__(self, bucket_limit: int = 5000):
        if bucket_limit < 1:
            raise ValueError("bucket_limit must be positive")
        self.bucket_limit = bucket_limit
        self.targets: dict[str, NormalizedRecord] = {}
        self._buckets: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._bucket_sizes: dict[tuple[str, str], int] = defaultdict(int)
        self._target_keys: dict[str, tuple[tuple[str, str], ...]] = {}

    @classmethod
    def from_records(cls, records: Iterable[NormalizedRecord], bucket_limit: int = 5000) -> "BlockingIndex":
        index = cls(bucket_limit=bucket_limit)
        for record in records:
            if record.raw.source in {2, 3}:
                index.add(record)
        return index

    @staticmethod
    def _keys(record: NormalizedRecord) -> tuple[tuple[str, str], ...]:
        keys: list[tuple[str, str]] = []
        name = record.name
        if name.core_folded:
            keys.append(("name_core_exact", name.core_folded))
        if name.folded:
            keys.append(("name_exact", name.folded))
        if name.token_sort:
            keys.append(("name_token_sort", name.token_sort))
        prefix = name.core_folded[:3]
        if len(prefix) >= 3:
            keys.append(("name_prefix", prefix))
        for token in name.core_folded.split():
            if len(token) >= 2 and token not in {"and", "the"}:
                keys.append(("name_token", token))
        address = record.address
        if address.postal_code:
            keys.append(("address_postal", address.postal_code))
        if address.house_number:
            keys.append(("address_house", address.house_number))
        for number in address.numbers:
            if len(number) >= 2:
                keys.append(("address_number", number))
        if address.city:
            keys.append(("address_city", address.city))
        for token in address.folded.split():
            if len(token) >= 4 and token not in {"near", "road", "street", "avenue"}:
                keys.append(("address_token", token))
        return tuple(dict.fromkeys(keys))

    def add(self, record: NormalizedRecord) -> None:
        if record.raw.source not in {2, 3}:
            raise ValueError("blocking targets must be Source 2 or Source 3")
        self.targets[record.raw.entity_id] = record
        keys = self._keys(record)
        self._target_keys[record.raw.entity_id] = keys
        for method, key in keys:
            bucket_key = (method, key)
            self._buckets[bucket_key].add(record.raw.entity_id)
            self._bucket_sizes[bucket_key] = len(self._buckets[bucket_key])

    def _bucket_ids(self, method: str, key: str) -> set[str]:
        bucket_key = (method, key)
        if self._bucket_sizes[bucket_key] > self.bucket_limit:
            return set()
        return set(self._buckets[bucket_key])

    def query(
        self,
        record: NormalizedRecord,
        cap: int = 50,
        country_fallback_limit: int = 10,
        stage1: int | None = None,
    ) -> tuple[CandidateEvidence, ...]:
        if cap < 1:
            return ()
        if country_fallback_limit < 0:
            raise ValueError("country_fallback_limit must be non-negative")
        evidence: dict[str, dict[str, object]] = {}
        for method, key in self._keys(record):
            for candidate_id in self._bucket_ids(method, key):
                candidate = self.targets[candidate_id]
                same_country = candidate.country == record.country
                base_score = float(_METHOD_WEIGHTS[method])
                if same_country:
                    base_score += 0.15
                entry = evidence.setdefault(
                    candidate_id,
                    {"score": 0.0, "methods": set()},
                )
                entry["score"] = float(entry["score"]) + base_score
                entry["methods"].add(method)
        return _rank_evidence(
            record,
            evidence,
            self.targets,
            cap,
            country_fallback_limit,
        )

    def query_many(
        self,
        records: Iterable[NormalizedRecord],
        cap: int = 50,
        country_fallback_limit: int = 10,
        stage1: int | None = None,
    ) -> tuple[tuple[CandidateEvidence, ...], ...]:
        return tuple(
            self.query(record, cap, country_fallback_limit) for record in records
        )

    def query_many_with_targets(
        self,
        records: Iterable[NormalizedRecord],
        cap: int = 50,
        country_fallback_limit: int = 10,
        stage1: int | None = None,
    ) -> tuple[
        tuple[tuple[CandidateEvidence, ...], ...],
        dict[str, NormalizedRecord],
    ]:
        record_rows = tuple(records)
        return (
            self.query_many(record_rows, cap, country_fallback_limit),
            self.targets,
        )

    def iter_candidates(
        self,
        source1_records: Iterable[NormalizedRecord],
        cap: int = 50,
        country_fallback_limit: int = 10,
    ) -> Iterable[tuple[str, tuple[CandidateEvidence, ...]]]:
        for record in source1_records:
            if record.raw.source != 1:
                raise ValueError("candidate queries must use Source 1 records")
            yield record.raw.entity_id, self.query(record, cap, country_fallback_limit)

    def candidate_map(
        self,
        source1_records: Iterable[NormalizedRecord],
        cap: int = 50,
        country_fallback_limit: int = 10,
    ) -> dict[str, tuple[CandidateEvidence, ...]]:
        return dict(self.iter_candidates(source1_records, cap, country_fallback_limit))


class SqliteBlockingIndex:
    def __init__(
        self, path: Path, bucket_limit: int = 5000, use_postings: bool = True
    ):
        if bucket_limit < 1:
            raise ValueError("bucket_limit must be positive")
        self.path = Path(path)
        self.bucket_limit = bucket_limit
        self._postings: _Postings | None = None
        self._last_query_targets: dict[str, NormalizedRecord] = {}
        self._similarity_cache: dict[str, tuple[set[int], set[str]]] = {}
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA foreign_keys = OFF")
        self.connection.execute("PRAGMA page_size = 8192")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self._create_schema()
        if use_postings:
            self._postings = _load_postings(self.path, self.metadata())

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS targets (
                row_id INTEGER PRIMARY KEY,
                entity_id TEXT NOT NULL UNIQUE,
                source INTEGER NOT NULL,
                business_name TEXT NOT NULL,
                business_address TEXT NOT NULL,
                country TEXT NOT NULL,
                country_key TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS target_keys (
                method TEXT NOT NULL,
                key TEXT NOT NULL,
                target_id INTEGER NOT NULL,
                PRIMARY KEY (method, key, target_id),
                FOREIGN KEY (target_id) REFERENCES targets(row_id)
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS targets_country
                ON targets (country);
            CREATE TABLE IF NOT EXISTS key_counts (
                method TEXT NOT NULL,
                key TEXT NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (method, key)
            );
            CREATE TABLE IF NOT EXISTS index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def set_metadata(self, values: Mapping[str, object]) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO index_metadata VALUES (?, ?)",
            ((str(key), str(value)) for key, value in values.items()),
        )
        self.connection.commit()

    def metadata(self) -> dict[str, str]:
        return {
            str(row[0]): str(row[1])
            for row in self.connection.execute(
                "SELECT key, value FROM index_metadata ORDER BY key"
            )
        }

    def is_complete(self) -> bool:
        return self.metadata().get("complete") == "1"

    def ensure_postings(self) -> None:
        if self._postings is not None:
            return
        directory = Path(f"{self.path}{_POSTINGS_DIR_SUFFIX}")
        fingerprint = _index_fingerprint(self.metadata())
        meta_path = directory / "meta.json"
        if meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text(encoding="utf-8"))
            except ValueError:
                existing = {}
            if existing.get("fingerprint") == fingerprint:
                loaded = _load_postings(self.path, self.metadata())
                if loaded is not None:
                    self._postings = loaded
                    return
        tmp = Path(f"{directory}.tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        try:
            self._export_postings(tmp, fingerprint)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            return
        shutil.rmtree(directory, ignore_errors=True)
        tmp.rename(directory)
        loaded = _load_postings(self.path, self.metadata())
        if loaded is not None:
            self._postings = loaded

    def _export_postings(self, directory: Path, fingerprint: str) -> None:
        metadata = self.metadata()
        row_count = int(metadata.get("row_count", "0"))
        country_ids: dict[str, int] = {}
        country_table = bytearray(row_count)
        expected_row = 1
        for row_id, country_key in self.connection.execute(
            "SELECT row_id, country_key FROM targets ORDER BY row_id"
        ):
            if int(row_id) != expected_row:
                raise RuntimeError("target row ids are not contiguous")
            expected_row += 1
            key = str(country_key)
            country_id = country_ids.get(key)
            if country_id is None:
                if len(country_ids) >= 255:
                    raise RuntimeError("too many country keys")
                country_id = len(country_ids)
                country_ids[key] = country_id
            country_table[int(row_id) - 1] = country_id
        if expected_row - 1 != row_count:
            raise RuntimeError("target row count mismatch")
        g_hash = array("Q")
        g_start = array("Q")
        g_count = array("I")
        target_buf: list[int] = []
        method_buf: list[int] = []
        cur_targets: list[int] = []
        cur_methods: list[int] = []
        last_key: tuple[str, str] | None = None
        n_postings = 0
        with (directory / "targets.bin").open("wb") as targets_file, (
            directory / "methods.bin"
        ).open("wb") as methods_file:

            def write_group() -> None:
                nonlocal n_postings
                if last_key is None:
                    return
                g_hash.append(_key_hash(last_key[0], last_key[1]))
                g_start.append(n_postings)
                g_count.append(len(cur_targets))
                n_postings += len(cur_targets)
                target_buf.extend(cur_targets)
                method_buf.extend(cur_methods)
                cur_targets.clear()
                cur_methods.clear()
                if len(target_buf) >= 1 << 20:
                    targets_file.write(array("i", target_buf).tobytes())
                    methods_file.write(array("B", method_buf).tobytes())
                    target_buf.clear()
                    method_buf.clear()

            for method, key, target_id in self.connection.execute(
                "SELECT method, key, target_id FROM target_keys "
                "ORDER BY method, key"
            ):
                pair = (str(method), str(key))
                if pair != last_key:
                    write_group()
                    last_key = pair
                method_id = _METHOD_IDS.get(pair[0])
                if method_id is None:
                    raise RuntimeError(f"unknown blocking method: {pair[0]}")
                cur_targets.append(int(target_id))
                cur_methods.append(method_id)
            write_group()
            if target_buf:
                targets_file.write(array("i", target_buf).tobytes())
                methods_file.write(array("B", method_buf).tobytes())
        hashes = np.frombuffer(g_hash.tobytes(), dtype=np.uint64)
        if hashes.size:
            order = np.argsort(hashes, kind="stable")
            sorted_hashes = hashes[order]
            if bool((sorted_hashes[1:] == sorted_hashes[:-1]).any()):
                raise RuntimeError("postings hash collision")
            starts = np.frombuffer(g_start.tobytes(), dtype=np.uint64)[order]
            counts = np.frombuffer(g_count.tobytes(), dtype=np.uint32)[order]
        else:
            sorted_hashes = hashes
            starts = np.frombuffer(g_start.tobytes(), dtype=np.uint64)
            counts = np.frombuffer(g_count.tobytes(), dtype=np.uint32)
        (directory / "hashes.bin").write_bytes(sorted_hashes.tobytes())
        (directory / "starts.bin").write_bytes(starts.tobytes())
        (directory / "counts.bin").write_bytes(counts.tobytes())
        (directory / "countries.bin").write_bytes(bytes(country_table))
        (directory / "meta.json").write_text(
            json.dumps(
                {
                    "schema": _POSTINGS_SCHEMA,
                    "fingerprint": fingerprint,
                    "row_count": row_count,
                    "n_groups": int(hashes.size),
                    "n_postings": int(n_postings),
                    "country_ids": country_ids,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def build(
        cls,
        records: Iterable[NormalizedRecord],
        path: Path,
        bucket_limit: int = 5000,
        batch_size: int = 50_000,
        overwrite: bool = False,
        metadata: Mapping[str, object] | None = None,
    ) -> "SqliteBlockingIndex":
        final_path = Path(path)
        if final_path.exists() and not overwrite and final_path.stat().st_size > 0:
            raise FileExistsError(f"blocking index already exists: {final_path}")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        for path_to_remove in (
            final_path,
            Path(f"{final_path}-wal"),
            Path(f"{final_path}-shm"),
        ):
            path_to_remove.unlink(missing_ok=True)
        index = cls(final_path, bucket_limit=bucket_limit)
        index.connection.execute("PRAGMA journal_mode = OFF")
        index.connection.execute("PRAGMA synchronous = OFF")
        index.connection.execute("PRAGMA locking_mode = EXCLUSIVE")
        index.connection.execute("PRAGMA cache_size = -1572864")
        index.connection.execute("PRAGMA temp_store = MEMORY")
        index.connection.execute("PRAGMA mmap_size = 4294967296")
        index.set_metadata({"complete": "0", "schema_version": "4"})
        target_count = 0
        target_batch: list[tuple[int, str, int, str, str, str, str]] = []
        key_batch: list[tuple[str, str, int]] = []
        try:
            for record in records:
                if record.raw.source not in {2, 3}:
                    continue
                target_count += 1
                row_id = target_count
                target_batch.append(
                    (
                        row_id,
                        record.raw.entity_id,
                        record.raw.source,
                        record.raw.business_name,
                        record.raw.business_address,
                        record.raw.country,
                        record.country,
                    )
                )
                for method, key in BlockingIndex._keys(record):
                    key_batch.append((method, key, row_id))
                if len(target_batch) >= batch_size:
                    index._insert_batch(target_batch, key_batch)
                    target_batch.clear()
                    key_batch.clear()
            if target_batch:
                index._insert_batch(target_batch, key_batch)
            index.connection.execute(
                "INSERT INTO key_counts (method, key, count) "
                "SELECT method, key, COUNT(*) FROM target_keys GROUP BY method, key"
            )
            index.connection.commit()
            index.connection.execute("PRAGMA journal_mode = WAL")
            index.connection.execute("PRAGMA synchronous = NORMAL")
            values = {
                "schema_version": "4",
                "normalization_version": "1",
                "row_count": target_count,
                "complete": "1",
            }
            values.update(metadata or {})
            index.set_metadata(values)
            index.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            index.connection.commit()
            index.close()
            return cls(final_path, bucket_limit=bucket_limit)
        except Exception:
            index.close()
            for path_to_remove in (
                final_path,
                Path(f"{final_path}-wal"),
                Path(f"{final_path}-shm"),
            ):
                path_to_remove.unlink(missing_ok=True)
            raise

    def _insert_batch(
        self,
        targets: list[tuple[int, str, int, str, str, str, str]],
        keys: list[tuple[str, str, int]],
    ) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO targets VALUES (?, ?, ?, ?, ?, ?, ?)", targets
        )
        self.connection.executemany(
            "INSERT OR IGNORE INTO target_keys VALUES (?, ?, ?)", keys
        )
        self.connection.commit()

    def _bucket_ids(self, method: str, key: str) -> tuple[str, ...]:
        rows = self.connection.execute(
            "SELECT t.entity_id FROM target_keys AS tk "
            "JOIN targets AS t ON t.row_id = tk.target_id "
            "WHERE tk.method = ? AND tk.key = ? LIMIT ?",
            (method, key, self.bucket_limit + 1),
        ).fetchall()
        if len(rows) > self.bucket_limit:
            return ()
        return tuple(row[0] for row in rows)

    def get(self, entity_id: str) -> NormalizedRecord | None:
        row = self.connection.execute(
            "SELECT entity_id, source, business_name, business_address, country "
            "FROM targets WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()
        if row is None:
            return None
        return normalize_record(
            Record(
                entity_id=row[0],
                business_name=row[2],
                business_address=row[3],
                country=row[4],
            )
        )

    def get_many(self, entity_ids: Iterable[str]) -> dict[str, NormalizedRecord]:
        unique_ids = tuple(dict.fromkeys(entity_ids))
        result: dict[str, NormalizedRecord] = {}
        for start in range(0, len(unique_ids), 900):
            chunk = unique_ids[start : start + 900]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"SELECT entity_id, business_name, business_address, country "
                f"FROM targets WHERE entity_id IN ({placeholders})",
                chunk,
            ).fetchall()
            for entity_id, name, address, country in rows:
                result[entity_id] = normalize_record(
                    Record(
                        entity_id=entity_id,
                        business_name=name,
                        business_address=address,
                        country=country,
                    )
                )
        return result

    def query_many(
        self,
        records: Iterable[NormalizedRecord],
        cap: int = 50,
        country_fallback_limit: int = 10,
        stage1: int | None = None,
    ) -> tuple[tuple[CandidateEvidence, ...], ...]:
        record_rows = tuple(records)
        self._last_query_targets = {}
        if cap < 1:
            return tuple(() for _ in record_rows)
        if country_fallback_limit < 0:
            raise ValueError("country_fallback_limit must be non-negative")
        if stage1 is not None and stage1 < 1:
            raise ValueError("stage1 must be positive")
        selection_limit = cap if stage1 is None else max(cap, stage1)
        rerank = bool(stage1 is not None and stage1 > cap)
        query_keys: list[tuple[int, str, str, str]] = []
        for request_id, record in enumerate(record_rows):
            query_keys.extend(
                (request_id, method, key, record.country)
                for method, key in BlockingIndex._keys(record)
            )
        results: list[tuple[CandidateEvidence, ...]] = [() for _ in record_rows]
        if not query_keys:
            return tuple(results)
        if self._postings is not None:
            grouped = self._group_postings(record_rows, query_keys)
        else:
            grouped = self._group_sql(query_keys, len(record_rows))
        pending: dict[int, list[tuple[int, float, int]]] = {}
        for request_id in range(len(record_rows)):
            ranked = sorted(
                grouped[request_id],
                key=lambda item: (-item[0], not item[1], item[2]),
            )
            chosen: list[tuple[int, float, int]] = []
            cross_country_count = 0
            for score, same_country, target_id, methods in ranked:
                if not same_country:
                    if cross_country_count >= country_fallback_limit:
                        continue
                    cross_country_count += 1
                chosen.append((target_id, score, methods))
                if len(chosen) >= selection_limit:
                    break
            pending[request_id] = chosen
        detail_ids = sorted(
            {target_id for chosen in pending.values() for target_id, _, _ in chosen}
        )
        if not detail_ids:
            return tuple(results)
        details: dict[int, tuple[str, str, str, str, str]] = {}
        for chunk_start in range(0, len(detail_ids), 900):
            chunk = detail_ids[chunk_start : chunk_start + 900]
            placeholders = ",".join("?" for _ in chunk)
            for detail_row in self.connection.execute(
                "SELECT row_id, entity_id, business_name, business_address, "
                "country, country_key FROM targets WHERE row_id IN ("
                + placeholders
                + ")",
                chunk,
            ):
                details[int(detail_row[0])] = (
                    str(detail_row[1]),
                    str(detail_row[2]),
                    str(detail_row[3]),
                    str(detail_row[4]),
                    str(detail_row[5]),
                )
        target_cache: dict[str, NormalizedRecord] = {}

        def emit(request_id: int) -> None:
            record = record_rows[request_id]
            rows_to_emit: list[tuple[int, str, float, str, str, str, int]] = []
            if rerank:
                query_sets = _similarity_sets(
                    record.raw.business_name,
                    record.raw.business_address,
                    self._similarity_cache,
                )
                scored: list[tuple[float, bool, str, str, str, str, int]] = []
                for target_id, score, methods in pending[request_id]:
                    entity_id, name, address, country, country_key = details[target_id]
                    total = score + _rerank_bonus(
                        query_sets,
                        _similarity_sets(name, address, self._similarity_cache),
                    )
                    scored.append(
                        (
                            total,
                            country_key != record.country,
                            entity_id,
                            name,
                            address,
                            country,
                            methods,
                        )
                    )
                scored.sort(key=lambda item: (-item[0], item[1], item[2]))
                rows_to_emit = [
                    (rank, item[2], item[0], item[3], item[4], item[5], item[6])
                    for rank, item in enumerate(scored[:cap], start=1)
                ]
            else:
                for rank, (target_id, score, methods) in enumerate(
                    pending[request_id], start=1
                ):
                    entity_id, name, address, country, _ = details[target_id]
                    rows_to_emit.append(
                        (rank, entity_id, score, name, address, country, methods)
                    )
                rows_to_emit.sort(key=lambda item: item[0])
            evidence: list[CandidateEvidence] = []
            for rank, entity_id, score, name, address, country, methods in rows_to_emit:
                target = target_cache.get(entity_id)
                if target is None:
                    target = normalize_record(
                        Record(
                            entity_id=entity_id,
                            business_name=name,
                            business_address=address,
                            country=country,
                        )
                    )
                    target_cache[entity_id] = target
                evidence.append(
                    CandidateEvidence(
                        candidate_entity_id=entity_id,
                        score=score,
                        methods=_MASK_TO_METHODS[methods],
                        best_rank=rank,
                        best_rank_score=score,
                    )
                )
            results[request_id] = tuple(evidence)

        for request_id in range(len(record_rows)):
            if pending[request_id]:
                emit(request_id)
        self._last_query_targets = target_cache
        return tuple(results)

    def _group_sql(
        self,
        query_keys: list[tuple[int, str, str, str]],
        n_records: int,
    ) -> dict[int, list[tuple[float, bool, int, int]]]:
        for table_name in ("query_keys", "allowed_keys"):
            self.connection.execute(f"DROP TABLE IF EXISTS temp.{table_name}")
        self.connection.execute(
            "CREATE TEMP TABLE query_keys "
            "(request_id INTEGER, method TEXT, key TEXT, country TEXT, "
            "PRIMARY KEY (request_id, method, key))"
        )
        self.connection.execute(
            "CREATE TEMP TABLE allowed_keys "
            "(request_id INTEGER, method TEXT, key TEXT, country TEXT, "
            "PRIMARY KEY (request_id, method, key))"
        )
        self.connection.executemany(
            "INSERT INTO query_keys VALUES (?, ?, ?, ?)", query_keys
        )
        counts = self.connection.execute(
            "SELECT q.request_id, q.method, q.key, q.country, COALESCE(kc.count, 0) "
            "FROM query_keys AS q "
            "LEFT JOIN key_counts AS kc "
            "ON kc.method = q.method AND kc.key = q.key"
        ).fetchall()
        allowed = [
            (int(row[0]), row[1], row[2], row[3])
            for row in counts
            if int(row[4]) <= self.bucket_limit
        ]
        grouped: dict[int, list[tuple[float, bool, int, int]]] = {
            request_id: [] for request_id in range(n_records)
        }
        if not allowed:
            return grouped
        self.connection.executemany(
            "INSERT INTO allowed_keys VALUES (?, ?, ?, ?)", allowed
        )
        weight_case = " ".join(
            f"WHEN '{method}' THEN {weight}"
            for method, weight in _METHOD_WEIGHTS.items()
        )
        aggregated = self.connection.execute(
            "SELECT q.request_id, tk.target_id, "
            "SUM(CASE tk.method " + weight_case + " ELSE 0 END), "
            "COUNT(*), "
            "MAX(CASE WHEN t.country_key = q.country THEN 1 ELSE 0 END), "
            "GROUP_CONCAT(DISTINCT tk.method) "
            "FROM allowed_keys AS q "
            "JOIN target_keys AS tk "
            "ON tk.method = q.method AND tk.key = q.key "
            "JOIN targets AS t ON t.row_id = tk.target_id "
            "GROUP BY q.request_id, tk.target_id"
        ).fetchall()
        for request_id, target_id, base_score, method_count, same_country, methods in aggregated:
            score = float(base_score)
            if same_country:
                score += 0.15 * int(method_count)
            mask = 0
            for name in str(methods).split(","):
                mask |= _METHOD_MASKS[name]
            grouped[int(request_id)].append(
                (score, bool(same_country), int(target_id), mask)
            )
        return grouped

    def _group_postings(
        self,
        record_rows: tuple[NormalizedRecord, ...],
        query_keys: list[tuple[int, str, str, str]],
    ) -> dict[int, list[tuple[float, bool, int, int]]]:
        grouped: dict[int, list[tuple[float, bool, int, int]]] = {
            request_id: [] for request_id in range(len(record_rows))
        }
        postings = self._postings
        if postings is None:
            return grouped
        n_groups = int(postings.hashes.shape[0])
        if n_groups == 0:
            return grouped
        unique_keys = list(dict.fromkeys(query_keys))
        count = len(unique_keys)
        q_hash = np.empty(count, dtype=np.uint64)
        key_request = np.zeros(count, dtype=np.int64)
        key_method = np.zeros(count, dtype=np.uint8)
        key_country = np.full(count, 255, dtype=np.uint8)
        usable = np.ones(count, dtype=bool)
        for index, (request_id, method, key, country) in enumerate(unique_keys):
            method_id = _METHOD_IDS.get(method)
            if method_id is None:
                usable[index] = False
                continue
            q_hash[index] = _key_hash(method, key)
            key_request[index] = request_id
            key_method[index] = method_id
            country_id = postings.country_ids.get(country)
            if country_id is not None:
                key_country[index] = country_id
        positions = np.searchsorted(postings.hashes, q_hash)
        clipped = np.minimum(positions, n_groups - 1)
        valid = (
            usable
            & (positions < n_groups)
            & (postings.hashes[clipped] == q_hash)
            & (postings.counts[clipped] <= self.bucket_limit)
        )
        if not bool(valid.any()):
            return grouped
        lengths = postings.counts[clipped][valid].astype(np.int64)
        starts = postings.starts[clipped][valid].astype(np.int64)
        key_request = key_request[valid]
        key_method = key_method[valid]
        key_country = key_country[valid]
        total = int(lengths.sum())
        bases = np.cumsum(lengths) - lengths
        offsets = (
            np.arange(total, dtype=np.int64)
            - np.repeat(bases, lengths)
            + np.repeat(starts, lengths)
        )
        targets = np.asarray(postings.targets[offsets])
        methods = np.asarray(postings.methods[offsets])
        request_of = np.repeat(key_request, lengths)
        country_of = np.repeat(key_country, lengths)
        same = postings.countries[targets - 1] == country_of
        stride = np.int64(postings.row_count) + 1
        composite = request_of * stride + targets
        unique_values, inverse = np.unique(composite, return_inverse=True)
        base_scores = np.bincount(inverse, weights=_POSTING_WEIGHTS[methods])
        row_counts = np.bincount(inverse)
        same_any = np.bincount(inverse, weights=same.astype(np.float64)) > 0
        scores = base_scores + 0.15 * row_counts * same_any
        method_span = len(_METHOD_ORDER)
        method_masks = np.zeros((unique_values.size, method_span), dtype=bool)
        method_masks[inverse, methods] = True
        method_masks = method_masks @ _METHOD_BIT_WEIGHTS
        n_records = len(record_rows)
        for request_id, target_id, score, same, mask in zip(
            (unique_values // stride).tolist(),
            (unique_values % stride).tolist(),
            scores.tolist(),
            same_any.tolist(),
            method_masks.tolist(),
        ):
            if request_id < n_records:
                grouped[request_id].append((score, same, target_id, int(mask)))
        return grouped

    def query_many_with_targets(
        self,
        records: Iterable[NormalizedRecord],
        cap: int = 50,
        country_fallback_limit: int = 10,
        stage1: int | None = None,
    ) -> tuple[
        tuple[tuple[CandidateEvidence, ...], ...],
        dict[str, NormalizedRecord],
    ]:
        evidence = self.query_many(records, cap, country_fallback_limit, stage1)
        return evidence, self._last_query_targets

    def query(
        self,
        record: NormalizedRecord,
        cap: int = 50,
        country_fallback_limit: int = 10,
        stage1: int | None = None,
    ) -> tuple[CandidateEvidence, ...]:
        if cap < 1:
            return ()
        if country_fallback_limit < 0:
            raise ValueError("country_fallback_limit must be non-negative")
        return self.query_many((record,), cap, country_fallback_limit, stage1)[0]

    def iter_candidates(
        self,
        source1_records: Iterable[NormalizedRecord],
        cap: int = 50,
        country_fallback_limit: int = 10,
    ) -> Iterable[tuple[str, tuple[CandidateEvidence, ...]]]:
        for record in source1_records:
            if record.raw.source != 1:
                raise ValueError("candidate queries must use Source 1 records")
            yield record.raw.entity_id, self.query(record, cap, country_fallback_limit)

    def candidate_map(
        self,
        source1_records: Iterable[NormalizedRecord],
        cap: int = 50,
        country_fallback_limit: int = 10,
    ) -> dict[str, tuple[CandidateEvidence, ...]]:
        return dict(self.iter_candidates(source1_records, cap, country_fallback_limit))

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SqliteBlockingIndex":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
