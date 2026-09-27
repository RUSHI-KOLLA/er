"""Sample 3000 US + 3000 India diag-valid S1s; query production-config candidates."""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/rushi/er/student_resource/code/business_entity_resolution")
from src.evaluation import entity_level_split
from src.io import Record
from src.normalize import normalize_record
from src.submit import open_or_build_index

ROOT = Path("/home/rushi/er/student_resource")
DIAG = ROOT / "output" / "diagnostic_v1"
PER_COUNTRY = 3000
CAP, FALLBACK, STAGE1 = 50, 10, 2000
SEED = 2026
COUNTRIES = tuple(a.strip().lower() for a in sys.argv[1:] if a.strip()) or ("us", "india")
assert set(COUNTRIES) <= {"us", "india"}, COUNTRIES


def rank_key(sid: str) -> int:
    d = hashlib.blake2b(f"diag-sample:{sid}".encode(), digest_size=8).digest()
    return int.from_bytes(d, "big")


t0 = time.time()
# 1. diag-valid ids per country
country_of = {}
with (ROOT / "dataset/train/train_source1.tsv").open(encoding="utf-8", newline="") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        country_of[row[0]] = row[3].strip().lower()
_, valid_ids = entity_level_split(set(country_of), validation_fraction=0.2, seed=SEED)
by_country: dict[str, list] = {"us": [], "india": []}
for sid in valid_ids:
    c = country_of.get(sid)
    if c in by_country and c in COUNTRIES:
        by_country[c].append(sid)
sampled = {}
for c, ids in by_country.items():
    ids.sort(key=rank_key)
    sampled[c] = ids[:PER_COUNTRY]
print(f"sampled {[ (c, len(sampled[c])) for c in COUNTRIES ]} elapsed={time.time()-t0:.1f}s", flush=True)

# 2. records + truth for sampled
want = set(sampled["us"]) | set(sampled["india"])
records: dict[str, Record] = {}
with (ROOT / "dataset/train/train_source1.tsv").open(encoding="utf-8", newline="") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        if row[0] in want:
            records[row[0]] = Record(row[0], row[1], row[2], row[3])
            if len(records) >= len(want):
                pass
truth: dict[str, list] = {sid: [] for sid in want}
with (ROOT / "dataset/train/train_ground_truth.tsv").open(encoding="utf-8", newline="") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        if row[0] in want:
            truth[row[0]] = [x.strip() for x in row[1].split(",") if x.strip()]
print(f"records={len(records)} elapsed={time.time()-t0:.1f}s", flush=True)
for c in COUNTRIES:
    with (DIAG / f"samples_{c}.jsonl").open("w", encoding="utf-8") as f:
        for sid in sampled[c]:
            rec = records[sid]
            f.write(json.dumps({"s1": sid, "country": c, "name": rec.business_name,
                                "address": rec.business_address, "truth": truth[sid]},
                               ensure_ascii=False) + "\n")

# 3. query per-country indexes
total_cands = 0
for c in COUNTRIES:
    out = (DIAG / f"candidates_{c}.jsonl").open("w", encoding="utf-8")
    idx = open_or_build_index(
        [ROOT / "dataset/train/train_source2.tsv", ROOT / "dataset/train/train_source3.tsv"],
        index_path=DIAG / f"train_blocking_{c}.sqlite",
        target_limit=None, bucket_limit=5000, country=c)
    try:
        sids = sampled[c]
        normed = [normalize_record(records[s]) for s in sids]
        for start in range(0, len(normed), 64):
            batch_ids = sids[start:start + 64]
            ev = idx.query_many(normed[start:start + 64], cap=CAP,
                                country_fallback_limit=FALLBACK, stage1=STAGE1)
            for sid, rows in zip(batch_ids, ev):
                cands = [{"id": e.candidate_entity_id, "score": e.score,
                          "rank": e.best_rank, "methods": sorted(e.methods)} for e in rows]
                total_cands += len(cands)
                out.write(json.dumps({"s1": sid, "candidates": cands}) + "\n")
            print(f"{c} queried {start+len(batch_ids)}/{len(sids)} elapsed={time.time()-t0:.1f}s", flush=True)
    finally:
        if hasattr(idx, "close"):
            idx.close()
    out.close()
print(f"DONE total_candidates={total_cands} elapsed={time.time()-t0:.1f}s", flush=True)
