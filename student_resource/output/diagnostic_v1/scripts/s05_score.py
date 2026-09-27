"""Score sampled validation candidates with production model (train-loop parity)."""
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/rushi/er/student_resource/code/business_entity_resolution")
from src.block import CandidateEvidence
from src.features import extract_pair_features
from src.io import Record
from src.normalize import normalize_record
from src.train import load_model

ROOT = Path("/home/rushi/er/student_resource")
DIAG = ROOT / "output" / "diagnostic_v1"
COUNTRIES = tuple(a.strip().lower() for a in sys.argv[1:] if a.strip()) or ("us", "india")
assert set(COUNTRIES) <= {"us", "india"}, COUNTRIES
t0 = time.time()

bundle = load_model(ROOT / "code/business_entity_resolution/artifacts/matcher.joblib")
print(f"model kind={bundle.kind} threshold={bundle.threshold} features={len(bundle.feature_names)}", flush=True)

samples: dict[str, dict] = {}
cands: dict[str, list] = {}
for c in COUNTRIES:
    with (DIAG / f"samples_{c}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            samples[d["s1"]] = d
    with (DIAG / f"candidates_{c}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cands[d["s1"]] = d["candidates"]
print(f"loaded samples={len(samples)} elapsed={time.time()-t0:.1f}s", flush=True)

for c in COUNTRIES:
    out = (DIAG / f"scores_{c}.jsonl").open("w", encoding="utf-8")
    sids = [d["s1"] for d in samples.values() if d["country"] == c]
    # distinct candidate ids wanted
    wanted: set[str] = set()
    for sid in sids:
        for e in cands.get(sid, []):
            wanted.add(e["id"])
    print(f"{c}: s1={len(sids)} wanted_targets={len(wanted)}", flush=True)
    con = sqlite3.connect(f"file:{DIAG / f'train_blocking_{c}.sqlite'}?mode=ro", uri=True)
    idx_list = [r[1] for r in con.execute("PRAGMA index_list(targets)")]
    print(f"{c}: targets indexes={idx_list}", flush=True)
    targets: dict[str, Record] = {}
    if any("entity_id" in str(r) for r in con.execute(
            "SELECT sql FROM sqlite_master WHERE name LIKE '%targets%'")):
        # indexed path: chunked lookup by entity_id
        wanted_list = sorted(wanted)
        for i in range(0, len(wanted_list), 900):
            chunk = wanted_list[i:i + 900]
            ph = ",".join("?" for _ in chunk)
            for eid, nm, ad, co in con.execute(
                    f"SELECT entity_id, business_name, business_address, country FROM targets WHERE entity_id IN ({ph})", chunk):
                targets[eid] = Record(eid, nm, ad, co)
    else:
        for row_id, eid, nm, ad, co in con.execute(
                "SELECT row_id, entity_id, business_name, business_address, country FROM targets"):
            if eid in wanted:
                targets[eid] = Record(eid, nm, ad, co)
                if len(targets) >= len(wanted):
                    pass
    con.close()
    print(f"{c}: targets_fetched={len(targets)} elapsed={time.time()-t0:.1f}s", flush=True)
    # features + scores
    pair_keys: list[tuple[str, str]] = []
    feat_rows: list[dict] = []
    for sid in sids:
        ev_rows = cands.get(sid, [])
        if not ev_rows:
            continue
        ref = normalize_record(Record(sid, samples[sid]["name"], samples[sid]["address"], samples[sid]["country"]))
        ordered = sorted(ev_rows, key=lambda e: (-e["score"], e["id"]))
        runner = float(ordered[1]["score"]) if len(ordered) > 1 else 0.0
        for e in ordered:
            tgt = targets.get(e["id"])
            if tgt is None:
                continue
            ev = CandidateEvidence(e["id"], float(e["score"]), frozenset(e["methods"]), int(e["rank"]), float(e["score"]))
            feat_rows.append(extract_pair_features(ref, normalize_record(tgt), evidence=ev,
                                                  candidate_count=len(ordered), runner_up_score=runner))
            pair_keys.append((sid, e["id"]))
    print(f"{c}: pairs={len(pair_keys)} elapsed={time.time()-t0:.1f}s", flush=True)
    probs: list[float] = []
    for i in range(0, len(feat_rows), 20000):
        probs.extend(bundle.predict_proba(feat_rows[i:i + 20000]))
        print(f"{c}: scored {min(i+20000,len(feat_rows))}/{len(feat_rows)} elapsed={time.time()-t0:.1f}s", flush=True)
    by_s1: dict[str, dict] = {}
    for (sid, cid), p in zip(pair_keys, probs):
        by_s1.setdefault(sid, {})[cid] = p
    for sid in sids:
        out.write(json.dumps({"s1": sid, "scores": by_s1.get(sid, {})}) + "\n")
    out.close()
print(f"DONE elapsed={time.time()-t0:.1f}s", flush=True)
