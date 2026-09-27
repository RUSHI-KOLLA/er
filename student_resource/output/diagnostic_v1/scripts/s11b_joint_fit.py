"""E3b: US examples + joint fit + threshold-select + sample eval. Needs US index present."""
import csv
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/rushi/er/student_resource/code/business_entity_resolution")
from src.block import CandidateEvidence
from src.evaluation import entity_level_split, macro_f_beta
from src.features import extract_pair_features
from src.io import Record
from src.match import decide_matches
from src.normalize import normalize_record
from src.submit import open_or_build_index
from src.train import build_training_examples, fit_model, save_model, select_threshold

ROOT = Path("/home/rushi/er/student_resource")
DIAG = ROOT / "output" / "diagnostic_v1"
SEL_LIMIT = 40000
CAP, FALLBACK, STAGE1 = 50, 10, 2000
t0 = time.time()

selected = []
with (ROOT / "dataset/train/train_source1.tsv").open(encoding="utf-8", newline="") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        if row[3].strip().lower() == "us":
            selected.append(row[0])
            if len(selected) >= SEL_LIMIT:
                break
tr_ids, va_ids = entity_level_split(set(selected), validation_fraction=0.2, seed=2026)
print(f"selected={len(selected)} train={len(tr_ids)} valid={len(va_ids)}", flush=True)
assert len(tr_ids) == 32054 and len(va_ids) == 7946, "must reproduce production split"
with (DIAG / "joint_valid_us.txt").open("w") as f:
    for s in sorted(va_ids):
        f.write(s + "\n")

want = set(tr_ids)
records = {}
with (ROOT / "dataset/train/train_source1.tsv").open(encoding="utf-8", newline="") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        if row[0] in want:
            records[row[0]] = Record(row[0], row[1], row[2], row[3])
truth = {}
with (ROOT / "dataset/train/train_ground_truth.tsv").open(encoding="utf-8", newline="") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        if row[0] in want:
            truth[row[0]] = {x.strip() for x in row[1].split(",") if x.strip()}

idx = open_or_build_index(
    [ROOT / "dataset/train/train_source2.tsv", ROOT / "dataset/train/train_source3.tsv"],
    index_path=DIAG / "train_blocking_us.sqlite",
    target_limit=None, bucket_limit=5000, country="us")
import sqlite3
con = sqlite3.connect(f"file:{DIAG / 'train_blocking_us.sqlite'}?mode=ro", uri=True)
cache = {}


def get_target(eid):
    rec = cache.get(eid)
    if rec is None:
        row = con.execute("SELECT business_name, business_address, country FROM targets WHERE entity_id=?", (eid,)).fetchone()
        rec = normalize_record(Record(eid, row[0], row[1], row[2])) if row else None
        cache[eid] = rec
    return rec


sids = sorted(tr_ids)
normed = [normalize_record(records[s]) for s in sids]
candmap = {}
for start in range(0, len(normed), 64):
    ev = idx.query_many(normed[start:start + 64], cap=CAP, country_fallback_limit=FALLBACK, stage1=STAGE1)
    for s, rows in zip(sids[start:start + 64], ev):
        candmap[s] = rows
    if start % 640 == 0:
        print(f"queried {start+64}/{len(normed)} elapsed={time.time()-t0:.1f}s", flush=True)
if hasattr(idx, "close"):
    idx.close()
us_examples = build_training_examples(normed, candmap, get_target, truth, max_pairs=500_000, seed=2026)
con.close()
del normed, candmap, cache, records
print(f"us examples={len(us_examples)} pos={sum(e.label==1 for e in us_examples)} elapsed={time.time()-t0:.1f}s", flush=True)

with (DIAG / "joint_examples_india.pkl").open("rb") as f:
    in_examples = pickle.load(f)
examples = us_examples + in_examples
del us_examples, in_examples
print(f"joint examples={len(examples)} elapsed={time.time()-t0:.1f}s", flush=True)
model = fit_model(examples, seed=2026, max_iterations=120)
del examples
print(f"joint model kind={model.kind} elapsed={time.time()-t0:.1f}s", flush=True)

# threshold-select + eval on cached 3000-samples (TSV target fetch, no index)
pool: dict[str, Record] = {}
for src in (2, 3):
    with (ROOT / f"dataset/train/train_source{src}.tsv").open(encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            pool[row[0]] = Record(row[0], row[1], row[2], row[3])
print(f"target pool={len(pool)} elapsed={time.time()-t0:.1f}s", flush=True)


def load_json(name):
    d = {}
    with (DIAG / name).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d[r["s1"]] = r
    return d


all_scores, all_truth, per_country = {}, {}, {}
for c in ("us", "india"):
    samples = load_json(f"samples_{c}.jsonl")
    cands = load_json(f"candidates_{c}.jsonl")
    sids_c = sorted(samples)
    tru = {s: set(samples[s]["truth"]) for s in sids_c}
    per_country[c] = (sids_c, tru)
    for s in sids_c:
        ev_rows = cands.get(s, {}).get("candidates", [])
        if not ev_rows:
            all_scores[s] = {}
            continue
        ref = normalize_record(Record(s, samples[s]["name"], samples[s]["address"], samples[s]["country"]))
        ordered = sorted(ev_rows, key=lambda e: (-e["score"], e["id"]))
        runner = float(ordered[1]["score"]) if len(ordered) > 1 else 0.0
        feats, keys = [], []
        for e in ordered:
            t = pool.get(e["id"])
            if t is None:
                continue
            ev = CandidateEvidence(e["id"], float(e["score"]), frozenset(e["methods"]), int(e["rank"]), float(e["score"]))
            feats.append(extract_pair_features(ref, normalize_record(t), evidence=ev,
                                              candidate_count=len(ordered), runner_up_score=runner))
            keys.append(e["id"])
        probs = model.predict_proba(feats)
        all_scores[s] = dict(zip(keys, (float(p) for p in probs)))
    print(f"scored {c} elapsed={time.time()-t0:.1f}s", flush=True)
del pool

all_truth = {}
for c in ("us", "india"):
    all_truth.update(per_country[c][1])
th, summ = select_threshold(all_scores, all_truth,
                            candidates={s: set(all_scores[s]) for s in all_scores}, enforce_exclusivity=True)
model.threshold = th
save_model(model, DIAG / "matcher_joint.joblib")
rep = {"threshold": th, "examples_us_in": True}
for c in ("us", "india"):
    sids_c, tru = per_country[c]
    pred = decide_matches({s: all_scores[s] for s in sids_c}, candidates={s: set(all_scores[s]) for s in sids_c},
                          threshold=th, enforce_exclusivity=True).predictions
    m = macro_f_beta(pred, tru, beta=0.5).as_dict()
    rep[c] = {"f05": m["f_0.5"], "p": m["precision"], "r": m["recall"]}
m_all = macro_f_beta(decide_matches(all_scores, candidates={s: set(all_scores[s]) for s in all_scores},
                                    threshold=th, enforce_exclusivity=True).predictions, all_truth, beta=0.5).as_dict()
rep["combined"] = {"f05": m_all["f_0.5"], "p": m_all["precision"], "r": m_all["recall"]}
(DIAG / "joint_summary.json").write_text(json.dumps(rep, indent=2))
print("JOINT " + json.dumps(rep) + f" elapsed={time.time()-t0:.1f}s DONE", flush=True)
