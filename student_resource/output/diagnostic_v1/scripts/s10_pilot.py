"""E2-pilot: small India-trained HistGB; eval on India + US samples (no US index needed)."""
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/rushi/er/student_resource/code/business_entity_resolution")
from src.block import CandidateEvidence
from src.evaluation import entity_level_split, macro_f_beta
from src.features import extract_pair_features
from src.io import Record
from src.match import decide_matches, independent_threshold
from src.normalize import normalize_record
from src.submit import open_or_build_index
from src.train import build_training_examples, fit_model, load_model, select_threshold

ROOT = Path("/home/rushi/er/student_resource")
DIAG = ROOT / "output" / "diagnostic_v1"
N_TRAIN = 8000
CAP, FALLBACK, STAGE1 = 50, 10, 2000
t0 = time.time()


def rank_key(sid, tag):
    import hashlib
    d = hashlib.blake2b(f"{tag}:{sid}".encode(), digest_size=8).digest()
    return int.from_bytes(d, "big")


# 1. diag-train-side India entities, deterministic
country_of = {}
with (ROOT / "dataset/train/train_source1.tsv").open(encoding="utf-8", newline="") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        country_of[row[0]] = row[3].strip().lower()
tr_ids, _ = entity_level_split(set(country_of), validation_fraction=0.2, seed=2026)
india_tr = sorted([s for s in tr_ids if country_of[s] == "india"], key=lambda s: rank_key(s, "pilot-train"))[:N_TRAIN]
print(f"pilot train entities={len(india_tr)} elapsed={time.time()-t0:.1f}s", flush=True)

# 2. records + truth
want = set(india_tr)
records = {}
with (ROOT / "dataset/train/train_source1.tsv").open(encoding="utf-8", newline="") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        if row[0] in want:
            records[row[0]] = Record(row[0], row[1], row[2], row[3])
truth_all = {}
with (ROOT / "dataset/train/train_ground_truth.tsv").open(encoding="utf-8", newline="") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        if row[0] in want:
            truth_all[row[0]] = {x.strip() for x in row[1].split(",") if x.strip()}

# 3. candidates via India index
idx = open_or_build_index(
    [ROOT / "dataset/train/train_source2.tsv", ROOT / "dataset/train/train_source3.tsv"],
    index_path=DIAG / "train_blocking_india.sqlite",
    target_limit=None, bucket_limit=5000, country="india")
import sqlite3
con = sqlite3.connect(f"file:{DIAG / 'train_blocking_india.sqlite'}?mode=ro", uri=True)
tgt_cache: dict[str, Record] = {}


def get_target(eid):
    rec = tgt_cache.get(eid)
    if rec is None:
        row = con.execute("SELECT business_name, business_address, country FROM targets WHERE entity_id=?", (eid,)).fetchone()
        rec = normalize_record(Record(eid, row[0], row[1], row[2])) if row else None
        tgt_cache[eid] = rec
    return rec


normed = [normalize_record(records[s]) for s in india_tr]
ev_all = []
for start in range(0, len(normed), 64):
    ev_all.extend(idx.query_many(normed[start:start + 64], cap=CAP, country_fallback_limit=FALLBACK, stage1=STAGE1))
    if start % 640 == 0:
        print(f"queried {start+64}/{len(normed)} elapsed={time.time()-t0:.1f}s", flush=True)
candmap = {s: ev for s, ev in zip(india_tr, ev_all)}
if hasattr(idx, "close"):
    idx.close()
print(f"candidates done elapsed={time.time()-t0:.1f}s", flush=True)

# 4. training examples (production parity: runner-up, candidate_count)
examples = build_training_examples(
    [normalize_record(records[s]) for s in india_tr], candmap, get_target, truth_all,
    max_pairs=300_000, seed=2026)
n_pos = sum(e.label == 1 for e in examples)
print(f"examples={len(examples)} pos={n_pos} neg={len(examples)-n_pos} elapsed={time.time()-t0:.1f}s", flush=True)
model = fit_model(examples, seed=2026, max_iterations=60)
print(f"pilot model kind={model.kind} elapsed={time.time()-t0:.1f}s", flush=True)
con.close()

# 5. eval on India sample (cached candidates) + US sample (TSV-fetched targets, cached evidence)
def load_json(name):
    d = {}
    with (DIAG / name).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d[r["s1"]] = r
    return d

for c in ("india", "us"):
    samples = load_json(f"samples_{c}.jsonl")
    cands = load_json(f"candidates_{c}.jsonl")
    sids = sorted(samples)
    tru = {s: set(samples[s]["truth"]) for s in sids}
    # target texts via TSV scan (no index needed)
    if c == "india":
        # reuse india index cache approach: reopen read-only
        con2 = sqlite3.connect(f"file:{DIAG / 'train_blocking_india.sqlite'}?mode=ro", uri=True)
        def fetch(eid):
            row = con2.execute("SELECT business_name, business_address, country FROM targets WHERE entity_id=?", (eid,)).fetchone()
            return Record(eid, row[0], row[1], row[2]) if row else None
    else:
        pool = {}
        for src in (2, 3):
            with (ROOT / f"dataset/train/train_source{src}.tsv").open(encoding="utf-8", newline="") as f:
                r = csv.reader(f, delimiter="\t")
                next(r)
                for row in r:
                    if row[3].strip().lower() == "us":
                        pool[row[0]] = Record(row[0], row[1], row[2], row[3])
        def fetch(eid):
            return pool.get(eid)
    scores = {}
    for s in sids:
        ev_rows = cands.get(s, {}).get("candidates", [])
        if not ev_rows:
            scores[s] = {}
            continue
        ref = normalize_record(Record(s, samples[s]["name"], samples[s]["address"], samples[s]["country"]))
        ordered = sorted(ev_rows, key=lambda e: (-e["score"], e["id"]))
        runner = float(ordered[1]["score"]) if len(ordered) > 1 else 0.0
        feats, keys = [], []
        for e in ordered:
            t = fetch(e["id"])
            if t is None:
                continue
            ev = CandidateEvidence(e["id"], float(e["score"]), frozenset(e["methods"]), int(e["rank"]), float(e["score"]))
            feats.append(extract_pair_features(ref, normalize_record(t), evidence=ev,
                                              candidate_count=len(ordered), runner_up_score=runner))
            keys.append(e["id"])
        probs = model.predict_proba(feats)
        scores[s] = dict(zip(keys, (float(p) for p in probs)))
    if c == "india":
        con2.close()
    th, summ = select_threshold(scores, tru, candidates={s: set(scores[s]) for s in sids}, enforce_exclusivity=True)
    m = summ.as_dict()
    m1 = macro_f_beta(decide_matches(scores, candidates={s: set(scores[s]) for s in sids},
                                     threshold=th, enforce_exclusivity=True).predictions, tru, beta=1.0).as_dict(beta=1.0)
    print(f"PILOT-{c}: th={th} F05={m['f_0.5']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} F1={m1['f_1.0']:.4f} "
          f"edges={sum(len(v) for v in scores.values())} elapsed={time.time()-t0:.1f}s", flush=True)
print(f"DONE elapsed={time.time()-t0:.1f}s", flush=True)
