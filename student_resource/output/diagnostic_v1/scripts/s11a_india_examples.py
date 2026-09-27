"""E3a: generate India training examples (first-40k file-order selection, production parity)."""
import csv
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/rushi/er/student_resource/code/business_entity_resolution")
from src.evaluation import entity_level_split
from src.io import Record
from src.normalize import normalize_record
from src.submit import open_or_build_index
from src.train import build_training_examples

ROOT = Path("/home/rushi/er/student_resource")
DIAG = ROOT / "output" / "diagnostic_v1"
SEL_LIMIT = 40000
CAP, FALLBACK, STAGE1 = 50, 10, 2000
t0 = time.time()

# first-40k India S1s in file order (mirrors _sample_entity_ids)
selected = []
with (ROOT / "dataset/train/train_source1.tsv").open(encoding="utf-8", newline="") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        if row[3].strip().lower() == "india":
            selected.append(row[0])
            if len(selected) >= SEL_LIMIT:
                break
tr_ids, va_ids = entity_level_split(set(selected), validation_fraction=0.2, seed=2026)
print(f"selected={len(selected)} train={len(tr_ids)} valid={len(va_ids)}", flush=True)
with (DIAG / "joint_valid_india.txt").open("w") as f:
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
print(f"records={len(records)} elapsed={time.time()-t0:.1f}s", flush=True)

idx = open_or_build_index(
    [ROOT / "dataset/train/train_source2.tsv", ROOT / "dataset/train/train_source3.tsv"],
    index_path=DIAG / "train_blocking_india.sqlite",
    target_limit=None, bucket_limit=5000, country="india")
import sqlite3
con = sqlite3.connect(f"file:{DIAG / 'train_blocking_india.sqlite'}?mode=ro", uri=True)
cache: dict[str, object] = {}


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
examples = build_training_examples(normed, candmap, get_target, truth, max_pairs=500_000, seed=2026)
con.close()
print(f"india examples={len(examples)} pos={sum(e.label==1 for e in examples)} elapsed={time.time()-t0:.1f}s", flush=True)
with (DIAG / "joint_examples_india.pkl").open("wb") as f:
    pickle.dump(examples, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"saved elapsed={time.time()-t0:.1f}s DONE", flush=True)
