"""Step 2: deterministic 80/20 S1 split stats over TRAIN + reconstruct model 40k split."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/rushi/er/student_resource/code/business_entity_resolution")
from src.evaluation import entity_level_split

ROOT = Path("/home/rushi/er/student_resource")
DIAG = ROOT / "output" / "diagnostic_v1"
SEED = 2026
FRAC = 0.2

s1_country = {}
order_us40k = []
with (ROOT / "dataset/train/train_source1.tsv").open(encoding="utf-8", newline="") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        s1_country[row[0]] = row[3].strip().lower()
        if row[3].strip().lower() == "us" and len(order_us40k) < 40000:
            order_us40k.append(row[0])

all_ids = set(s1_country)
train_ids, valid_ids = entity_level_split(all_ids, validation_fraction=FRAC, seed=SEED)

# reconstruct production model split: first 40000 US S1s in file order
m_train, m_valid = entity_level_split(set(order_us40k), validation_fraction=FRAC, seed=SEED)

stats = {
    "s1_total": len(all_ids),
    "split_train": len(train_ids),
    "split_valid": len(valid_ids),
    "model_40k_us_selection": len(order_us40k),
    "model_train": len(m_train),
    "model_valid": len(m_valid),
}
# overlap of diagnostic-validation with model-training (leakage to exclude in step 5)
leak = valid_ids & m_train
stats["diag_valid_overlap_model_train"] = len(leak)

valid_edges = valid_singletons = valid_nonsingle = 0
valid_s2 = valid_s3 = 0
train_edges = 0
per_country = {}
with (ROOT / "dataset/train/train_ground_truth.tsv").open(encoding="utf-8", newline="") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        sid = row[0]
        mids = [x.strip() for x in row[1].split(",") if x.strip()]
        if sid in valid_ids:
            valid_edges += len(mids)
            if mids:
                valid_nonsingle += 1
            else:
                valid_singletons += 1
            for m in mids:
                if m.startswith("S2-"):
                    valid_s2 += 1
                elif m.startswith("S3-"):
                    valid_s3 += 1
            c = s1_country.get(sid, "?")
            d = per_country.setdefault(c, {"s1": 0, "edges": 0, "singletons": 0})
            d["s1"] += 1
            d["edges"] += len(mids)
            d["singletons"] += int(not mids)
        else:
            train_edges += len(mids)

stats.update({
    "valid_edges": valid_edges,
    "valid_singletons": valid_singletons,
    "valid_nonsingletons": valid_nonsingle,
    "valid_s2_edges": valid_s2,
    "valid_s3_edges": valid_s3,
    "train_side_edges": train_edges,
    "valid_per_country": per_country,
})
(DIAG / "split_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
# save model-train ids for exclusion in step 5
with (DIAG / "model_train_ids.txt").open("w") as f:
    for sid in sorted(m_train):
        f.write(sid + "\n")
print(json.dumps(stats, indent=2, sort_keys=True))
