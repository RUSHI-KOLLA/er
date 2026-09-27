"""Unlabeled test-distribution inspection: France vs US/India (no labels used)."""
import csv
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/rushi/er/student_resource/code/business_entity_resolution")
from src.io import Record
from src.normalize import normalize_record

ROOT = Path("/home/rushi/er/student_resource")
t0 = time.time()
N = 50000
stats = {}
toks = {}
for c in ("france", "us", "india"):
    stats[c] = {"n": 0, "postal": 0, "city": 0, "house": 0, "nonascii_name": 0,
                "name_tokens": 0, "addr_tokens": 0, "blank_addr": 0}
    toks[c] = Counter()

with (ROOT / "dataset/test/test_source1.tsv").open(encoding="utf-8", newline="") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        c = row[3].strip().lower()
        if c not in stats or stats[c]["n"] >= N:
            continue
        rec = Record(row[0], row[1], row[2], row[3])
        nr = normalize_record(rec)
        d = stats[c]
        d["n"] += 1
        d["postal"] += int(bool(nr.address.postal_code))
        d["city"] += int(bool(nr.address.city))
        d["house"] += int(bool(nr.address.house_number))
        d["nonascii_name"] += int(any(ord(ch) > 127 for ch in row[1]))
        d["name_tokens"] += len((nr.name.core_folded or nr.name.folded).split())
        d["addr_tokens"] += len(nr.address.folded.split())
        d["blank_addr"] += int(not row[2].strip())
        for t in (nr.name.core_folded or nr.name.folded).split():
            toks[c][t] += 1
        if all(v["n"] >= N for v in stats.values()):
            break

for c, d in stats.items():
    n = d["n"]
    print(f"{c}: n={n} postal={d['postal']/n:.3f} city={d['city']/n:.3f} house={d['house']/n:.3f} "
          f"nonascii={d['nonascii_name']/n:.3f} name_tok={d['name_tokens']/n:.2f} "
          f"addr_tok={d['addr_tokens']/n:.2f} blank_addr={d['blank_addr']/n:.4f}")
    print(f"  top name tokens: {toks[c].most_common(15)}")
print(f"elapsed={time.time()-t0:.1f}s", flush=True)
