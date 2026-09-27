"""Mix approved rows: joint France + joint India + BASELINE US (validated-best per country)."""
import csv
import sys
import time
from pathlib import Path

ROOT = Path("/home/rushi/er/student_resource")
JOINT = ROOT / "output_joint"
BASE = ROOT / "output"
OUT = ROOT / "output_mixed"
t0 = time.time()
OUT.mkdir(parents=True, exist_ok=True)


def load_matching(path):
    d = {}
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        h = next(r)
        assert h == ["source1_entity_id", "matched_entity_ids"], h
        for row in r:
            d[row[0]] = row[1] if len(row) > 1 else ""
    return d


jf = load_matching(JOINT / "parts/france/matching_results.tsv")
ji = load_matching(JOINT / "parts/india/matching_results.tsv")
bu = load_matching(BASE / "parts/us/matching_results.tsv")
print(f"loaded fr={len(jf)} in={len(ji)} us={len(bu)} elapsed={time.time()-t0:.1f}s", flush=True)
assert not (set(jf) & set(ji) & set(bu)), "cross-country S1 overlap!"

# country of each S1 from test_source1 (guard against misplaced rows)
country_of = {}
with (ROOT / "dataset/test/test_source1.tsv").open(encoding="utf-8", newline="") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        country_of[row[0]] = row[3].strip().lower()
for s in jf:
    assert country_of[s] == "france", s
for s in ji:
    assert country_of[s] == "india", s
for s in bu:
    assert country_of[s] == "us", s
print("country guards pass", flush=True)

order = sorted(country_of, key=lambda s: ({"france": 0, "india": 1, "us": 2}[country_of[s]], s))
# note: production order is file order, not sorted; validator is order-insensitive.
# use test file order instead:
order = list(country_of)
merged = {}
merged.update(jf)
merged.update(ji)
merged.update(bu)
assert len(merged) == 1732544, len(merged)
assert set(merged) == set(country_of)
with (OUT / "matching_results.tsv").open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["source1_entity_id", "matched_entity_ids"])
    for s in order:
        w.writerow([s, merged[s]])
print(f"wrote {OUT/'matching_results.tsv'} rows={len(merged)} elapsed={time.time()-t0:.1f}s DONE", flush=True)
