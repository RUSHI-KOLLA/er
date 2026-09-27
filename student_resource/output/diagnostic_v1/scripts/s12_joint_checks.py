"""Post-inference checks for output_joint: counts, validator, distribution comparison."""
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/rushi/er/student_resource")
JOINT = ROOT / "output_joint"
BASE = ROOT / "output"


def dist(path):
    c = Counter()
    n = 0
    total_edges = 0
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r)
        for row in r:
            n += 1
            ids = [x for x in row[1].split(",") if x.strip()] if len(row) > 1 else []
            k = len(ids) if len(ids) <= 25 else ">25"
            c[k] += 1
            total_edges += len(ids)
    return header, n, total_edges, c


for label, path in (("joint_match", JOINT / "matching_results.tsv"),
                    ("joint_cand", JOINT / "candidate_pairs.tsv")):
    h, n, e, c = dist(path)
    print(f"{label}: header={h} rows={n} edges={e} empty={c[0]}")
    assert n == 1732544, f"{label} row count {n}"

h, n, e, c = dist(BASE / "matching_results.tsv")
print(f"base_match: rows={n} edges={e} empty={c[0]}")
hj, nj, ej, cj = dist(JOINT / "matching_results.tsv")
print(f"edge delta joint-base: {ej - e:+d}; empty delta: {cj[0] - c[0]:+d}")

# official validator
p = subprocess.run(
    [sys.executable, "utils/validate_submission.py", "--matching",
     str(JOINT / "matching_results.tsv"), "--candidate",
     str(JOINT / "candidate_pairs.tsv"), "--test-dir", "dataset/test"],
    cwd=ROOT, capture_output=True, text=True, timeout=1200)
print("VALIDATOR rc=", p.returncode)
print((p.stdout + p.stderr)[-2000:])
(Path("/home/rushi/er/student_resource/output/diagnostic_v1") / "joint_checks.json").write_text(
    json.dumps({"joint_rows": nj, "joint_edges": ej, "joint_empty": cj[0],
                "base_edges": e, "base_empty": c[0], "validator_rc": p.returncode}, indent=2))
print("CHECKS DONE")
