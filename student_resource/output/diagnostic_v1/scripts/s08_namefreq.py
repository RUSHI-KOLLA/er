"""Generic-name frequency analysis per country (watchlist counting over train S2/S3)."""
import csv
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/rushi/er/student_resource/code/business_entity_resolution")
from src.io import Record
from src.normalize import normalize_record

ROOT = Path("/home/rushi/er/student_resource")
DIAG = ROOT / "output" / "diagnostic_v1"
COUNTRIES = tuple(a.strip().lower() for a in sys.argv[1:] if a.strip()) or ("us",)
assert set(COUNTRIES) <= {"us", "india"}, COUNTRIES
t0 = time.time()


def canon(name: str) -> str:
    n = normalize_record(Record("S2-0", name, "", "us")).name
    return n.core_folded or n.folded


for c in COUNTRIES:
    samples, cands, scores = {}, {}, {}
    for name, store in (("samples", samples), ("candidates", cands), ("scores", scores)):
        with (DIAG / f"{name}_{c}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                store[d["s1"]] = d
    truth = {s: set(samples[s]["truth"]) for s in samples}
    cand_ids = {s: [e["id"] for e in cands.get(s, {}).get("candidates", [])] for s in samples}
    # names for candidates + truth targets
    need = sorted({i for s in samples for i in (cand_ids[s] + list(truth[s]))})
    name_of: dict[str, str] = {}
    con = sqlite3.connect(f"file:{DIAG / f'train_blocking_{c}.sqlite'}?mode=ro", uri=True)
    for i in range(0, len(need), 900):
        ch = need[i:i + 900]
        ph = ",".join("?" for _ in ch)
        for eid, nm in con.execute(f"SELECT entity_id, business_name FROM targets WHERE entity_id IN ({ph})", ch):
            name_of[eid] = nm
    con.close()
    # canon names for s1 + all targets
    watch: dict[str, str] = {}  # entity -> canon
    for s in samples:
        watch[s] = canon(samples[s]["name"])
    for eid, nm in name_of.items():
        watch[eid] = canon(nm)
    watch_names = set(watch.values())
    print(f"{c}: watch entities={len(watch)} distinct names={len(watch_names)}", flush=True)
    freq = Counter()
    for src in (2, 3):
        with (ROOT / f"dataset/train/train_source{src}.tsv").open(encoding="utf-8", newline="") as f:
            r = csv.reader(f, delimiter="\t")
            next(r)
            for row in r:
                cn = canon(row[1])
                if cn in watch_names:
                    freq[cn] += 1
    print(f"{c}: freq counted elapsed={time.time()-t0:.1f}s", flush=True)

    def bucket(n):
        if n <= 1:
            return "1"
        if n <= 5:
            return "2-5"
        if n <= 10:
            return "6-10"
        if n <= 50:
            return "11-50"
        if n <= 100:
            return "51-100"
        return ">100"

    stats: dict[str, dict] = {}
    for s in samples:
        for cid in cand_ids[s]:
            cn = watch.get(cid, "")
            b = bucket(freq.get(cn, 0))
            d = stats.setdefault(b, {"pairs": 0, "tp": 0, "pred": 0, "true": 0})
            sc = scores.get(s, {}).get("scores", {}).get(cid, 0.0)
            is_t = cid in truth[s]
            is_p = sc >= 0.9
            d["pairs"] += 1
            d["true"] += int(is_t)
            d["pred"] += int(is_p)
            d["tp"] += int(is_t and is_p)
    for b, d in stats.items():
        d["precision"] = d["tp"] / d["pred"] if d["pred"] else 1.0
        d["recall"] = d["tp"] / d["true"] if d["true"] else 1.0
    (DIAG / f"namefreq_{c}.json").write_text(json.dumps(
        {"buckets": stats, "top_names": freq.most_common(20)}, indent=2))
    print(f"{c}: " + json.dumps({b: {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()} for b, d in sorted(stats.items())}), flush=True)
print("DONE", flush=True)
