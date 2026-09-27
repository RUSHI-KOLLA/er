"""Miss attribution + top FN/FP tables per country."""
import csv
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/rushi/er/student_resource/code/business_entity_resolution")
from src.block import BlockingIndex
from src.io import Record
from src.normalize import normalize_record

ROOT = Path("/home/rushi/er/student_resource")
DIAG = ROOT / "output" / "diagnostic_v1"
COUNTRIES = tuple(a.strip().lower() for a in sys.argv[1:] if a.strip()) or ("us",)
assert set(COUNTRIES) <= {"us", "india"}, COUNTRIES
t0 = time.time()


def tok_jac(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


for c in COUNTRIES:
    samples, candrows, scorerows = {}, {}, {}
    for name, store in (("samples", samples), ("candidates", candrows), ("scores", scorerows)):
        with (DIAG / f"{name}_{c}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                store[d["s1"]] = d
    con = sqlite3.connect(f"file:{DIAG / f'train_blocking_{c}.sqlite'}?mode=ro", uri=True)
    try:
        kc_cols = [r[1] for r in con.execute("PRAGMA table_info(key_counts)")]
    except Exception:
        kc_cols = []
    print(f"{c}: key_counts cols={kc_cols}", flush=True)

    def bucket_size(method, key):
        if not kc_cols:
            return -1
        try:
            r = con.execute("SELECT count FROM key_counts WHERE method=? AND key=?", (method, key)).fetchone()
            return int(r[0]) if r else 0
        except Exception:
            return -1

    # gather all needed target ids: missed truth + fp cands
    truth = {s: set(samples[s]["truth"]) for s in samples}
    candset = {s: {e["id"] for e in candrows.get(s, {}).get("candidates", [])} for s in samples}
    scores = {s: dict(scorerows.get(s, {}).get("scores", {})) for s in samples}
    missed = [(s, t) for s in samples for t in truth[s] if t not in candset[s]]
    fp_pairs = [(s, p, scores[s][p]) for s in samples for p in
                ({p for p, sc in scores[s].items() if sc >= 0.9} - truth[s])]
    fn_scored = [(s, t, scores[s].get(t, 0.0)) for s in samples for t in truth[s]
                 if t in scores[s] and scores[s][t] < 0.9]
    need = {t for _, t in missed} | {p for _, p, _ in fp_pairs} | {t for _, t, _ in fn_scored}
    tgt: dict[str, Record] = {}
    need_list = sorted(need)
    for i in range(0, len(need_list), 900):
        chunk = need_list[i:i + 900]
        ph = ",".join("?" for _ in chunk)
        for eid, nm, ad, co in con.execute(
                f"SELECT entity_id, business_name, business_address, country FROM targets WHERE entity_id IN ({ph})", chunk):
            tgt[eid] = Record(eid, nm, ad, co)
    print(f"{c}: missed={len(missed)} fp={len(fp_pairs)} fn_scored={len(fn_scored)} targets_fetched={len(tgt)}", flush=True)

    # miss probes
    method_stats: dict[str, int] = {}
    bucket_dropped = 0
    no_shared = 0
    rows = []
    for s, t in missed:
        srec = samples[s]
        trec = tgt.get(t)
        s_norm = normalize_record(Record(s, srec["name"], srec["address"], srec["country"]))
        s_keys = set(BlockingIndex._keys(s_norm))
        if trec is None:
            rows.append({"s1": s, "true_id": t, "shared": "TARGET_NOT_IN_INDEX", "s_name": srec["name"],
                         "t_name": "", "s_addr": srec["address"], "t_addr": "", "country": srec["country"]})
            continue
        t_norm = normalize_record(trec)
        t_keys = set(BlockingIndex._keys(t_norm))
        shared = sorted(set(s_keys) & set(t_keys))
        for m, k in shared:
            method_stats[m] = method_stats.get(m, 0) + 1
        over = [f"{m}:{bucket_size(m, k)}" for m, k in s_keys if bucket_size(m, k) > 5000]
        if over:
            bucket_dropped += 1
        if not shared:
            no_shared += 1
        cf = s_norm.name.core_folded or s_norm.name.folded
        tf = t_norm.name.core_folded or t_norm.name.folded
        rows.append({
            "s1": s, "true_id": t,
            "shared_methods": "|".join(m for m, _ in shared) or "NONE",
            "overlimit_s1_keys": "|".join(over) or "none",
            "prefix4": int(cf[:4] == tf[:4]) if len(cf) >= 4 and len(tf) >= 4 else 0,
            "prefix6": int(cf[:6] == tf[:6]) if len(cf) >= 6 and len(tf) >= 6 else 0,
            "prefix8": int(cf[:8] == tf[:8]) if len(cf) >= 8 and len(tf) >= 8 else 0,
            "name_tokjac": round(tok_jac(cf, tf), 3),
            "addr_tokjac": round(tok_jac(s_norm.address.folded, t_norm.address.folded), 3),
            "postal_agree": int(bool(s_norm.address.postal_code) and s_norm.address.postal_code == t_norm.address.postal_code),
            "s_name": srec["name"], "t_name": trec.business_name,
            "s_addr": srec["address"], "t_addr": trec.business_address,
            "country": srec["country"]})
    with (DIAG / f"misses_{c}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{c}: no_shared_key={no_shared}/{len(missed)} overlimit_involved={bucket_dropped}/{len(missed)}", flush=True)
    print(f"{c}: shared_method_counts={json.dumps(method_stats, sort_keys=True)}", flush=True)

    # top FN (lowest... representative: sort by score desc = near-misses first)
    fn_scored.sort(key=lambda x: -x[2])
    with (DIAG / f"fn_top100_{c}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["s1", "true_id", "score", "s_name", "t_name", "s_addr", "t_addr", "country"])
        for s, t, sc in fn_scored[:100]:
            tr = tgt.get(t)
            w.writerow([s, t, round(sc, 4), samples[s]["name"], tr.business_name if tr else "",
                        samples[s]["address"], tr.business_address if tr else "", samples[s]["country"]])
    fp_pairs.sort(key=lambda x: -x[2])
    with (DIAG / f"fp_top100_{c}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["s1", "fp_id", "score", "s_name", "t_name", "s_addr", "t_addr", "country"])
        for s, p, sc in fp_pairs[:100]:
            tr = tgt.get(p)
            w.writerow([s, p, round(sc, 4), samples[s]["name"], tr.business_name if tr else "",
                        samples[s]["address"], tr.business_address if tr else "", samples[s]["country"]])
    con.close()
print("DONE", flush=True)
