"""Per-country diagnostic analysis: recall, matcher, thresholds, margins, misses, singletons."""
import csv
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/rushi/er/student_resource/code/business_entity_resolution")
from src.block import BlockingIndex
from src.evaluation import blocking_recall_at_k, f_beta, macro_f_beta
from src.io import Record
from src.match import decide_matches, independent_threshold
from src.normalize import normalize_record

ROOT = Path("/home/rushi/er/student_resource")
DIAG = ROOT / "output" / "diagnostic_v1"
COUNTRIES = tuple(a.strip().lower() for a in sys.argv[1:] if a.strip()) or ("us",)
assert set(COUNTRIES) <= {"us", "india"}, COUNTRIES
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
t0 = time.time()


def load_jsonl(path):
    out = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[d["s1"]] = d
    return out


for c in COUNTRIES:
    samples = load_jsonl(DIAG / f"samples_{c}.jsonl")
    candrows = load_jsonl(DIAG / f"candidates_{c}.jsonl")
    scorerows = load_jsonl(DIAG / f"scores_{c}.jsonl")
    sids = sorted(samples)
    truth = {s: set(samples[s]["truth"]) for s in sids}
    candmap = {s: [e["id"] for e in candrows.get(s, {}).get("candidates", [])] for s in sids}
    candset = {s: set(v) for s, v in candmap.items()}
    scores = {s: dict(scorerows.get(s, {}).get("scores", {})) for s in sids}
    res: dict = {"country": c, "n_s1": len(sids)}

    # ---- candidate recall ----
    n_true = sum(len(v) for v in truth.values())
    n_rec = sum(len(candset[s] & truth[s]) for s in sids)
    res["cand_recall_overall"] = n_rec / n_true if n_true else 1.0
    res["truth_edges"] = n_true
    res["recovered_edges"] = n_rec
    for prefix, label in (("S2-", "s2"), ("S3-", "s3")):
        tt = sum(1 for s in sids for t in truth[s] if t.startswith(prefix))
        rr = sum(1 for s in sids for t in truth[s] if t.startswith(prefix) and t in candset[s])
        res[f"cand_recall_{label}"] = rr / tt if tt else 1.0
        res[f"truth_edges_{label}"] = tt
    counts = sorted(len(candmap[s]) for s in sids)
    import statistics
    res["cand_count"] = {"mean": sum(counts) / len(counts), "min": counts[0], "max": counts[-1],
                         "p50": counts[len(counts) // 2],
                         "p90": counts[int(len(counts) * 0.90)], "p99": counts[int(len(counts) * 0.99)]}
    buckets = {"0": 0, "1-5": 0, "6-10": 0, "11-20": 0, "21-50": 0, ">50": 0}
    for n in counts:
        if n == 0:
            buckets["0"] += 1
        elif n <= 5:
            buckets["1-5"] += 1
        elif n <= 10:
            buckets["6-10"] += 1
        elif n <= 20:
            buckets["11-20"] += 1
        elif n <= 50:
            buckets["21-50"] += 1
        else:
            buckets[">50"] += 1
    res["cand_buckets"] = buckets
    # per-s1 recall rows
    per_s1_recall = []
    for s in sids:
        t = len(truth[s])
        r = len(candset[s] & truth[s]) if t else -1
        per_s1_recall.append((s, len(candmap[s]), t, r))

    # ---- matcher @0.9 ----
    def split_metrics(pred, tru, tag):
        m = macro_f_beta(pred, tru, beta=0.5).as_dict()
        m1 = macro_f_beta(pred, tru, beta=1.0).as_dict(beta=1.0)
        pe = sum(len(v) for v in pred.values())
        te = sum(len(v) for v in tru.values())
        fp = sum(len(set(pred.get(s, ())) - tru.get(s, set())) for s in tru)
        fn = sum(len(tru.get(s, set()) - set(pred.get(s, ()))) for s in tru)
        return {"f05": m["f_0.5"], "precision": m["precision"], "recall": m["recall"],
                "f1": m1["f_1.0"], "pred_edges": pe, "true_edges": te, "fp": fp, "fn": fn,
                "pred_singletons": m["predicted_singletons"], "true_singletons": m["true_singletons"],
                "correct_singletons": m["correct_singletons"], "entities": m["entity_count"]}

    pred_ind = independent_threshold(scores, 0.9)
    pred_exc = decide_matches(scores, candidates=candset, threshold=0.9, enforce_exclusivity=True).predictions
    res["matcher_ind_09"] = split_metrics(pred_ind, truth, "ind")
    res["matcher_exc_09"] = split_metrics(pred_exc, truth, "exc")
    for prefix, label in (("S2-", "s2"), ("S3-", "s3")):
        tru = {s: {t for t in truth[s] if t.startswith(prefix)} for s in sids}
        res[f"matcher_exc_09_{label}"] = split_metrics(
            {s: {p for p in pred_exc.get(s, set()) if p.startswith(prefix)} for s in sids}, tru, label)
    # singleton split
    for want_single, label in ((True, "singleton"), (False, "nonsingleton")):
        sub = [s for s in sids if (len(truth[s]) == 0) == want_single]
        tru = {s: truth[s] for s in sub}
        pr = {s: pred_exc.get(s, set()) for s in sub}
        res[f"matcher_exc_09_{label}"] = split_metrics(pr, tru, label)

    # ---- threshold curves ----
    curve_ind, curve_exc = [], []
    for th in THRESHOLDS:
        pi = independent_threshold(scores, th)
        mi = macro_f_beta(pi, truth, beta=0.5).as_dict()
        curve_ind.append({"th": th, "f05": mi["f_0.5"], "p": mi["precision"], "r": mi["recall"],
                          "edges": sum(len(v) for v in pi.values()),
                          "empty": mi["predicted_singletons"]})
        pe = decide_matches(scores, candidates=candset, threshold=th, enforce_exclusivity=True).predictions
        me = macro_f_beta(pe, truth, beta=0.5).as_dict()
        curve_exc.append({"th": th, "f05": me["f_0.5"], "p": me["precision"], "r": me["recall"],
                          "edges": sum(len(v) for v in pe.values()),
                          "empty": me["predicted_singletons"]})
    res["curve_ind"] = curve_ind
    res["curve_exc"] = curve_exc

    # ---- margins ----
    margins, best_scores = [], []
    for s in sids:
        sc = sorted(scores[s].values(), reverse=True)
        best = sc[0] if sc else 0.0
        second = sc[1] if len(sc) > 1 else 0.0
        best_scores.append(best)
        margins.append(best - second)
    res["margin"] = {"mean_best": sum(best_scores) / len(best_scores),
                     "mean_margin": sum(margins) / len(margins)}
    for mcut in (0.05, 0.10, 0.20):
        res["margin"][f"n_ge09_margin_ge_{mcut}"] = sum(
            1 for s in sids for _ in [0]
            if scores[s] and max(scores[s].values()) >= 0.9 and
            (max(scores[s].values()) - (sorted(scores[s].values(), reverse=True)[1] if len(scores[s]) > 1 else 0.0)) >= mcut)
    # true vs false score distributions
    true_sc, false_sc = [], []
    for s in sids:
        for cid, sc in scores[s].items():
            (true_sc if cid in truth[s] else false_sc).append(sc)
    def dist(v):
        v = sorted(v)
        return {"n": len(v), "mean": sum(v) / len(v) if v else 0.0,
                "p50": v[len(v) // 2] if v else 0.0, "p10": v[int(len(v) * 0.1)] if v else 0.0,
                "frac_ge_09": sum(1 for x in v if x >= 0.9) / len(v) if v else 0.0}
    res["score_dist_true"] = dist(true_sc)
    res["score_dist_false"] = dist(false_sc)

    (DIAG / f"metrics_{c}.json").write_text(json.dumps(res, indent=2))
    print(f"{c}: recall={res['cand_recall_overall']:.4f} ind09_f05={res['matcher_ind_09']['f05']:.4f} "
          f"exc09_f05={res['matcher_exc_09']['f05']:.4f} elapsed={time.time()-t0:.1f}s", flush=True)
print("DONE", flush=True)
