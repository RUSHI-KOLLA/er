# Night session summary (27 Sep 2026) — read this first for any future use

## What this repo state contains (new vs Day-1 baseline)
- Day-1 baseline: US-only HistGB matcher (`code/.../artifacts/matcher.joblib`, threshold 0.9,
  train F0.5 0.9196) → leaderboard ~0.80. Its outputs are NOT in git (see `output/`, git-ignored).
- New: **joint US+India matcher** (`student_resource/output/diagnostic_v1/matcher_joint.joblib`,
  threshold 0.88). Validated on 6000 held-out train S1s (zero overlap with baseline training IDs):
  US F0.5 0.9050 (was 0.9138, −0.009), India F0.5 **0.8480** (was 0.7418, **+0.106**),
  combined 0.8765 (was ~0.828, +0.048).
- New: full diagnostic evidence under `student_resource/output/diagnostic_v1/`:
  `DIAGNOSTIC_REPORT.md` (start here), `EXPERIMENTS.md`, `metrics_{us,india}.json`,
  `misses_{us,india}.csv` + top-100 FN/FP tables, `namefreq_*.json`, `joint_summary.json`,
  `scripts/s02–s13_*.py` (runnable pipeline mirrors), `matcher_joint.joblib`.
- New: code changes — `src/block.py`, `src/submit.py`, `tests/test_partitioning.py`,
  `run_ranges.py`, `tests/test_resume.py`, `experiments.csv` row(s).
- Salvaged output: `student_resource/output_joint/parts/france/matching_results.tsv`
  (259,452 rows, joint-model France partition — the only finished joint inference output).

## Key validated findings
1. Bottleneck = matcher country-degradation (US-only training), NOT blocking (recall
   US 0.9167 / India 0.9039), NOT threshold (curves flat ±0.009), NOT generic names (US
   precision flat across frequency buckets).
2. Blocking misses: 20% bucket-overflow (all shared keys >5000), 80% stage1=2000/cap=50
   rank cuts, 0% keyless. Raising stage1/bucket is the next lever (+0.005–0.015 est.).
3. Test-weighted estimate with joint model ≈ 0.85–0.87. 0.95 was mathematically out of
   reach (0.38×0.905 + 0.47×0.848 = 0.743 banked; France would need >1.0).
4. Production test inference was killed mid-India-partition when the hackathon ended;
   to reproduce: `infer --split test --output-dir <dir> --model <matcher_joint.joblib>
   --candidate-cap 50 --bucket-limit 5000 --stage1 2000 --partition-by-country --shard-count 6`
   then mix joint France+India rows with baseline US rows (baseline US 0.914 > joint 0.905).

## NOT in git (were git-ignored, deleted with local setup)
- `student_resource/output/` (baseline 0.80/0.804 submissions, 1.1 GB candidates),
  `student_resource/dataset/` (2.5 GB challenge data, re-download from portal),
  all `*.sqlite` blocking/scoring indexes, `output_joint/` India/US partitions.
