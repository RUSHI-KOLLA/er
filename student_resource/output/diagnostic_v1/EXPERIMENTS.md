# Experiment Log — overnight F0.5 maximization (evidence-first)

## Baseline (recorded, NOT modified)
- Source: `output/training_summary.json` + `inference_summary.json` + `experiments.csv`
- Model: HistGradientBoosting (`artifacts/matcher.joblib`), trained on 32,054 US S1 (first-40k US file-order subset, entity_level_split seed 2026), validated on 7,946 US S1.
- Validation: F0.5 0.9196, P 0.9726, R 0.8506, threshold 0.9, blocking R@20 0.9033 / R@50 0.9187.
- Inference: bucket 5000, stage1 2000, cap 50, fallback 10, per-country partitioned, 6 shards.
- Leaderboard: ~0.80 (matching_results.tsv), boosted variant 0.804.
- Production files: UNTOUCHED. All diagnostic artifacts in `output/diagnostic_v1/`.

## Validation evidence (fresh 3000-US-S1 sample, zero overlap with model-train IDs)
- Candidate recall 0.9167 (S2 0.9237, S3 0.9104). Ceiling loss = 8.3% of true edges.
- Matcher @0.9: F0.5 0.9138, P 0.9717, R 0.8436, FP 244, FN 1621 (= 867 blocking + 754 scoring).
- Threshold curve (independent): 0.90 → 0.9138; 0.92 → 0.9149 (+0.0011); 0.95 → 0.9118. Flat 0.90–0.94.
- Miss split (867): 173 all-shared-keys-over-bucket-limit; 694 in SQL pool, lost at stage1/cap rank cut; 0 keyless.
- Generic names: precision flat 0.962–1.0 across frequency buckets. NOT a bottleneck.
- Sample-local exclusivity: zero collisions (ind == exc). Production test conflicts (5.27% edges on multi-claimed targets) unmeasurable sample-locally.

## Experiments
### E1 — threshold 0.9 → 0.92
- Hypothesis: curve peak transfers.
- Result (US valid): F0.5 0.9138 → 0.9149 (+0.0011), P +0.007, R −0.007.
- Decision: NOTE ONLY. Gain negligible; threshold choice deferred until India evidence. No production change.

### (India evidence — MEASURED)
- India valid (n=3000, zero model-train overlap): recall 0.9039 (S2 0.9085, S3 0.8994).
- Matcher @0.9: F0.5 **0.7418** (P 0.8566, R 0.6865), FP 1291 (5.3x US rate), FN 3164 (30.1%: 1010 blocking + 2154 scoring).
- India threshold curve peaks at 0.96 → 0.7506 (+0.0088). Calibration is NOT the fix.
- True-score p10 0.343 (US 0.932): systematic underconfidence on India truths. False ≥0.9 rate 0.92% (US 0.17%).
- India singletons F0.5 0.549 (US 0.847). Name-freq buckets 6–50 show P dip + R collapse (0.44).
- Test-weighted estimate (47% India): matches LB ~0.80. **Matcher country-degradation is the primary bottleneck.**
- France EDA (unlabeled): suffix list lacks eurl/sasu/sci; city/house extraction OK; non-ASCII 15.7%. No France action without labels.

### E2-pilot — small India-trained model (VALIDATED, lever proven)
- Hypothesis: country-specific training data (not architecture) is the lever.
- Change: train HistGB (60 iters) on 8000 India diag-train entities, production-identical pipeline; eval on India + US samples.
- India-pilot-model on India: F0.5 **0.8511** (th 0.77, P 0.9352, R 0.7663) vs US-model 0.7418 → **+0.109**.
- Same pilot on US: 0.8694 vs US-model 0.9138 (−0.044). Specialization confirmed both directions.
- 8000 entities / 60 iters / 175k examples were sufficient to prove the lever.
- Decision: **KEEP hypothesis → proceed to E3 full joint retrain** (40k US + 40k India, 120 iters, production parity).

### E3a — India training examples (DONE)
- 40k India file-order selection → split 31,902 train / 8,098 valid. Examples: 349,825 (pos 99,825).
- Saved `joint_examples_india.pkl` (146MB). India index deleted to free disk; US index rebuilding (~30 min).

### E3b — joint fit (VALIDATED, KEEP)
- US examples 351,557 (pos **101,557 = production count exactly** — training parity).
- Joint 701,382 examples, HistGB 120 iters (fit: 18 s). Threshold-select on combined samples → **0.88**.
- Joint @0.88 — US: F0.5 0.9050 (P 0.9672, R 0.8312) vs 0.9138 (−0.0088). India: **0.8480** (P 0.9302, R 0.7726) vs 0.7418 (**+0.106**). Combined 0.8765 vs ~0.828 (**+0.048**).
- Saved `diagnostic_v1/matcher_joint.joblib`. Production `artifacts/matcher.joblib` UNTOUCHED.
- Decision: **KEEP. Proceed to E4 full test re-inference with joint model, production-identical config, into `output_joint/` (baseline outputs preserved).**

### E4 — full test re-inference with joint model (RUNNING)
- Command (from `code/business_entity_resolution`): `infer --split test --output-dir ../../output_joint --model ../../output/diagnostic_v1/matcher_joint.joblib --candidate-cap 50 --bucket-limit 5000 --stage1 2000 --partition-by-country --shard-count 6 --keep-work`.
- Production-identical config; only model artifact differs. Threshold resolves to bundle 0.88. Baseline `output/` UNTOUCHED.
### E5 — fast path: SKIP US re-inference, mix validated-best rows (DECIDED)
- Baseline US (0.914) > joint US (0.905) on validation → re-inferring US costs 5 h for −0.003.
- Final file = joint France + joint India + BASELINE US (`s13_concat_mixed.py`, country guards,
  1,732,544 rows). Candidates = baseline `candidate_pairs.tsv` (blocking identical/deterministic).
- ETA ~2 PM (India merge) instead of ~8 PM. Manager to be killed before US index build.
