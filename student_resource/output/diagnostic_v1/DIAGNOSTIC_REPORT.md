# AMAZON ML PRE-SUBMISSION DIAGNOSTIC
==================================================
Current leaderboard: ~0.80
Submissions remaining: 3

Candidate Recall: US 91.67% / India 90.39% (3000-S1 samples, production config)
Current Local Precision: 94.87%
Current Local Recall: 80.19%
Current Local F0.5: 0.8765 (joint US+India model @0.88, 6000-S1 combined sample)

Primary Bottleneck:
MATCHER

Confidence:
HIGH
==================================================

All measurements use TRAIN ONLY with held-out S1 splits (entity_level_split, seed 2026,
fraction 0.2). Diagnostic validation sets have ZERO overlap with the production model's
32,054 training IDs (same hash function: overlap is structurally impossible; the 7,946
production-validation IDs are a subset of the diagnostic-validation pool). No test labels
used. No production files modified. Production baseline (`artifacts/matcher.joblib`,
`output/matching_results.tsv` LB ~0.80) preserved untouched.

## 1. Candidate recall
- Method: per-country TRAIN blocking indexes rebuilt with production config (bucket 5000,
  stage1 2000, cap 50, fallback 10; metadata row_counts match production exactly:
  US 6,186,873 / India 4,133,346 targets). 3000 diag-valid S1s per country, `query_many`.
- US: 9,544/10,411 = **0.9167** (S2 0.9237, S3 0.9104). India: 9,496/10,506 = **0.9039**
  (S2 0.9085, S3 0.8994). 95% CI ≈ ±0.6%.
- Candidate counts: mean 49.9 (US) / 49.8 (India); p50/p90/p99 = 50; min 9 (US) / 0 (India,
  3 S1s). Buckets US: 0→0, 1–5→0, 6–10→2, 11–20→2, 21–50→2996. Cap almost always fills.
- France recall: UNMEASURABLE (no train GT contains France; France is test-only).
- Singletons are excluded from recall denominator by definition (no true edges); 150 US /
  153 India true singletons in samples, all with ~50 candidates available.

## 2. Candidate-generation failures
- Miss tables: `misses_us.csv` (867 rows), `misses_india.csv` (1,010 rows) with record texts.
- Exact attribution via pipeline's own `BlockingIndex._keys` + indexed `key_counts`:
  - US: **0/867** share no key; 173 (20.0%) all shared keys over bucket_limit → invisible;
    694 (80.0%) had ≥1 under-limit shared key (e.g. address_token 'phillips' n=2467) but lost
    at stage1=2000 rerank cut or cap=50 cut.
  - India: 0/1,010 keyless; 1,010/1,010 touch an over-limit key (refinement by shared-key
    under-limit status pending; same pattern expected).
- Shared-method counts (US misses): address_token 1752 events, name_token 1246, name_prefix
  704, address_number 419, address_house 304, address_city 269, name_core_exact 78,
  name_exact 19, name_token_sort 5. Misses share weak keys, never exact-name keys.
- Conclusion: no missing key TYPE (0 keyless). Losses split bucket-overflow vs rank-depth.

## 3. Current matcher metrics (production US-only model @0.9, exclusive)
- US (n=3000): F0.5 0.9138, P 0.9717, R 0.8436, F1 0.8786; edges 9,034/10,411; FP 244; FN 1,621.
- India (n=3000): F0.5 **0.7418**, P 0.8566, R 0.6865, F1 0.7037; edges 8,633/10,506;
  FP 1,291 (5.3x US rate); FN 3,164 (30.1%).
- S2 vs S3: US 0.8924/0.8859; India 0.6873/0.7377. No S2/S3-specific pathology.
- Score separation US: true mean 0.965 (p10 0.932), false mean 0.014 (0.17% ≥0.9).
  India: true mean 0.869 (p10 **0.343**), false mean 0.035 (0.92% ≥0.9) — systematic
  underconfidence on India truths + 5.4x FP rate.
- Sample-local exclusivity changed NOTHING (ind == exc exactly, both countries): collisions
  are diffuse. Production test outputs show 117,791 multi-claimed targets (5.27% of
  5,566,403 edges, max multiplicity 6) — real but unmeasurable sample-locally.

## 4. Threshold curve (independent; exclusive identical on samples)
- US: 0.85→0.9119, 0.90→0.9138, **0.92→0.9149 (+0.0011)**, 0.94→0.9142, 0.95→0.9118.
  Flat 0.90–0.94. Production 0.9 is near-optimal on US.
- India: 0.90→0.7418 … 0.94→0.7482, 0.95→0.7487, **0.96→0.7506 (+0.0088)**, 0.97→0.7467.
  Calibration helps India marginally (+0.009) but cannot close a 0.17 gap.
- Full curves in `metrics_us.json` / `metrics_india.json` (`curve_ind`, `curve_exc`).

## 5. Score-margin analysis
- Mean best score 0.960 (US) / 0.957 (India); mean margin 0.077 / 0.086.
- S1s with best ≥0.9 AND margin ≥0.05/0.10/0.20: US 390/314/267 of 3000; India 571/465/327.
- Margins do not isolate a clean high-precision subset beyond the threshold itself
  (score separation already excellent on US; overlapping on India).

## 6. Generic-name analysis (`namefreq_*.json`, frequencies counted over all 10.3M train S2/S3)
- US: precision flat 0.962–1.000 across buckets 1 / 2–5 / 6–10 / 11–50 / 51–100 / >100.
  NO generic-name FP concentration. Recall 0.85–0.95 everywhere.
- India: precision dips 6–10 (0.915), 11–50 (0.896); recall collapses 11–50 (**0.44**),
  6–10 (0.66). Mid-frequency names hurt on India only — consistent with matcher
  country-degradation, not a blocking defect.

## 7. Singleton analysis (@0.9 exclusive)
- US: 150 true singletons → F0.5 0.8467 (127 correct, 23 FP... 27 FP edges); 65
  non-singleton S1s predicted empty (all FN).
- India: 153 true singletons → F0.5 **0.549** (84 correct, 100 FP edges).
- Non-singleton F0.5: US 0.9174, India 0.7522.
- No candidate-less S1s on US (min 9 cands); India has 3 zero-candidate S1s.

## 8. Conflict analysis (existing TEST outputs — labels unknown, structure only)
- 5,566,403 matched edges; multiplicity: ×1 5,273,242 / ×2 85,466 / ×3 17,635 / ×4 7,289 /
  ×5 4,238 / ×6 3,163. Distinct conflicted targets 117,791 (2.18%); edges on conflicted
  targets 293,161 (**5.27%**). Saved `conflicts.json`.
- Prediction-count distribution: 0→124,823 / 1→203,834 / 2→314,750 / 3→361,040 /
  4→313,724 / 5→215,530 / 6→118,492 / 7→52,080 / 8→18,702 / ≥9→~9k; max 25.
- Mean predicted edges/entity 3.21 overall, 3.47 non-empty vs GT 3.46 — aggregates calibrated.

## 9. S2/S3 breakdown
- Recall: US S2 0.9237 > S3 0.9104; India S2 0.9085 > S3 0.8994. Small consistent S2 edge.
- Matcher F0.5 gaps (US→India): S2 0.8924→0.6873 (−0.205), S3 0.8859→0.7377 (−0.148).

## 10. Country breakdown
- US-model validated: US 0.9138 vs India 0.7418 (**−0.172**). Model trained on 32,054 US
  entities only; India (46.7% of test S1s) is effectively out-of-distribution.
- Test-weighted estimate 0.38×0.914 + 0.47×0.742 + 0.15×France(≈0.7–0.8) ≈ 0.79–0.81 —
  reproduces LB ~0.80. Country degradation EXPLAINS the leaderboard.
- France: unlabeled EDA only (50k S1s/country). Normalizer lacks eurl/sasu/sci suffix
  stripping (top France tokens); city 0.841 ≈ US, house 0.996, postal 0.005 (India 0.003);
  non-ASCII names 15.7% (US/India 0%). No France action taken (no labels to validate with).

## 11. Top false negatives / false positives
- `fn_top100_us.csv`, `fp_top100_us.csv`, `misses_us.csv` (867), and India equivalents —
  all with S1/target names, addresses, scores. Representative US near-miss FNs score
  0.80–0.89 (threshold-adjacent); India FNs spread down to ~0.34 (systematic, not borderline).

## 12. Decisive follow-up experiments (all validated BEFORE any production change)
- E1 threshold 0.9→0.92 on US: +0.0011. NOTED ONLY, too small to act on.
- E2-pilot (India-only HistGB, 8k entities, 60 iters): India 0.7418→**0.8511 (+0.109)**,
  US 0.9138→0.8694 (−0.044). Lever PROVEN both directions.
- E3 joint model (701,382 examples: 351,557 US with pos count EXACTLY reproducing
  production's 101,557 + 349,825 India; HistGB 120 iters, same architecture; threshold
  0.88 selected on combined samples): US 0.9050 (−0.0088), India **0.8480 (+0.106)**,
  combined 0.8765 vs ~0.828 (**+0.048**). Saved `diagnostic_v1/matcher_joint.joblib`
  (sha256 c2e165d6…); baseline artifact sha unchanged (0d243c4a…).
- E4 (RUNNING at report time): full test re-inference, production-identical config
  (bucket 5000/stage1 2000/cap 50/fallback 10/partition/shard-6), only model swapped,
  into `output_joint/` with `--keep-work`. Baseline `output/` preserved.

## FINAL DECISION
B. "MATCHER FIRST"

Evidence: blocking recall 0.90–0.92 on both measured countries (ceiling loss only 8–10%,
stage-1/rank-depth dominated, zero keyless misses); threshold curves flat (±0.009);
generic names cleared on US; conflicts diffuse and unmeasurable sample-locally; singleton
handling degrades only as a symptom (India singleton F0.5 0.549 under the same model that
scores 0.847 on US). The single measured −0.17 cross-country gap, its +0.109 reversal with
India training data, and the joint model's +0.048 combined lift numerically dominate every
alternative. Retraining the EXISTING matcher (same features, same architecture, same
blocking) was the justified change; no architecture redesign was undertaken.

## Performance / provenance
- TOTAL RUNTIME ≈ 9 h wall (index builds US 1776 s + India 767 s + US-rebuild 958 s;
  queries ≈ 15 min per 3k-sample per country; s11a examples ≈ 70 min; s11b ≈ 63 min
  incl. 18 s fit; namefreq passes ≈ 4 min each over 10.3M rows).
- PEAK MEMORY ≈ 3.9 GB (example-building reservoirs + unbounded target cache; 7 GB box).
- FILES CREATED: all under `output/diagnostic_v1/` (scripts/, *.jsonl, *.json, *.csv,
  *.log, EXPERIMENTS.md, matcher_joint.joblib, this report). Zero production files
  modified (verified: baseline model sha256 unchanged).
- KEY METRICS: recall US 0.9167 / India 0.9039; joint-validated F0.5 US 0.9050 / India
  0.8480 / combined 0.8765 @0.88.
- PRIMARY BOTTLENECK: matcher country-degradation (US-only training data).
- NEXT EXPERIMENT TO RUN: none until E4 inference completes (~13 h); then validate
  `output_joint/` format + row counts + sanity stats, and decide submission (3 left —
  spend at most one on the joint pipeline).
