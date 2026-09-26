# ML Challenge 2026: Business Entity Resolution Methodology

**Team Name:** ER Baseline Team (replace before submission)  
**Team Members:** [Add team members]  
**Submission Date:** [Add submission date]

---

## 1. Executive Summary

The solution is a CPU-first hybrid of rule-based normalization, bounded multi-method blocking, engineered pairwise similarity features, and a scikit-learn histogram gradient-boosting classifier. Country is used as a ranking signal and bounded fallback rather than a hard gate, and a target-side exclusivity pass prevents two S1 rows from claiming the same S2/S3 record. The implementation uses SQLite work databases and streamed TSV output so the pipeline does not materialize the full 24M-row source corpus or an all-pairs comparison matrix in memory.

## 2. Methodology

### 2.1 Problem Analysis

The supplied data contains 2,206,821 train S1 rows, 5,034,616 train S2 rows, 5,285,603 train S3 rows, 1,732,544 test S1 rows, 4,887,273 test S2 rows, and 5,082,316 test S3 rows. Training country labels are US and India; test also contains France. Country is therefore handled as an open normalized string rather than a two-class categorical assumption.

The training ground truth has 123,247 singleton S1 entities and 7,638,365 positive links overall. Names contain legal-suffix combinations, punctuation differences, DBA/trade-name variants, word-order changes, typos, domain-like substitutions, and substantial non-Latin script variation. Target addresses are blank for roughly 3.3% of train S2/S3 rows, and postal codes are not consistently present. Numeric agreement is useful but cannot be a hard key.

The implementation logs normalization anomalies, including blank addresses, addresses without numeric components, and repeated legal suffixes. A fixed 20-case synthetic suite covers suffix swaps, DBA variants, word-order changes, abbreviations, missing postal codes, landmark addresses, Unicode accent changes, component reordering, and deliberate near-duplicate non-matches.

### 2.2 Solution Strategy

The stages are:

1. Stream and validate TSV input.
2. Normalize names and addresses without overwriting raw fields.
3. Generate a bounded union of candidates using exact/core-name, token, prefix, postal, house-number, numeric, city, and address-token blocks, then re-rank the top stage-1 pool with character n-gram and token overlap before capping at 50.
4. Compute name, address, cross-field, missingness, ambiguity, and retrieval-provenance features for each pair.
5. Train a histogram gradient-boosting classifier on positive labels and retrieval hard negatives.
6. Tune the threshold directly on entity-level validation macro F0.5 using the same target-exclusive decision rule used at inference.
7. Resolve target collisions deterministically and write the final model-input candidate set and final matches.

**Approach Type:** Two-Stage Blocking + Classifier + Target-Side Graph Assignment  
**Core Innovation:** A memory-bounded, provenance-aware blocking index that unions generic name/address signals, keeps country as a soft ranking signal, re-ranks each stage-1 pool with cheap character 3-gram/token similarity before the candidate cap, and carries retrieval evidence into a precision-oriented pairwise model.

## 3. Candidate Generation (Blocking)

**Blocking keys used:**

- Legal-suffix-stripped normalized name
- Full normalized name and token-sorted name
- Name prefixes and individual meaningful name tokens
- Postal code, house number, numeric address tokens, and inferred city
- Meaningful address tokens

Retrieval is two-stage. The SQL query scores and aggregates block keys without normalization and returns the top `stage1` candidates (default 2,000, query-time configurable). Each survivor is then re-ranked with weighted character 3-gram and token Jaccard similarities on light-normalized names and addresses, combined with the uncapped block score, and only the top 50 are emitted. Re-ranking is what closes the gap between block-key agreement and textual similarity: without it, true links found only through weak keys such as address tokens were pushed below the cap.

The default cap is 50 candidates per S1 entity. Same-country candidates receive a ranking boost, while a bounded cross-country fallback remains available. The SQLite index stores source paths, target limits, bucket settings, normalization version, row count, and a completion marker. It is built atomically and rebuilt if metadata is missing or inconsistent.

**Candidate pairs generated:** The final count is recorded in `output/inference_summary.json`; the streaming implementation writes candidates before model inference and caps the edge count at the configured candidate limit per S1.

**How true matches were protected:**

- Multiple block families are unioned rather than selecting one fragile key.
- Country is not a hard gate.
- Legal suffixes, address abbreviations, token order, Unicode accents, and missing components are normalized before retrieval.
- Recall@20 and recall@50 are measured separately on an entity-level validation split, and the stage-1 pool size is tuned on that split (see 5.1).
- A bounded sample run with a deliberately limited target slice is a pipeline sanity check, not evidence of final recall. The full target index and held-out recall report must be generated before submission.

## 4. Matching Model

**Features used:**

- Name: exact, suffix-stripped exact, token-sort exact, token Jaccard, normalized Levenshtein similarity, sequence ratio, character n-gram cosine, shared rare-token count, and rare-token score.
- Address: exact, postal, house-number, city, token Jaccard, normalized Levenshtein similarity, sequence ratio, character n-gram cosine, and numeric overlap.
- Cross-field: country agreement/conflict, name/address disagreement, address conflict, city conflict, name/address name conflict, agreeing-field count, and postal-conditioned name similarity.
- Ambiguity/missingness: candidate count, runner-up score gap, retrieval rank, retrieval score, retrieval-method count, and missing name/address/country indicators.

**Model type:** `sklearn.ensemble.HistGradientBoostingClassifier`, trained on CPU. A deterministic heuristic scorer is available only for explicit smoke/inference testing. No external model weights or network downloads are used.

**Threshold selection method:** A bounded grid of candidate thresholds is evaluated against exact per-entity macro F0.5, including singleton rows, with target-side exclusivity enabled. This avoids tuning on AUC, accuracy, or F1 and avoids quadratic rescans when validation contains many unique scores.

## 5. Results & Error Analysis

### 5.1 Completed Checks

- The fixed synthetic suite passes all 20 cases at threshold 0.72, including deliberate precision stress cases.
- The foundation, normalization, blocking, feature, model, assignment, audit, and submission tests pass.
- A real 2,000-row S1 smoke run with a 10,000-row-per-source target index produced 2,000 candidate rows and passed the supplied validator with `--check-ids` against the corresponding temporary test slice.
- On a linked-target diagnostic containing the 1,778 ground-truth targets for 500 S1 entities, recall@20 was 0.9966 and recall@50 was 0.9994. This validates retrieval logic on known links but is not a substitute for a full-pool recall measurement because the target fixture intentionally omitted unrelated distractors.
- The real audit sample confirmed France in test, blank target addresses, and repeated legal-suffix patterns.
- On 2,000 US validation S1 entities against the full 6.19M-row target pool (6,898 ground-truth links), block scores alone gave recall@50 0.851 at 70 ms/entity; two-stage re-ranking at `stage1 2000` raised recall@20 to 0.908 and recall@50 to 0.928 at 157 ms/entity, measured on a 404-entity subset of that split.
- A sweep over `bucket_limit` and `stage1` on the same 404 entities showed recall@50 rising from 0.928 (`bucket 5000`, `stage1 2000`) to 0.950 (`bucket 20000`, `stage1 5000`), with query cost growing from 157 to about 600 ms/entity. The shipped configuration is `bucket 5000` / `stage1 2000` because the larger settings roughly quadruple runtime for about two recall points.
- Shipped training at `stage1 2000` on 40,000 sampled S1 entities against the full US target index; a 307-entity smoke run at the same settings already reached validation precision 0.951, recall 0.809, macro F0.5 0.887.

### 5.2 Metrics Pending Full Run

- **Best full held-out macro F0.5:** [Insert after full target-index validation]
- **Blocking recall@20 / recall@50:** [Insert after full target-index validation]
- **Full synthetic pass rate:** 20/20 in the current CPU heuristic check; rerun after the final model artifact.
- **Singleton behavior:** [Insert predicted versus actual singleton rates from the full validation run]

The small target-limited training diagnostic is not a leaderboard estimate: it used only 10 positive training examples and a deliberately incomplete target slice, so its low recall is expected.

**Common false positives to inspect:** same-name/different-address businesses, shared landmark/address text with different names, and cross-country collisions. Explicit conflict features and target-side exclusivity are intended to reduce these errors.

**Common false negatives to inspect:** transliterated or non-Latin names, missing target addresses, domain-like name substitutions, reordered address components, and true links outside the initial candidate cap. These categories should be recorded in the anomaly/error log during the full run.

## 6. Conclusion

The implementation establishes a reproducible baseline that follows the precision-heavy F0.5 objective and the scale constraints in the project context. The main remaining work is data-dependent validation on the complete target pool, followed by threshold/model tuning and final packaging; no external lookup is required at any stage.

## Appendix

### A. Code Artefacts

The runnable package is under `code/business_entity_resolution/`:

- `src/normalize.py`: generic name/address normalization and anomaly flags
- `src/block.py`: in-memory and SQLite blocking indexes
- `src/features.py`: bounded pairwise feature extraction
- `src/train.py`: hard-negative sampling, CPU GBM training, and threshold selection
- `src/match.py`: thresholding and target-side exclusivity
- `src/submit.py`: streaming inference and TSV output
- `src/audit.py`: train/test EDA and recall reports
- `src/synthetic.py`: fixed edge-case suite
- `tests/`: unit and integration tests

The primary entry points are:

```bash
python3 -m src.submit train --country us --output-dir ../../output --model ../../artifacts/matcher.joblib --index-db ../../artifacts/us_train_index.sqlite --training-entity-limit 40000 --bucket-limit 5000 --stage1 2000
python3 -m src.submit infer --data-root ../../dataset --split test --output-dir ../../output --model ../../artifacts/matcher.joblib --bucket-limit 5000 --stage1 2000 --partition-by-country --shard-count 6
```

Validate outputs with `student_resource/utils/validate_submission.py` before packaging.

`--partition-by-country` builds, uses, and destroys one target index per country so peak disk holds a single country. `--shard-count 6` runs six hash-partitioned workers per country; the trade-off is that target exclusivity then holds only within each worker, so two S1 rows in different shards can claim the same target. The supplied validator does not test cross-row exclusivity, and the candidate and matching files are sharded consistently, so outputs remain internally valid; this trade-off is documented rather than hidden.

### B. Additional Results

The audit JSON, blocking recall JSON, inference summary, synthetic results, and experiment CSV should be copied into the final package after the full run. The current repository contains the implementation and reproducible commands; sample artifacts are not final leaderboard evidence.
