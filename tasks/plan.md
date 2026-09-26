# Implementation Plan: Business Entity Resolution

## Overview

Build a CPU-first, leakage-safe entity-resolution pipeline under `student_resource/code/business_entity_resolution/`. The pipeline will stream the large TSV files, normalize names and addresses with generic rules, generate a bounded union of retrieval candidates, score pairs with engineered similarity features and a histogram gradient-boosting model, tune against macro F0.5, apply target-side exclusivity, and emit strict submission TSVs.

## Architecture Decisions

- Keep the exact context module contract: `config.py`, `normalize.py`, `block.py`, `features.py`, `train.py`, `match.py`, and `submit.py`.
- Add small shared modules only where they remove duplicated data/evaluation logic; keep the public entry points under `src/`.
- Use `csv`/chunked readers rather than loading all 24M source rows or all candidate edges at once.
- Use scikit-learn's `HistGradientBoostingClassifier` as the available CPU GBM baseline; keep a deterministic heuristic fallback so smoke tests do not require a trained model.
- Retrieve same-country candidates first, with a bounded cross-country fallback, rather than using country as an absolute gate.
- Treat target-side exclusivity as capacitated b-matching: one target cannot be assigned to two S1 rows, while one S1 row may retain multiple valid targets.
- Evaluate every decision with entity-level validation, exact macro F0.5, blocking recall@K, singleton behavior, and a fixed synthetic edge-case suite.
- Never add external lookups, geocoding, or network-dependent model downloads.

## Ordered Build Paths

### Path 1: Foundation, I/O, and evaluation

- Add configuration, streaming TSV readers, ground-truth parsing, anomaly-safe ID validation, and exact macro F0.5 utilities.
- Add unit tests for empty/singleton/list scoring, malformed labels, deterministic splitting, and TSV output escaping.
- Verify the foundation on a small synthetic fixture and a streaming sample of the real data.

### Checkpoint 1

- All focused tests pass.
- No full dataset is loaded into memory.
- A small end-to-end output can be validated by the supplied validator.

### Path 2: Normalization

- Implement Unicode/case/punctuation cleanup, legal suffix extraction, token sorting, address abbreviations, postal/house/city extraction, digit features, and anomaly flags.
- Preserve raw fields alongside normalized fields.
- Add fixed tests for suffixes, `&`, word order, abbreviations, missing fields, transliteration-compatible Unicode cleanup, and France/open-country labels.

### Path 3: Candidate generation

- Build streaming normalized indexes and union exact/core-name, token, numeric/address, prefix, and bounded cross-country retrieval.
- Cap candidates per S1 and retain retrieval provenance/rank for feature engineering.
- Add tests for cap enforcement, deduplication, singleton rows, country fallback, and deterministic ordering.
- Measure recall@20/@50 on a held-out S1 split before proceeding.

### Checkpoint 2

- Blocking recall@K is measured on a real validation sample.
- Candidate memory stays bounded by chunk size and K.
- Candidate output contains every required S1 row.

### Path 4: Pair features and model

- Implement pairwise name/address/cross-field/missingness/ambiguity features over candidate shards.
- Expand ground-truth positives, retrieve hard negatives, split by S1, and train a deterministic CPU GBM.
- Tune the decision threshold directly on validation macro F0.5.
- Add unit tests for feature finiteness, label expansion, hard-negative selection, and threshold selection.

### Checkpoint 3

- The model trains and scores a small labeled fixture.
- Validation metrics and singleton rate are logged.
- No all-pairs feature matrix is created.

### Path 5: Global decision and assignment

- Add high-confidence edge selection, right-side exclusivity, unmatched handling, and output-safe deterministic ordering.
- Validate that assignment cannot delete a valid S1 row and cannot assign one target to multiple S1 rows.
- Compare independent thresholding versus assignment on validation.

### Checkpoint 6

- Assignment improves or preserves the validation score on the test fixture.
- No duplicate target assignments occur.
- All threshold and assignment choices are configurable.

### Path 6: Submission, audit, and packaging

- Add `submit.py` entry point, output generation, validator integration, anomaly log, experiment summary, synthetic suite runner, README, requirements, and filled methodology document.
- Run smoke tests, then run the full CPU pipeline on the available data with checkpoints.
- Generate `output/candidate_pairs.tsv` and `output/matching_results.tsv` and run `validate_submission.py` with ID checks when memory permits.

## Checkpoint: Complete

- Unit/integration/synthetic tests pass.
- Lint/type checks (or the closest available static checks) pass.
- Full outputs validate against the required schema and source IDs.
- Reproducible run instructions and methodology documentation are complete.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Full data exceeds available RAM | High | Stream TSVs, process candidate/feature shards, persist only bounded model artifacts |
| Blocking recall ceiling | High | Union multiple retrieval methods, measure recall@K, keep cross-country fallback |
| Validator memory on full candidates | Medium | Run in a clean process, use `--check-ids` only when memory allows, and validate matching separately |
| Country labels are noisy | High | Use country as a soft ranking feature and bounded fallback, never a final hard gate |
| One-to-one wording conflicts with multi-match labels | High | Enforce exclusivity on targets with multi-target capacity per S1 |
| No LightGBM/XGBoost installed | Medium | Use available `HistGradientBoostingClassifier`; isolate the model behind a small interface |
| Optional embedding dependency unavailable | Low | Keep semantic/embedding retrieval optional and disabled by default |

## Open Questions / Defaults

- The team name, member names, and final submission date are not present; methodology placeholders will remain explicit until supplied.
- No leaderboard portal or external lookup is used.
- The default candidate cap is 50 total candidates per S1 and can be raised only after recall measurement.
- The default validation split is deterministic and entity-level, stratified only where practical without loading all labels.
