# Build Checklist

## Path 1: Foundation, I/O, and evaluation

- [x] Add configuration and deterministic path/seed settings.
- [x] Add streaming TSV and ground-truth readers.
- [x] Add exact macro F0.5, precision, recall, and singleton metrics.
- [x] Add entity-level split and output serialization helpers.
- [x] Add focused foundation tests and run them.

## Path 2: Normalization

- [x] Normalize names while preserving raw values.
- [x] Normalize addresses and extract numeric/location signals.
- [x] Add legal suffix, abbreviation, Unicode, and missing-field tests.
- [x] Run normalization on a real sample and inspect anomaly counters.

## Path 3: Blocking

- [x] Build normalized indexes and union retrieval methods.
- [x] Enforce bounded candidates and deterministic ordering.
- [x] Add cross-country fallback without a hard country gate.
- [x] Measure validation recall@20 and recall@50 on linked targets.
- [x] Run blocking on a real sample and inspect candidates.

## Path 4: Features and model

- [x] Add bounded pairwise feature computation.
- [x] Expand labels and select hard negatives.
- [x] Train the available CPU GBM.
- [x] Tune threshold directly on macro F0.5.
- [x] Run feature/model tests and validation.

## Path 5: Decision and assignment

- [x] Add confidence filtering and target-side exclusivity.
- [x] Add unmatched behavior and deterministic assignment.
- [x] Compare independent and assigned decisions.
- [x] Run assignment tests.

## Path 6: Submission and packaging

- [x] Generate both required TSV outputs on a smoke fixture.
- [x] Add anomaly and experiment logging.
- [x] Add synthetic edge-case suite.
- [x] Add README, requirements, and methodology content.
- [x] Run smoke, static, and sample validation checks.

## Checkpoints

- [x] Foundation checkpoint passes.
- [x] Blocking recall checkpoint passes on linked targets.
- [x] Model/assignment checkpoint passes.
- [x] Submission validator passes on a bounded real-data slice.
- [x] Full test/lint/typecheck suite passes (58 tests, flake8 clean).

## Phase 7: Full-scale run

- [x] Two-stage rerank in `block.py` (`--stage1`), recall 0.851 -> 0.928 @50 on the US validation split.
- [x] Rerank/config sweep recorded in `experiments.csv` (b5000/s2000 chosen).
- [x] `query_batch_size=64` to bound SQLite temp/RAM (40k-record single query_many caused a 6GB temp file; killed and batched).
- [x] France test-country index prebuilt at `output/parts/france/blocking_index.sqlite` (1,434,993 rows).
- [x] Sharded partitioned inference: `--shard-count` (tests/test_sharding.py, 58 tests green).
- [x] README + Documentation_template.md updated with stage1/shard design and exclusivity caveat.
- [x] Train model: 40,000 US entities at stage1 2000 -> `artifacts/matcher.joblib` (threshold 0.9, macro F0.5 0.9196 on validation).
- [x] Smoke real-model sharded inference on a small test slice (RSS/temp/disk check).
- [x] Numpy posting-list retrieval path in `block.py` (sidecar `<index>.postings`, exact SQL parity: 0/400 mismatches on France, blocking recall identical to 10 decimals: 0.9080962800875274 / 0.9277899343544858).
- [x] Exact-value speedups: Levenshtein rewrite (40,008-case parity), method-mask aggregation, row_id IN detail fetch, sequence-ratio shortcut, cache halving; pipeline 118 -> 67.6 ms/entity with identical outputs.
- [x] Disk survival plan: per-country index+postings cleanup, streaming shard concat, deleted 5.2GB US train index after final audit (11.5GB free at launch).
- [ ] Full test inference: `infer --partition-by-country --shard-count 6 --stage1 2000 --bucket-limit 5000` (running since 03:24, PID 135356).
- [ ] Validate with `utils/validate_submission.py --check-ids`.
- [ ] Record final metrics in `Documentation_template.md`, copy artifacts.
