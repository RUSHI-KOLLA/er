# Business Entity Resolution Pipeline

This directory contains the reproducible CPU pipeline for the ML Challenge 2026 business entity resolution task. It uses only the provided TSV files: no network access, external business lookup, geocoding service, or registry is used.

## Quick Start

Run commands from this directory (`student_resource/code/business_entity_resolution`). The dataset is expected at `../../dataset` relative to this directory.

```bash
python3 -m unittest discover -s tests -v
flake8 src tests
python3 -m src.audit dataset --data-root ../../dataset --output artifacts/eda.json
python3 -m src.audit blocking --data-root ../../dataset --source1-limit 20000 --output artifacts/blocking_recall.json
```

Train a model with a bounded entity sample and a persistent target index:

```bash
python3 -m src.submit train \
  --data-root ../../dataset \
  --output-dir ../../output \
  --model ../../artifacts/matcher.joblib \
  --index-db ../../artifacts/train_blocking_index.sqlite \
  --training-entity-limit 40000 \
  --candidate-cap 50 \
  --bucket-limit 5000 \
  --stage1 2000
```

Run test inference and write both required files:

```bash
python3 -m src.submit infer \
  --data-root ../../dataset \
  --split test \
  --output-dir ../../output \
  --model ../../artifacts/matcher.joblib \
  --candidate-cap 50 \
  --bucket-limit 5000 \
  --stage1 2000 \
  --partition-by-country \
  --shard-count 6
```

`infer` requires an explicit trained `--model` or `--heuristic` flag. `--partition-by-country` builds one target index per country, destroys it before moving on, and concatenates the parts; `--shard-count N` then runs N independent workers per country, each emitting every `N`-th S1 entity by hash, and merges their parts. `--shard-count` requires `--partition-by-country`. The default non-inference index is SQLite; `--in-memory` is intended only for small smoke fixtures and is guarded by a target-count limit.

Validate a completed submission from `student_resource/`:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test \
  --check-ids
```

## Pipeline

1. `config.py` centralizes paths, seeds, candidate limits, batch sizes, and safety limits.
2. `io.py` streams tab-separated source and ground-truth files without loading the full dataset.
3. `normalize.py` applies Unicode cleanup, legal-suffix extraction, token sorting, address abbreviations, numeric/location signals, and anomaly flags while retaining raw fields.
4. `block.py` builds a union of exact, suffix-stripped, token, prefix, numeric, postal, city, and address blocks. `SqliteBlockingIndex` is the full-data path; it stores completion metadata and uses set-based batch retrieval. Retrieval is two-stage: the SQL query returns the top `--stage1` candidates by block score, then each candidate is re-ranked with character 3-gram and token Jaccard similarities on light-normalized names and addresses, and only the top `--candidate-cap` survivors are emitted.
5. `features.py` computes name, address, cross-field, missingness, retrieval, and ambiguity features.
6. `train.py` expands labels, samples hard negatives, trains scikit-learn `HistGradientBoostingClassifier`, and selects the threshold against the final target-exclusive macro F0.5 decision.
7. `match.py` applies the threshold and target-side exclusivity while allowing one S1 entity to retain multiple targets.
8. `submit.py` streams candidates and scores through a disk-backed work database, resolves collisions, and atomically writes the final TSVs.
9. `audit.py`, `synthetic.py`, and `evaluation.py` provide data audits, recall measurement, exact metrics, and the fixed edge-case suite.

## Output Contract

`candidate_pairs.tsv` contains the final model-input candidate set for every S1 row. `matching_results.tsv` contains only a subset of those candidates after thresholding and target-side assignment. Empty lists are valid and required for singletons. Both files are UTF-8 TSVs with one row per S1 entity and no duplicate IDs.

## Validation and Scale Notes

- The candidate cap defaults to 50 and the country fallback is bounded, but country is a ranking signal rather than a hard retrieval gate.
- Target indexes are built atomically and record source paths, limits, bucket settings, normalization version, row count, and completion state. An incomplete or mismatched index is rebuilt.
- The full target pool is approximately 10M records. The default index and inference paths avoid loading all normalized targets or all candidate edges into Python memory.
- Validation must be entity-level. The audit command measures recall@20 and recall@50 independently from final F0.5.
- Two-stage re-ranking raised held-out blocking recall@50 on a US validation split from 0.851 (block scores alone) to 0.928 at `--stage1 2000` and 0.950 at `--stage1 5000`; `--stage1` is a query-time knob, so index provenance does not record it. Stage-1 scores are deliberately uncapped: capping destroyed their ordering.
- Exclusivity of targets holds within each inference run, but `--shard-count` relaxes it across shards: two shards may claim the same target because they resolve assignments independently. The validator does not check cross-row exclusivity, so this is documented rather than hidden.
- The bundled sample run used only a small target slice and is not evidence of the final leaderboard score. Replace its sample metrics with a full held-out run before packaging.

## Dependencies

Install the pinned runtime and development dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

The pipeline uses only the Python standard library plus NumPy, SciPy, and scikit-learn. The optional embedding ideas in the project context are intentionally not enabled by default because they would require additional model assets; the two-stage character 3-gram re-ranker is implemented in plain Python and is enabled by default.
