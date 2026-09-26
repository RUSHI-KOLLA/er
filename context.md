# Business Entity Resolution — Project Context
**Amazon ML Challenge 2026** | Repo: `RUSHI-KOLLA/er` (student_resource skeleton)

This file is the single source of truth for building this project. Read it before writing any code.
Goal: win this — every section below exists because it's a place teams typically lose points.

---

## 1. Problem Summary

Given business records from 3 independent sources (no shared IDs), find every Source 2 / Source 3
record that refers to the same real-world business as each Source 1 (reference) entity. A Source 1
entity may match zero, one, or many records. This is Clean-Clean Entity Resolution.

---

## 2. Data

### 2.1 Files & schema
All files are `.tsv` (tab-separated — commas appear inside addresses and ID lists, so `sep="\t"` is mandatory).

| File | Columns |
|---|---|
| `train/test _source{1,2,3}.tsv` | `entity_id` (prefix `S1-`/`S2-`/`S3-`), `business_name`, `business_address`, `country` |
| `train_ground_truth.tsv` | `source1_entity_id`, `matched_entity_ids` (comma-sep, empty = singleton) |

No separate "source" column — the ID prefix + file it's in tells you the source.
`country`: train = `{US, India}`; **test adds `France`, unseen in training.** Treat as an open string set — never hardcode or one-hot to just US/India.

### 2.2 Scale (confirmed from the repo's Git LFS pointer files — not yet downloaded/inspected as raw rows)

| File | Size | Rough row estimate |
|---|---|---|
| train_source1.tsv | ~210 MB | ~1.1–1.3M |
| train_source2.tsv | ~489 MB | ~2.5–2.8M |
| train_source3.tsv | ~504 MB | ~2.6–2.9M |
| train_ground_truth.tsv | ~127 MB | roughly = source1 row count |
| test_source1.tsv | ~175 MB | ~0.9–1.0M |
| test_source2.tsv | ~509 MB | ~2.7–2.9M |
| test_source3.tsv | ~506 MB | ~2.7–2.9M |

**Total ≈ 2.5 GB.** Millions of records per source → brute-force S1×S2×S3 comparison is computationally
impossible. Blocking is not optional polish, it's the thing that makes the problem tractable at all.
Row estimates are approximate — confirm exact counts with `wc -l` once you `git lfs pull` for real.

### 2.3 Compute — CPU is sufficient for the entire pipeline

No stage of this plan requires gradient-based training or fine-tuning:
- GBM (XGBoost/LightGBM/CatBoost) trains via tree-splitting, not backprop — CPU handles millions of
  rows and high boosting-round counts fine.
- The optional embedding pass for blocking is **inference-only**, no training — a small model
  (e.g. MiniLM family) does on the order of 10k+ sentences/sec on CPU, so even a few million records
  is minutes, not hours.
- GPU only becomes relevant if you reach the optional Stage 4 LLM reranker, and even then it only
  scores a small fraction of borderline pairs. Don't provision for GPU as a baseline assumption.

---

## 3. Constraints (hard rules — violating any of these = disqualification or rejection)

- **No external data/lookups.** No entity-resolution APIs, no geocoding APIs, no government business
  registries, no internet augmentation of any kind. Build your own regex/dictionary address & name
  normalizers instead of libraries like libpostal — safer given the explicit "no geocoding APIs" rule.
- **Final model: MIT or Apache-2.0 licensed, ≤8B parameters.**
  Qualifies: Mistral-7B (Apache 2.0), Qwen2.5-7B (Apache 2.0).
  **Does NOT qualify: Llama (custom Meta license), Gemma (custom license).**
- Output format is strict (see §5) — a submission that fails validation isn't scored at all.
- `matched_entity_ids` / `candidate_entity_ids`: S2-/S3- IDs only, no self-matches to S1, no IDs
  outside the current split, no duplicates within a list, no duplicate `source1_entity_id` rows.
- Every S1 test entity needs exactly one output row (empty match list is valid for singletons).

---

## 4. Evaluation Metric

**F0.5, macro-averaged per Source 1 entity** (singletons included in the average):

```
F0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

Precision weighted 2× over recall. A correctly-predicted singleton (empty list) scores 1.0; a false
merge on a true singleton scores 0.0. **When uncertain, prefer no match over a wrong match.**

---

## 5. Output Format Contract

`output/matching_results.tsv` (scored) and `output/candidate_pairs.tsv` (not scored, used to audit
blocking quality — recall ceiling & reduction ratio). Same schema for both:

```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

`matching_results` must be a **subset** of `candidate_pairs` for every S1 entity. Run
`utils/validate_submission.py --matching ... --candidate ... --test-dir ...` before every leaderboard
upload — it catches every format rule above locally, for free, before burning a submission.

---

## 6. Recommended Architecture

**Pipeline:** normalize → block (union of methods) → feature-engineer pairs → GBM classifier →
threshold on F0.5 → global one-to-one assignment → singleton-aware final decision.

### 6.1 Normalization (rule-based, no external libraries)
- Name: Unicode-normalize, lowercase, strip punctuation, `&`→`and`, strip legal suffixes into a
  separate field (Pvt/Private, Ltd/Limited, Corp/Corporation, LLP, Inc, LLC, Co...), token-sorted
  variant for word-order invariance.
- Address: same base cleanup + abbreviation dictionary (Rd/Road, St/Street, Ave/Avenue, Nr/Near...),
  regex-extract postal code / house number / city where detectable. Keep original fields alongside
  normalized ones — don't overwrite.
- Build country-specific abbreviation *additions*, not country-specific branches — the pipeline must
  degrade gracefully on France, which has none of these rules written for it.

### 6.2 Blocking / candidate generation — union of methods, capped per S1 (~50–100 candidates)
- **Country as a strong filter, not an absolute gate.** (See §8 — this is a deliberate deviation from
  "never compare across countries": measure blocking recall with and without a hard country filter
  before committing, since a mislabeled/missing country field would otherwise silently kill a true
  match at the blocking stage — unrecoverable downstream.)
- Exact/prefix blocks: `name_clean + postal_code`, `name_clean + city`, first-N-char prefixes.
- Character n-gram TF-IDF cosine similarity (name, address, and concatenated) — top-K per source.
- Token-level TF-IDF/BM25 for word-order changes.
- Optional: small multilingual sentence-embedding model (MIT/Apache licensed, e.g. MiniLM family) +
  ANN search as a semantic-variant safety net.
- **Measure blocking recall on a held-out validation split before building the matcher.** Target
  ≥99% recall at K≤50 — anything missed here can never be recovered by the classifier.

### 6.3 Pairwise features (per S1–candidate pair)
- Name: exact match (full / suffix-stripped), Jaccard on tokens, normalized Levenshtein, token-sort
  ratio, char n-gram TF-IDF cosine, shared rare-token count.
- Address: exact match, postal/house-number exact match, Jaccard, Levenshtein, TF-IDF cosine.
- Cross-field: name-similarity-given-same-postal, count of exactly-agreeing fields, number of
  retrieval methods that surfaced this candidate, TF-IDF/embedding rank & score.
- Ambiguity/missingness: candidate count for this S1, score gap to runner-up, fraction of missing
  fields, name frequency in-country (common vs. rare).

### 6.4 Matching model
GBM (LightGBM / XGBoost / CatBoost) over the feature vector — not a fine-tuned transformer as the
primary matcher. Reasoning: trains in minutes on CPU, scores millions of pairs cheaply, gives a
calibrated probability you can threshold precisely against F0.5, and is far easier to document/debug
for the methodology write-up than a fine-tuned model, all under real time pressure.

**Training labels:**
- Positives: expand `train_ground_truth.tsv`.
- Hard negatives (critical for precision): same-name/different-address, same-address/different-name,
  top retrieval candidates that aren't true matches, "runner-up" candidates that lost to a different
  S1's true match.

### 6.5 Threshold selection
Tune directly against **macro F0.5 on an entity-level validation split** (no leakage) — never against
AUC, accuracy, or F1. Expect the optimal threshold to sit high given the 2× precision weighting.

### 6.6 Global one-to-one consistency (do not skip)
A given S2/S3 record realistically belongs to only one real business. Build a bipartite graph
(S1 entities ↔ S2/S3 entities, edges = high-confidence candidate pairs weighted by calibrated
probability), decompose into connected components, and solve max-weight bipartite matching per
component (with an unmatched option) instead of independently thresholding every S1 row in
isolation. Prevents precision-killing collisions where two similar S1 entities both claim one record —
a direct precision cost under F0.5 with no other safety net catching it.

### 6.7 France / OOD robustness
- Country used as signal, never baked into language-specific rules.
- Prefer language-agnostic features: char n-gram similarity, digit agreement, token overlap.
- Validate cross-country before test day: train on US, validate on India (and vice versa) — check
  whether thresholds and feature importances hold up. Large drops mean simplify toward generic
  similarity features rather than country-specific heuristics.
- Generic address-convention knowledge (e.g., French postal codes are 5 digits, "Rue"/"Avenue" street
  prefixes) is general pattern knowledge, not external lookup — fine to use as a soft signal, with the
  same caution as country blocking above: don't let one rule silently gate out a true match.

---

## 7. Hidden Data Quirks — Discovery & Reaction Protocol

Known noise patterns (abbreviations, legal suffixes, transliteration, landmark addresses) are already
handled by normalization + blocking. This section is for **quirks nobody anticipated** — a deliberate
mechanism to surface these, not just react once they cause visible score damage.

- **Run full EDA on the test set too, not just train.** France only exists in test — country-string
  variants ("France"/"FR"/"french republic"), postal formats, encoding issues specific to it will
  never show up in train-only EDA.
- **Country-label sanity check.** Don't trust `country` blindly — check whether it plausibly matches
  the address content (postal code shape, script). Log mismatches; a wrong label upstream silently
  breaks any country-aware blocking or feature.
- **Keep a running anomaly log**, separate from `experiments.csv` — any unexpected format, encoding
  artifact, off-pattern record, or inconsistent label, logged the moment someone finds it, not
  reconstructed from memory at the end.
- **Normalization/blocking should log what it can't confidently parse**, not silently pass it through
  or drop it. Unparsed residue is usually exactly where quirks hide.
- **Treat error analysis as continuous, not a single scheduled stage.** Feed discoveries back into
  normalization/blocking as soon as found, rather than batching fixes for later.

---

## 8. Key Design Decisions & Rationale (for the methodology doc)

| Decision | Why | Rejected alternative & why |
|---|---|---|
| GBM over engineered features as primary matcher | Fast, calibratable, interpretable, no GPU needed, scores millions of pairs cheaply | Fine-tuned transformer (Ditto-style): better semantic matching in theory, but slower to build/calibrate under a 72h clock and harder to document |
| Union of token + embedding blocking | Blocking recall is a hard ceiling; the two methods fail on disjoint cases | Either alone: cheaper, but caps max achievable score lower |
| Country as strong feature, not hard gate | A mislabeled/missing country field is exactly the kind of noise the brief warns about; a hard gate would silently drop that match at blocking, unrecoverably | Hard per-country blocking gate: cuts compute more, but riskier — validate before adopting |
| Custom rule-based address/name normalizer | Keeps you clearly clear of the "no geocoding APIs" rule; also lighter/faster | libpostal or similar: technically offline, but close enough to the banned category to not be worth the risk |
| Threshold tuned on F0.5 directly | The actual scored metric, precision-weighted 2× | Default 0.5 cutoff / tuning on F1 or accuracy: consistently the top reason teams underperform in these competitions |
| Bipartite one-to-one assignment as a final pass | Real businesses are one entity; prevents double-claiming | Independent per-S1 thresholding only: simpler, but allows collisions that cost precision |

---

## 9. Development Workflow

### 9.1 Compute split
- **CLI agent (Antigravity/OpenCode/Claude Code) in VS Code** → writes and iterates on the reusable
  `src/` pipeline code. This is CPU-only work (see §2.3) — no GPU dependency for the core pipeline.
- **Kaggle notebook directly** → only needed if/when the optional embedding or reranker stage is
  built, since that's the one place GPU and interactive debugging (checking tensor shapes, CUDA
  errors) genuinely help. Don't default to notebook-first for the GBM-driven core.
- To move CLI-written code onto Kaggle: push `src/` as a private Kaggle Dataset
  (`kaggle datasets version -p <src-folder> -m "update"`), attach it in the notebook, `sys.path.append`
  and import — keeps internet off (a clean guardrail against the "no external lookup" rule) and avoids
  re-cloning on every change.

### 9.2 Keeping `src/` from turning into a mess
- **Fix the module layout before writing any code** (see §11 for the exact structure) and give it to
  every CLI session as a fixed contract — don't let each person's agent invent its own boundaries.
- **One module per CLI request.** Ask for `normalize.py`, review it, then `block.py`, review it — not
  "build the pipeline" in one shot. Apply your existing request-review habit per-file.
- **Give the agent the target function signature, not just the goal** — e.g. specify
  `normalize_name(s: str) -> str` up front, don't just say "clean the data."
- Agree on the module contract as a team once, keep it in this file, so different members' CLI
  sessions build compatible code instead of three incompatible structures.

### 9.3 Why understanding the code matters more than usual here
Top teams' submissions get their code reviewed, and finalists present live to Amazon Scientists. If
nobody on the team can explain *why* the blocking or threshold logic works when asked, that's a real
risk, not just a style preference. Keep request-review discipline strict specifically on blocking and
match-decision logic (the parts you'll have to defend) — be more relaxed about it for boilerplate
(TSV I/O, the validator wrapper, `requirements.txt`).

---

## 10. Testing & Validation Protocol

Two tracks. Track A tells you the aggregate score. Track B tells you *why* — and catches regressions
Track A's single number can hide.

### 10.1 Track A — Statistical validation (primary signal)
- **Entity-level split** (by Source 1 entity, not random pair-splitting) — no leakage between train
  and validation.
- **Exact local F0.5 evaluator** matching the competition's per-entity macro formula precisely —
  implement this before touching a real model, and treat it as ground truth for every decision.
- **Blocking recall@K measured separately from final F0.5** — a true match lost at blocking can never
  be recovered downstream, so this number must be checked on its own, not inferred from the final score.
- **Error analysis broken down by category**: false positives, false negatives, singleton errors,
  blocking misses, country/noise-specific failures — not just one aggregate error rate.

### 10.2 Track B — Synthetic edge-case suite (catches what aggregate metrics hide)
Build a small, hand-labeled, fixed set of ~20–50 record pairs, each covering exactly one noise pattern
from the brief:
- Legal suffix swap (Pvt Ltd ↔ Private Limited)
- DBA / trade name variant
- Word-order transposition
- Abbreviation (Rd ↔ Road, St ↔ Street)
- Missing PIN / postal code
- Landmark-based address ("Near SBI ATM")
- Transliteration variant
- **Deliberate near-duplicate DIFFERENT businesses that must NOT match** (this is the precision
  stress-test — equally important as the positive cases)

Run this fixed suite through the full pipeline after every meaningful change, like a unit test suite.
A big dataset's macro F0.5 can stay flat even while one entire noise category silently breaks — this
is the check that catches that.

### 10.3 Run summary — "is this good or not," answered in one glance
After every run, log a compact summary (feeds directly into `experiments.csv`):

| Field | What it tells you |
|---|---|
| Blocking recall@K | Is the ceiling still intact? |
| Precision / Recall / macro F0.5 (validation) | The actual scored signal |
| Synthetic-suite pass rate | Did a specific noise category regress? |
| Predicted vs. actual singleton rate | Is the model over- or under-matching overall? |
| Worst 5–10 false positives (eyeballed) | What kind of wrong merge is happening? |
| Worst 5–10 false negatives (eyeballed) | What kind of real match is being missed? |

If any run doesn't produce this full summary, don't trust a single F0.5 number from it as "better" —
re-run with the full checklist before deciding to keep a change.

---

## 11. Repo / File Structure (final submission zip)

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/business_entity_resolution/
│   ├── src/
│   │   ├── normalize.py     # name/address cleaning
│   │   ├── block.py          # candidate generation
│   │   ├── features.py       # pairwise similarity features
│   │   ├── train.py           # GBM training
│   │   ├── match.py          # threshold + bipartite assignment
│   │   ├── submit.py         # generates + validates output TSVs
│   │   └── config.py         # thresholds, paths, seeds — no magic numbers scattered in code
│   ├── README.md         # exact run instructions, data → blocking → matching → output
│   └── requirements.txt  # pinned versions
└── Documentation_template.md   # filled in, see §13
```

---

## 12. 72-Hour Execution Plan

| Hours | Focus |
|---|---|
| 0–6 | Data audit on **both train and test** (null rates, country distribution incl. France, singleton ratio, country-label sanity check); start the anomaly log; rule-based exact-match baseline; build the exact local F0.5 evaluator; validate submission format end-to-end early |
| 6–18 | Blocking engine: normalization, exact/prefix/TF-IDF blocks; measure recall@20/@50 on validation split; tune until ≥99% recall at K≤50; build the synthetic edge-case suite in parallel |
| 18–36 | Feature computation; train GBM with hard negatives; evaluate macro F0.5 on validation; tune threshold; run synthetic suite after each change |
| 36–48 | Bipartite one-to-one assignment; singleton behavior analysis; re-evaluate macro F0.5 with full assignment logic |
| 48–60 | Cross-country validation (US↔India as France proxy); remove any hard-coded country logic; optional second model/blend if time allows |
| 60–72 | Full pipeline on test set; generate both output files; validate with `validate_submission.py`; write methodology doc; package zip |

---

## 13. Methodology Doc Mapping (`Documentation_template.md`)

1. Executive Summary — 2–3 sentences on approach + core innovation
2. Methodology — EDA insights (noise patterns actually observed) + solution strategy (blocking + classifier hybrid)
3. Candidate Generation — blocking keys used, candidate count, recall-protection evidence (recall@K numbers)
4. Matching Model — features list, model type, threshold selection method (F0.5 optimization on validation)
5. Results & Error Analysis — validation macro F0.5, synthetic-suite pass rate, common false-positive/false-negative patterns
6. Conclusion — summary + lessons learned
7. Appendix — code structure, entry points to reproduce both output files

---

## 14. Open Items / Logistics

- AWS Builder Center registration is separate onboarding (not a rules requirement) — gives AWS Free
  Tier/compute credits if you want cloud compute; not mandatory, you can build entirely locally/Colab/Kaggle.
- Exact leaderboard daily submission cap: **not stated in the problem spec — check the Unstop portal
  directly.**
- Code + methodology doc zip is due the **same Day-3 deadline** as the leaderboard, not later.
- Don't spend every submission chasing the public leaderboard — final ranking is private-leaderboard
  only; trust your own entity-level validation split over repeated public-LB pokes.

---

## 15. Win Checklist

Before considering the pipeline done, every one of these should be true:

- [ ] Blocking recall@K measured and ≥99% at K≤50 on validation
- [ ] Local F0.5 evaluator matches the competition's exact per-entity macro formula
- [ ] Threshold tuned against F0.5 specifically, not F1/accuracy/AUC
- [ ] Hard negatives included in GBM training, not just random negatives
- [ ] Bipartite one-to-one assignment applied, not independent per-row thresholding
- [ ] Country used as a feature, not a hard blocking gate (or the tradeoff was explicitly measured)
- [ ] Synthetic edge-case suite passes, including the deliberate non-match cases
- [ ] EDA run on test set too, not just train — France-specific quirks checked
- [ ] `experiments.csv` up to date with every meaningful run's full summary (§10.3)
- [ ] `validate_submission.py` passes before every leaderboard upload
- [ ] Every team member can explain the blocking and match-decision logic unprompted
- [ ] Methodology doc filled section-by-section as work happens, not reconstructed at the end
