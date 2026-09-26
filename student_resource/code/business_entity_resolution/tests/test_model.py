import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.block import BlockingIndex
from src.io import Record
from src.normalize import normalize_record
from src.train import (
    build_training_examples,
    fit_model,
    load_model,
    save_model,
    select_threshold,
    train_from_index,
)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.reference = [
            normalize_record(Record("S1-1", "Acme Pvt Ltd", "12 Park Rd", "US")),
            normalize_record(Record("S1-2", "Cafe", "", "France")),
        ]
        self.targets = [
            normalize_record(Record("S2-1", "Acme Private Limited", "12 Park Road", "US")),
            normalize_record(Record("S2-2", "Acme LLC", "99 Other Road", "US")),
            normalize_record(Record("S3-1", "Cafe", "", "France")),
            normalize_record(Record("S3-2", "Cafe", "10 Rue", "India")),
        ]
        self.index = BlockingIndex.from_records(self.targets)
        self.candidates = self.index.candidate_map(self.reference, cap=10, country_fallback_limit=2)
        self.truth = {
            "S1-1": {"S2-1"},
            "S1-2": {"S3-1"},
        }

    def test_training_examples_label_positive_and_hard_negative(self):
        examples = build_training_examples(
            self.reference,
            self.candidates,
            {record.raw.entity_id: record for record in self.targets},
            self.truth,
            max_pairs=100,
            seed=4,
        )
        labels = {(row.source1_entity_id, row.candidate_entity_id): row.label for row in examples}
        self.assertEqual(labels[("S1-1", "S2-1")], 1)
        self.assertEqual(labels[("S1-1", "S2-2")], 0)
        self.assertTrue(all(row.features for row in examples))

    def test_training_pair_cap_is_respected(self):
        examples = build_training_examples(
            self.reference,
            self.candidates,
            {record.raw.entity_id: record for record in self.targets},
            self.truth,
            max_pairs=2,
            seed=4,
        )
        self.assertLessEqual(len(examples), 2)

    def test_model_scores_clear_positive_above_negative(self):
        examples = build_training_examples(
            self.reference,
            self.candidates,
            {record.raw.entity_id: record for record in self.targets},
            self.truth,
            max_pairs=100,
            seed=4,
        )
        model = fit_model(examples, seed=4)
        positive = next(row for row in examples if row.label == 1)
        negative = next(row for row in examples if row.label == 0)
        positive_score = model.predict_proba([positive.features])[0]
        negative_score = model.predict_proba([negative.features])[0]
        self.assertGreaterEqual(positive_score, negative_score)
        self.assertGreaterEqual(positive_score, 0.0)
        self.assertLessEqual(positive_score, 1.0)

    def test_threshold_selection_optimizes_macro_fbeta(self):
        scores = {
            "S1-1": {"S2-1": 0.91, "S2-2": 0.20},
            "S1-2": {"S3-1": 0.88, "S3-2": 0.10},
        }
        truth = {"S1-1": {"S2-1"}, "S1-2": set()}
        threshold, summary = select_threshold(scores, truth)
        self.assertGreater(threshold, 0.2)
        self.assertEqual(summary.entity_count, 2)
        self.assertAlmostEqual(summary.f_beta, 1.0)

    def test_train_from_index_writes_model_and_validation_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source1_path = root / "source1.tsv"
            truth_path = root / "truth.tsv"
            source1_path.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "S1-1\tAcme Pvt Ltd\t12 Park Rd\tUS\n"
                "S1-2\tCafe\t\tFrance\n",
                encoding="utf-8",
            )
            truth_path.write_text(
                "source1_entity_id\tmatched_entity_ids\n"
                "S1-1\tS2-1\n"
                "S1-2\tS3-1\n",
                encoding="utf-8",
            )
            model_path = root / "model.joblib"
            summary = train_from_index(
                reference_path=source1_path,
                truth_path=truth_path,
                target_index=self.index,
                model_path=model_path,
                training_ids={"S1-1"},
                validation_ids={"S1-2"},
                max_pairs=100,
                candidate_cap=10,
            )
            self.assertTrue(model_path.exists())
            self.assertEqual(summary.validation_entity_count, 1)
            self.assertGreaterEqual(summary.threshold, 0.0)
            self.assertLessEqual(summary.threshold, 1.0)
            self.assertGreaterEqual(summary.positive_training_examples, 1)
            self.assertGreaterEqual(summary.negative_training_examples, 1)

    def test_threshold_selection_accounts_for_target_collisions(self):
        scores = {
            "S1-1": {"S2-1": 0.60},
            "S1-2": {"S2-1": 0.95, "S2-2": 0.90},
        }
        candidates = {
            "S1-1": {"S2-1"},
            "S1-2": {"S2-1", "S2-2"},
        }
        truth = {"S1-1": {"S2-1"}, "S1-2": {"S2-2"}}
        threshold, summary = select_threshold(
            scores,
            truth,
            candidates=candidates,
            enforce_exclusivity=True,
        )
        self.assertGreater(threshold, 0.8)
        self.assertGreater(summary.f_beta, 0.0)

    def test_model_round_trip(self):
        examples = build_training_examples(
            self.reference,
            self.candidates,
            {record.raw.entity_id: record for record in self.targets},
            self.truth,
            max_pairs=100,
            seed=4,
        )
        model = fit_model(examples, seed=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl"
            save_model(model, path)
            loaded = load_model(path)
            self.assertEqual(loaded.feature_names, model.feature_names)
            self.assertEqual(
                loaded.predict_proba([examples[0].features]),
                model.predict_proba([examples[0].features]),
            )


if __name__ == "__main__":
    unittest.main()
