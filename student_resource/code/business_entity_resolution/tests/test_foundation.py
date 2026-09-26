import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DEFAULT_SETTINGS
from src.evaluation import (
    blocking_recall_at_k,
    entity_level_split,
    macro_f_beta,
    score_entity,
    threshold_predictions,
)
from src.io import (
    DataFormatError,
    iter_ground_truth,
    iter_records,
    load_ground_truth,
    read_output_rows,
    write_candidate_pairs,
    write_matching_results,
)


class FoundationTests(unittest.TestCase):
    def test_config_points_to_student_resource(self):
        self.assertEqual(DEFAULT_SETTINGS.train_source1_path.name, "train_source1.tsv")
        self.assertEqual(DEFAULT_SETTINGS.test_source1_path.name, "test_source1.tsv")
        self.assertEqual(DEFAULT_SETTINGS.candidate_cap, 50)

    def test_records_are_tab_separated_and_prefix_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.tsv"
            path.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "S1-1\tA & B\t1 Road\tUS\n"
                "S2-2\tC\t\tIndia\n",
                encoding="utf-8",
            )
            records = list(iter_records(path))
            self.assertEqual(records[0].entity_id, "S1-1")
            self.assertEqual(records[0].business_address, "1 Road")
            self.assertEqual(records[1].source, 2)
            with self.assertRaises(DataFormatError):
                list(iter_records(path, source=1))

    def test_ground_truth_parses_singletons_and_deduplicates_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truth.tsv"
            path.write_text(
                "source1_entity_id\tmatched_entity_ids\n"
                "S1-1\tS2-2,S2-2,S3-3\n"
                "S1-2\t\n",
                encoding="utf-8",
            )
            rows = list(iter_ground_truth(path))
            self.assertEqual(rows[0].matched_entity_ids, ("S2-2", "S3-3"))
            self.assertEqual(rows[1].matched_entity_ids, ())
            self.assertEqual(load_ground_truth(path)["S1-1"], {"S2-2", "S3-3"})

    def test_output_writer_and_reader_preserve_tabs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matching = root / "matching.tsv"
            candidates = root / "candidates.tsv"
            write_matching_results(matching, [("S1-1", ["S2-2", "S3-3"]), ("S1-2", [])])
            write_candidate_pairs(candidates, [("S1-1", ["S2-2", "S3-3", "S3-3"])])
            self.assertEqual(
                read_output_rows(
                    matching,
                    ("source1_entity_id", "matched_entity_ids"),
                )["S1-1"],
                ("S2-2", "S3-3"),
            )
            self.assertEqual(
                read_output_rows(
                    candidates,
                    ("source1_entity_id", "candidate_entity_ids"),
                )["S1-1"],
                ("S2-2", "S3-3"),
            )

    def test_entity_fbeta_handles_singletons(self):
        self.assertEqual(score_entity(set(), set()).f_beta, 1.0)
        self.assertEqual(score_entity(set(), {"S2-1"}).f_beta, 0.0)
        self.assertEqual(score_entity({"S2-1"}, set()).f_beta, 0.0)
        self.assertAlmostEqual(score_entity({"S2-1", "S3-1"}, {"S2-1"}).f_beta, 5 / 9)

    def test_macro_fbeta_uses_all_truth_entities(self):
        summary = macro_f_beta(
            {"S1-1": {"S2-1"}, "S1-2": set()},
            {"S1-1": {"S2-1"}, "S1-2": set(), "S1-3": {"S3-3"}},
        )
        self.assertEqual(summary.entity_count, 3)
        self.assertEqual(summary.missing_prediction_rows, 1)
        self.assertAlmostEqual(summary.f_beta, 2 / 3)

    def test_blocking_recall_is_link_based(self):
        metrics = blocking_recall_at_k(
            {"S1-1": ["S2-1", "S2-9"]},
            {"S1-1": {"S2-1", "S3-3"}, "S1-2": {"S2-2"}},
            k=1,
        )
        self.assertEqual(metrics["recovered_link_count"], 1)
        self.assertEqual(metrics["truth_link_count"], 3)
        self.assertAlmostEqual(metrics["blocking_recall"], 1 / 3)

    def test_split_is_deterministic_and_entity_level(self):
        ids = [f"S1-{index}" for index in range(100)]
        first = entity_level_split(ids, 0.2, seed=7)
        second = entity_level_split(iter(ids), 0.2, seed=7)
        self.assertEqual(first, second)
        self.assertTrue(first[0].isdisjoint(first[1]))
        self.assertEqual(first[0] | first[1], set(ids))

    def test_threshold_predictions_are_inclusive(self):
        self.assertEqual(
            threshold_predictions({"S1-1": {"S2-1": 0.7, "S2-2": 0.71}}, 0.7),
            {"S1-1": {"S2-1", "S2-2"}},
        )


if __name__ == "__main__":
    unittest.main()
