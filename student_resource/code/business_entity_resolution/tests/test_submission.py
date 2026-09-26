import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.block import BlockingIndex
from src.io import Record, read_output_rows
from src.normalize import normalize_record
from src.submit import run_inference, run_inference_streaming


class SubmissionTests(unittest.TestCase):
    def test_end_to_end_outputs_preserve_every_source1_row(self):
        targets = [
            normalize_record(Record("S2-1", "Acme Private Limited", "12 Park Road", "US")),
            normalize_record(Record("S3-1", "Acme Private Limited", "12 Park Road", "US")),
        ]
        index = BlockingIndex.from_records(targets)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source1_path = root / "source1.tsv"
            source1_path.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "S1-1\tAcme Pvt Ltd\t12 Park Road\tUS\n"
                "S1-2\tUnlisted\t7777 Nowhere Lane\tUS\n",
                encoding="utf-8",
            )
            summary = run_inference(
                source1_path=source1_path,
                target_index=index,
                output_dir=root / "output",
                threshold=0.5,
            )
            self.assertEqual(summary.source1_count, 2)
            self.assertEqual(summary.candidate_edge_count, 2)
            self.assertEqual(
                read_output_rows(
                    root / "output" / "candidate_pairs.tsv",
                    ("source1_entity_id", "candidate_entity_ids"),
                )["S1-2"],
                (),
            )
            self.assertEqual(
                set(
                    read_output_rows(
                        root / "output" / "matching_results.tsv",
                        ("source1_entity_id", "matched_entity_ids"),
                    )
                ),
                {"S1-1", "S1-2"},
            )

    def test_streaming_mode_writes_same_contract(self):
        targets = [
            normalize_record(Record("S2-1", "Acme Private Limited", "12 Park Road", "US")),
            normalize_record(Record("S3-1", "Acme Private Limited", "12 Park Road", "US")),
        ]
        index = BlockingIndex.from_records(targets)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source1_path = root / "source1.tsv"
            source1_path.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "S1-1\tAcme Pvt Ltd\t12 Park Road\tUS\n"
                "S1-2\tUnlisted\t7777 Nowhere Lane\tUS\n",
                encoding="utf-8",
            )
            summary = run_inference_streaming(
                source1_path=source1_path,
                target_index=index,
                output_dir=root / "output",
                threshold=0.5,
            )
            self.assertEqual(summary.source1_count, 2)
            self.assertEqual(summary.candidate_edge_count, 2)
            self.assertEqual(
                set(
                    read_output_rows(
                        root / "output" / "matching_results.tsv",
                        ("source1_entity_id", "matched_entity_ids"),
                    )
                ),
                {"S1-1", "S1-2"},
            )

    def test_streaming_keeps_empty_rows_and_resolves_collisions(self):
        targets = [
            normalize_record(Record("S2-1", "Acme Private Limited", "12 Park Road", "US")),
        ]
        index = BlockingIndex.from_records(targets)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source1_path = root / "source1.tsv"
            source1_path.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "S1-1\tAcme Pvt Ltd\t12 Park Road\tUS\n"
                "S1-2\tAcme Private Limited\t12 Park Road\tUS\n"
                "S1-3\tUnlisted\t7777 Nowhere Lane\tUS\n",
                encoding="utf-8",
            )
            summary = run_inference_streaming(
                source1_path=source1_path,
                target_index=index,
                output_dir=root / "output",
                threshold=0.5,
            )
            rows = read_output_rows(
                root / "output" / "matching_results.tsv",
                ("source1_entity_id", "matched_entity_ids"),
            )
            self.assertEqual(set(rows), {"S1-1", "S1-2", "S1-3"})
            self.assertEqual(rows["S1-3"], ())
            self.assertEqual(sum(len(ids) for ids in rows.values()), 1)
            self.assertGreaterEqual(summary.collision_count, 1)

    def test_inference_limit_is_reported_as_incomplete(self):
        targets = [normalize_record(Record("S2-1", "Acme", "1 Road", "US"))]
        index = BlockingIndex.from_records(targets)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source1_path = root / "source1.tsv"
            source1_path.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "S1-1\tAcme\t1 Road\tUS\n"
                "S1-2\tOther\t2 Road\tUS\n",
                encoding="utf-8",
            )
            summary = run_inference_streaming(
                source1_path=source1_path,
                target_index=index,
                output_dir=root / "output",
                source1_limit=1,
                threshold=0.5,
            )
            self.assertFalse(summary.complete)


if __name__ == "__main__":
    unittest.main()
