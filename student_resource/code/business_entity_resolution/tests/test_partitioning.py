import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.block import SqliteBlockingIndex
from src.io import Record, read_output_rows
from src.normalize import normalize_record
from src.submit import build_target_index, main


HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


def _write(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(HEADER)
        for row in rows:
            handle.write("\t".join(row) + "\n")


class PartitioningTests(unittest.TestCase):
    def test_build_writes_final_path_without_temporary_files(self):
        records = [
            normalize_record(Record("S2-1", "Acme Limited", "12 Park Road", "US")),
            normalize_record(Record("S3-1", "Acme Limited", "12 Park Road", "US")),
        ]
        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "index.sqlite"
            index = SqliteBlockingIndex.build(records, index_path, overwrite=True)
            metadata = index.metadata()
            complete = index.is_complete()
            index.close()
            self.assertTrue(index_path.exists())
            self.assertEqual(
                sorted(
                    path.name
                    for path in Path(directory).iterdir()
                    if path.name != "index.sqlite"
                ),
                [],
            )
            self.assertTrue(complete)
            self.assertEqual(metadata["row_count"], "2")

    def test_failed_build_removes_partial_index(self):
        def broken_records():
            yield normalize_record(Record("S2-1", "Acme Limited", "12 Park Road", "US"))
            raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "index.sqlite"
            with self.assertRaises(RuntimeError):
                SqliteBlockingIndex.build(broken_records(), index_path, overwrite=True)
            self.assertFalse(index_path.exists())

    def test_country_filtered_index_only_serves_that_country(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source2 = root / "source2.tsv"
            source3 = root / "source3.tsv"
            _write(source2, [("S2-1", "Acme Limited", "12 Park Road", "US")])
            _write(source3, [("S3-1", "Acme Limited", "12 Park Road", "FR")])
            index = build_target_index(
                [source2, source3],
                index_path=root / "index.sqlite",
                country="us",
            )
            try:
                metadata = index.metadata()
                self.assertEqual(metadata["country"], "us")
                self.assertEqual(metadata["row_count"], "1")
                evidence = index.query(
                    normalize_record(Record("S1-9", "Acme Limited", "12 Park Road", "US"))
                )
                self.assertEqual(
                    [item.candidate_entity_id for item in evidence], ["S2-1"]
                )
                self.assertNotIn("S3-1", {item.candidate_entity_id for item in evidence})
            finally:
                index.close()

    def test_country_provenance_triggers_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source2 = root / "source2.tsv"
            source3 = root / "source3.tsv"
            _write(source2, [("S2-1", "Acme Limited", "12 Park Road", "US")])
            _write(source3, [("S3-1", "Acme Limited", "12 Park Road", "FR")])
            index_path = root / "index.sqlite"
            first = build_target_index(
                [source2, source3], index_path=index_path, country="us"
            )
            first.close()
            rebuilt = build_target_index(
                [source2, source3], index_path=index_path, country="fr"
            )
            try:
                self.assertEqual(rebuilt.metadata()["country"], "fr")
                self.assertEqual(rebuilt.metadata()["row_count"], "1")
                connection = sqlite3.connect(index_path)
                try:
                    row_id = connection.execute(
                        "SELECT entity_id FROM targets"
                    ).fetchone()[0]
                finally:
                    connection.close()
                self.assertEqual(row_id, "S3-1")
            finally:
                rebuilt.close()

    def test_partition_by_country_covers_every_source1_row_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "dataset"
            output_dir = root / "output"
            _write(
                data_root / "test" / "test_source1.tsv",
                [
                    ("S1-1", "Acme Limited", "12 Park Road", "US"),
                    ("S1-2", "Bistro Chez", "5 Rue Victor Hugo", "FR"),
                    ("S1-3", "No Match Here", "9 Empty Street", "US"),
                ],
            )
            _write(
                data_root / "test" / "test_source2.tsv",
                [
                    ("S2-1", "Acme Limited", "12 Park Road", "US"),
                    ("S2-2", "Bistro Chez", "5 Rue Victor Hugo", "FR"),
                ],
            )
            _write(
                data_root / "test" / "test_source3.tsv",
                [("S3-1", "Acme Private Limited", "12 Park Road", "US")],
            )
            exit_code = main(
                [
                    "infer",
                    "--data-root",
                    str(data_root),
                    "--output-dir",
                    str(output_dir),
                    "--split",
                    "test",
                    "--heuristic",
                    "--threshold",
                    "0.5",
                    "--partition-by-country",
                ]
            )
            self.assertEqual(exit_code, 0)
            candidate_rows = read_output_rows(
                output_dir / "candidate_pairs.tsv",
                ("source1_entity_id", "candidate_entity_ids"),
            )
            matching_rows = read_output_rows(
                output_dir / "matching_results.tsv",
                ("source1_entity_id", "matched_entity_ids"),
            )
            self.assertEqual(set(candidate_rows), {"S1-1", "S1-2", "S1-3"})
            self.assertEqual(set(matching_rows), {"S1-1", "S1-2", "S1-3"})
            for entity_id, matched in matching_rows.items():
                self.assertTrue(
                    set(matched).issubset(candidate_rows[entity_id]),
                    f"matching rows must stay inside candidates for {entity_id}",
                )
            self.assertEqual(matching_rows["S1-3"], ())
            self.assertTrue((output_dir / "parts" / "fr").exists())
            self.assertTrue((output_dir / "parts" / "us").exists())
            summary = (output_dir / "inference_summary.json").read_text(
                encoding="utf-8"
            )
            self.assertIn('"partitioned_by_country": true', summary)

    def test_countries_filter_processes_only_requested_country(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "dataset"
            output_dir = root / "output"
            _write(
                data_root / "test" / "test_source1.tsv",
                [
                    ("S1-1", "Acme Limited", "12 Park Road", "US"),
                    ("S1-2", "Bistro Chez", "5 Rue Victor Hugo", "FR"),
                ],
            )
            _write(
                data_root / "test" / "test_source2.tsv",
                [
                    ("S2-1", "Acme Limited", "12 Park Road", "US"),
                    ("S2-2", "Bistro Chez", "5 Rue Victor Hugo", "FR"),
                ],
            )
            _write(
                data_root / "test" / "test_source3.tsv",
                [("S3-1", "Acme Private Limited", "12 Park Road", "US")],
            )
            exit_code = main(
                [
                    "infer",
                    "--data-root",
                    str(data_root),
                    "--output-dir",
                    str(output_dir),
                    "--split",
                    "test",
                    "--heuristic",
                    "--threshold",
                    "0.5",
                    "--partition-by-country",
                    "--countries",
                    "us",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertFalse((output_dir / "parts" / "fr").exists())
            self.assertTrue((output_dir / "parts" / "us" / "matching_results.tsv").exists())
            candidate_rows = read_output_rows(
                output_dir / "candidate_pairs.tsv",
                ("source1_entity_id", "candidate_entity_ids"),
            )
            self.assertEqual(set(candidate_rows), {"S1-1"})
            with self.assertRaises(ValueError):
                main(
                    [
                        "infer",
                        "--data-root",
                        str(data_root),
                        "--output-dir",
                        str(output_dir),
                        "--split",
                        "test",
                        "--heuristic",
                        "--threshold",
                        "0.5",
                        "--partition-by-country",
                        "--countries",
                        "zz",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
