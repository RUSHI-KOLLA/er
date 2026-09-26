import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import read_output_rows
from src.submit import _iter_normalized, main


HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


def _write(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(HEADER)
        for row in rows:
            handle.write("\t".join(row) + "\n")


class ShardingTests(unittest.TestCase):
    def test_shards_partition_source1_exactly_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test_source1.tsv"
            rows = [
                (f"S1-{index}", "Acme Limited", f"{index} Park Road", "US")
                for index in range(40)
            ]
            _write(path, rows)
            shard_ids = []
            for shard_index in range(4):
                shard_ids.append(
                    {
                        record.raw.entity_id
                        for record in _iter_normalized(
                            path, source=1, shard=(shard_index, 4)
                        )
                    }
                )
            union = set().union(*shard_ids)
            self.assertEqual(union, {row[0] for row in rows})
            for index in range(4):
                for other in range(index + 1, 4):
                    self.assertFalse(shard_ids[index] & shard_ids[other])

    def test_partition_by_country_with_shards_covers_every_row_once(self):
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
                    ("S1-4", "Acme Private Limited", "12 Park Road", "US"),
                    ("S1-5", "Corner Cafe", "77 Main Street", "FR"),
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
                    "--shard-count",
                    "2",
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
            expected = {"S1-1", "S1-2", "S1-3", "S1-4", "S1-5"}
            self.assertEqual(set(candidate_rows), expected)
            self.assertEqual(set(matching_rows), expected)
            for entity_id, matched in matching_rows.items():
                self.assertTrue(
                    set(matched).issubset(candidate_rows[entity_id]),
                    f"matching rows must stay inside candidates for {entity_id}",
                )
            summary = (output_dir / "inference_summary.json").read_text(
                encoding="utf-8"
            )
            self.assertIn('"shards": 2', summary)
            parts_root = output_dir / "parts"
            shard_dirs = [
                path for path in parts_root.iterdir() if path.name.endswith("shard0")
                or path.name.endswith("shard1")
            ]
            self.assertEqual(shard_dirs, [])
            self.assertTrue((parts_root / "fr" / "matching_results.tsv").exists())
            self.assertTrue((parts_root / "us" / "matching_results.tsv").exists())


if __name__ == "__main__":
    unittest.main()
