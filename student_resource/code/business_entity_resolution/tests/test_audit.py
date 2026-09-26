import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audit import audit_file, measure_blocking_recall


class AuditTests(unittest.TestCase):
    def test_audit_file_counts_countries_and_missingness(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.tsv"
            path.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "S1-1\tAcme\t1 Road\tUS\n"
                "S1-2\tOther\t\tFrance\n",
                encoding="utf-8",
            )
            report = audit_file(path, source=1)
            self.assertEqual(report["row_count"], 2)
            self.assertEqual(report["country_counts"], {"france": 1, "us": 1})
            self.assertEqual(report["blank_address_count"], 1)

    def test_blocking_recall_uses_validation_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source1 = root / "source1.tsv"
            source2 = root / "source2.tsv"
            truth = root / "truth.tsv"
            source1.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "S1-1\tAcme\t1 Road\tUS\n"
                "S1-2\tOther\t2 Road\tUS\n",
                encoding="utf-8",
            )
            source2.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "S2-1\tAcme Limited\t1 Road\tUS\n"
                "S2-9\tOther Name\t9 Road\tUS\n",
                encoding="utf-8",
            )
            truth.write_text(
                "source1_entity_id\tmatched_entity_ids\n"
                "S1-1\tS2-1\n"
                "S1-2\t\n",
                encoding="utf-8",
            )
            report = measure_blocking_recall(
                source1_path=source1,
                target_paths=[source2],
                truth_path=truth,
                source1_limit=2,
                target_limit=10,
                validation_fraction=1.0,
            )
            self.assertEqual(report["recall_at_20"], 1.0)
            self.assertEqual(report["truth_link_count"], 1)


if __name__ == "__main__":
    unittest.main()
