import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import Record
from src.normalize import normalize_record
from src.block import BlockingIndex, SqliteBlockingIndex


class BlockingTests(unittest.TestCase):
    def setUp(self):
        self.targets = [
            normalize_record(Record("S2-1", "Acme Private Limited", "12 Park Rd", "US")),
            normalize_record(Record("S2-2", "Acme LLC", "99 Other Road", "US")),
            normalize_record(Record("S3-1", "Acme Private Limited", "12 Park Road", "US")),
            normalize_record(Record("S3-2", "Cafe Ecole", "10 Rue de Paris", "France")),
            normalize_record(Record("S3-3", "Cafe Ecole", "10 Rue de Paris", "India")),
        ]
        self.index = BlockingIndex.from_records(self.targets)

    def query(self, entity_id, name, address, country, cap=50):
        record = normalize_record(Record(entity_id, name, address, country))
        return self.index.query(record, cap=cap, country_fallback_limit=2)

    def test_union_methods_deduplicates_and_keeps_provenance(self):
        candidates = self.query("S1-1", "Acme Pvt Ltd", "12 Park Road", "US")
        ids = [candidate.candidate_entity_id for candidate in candidates]
        self.assertEqual(ids.count("S2-1"), 1)
        self.assertIn("S2-1", ids)
        self.assertIn("S3-1", ids)
        evidence = next(candidate for candidate in candidates if candidate.candidate_entity_id == "S2-1")
        self.assertGreaterEqual(len(evidence.methods), 2)
        self.assertGreaterEqual(evidence.score, evidence.best_rank_score)

    def test_singleton_has_no_candidates(self):
        candidates = self.query("S1-2", "Unlisted Business", "7777 Nowhere Lane", "US")
        self.assertEqual(candidates, ())

    def test_country_is_a_rank_signal_and_fallback_is_not_a_gate(self):
        same_country = self.query("S1-3", "Cafe Ecole", "10 Rue de Paris", "France")
        self.assertEqual(same_country[0].candidate_entity_id, "S3-2")
        self.assertIn("S3-3", [candidate.candidate_entity_id for candidate in same_country])
        cross_country = self.query("S1-4", "Unknown Name", "No Shared Address", "France", cap=2)
        self.assertNotIn("S3-3", [candidate.candidate_entity_id for candidate in cross_country])

    def test_cap_is_deterministic_and_sorted(self):
        records = [
            normalize_record(Record(f"S2-{index}", "Common Name", f"{index} Main Road", "US"))
            for index in range(10)
        ]
        index = BlockingIndex.from_records(records)
        record = normalize_record(Record("S1-1", "Common Name", "1 Main Road", "US"))
        first = index.query(record, cap=3)
        second = index.query(record, cap=3)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual([candidate.best_rank for candidate in first], [1, 2, 3])

    def test_target_source_prefix_is_preserved(self):
        self.assertEqual(self.index.targets["S3-2"].raw.source, 3)

    def test_sqlite_index_round_trips_without_memory_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "targets.sqlite"
            with SqliteBlockingIndex.build(self.targets, path) as index:
                self.assertTrue(index.is_complete())
                self.assertEqual(index.metadata()["row_count"], "5")
                record = normalize_record(
                    Record("S1-1", "Acme Pvt Ltd", "12 Park Road", "US")
                )
                candidates = index.query(record, cap=10, country_fallback_limit=2)
                self.assertIn("S2-1", [item.candidate_entity_id for item in candidates])
                batch = index.query_many(
                    [
                        record,
                        normalize_record(
                            Record("S1-2", "Unlisted", "7777 Nowhere Lane", "US")
                        ),
                    ],
                    cap=10,
                    country_fallback_limit=2,
                )
                self.assertIn("S2-1", [item.candidate_entity_id for item in batch[0]])
                self.assertEqual(batch[1], ())
                target = index.get("S2-1")
                self.assertIsNotNone(target)
                self.assertEqual(target.raw.business_name, "Acme Private Limited")


if __name__ == "__main__":
    unittest.main()
