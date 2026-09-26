import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.block import SqliteBlockingIndex
from src.io import Record
from src.normalize import normalize_record


class RerankTests(unittest.TestCase):
    def _build_index(self, records):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "index.sqlite"
        index = SqliteBlockingIndex.build(
            [normalize_record(record) for record in records],
            path,
            overwrite=True,
        )
        self.addCleanup(index.close)
        return index

    def _flip_index(self):
        return self._build_index(
            [
                Record("S2-1", "Totally Different Traders", "12 Oak Avenue, Shelbyville 99999", "US"),
                Record("S2-2", "Alpha Bravu Charliu", "888 Nowhere Boulevard", "US"),
            ]
        )

    def test_stage1_rerank_reorders_by_text_similarity(self):
        index = self._flip_index()
        record = normalize_record(
            Record("S1-1", "Alpha Bravo Charlie", "12 Park Road, Springfield 57100", "US")
        )
        plain = index.query(record, cap=10, country_fallback_limit=2)
        reranked = index.query(record, cap=10, country_fallback_limit=2, stage1=100)
        self.assertEqual(plain[0].candidate_entity_id, "S2-1")
        self.assertEqual(reranked[0].candidate_entity_id, "S2-2")
        self.assertEqual(
            {item.candidate_entity_id for item in plain},
            {item.candidate_entity_id for item in reranked},
        )

    def test_stage1_truncates_the_pool_before_the_cap(self):
        records = [
            Record(f"S2-{index}", "Shared Name", f"{index} Park Road", "US")
            for index in range(12)
        ]
        index = self._build_index(records)
        record = normalize_record(Record("S1-1", "Shared Name", "1 Park Road", "US"))
        evidence = index.query(record, cap=4, country_fallback_limit=0, stage1=6)
        self.assertEqual(len(evidence), 4)
        self.assertEqual(
            [item.best_rank for item in evidence], [1, 2, 3, 4]
        )
        self.assertEqual(
            len({item.candidate_entity_id for item in evidence}), 4
        )

    def test_stage1_must_be_positive(self):
        index = self._flip_index()
        record = normalize_record(Record("S1-1", "Alpha Bravo Charlie", "1 Road", "US"))
        with self.assertRaises(ValueError):
            index.query(record, cap=5, stage1=0)

    def test_stage1_disabled_matches_default_behaviour(self):
        index = self._flip_index()
        record = normalize_record(
            Record("S1-1", "Alpha Bravo Charlie", "12 Park Road, Springfield 57100", "US")
        )
        default_rows = index.query(record, cap=5, stage1=None)
        plain_rows = index.query(record, cap=5)
        self.assertEqual(default_rows, plain_rows)

    def test_stage1_returns_normalized_targets_for_candidates(self):
        index = self._flip_index()
        record = normalize_record(
            Record("S1-1", "Alpha Bravo Charlie", "12 Park Road, Springfield 57100", "US")
        )
        evidence, targets = index.query_many_with_targets(
            (record,), cap=5, stage1=100
        )
        self.assertEqual(len(evidence), 1)
        for item in evidence[0]:
            self.assertIn(item.candidate_entity_id, targets)
            self.assertEqual(
                targets[item.candidate_entity_id].raw.entity_id,
                item.candidate_entity_id,
            )


if __name__ == "__main__":
    unittest.main()
