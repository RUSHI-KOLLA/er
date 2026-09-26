import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.block import SqliteBlockingIndex, _folded_sets, _jaccard
from src.io import Record
from src.normalize import normalize_record


def _record(entity_id, name, address, country):
    return normalize_record(
        Record(
            entity_id=entity_id,
            business_name=name,
            business_address=address,
            country=country,
        )
    )


def _records():
    return (
        _record("S1-1", "Acme Limited", "12 Park Road, Springfield", "us"),
        _record("S1-2", "Bistro Chez", "5 Rue Victor Hugo", "fr"),
        _record("S1-3", "Acme Private Limited", "12 Park Road", "us"),
        _record("S1-4", "No Match Anyone", "9 Empty Street", "us"),
    )


def _targets():
    return (
        _record("S2-1", "Acme Limited", "12 Park Road, Springfield", "us"),
        _record("S2-2", "Acme Private Ltd", "12 Park Road", "us"),
        _record("S3-1", "Acme Limited", "12 Park Road", "in"),
        _record("S2-3", "Totally Different Traders", "88 Nowhere", "us"),
        _record("S2-4", "Bistro Chez", "5 Rue Victor Hugo", "fr"),
        _record("S2-5", "Acme", "1 Park Road", "us"),
    )


class PostingsParityTests(unittest.TestCase):
    def _build(self, directory: Path, bucket_limit: int = 5000):
        index_path = directory / "index.sqlite"
        built = SqliteBlockingIndex.build(_targets(), index_path, overwrite=True)
        built.close()
        with_postings = SqliteBlockingIndex(index_path, bucket_limit=bucket_limit)
        with_postings.ensure_postings()
        self.assertIsNotNone(with_postings._postings)
        sql_only = SqliteBlockingIndex(
            index_path, bucket_limit=bucket_limit, use_postings=False
        )
        self.assertIsNone(sql_only._postings)
        return with_postings, sql_only

    def test_postings_and_sql_return_identical_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            fast, sql = self._build(Path(directory))
            try:
                records = _records()
                for stage1, cap in ((None, 50), (5, 50), (4, 2), (3, 3)):
                    fast_rows = fast.query_many(
                        records, cap=cap, country_fallback_limit=2, stage1=stage1
                    )
                    sql_rows = sql.query_many(
                        records, cap=cap, country_fallback_limit=2, stage1=stage1
                    )
                    for fast_evidence, sql_evidence in zip(fast_rows, sql_rows):
                        self.assertEqual(len(fast_evidence), len(sql_evidence))
                        for fast_item, sql_item in zip(fast_evidence, sql_evidence):
                            self.assertEqual(
                                fast_item.candidate_entity_id,
                                sql_item.candidate_entity_id,
                            )
                            self.assertAlmostEqual(
                                fast_item.score, sql_item.score, places=12
                            )
                            self.assertEqual(fast_item.methods, sql_item.methods)
                            self.assertEqual(fast_item.best_rank, sql_item.best_rank)
            finally:
                fast.close()
                sql.close()

    def test_postings_respect_bucket_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            fast, sql = self._build(Path(directory), bucket_limit=1)
            try:
                records = _records()
                fast_rows = fast.query_many(records, cap=50, stage1=20)
                sql_rows = sql.query_many(records, cap=50, stage1=20)
                self.assertEqual(
                    [
                        tuple(item.candidate_entity_id for item in row)
                        for row in fast_rows
                    ],
                    [
                        tuple(item.candidate_entity_id for item in row)
                        for row in sql_rows
                    ],
                )
                self.assertTrue(any(row for row in sql_rows))
            finally:
                fast.close()
                sql.close()

    def test_int_gram_jaccard_matches_string_reference(self):
        import re as _re

        samples = [
            "Acme Limited",
            "12 Park Road, Springfield 57100",
            "bistro chez",
            "",
            "a",
            "AAA  ___  999",
            "Rue Victor Hugo",
            "alpha bravo charlie delta",
        ]

        def original_sets(text: str):
            folded = _re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
            padded = folded.replace(" ", "_")
            span = len(padded) - 2
            grams = (
                {padded[index : index + 3] for index in range(span)}
                if span > 0
                else set()
            )
            return grams, set(folded.split())

        for left in samples:
            for right in samples:
                old_left, old_right = original_sets(left), original_sets(right)
                new_left, new_right = _folded_sets(left), _folded_sets(right)
                self.assertEqual(
                    _jaccard(old_left[0], old_right[0]),
                    _jaccard(new_left[0], new_right[0]),
                )
                self.assertEqual(
                    _jaccard(old_left[1], old_right[1]),
                    _jaccard(new_left[1], new_right[1]),
                )


if __name__ == "__main__":
    unittest.main()
