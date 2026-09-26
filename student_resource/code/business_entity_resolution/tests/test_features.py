import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.block import CandidateEvidence
from src.features import extract_pair_features, feature_names
from src.io import Record
from src.normalize import normalize_record


class FeatureTests(unittest.TestCase):
    def pair(self, left, right, evidence=None, candidate_count=1):
        reference = normalize_record(Record("S1-1", left[0], left[1], left[2]))
        candidate = normalize_record(Record("S2-1", right[0], right[1], right[2]))
        return extract_pair_features(
            reference,
            candidate,
            evidence=evidence,
            candidate_count=candidate_count,
            runner_up_score=0.1,
            name_frequency={"acme": 1, "cafe": 1},
        )

    def test_exact_suffix_variant_has_strong_name_features(self):
        features = self.pair(
            ("Acme Pvt Ltd", "12 Park Road", "US"),
            ("Acme Private Limited", "12 Park Road", "US"),
        )
        self.assertEqual(features["name_core_exact"], 1.0)
        self.assertEqual(features["address_exact"], 1.0)
        self.assertGreater(features["name_token_jaccard"], 0.5)

    def test_word_order_is_similar_but_different_names_are_not(self):
        reordered = self.pair(
            ("Learning Center Moncada", "1 Main Road", "US"),
            ("Moncada Learning Center", "1 Main Road", "US"),
        )
        different = self.pair(
            ("Learning Center Moncada", "1 Main Road", "US"),
            ("Unrelated Bakery", "1 Main Road", "US"),
        )
        self.assertGreater(reordered["name_token_jaccard"], different["name_token_jaccard"])
        self.assertEqual(different["name_core_exact"], 0.0)

    def test_numeric_agreement_and_country_are_separate_signals(self):
        features = self.pair(
            ("Cafe", "12 Park Road", "US"),
            ("Cafe", "12 Other Road", "India"),
        )
        self.assertEqual(features["numeric_overlap"], 1.0)
        self.assertEqual(features["country_exact"], 0.0)
        self.assertEqual(features["address_house_exact"], 1.0)

    def test_missing_fields_are_finite_and_explicit(self):
        features = self.pair(("Cafe", "", "US"), ("Cafe", "", "US"))
        self.assertTrue(all(math.isfinite(value) for value in features.values()))
        self.assertEqual(features["missing_address"], 1.0)
        self.assertEqual(features["address_exact"], 0.0)

    def test_retrieval_provenance_is_included(self):
        evidence = CandidateEvidence("S2-1", 1.4, frozenset({"name_core_exact", "address_house"}), 1, 1.4)
        features = self.pair(
            ("Acme", "1 Main Road", "US"),
            ("Acme Pvt Ltd", "1 Main Road", "US"),
            evidence=evidence,
            candidate_count=7,
        )
        self.assertEqual(features["retrieval_method_count"], 2.0)
        self.assertEqual(features["retrieval_rank"], 1.0)
        self.assertEqual(features["candidate_count"], 7.0)
        self.assertIn("name_core_exact", feature_names())


if __name__ == "__main__":
    unittest.main()
