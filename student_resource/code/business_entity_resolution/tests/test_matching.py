import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.match import decide_matches, independent_threshold


class MatchTests(unittest.TestCase):
    def test_independent_threshold_keeps_all_edges_above_cutoff(self):
        scores = {
            "S1-1": {"S2-1": 0.9, "S2-2": 0.4},
            "S1-2": {},
        }
        self.assertEqual(
            independent_threshold(scores, 0.7),
            {"S1-1": {"S2-1"}, "S1-2": set()},
        )

    def test_target_collision_keeps_highest_confidence_owner(self):
        scores = {
            "S1-1": {"S2-1": 0.91, "S2-2": 0.80},
            "S1-2": {"S2-1": 0.95},
            "S1-3": {},
        }
        result = decide_matches(scores, threshold=0.7, enforce_exclusivity=True)
        self.assertEqual(result.predictions["S1-1"], {"S2-2"})
        self.assertEqual(result.predictions["S1-2"], {"S2-1"})
        self.assertEqual(result.predictions["S1-3"], set())
        self.assertEqual(result.collision_count, 1)

    def test_source1_can_retain_multiple_targets(self):
        scores = {"S1-1": {"S2-1": 0.9, "S3-1": 0.8}}
        result = decide_matches(scores, threshold=0.7)
        self.assertEqual(result.predictions["S1-1"], {"S2-1", "S3-1"})
        self.assertEqual(result.collision_count, 0)

    def test_candidate_filter_and_empty_rows_are_preserved(self):
        scores = {"S1-1": {"S2-1": 0.9, "S2-9": 0.99}, "S1-2": {}}
        candidates = {"S1-1": {"S2-1"}, "S1-2": set()}
        result = decide_matches(
            scores,
            candidates=candidates,
            threshold=0.7,
            enforce_exclusivity=True,
        )
        self.assertEqual(result.predictions, {"S1-1": {"S2-1"}, "S1-2": set()})

    def test_tie_breaking_is_deterministic(self):
        scores = {"S1-2": {"S2-1": 0.9}, "S1-1": {"S2-1": 0.9}}
        first = decide_matches(scores, threshold=0.7)
        second = decide_matches(scores, threshold=0.7)
        self.assertEqual(first.predictions, second.predictions)
        self.assertEqual(first.predictions["S1-1"], {"S2-1"})


if __name__ == "__main__":
    unittest.main()
