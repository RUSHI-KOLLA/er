import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic import SYNTHETIC_CASES, run_synthetic_suite


class SyntheticTests(unittest.TestCase):
    def test_suite_covers_positive_and_precision_stress_cases(self):
        self.assertGreaterEqual(len(SYNTHETIC_CASES), 20)
        self.assertTrue(any(case.should_match for case in SYNTHETIC_CASES))
        self.assertTrue(any(not case.should_match for case in SYNTHETIC_CASES))
        self.assertTrue(any("near_duplicate" in case.name for case in SYNTHETIC_CASES))

    def test_suite_passes_at_conservative_threshold(self):
        results = run_synthetic_suite()
        failures = [result for result in results if not result.passed]
        self.assertEqual(failures, [], "\n".join(f"{item.name}: {item.detail}" for item in failures))


if __name__ == "__main__":
    unittest.main()
