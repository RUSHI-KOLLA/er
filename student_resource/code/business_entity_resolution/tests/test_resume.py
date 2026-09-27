import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.submit import _open_or_build_index, _run_shard_worker, merge_shard_parts


HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


def _write(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(HEADER)
        for row in rows:
            handle.write("\t".join(row) + "\n")


class ResumeMergeTests(unittest.TestCase):
    def test_split_resume_merge_matches_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "dataset" / "test"
            source1_rows = []
            source2_rows = []
            for index in range(30):
                name = "Acme Limited" if index % 2 == 0 else "Bistro Chez"
                address = f"{index} Park Road" if index % 3 else "12 Park Road"
                source1_rows.append((f"S1-{index}", name, address, "US"))
                source2_rows.append((f"S2-{index}", name, address, "US"))
            _write(data / "test_source1.tsv", source1_rows)
            _write(data / "test_source2.tsv", source2_rows)
            _write(
                data / "test_source3.tsv",
                [(f"S3-{index}", name, address, "US") for index, (_, name, address, _) in enumerate(source1_rows)],
            )
            index_path = root / "blocking_index.sqlite"
            built = _open_or_build_index(
                [data / "test_source2.tsv", data / "test_source3.tsv"],
                index_path,
                None,
                5000,
                "us",
            )
            built.close()
            worker_kwargs = {
                "source1_path": data / "test_source1.tsv",
                "index_path": index_path,
                "country": "us",
                "shard": (0, 2),
                "model": None,
                "threshold": 0.5,
                "candidate_cap": 50,
                "country_fallback_limit": 10,
                "stage1": 2000,
                "source1_limit": None,
                "enforce_exclusivity": True,
                "bucket_limit": 5000,
            }
            baseline_dir = root / "baseline"
            _run_shard_worker(output_dir=baseline_dir, **worker_kwargs)
            baseline_summary = json.loads(
                (baseline_dir / "shard_summary.json").read_text(encoding="utf-8")
            )
            group_size = baseline_summary["source1_count"]
            self.assertGreaterEqual(group_size, 6)
            self.assertGreater(baseline_summary["candidate_edge_count"], 0)
            self.assertGreater(baseline_summary["matched_edge_count"], 0)
            keep_limit = group_size // 3
            stop_limit = keep_limit + 2
            middle = keep_limit + max(1, (group_size - keep_limit) // 2)
            old_dir = root / "old"
            part_a_dir = root / "part_a"
            part_b_dir = root / "part_b"
            _run_shard_worker(
                output_dir=old_dir,
                start_ordinal=0,
                stop_ordinal=stop_limit,
                keep_work=True,
                **worker_kwargs,
            )
            _run_shard_worker(
                output_dir=part_a_dir,
                start_ordinal=keep_limit,
                stop_ordinal=middle,
                keep_work=True,
                **worker_kwargs,
            )
            _run_shard_worker(
                output_dir=part_b_dir,
                start_ordinal=middle,
                keep_work=True,
                **worker_kwargs,
            )
            summary = merge_shard_parts(
                old_dir,
                [part_a_dir, part_b_dir],
                keep_ordinal_limit=keep_limit,
                expected_source1_count=group_size,
            )
            for file_name in ("candidate_pairs.tsv", "matching_results.tsv"):
                self.assertEqual(
                    (baseline_dir / file_name).read_bytes(),
                    (old_dir / file_name).read_bytes(),
                    file_name,
                )
            for key in (
                "source1_count",
                "candidate_edge_count",
                "matched_edge_count",
                "collision_count",
                "complete",
            ):
                self.assertEqual(baseline_summary[key], summary[key], key)
            self.assertFalse((old_dir / ".incomplete").exists())
            self.assertFalse((old_dir / "scoring.sqlite").exists())


if __name__ == "__main__":
    unittest.main()
