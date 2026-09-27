import argparse
import json
import multiprocessing
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.block import SqliteBlockingIndex
from src.config import DEFAULT_SETTINGS
from src.submit import _open_or_build_index, _run_shard_worker, set_similarity_cache_limit
from src.train import load_model

set_similarity_cache_limit(6000)


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def target_paths(data_root: str, split: str) -> list[Path]:
    test_dir = Path(data_root) / split
    return [test_dir / "test_source2.tsv", test_dir / "test_source3.tsv"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranges", default=None, help='JSON file: [{"shard":0,"start":N,"stop":M}]')
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--country", required=True)
    parser.add_argument("--data-root", default="../../dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--source1", default=None)
    parser.add_argument("--index", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bucket-limit", type=int, default=DEFAULT_SETTINGS.bucket_limit)
    parser.add_argument("--similarity-cache", type=int, default=6000)
    parser.add_argument("--stage1", type=int, default=DEFAULT_SETTINGS.blocking_stage1)
    args = parser.parse_args()

    if args.build_only:
        built = _open_or_build_index(
            target_paths(args.data_root, args.split),
            Path(args.index),
            None,
            args.bucket_limit,
            args.country,
        )
        if isinstance(built, SqliteBlockingIndex):
            built.close()
        log(f"index ready at {args.index}")
        return 0

    if not args.ranges:
        parser.error("--ranges is required unless --build-only is given")
    ranges = json.loads(Path(args.ranges).read_text(encoding="utf-8"))
    source1 = Path(args.source1) if args.source1 else Path(args.data_root) / args.split / "test_source1.tsv"
    model = load_model(Path(args.model))
    set_similarity_cache_limit(args.similarity_cache)
    log(f"model threshold={model.threshold}; ranges={len(ranges)}")

    context = multiprocessing.get_context()
    processes = []
    for entry in ranges:
        output_dir = Path(args.output_dir) / f"r{entry['shard']}"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        process = context.Process(
            target=_run_shard_worker,
            kwargs={
                "source1_path": source1,
                "index_path": Path(args.index),
                "output_dir": output_dir,
                "country": args.country,
                "shard": (int(entry["shard"]), 6),
                "model": model,
                "threshold": model.threshold,
                "candidate_cap": DEFAULT_SETTINGS.candidate_cap,
                "country_fallback_limit": DEFAULT_SETTINGS.country_fallback_limit,
                "stage1": args.stage1,
                "source1_limit": None,
                "enforce_exclusivity": True,
                "bucket_limit": args.bucket_limit,
                "start_ordinal": int(entry["start"]),
                "stop_ordinal": entry.get("stop"),
                "keep_work": True,
            },
        )
        processes.append((entry["shard"], output_dir, process))
    for _, _, process in processes:
        process.start()
    failures = 0
    for shard, output_dir, process in processes:
        process.join()
        ok = process.exitcode == 0 and (output_dir / "shard_summary.json").exists()
        if not ok:
            failures += 1
        summary_path = output_dir / "shard_summary.json"
        detail = summary_path.read_text(encoding="utf-8").replace("\n", " ") if summary_path.exists() else "no summary"
        log(f"shard{shard}: exit={process.exitcode} ok={ok} {detail}")
    if failures:
        log(f"FAILED ranges: {failures}")
        return 1
    log("ALL RANGES DONE — zip this output directory and send it back")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
