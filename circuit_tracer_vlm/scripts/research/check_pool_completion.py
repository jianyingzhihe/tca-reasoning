#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def count_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def count_unique_sample_ids(csv_path: Path) -> tuple[int, int]:
    if not csv_path.exists():
        return 0, 0
    seen: set[str] = set()
    raw = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw += 1
            sid = (row.get("sample_id") or "").strip()
            if sid:
                seen.add(sid)
    return len(seen), raw


def main() -> int:
    parser = argparse.ArgumentParser(description="Check pool run completion by unique sample_id counts.")
    parser.add_argument("--run-dir", required=True, help="Pool run dir, e.g. outputs/phase_ab/eval_pool/<run_tag>")
    parser.add_argument("--workers", type=int, default=0, help="Worker count; 0 means auto-detect from splits/part_*.csv")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if incomplete or duplicates detected.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    split_dir = run_dir / "splits"
    if not split_dir.exists():
        raise FileNotFoundError(f"split dir not found: {split_dir}")

    if args.workers > 0:
        workers = args.workers
    else:
        workers = len(sorted(split_dir.glob("part_*.csv")))
        if workers == 0:
            raise ValueError(f"no split csv files found in {split_dir}")

    total_done = 0
    total_target = 0
    has_dup = False
    incomplete = False

    for i in range(workers):
        split_csv = split_dir / f"part_{i}.csv"
        eval_csv = run_dir / f"part_{i}_eval.csv"
        target = count_rows(split_csv)
        unique_done, raw_rows = count_unique_sample_ids(eval_csv)
        done = min(unique_done, target)
        dup = max(0, raw_rows - unique_done)
        miss = max(0, target - done)
        total_done += done
        total_target += target
        has_dup = has_dup or dup > 0
        incomplete = incomplete or miss > 0
        print(f"W{i}: done={done}/{target} missing={miss} dup_rows={dup} file={eval_csv}")

    pct = (100.0 * total_done / total_target) if total_target else 0.0
    print(f"TOTAL: done={total_done}/{total_target} ({pct:.2f}%)")
    print(f"STATUS: {'COMPLETE' if (not incomplete) else 'INCOMPLETE'} | DUPLICATES: {'YES' if has_dup else 'NO'}")

    if args.strict and (incomplete or has_dup):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
