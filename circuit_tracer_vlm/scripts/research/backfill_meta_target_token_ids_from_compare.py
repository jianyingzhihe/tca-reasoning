#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill missing target_token_id values in answer-aligned meta CSVs from sample_compare_controlled.csv."
    )
    parser.add_argument("--meta-csv", required=True)
    parser.add_argument("--compare-csv", required=True)
    parser.add_argument("--which", required=True, choices=["a", "b"])
    parser.add_argument("--output", default="")
    parser.add_argument("--backup-suffix", default=".bak")
    parser.add_argument(
        "--same-target-only",
        action="store_true",
        help="Only backfill when a_target_token_id == b_target_token_id and both are non-empty.",
    )
    args = parser.parse_args()

    meta_path = Path(args.meta_csv).expanduser().resolve()
    compare_path = Path(args.compare_csv).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else meta_path

    meta_rows = _read_csv(meta_path)
    compare_rows = _read_csv(compare_path)
    if not meta_rows:
        raise ValueError(f"no rows loaded from {meta_path}")
    if not compare_rows:
        raise ValueError(f"no rows loaded from {compare_path}")

    compare_by_sample = {row.get("sample_id", ""): row for row in compare_rows}
    compare_col = f"{args.which}_target_token_id"

    updated = 0
    skipped_already_present = 0
    skipped_missing_compare = 0
    skipped_same_target = 0

    for row in meta_rows:
        if (row.get("target_token_id") or "").strip():
            skipped_already_present += 1
            continue

        sample_id = row.get("sample_id", "")
        compare_row = compare_by_sample.get(sample_id)
        if compare_row is None:
            skipped_missing_compare += 1
            continue

        compare_value = (compare_row.get(compare_col) or "").strip()
        a_value = (compare_row.get("a_target_token_id") or "").strip()
        b_value = (compare_row.get("b_target_token_id") or "").strip()
        if args.same_target_only and (not a_value or a_value != b_value):
            skipped_same_target += 1
            continue
        if not compare_value:
            skipped_missing_compare += 1
            continue

        row["target_token_id"] = compare_value
        updated += 1

    if output_path == meta_path:
        backup_path = meta_path.with_name(meta_path.name + args.backup_suffix)
        if not backup_path.exists():
            backup_path.write_text(meta_path.read_text(encoding="utf-8"), encoding="utf-8")

    _write_csv(output_path, meta_rows, list(meta_rows[0].keys()))

    print(f"[done] output={output_path}")
    print(f"[done] updated={updated}")
    print(f"[done] skipped_already_present={skipped_already_present}")
    print(f"[done] skipped_missing_compare={skipped_missing_compare}")
    print(f"[done] skipped_same_target={skipped_same_target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
