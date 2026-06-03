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
        description=(
            "Filter a region-mask manifest down to rows whose sample_id appears in a chosen clean-subset flag "
            "(for example clean_core_ab_pair or clean_core_any_run)."
        )
    )
    parser.add_argument("--clean-subset-csv", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--subset-flag", default="clean_core_ab_pair")
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    clean_rows = _read_csv(Path(args.clean_subset_csv).expanduser().resolve())
    manifest_rows = _read_csv(Path(args.manifest_csv).expanduser().resolve())
    if not clean_rows:
        raise ValueError("no clean-subset rows loaded")
    if not manifest_rows:
        raise ValueError("no manifest rows loaded")
    if args.subset_flag not in clean_rows[0]:
        raise ValueError(f"subset flag {args.subset_flag!r} missing from clean-subset CSV")

    selected_ids = {
        (row.get("sample_id") or "").strip()
        for row in clean_rows
        if row.get(args.subset_flag) == "True" and (row.get("sample_id") or "").strip()
    }
    if not selected_ids:
        raise ValueError(f"no selected sample ids for subset flag {args.subset_flag!r}")

    kept = [row for row in manifest_rows if (row.get("sample_id") or "").strip() in selected_ids]
    if not kept:
        raise ValueError("no overlap rows between clean subset and region-mask manifest")

    # Add a small breadcrumb for downstream bookkeeping.
    for row in kept:
        row["followup_subset_flag"] = args.subset_flag

    fieldnames = list(kept[0].keys())
    _write_csv(Path(args.out_csv).expanduser().resolve(), kept, fieldnames)
    print(f"[done] out_csv={Path(args.out_csv).expanduser().resolve()}")
    print(f"[done] selected_sample_ids={len(selected_ids)}")
    print(f"[done] kept_rows={len(kept)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
