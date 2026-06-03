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
            "Build a focused Stage 1 follow-up smoke CSV by filtering bucket-level intervention_smoke files "
            "using flags from alignment_clean_subset_rows.csv."
        )
    )
    parser.add_argument("--clean-subset-csv", required=True)
    parser.add_argument("--outputs-root", default="outputs/phase_ab/ab_answer_aligned")
    parser.add_argument("--run-tag-base", required=True)
    parser.add_argument(
        "--subset-flag",
        default="clean_core_ab_pair",
        help="Column in the clean-subset CSV to require as True, e.g. clean_core_ab_pair or clean_core_any_run.",
    )
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    clean_rows = _read_csv(Path(args.clean_subset_csv).expanduser().resolve())
    if not clean_rows:
        raise ValueError("no clean-subset rows loaded")
    if args.subset_flag not in clean_rows[0]:
        raise ValueError(f"subset flag {args.subset_flag!r} missing from clean-subset CSV")

    selected: dict[str, set[str]] = {}
    extra_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in clean_rows:
        if row.get(args.subset_flag) != "True":
            continue
        bucket = row.get("bucket", "")
        sample_id = row.get("sample_id", "")
        if not bucket or not sample_id:
            continue
        selected.setdefault(bucket, set()).add(sample_id)
        extra_by_key[(bucket, sample_id)] = row

    if not selected:
        raise ValueError(f"no rows selected for subset flag {args.subset_flag!r}")

    outputs_root = Path(args.outputs_root).expanduser().resolve()
    out_rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for bucket, sample_ids in sorted(selected.items()):
        smoke_csv = outputs_root / f"{args.run_tag_base}_{bucket}" / f"intervention_smoke_{bucket}.csv"
        smoke_rows = _read_csv(smoke_csv)
        for row in smoke_rows:
            sample_id = row.get("sample_id", "")
            if sample_id not in sample_ids:
                continue
            merged = dict(row)
            extra = extra_by_key.get((bucket, sample_id), {})
            merged["followup_subset_flag"] = args.subset_flag
            merged["followup_priority"] = extra.get("priority", "")
            merged["followup_rank"] = extra.get("rank", "")
            merged["followup_visual_type_label"] = extra.get("visual_type_label", "")
            merged["followup_knowledge_level_label"] = extra.get("knowledge_level_label", "")
            out_rows.append(merged)
            if fieldnames is None:
                fieldnames = list(merged.keys())

    if not out_rows or fieldnames is None:
        raise ValueError("no smoke rows selected")

    _write_csv(Path(args.out_csv).expanduser().resolve(), out_rows, fieldnames)
    print(f"[done] out_csv={Path(args.out_csv).expanduser().resolve()}")
    print(f"[done] selected_rows={len(out_rows)}")
    print(f"[done] selected_buckets={len(selected)}")
    for bucket, sample_ids in sorted(selected.items()):
        print(f"[done] bucket={bucket} selected_samples={len(sample_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
