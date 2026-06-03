#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
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
            "Filter bucket-level intervention_smoke CSVs down to the sample IDs listed in a "
            "modality follow-up pack. This lets later modality pilots reuse the existing smoke "
            "cache while keeping sample selection aligned with human labels."
        )
    )
    parser.add_argument("--pack-csv", required=True)
    parser.add_argument("--smoke-cache-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    pack_rows = _read_csv(Path(args.pack_csv).expanduser().resolve())
    smoke_cache_dir = Path(args.smoke_cache_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not pack_rows:
        raise ValueError("no pack rows loaded")

    pack_by_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    sample_ids_by_bucket: dict[str, set[str]] = defaultdict(set)
    for row in pack_rows:
        bucket = (row.get("bucket") or "").strip()
        sample_id = (row.get("sample_id") or "").strip()
        if not bucket or not sample_id:
            continue
        pack_by_bucket[bucket].append(row)
        sample_ids_by_bucket[bucket].add(sample_id)

    if not sample_ids_by_bucket:
        raise ValueError("no valid bucket/sample rows found in pack CSV")

    selected_rows_all: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []
    selected_fieldnames: list[str] | None = None

    for bucket in sorted(sample_ids_by_bucket):
        smoke_path = smoke_cache_dir / f"intervention_smoke_{bucket}.csv"
        smoke_rows = _read_csv(smoke_path)
        selected_ids = sample_ids_by_bucket[bucket]
        pack_meta_by_sample = {
            (row.get("sample_id") or "").strip(): row
            for row in pack_by_bucket[bucket]
        }

        selected_rows: list[dict[str, str]] = []
        run_counts: Counter[str] = Counter()
        sign_counts: Counter[str] = Counter()
        for row in smoke_rows:
            sample_id = (row.get("sample_id") or "").strip()
            if sample_id not in selected_ids:
                continue
            merged = dict(row)
            pack_meta = pack_meta_by_sample.get(sample_id, {})
            merged["followup_visual_type_label"] = pack_meta.get("visual_type_label", "")
            merged["followup_knowledge_level_label"] = pack_meta.get("knowledge_level_label", "")
            merged["followup_priority"] = pack_meta.get("priority", "")
            merged["followup_display_question"] = pack_meta.get("display_question", "")
            selected_rows.append(merged)
            run_counts[row.get("run", "")] += 1
            try:
                delta = float(row.get("delta_target_logit") or "")
            except Exception:
                delta = 0.0
            sign_counts["support_like" if delta < 0 else "suppressor_like" if delta > 0 else "zero"] += 1

        if selected_rows and selected_fieldnames is None:
            selected_fieldnames = list(selected_rows[0].keys())
        selected_rows_all.extend(selected_rows)
        _write_csv(out_dir / f"intervention_smoke_{bucket}.csv", selected_rows, selected_fieldnames or [])

        summary_rows.append(
            {
                "bucket": bucket,
                "n_pack_samples": str(len(selected_ids)),
                "n_selected_rows": str(len(selected_rows)),
                "n_selected_sample_ids": str(len({row.get('sample_id', '') for row in selected_rows})),
                "n_run_A_rows": str(run_counts.get("A", 0)),
                "n_run_B_rows": str(run_counts.get("B", 0)),
                "n_support_like_rows": str(sign_counts.get("support_like", 0)),
                "n_suppressor_like_rows": str(sign_counts.get("suppressor_like", 0)),
                "n_zero_rows": str(sign_counts.get("zero", 0)),
            }
        )

    if selected_rows_all and selected_fieldnames:
        _write_csv(out_dir / "intervention_smoke_all.csv", selected_rows_all, selected_fieldnames)
    _write_csv(
        out_dir / "summary.csv",
        summary_rows,
        [
            "bucket",
            "n_pack_samples",
            "n_selected_rows",
            "n_selected_sample_ids",
            "n_run_A_rows",
            "n_run_B_rows",
            "n_support_like_rows",
            "n_suppressor_like_rows",
            "n_zero_rows",
        ],
    )
    print(f"[done] out_dir={out_dir}")
    print(f"[done] total_rows={len(selected_rows_all)}")
    for row in summary_rows:
        print(
            "[done] bucket={bucket} samples={n_selected_sample_ids}/{n_pack_samples} rows={n_selected_rows}".format(
                **row
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
