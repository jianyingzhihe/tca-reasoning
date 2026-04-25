#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from validate_answer_aligned_queue import inspect_queue


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _safe_float(row: dict[str, str], key: str) -> float | None:
    value = (row.get(key) or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def main() -> int:
    parser = argparse.ArgumentParser(description="Show a compact summary of answer-aligned queue outputs.")
    parser.add_argument("--run-tag-base", required=True, help="Queue run tag base, without the trailing bucket name.")
    parser.add_argument("--buckets", default="A0_B1,A1_B1,A0_B0,A1_B0")
    parser.add_argument("--outputs-root", default="outputs/phase_ab/ab_answer_aligned")
    parser.add_argument("--work-root", default="research/work/ab_answer_aligned")
    args = parser.parse_args()

    expected_buckets = [x.strip() for x in args.buckets.split(",") if x.strip()]
    outputs_root = Path(args.outputs_root).expanduser().resolve()
    work_root = Path(args.work_root).expanduser().resolve()

    results, status_tsv = inspect_queue(
        run_tag_base=args.run_tag_base,
        expected_buckets=expected_buckets,
        outputs_root=outputs_root,
        work_root=work_root,
    )

    print(f"[info] status_tsv={status_tsv}")
    total_selected = 0
    total_success_a = 0
    total_success_b = 0
    total_size = 0

    for res in results:
        total_selected += res.selected_count
        total_success_a += res.success_a_count
        total_success_b += res.success_b_count
        total_size += res.output_size_bytes

        compare_dir = outputs_root / res.run_tag / res.run_tag
        compare_row: dict[str, str] | None = None
        if (compare_dir / "bucket_summary_controlled.csv").exists():
            rows = _read_csv(compare_dir / "bucket_summary_controlled.csv")
            compare_row = next((r for r in rows if (r.get("bucket") or "").strip() == "__all__"), None)

        parts = [
            f"[bucket] {res.bucket}",
            f"status={'ok' if res.status_ok else 'fail'}",
            f"selected={res.selected_count}",
            f"A_ok={res.success_a_count}",
            f"B_ok={res.success_b_count}",
            f"pt_a={res.pt_a_count}",
            f"pt_b={res.pt_b_count}",
            f"size={_human_bytes(res.output_size_bytes)}",
        ]

        if compare_row is not None:
            node_overlap = _safe_float(compare_row, "node_overlap_jaccard")
            edge_overlap = _safe_float(compare_row, "edge_overlap_jaccard")
            delta_error = _safe_float(compare_row, "delta_target_error_ratio")
            delta_top3 = _safe_float(compare_row, "delta_target_top3_concentration")
            if node_overlap is not None:
                parts.append(f"node_jaccard={node_overlap:.4f}")
            if edge_overlap is not None:
                parts.append(f"edge_jaccard={edge_overlap:.4f}")
            if delta_error is not None:
                parts.append(f"delta_error={delta_error:.4f}")
            if delta_top3 is not None:
                parts.append(f"delta_top3={delta_top3:.4f}")
        else:
            parts.append("compare=<skipped-or-missing>")

        print(" ".join(parts))

        if not res.status_ok:
            for msg in res.messages:
                print(f"  - {msg}")

    print(
        f"[total] buckets={len(results)} selected={total_selected} "
        f"A_ok={total_success_a} B_ok={total_success_b} size={_human_bytes(total_size)}"
    )

    compare_sample_rows: list[dict[str, str]] = []
    for res in results:
        compare_csv = outputs_root / res.run_tag / res.run_tag / "sample_compare_controlled.csv"
        if compare_csv.exists():
            compare_sample_rows.extend(_read_csv(compare_csv))

    if compare_sample_rows:
        node_vals = [v for r in compare_sample_rows if (v := _safe_float(r, "node_overlap_jaccard")) is not None]
        edge_vals = [v for r in compare_sample_rows if (v := _safe_float(r, "edge_overlap_jaccard")) is not None]
        error_vals = [v for r in compare_sample_rows if (v := _safe_float(r, "delta_target_error_ratio")) is not None]
        top3_vals = [v for r in compare_sample_rows if (v := _safe_float(r, "delta_target_top3_concentration")) is not None]
        print(
            "[compare] "
            f"rows={len(compare_sample_rows)} "
            f"mean_node_jaccard={_mean(node_vals):.4f} "
            f"mean_edge_jaccard={_mean(edge_vals):.4f} "
            f"mean_delta_error={_mean(error_vals):.4f} "
            f"mean_delta_top3={_mean(top3_vals):.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
