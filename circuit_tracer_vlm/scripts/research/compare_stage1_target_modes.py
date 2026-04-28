#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


CORE_METRICS = [
    "node_overlap_jaccard",
    "edge_overlap_jaccard",
    "delta_target_error_ratio",
    "delta_target_feature_ratio",
    "delta_target_token_ratio",
    "delta_target_top3_concentration",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: str | None) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except Exception:
        return math.nan


def _fmt(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.10g}"


def _load_metric_map(summary_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = _read_csv(summary_dir / "stage1_metric_summary.csv")
    return {(r["bucket"], r["metric"]): r for r in rows}


def _same_target_counts(summary_dir: Path) -> dict[str, tuple[int, int]]:
    rows = _read_csv(summary_dir / "stage1_combined_samples.csv")
    out: dict[str, tuple[int, int]] = {}
    by_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_bucket[row["bucket"]].append(row)
    by_bucket["__all__"] = rows
    for bucket, part in by_bucket.items():
        same = sum(r.get("a_target_token_id", "") == r.get("b_target_token_id", "") for r in part)
        out[bucket] = (same, len(part))
    return out


def _aggregate_filtered_overlap(summary_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    path = summary_dir / "stage1_generic_filtered_overlap.csv"
    if not path.exists():
        return {}
    rows = _read_csv(path)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        bucket = row["bucket"]
        same_flag = row.get("same_target_token", "")
        grouped[(bucket, "all")].append(row)
        grouped[("__all__", "all")].append(row)
        grouped[(bucket, "same" if same_flag == "True" else "diff")].append(row)
        grouped[("__all__", "same" if same_flag == "True" else "diff")].append(row)

    out: dict[tuple[str, str], dict[str, str]] = {}
    for key, part in grouped.items():
        raw_vals = [_safe_float(r.get("raw_node_overlap_jaccard")) for r in part]
        filtered_vals = [_safe_float(r.get("generic_filtered_node_overlap_jaccard")) for r in part]
        removed_a = [_safe_float(r.get("generic_nodes_removed_a")) for r in part]
        removed_b = [_safe_float(r.get("generic_nodes_removed_b")) for r in part]
        out[key] = {
            "bucket": key[0],
            "subset": key[1],
            "n": str(len(part)),
            "raw_node_overlap_mean": _fmt(sum(raw_vals) / len(raw_vals) if raw_vals else math.nan),
            "filtered_node_overlap_mean": _fmt(
                sum(filtered_vals) / len(filtered_vals) if filtered_vals else math.nan
            ),
            "removed_a_mean": _fmt(sum(removed_a) / len(removed_a) if removed_a else math.nan),
            "removed_b_mean": _fmt(sum(removed_b) / len(removed_b) if removed_b else math.nan),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare predicted-target and gold-target Stage 1 summaries.")
    parser.add_argument("--predicted-summary-dir", required=True)
    parser.add_argument("--gold-summary-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--predicted-label", default="predicted_target")
    parser.add_argument("--gold-label", default="gold_target")
    args = parser.parse_args()

    pred_dir = Path(args.predicted_summary_dir).expanduser().resolve()
    gold_dir = Path(args.gold_summary_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    pred_metric_map = _load_metric_map(pred_dir)
    gold_metric_map = _load_metric_map(gold_dir)
    pred_same = _same_target_counts(pred_dir)
    gold_same = _same_target_counts(gold_dir)
    pred_filtered = _aggregate_filtered_overlap(pred_dir)
    gold_filtered = _aggregate_filtered_overlap(gold_dir)

    buckets = ["A0_B0", "A0_B1", "A1_B0", "A1_B1", "__all__"]
    metric_rows: list[dict[str, str]] = []
    for bucket in buckets:
        for metric in CORE_METRICS:
            pred = pred_metric_map.get((bucket, metric))
            gold = gold_metric_map.get((bucket, metric))
            if not pred or not gold:
                continue
            pred_mean = _safe_float(pred.get("mean"))
            gold_mean = _safe_float(gold.get("mean"))
            metric_rows.append(
                {
                    "bucket": bucket,
                    "metric": metric,
                    f"{args.predicted_label}_mean": _fmt(pred_mean),
                    f"{args.gold_label}_mean": _fmt(gold_mean),
                    f"delta_{args.gold_label}_minus_{args.predicted_label}": _fmt(gold_mean - pred_mean),
                }
            )

    same_rows: list[dict[str, str]] = []
    for bucket in buckets:
        pred_pair = pred_same.get(bucket)
        gold_pair = gold_same.get(bucket)
        if not pred_pair or not gold_pair:
            continue
        pred_same_count, pred_total = pred_pair
        gold_same_count, gold_total = gold_pair
        same_rows.append(
            {
                "bucket": bucket,
                f"{args.predicted_label}_same_target_count": str(pred_same_count),
                f"{args.predicted_label}_total": str(pred_total),
                f"{args.predicted_label}_same_target_rate": _fmt(pred_same_count / pred_total if pred_total else math.nan),
                f"{args.gold_label}_same_target_count": str(gold_same_count),
                f"{args.gold_label}_total": str(gold_total),
                f"{args.gold_label}_same_target_rate": _fmt(gold_same_count / gold_total if gold_total else math.nan),
            }
        )

    filtered_rows: list[dict[str, str]] = []
    for bucket in buckets:
        for subset in ["all", "same", "diff"]:
            pred = pred_filtered.get((bucket, subset))
            gold = gold_filtered.get((bucket, subset))
            if not pred and not gold:
                continue
            filtered_rows.append(
                {
                    "bucket": bucket,
                    "subset": subset,
                    f"{args.predicted_label}_raw_node_overlap_mean": pred.get("raw_node_overlap_mean", "") if pred else "",
                    f"{args.predicted_label}_filtered_node_overlap_mean": pred.get("filtered_node_overlap_mean", "") if pred else "",
                    f"{args.gold_label}_raw_node_overlap_mean": gold.get("raw_node_overlap_mean", "") if gold else "",
                    f"{args.gold_label}_filtered_node_overlap_mean": gold.get("filtered_node_overlap_mean", "") if gold else "",
                }
            )

    _write_csv(
        out_dir / "target_mode_metric_comparison.csv",
        metric_rows,
        [
            "bucket",
            "metric",
            f"{args.predicted_label}_mean",
            f"{args.gold_label}_mean",
            f"delta_{args.gold_label}_minus_{args.predicted_label}",
        ],
    )
    _write_csv(
        out_dir / "target_mode_same_target_comparison.csv",
        same_rows,
        [
            "bucket",
            f"{args.predicted_label}_same_target_count",
            f"{args.predicted_label}_total",
            f"{args.predicted_label}_same_target_rate",
            f"{args.gold_label}_same_target_count",
            f"{args.gold_label}_total",
            f"{args.gold_label}_same_target_rate",
        ],
    )
    if filtered_rows:
        _write_csv(
            out_dir / "target_mode_filtered_overlap_comparison.csv",
            filtered_rows,
            [
                "bucket",
                "subset",
                f"{args.predicted_label}_raw_node_overlap_mean",
                f"{args.predicted_label}_filtered_node_overlap_mean",
                f"{args.gold_label}_raw_node_overlap_mean",
                f"{args.gold_label}_filtered_node_overlap_mean",
            ],
        )

    lines = [
        "# Target Mode Comparison",
        "",
        f"- predicted summary: `{pred_dir}`",
        f"- gold summary: `{gold_dir}`",
        "",
        "## Same Target Rate",
        "",
        "| bucket | predicted same-target | gold same-target |",
        "|---|---:|---:|",
    ]
    for row in same_rows:
        lines.append(
            "| {bucket} | {pred} | {gold} |".format(
                bucket=row["bucket"],
                pred=row[f"{args.predicted_label}_same_target_rate"],
                gold=row[f"{args.gold_label}_same_target_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## Core Metrics",
            "",
            "| bucket | metric | predicted | gold | gold-predicted |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in metric_rows:
        lines.append(
            "| {bucket} | {metric} | {pred} | {gold} | {delta} |".format(
                bucket=row["bucket"],
                metric=row["metric"],
                pred=row[f"{args.predicted_label}_mean"],
                gold=row[f"{args.gold_label}_mean"],
                delta=row[f"delta_{args.gold_label}_minus_{args.predicted_label}"],
            )
        )
    if filtered_rows:
        lines.extend(
            [
                "",
                "## Generic-Filtered Overlap",
                "",
                "| bucket | subset | predicted filtered | gold filtered |",
                "|---|---|---:|---:|",
            ]
        )
        for row in filtered_rows:
            lines.append(
                "| {bucket} | {subset} | {pred} | {gold} |".format(
                    bucket=row["bucket"],
                    subset=row["subset"],
                    pred=row.get(f"{args.predicted_label}_filtered_node_overlap_mean", ""),
                    gold=row.get(f"{args.gold_label}_filtered_node_overlap_mean", ""),
                )
            )

    md_path = out_dir / "target_mode_comparison.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
