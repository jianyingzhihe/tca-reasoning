#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


METRICS = [
    "node_overlap_jaccard",
    "edge_overlap_jaccard",
    "delta_target_total_in_abs",
    "delta_target_top1_concentration",
    "delta_target_top3_concentration",
    "delta_target_top10_concentration",
    "delta_target_error_ratio",
    "delta_target_feature_ratio",
    "delta_target_token_ratio",
    "delta_traced_nodes",
    "delta_traced_edges",
    "delta_traced_max_depth",
    "delta_traced_total_path_mass",
    "delta_traced_min_abs_weight",
    "delta_traced_max_abs_weight",
]


RANK_METRICS = [
    ("node_overlap_jaccard", False),
    ("edge_overlap_jaccard", False),
    ("delta_target_error_ratio", True),
    ("delta_target_feature_ratio", False),
    ("delta_target_token_ratio", True),
    ("delta_traced_edges", True),
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
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


def _mean(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return math.nan
    return sum(vals) / len(vals)


def _median(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return math.nan
    return statistics.median(vals)


def _stdev(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    if len(vals) < 2:
        return 0.0
    return statistics.stdev(vals)


def _fmt(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.10g}"


def _jaccard_tuples(a: set[tuple[str, str, str, str]], b: set[tuple[str, str, str, str]]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    if union == 0:
        return 1.0
    return len(a & b) / union


def _infer_bucket(path: Path) -> str:
    match = re.search(r"(A[01]_B[01])$", path.name)
    if match:
        return match.group(1)
    match = re.search(r"(A[01]_B[01])", path.name)
    if match:
        return match.group(1)
    return path.name


def _discover_compare_dirs(root: Path, path_contains: str = "") -> list[Path]:
    if (root / "sample_compare_controlled.csv").exists():
        dirs = [root]
    else:
        dirs = sorted({p.parent for p in root.rglob("sample_compare_controlled.csv")})
    if path_contains:
        dirs = [p for p in dirs if path_contains in str(p)]
    return dirs


def _summarize_metric(rows: list[dict[str, str]], metric: str) -> dict[str, str]:
    values = [_safe_float(r.get(metric)) for r in rows]
    vals = [v for v in values if not math.isnan(v)]
    pos = sum(v > 0 for v in vals)
    neg = sum(v < 0 for v in vals)
    zero = sum(v == 0 for v in vals)
    return {
        "metric": metric,
        "n": str(len(vals)),
        "mean": _fmt(_mean(vals)),
        "median": _fmt(_median(vals)),
        "stdev": _fmt(_stdev(vals)),
        "min": _fmt(min(vals) if vals else math.nan),
        "max": _fmt(max(vals) if vals else math.nan),
        "pos": str(pos),
        "neg": str(neg),
        "zero": str(zero),
    }


def _build_metric_summary(sample_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    by_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sample_rows:
        by_bucket[row["bucket"]].append(row)

    for bucket in sorted(by_bucket):
        rows = by_bucket[bucket]
        for metric in METRICS:
            if metric in rows[0]:
                item = {"bucket": bucket}
                item.update(_summarize_metric(rows, metric))
                out.append(item)

    for metric in METRICS:
        if metric in sample_rows[0]:
            item = {"bucket": "__all__"}
            item.update(_summarize_metric(sample_rows, metric))
            out.append(item)
    return out


def _build_repeated_nodes(compare_dirs: list[Path]) -> list[dict[str, str]]:
    counts: Counter[tuple[str, str, str, str, str, str]] = Counter()
    sample_counts: Counter[tuple[str, str, str, str, str, str]] = Counter()
    for compare_dir in compare_dirs:
        bucket = _infer_bucket(compare_dir)
        nodes_csv = compare_dir / "nodes_detailed_controlled.csv"
        if not nodes_csv.exists():
            continue
        seen_in_sample = set()
        for row in _read_csv(nodes_csv):
            key = (
                bucket,
                row.get("run", ""),
                row.get("node_type", ""),
                row.get("layer", ""),
                row.get("pos", ""),
                row.get("feature_id", "") or row.get("token_id", ""),
            )
            counts[key] += 1
            seen_in_sample.add((row.get("sample_id", ""), key))
        for _, key in seen_in_sample:
            sample_counts[key] += 1

    rows = []
    for key, count in counts.most_common():
        bucket, run, node_type, layer, pos, node_value = key
        rows.append(
            {
                "bucket": bucket,
                "run": run,
                "node_type": node_type,
                "layer": layer,
                "pos": pos,
                "feature_or_token_id": node_value,
                "node_count": str(count),
                "sample_count": str(sample_counts[key]),
            }
        )
    return rows


def _node_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("node_type", ""),
        row.get("layer", ""),
        row.get("pos", ""),
        row.get("feature_id", "") or row.get("token_id", ""),
    )


def _load_nodes_by_sample(compare_dirs: list[Path]) -> dict[tuple[str, str, str], set[tuple[str, str, str, str]]]:
    out: dict[tuple[str, str, str], set[tuple[str, str, str, str]]] = defaultdict(set)
    for compare_dir in compare_dirs:
        nodes_csv = compare_dir / "nodes_detailed_controlled.csv"
        if not nodes_csv.exists():
            continue
        for row in _read_csv(nodes_csv):
            sample_id = row.get("sample_id", "")
            bucket = row.get("bucket", "") or _infer_bucket(compare_dir)
            run = row.get("run", "")
            if sample_id and run:
                out[(bucket, sample_id, run)].add(_node_key(row))
    return out


def _build_generic_filtered_overlap(
    sample_rows: list[dict[str, str]],
    compare_dirs: list[Path],
    *,
    generic_sample_threshold: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    nodes_by_sample = _load_nodes_by_sample(compare_dirs)
    node_sample_counter: Counter[tuple[str, str, str, str]] = Counter()
    for nodes in nodes_by_sample.values():
        for key in nodes:
            node_sample_counter[key] += 1

    generic_nodes = {
        key for key, count in node_sample_counter.items() if count >= generic_sample_threshold
    }

    generic_rows = [
        {
            "node_type": key[0],
            "layer": key[1],
            "pos": key[2],
            "feature_or_token_id": key[3],
            "sample_count": str(count),
            "is_generic": str(key in generic_nodes),
        }
        for key, count in node_sample_counter.most_common()
    ]

    filtered_rows: list[dict[str, str]] = []
    for row in sample_rows:
        bucket = row["bucket"]
        sample_id = row["sample_id"]
        nodes_a = nodes_by_sample.get((bucket, sample_id, "A"), set())
        nodes_b = nodes_by_sample.get((bucket, sample_id, "B"), set())
        nodes_a_filtered = nodes_a - generic_nodes
        nodes_b_filtered = nodes_b - generic_nodes
        filtered_rows.append(
            {
                "sample_id": sample_id,
                "bucket": bucket,
                "same_target_token": row.get("same_target_token", ""),
                "raw_node_overlap_jaccard": row.get("node_overlap_jaccard", ""),
                "generic_filtered_node_overlap_jaccard": _fmt(
                    _jaccard_tuples(nodes_a_filtered, nodes_b_filtered)
                ),
                "a_nodes_raw": str(len(nodes_a)),
                "b_nodes_raw": str(len(nodes_b)),
                "a_nodes_filtered": str(len(nodes_a_filtered)),
                "b_nodes_filtered": str(len(nodes_b_filtered)),
                "generic_nodes_removed_a": str(len(nodes_a & generic_nodes)),
                "generic_nodes_removed_b": str(len(nodes_b & generic_nodes)),
            }
        )
    return filtered_rows, generic_rows


def _build_divergent_cases(sample_rows: list[dict[str, str]], top_n: int) -> list[dict[str, str]]:
    rows = []
    for metric, descending in RANK_METRICS:
        if metric not in sample_rows[0]:
            continue
        ranked = sorted(
            sample_rows,
            key=lambda r: _safe_float(r.get(metric)),
            reverse=descending,
        )
        for rank, row in enumerate(ranked[:top_n], start=1):
            rows.append(
                {
                    "rank_metric": metric,
                    "rank_direction": "desc" if descending else "asc",
                    "rank": str(rank),
                    "sample_id": row.get("sample_id", ""),
                    "bucket": row.get("bucket", ""),
                    "metric_value": row.get(metric, ""),
                    "a_target_token_id": row.get("a_target_token_id", ""),
                    "b_target_token_id": row.get("b_target_token_id", ""),
                    "same_target_token": str(row.get("a_target_token_id", "") == row.get("b_target_token_id", "")),
                    "node_overlap_jaccard": row.get("node_overlap_jaccard", ""),
                    "edge_overlap_jaccard": row.get("edge_overlap_jaccard", ""),
                    "delta_target_error_ratio": row.get("delta_target_error_ratio", ""),
                    "delta_target_feature_ratio": row.get("delta_target_feature_ratio", ""),
                    "delta_target_token_ratio": row.get("delta_target_token_ratio", ""),
                    "delta_traced_edges": row.get("delta_traced_edges", ""),
                }
            )
    return rows


def _write_markdown_summary(
    path: Path,
    *,
    sample_rows: list[dict[str, str]],
    metric_rows: list[dict[str, str]],
    repeated_rows: list[dict[str, str]],
    divergent_rows: list[dict[str, str]],
) -> None:
    by_bucket = defaultdict(list)
    for row in sample_rows:
        by_bucket[row["bucket"]].append(row)

    def metric_mean(bucket: str, metric: str) -> str:
        for row in metric_rows:
            if row["bucket"] == bucket and row["metric"] == metric:
                return row["mean"]
        return ""

    lines = [
        "# Stage 1 Generated Summary",
        "",
        f"- samples: {len(sample_rows)}",
        f"- buckets: {', '.join(sorted(by_bucket))}",
        "",
        "## Bucket Means",
        "",
        "| bucket | n | node Jaccard | edge Jaccard | delta error | delta feature | delta token |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for bucket in sorted(by_bucket):
        lines.append(
            "| {bucket} | {n} | {node} | {edge} | {err} | {feat} | {tok} |".format(
                bucket=bucket,
                n=len(by_bucket[bucket]),
                node=metric_mean(bucket, "node_overlap_jaccard"),
                edge=metric_mean(bucket, "edge_overlap_jaccard"),
                err=metric_mean(bucket, "delta_target_error_ratio"),
                feat=metric_mean(bucket, "delta_target_feature_ratio"),
                tok=metric_mean(bucket, "delta_target_token_ratio"),
            )
        )

    same_target = sum(r.get("a_target_token_id") == r.get("b_target_token_id") for r in sample_rows)
    lines.extend(
        [
            "",
            "## Target Token Control Check",
            "",
            f"- same target token: {same_target}/{len(sample_rows)}",
            "",
            "## Top Repeated Nodes",
            "",
            "| bucket | run | type | layer | feature/token | sample count | node count |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in repeated_rows[:20]:
        lines.append(
            "| {bucket} | {run} | {typ} | {layer} | {val} | {samples} | {nodes} |".format(
                bucket=row["bucket"],
                run=row["run"],
                typ=row["node_type"],
                layer=row["layer"],
                val=row["feature_or_token_id"],
                samples=row["sample_count"],
                nodes=row["node_count"],
            )
        )

    lines.extend(
        [
            "",
            "## Top Divergent Cases",
            "",
            "| metric | rank | sample | bucket | value | same target |",
            "|---|---:|---|---|---:|---|",
        ]
    )
    for row in divergent_rows[:30]:
        lines.append(
            "| {metric} | {rank} | {sample} | {bucket} | {value} | {same} |".format(
                metric=row["rank_metric"],
                rank=row["rank"],
                sample=row["sample_id"],
                bucket=row["bucket"],
                value=row["metric_value"],
                same=row["same_target_token"],
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize answer-aligned A/B compare outputs.")
    parser.add_argument("--compare-root", required=True, help="Root containing compare dirs or one compare dir.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--path-contains",
        default="",
        help="Optional substring filter for recursively discovered compare dirs.",
    )
    parser.add_argument(
        "--generic-sample-threshold",
        type=int,
        default=30,
        help=(
            "Mark a node as generic if it appears in at least this many "
            "sample-run graphs across the summarized run."
        ),
    )
    args = parser.parse_args()

    compare_root = Path(args.compare_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    compare_dirs = _discover_compare_dirs(compare_root, args.path_contains)
    if not compare_dirs:
        raise ValueError(f"no compare dirs found under {compare_root}")

    sample_rows: list[dict[str, str]] = []
    for compare_dir in compare_dirs:
        bucket = _infer_bucket(compare_dir)
        for row in _read_csv(compare_dir / "sample_compare_controlled.csv"):
            row = dict(row)
            row["source_dir"] = str(compare_dir)
            row["bucket"] = row.get("bucket") or bucket
            row["same_target_token"] = str(row.get("a_target_token_id", "") == row.get("b_target_token_id", ""))
            sample_rows.append(row)

    if not sample_rows:
        raise ValueError("no sample rows loaded")

    combined_fields = list(sample_rows[0].keys())
    if "source_dir" not in combined_fields:
        combined_fields.append("source_dir")
    if "same_target_token" not in combined_fields:
        combined_fields.append("same_target_token")

    metric_rows = _build_metric_summary(sample_rows)
    repeated_rows = _build_repeated_nodes(compare_dirs)
    filtered_overlap_rows, generic_node_rows = _build_generic_filtered_overlap(
        sample_rows,
        compare_dirs,
        generic_sample_threshold=args.generic_sample_threshold,
    )
    divergent_rows = _build_divergent_cases(sample_rows, args.top_n)

    _write_csv(out_dir / "stage1_combined_samples.csv", sample_rows, combined_fields)
    _write_csv(
        out_dir / "stage1_metric_summary.csv",
        metric_rows,
        ["bucket", "metric", "n", "mean", "median", "stdev", "min", "max", "pos", "neg", "zero"],
    )
    _write_csv(
        out_dir / "stage1_repeated_nodes.csv",
        repeated_rows,
        ["bucket", "run", "node_type", "layer", "pos", "feature_or_token_id", "node_count", "sample_count"],
    )
    _write_csv(
        out_dir / "stage1_generic_nodes.csv",
        generic_node_rows,
        ["node_type", "layer", "pos", "feature_or_token_id", "sample_count", "is_generic"],
    )
    _write_csv(
        out_dir / "stage1_generic_filtered_overlap.csv",
        filtered_overlap_rows,
        [
            "sample_id",
            "bucket",
            "same_target_token",
            "raw_node_overlap_jaccard",
            "generic_filtered_node_overlap_jaccard",
            "a_nodes_raw",
            "b_nodes_raw",
            "a_nodes_filtered",
            "b_nodes_filtered",
            "generic_nodes_removed_a",
            "generic_nodes_removed_b",
        ],
    )
    _write_csv(
        out_dir / "stage1_divergent_cases.csv",
        divergent_rows,
        [
            "rank_metric",
            "rank_direction",
            "rank",
            "sample_id",
            "bucket",
            "metric_value",
            "a_target_token_id",
            "b_target_token_id",
            "same_target_token",
            "node_overlap_jaccard",
            "edge_overlap_jaccard",
            "delta_target_error_ratio",
            "delta_target_feature_ratio",
            "delta_target_token_ratio",
            "delta_traced_edges",
        ],
    )
    _write_markdown_summary(
        out_dir / "stage1_generated_summary.md",
        sample_rows=sample_rows,
        metric_rows=metric_rows,
        repeated_rows=repeated_rows,
        divergent_rows=divergent_rows,
    )

    print(f"[done] compare_dirs={len(compare_dirs)}")
    print(f"[done] samples={len(sample_rows)}")
    print(f"[done] out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
