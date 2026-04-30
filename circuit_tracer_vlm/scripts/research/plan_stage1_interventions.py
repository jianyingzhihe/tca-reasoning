#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


BUCKETS = ["A0_B0", "A0_B1", "A1_B0", "A1_B1"]


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
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except Exception:
        return math.nan


def _fmt(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.10g}"


def _same_nonempty_target(row: dict[str, str], overlap_row: dict[str, str]) -> bool:
    a = (row.get("a_target_token_id") or overlap_row.get("a_target_token_id") or "").strip()
    b = (row.get("b_target_token_id") or overlap_row.get("b_target_token_id") or "").strip()
    return bool(a) and a == b


def _load_generic_features(path: Path | None) -> set[tuple[str, str, str]]:
    if path is None or not path.exists():
        return set()
    out = set()
    for row in _read_csv(path):
        if row.get("is_generic", "").lower() != "true":
            continue
        if row.get("node_type") != "feature":
            continue
        out.add((row.get("layer", ""), row.get("pos", ""), row.get("feature_or_token_id", "")))
    return out


def _sample_score(row: dict[str, str], filtered_overlap: float) -> float:
    feature_down = max(0.0, -_safe_float(row.get("delta_target_feature_ratio")))
    token_up = max(0.0, _safe_float(row.get("delta_target_token_ratio")))
    error_up = max(0.0, _safe_float(row.get("delta_target_error_ratio")))
    overlap_term = 1.0 - filtered_overlap if not math.isnan(filtered_overlap) else 0.0
    return 0.25 * overlap_term + 2.0 * feature_down + token_up + 0.5 * error_up


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a focused Stage 1 intervention plan from completed compare outputs.")
    parser.add_argument("--run-tag-base", required=True)
    parser.add_argument("--outputs-root", default="outputs/phase_ab/ab_answer_aligned")
    parser.add_argument("--summary-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--generic-nodes-csv", default="")
    parser.add_argument("--same-target-only", action="store_true")
    parser.add_argument("--max-samples-per-bucket", type=int, default=2)
    parser.add_argument("--top-features-per-run", type=int, default=2)
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root).expanduser().resolve()
    summary_dir = Path(args.summary_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    generic_features = _load_generic_features(
        Path(args.generic_nodes_csv).expanduser().resolve() if args.generic_nodes_csv else None
    )
    combined_sample_rows = _read_csv(summary_dir / "stage1_combined_samples.csv")
    combined_by_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in combined_sample_rows:
        combined_by_bucket[row.get("bucket", "")].append(row)

    filtered_overlap_rows = _read_csv(summary_dir / "stage1_generic_filtered_overlap.csv")
    filtered_overlap_map = {
        (row["bucket"], row["sample_id"]): row
        for row in filtered_overlap_rows
    }

    candidate_sample_rows: list[dict[str, str]] = []
    candidate_feature_rows: list[dict[str, str]] = []
    sample_ids_dir = out_dir / "bucket_sample_ids"
    sample_ids_dir.mkdir(parents=True, exist_ok=True)

    for bucket in BUCKETS:
        run_tag = f"{args.run_tag_base}_{bucket}"
        compare_dir = outputs_root / run_tag / run_tag
        sample_compare_path = compare_dir / "sample_compare_controlled.csv"
        nodes_path = compare_dir / "nodes_detailed_controlled.csv"
        sample_rows = _read_csv(sample_compare_path) if sample_compare_path.exists() else combined_by_bucket.get(bucket, [])
        nodes_rows = _read_csv(nodes_path) if nodes_path.exists() else []
        if not sample_rows:
            continue

        scored_rows = []
        for row in sample_rows:
            sample_id = row.get("sample_id", "")
            overlap_row = filtered_overlap_map.get((bucket, sample_id), {})
            same_target = overlap_row.get("same_target_token", "")
            if args.same_target_only and (same_target != "True" or not _same_nonempty_target(row, overlap_row)):
                continue
            filtered_overlap = _safe_float(overlap_row.get("generic_filtered_node_overlap_jaccard"))
            score = _sample_score(row, filtered_overlap)
            scored_rows.append((score, row, overlap_row))

        scored_rows.sort(key=lambda item: item[0], reverse=True)
        chosen = scored_rows[: args.max_samples_per_bucket]

        bucket_sample_rows = []
        for rank, (score, row, overlap_row) in enumerate(chosen, start=1):
            sample_id = row.get("sample_id", "")
            bucket_sample_rows.append({"sample_id": sample_id})
            candidate_sample_rows.append(
                {
                    "bucket": bucket,
                    "rank": str(rank),
                    "sample_id": sample_id,
                    "intervention_priority_score": _fmt(score),
                    "same_target_token": overlap_row.get("same_target_token", ""),
                    "filtered_node_overlap_jaccard": overlap_row.get("generic_filtered_node_overlap_jaccard", ""),
                    "raw_node_overlap_jaccard": row.get("node_overlap_jaccard", ""),
                    "edge_overlap_jaccard": row.get("edge_overlap_jaccard", ""),
                    "delta_target_error_ratio": row.get("delta_target_error_ratio", ""),
                    "delta_target_feature_ratio": row.get("delta_target_feature_ratio", ""),
                    "delta_target_token_ratio": row.get("delta_target_token_ratio", ""),
                }
            )

            per_run_counts: dict[str, int] = defaultdict(int)
            sample_feature_rows = [
                n for n in nodes_rows
                if n.get("sample_id") == sample_id and n.get("node_type") == "feature"
            ]
            sample_feature_rows.sort(key=lambda n: _safe_float(n.get("path_mass_best")), reverse=True)
            for node in sample_feature_rows:
                run = node.get("run", "")
                if per_run_counts[run] >= args.top_features_per_run:
                    continue
                key = (node.get("layer", ""), node.get("pos", ""), node.get("feature_id", ""))
                if key in generic_features:
                    continue
                candidate_feature_rows.append(
                    {
                        "bucket": bucket,
                        "sample_id": sample_id,
                        "run": run,
                        "feature_rank_within_run": str(per_run_counts[run] + 1),
                        "layer": node.get("layer", ""),
                        "pos": node.get("pos", ""),
                        "feature_id": node.get("feature_id", ""),
                        "path_mass_best": node.get("path_mass_best", ""),
                        "depth_from_target": node.get("depth_from_target", ""),
                    }
                )
                per_run_counts[run] += 1

        if bucket_sample_rows:
            _write_csv(sample_ids_dir / f"{bucket}.csv", bucket_sample_rows, ["sample_id"])

    _write_csv(
        out_dir / "candidate_samples.csv",
        candidate_sample_rows,
        [
            "bucket",
            "rank",
            "sample_id",
            "intervention_priority_score",
            "same_target_token",
            "filtered_node_overlap_jaccard",
            "raw_node_overlap_jaccard",
            "edge_overlap_jaccard",
            "delta_target_error_ratio",
            "delta_target_feature_ratio",
            "delta_target_token_ratio",
        ],
    )
    _write_csv(
        out_dir / "candidate_features.csv",
        candidate_feature_rows,
        [
            "bucket",
            "sample_id",
            "run",
            "feature_rank_within_run",
            "layer",
            "pos",
            "feature_id",
            "path_mass_best",
            "depth_from_target",
        ],
    )

    lines = [
        "# Stage 1 Intervention Plan",
        "",
        f"- run tag base: `{args.run_tag_base}`",
        f"- same target only: `{args.same_target_only}`",
        f"- max samples per bucket: `{args.max_samples_per_bucket}`",
        f"- top features per run: `{args.top_features_per_run}`",
        "",
        "## Recommended Samples",
        "",
        "| bucket | rank | sample_id | score | same target | filtered overlap | delta feature | delta token | delta error |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in candidate_sample_rows:
        lines.append(
            "| {bucket} | {rank} | {sample_id} | {intervention_priority_score} | {same_target_token} | {filtered_node_overlap_jaccard} | {delta_target_feature_ratio} | {delta_target_token_ratio} | {delta_target_error_ratio} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Recommended Feature Nodes",
            "",
            "| bucket | sample_id | run | rank | feature | path mass | depth |",
            "|---|---|---|---:|---|---:|---:|",
        ]
    )
    for row in candidate_feature_rows:
        lines.append(
            "| {bucket} | {sample_id} | {run} | {feature_rank_within_run} | L{layer}:P{pos}:F{feature_id} | {path_mass_best} | {depth_from_target} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Suggested Server Run",
            "",
            "```bash",
            f"RUN_TAG_BASE={args.run_tag_base} \\",
            f"SAMPLE_IDS_DIR={sample_ids_dir.as_posix()} \\",
            f"GENERIC_NODES_CSV={Path(args.generic_nodes_csv).expanduser().resolve().as_posix() if args.generic_nodes_csv else '<generic_nodes_csv>'} \\",
            "REQUIRE_SAME_TARGET=1 \\",
            f"TOP_FEATURES_PER_SAMPLE={args.top_features_per_run} \\",
            f"MAX_SAMPLES={args.max_samples_per_bucket} \\",
            "bash scripts/server/run_stage1_intervention_smoke.sh",
            "```",
        ]
    )

    (out_dir / "intervention_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
