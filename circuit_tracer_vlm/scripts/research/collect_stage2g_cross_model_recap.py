#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarize_rows(
    *,
    model: str,
    evidence_level: str,
    rows: list[dict[str, str]],
    bucket_names: set[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("bucket") not in bucket_names:
            continue
        grouped[(row.get("layer", ""), row.get("condition", ""), row.get("bucket", ""))].append(row)

    out = []
    for (layer, condition, bucket), group in sorted(
        grouped.items(),
        key=lambda item: (int(item[0][0]) if str(item[0][0]).isdigit() else -1, item[0][1], item[0][2]),
    ):
        drops = [float(row["mean_topk_drop"]) for row in group]
        by_case: dict[str, list[float]] = defaultdict(list)
        by_prompt: dict[str, list[float]] = defaultdict(list)
        for row in group:
            by_case[row["sample_id"]].append(float(row["mean_topk_drop"]))
            by_prompt[row["prompt_name"]].append(float(row["mean_topk_drop"]))
        case_means = {case: _mean(vals) for case, vals in by_case.items()}
        prompt_means = {prompt: _mean(vals) for prompt, vals in by_prompt.items()}
        positives = sum(1 for value in drops if value > 0)
        consistency = positives / len(drops) if drops else 0.0
        prompt_delta = ""
        if {"B_direct", "D_visual_only"}.issubset(prompt_means):
            prompt_delta = prompt_means["D_visual_only"] - prompt_means["B_direct"]
        if consistency == 1.0 and _mean(drops) > 0:
            readout_status = "stable_positive"
        elif _mean(drops) > 0:
            readout_status = "positive_heterogeneous"
        elif _mean(drops) < 0:
            readout_status = "negative_or_reversal"
        else:
            readout_status = "weak_or_zero"
        out.append(
            {
                "model": model,
                "layer": layer,
                "bucket": bucket,
                "condition": condition,
                "n_rows": len(group),
                "mean_topk_drop": round(_mean(drops), 6),
                "min_topk_drop": round(min(drops), 6),
                "max_topk_drop": round(max(drops), 6),
                "positive_fraction": round(consistency, 6),
                "case_mean_json": json.dumps(case_means, ensure_ascii=False, sort_keys=True),
                "prompt_mean_json": json.dumps(prompt_means, ensure_ascii=False, sort_keys=True),
                "prompt_delta_D_minus_B": round(prompt_delta, 6) if prompt_delta != "" else "",
                "readout_status": readout_status,
                "evidence_level": evidence_level,
                "has_intervention": "no",
                "has_control": "readout_only_control_not_available",
                "causal_route_status": "not_tested",
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Stage 2G cross-model recap tables.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    base = Path(args.artifact_dir)
    rows: list[dict[str, Any]] = []
    qwen_path = base / "stage2f_qwen_clean_vs_mask_feature_readout_3case_summary.csv"
    if qwen_path.exists():
        rows.extend(
            _summarize_rows(
                model="Qwen2.5-VL-7B",
                evidence_level="readout_replication_only",
                rows=_read_csv(qwen_path),
                bucket_names={"image_marker_or_span"},
            )
        )
    llava0_path = base / "stage2f_llava_clean_vs_mask_feature_readout_3case_summary.csv"
    if llava0_path.exists():
        rows.extend(
            _summarize_rows(
                model="LLaVA-1.5-7B",
                evidence_level="readout_replication_only",
                rows=_read_csv(llava0_path),
                bucket_names={"image_token_span"},
            )
        )
    llava_high_path = base / "stage2f_llava_clean_vs_mask_feature_readout_3case_layers15_30_summary.csv"
    if llava_high_path.exists():
        rows.extend(
            _summarize_rows(
                model="LLaVA-1.5-7B",
                evidence_level="readout_replication_only",
                rows=_read_csv(llava_high_path),
                bucket_names={"image_token_span"},
            )
        )

    rows.append(
        {
            "model": "Gemma3-4B-IT",
            "layer": "multiple",
            "bucket": "answer_adjacent_source_routes",
            "condition": "wrong_image_and_region_mask",
            "n_rows": "",
            "mean_topk_drop": "",
            "min_topk_drop": "",
            "max_topk_drop": "",
            "positive_fraction": "",
            "case_mean_json": "",
            "prompt_mean_json": "",
            "prompt_delta_D_minus_B": "",
            "readout_status": "mainline_success",
            "evidence_level": "causal_route_mainline",
            "has_intervention": "yes_node_zeroing",
            "has_control": "yes_nearest_and_random_controls",
            "causal_route_status": "supported_for_localized_strong_image_dependence_cases",
        }
    )

    fieldnames = [
        "model",
        "layer",
        "bucket",
        "condition",
        "n_rows",
        "mean_topk_drop",
        "min_topk_drop",
        "max_topk_drop",
        "positive_fraction",
        "case_mean_json",
        "prompt_mean_json",
        "prompt_delta_D_minus_B",
        "readout_status",
        "evidence_level",
        "has_intervention",
        "has_control",
        "causal_route_status",
    ]
    _write_csv(Path(args.out_csv), rows, fieldnames)

    md_lines = [
        "# Stage 2G Cross-Model Recap",
        "",
        "This recap deliberately separates readout-level evidence from causal-route evidence.",
        "",
        "| model | layer | condition | mean drop | positive fraction | status | evidence level | causal status |",
        "|---|---:|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        md_lines.append(
            "| {model} | {layer} | {condition} | {mean_topk_drop} | {positive_fraction} | {readout_status} | {evidence_level} | {causal_route_status} |".format(
                **row
            )
        )
    md_lines.extend(
        [
            "",
            "Conservative reading:",
            "",
            "- Qwen and LLaVA show evidence-region-sensitive feature readouts.",
            "- Gemma3 remains the only model with the full source/control causal route chain.",
            "- Cross-model causal route replication is not established until feature intervention and control tests pass.",
        ]
    )
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
