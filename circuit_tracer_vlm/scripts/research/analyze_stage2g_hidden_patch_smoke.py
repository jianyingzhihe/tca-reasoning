#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


CONDITION_RE = re.compile(r"^hidden_bucket_restore_s(\d+)p(\d+)$")


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _mean(values: list[float]) -> float:
    vals = [value for value in values if not math.isnan(value)]
    return sum(vals) / len(vals) if vals else float("nan")


def _round(value: float) -> float | str:
    return "" if math.isnan(value) else round(value, 6)


def _scale(condition: str) -> float | str:
    match = CONDITION_RE.match(condition)
    if not match:
        return ""
    whole, frac = match.groups()
    return float(f"{whole}.{frac}")


def _prepare(model: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for raw in rows:
        row: dict[str, Any] = {
            **raw,
            "model_family": raw.get("model_family") or model,
            "scale": _scale(raw["condition"]),
            "logit_restore": _float(raw.get("logit_restore_vs_union", "nan")),
            "rank_restore": _float(raw.get("rank_restore_vs_union", "nan")),
            "gap_closure": _float(raw.get("logit_gap_closure", "nan")),
            "clean_union_logit_gap_f": _float(raw.get("clean_union_logit_gap", "nan")),
            "clean_union_rank_gap_f": _float(raw.get("clean_union_rank_gap", "nan")),
            "top1_changed_bool": str(raw.get("top1_changed_vs_union", "")).lower() == "true",
        }
        out.append(row)
    return out


def _baseline_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["condition"] == "union_mask":
            grouped[(row["model_family"], row["bucket"])].append(row)
    out = []
    for (model, bucket), group in sorted(grouped.items()):
        logit_gaps = [row["clean_union_logit_gap_f"] for row in group]
        rank_gaps = [row["clean_union_rank_gap_f"] for row in group]
        out.append(
            {
                "model_family": model,
                "bucket": bucket,
                "n_runs": len(group),
                "mean_clean_union_logit_gap": _round(_mean(logit_gaps)),
                "positive_logit_gap_n": sum(value > 0 for value in logit_gaps if not math.isnan(value)),
                "mean_clean_union_rank_gap": _round(_mean(rank_gaps)),
                "positive_rank_gap_n": sum(value > 0 for value in rank_gaps if not math.isnan(value)),
                "top1_changed_clean_to_union_n": sum(
                    str(row.get("clean_top1_token", "")) != str(row.get("union_top1_token", "")) for row in group
                ),
            }
        )
    return out


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "hidden_bucket_restore" not in row["condition"]:
            continue
        grouped[(row["model_family"], row["layer"], row["bucket"], row["condition"], row["scale"])].append(row)
    out = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        model, layer, bucket, condition, scale = key
        logit_restore = [row["logit_restore"] for row in group]
        rank_restore = [row["rank_restore"] for row in group]
        gap_closure = [row["gap_closure"] for row in group]
        out.append(
            {
                "model_family": model,
                "layer": layer,
                "bucket": bucket,
                "condition": condition,
                "scale": scale,
                "n_rows": len(group),
                "n_samples": len({row["sample_id"] for row in group}),
                "n_prompts": len({row["prompt_name"] for row in group}),
                "mean_logit_restore": _round(_mean(logit_restore)),
                "positive_logit_restore_n": sum(value > 0 for value in logit_restore if not math.isnan(value)),
                "mean_rank_restore": _round(_mean(rank_restore)),
                "positive_rank_restore_n": sum(value > 0 for value in rank_restore if not math.isnan(value)),
                "mean_logit_gap_closure": _round(_mean(gap_closure)),
                "top1_changed_vs_union_n": sum(row["top1_changed_bool"] for row in group),
                "nan_logit_restore_n": sum(math.isnan(value) for value in logit_restore),
            }
        )
    return out


def _case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = {"clean", "union_mask", "hidden_bucket_restore_s1p0"}
    out = []
    for row in rows:
        if row["condition"] not in keep:
            continue
        out.append(
            {
                "model_family": row["model_family"],
                "sample_id": row["sample_id"],
                "prompt_name": row["prompt_name"],
                "condition": row["condition"],
                "target_answer": row["target_answer"],
                "target_rank": row["target_rank"],
                "target_logit": row["target_logit"],
                "clean_target_rank": row["clean_target_rank"],
                "union_target_rank": row["union_target_rank"],
                "clean_union_logit_gap": row["clean_union_logit_gap"],
                "clean_union_rank_gap": row["clean_union_rank_gap"],
                "logit_restore_vs_union": row["logit_restore_vs_union"],
                "rank_restore_vs_union": row["rank_restore_vs_union"],
                "logit_gap_closure": row["logit_gap_closure"],
                "top1_token": row["top1_token"],
                "clean_top1_token": row["clean_top1_token"],
                "union_top1_token": row["union_top1_token"],
            }
        )
    return sorted(out, key=lambda row: (row["model_family"], row["sample_id"], row["prompt_name"], row["condition"]))


def _decision(model: str, summary: list[dict[str, Any]]) -> dict[str, Any]:
    model_summary = [row for row in summary if row["model_family"] == model]
    primary = next((row for row in model_summary if float(row["scale"]) == 1.0), None)
    if (
        primary is not None
        and int(primary["positive_logit_restore_n"]) >= 4
        and int(primary["positive_rank_restore_n"]) >= 3
        and _float(str(primary["mean_logit_gap_closure"])) > 0.1
    ):
        status = "supported_hidden_state_upper_bound"
    elif primary is not None and int(primary["positive_logit_restore_n"]) >= 3:
        status = "partial_hidden_state_upper_bound"
    else:
        status = "not_supported_hidden_state_upper_bound"
    return {
        "model_family": model,
        "primary_condition": primary["condition"] if primary else "",
        "primary_mean_logit_restore": primary["mean_logit_restore"] if primary else "",
        "primary_positive_logit_restore_n": primary["positive_logit_restore_n"] if primary else "",
        "primary_mean_rank_restore": primary["mean_rank_restore"] if primary else "",
        "primary_positive_rank_restore_n": primary["positive_rank_restore_n"] if primary else "",
        "primary_mean_logit_gap_closure": primary["mean_logit_gap_closure"] if primary else "",
        "decision_status": status,
        "claim_boundary": (
            "Hidden-state patch is an upper bound for layer/bucket causality. It does not provide "
            "feature-level specificity or source-control route replication."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage 2G hidden-state clean patch upper-bound smoke.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--out-baseline-csv", required=True)
    parser.add_argument("--out-summary-csv", required=True)
    parser.add_argument("--out-case-csv", required=True)
    parser.add_argument("--out-decision-json", required=True)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    all_rows = []
    for model, filename in {
        "qwen": "stage2g_qwen_hidden_patch_smoke.csv",
        "llava": "stage2g_llava_hidden_patch_smoke.csv",
    }.items():
        path = artifact_dir / filename
        if path.exists():
            all_rows.extend(_prepare(model, _read_csv(path)))

    baseline = _baseline_summary(all_rows)
    summary = _summary(all_rows)
    cases = _case_rows(all_rows)
    decisions = {
        "evidence_ladder": {
            "readout_replication": "supported by Stage 2F for Qwen/LLaVA",
            "feature_level_restoration": "not established by Stage 2G-5",
            "hidden_state_upper_bound": "evaluated here",
            "source_control_causal_replication": "not established cross-model; Gemma3 remains the only full-chain model",
        },
        "model_decisions": {
            model: _decision(model, summary)
            for model in sorted({row["model_family"] for row in all_rows})
        },
    }

    baseline_fields = [
        "model_family",
        "bucket",
        "n_runs",
        "mean_clean_union_logit_gap",
        "positive_logit_gap_n",
        "mean_clean_union_rank_gap",
        "positive_rank_gap_n",
        "top1_changed_clean_to_union_n",
    ]
    summary_fields = [
        "model_family",
        "layer",
        "bucket",
        "condition",
        "scale",
        "n_rows",
        "n_samples",
        "n_prompts",
        "mean_logit_restore",
        "positive_logit_restore_n",
        "mean_rank_restore",
        "positive_rank_restore_n",
        "mean_logit_gap_closure",
        "top1_changed_vs_union_n",
        "nan_logit_restore_n",
    ]
    case_fields = [
        "model_family",
        "sample_id",
        "prompt_name",
        "condition",
        "target_answer",
        "target_rank",
        "target_logit",
        "clean_target_rank",
        "union_target_rank",
        "clean_union_logit_gap",
        "clean_union_rank_gap",
        "logit_restore_vs_union",
        "rank_restore_vs_union",
        "logit_gap_closure",
        "top1_token",
        "clean_top1_token",
        "union_top1_token",
    ]
    _write_csv(Path(args.out_baseline_csv), baseline, baseline_fields)
    _write_csv(Path(args.out_summary_csv), summary, summary_fields)
    _write_csv(Path(args.out_case_csv), cases, case_fields)
    _write_json(Path(args.out_decision_json), decisions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
