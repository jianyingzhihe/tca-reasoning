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


CONDITION_RE = re.compile(r"^(evidence|control)_(top1|topk)_restore_s(\d+)p(\d+)$")


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


def _condition_parts(condition: str) -> dict[str, Any]:
    if condition in {"clean", "union_mask"}:
        return {"feature_group": condition, "feature_width": "", "scale": ""}
    match = CONDITION_RE.match(condition)
    if not match:
        return {"feature_group": "unknown", "feature_width": "", "scale": ""}
    group, width, whole, frac = match.groups()
    return {"feature_group": group, "feature_width": width, "scale": float(f"{whole}.{frac}")}


def _prepare_rows(model: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for raw in rows:
        parts = _condition_parts(raw["condition"])
        row: dict[str, Any] = {
            **raw,
            "model_family": raw.get("model_family") or model,
            **parts,
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


def _condition_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "restore" not in row["condition"]:
            continue
        key = (
            row["model_family"],
            row["layer"],
            row["bucket"],
            row["condition"],
            row["feature_group"],
            row["feature_width"],
            row["scale"],
        )
        grouped[key].append(row)
    out = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        model, layer, bucket, condition, feature_group, feature_width, scale = key
        logit_restore = [row["logit_restore"] for row in group]
        rank_restore = [row["rank_restore"] for row in group]
        gap_closure = [row["gap_closure"] for row in group]
        out.append(
            {
                "model_family": model,
                "layer": layer,
                "bucket": bucket,
                "condition": condition,
                "feature_group": feature_group,
                "feature_width": feature_width,
                "scale": scale,
                "n_rows": len(group),
                "n_samples": len({row["sample_id"] for row in group}),
                "n_prompts": len({row["prompt_name"] for row in group}),
                "mean_logit_restore": _round(_mean(logit_restore)),
                "positive_logit_restore_n": sum(value > 0 for value in logit_restore if not math.isnan(value)),
                "positive_logit_restore_frac": _round(
                    sum(value > 0 for value in logit_restore if not math.isnan(value))
                    / max(1, sum(not math.isnan(value) for value in logit_restore))
                ),
                "mean_rank_restore": _round(_mean(rank_restore)),
                "positive_rank_restore_n": sum(value > 0 for value in rank_restore if not math.isnan(value)),
                "positive_rank_restore_frac": _round(
                    sum(value > 0 for value in rank_restore if not math.isnan(value))
                    / max(1, sum(not math.isnan(value) for value in rank_restore))
                ),
                "mean_logit_gap_closure": _round(_mean(gap_closure)),
                "top1_changed_vs_union_n": sum(row["top1_changed_bool"] for row in group),
                "top1_changed_vs_union_frac": _round(sum(row["top1_changed_bool"] for row in group) / len(group)),
            }
        )
    return out


def _specificity_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (
            row["model_family"],
            row["layer"],
            row["bucket"],
            row["feature_width"],
            row["scale"],
            row["feature_group"],
        ): row
        for row in summary
    }
    out = []
    for key, evidence in by_key.items():
        model, layer, bucket, width, scale, group = key
        if group != "evidence" or width != "topk":
            continue
        control = by_key.get((model, layer, bucket, "topk", scale, "control"))
        if control is None:
            continue
        evidence_logit = _float(str(evidence["mean_logit_restore"]))
        control_logit = _float(str(control["mean_logit_restore"]))
        evidence_rank = _float(str(evidence["mean_rank_restore"]))
        control_rank = _float(str(control["mean_rank_restore"]))
        evidence_closure = _float(str(evidence["mean_logit_gap_closure"]))
        control_closure = _float(str(control["mean_logit_gap_closure"]))
        out.append(
            {
                "model_family": model,
                "layer": layer,
                "bucket": bucket,
                "scale": scale,
                "evidence_condition": evidence["condition"],
                "control_condition": control["condition"],
                "evidence_mean_logit_restore": evidence["mean_logit_restore"],
                "control_mean_logit_restore": control["mean_logit_restore"],
                "evidence_minus_control_logit_restore": _round(evidence_logit - control_logit),
                "evidence_mean_rank_restore": evidence["mean_rank_restore"],
                "control_mean_rank_restore": control["mean_rank_restore"],
                "evidence_minus_control_rank_restore": _round(evidence_rank - control_rank),
                "evidence_mean_logit_gap_closure": evidence["mean_logit_gap_closure"],
                "control_mean_logit_gap_closure": control["mean_logit_gap_closure"],
                "evidence_minus_control_gap_closure": _round(evidence_closure - control_closure),
                "evidence_logit_positive_n": evidence["positive_logit_restore_n"],
                "control_logit_positive_n": control["positive_logit_restore_n"],
                "evidence_rank_positive_n": evidence["positive_rank_restore_n"],
                "control_rank_positive_n": control["positive_rank_restore_n"],
            }
        )
    return sorted(out, key=lambda row: (row["model_family"], float(row["scale"])))


def _case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = {
        "clean",
        "union_mask",
        "evidence_topk_restore_s1p0",
        "evidence_topk_restore_s1p5",
        "control_topk_restore_s1p0",
        "control_topk_restore_s1p5",
    }
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


def _decision(model: str, summary: list[dict[str, Any]], specificity: list[dict[str, Any]]) -> dict[str, Any]:
    model_summary = [row for row in summary if row["model_family"] == model]
    model_specificity = [row for row in specificity if row["model_family"] == model]
    candidates = [
        row
        for row in model_summary
        if row["feature_group"] == "evidence"
        and row["feature_width"] == "topk"
        and float(row["scale"]) in {1.0, 1.5}
    ]
    best = max(candidates, key=lambda row: (_float(str(row["mean_logit_restore"])), _float(str(row["mean_rank_restore"]))), default=None)
    best_spec = None
    if best is not None:
        best_spec = next(
            (
                row
                for row in model_specificity
                if float(row["scale"]) == float(best["scale"])
            ),
            None,
        )
    if (
        best is not None
        and int(best["positive_logit_restore_n"]) >= 4
        and int(best["positive_rank_restore_n"]) >= 2
        and best_spec is not None
        and _float(str(best_spec["evidence_minus_control_logit_restore"])) > 0
        and _float(str(best_spec["evidence_minus_control_rank_restore"])) > 0
    ):
        status = "supported_feature_level_restoration"
    elif (
        best is not None
        and int(best["positive_logit_restore_n"]) >= 3
        and best_spec is not None
        and _float(str(best_spec["evidence_minus_control_logit_restore"])) > 0
    ):
        status = "partial_logit_restoration_only"
    else:
        status = "not_supported_feature_level_restoration"
    return {
        "model_family": model,
        "best_evidence_condition": best["condition"] if best else "",
        "best_mean_logit_restore": best["mean_logit_restore"] if best else "",
        "best_mean_rank_restore": best["mean_rank_restore"] if best else "",
        "best_positive_logit_restore_n": best["positive_logit_restore_n"] if best else "",
        "best_positive_rank_restore_n": best["positive_rank_restore_n"] if best else "",
        "best_mean_logit_gap_closure": best["mean_logit_gap_closure"] if best else "",
        "best_specificity_logit": best_spec["evidence_minus_control_logit_restore"] if best_spec else "",
        "best_specificity_rank": best_spec["evidence_minus_control_rank_restore"] if best_spec else "",
        "decision_status": status,
        "claim_boundary": (
            "No source-control causal route replication unless evidence restoration clearly beats control "
            "on rank/logit restoration."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage 2G mask-to-clean feature restoration smoke.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--out-baseline-csv", required=True)
    parser.add_argument("--out-summary-csv", required=True)
    parser.add_argument("--out-specificity-csv", required=True)
    parser.add_argument("--out-case-csv", required=True)
    parser.add_argument("--out-decision-json", required=True)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    model_files = {
        "qwen": artifact_dir / "stage2g_qwen_feature_restoration_smoke.csv",
        "llava": artifact_dir / "stage2g_llava_feature_restoration_smoke.csv",
    }
    all_rows = []
    for model, path in model_files.items():
        if path.exists():
            all_rows.extend(_prepare_rows(model, _read_csv(path)))

    baseline = _baseline_summary(all_rows)
    summary = _condition_summary(all_rows)
    specificity = _specificity_rows(summary)
    cases = _case_rows(all_rows)
    decisions = {
        "evidence_ladder": {
            "readout_replication": "supported by Stage 2F for Qwen/LLaVA",
            "feature_direction_intervention": "not established by Stage 2G dose/sign probe",
            "mask_to_clean_restoration": "evaluated here",
            "source_control_causal_replication": "not established cross-model; Gemma3 remains the only full-chain model",
        },
        "model_decisions": {
            model: _decision(model, summary, specificity)
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
        "feature_group",
        "feature_width",
        "scale",
        "n_rows",
        "n_samples",
        "n_prompts",
        "mean_logit_restore",
        "positive_logit_restore_n",
        "positive_logit_restore_frac",
        "mean_rank_restore",
        "positive_rank_restore_n",
        "positive_rank_restore_frac",
        "mean_logit_gap_closure",
        "top1_changed_vs_union_n",
        "top1_changed_vs_union_frac",
    ]
    specificity_fields = [
        "model_family",
        "layer",
        "bucket",
        "scale",
        "evidence_condition",
        "control_condition",
        "evidence_mean_logit_restore",
        "control_mean_logit_restore",
        "evidence_minus_control_logit_restore",
        "evidence_mean_rank_restore",
        "control_mean_rank_restore",
        "evidence_minus_control_rank_restore",
        "evidence_mean_logit_gap_closure",
        "control_mean_logit_gap_closure",
        "evidence_minus_control_gap_closure",
        "evidence_logit_positive_n",
        "control_logit_positive_n",
        "evidence_rank_positive_n",
        "control_rank_positive_n",
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
    _write_csv(Path(args.out_specificity_csv), specificity, specificity_fields)
    _write_csv(Path(args.out_case_csv), cases, case_fields)
    _write_json(Path(args.out_decision_json), decisions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
