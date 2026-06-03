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


CONDITION_RE = re.compile(r"^(evidence|control)_(top1|topk)_(subtract|add)_s(\d+)p(\d+)$")


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


def _median(values: list[float]) -> float:
    vals = sorted(value for value in values if not math.isnan(value))
    if not vals:
        return float("nan")
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def _round(value: float) -> float | str:
    return "" if math.isnan(value) else round(value, 6)


def _condition_parts(condition: str) -> dict[str, Any]:
    if condition == "baseline":
        return {"feature_group": "baseline", "feature_width": "", "direction": "", "scale": ""}
    match = CONDITION_RE.match(condition)
    if not match:
        return {"feature_group": "unknown", "feature_width": "", "direction": "", "scale": ""}
    group, width, direction, whole, frac = match.groups()
    return {
        "feature_group": group,
        "feature_width": width,
        "direction": direction,
        "scale": float(f"{whole}.{frac}"),
    }


def _prepare_rows(model: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in rows:
        parts = _condition_parts(raw["condition"])
        delta_logit = _float(raw.get("delta_logit_vs_baseline", "nan"))
        rank_damage = _float(raw.get("rank_damage_vs_baseline", "nan"))
        row: dict[str, Any] = {
            **raw,
            "model_family": raw.get("model_family") or model,
            **parts,
            "delta_logit": delta_logit,
            "logit_damage": -delta_logit if not math.isnan(delta_logit) else float("nan"),
            "rank_damage": rank_damage,
            "top1_changed_bool": str(raw.get("top1_changed", "")).lower() == "true",
        }
        out.append(row)
    return out


def _condition_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["condition"] == "baseline":
            continue
        key = (
            row["model_family"],
            row["layer"],
            row["bucket"],
            row["feature_group"],
            row["feature_width"],
            row["direction"],
            row["scale"],
            row["condition"],
        )
        grouped[key].append(row)

    out: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        (
            model,
            layer,
            bucket,
            feature_group,
            feature_width,
            direction,
            scale,
            condition,
        ) = key
        rank_damage = [row["rank_damage"] for row in group]
        logit_damage = [row["logit_damage"] for row in group]
        delta_logit = [row["delta_logit"] for row in group]
        valid_logit = [value for value in logit_damage if not math.isnan(value)]
        out.append(
            {
                "model_family": model,
                "layer": layer,
                "bucket": bucket,
                "condition": condition,
                "feature_group": feature_group,
                "feature_width": feature_width,
                "direction": direction,
                "scale": scale,
                "n_rows": len(group),
                "n_samples": len({row["sample_id"] for row in group}),
                "n_prompts": len({row["prompt_name"] for row in group}),
                "mean_rank_damage": _round(_mean(rank_damage)),
                "median_rank_damage": _round(_median(rank_damage)),
                "rank_damage_positive_n": sum(value > 0 for value in rank_damage if not math.isnan(value)),
                "rank_damage_positive_frac": _round(
                    sum(value > 0 for value in rank_damage if not math.isnan(value))
                    / max(1, sum(not math.isnan(value) for value in rank_damage))
                ),
                "mean_delta_logit": _round(_mean(delta_logit)),
                "mean_logit_damage": _round(_mean(logit_damage)),
                "median_logit_damage": _round(_median(logit_damage)),
                "logit_damage_positive_n": sum(value > 0 for value in logit_damage if not math.isnan(value)),
                "logit_damage_positive_frac": _round(
                    sum(value > 0 for value in logit_damage if not math.isnan(value))
                    / max(1, len(valid_logit))
                ),
                "nan_logit_n": len(group) - len(valid_logit),
                "top1_changed_n": sum(row["top1_changed_bool"] for row in group),
                "top1_changed_frac": _round(sum(row["top1_changed_bool"] for row in group) / len(group)),
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
            row["direction"],
            row["scale"],
            row["feature_group"],
        ): row
        for row in summary
    }
    out: list[dict[str, Any]] = []
    for key, evidence in by_key.items():
        model, layer, bucket, width, direction, scale, group = key
        if group != "evidence" or width != "topk":
            continue
        control = by_key.get((model, layer, bucket, "topk", direction, scale, "control"))
        if control is None:
            continue
        evidence_rank = _float(str(evidence["mean_rank_damage"]))
        control_rank = _float(str(control["mean_rank_damage"]))
        evidence_logit = _float(str(evidence["mean_logit_damage"]))
        control_logit = _float(str(control["mean_logit_damage"]))
        out.append(
            {
                "model_family": model,
                "layer": layer,
                "bucket": bucket,
                "direction": direction,
                "scale": scale,
                "evidence_condition": evidence["condition"],
                "control_condition": control["condition"],
                "evidence_mean_rank_damage": evidence["mean_rank_damage"],
                "control_mean_rank_damage": control["mean_rank_damage"],
                "evidence_minus_control_rank_damage": _round(evidence_rank - control_rank),
                "evidence_mean_logit_damage": evidence["mean_logit_damage"],
                "control_mean_logit_damage": control["mean_logit_damage"],
                "evidence_minus_control_logit_damage": _round(evidence_logit - control_logit),
                "evidence_rank_positive_n": evidence["rank_damage_positive_n"],
                "control_rank_positive_n": control["rank_damage_positive_n"],
                "evidence_logit_positive_n": evidence["logit_damage_positive_n"],
                "control_logit_positive_n": control["logit_damage_positive_n"],
            }
        )
    return sorted(out, key=lambda row: (row["model_family"], row["direction"], float(row["scale"])))


def _case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = {
        "baseline",
        "evidence_topk_subtract_s1p0",
        "evidence_topk_subtract_s3p0",
        "evidence_topk_subtract_s5p0",
        "evidence_topk_add_s1p0",
        "evidence_topk_add_s3p0",
        "evidence_topk_add_s5p0",
        "control_topk_subtract_s5p0",
        "control_topk_add_s5p0",
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["condition"] not in keep:
            continue
        out.append(
            {
                "model_family": row["model_family"],
                "sample_id": row["sample_id"],
                "prompt_name": row["prompt_name"],
                "layer": row["layer"],
                "bucket": row["bucket"],
                "condition": row["condition"],
                "target_answer": row["target_answer"],
                "target_token": row["target_token"],
                "target_rank": row["target_rank"],
                "baseline_target_rank": row["baseline_target_rank"],
                "rank_damage_vs_baseline": row["rank_damage_vs_baseline"],
                "delta_logit_vs_baseline": row["delta_logit_vs_baseline"],
                "logit_damage": _round(row["logit_damage"]),
                "top1_token": row["top1_token"],
                "baseline_top1_token": row["baseline_top1_token"],
                "top1_changed": row["top1_changed"],
            }
        )
    return sorted(out, key=lambda row: (row["model_family"], row["sample_id"], row["prompt_name"], row["condition"]))


def _decision(model: str, rows: list[dict[str, Any]], summary: list[dict[str, Any]], specificity: list[dict[str, Any]]) -> dict[str, Any]:
    model_summary = [row for row in summary if row["model_family"] == model]
    model_specificity = [row for row in specificity if row["model_family"] == model]

    primary = next(
        (
            row
            for row in model_summary
            if row["feature_group"] == "evidence"
            and row["feature_width"] == "topk"
            and row["direction"] == "subtract"
            and float(row["scale"]) == 5.0
        ),
        None,
    )
    strongest_add = next(
        (
            row
            for row in model_summary
            if row["feature_group"] == "evidence"
            and row["feature_width"] == "topk"
            and row["direction"] == "add"
            and float(row["scale"]) == 5.0
        ),
        None,
    )
    add_spec = next(
        (
            row
            for row in model_specificity
            if row["direction"] == "add" and float(row["scale"]) == 5.0
        ),
        None,
    )
    subtract_spec = next(
        (
            row
            for row in model_specificity
            if row["direction"] == "subtract" and float(row["scale"]) == 5.0
        ),
        None,
    )

    usable_runs = len({(row["sample_id"], row["prompt_name"]) for row in rows if row["condition"] == "baseline"})
    samples = sorted({row["sample_id"] for row in rows})

    primary_rank_hits = int(primary["rank_damage_positive_n"]) if primary else 0
    primary_logit_hits = int(primary["logit_damage_positive_n"]) if primary else 0
    add_rank_hits = int(strongest_add["rank_damage_positive_n"]) if strongest_add else 0
    add_logit_hits = int(strongest_add["logit_damage_positive_n"]) if strongest_add else 0

    if primary and primary_rank_hits >= 2 and subtract_spec and _float(str(subtract_spec["evidence_minus_control_rank_damage"])) > 0:
        status = "supported_by_primary_ablation_rank"
    elif primary and primary_logit_hits >= 4 and subtract_spec and _float(str(subtract_spec["evidence_minus_control_logit_damage"])) > 0:
        status = "partial_primary_ablation_logit_only"
    elif strongest_add and (add_rank_hits >= 2 or add_logit_hits >= 4) and add_spec and _float(str(add_spec["evidence_minus_control_logit_damage"])) > 0:
        status = "partial_signed_high_dose_effect_not_ablation"
    else:
        status = "not_supported_intervention_replication"

    return {
        "model_family": model,
        "usable_runs": usable_runs,
        "samples": samples,
        "primary_condition": "evidence_topk_subtract_s5p0",
        "primary_rank_hits": primary_rank_hits,
        "primary_logit_hits": primary_logit_hits,
        "primary_mean_rank_damage": primary["mean_rank_damage"] if primary else "",
        "primary_mean_logit_damage": primary["mean_logit_damage"] if primary else "",
        "primary_specificity_rank": subtract_spec["evidence_minus_control_rank_damage"] if subtract_spec else "",
        "primary_specificity_logit": subtract_spec["evidence_minus_control_logit_damage"] if subtract_spec else "",
        "signed_add_condition": "evidence_topk_add_s5p0",
        "signed_add_rank_hits": add_rank_hits,
        "signed_add_logit_hits": add_logit_hits,
        "signed_add_mean_rank_damage": strongest_add["mean_rank_damage"] if strongest_add else "",
        "signed_add_mean_logit_damage": strongest_add["mean_logit_damage"] if strongest_add else "",
        "signed_add_specificity_rank": add_spec["evidence_minus_control_rank_damage"] if add_spec else "",
        "signed_add_specificity_logit": add_spec["evidence_minus_control_logit_damage"] if add_spec else "",
        "decision_status": status,
        "claim_boundary": (
            "Readout-level evidence sensitivity remains supported; this feature-direction probe "
            "does not establish source-control causal route replication."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage 2G feature intervention dose/sign probes.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--out-summary-csv", required=True)
    parser.add_argument("--out-specificity-csv", required=True)
    parser.add_argument("--out-case-csv", required=True)
    parser.add_argument("--out-decision-json", required=True)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    model_files = {
        "qwen": artifact_dir / "stage2g_qwen_feature_intervention_dose_probe.csv",
        "llava": artifact_dir / "stage2g_llava_feature_intervention_dose_probe.csv",
    }

    all_rows: list[dict[str, Any]] = []
    for model, path in model_files.items():
        if path.exists():
            all_rows.extend(_prepare_rows(model, _read_csv(path)))

    summary = _condition_summary(all_rows)
    specificity = _specificity_rows(summary)
    cases = _case_rows(all_rows)
    decisions = {
        "evidence_ladder": {
            "readout_replication": "supported by Stage 2F for Qwen/LLaVA",
            "intervention_replication": "evaluated here; only partial signed/high-dose responses, no primary ablation success",
            "source_control_causal_replication": "not established cross-model; Gemma3 remains the only full-chain model",
        },
        "model_decisions": {
            model: _decision(model, all_rows, summary, specificity)
            for model in sorted({row["model_family"] for row in all_rows})
        },
    }

    summary_fields = [
        "model_family",
        "layer",
        "bucket",
        "condition",
        "feature_group",
        "feature_width",
        "direction",
        "scale",
        "n_rows",
        "n_samples",
        "n_prompts",
        "mean_rank_damage",
        "median_rank_damage",
        "rank_damage_positive_n",
        "rank_damage_positive_frac",
        "mean_delta_logit",
        "mean_logit_damage",
        "median_logit_damage",
        "logit_damage_positive_n",
        "logit_damage_positive_frac",
        "nan_logit_n",
        "top1_changed_n",
        "top1_changed_frac",
    ]
    specificity_fields = [
        "model_family",
        "layer",
        "bucket",
        "direction",
        "scale",
        "evidence_condition",
        "control_condition",
        "evidence_mean_rank_damage",
        "control_mean_rank_damage",
        "evidence_minus_control_rank_damage",
        "evidence_mean_logit_damage",
        "control_mean_logit_damage",
        "evidence_minus_control_logit_damage",
        "evidence_rank_positive_n",
        "control_rank_positive_n",
        "evidence_logit_positive_n",
        "control_logit_positive_n",
    ]
    case_fields = [
        "model_family",
        "sample_id",
        "prompt_name",
        "layer",
        "bucket",
        "condition",
        "target_answer",
        "target_token",
        "target_rank",
        "baseline_target_rank",
        "rank_damage_vs_baseline",
        "delta_logit_vs_baseline",
        "logit_damage",
        "top1_token",
        "baseline_top1_token",
        "top1_changed",
    ]
    _write_csv(Path(args.out_summary_csv), summary, summary_fields)
    _write_csv(Path(args.out_specificity_csv), specificity, specificity_fields)
    _write_csv(Path(args.out_case_csv), cases, case_fields)
    _write_json(Path(args.out_decision_json), decisions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
