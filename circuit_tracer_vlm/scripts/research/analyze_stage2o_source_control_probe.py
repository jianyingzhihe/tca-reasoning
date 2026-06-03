#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
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


def _float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _mean(values: list[float]) -> float | str:
    if not values:
        return ""
    return round(mean(values), 6)


def _bootstrap(values: list[float], seed: int = 1729, n_boot: int = 2000) -> tuple[float | str, float | str, str]:
    if not values:
        return "", "", "missing"
    if len(values) == 1:
        value = round(values[0], 6)
        return value, value, "single_row_positive" if values[0] > 0 else "not_positive"
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _idx in range(len(values))]
        means.append(mean(sample))
    means.sort()
    low = means[int(0.025 * (len(means) - 1))]
    high = means[int(0.975 * (len(means) - 1))]
    status = "stable_positive" if low > 0 else ("weak_or_heterogeneous_positive" if mean(values) > 0 else "not_positive")
    return round(low, 6), round(high, 6), status


def _group(rows: list[dict[str, str]], keys: list[str]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    out: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[tuple(row.get(key, "") for key in keys)].append(row)
    return out


def _summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = ["model_family", "mask_condition", "mask_variant", "intervention", "feature_role"]
    for key, items in sorted(_group(rows, keys).items()):
        model, mask_condition, mask_variant, intervention, role = key
        effects = [value for value in (_float(row.get("effect_logit")) for row in items) if value is not None]
        ranks = [value for value in (_float(row.get("effect_rank")) for row in items) if value is not None]
        correct_wrong = [value for value in (_float(row.get("correct_minus_wrong_logit")) for row in items) if value is not None]
        ci_low, ci_high, status = _bootstrap(effects)
        out.append(
            {
                "model_family": model,
                "mask_condition": mask_condition,
                "mask_variant": mask_variant,
                "intervention": intervention,
                "feature_role": role,
                "n_rows": len(items),
                "positive_logit_n": sum(1 for value in effects if value > 0),
                "positive_rank_n": sum(1 for value in ranks if value > 0),
                "mean_effect_logit": _mean(effects),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "effect_status": status,
                "mean_effect_rank": _mean(ranks),
                "mean_gap_closure": _mean([value for value in (_float(row.get("gap_closure")) for row in items) if value is not None]),
                "mean_correct_minus_wrong_logit": _mean(correct_wrong),
            }
        )
    return out


def _specificity(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    grouped = _group(rows, ["model_family", "sample_id", "prompt_name", "mask_condition", "mask_variant", "intervention"])
    for key, items in sorted(grouped.items()):
        model, sample_id, prompt_name, mask_condition, mask_variant, intervention = key
        by_role = {row.get("feature_role", ""): row for row in items}
        source = by_role.get("source")
        control = by_role.get("matched_control")
        if source is None or control is None:
            continue
        source_effect = _float(source.get("effect_logit"))
        control_effect = _float(control.get("effect_logit"))
        source_rank = _float(source.get("effect_rank"))
        control_rank = _float(control.get("effect_rank"))
        out.append(
            {
                "model_family": model,
                "sample_id": sample_id,
                "prompt_name": prompt_name,
                "mask_condition": mask_condition,
                "mask_variant": mask_variant,
                "intervention": intervention,
                "source_feature_id": source.get("feature_id", ""),
                "control_feature_id": control.get("feature_id", ""),
                "control_group": control.get("feature_group", ""),
                "source_effect_logit": source_effect,
                "control_effect_logit": control_effect,
                "source_minus_control_logit": "" if source_effect is None or control_effect is None else source_effect - control_effect,
                "source_effect_rank": source_rank,
                "control_effect_rank": control_rank,
                "source_minus_control_rank": "" if source_rank is None or control_rank is None else source_rank - control_rank,
                "source_correct_minus_wrong_logit": source.get("correct_minus_wrong_logit", ""),
                "control_correct_minus_wrong_logit": control.get("correct_minus_wrong_logit", ""),
            }
        )
    return out


def _mask_specificity(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    grouped = _group(rows, ["model_family", "sample_id", "prompt_name", "mask_condition", "feature_role", "intervention"])
    for key, items in sorted(grouped.items()):
        model, sample_id, prompt_name, mask_condition, role, intervention = key
        if intervention != "restore":
            continue
        by_variant = {row.get("mask_variant", ""): row for row in items}
        real = by_variant.get("real_mask")
        shuffled = by_variant.get("mask_shuffled")
        if real is None or shuffled is None:
            continue
        real_effect = _float(real.get("effect_logit"))
        shuffled_effect = _float(shuffled.get("effect_logit"))
        out.append(
            {
                "model_family": model,
                "sample_id": sample_id,
                "prompt_name": prompt_name,
                "mask_condition": mask_condition,
                "feature_role": role,
                "real_effect_logit": real_effect,
                "shuffled_effect_logit": shuffled_effect,
                "real_minus_shuffled_logit": "" if real_effect is None or shuffled_effect is None else real_effect - shuffled_effect,
            }
        )
    return out


def _aggregate_gap(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [value for value in (_float(row.get(key)) for row in rows) if value is not None]
    ci_low, ci_high, status = _bootstrap(values)
    return {
        "n": len(values),
        "positive_n": sum(1 for value in values if value > 0),
        "mean": _mean(values),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "status": status,
    }


def _decision(summary_rows: list[dict[str, Any]], specificity_rows: list[dict[str, Any]], mask_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "claim_boundary": (
            "Stage 2O Experiment 2 is an approximate source-control route probe. It can support source-control-like "
            "evidence, but not full Gemma-style route replication without source tracing adapter."
        ),
        "model_decisions": {},
    }
    for model in sorted({row["model_family"] for row in summary_rows}):
        real_spec = [
            row
            for row in specificity_rows
            if row["model_family"] == model and row["mask_variant"] == "real_mask"
        ]
        restore_spec = [row for row in real_spec if row["intervention"] == "restore"]
        zero_spec = [row for row in real_spec if row["intervention"] == "zeroing"]
        mask_spec = [row for row in mask_rows if row["model_family"] == model and row["feature_role"] == "source"]
        source_control_restore = _aggregate_gap(restore_spec, "source_minus_control_logit")
        source_control_zero = _aggregate_gap(zero_spec, "source_minus_control_logit")
        real_shuffled = _aggregate_gap(mask_spec, "real_minus_shuffled_logit")
        correct_wrong = _aggregate_gap(
            [row for row in summary_rows if row["model_family"] == model and row["mask_variant"] == "real_mask" and row["feature_role"] == "source"],
            "mean_correct_minus_wrong_logit",
        )
        supported_parts = sum(
            1
            for item in [source_control_restore, source_control_zero, real_shuffled, correct_wrong]
            if item["status"] == "stable_positive"
        )
        if source_control_restore["status"] == "stable_positive" and source_control_zero["status"] == "stable_positive" and real_shuffled["status"] == "stable_positive":
            status = "approx_source_control_route_supported"
        elif supported_parts >= 2:
            status = "source_control_probe_partial"
        else:
            status = "source_control_route_not_established"
        out["model_decisions"][model] = {
            "status": status,
            "source_control_restore": source_control_restore,
            "source_control_zeroing": source_control_zero,
            "real_minus_shuffled": real_shuffled,
            "correct_minus_wrong": correct_wrong,
        }
    if not out["model_decisions"]:
        out["overall_status"] = "blocked_no_rows"
    elif any(item["status"] == "approx_source_control_route_supported" for item in out["model_decisions"].values()):
        out["overall_status"] = "model_specific_approx_source_control_support"
    elif any(item["status"] == "source_control_probe_partial" for item in out["model_decisions"].values()):
        out["overall_status"] = "source_control_probe_partial"
    else:
        out["overall_status"] = "source_control_route_not_established"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage 2O source-control probe.")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--out-specificity", required=True)
    parser.add_argument("--out-mask-specificity", required=True)
    parser.add_argument("--out-decision", required=True)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for raw in args.inputs.split(","):
        rows.extend(_read_csv(Path(raw.strip())))
    summary_rows = _summary(rows)
    specificity_rows = _specificity(rows)
    mask_rows = _mask_specificity(rows)
    decision = _decision(summary_rows, specificity_rows, mask_rows)

    _write_csv(
        Path(args.out_summary),
        summary_rows,
        [
            "model_family",
            "mask_condition",
            "mask_variant",
            "intervention",
            "feature_role",
            "n_rows",
            "positive_logit_n",
            "positive_rank_n",
            "mean_effect_logit",
            "ci95_low",
            "ci95_high",
            "effect_status",
            "mean_effect_rank",
            "mean_gap_closure",
            "mean_correct_minus_wrong_logit",
        ],
    )
    _write_csv(
        Path(args.out_specificity),
        specificity_rows,
        [
            "model_family",
            "sample_id",
            "prompt_name",
            "mask_condition",
            "mask_variant",
            "intervention",
            "source_feature_id",
            "control_feature_id",
            "control_group",
            "source_effect_logit",
            "control_effect_logit",
            "source_minus_control_logit",
            "source_effect_rank",
            "control_effect_rank",
            "source_minus_control_rank",
            "source_correct_minus_wrong_logit",
            "control_correct_minus_wrong_logit",
        ],
    )
    _write_csv(
        Path(args.out_mask_specificity),
        mask_rows,
        [
            "model_family",
            "sample_id",
            "prompt_name",
            "mask_condition",
            "feature_role",
            "real_effect_logit",
            "shuffled_effect_logit",
            "real_minus_shuffled_logit",
        ],
    )
    _write_json(Path(args.out_decision), decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
