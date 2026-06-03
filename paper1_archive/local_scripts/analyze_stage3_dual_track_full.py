#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(r"E:\Bridging")
STAGE3_CROSS = ROOT / "doc" / "experiments" / "stage3" / "cross_model"

FEATURE_INPUTS = [
    ("qwen2p5vl_plt", STAGE3_CROSS / "stage3_qwen2p5vl_plt_feature_union.csv"),
    ("qwen2p5vl_clt", STAGE3_CROSS / "stage3_qwen2p5vl_clt_feature_union.csv"),
    ("llava15_clt", STAGE3_CROSS / "stage3_llava15_clt_feature_union.csv"),
]

SOURCE_INPUTS = [
    ("qwen2p5vl_plt", STAGE3_CROSS / "stage3_qwen2p5vl_plt_source_control.csv"),
    ("qwen2p5vl_clt", STAGE3_CROSS / "stage3_qwen2p5vl_clt_source_control.csv"),
    ("llava15_clt", STAGE3_CROSS / "stage3_llava15_clt_source_control.csv"),
]

CONTROL_GROUPS = {
    "activation_matched_topk",
    "drop_matched_topk",
    "attribution_matched_mask_insensitive_topk",
    "random_active_topk",
}


def _bootstrap_ci(values: list[float], *, seed: int = 1729, rounds: int = 2000) -> tuple[float | str, float | str]:
    if not values:
        return "", ""
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed + len(values))
    means = []
    for _ in range(rounds):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(mean(sample))
    means.sort()
    lo = means[int(0.025 * (rounds - 1))]
    hi = means[int(0.975 * (rounds - 1))]
    return lo, hi


def _stable_seed(*parts: str) -> int:
    value = 1729
    for part in parts:
        for char in part:
            value = (value * 131 + ord(char)) % (2**31)
    return value


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


def _float(value: str) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except Exception:
        return None


def _feature_effect(row: dict[str, str]) -> float | None:
    if row.get("direction") == "restore":
        return _float(row.get("logit_restore_vs_mask", ""))
    if row.get("direction") == "corrupt":
        return _float(row.get("logit_damage_vs_clean", ""))
    return None


def _feature_summary(asset_id: str, rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(dict)
    for row in rows:
        effect = _feature_effect(row)
        if effect is None:
            continue
        key = (row.get("sample_id", ""), row.get("prompt_name", ""), row.get("position_group", ""), row.get("direction", ""))
        group = row.get("feature_group", "")
        if group == "evidence_attribution_topk":
            by_key[key]["evidence"] = effect
        elif group in CONTROL_GROUPS:
            by_key[key].setdefault("controls", []).append(effect)

    comparisons: list[dict[str, Any]] = []
    for (sample_id, prompt_name, position_group, direction), values in sorted(by_key.items()):
        controls = values.get("controls", [])
        if "evidence" not in values or not controls:
            continue
        evidence = float(values["evidence"])
        control_mean = mean(controls)
        comparisons.append(
            {
                "asset_id": asset_id,
                "sample_id": sample_id,
                "prompt_name": prompt_name,
                "position_group": position_group,
                "direction": direction,
                "evidence_effect": evidence,
                "control_mean": control_mean,
                "evidence_minus_control": evidence - control_mean,
                "evidence_above_control": evidence > control_mean,
            }
        )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        grouped[(row["position_group"], row["direction"])].append(row)
    slices = []
    for (position_group, direction), items in sorted(grouped.items()):
        diffs = [float(item["evidence_minus_control"]) for item in items]
        ci_low, ci_high = _bootstrap_ci(diffs, seed=_stable_seed(asset_id, position_group, direction))
        slices.append(
            {
                "asset_id": asset_id,
                "position_group": position_group,
                "direction": direction,
                "comparison_count": len(items),
                "mean_evidence_effect": mean(float(item["evidence_effect"]) for item in items),
                "mean_control_effect": mean(float(item["control_mean"]) for item in items),
                "mean_evidence_minus_control": mean(diffs),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "positive_count": sum(diff > 0 for diff in diffs),
            }
        )
    return slices, comparisons


def _source_summary(asset_id: str, rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key: dict[tuple[str, str, str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (
            row.get("sample_id", ""),
            row.get("prompt_name", ""),
            row.get("mask_condition", ""),
            row.get("intervention", ""),
            row.get("mask_variant", ""),
        )
        by_key[key][row.get("feature_role", "")] = row

    comparisons: list[dict[str, Any]] = []
    for (sample_id, prompt_name, mask_condition, intervention, mask_variant), values in sorted(by_key.items()):
        source = values.get("source")
        control = values.get("matched_control")
        if source is None or control is None:
            continue
        source_effect = _float(source.get("effect_logit", ""))
        control_effect = _float(control.get("effect_logit", ""))
        source_correct_wrong = _float(source.get("correct_minus_wrong_logit", ""))
        control_correct_wrong = _float(control.get("correct_minus_wrong_logit", ""))
        if source_effect is None or control_effect is None:
            continue
        comparisons.append(
            {
                "asset_id": asset_id,
                "sample_id": sample_id,
                "prompt_name": prompt_name,
                "mask_condition": mask_condition,
                "intervention": intervention,
                "mask_variant": mask_variant,
                "source_effect": source_effect,
                "control_effect": control_effect,
                "source_minus_control": source_effect - control_effect,
                "source_above_control": source_effect > control_effect,
                "source_correct_minus_wrong": source_correct_wrong if source_correct_wrong is not None else "",
                "control_correct_minus_wrong": control_correct_wrong if control_correct_wrong is not None else "",
            }
        )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        grouped[(row["mask_condition"], row["intervention"], row["mask_variant"])].append(row)
    slices = []
    for (mask_condition, intervention, mask_variant), items in sorted(grouped.items()):
        diffs = [float(item["source_minus_control"]) for item in items]
        ci_low, ci_high = _bootstrap_ci(diffs, seed=_stable_seed(asset_id, mask_condition, intervention, mask_variant))
        source_effects = [float(item["source_effect"]) for item in items]
        control_effects = [float(item["control_effect"]) for item in items]
        correct_wrong = [
            float(item["source_correct_minus_wrong"])
            for item in items
            if item["source_correct_minus_wrong"] != ""
        ]
        slices.append(
            {
                "asset_id": asset_id,
                "mask_condition": mask_condition,
                "intervention": intervention,
                "mask_variant": mask_variant,
                "comparison_count": len(items),
                "mean_source_effect": mean(source_effects),
                "mean_control_effect": mean(control_effects),
                "mean_source_minus_control": mean(diffs),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "positive_count": sum(diff > 0 for diff in diffs),
                "mean_source_correct_minus_wrong": mean(correct_wrong) if correct_wrong else "",
                "correct_wrong_positive_count": sum(value > 0 for value in correct_wrong),
            }
        )
    return slices, comparisons


def main() -> int:
    feature_slices: list[dict[str, Any]] = []
    feature_comparisons: list[dict[str, Any]] = []
    source_slices: list[dict[str, Any]] = []
    source_comparisons: list[dict[str, Any]] = []
    status: list[dict[str, Any]] = []

    for asset_id, path in FEATURE_INPUTS:
        rows = _read_csv(path)
        status.append({"asset_id": asset_id, "kind": "feature", "path": str(path), "row_count": len(rows), "exists": path.exists()})
        slices, comparisons = _feature_summary(asset_id, rows)
        feature_slices.extend(slices)
        feature_comparisons.extend(comparisons)

    for asset_id, path in SOURCE_INPUTS:
        rows = _read_csv(path)
        status.append({"asset_id": asset_id, "kind": "source_control", "path": str(path), "row_count": len(rows), "exists": path.exists()})
        slices, comparisons = _source_summary(asset_id, rows)
        source_slices.extend(slices)
        source_comparisons.extend(comparisons)

    payload = {
        "claim_boundary": (
            "Stage3 full summary. Source-control here is an approximate cross-model probe, "
            "not full Gemma-style source tracing."
        ),
        "status": status,
    }
    (STAGE3_CROSS / "stage3_full_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(
        STAGE3_CROSS / "stage3_full_feature_summary_slices.csv",
        feature_slices,
        [
            "asset_id",
            "position_group",
            "direction",
            "comparison_count",
            "mean_evidence_effect",
            "mean_control_effect",
            "mean_evidence_minus_control",
            "ci95_low",
            "ci95_high",
            "positive_count",
        ],
    )
    _write_csv(
        STAGE3_CROSS / "stage3_full_feature_comparisons.csv",
        feature_comparisons,
        [
            "asset_id",
            "sample_id",
            "prompt_name",
            "position_group",
            "direction",
            "evidence_effect",
            "control_mean",
            "evidence_minus_control",
            "evidence_above_control",
        ],
    )
    _write_csv(
        STAGE3_CROSS / "stage3_full_source_control_summary_slices.csv",
        source_slices,
        [
            "asset_id",
            "mask_condition",
            "intervention",
            "mask_variant",
            "comparison_count",
            "mean_source_effect",
            "mean_control_effect",
            "mean_source_minus_control",
            "ci95_low",
            "ci95_high",
            "positive_count",
            "mean_source_correct_minus_wrong",
            "correct_wrong_positive_count",
        ],
    )
    _write_csv(
        STAGE3_CROSS / "stage3_full_source_control_comparisons.csv",
        source_comparisons,
        [
            "asset_id",
            "sample_id",
            "prompt_name",
            "mask_condition",
            "intervention",
            "mask_variant",
            "source_effect",
            "control_effect",
            "source_minus_control",
            "source_above_control",
            "source_correct_minus_wrong",
            "control_correct_minus_wrong",
        ],
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
