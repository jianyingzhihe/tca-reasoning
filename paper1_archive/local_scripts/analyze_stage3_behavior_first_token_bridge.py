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
STAGE3 = ROOT / "doc" / "experiments" / "stage3"
STAGE3_CROSS = STAGE3 / "cross_model"

SOURCE_INPUTS = [
    ("qwen2p5vl_plt", STAGE3_CROSS / "stage3_qwen2p5vl_plt_source_control.csv"),
    ("qwen2p5vl_clt", STAGE3_CROSS / "stage3_qwen2p5vl_clt_source_control.csv"),
    ("llava15_clt", STAGE3_CROSS / "stage3_llava15_clt_source_control.csv"),
]

OUT_COMPARISONS = STAGE3_CROSS / "stage3_behavior_first_token_comparisons.csv"
OUT_SUMMARY = STAGE3_CROSS / "stage3_behavior_first_token_summary.csv"
OUT_DECISION = STAGE3_CROSS / "stage3_behavior_first_token_decision.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def bootstrap_ci(values: list[float], *, seed: int = 1729, rounds: int = 2000) -> tuple[float | str, float | str]:
    if not values:
        return "", ""
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed + len(values))
    boot = []
    for _ in range(rounds):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boot.append(mean(sample))
    boot.sort()
    return boot[int(0.025 * (rounds - 1))], boot[int(0.975 * (rounds - 1))]


def stable_seed(*parts: str) -> int:
    value = 1729
    for part in parts:
        for char in part:
            value = (value * 131 + ord(char)) % (2**31)
    return value


def build_comparisons(asset_id: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (
            row.get("sample_id", ""),
            row.get("prompt_name", ""),
            row.get("mask_condition", ""),
            row.get("mask_variant", ""),
            row.get("intervention", ""),
            row.get("position_group", ""),
        )
        by_key[key][row.get("feature_role", "")] = row

    out: list[dict[str, Any]] = []
    for (sample_id, prompt_name, mask_condition, mask_variant, intervention, position_group), values in sorted(by_key.items()):
        source = values.get("source")
        control = values.get("matched_control")
        if source is None or control is None:
            continue
        source_logit = to_float(source.get("effect_logit"))
        control_logit = to_float(control.get("effect_logit"))
        source_rank = to_float(source.get("effect_rank"))
        control_rank = to_float(control.get("effect_rank"))
        source_gap_closure = to_float(source.get("gap_closure"))
        control_gap_closure = to_float(control.get("gap_closure"))
        source_correct_wrong = to_float(source.get("correct_minus_wrong_logit"))
        control_correct_wrong = to_float(control.get("correct_minus_wrong_logit"))
        if source_logit is None or control_logit is None:
            continue
        out.append(
            {
                "asset_id": asset_id,
                "model_family": source.get("model_family", ""),
                "sample_id": sample_id,
                "prompt_name": prompt_name,
                "mask_condition": mask_condition,
                "mask_variant": mask_variant,
                "intervention": intervention,
                "position_group": position_group,
                "target_answer": source.get("target_answer", ""),
                "target_token_id": source.get("target_token_id", ""),
                "target_token": source.get("target_token", ""),
                "source_effect_logit": source_logit,
                "control_effect_logit": control_logit,
                "source_minus_control_logit": source_logit - control_logit,
                "source_effect_rank": source_rank if source_rank is not None else "",
                "control_effect_rank": control_rank if control_rank is not None else "",
                "source_minus_control_rank": "" if source_rank is None or control_rank is None else source_rank - control_rank,
                "source_gap_closure": source_gap_closure if source_gap_closure is not None else "",
                "control_gap_closure": control_gap_closure if control_gap_closure is not None else "",
                "source_minus_control_gap_closure": ""
                if source_gap_closure is None or control_gap_closure is None
                else source_gap_closure - control_gap_closure,
                "source_correct_minus_wrong_logit": source_correct_wrong if source_correct_wrong is not None else "",
                "control_correct_minus_wrong_logit": control_correct_wrong if control_correct_wrong is not None else "",
                "source_minus_control_correct_wrong_logit": ""
                if source_correct_wrong is None or control_correct_wrong is None
                else source_correct_wrong - control_correct_wrong,
                "source_top1_token": source.get("top1_token", ""),
                "reference_top1_token": source.get("reference_top1_token", ""),
            }
        )
    return out


def summarize(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        key = (
            row["asset_id"],
            row["mask_condition"],
            row["mask_variant"],
            row["intervention"],
            row["position_group"],
        )
        grouped[key].append(row)

    out: list[dict[str, Any]] = []
    for (asset_id, mask_condition, mask_variant, intervention, position_group), rows in sorted(grouped.items()):
        logit_diffs = [float(row["source_minus_control_logit"]) for row in rows]
        rank_diffs = [
            float(row["source_minus_control_rank"])
            for row in rows
            if row.get("source_minus_control_rank") != ""
        ]
        gap_diffs = [
            float(row["source_minus_control_gap_closure"])
            for row in rows
            if row.get("source_minus_control_gap_closure") != ""
        ]
        correct_wrong_diffs = [
            float(row["source_minus_control_correct_wrong_logit"])
            for row in rows
            if row.get("source_minus_control_correct_wrong_logit") != ""
        ]
        ci_low, ci_high = bootstrap_ci(
            logit_diffs,
            seed=stable_seed(asset_id, mask_condition, mask_variant, intervention, position_group, "logit"),
        )
        rank_ci_low, rank_ci_high = bootstrap_ci(
            rank_diffs,
            seed=stable_seed(asset_id, mask_condition, mask_variant, intervention, position_group, "rank"),
        )
        out.append(
            {
                "asset_id": asset_id,
                "mask_condition": mask_condition,
                "mask_variant": mask_variant,
                "intervention": intervention,
                "position_group": position_group,
                "comparison_count": len(rows),
                "mean_source_minus_control_logit": mean(logit_diffs),
                "ci95_low_logit": ci_low,
                "ci95_high_logit": ci_high,
                "positive_logit_count": sum(value > 0 for value in logit_diffs),
                "mean_source_minus_control_rank": mean(rank_diffs) if rank_diffs else "",
                "ci95_low_rank": rank_ci_low,
                "ci95_high_rank": rank_ci_high,
                "positive_rank_count": sum(value > 0 for value in rank_diffs),
                "mean_source_minus_control_gap_closure": mean(gap_diffs) if gap_diffs else "",
                "positive_gap_closure_count": sum(value > 0 for value in gap_diffs),
                "mean_source_minus_control_correct_wrong_logit": mean(correct_wrong_diffs) if correct_wrong_diffs else "",
                "positive_correct_wrong_count": sum(value > 0 for value in correct_wrong_diffs),
            }
        )
    return out


def decide(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    for asset_id in sorted({row["asset_id"] for row in summary_rows}):
        primary = [
            row
            for row in summary_rows
            if row["asset_id"] == asset_id
            and row["mask_variant"] == "real_mask"
            and row["mask_condition"] in {"answer_mask", "union_mask"}
            and row["intervention"] in {"restore", "zeroing"}
            and row["position_group"] == "top_hidden_delta_plus_answer_adjacent"
        ]
        supported_items = []
        partial_items = []
        for row in primary:
            lo = to_float(row.get("ci95_low_logit"))
            rank_mean = to_float(row.get("mean_source_minus_control_rank"))
            if lo is not None and lo > 0 and (rank_mean is None or rank_mean >= 0):
                supported_items.append(row)
            elif to_float(row.get("mean_source_minus_control_logit")) and to_float(row.get("mean_source_minus_control_logit")) > 0:
                partial_items.append(row)
        if len(supported_items) >= 4:
            status = "supported_first_token_rank_bridge"
        elif supported_items:
            status = "partial_first_token_rank_bridge"
        elif partial_items:
            status = "weak_logit_only_bridge"
        else:
            status = "not_supported"
        decisions[asset_id] = {
            "status": status,
            "primary_rows": len(primary),
            "supported_primary_rows": len(supported_items),
            "partial_positive_mean_rows": len(partial_items),
            "interpretation": {
                "qwen2p5vl_plt": (
                    "PLT-aligned Qwen source/control effects carry through to the first answer token logit/rank "
                    "when supported; this is still not decoded generation."
                ),
                "qwen2p5vl_clt": (
                    "CLT Qwen source/control effects are expected to be larger; positive results are auxiliary "
                    "robustness evidence, not PLT mainline replacement."
                ),
                "llava15_clt": (
                    "LLaVA-CLT can show weak first-token effects, but unstable restoration/source specificity "
                    "should remain hidden/weak feature support unless controls are clearly beaten."
                ),
            }.get(asset_id, ""),
        }
    return decisions


def main() -> None:
    all_comparisons: list[dict[str, Any]] = []
    input_rows: dict[str, int] = {}
    for asset_id, path in SOURCE_INPUTS:
        rows = read_csv(path)
        input_rows[asset_id] = len(rows)
        all_comparisons.extend(build_comparisons(asset_id, rows))

    summary_rows = summarize(all_comparisons)
    decision = {
        "status": "completed",
        "input_rows": input_rows,
        "comparison_rows": len(all_comparisons),
        "asset_decisions": decide(summary_rows),
        "artifacts": {
            "comparisons": str(OUT_COMPARISONS),
            "summary": str(OUT_SUMMARY),
        },
        "claim_boundary": (
            "This analyzes first answer token logit/rank bridge already present in Stage3 source-control probes. "
            "It does not test short greedy decoded generation and does not establish full Gemma-style source tracing."
        ),
    }

    write_csv(
        OUT_COMPARISONS,
        all_comparisons,
        [
            "asset_id",
            "model_family",
            "sample_id",
            "prompt_name",
            "mask_condition",
            "mask_variant",
            "intervention",
            "position_group",
            "target_answer",
            "target_token_id",
            "target_token",
            "source_effect_logit",
            "control_effect_logit",
            "source_minus_control_logit",
            "source_effect_rank",
            "control_effect_rank",
            "source_minus_control_rank",
            "source_gap_closure",
            "control_gap_closure",
            "source_minus_control_gap_closure",
            "source_correct_minus_wrong_logit",
            "control_correct_minus_wrong_logit",
            "source_minus_control_correct_wrong_logit",
            "source_top1_token",
            "reference_top1_token",
        ],
    )
    write_csv(
        OUT_SUMMARY,
        summary_rows,
        [
            "asset_id",
            "mask_condition",
            "mask_variant",
            "intervention",
            "position_group",
            "comparison_count",
            "mean_source_minus_control_logit",
            "ci95_low_logit",
            "ci95_high_logit",
            "positive_logit_count",
            "mean_source_minus_control_rank",
            "ci95_low_rank",
            "ci95_high_rank",
            "positive_rank_count",
            "mean_source_minus_control_gap_closure",
            "positive_gap_closure_count",
            "mean_source_minus_control_correct_wrong_logit",
            "positive_correct_wrong_count",
        ],
    )
    OUT_DECISION.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
