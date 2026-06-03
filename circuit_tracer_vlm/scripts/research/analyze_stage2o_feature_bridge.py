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


EVIDENCE_GROUP = "evidence_attribution_topk"
CONTROL_GROUPS = [
    "activation_matched_topk",
    "drop_matched_topk",
    "attribution_matched_mask_insensitive_topk",
    "random_active_topk",
]


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


def _effect(row: dict[str, str]) -> float | None:
    if row.get("direction") == "restore":
        return _float(row.get("logit_restore_vs_mask"))
    if row.get("direction") == "corrupt":
        return _float(row.get("logit_damage_vs_clean"))
    return None


def _rank_effect(row: dict[str, str]) -> float | None:
    if row.get("direction") == "restore":
        return _float(row.get("rank_restore_vs_mask"))
    if row.get("direction") == "corrupt":
        return _float(row.get("rank_damage_vs_clean"))
    return None


def _summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, items in sorted(_group(rows, ["model_family", "mask_condition", "position_group", "direction", "feature_group"]).items()):
        model, mask_condition, position_group, direction, feature_group = key
        effects = [value for value in (_effect(row) for row in items) if value is not None]
        ranks = [value for value in (_rank_effect(row) for row in items) if value is not None]
        ci_low, ci_high, status = _bootstrap(effects)
        out.append(
            {
                "model_family": model,
                "mask_condition": mask_condition,
                "position_group": position_group,
                "direction": direction,
                "feature_group": feature_group,
                "n_rows": len(items),
                "positive_logit_n": sum(1 for value in effects if value > 0),
                "positive_rank_n": sum(1 for value in ranks if value > 0),
                "mean_logit_effect": _mean(effects),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "effect_status": status,
                "mean_rank_effect": _mean(ranks),
                "mean_gap_closure": _mean([value for value in (_float(row.get("gap_closure")) for row in items) if value is not None]),
            }
        )
    return out


def _specificity(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    grouped = _group(rows, ["model_family", "sample_id", "prompt_name", "mask_condition", "position_group", "direction"])
    for key, items in sorted(grouped.items()):
        model, sample_id, prompt_name, mask_condition, position_group, direction = key
        by_group = {row.get("feature_group", ""): row for row in items}
        evidence = by_group.get(EVIDENCE_GROUP)
        if evidence is None:
            continue
        evidence_effect = _effect(evidence)
        evidence_rank = _rank_effect(evidence)
        row_out: dict[str, Any] = {
            "model_family": model,
            "sample_id": sample_id,
            "prompt_name": prompt_name,
            "mask_condition": mask_condition,
            "position_group": position_group,
            "direction": direction,
            "target_answer": evidence.get("target_answer", ""),
            "evidence_logit_effect": evidence_effect,
            "evidence_rank_effect": evidence_rank,
            "feature_ids": evidence.get("feature_ids", ""),
        }
        positive_controls = 0
        for control in CONTROL_GROUPS:
            control_row = by_group.get(control)
            control_effect = _effect(control_row) if control_row else None
            control_rank = _rank_effect(control_row) if control_row else None
            logit_gap = "" if evidence_effect is None or control_effect is None else evidence_effect - control_effect
            rank_gap = "" if evidence_rank is None or control_rank is None else evidence_rank - control_rank
            row_out[f"evidence_minus_{control}_logit"] = logit_gap
            row_out[f"evidence_minus_{control}_rank"] = rank_gap
            if isinstance(logit_gap, float) and logit_gap > 0:
                positive_controls += 1
        row_out["positive_control_count"] = positive_controls
        row_out["above_all_controls"] = positive_controls == len(CONTROL_GROUPS)
        out.append(row_out)
    return out


def _model_decision(summary_rows: list[dict[str, Any]], specificity_rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    primary = [
        row
        for row in specificity_rows
        if row["model_family"] == model and row["position_group"] == "top_hidden_delta_plus_answer_adjacent"
    ]
    direction_decisions: dict[str, Any] = {}
    for direction in ["restore", "corrupt"]:
        rows = [row for row in primary if row["direction"] == direction]
        all_control = [row for row in rows if row.get("above_all_controls") is True or row.get("above_all_controls") == "True"]
        counts = [int(row.get("positive_control_count", 0)) for row in rows]
        evidence_summary = [
            row
            for row in summary_rows
            if row["model_family"] == model
            and row["position_group"] == "top_hidden_delta_plus_answer_adjacent"
            and row["direction"] == direction
            and row["feature_group"] == EVIDENCE_GROUP
        ]
        stable_evidence = any(row.get("effect_status") == "stable_positive" for row in evidence_summary)
        status = "supported" if len(all_control) >= max(1, int(0.6 * len(rows))) and stable_evidence else "partial_or_weak"
        if not rows:
            status = "missing"
        direction_decisions[direction] = {
            "status": status,
            "n_rows": len(rows),
            "above_all_controls_n": len(all_control),
            "mean_positive_control_count": _mean(counts),
            "evidence_summary": evidence_summary,
        }
    if direction_decisions.get("restore", {}).get("status") == "supported" and direction_decisions.get("corrupt", {}).get("status") == "supported":
        status = "feature_bridge_bidirectional_supported"
    elif direction_decisions.get("restore", {}).get("status") == "supported" or direction_decisions.get("corrupt", {}).get("status") == "supported":
        status = "feature_bridge_one_direction_supported"
    elif any(item.get("status") == "partial_or_weak" for item in direction_decisions.values()):
        status = "feature_bridge_partial_or_weak"
    else:
        status = "feature_bridge_not_established"
    return {"status": status, "direction_decisions": direction_decisions}


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage 2O attribution-weighted feature bridge.")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--out-specificity", required=True)
    parser.add_argument("--out-decision", required=True)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for raw in args.inputs.split(","):
        path = Path(raw.strip())
        rows.extend(_read_csv(path))
    summary_rows = _summary(rows)
    specificity_rows = _specificity(rows)
    models = sorted({row["model_family"] for row in summary_rows})
    decision = {
        "claim_boundary": (
            "Stage 2O Experiment 1 checks attribution-weighted feature-level bridge. "
            "It does not establish Gemma-style source-control route replication."
        ),
        "model_decisions": {model: _model_decision(summary_rows, specificity_rows, model) for model in models},
    }
    if not models:
        decision["overall_status"] = "blocked_no_rows"
    elif any(item["status"] in {"feature_bridge_bidirectional_supported", "feature_bridge_one_direction_supported"} for item in decision["model_decisions"].values()):
        decision["overall_status"] = "partial_or_model_specific_feature_bridge_support"
    elif any(item["status"] == "feature_bridge_partial_or_weak" for item in decision["model_decisions"].values()):
        decision["overall_status"] = "feature_bridge_partial_or_weak"
    else:
        decision["overall_status"] = "feature_bridge_not_established"

    _write_csv(
        Path(args.out_summary),
        summary_rows,
        [
            "model_family",
            "mask_condition",
            "position_group",
            "direction",
            "feature_group",
            "n_rows",
            "positive_logit_n",
            "positive_rank_n",
            "mean_logit_effect",
            "ci95_low",
            "ci95_high",
            "effect_status",
            "mean_rank_effect",
            "mean_gap_closure",
        ],
    )
    specificity_fields = [
        "model_family",
        "sample_id",
        "prompt_name",
        "mask_condition",
        "position_group",
        "direction",
        "target_answer",
        "evidence_logit_effect",
        "evidence_rank_effect",
        "feature_ids",
        "positive_control_count",
        "above_all_controls",
    ]
    for control in CONTROL_GROUPS:
        specificity_fields.extend([f"evidence_minus_{control}_logit", f"evidence_minus_{control}_rank"])
    _write_csv(Path(args.out_specificity), specificity_rows, specificity_fields)
    _write_json(Path(args.out_decision), decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
