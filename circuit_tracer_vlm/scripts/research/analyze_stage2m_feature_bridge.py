#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


CONTROL_GROUPS = [
    "activation_matched_topk",
    "drop_matched_topk",
    "mask_insensitive_topk",
    "random_topk",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
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


def _group(rows: list[dict[str, str]], keys: list[str]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    out: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[tuple(row.get(key, "") for key in keys)].append(row)
    return out


def _effect(row: dict[str, str]) -> float | None:
    if row.get("direction") == "restore":
        return _float(row.get("logit_restore_vs_union"))
    if row.get("direction") == "corrupt":
        return _float(row.get("logit_damage_vs_clean"))
    return None


def _rank_effect(row: dict[str, str]) -> float | None:
    if row.get("direction") == "restore":
        return _float(row.get("rank_restore_vs_union"))
    if row.get("direction") == "corrupt":
        return _float(row.get("rank_damage_vs_clean"))
    return None


def _summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, items in sorted(_group(rows, ["model_family", "direction", "feature_group"]).items()):
        model, direction, feature_group = key
        if direction == "baseline":
            continue
        effects = [value for value in (_effect(row) for row in items) if value is not None]
        rank_effects = [value for value in (_rank_effect(row) for row in items) if value is not None]
        out.append(
            {
                "model_family": model,
                "direction": direction,
                "feature_group": feature_group,
                "n_rows": len(items),
                "positive_logit_n": sum(1 for value in effects if value > 0),
                "positive_rank_n": sum(1 for value in rank_effects if value > 0),
                "mean_logit_effect": _mean(effects),
                "mean_rank_effect": _mean(rank_effects),
                "mean_gap_closure": _mean([value for value in (_float(row.get("gap_closure")) for row in items) if value is not None]),
            }
        )
    return out


def _specificity(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    grouped = _group(rows, ["model_family", "sample_id", "prompt_name", "direction"])
    for key, items in sorted(grouped.items()):
        model, sample_id, prompt_name, direction = key
        if direction == "baseline":
            continue
        by_group = {row.get("feature_group", ""): row for row in items}
        evidence = by_group.get("evidence_topk")
        if evidence is None:
            continue
        evidence_effect = _effect(evidence)
        evidence_rank = _rank_effect(evidence)
        row_out: dict[str, Any] = {
            "model_family": model,
            "sample_id": sample_id,
            "prompt_name": prompt_name,
            "direction": direction,
            "target_answer": evidence.get("target_answer", ""),
            "evidence_logit_effect": evidence_effect,
            "evidence_rank_effect": evidence_rank,
        }
        for control in CONTROL_GROUPS:
            control_row = by_group.get(control)
            control_effect = _effect(control_row) if control_row else None
            control_rank = _rank_effect(control_row) if control_row else None
            row_out[f"evidence_minus_{control}_logit"] = "" if evidence_effect is None or control_effect is None else evidence_effect - control_effect
            row_out[f"evidence_minus_{control}_rank"] = "" if evidence_rank is None or control_rank is None else evidence_rank - control_rank
        out.append(row_out)
    return out


def _decision(summary_rows: list[dict[str, Any]], specificity_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "claim_boundary": (
            "Feature-level causal bridge smoke only. Positive results can support a feature-level bridge, "
            "but do not establish Gemma-style source-control route replication."
        ),
        "model_decisions": {},
    }
    for model in sorted({row["model_family"] for row in summary_rows}):
        model_spec = [row for row in specificity_rows if row["model_family"] == model]
        model_summary = [row for row in summary_rows if row["model_family"] == model]
        direction_status: dict[str, Any] = {}
        for direction in ["restore", "corrupt"]:
            rows = [row for row in model_spec if row["direction"] == direction]
            if not rows:
                direction_status[direction] = {"status": "missing"}
                continue
            control_positive_counts = {}
            for control in CONTROL_GROUPS:
                key = f"evidence_minus_{control}_logit"
                values = [_float(row.get(key)) for row in rows]
                valid = [value for value in values if value is not None]
                control_positive_counts[control] = {
                    "mean": _mean(valid),
                    "positive_n": sum(1 for value in valid if value > 0),
                    "n": len(valid),
                }
            strong_controls = [
                control
                for control, stats in control_positive_counts.items()
                if stats["n"] and stats["positive_n"] >= max(1, int(0.6 * stats["n"])) and stats["mean"] != "" and stats["mean"] > 0
            ]
            direction_status[direction] = {
                "status": "supported" if len(strong_controls) >= 3 else "partial_or_weak",
                "strong_controls": strong_controls,
                "control_stats": control_positive_counts,
            }

        evidence_rows = [
            row
            for row in model_summary
            if row["feature_group"] == "evidence_topk" and row["direction"] in {"restore", "corrupt"}
        ]
        if direction_status.get("restore", {}).get("status") == "supported" and direction_status.get("corrupt", {}).get("status") == "supported":
            status = "feature_bridge_bidirectional_supported"
        elif direction_status.get("restore", {}).get("status") == "supported" or direction_status.get("corrupt", {}).get("status") == "supported":
            status = "feature_bridge_one_direction_supported"
        else:
            status = "feature_bridge_not_established"
        out["model_decisions"][model] = {
            "status": status,
            "direction_status": direction_status,
            "evidence_summary": evidence_rows,
        }
    if not out["model_decisions"]:
        out["overall_status"] = "blocked_no_model_rows"
    elif all(item["status"] == "feature_bridge_bidirectional_supported" for item in out["model_decisions"].values()):
        out["overall_status"] = "cross_model_feature_bridge_supported"
    elif any(item["status"] != "feature_bridge_not_established" for item in out["model_decisions"].values()):
        out["overall_status"] = "partial_feature_bridge_support"
    else:
        out["overall_status"] = "feature_bridge_not_established"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage 2M feature bridge smoke.")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--out-specificity", required=True)
    parser.add_argument("--out-decision", required=True)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for raw in args.inputs.split(","):
        path = Path(raw.strip())
        if path.exists():
            rows.extend(_read_csv(path))
    summary_rows = _summary(rows)
    specificity_rows = _specificity(rows)
    decision = _decision(summary_rows, specificity_rows)
    _write_csv(
        Path(args.out_summary),
        summary_rows,
        [
            "model_family",
            "direction",
            "feature_group",
            "n_rows",
            "positive_logit_n",
            "positive_rank_n",
            "mean_logit_effect",
            "mean_rank_effect",
            "mean_gap_closure",
        ],
    )
    specificity_fields = [
        "model_family",
        "sample_id",
        "prompt_name",
        "direction",
        "target_answer",
        "evidence_logit_effect",
        "evidence_rank_effect",
    ]
    for control in CONTROL_GROUPS:
        specificity_fields.extend([f"evidence_minus_{control}_logit", f"evidence_minus_{control}_rank"])
    _write_csv(Path(args.out_specificity), specificity_rows, specificity_fields)
    _write_json(Path(args.out_decision), decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
