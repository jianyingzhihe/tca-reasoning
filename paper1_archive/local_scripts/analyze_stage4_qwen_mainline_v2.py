#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"

EVIDENCE_RAW = CROSS / "stage4_qwen_evidence_linked_cutter_v2_raw.csv"
SUMMARY = CROSS / "stage4_qwen_mainline_v2_summary.csv"
SPECIFICITY = CROSS / "stage4_qwen_mainline_v2_specificity.csv"
DECISION = CROSS / "stage4_qwen_mainline_v2_decision.json"

CONTROL_GROUPS = [
    "same_position_matched_feature_control",
    "same_feature_random_position_control",
    "random_active_feature_control",
]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "")) if row.get(key, "") != "" else default
    except ValueError:
        return default


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _ci(values: list[float], n: int = 2000, seed: int = 4242) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(_mean(sample))
    means.sort()
    return means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]


def _summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.get("record_type", ""),
                row.get("analysis_group", ""),
                row.get("mask_condition", ""),
                row.get("control_group", ""),
                row.get("top_k", ""),
            )
        ].append(row)
    out = []
    for key, group in sorted(grouped.items()):
        record_type, analysis_group, mask_condition, control_group, top_k = key
        metric = "activation_drop_sum" if record_type == "activation_drop" else "target_logit_effect"
        vals = [_f(row, metric) for row in group]
        lo, hi = _ci(vals)
        out.append(
            {
                "record_type": record_type,
                "analysis_group": analysis_group,
                "mask_condition": mask_condition,
                "control_group": control_group,
                "top_k": top_k,
                "rows": len(group),
                "prompt_runs": len({(row.get("sample_id", ""), row.get("prompt_name", "")) for row in group}),
                "mean_value": _mean(vals),
                "ci95_low": lo,
                "ci95_high": hi,
                "positive_frac": sum(1 for val in vals if val > 0) / len(vals) if vals else 0.0,
            }
        )
    return out


def _index(rows: list[dict[str, str]], metric: str) -> dict[tuple[str, str, str, str, str], float]:
    out = {}
    for row in rows:
        key = (
            row.get("sample_id", "") + "::" + row.get("prompt_name", ""),
            row.get("record_type", ""),
            row.get("mask_condition", ""),
            row.get("control_group", ""),
            row.get("top_k", ""),
        )
        out[key] = _f(row, metric)
    return out


def _specificity(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    groups = sorted({row.get("analysis_group", "") for row in rows} | {"all"})
    topks = sorted({row.get("top_k", "") for row in rows if row.get("top_k", "")}, key=lambda raw: int(raw))
    run_group = {
        row.get("sample_id", "") + "::" + row.get("prompt_name", ""): row.get("analysis_group", "")
        for row in rows
    }
    restore = _index([row for row in rows if row.get("record_type") == "group_restore"], "target_logit_effect")
    drop = _index([row for row in rows if row.get("record_type") == "activation_drop"], "activation_drop_sum")

    def add(metric: str, analysis_group: str, top_k: str, values: list[float]) -> None:
        lo, hi = _ci(values)
        out.append(
            {
                "metric": metric,
                "analysis_group": analysis_group,
                "top_k": top_k,
                "paired_n": len(values),
                "mean_diff": _mean(values),
                "ci95_low": lo,
                "ci95_high": hi,
                "positive_frac": sum(1 for x in values if x > 0) / len(values) if values else 0.0,
            }
        )

    run_ids = sorted({row.get("sample_id", "") + "::" + row.get("prompt_name", "") for row in rows})
    for analysis_group in groups:
        subset = [rid for rid in run_ids if analysis_group == "all" or run_group.get(rid) == analysis_group]
        for top_k in topks:
            for real in ["answer_mask", "union_mask"]:
                restore_source_controls = []
                restore_real_shifted = []
                restore_real_shuffled = []
                drop_real_shifted = []
                drop_real_shuffled = []
                correct_wrong = []
                for rid in subset:
                    source = restore.get((rid, "group_restore", real, "source", top_k))
                    controls = [
                        restore[(rid, "group_restore", real, control, top_k)]
                        for control in CONTROL_GROUPS
                        if (rid, "group_restore", real, control, top_k) in restore
                    ]
                    shifted = restore.get((rid, "group_restore", "shifted_mask", "source", top_k))
                    shuffled = restore.get((rid, "group_restore", "shuffled_mask", "source", top_k))
                    if source is not None and controls:
                        restore_source_controls.append(source - _mean(controls))
                    if source is not None and shifted is not None:
                        restore_real_shifted.append(source - shifted)
                    if source is not None and shuffled is not None:
                        restore_real_shuffled.append(source - shuffled)

                    source_drop = drop.get((rid, "activation_drop", real, "source", top_k))
                    shifted_drop = drop.get((rid, "activation_drop", "shifted_mask", "source", top_k))
                    shuffled_drop = drop.get((rid, "activation_drop", "shuffled_mask", "source", top_k))
                    if source_drop is not None and shifted_drop is not None:
                        drop_real_shifted.append(source_drop - shifted_drop)
                    if source_drop is not None and shuffled_drop is not None:
                        drop_real_shuffled.append(source_drop - shuffled_drop)
                add(f"{real}_restore_source_minus_controls", analysis_group, top_k, restore_source_controls)
                add(f"{real}_restore_real_minus_shifted", analysis_group, top_k, restore_real_shifted)
                add(f"{real}_restore_real_minus_shuffled", analysis_group, top_k, restore_real_shuffled)
                add(f"{real}_activation_drop_real_minus_shifted", analysis_group, top_k, drop_real_shifted)
                add(f"{real}_activation_drop_real_minus_shuffled", analysis_group, top_k, drop_real_shuffled)
    return out


def _metric(spec: list[dict[str, Any]], name: str, top_k: str, group: str = "all") -> dict[str, Any]:
    for row in spec:
        if row["metric"] == name and row["top_k"] == top_k and row["analysis_group"] == group:
            return row
    return {"paired_n": 0, "mean_diff": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "positive_frac": 0.0}


def _positive(row: dict[str, Any], min_n: int = 8) -> bool:
    return int(row["paired_n"]) >= min_n and float(row["ci95_low"]) > 0 and float(row["positive_frac"]) >= 0.6


def _decision(rows: list[dict[str, str]], spec: list[dict[str, Any]]) -> dict[str, Any]:
    best = {"status": "blocked", "top_k": "", "mask_condition": "", "restore_metric": {}, "drop_metric": {}}
    if not rows:
        return {"created_at": _now(), "status": "blocked", "reason": "no_evidence_v2_rows"}
    for top_k in sorted({row.get("top_k", "") for row in rows if row.get("top_k", "")}, key=lambda raw: int(raw)):
        for condition in ["answer_mask", "union_mask"]:
            restore_controls = _metric(spec, f"{condition}_restore_source_minus_controls", top_k)
            restore_shifted = _metric(spec, f"{condition}_restore_real_minus_shifted", top_k)
            restore_shuffled = _metric(spec, f"{condition}_restore_real_minus_shuffled", top_k)
            drop_shifted = _metric(spec, f"{condition}_activation_drop_real_minus_shifted", top_k)
            drop_shuffled = _metric(spec, f"{condition}_activation_drop_real_minus_shuffled", top_k)
            if _positive(restore_controls) and _positive(restore_shifted) and _positive(restore_shuffled):
                status = "qwen_native_evidence_linked_supported"
            elif _positive(drop_shifted) and _positive(drop_shuffled) and _positive(restore_controls):
                status = "partial_evidence_linked_activation_supported"
            else:
                status = ""
            if status:
                return {
                    "created_at": _now(),
                    "status": status,
                    "top_k": top_k,
                    "mask_condition": condition,
                    "restore_source_minus_controls": restore_controls,
                    "restore_real_minus_shifted": restore_shifted,
                    "restore_real_minus_shuffled": restore_shuffled,
                    "activation_drop_real_minus_shifted": drop_shifted,
                    "activation_drop_real_minus_shuffled": drop_shuffled,
                    "claim_boundary": "Evidence-link V2 is Qwen-native cutter support, not full Gemma-style tracing unless automatic tracing gates also pass.",
                }
    # If no evidence-link support, keep the known cutter-only status if rows exist.
    return {
        "created_at": _now(),
        "status": "qwen_cutter_only_supported",
        "reason": "grouped restore or activation-drop specificity did not pass shifted/shuffled controls",
        "raw_rows": len(rows),
        "prompt_runs": len({(row.get("sample_id", ""), row.get("prompt_name", "")) for row in rows}),
        "claim_boundary": "Qwen has causal cutter evidence from Stage4-014/expanded discovery, but evidence-link is not established by this analyzer.",
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_csv(args.evidence_raw)
    summary = _summary(rows)
    spec = _specificity(rows)
    decision = _decision(rows, spec)
    _write_csv(
        args.summary_csv,
        summary,
        ["record_type", "analysis_group", "mask_condition", "control_group", "top_k", "rows", "prompt_runs", "mean_value", "ci95_low", "ci95_high", "positive_frac"],
    )
    _write_csv(args.specificity_csv, spec, ["metric", "analysis_group", "top_k", "paired_n", "mean_diff", "ci95_low", "ci95_high", "positive_frac"])
    _write_json(args.decision_json, decision)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage4-016 Qwen mainline V2 evidence-link artifacts.")
    parser.add_argument("--evidence-raw", type=Path, default=EVIDENCE_RAW)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY)
    parser.add_argument("--specificity-csv", type=Path, default=SPECIFICITY)
    parser.add_argument("--decision-json", type=Path, default=DECISION)
    args = parser.parse_args()
    decision = analyze(args)
    print(json.dumps({"status": decision["status"], "top_k": decision.get("top_k", "")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
