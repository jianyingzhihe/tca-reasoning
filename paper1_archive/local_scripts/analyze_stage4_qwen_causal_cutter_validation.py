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
STAGE4 = ROOT / "doc" / "experiments" / "stage4"
CROSS = STAGE4 / "cross_model"

RAW = CROSS / "stage4_qwen_causal_cutter_validation_raw.csv"
SUMMARY = CROSS / "stage4_qwen_causal_cutter_validation_summary.csv"
SPECIFICITY = CROSS / "stage4_qwen_causal_cutter_validation_specificity.csv"
CASE_TABLE = CROSS / "stage4_qwen_causal_cutter_validation_case_table.csv"
DECISION = CROSS / "stage4_qwen_causal_cutter_validation_decision.json"


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


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "")) if row.get(key, "") != "" else default
    except ValueError:
        return default


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _bootstrap_ci(values: list[float], n: int = 2000, seed: int = 20260524) -> tuple[float, float]:
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
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return lo, hi


def _summarize_group(rows: list[dict[str, str]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    out = []
    for key, group_rows in sorted(grouped.items()):
        effects = [_f(row, "logit_effect") for row in group_rows if row.get("status", "ok") == "ok"]
        ranks = [_f(row, "rank_effect") for row in group_rows if row.get("status", "ok") == "ok"]
        ci_lo, ci_hi = _bootstrap_ci(effects)
        payload = {name: value for name, value in zip(keys, key, strict=False)}
        payload.update(
            {
                "rows": len(group_rows),
                "candidate_count": len({row.get("candidate_id", "") for row in group_rows}),
                "mean_logit_effect": _mean(effects),
                "ci95_low": ci_lo,
                "ci95_high": ci_hi,
                "positive_logit_frac": sum(1 for x in effects if x > 0) / len(effects) if effects else 0.0,
                "mean_rank_effect": _mean(ranks),
                "positive_rank_frac": sum(1 for x in ranks if x > 0) / len(ranks) if ranks else 0.0,
            }
        )
        out.append(payload)
    return out


def _index_effects(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str, str], float]:
    # key: candidate, intervention_kind, mask_condition, control_group, token_scored
    out = {}
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        key = (
            row.get("candidate_id", ""),
            row.get("intervention_kind", ""),
            row.get("mask_condition", ""),
            row.get("control_group", ""),
            row.get("token_scored", ""),
        )
        out[key] = _f(row, "logit_effect")
    return out


def _specificity(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    effects = _index_effects(rows)
    candidates = sorted({row.get("candidate_id", "") for row in rows if row.get("candidate_id", "")})
    out: list[dict[str, Any]] = []

    def add_metric(metric: str, analysis_group: str, diffs: list[float]) -> None:
        lo, hi = _bootstrap_ci(diffs)
        out.append(
            {
                "metric": metric,
                "analysis_group": analysis_group,
                "paired_n": len(diffs),
                "mean_diff": _mean(diffs),
                "ci95_low": lo,
                "ci95_high": hi,
                "positive_frac": sum(1 for x in diffs if x > 0) / len(diffs) if diffs else 0.0,
            }
        )

    group_by_candidate = {
        cid: next((row.get("analysis_group", "") for row in rows if row.get("candidate_id") == cid), "")
        for cid in candidates
    }
    for group_name in ["all", "numeric", "non_numeric"]:
        candidate_subset = [
            cid
            for cid in candidates
            if group_name == "all" or group_by_candidate.get(cid) == group_name
        ]
        clean_source_vs_controls = []
        correct_minus_wrong = []
        for cid in candidate_subset:
            source = effects.get((cid, "clean_zeroing", "clean", "source", "target"))
            wrong = effects.get((cid, "clean_zeroing", "clean", "source", "wrong"))
            controls = [
                effects[(cid, "clean_zeroing", "clean", control, "target")]
                for control in CONTROL_GROUPS
                if (cid, "clean_zeroing", "clean", control, "target") in effects
            ]
            if source is not None and controls:
                clean_source_vs_controls.append(source - _mean(controls))
            if source is not None and wrong is not None:
                correct_minus_wrong.append(source - wrong)
        add_metric("clean_source_minus_controls", group_name, clean_source_vs_controls)
        add_metric("clean_correct_minus_wrong", group_name, correct_minus_wrong)

        for real_condition in ["answer_mask", "union_mask"]:
            source_vs_controls = []
            real_minus_shifted = []
            real_minus_shuffled = []
            restore_correct_wrong = []
            for cid in candidate_subset:
                source = effects.get((cid, "mask_restore", real_condition, "source", "target"))
                wrong = effects.get((cid, "mask_restore", real_condition, "source", "wrong"))
                controls = [
                    effects[(cid, "mask_restore", real_condition, control, "target")]
                    for control in CONTROL_GROUPS
                    if (cid, "mask_restore", real_condition, control, "target") in effects
                ]
                shifted = effects.get((cid, "mask_restore", "shifted_mask", "source", "target"))
                shuffled = effects.get((cid, "mask_restore", "shuffled_mask", "source", "target"))
                if source is not None and controls:
                    source_vs_controls.append(source - _mean(controls))
                if source is not None and shifted is not None:
                    real_minus_shifted.append(source - shifted)
                if source is not None and shuffled is not None:
                    real_minus_shuffled.append(source - shuffled)
                if source is not None and wrong is not None:
                    restore_correct_wrong.append(source - wrong)
            add_metric(f"{real_condition}_restore_source_minus_controls", group_name, source_vs_controls)
            add_metric(f"{real_condition}_restore_real_minus_shifted", group_name, real_minus_shifted)
            add_metric(f"{real_condition}_restore_real_minus_shuffled", group_name, real_minus_shuffled)
            add_metric(f"{real_condition}_restore_correct_minus_wrong", group_name, restore_correct_wrong)
    return out


def _case_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    effects = _index_effects(rows)
    by_candidate: dict[str, dict[str, str]] = {}
    for row in rows:
        by_candidate.setdefault(row.get("candidate_id", ""), row)
    out = []
    for cid, row in sorted(by_candidate.items()):
        controls = [
            effects[(cid, "clean_zeroing", "clean", control, "target")]
            for control in CONTROL_GROUPS
            if (cid, "clean_zeroing", "clean", control, "target") in effects
        ]
        source_clean = effects.get((cid, "clean_zeroing", "clean", "source", "target"), 0.0)
        answer_restore = effects.get((cid, "mask_restore", "answer_mask", "source", "target"), "")
        union_restore = effects.get((cid, "mask_restore", "union_mask", "source", "target"), "")
        shifted_restore = effects.get((cid, "mask_restore", "shifted_mask", "source", "target"), "")
        shuffled_restore = effects.get((cid, "mask_restore", "shuffled_mask", "source", "target"), "")
        out.append(
            {
                "candidate_id": cid,
                "analysis_group": row.get("analysis_group", ""),
                "sample_id": row.get("sample_id", ""),
                "prompt_name": row.get("prompt_name", ""),
                "target_answer": row.get("target_answer", ""),
                "target_token": row.get("target_token", ""),
                "source_feature_id": row.get("source_feature_id", ""),
                "source_pos": row.get("source_pos", ""),
                "source_clean_damage": source_clean,
                "mean_control_clean_damage": _mean(controls),
                "source_minus_controls": source_clean - _mean(controls) if controls else "",
                "answer_restore": answer_restore,
                "union_restore": union_restore,
                "shifted_restore": shifted_restore,
                "shuffled_restore": shuffled_restore,
            }
        )
    return out


def _metric_lookup(spec_rows: list[dict[str, Any]], metric: str, group: str = "all") -> dict[str, Any]:
    for row in spec_rows:
        if row["metric"] == metric and row["analysis_group"] == group:
            return row
    return {"paired_n": 0, "mean_diff": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "positive_frac": 0.0}


def _decision(spec_rows: list[dict[str, Any]], raw_rows: list[dict[str, str]]) -> dict[str, Any]:
    clean = _metric_lookup(spec_rows, "clean_source_minus_controls", "all")
    correct_wrong = _metric_lookup(spec_rows, "clean_correct_minus_wrong", "all")
    answer_shifted = _metric_lookup(spec_rows, "answer_mask_restore_real_minus_shifted", "all")
    answer_shuffled = _metric_lookup(spec_rows, "answer_mask_restore_real_minus_shuffled", "all")
    union_shifted = _metric_lookup(spec_rows, "union_mask_restore_real_minus_shifted", "all")
    union_shuffled = _metric_lookup(spec_rows, "union_mask_restore_real_minus_shuffled", "all")
    numeric_clean = _metric_lookup(spec_rows, "clean_source_minus_controls", "numeric")
    non_numeric_clean = _metric_lookup(spec_rows, "clean_source_minus_controls", "non_numeric")

    def supported(row: dict[str, Any], min_n: int = 6) -> bool:
        return int(row["paired_n"]) >= min_n and float(row["ci95_low"]) > 0 and float(row["positive_frac"]) >= 0.6

    causal = supported(clean)
    target_specific = supported(correct_wrong)
    answer_evidence = supported(answer_shifted) and supported(answer_shuffled)
    union_evidence = supported(union_shifted) and supported(union_shuffled)
    evidence = causal and target_specific and (answer_evidence or union_evidence)
    numeric_only = (not causal) and supported(numeric_clean, min_n=3) and not supported(non_numeric_clean, min_n=3)
    if evidence:
        status = "evidence_linked_cutter_supported"
    elif causal:
        status = "causal_cutter_supported"
    elif numeric_only:
        status = "numeric_only_supported"
    elif raw_rows:
        status = "not_supported"
    else:
        status = "blocked"
    return {
        "created_at": _now(),
        "status": status,
        "raw_rows": len(raw_rows),
        "candidate_count": len({row.get("candidate_id", "") for row in raw_rows if row.get("candidate_id", "")}),
        "primary_metrics": {
            "clean_source_minus_controls": clean,
            "clean_correct_minus_wrong": correct_wrong,
            "answer_real_minus_shifted": answer_shifted,
            "answer_real_minus_shuffled": answer_shuffled,
            "union_real_minus_shifted": union_shifted,
            "union_real_minus_shuffled": union_shuffled,
            "numeric_clean_source_minus_controls": numeric_clean,
            "non_numeric_clean_source_minus_controls": non_numeric_clean,
        },
        "claim_boundary": (
            "Positive status supports Qwen-native causal-screened cutter nodes. "
            "It does not establish Gemma-style automatic source tracing replication."
        ),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    raw_rows = [row for row in _read_csv(args.raw_csv) if row.get("status", "ok") == "ok"]
    summary_rows = _summarize_group(
        raw_rows,
        ["analysis_group", "intervention_kind", "mask_condition", "control_group", "token_scored"],
    )
    spec_rows = _specificity(raw_rows)
    case_rows = _case_table(raw_rows)
    decision = _decision(spec_rows, raw_rows)
    _write_csv(
        args.summary_csv,
        summary_rows,
        [
            "analysis_group",
            "intervention_kind",
            "mask_condition",
            "control_group",
            "token_scored",
            "rows",
            "candidate_count",
            "mean_logit_effect",
            "ci95_low",
            "ci95_high",
            "positive_logit_frac",
            "mean_rank_effect",
            "positive_rank_frac",
        ],
    )
    _write_csv(
        args.specificity_csv,
        spec_rows,
        ["metric", "analysis_group", "paired_n", "mean_diff", "ci95_low", "ci95_high", "positive_frac"],
    )
    _write_csv(
        args.case_table_csv,
        case_rows,
        [
            "candidate_id",
            "analysis_group",
            "sample_id",
            "prompt_name",
            "target_answer",
            "target_token",
            "source_feature_id",
            "source_pos",
            "source_clean_damage",
            "mean_control_clean_damage",
            "source_minus_controls",
            "answer_restore",
            "union_restore",
            "shifted_restore",
            "shuffled_restore",
        ],
    )
    _write_json(args.decision_json, decision)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage4-014 Qwen causal cutter validation.")
    parser.add_argument("--raw-csv", type=Path, default=RAW)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY)
    parser.add_argument("--specificity-csv", type=Path, default=SPECIFICITY)
    parser.add_argument("--case-table-csv", type=Path, default=CASE_TABLE)
    parser.add_argument("--decision-json", type=Path, default=DECISION)
    args = parser.parse_args()
    decision = analyze(args)
    print(json.dumps({"status": decision["status"], "candidate_count": decision["candidate_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
