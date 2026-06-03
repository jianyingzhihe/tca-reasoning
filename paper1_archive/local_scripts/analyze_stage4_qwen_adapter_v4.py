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

CONTROL_GROUPS = [
    "same_position_matched_feature_control",
    "same_feature_random_position_control",
    "random_active_feature_control",
]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
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


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value != "" else default
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _ci(values: list[float], n: int = 2000, seed: int = 20260526) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        means.append(_mean([values[rng.randrange(len(values))] for _ in values]))
    means.sort()
    return means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]


def _metric_row(metric: str, pack: str, values: list[float], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    lo, hi = _ci(values)
    row: dict[str, Any] = {
        "metric": metric,
        "pack": pack,
        "n": len(values),
        "mean": _mean(values),
        "ci95_low": lo,
        "ci95_high": hi,
        "positive_frac": sum(1 for value in values if value > 0) / len(values) if values else 0.0,
    }
    if extra:
        row.update(extra)
    return row


def _positive(row: dict[str, Any], min_n: int = 8) -> bool:
    return int(row.get("n", 0)) >= min_n and float(row.get("ci95_low", 0.0)) > 0 and float(row.get("positive_frac", 0.0)) >= 0.6


def _prefix(pack: str, mode: str, layer: int, tag: str) -> str:
    suffix = f"_{tag}" if tag else ""
    return f"stage4_qwen_adapter_v4_{pack}_{mode}_L{layer}{suffix}"


def _candidate_summary(rows: list[dict[str, str]], pack: str) -> list[dict[str, Any]]:
    out = []
    groups = ["all"] + sorted({row.get("analysis_group", "") for row in rows if row.get("analysis_group", "")})
    for group in groups:
        subset = rows if group == "all" else [row for row in rows if row.get("analysis_group") == group]
        out.append(
            {
                "record_type": "candidate_manifest",
                "pack": pack,
                "group": group,
                "rows": len(subset),
                "main_rows": sum(1 for row in subset if row.get("include_main") == "1"),
                "prompt_runs": len({(row.get("sample_id"), row.get("prompt_name")) for row in subset}),
                "mean_adapter_v4_score": _mean([_f(row, "adapter_v4_score") for row in subset]),
                "mean_zeroing_damage": _mean([_f(row, "zeroing_damage") for row in subset]),
                "exact_match_rows": sum(1 for row in subset if row.get("v4_match_level") == "exact_pos_feature"),
                "same_feature_rows": sum(1 for row in subset if row.get("v4_match_level") == "same_feature"),
            }
        )
    return out


def _zeroing_specificity(rows: list[dict[str, str]], pack: str) -> list[dict[str, Any]]:
    target = [row for row in rows if row.get("status", "ok") in {"", "ok"} and row.get("intervention_kind") == "clean_zeroing" and row.get("token_scored") == "target"]
    wrong = [row for row in rows if row.get("status", "ok") in {"", "ok"} and row.get("intervention_kind") == "clean_zeroing" and row.get("token_scored") == "wrong"]
    by_candidate: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in target:
        by_candidate[row.get("candidate_id", "")][row.get("control_group", "")] = row
    wrong_by_candidate = {row.get("candidate_id", ""): row for row in wrong if row.get("control_group") == "source"}
    source_minus_controls = []
    correct_minus_wrong = []
    source_rank = []
    for candidate_id, group in by_candidate.items():
        source = group.get("source")
        controls = [group[name] for name in CONTROL_GROUPS if name in group]
        if source and controls:
            source_minus_controls.append(_f(source, "logit_effect") - _mean([_f(row, "logit_effect") for row in controls]))
            source_rank.append(_f(source, "rank_effect"))
        wrong_row = wrong_by_candidate.get(candidate_id)
        if source and wrong_row:
            correct_minus_wrong.append(_f(source, "logit_effect") - _f(wrong_row, "logit_effect"))
    return [
        _metric_row("clean_zeroing_source_minus_controls", pack, source_minus_controls),
        _metric_row("clean_zeroing_correct_minus_wrong", pack, correct_minus_wrong),
        _metric_row("clean_zeroing_target_rank_effect", pack, source_rank),
    ]


def _group_specificity(rows: list[dict[str, str]], pack: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    restore = [row for row in rows if row.get("record_type") == "group_restore"]
    drops = [row for row in rows if row.get("record_type") == "activation_drop"]

    def key(row: dict[str, str]) -> tuple[str, str, str, str]:
        return row.get("sample_id", "") + "::" + row.get("prompt_name", ""), row.get("mask_condition", ""), row.get("control_group", ""), row.get("top_k", "")

    restore_idx = {key(row): row for row in restore}
    drop_idx = {key(row): row for row in drops}
    run_ids = sorted({row.get("sample_id", "") + "::" + row.get("prompt_name", "") for row in rows})
    topks = sorted({row.get("top_k", "") for row in rows if row.get("top_k", "")}, key=lambda raw: int(raw))
    for top_k in topks:
        for condition in ["answer_mask", "union_mask"]:
            restore_source_controls = []
            restore_real_shifted = []
            restore_real_shuffled = []
            restore_correct_wrong = []
            restore_rank = []
            drop_real_shifted = []
            drop_real_shuffled = []
            for run_id in run_ids:
                source = restore_idx.get((run_id, condition, "source", top_k))
                controls = [restore_idx[(run_id, condition, name, top_k)] for name in CONTROL_GROUPS if (run_id, condition, name, top_k) in restore_idx]
                shifted = restore_idx.get((run_id, "shifted_mask", "source", top_k))
                shuffled = restore_idx.get((run_id, "shuffled_mask", "source", top_k))
                if source and controls:
                    restore_source_controls.append(_f(source, "target_logit_effect") - _mean([_f(row, "target_logit_effect") for row in controls]))
                    restore_correct_wrong.append(_f(source, "target_logit_effect") - _f(source, "wrong_logit_effect"))
                    restore_rank.append(_f(source, "target_rank_effect"))
                if source and shifted:
                    restore_real_shifted.append(_f(source, "target_logit_effect") - _f(shifted, "target_logit_effect"))
                if source and shuffled:
                    restore_real_shuffled.append(_f(source, "target_logit_effect") - _f(shuffled, "target_logit_effect"))
                source_drop = drop_idx.get((run_id, condition, "source", top_k))
                shifted_drop = drop_idx.get((run_id, "shifted_mask", "source", top_k))
                shuffled_drop = drop_idx.get((run_id, "shuffled_mask", "source", top_k))
                if source_drop and shifted_drop:
                    drop_real_shifted.append(_f(source_drop, "activation_drop_sum") - _f(shifted_drop, "activation_drop_sum"))
                if source_drop and shuffled_drop:
                    drop_real_shuffled.append(_f(source_drop, "activation_drop_sum") - _f(shuffled_drop, "activation_drop_sum"))
            extra = {"mask_condition": condition, "top_k": top_k}
            out.append(_metric_row("restore_source_minus_controls", pack, restore_source_controls, extra))
            out.append(_metric_row("restore_real_minus_shifted", pack, restore_real_shifted, extra))
            out.append(_metric_row("restore_real_minus_shuffled", pack, restore_real_shuffled, extra))
            out.append(_metric_row("restore_correct_minus_wrong", pack, restore_correct_wrong, extra))
            out.append(_metric_row("restore_target_rank_effect", pack, restore_rank, extra))
            out.append(_metric_row("activation_drop_real_minus_shifted", pack, drop_real_shifted, extra))
            out.append(_metric_row("activation_drop_real_minus_shuffled", pack, drop_real_shuffled, extra))
    return out


def _best(spec: list[dict[str, Any]], pack: str, metric: str, condition: str = "", top_k: str = "") -> dict[str, Any]:
    rows = [
        row for row in spec
        if row.get("pack") == pack
        and row.get("metric") == metric
        and (not condition or row.get("mask_condition") == condition)
        and (not top_k or row.get("top_k") == top_k)
    ]
    if not rows:
        return {"n": 0, "mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "positive_frac": 0.0}
    return max(rows, key=lambda row: (float(row.get("ci95_low", 0.0)), float(row.get("mean", 0.0))))


def _gate(spec: list[dict[str, Any]], pack: str) -> dict[str, Any]:
    zero_controls = _best(spec, pack, "clean_zeroing_source_minus_controls")
    zero_wrong = _best(spec, pack, "clean_zeroing_correct_minus_wrong")
    zero_rank = _best(spec, pack, "clean_zeroing_target_rank_effect")
    best_restore: dict[str, Any] = {"passed": False, "score": -999999.0}
    for condition in ["answer_mask", "union_mask"]:
        for top_k in ["1", "4", "8", "16", "32", "64"]:
            restore_controls = _best(spec, pack, "restore_source_minus_controls", condition, top_k)
            real_shifted = _best(spec, pack, "restore_real_minus_shifted", condition, top_k)
            real_shuffled = _best(spec, pack, "restore_real_minus_shuffled", condition, top_k)
            restore_wrong = _best(spec, pack, "restore_correct_minus_wrong", condition, top_k)
            restore_rank = _best(spec, pack, "restore_target_rank_effect", condition, top_k)
            passed = all(_positive(row) for row in [restore_controls, real_shifted, real_shuffled, restore_wrong]) and float(restore_rank.get("mean", 0.0)) > 0
            score = sum(float(row.get("ci95_low", 0.0)) for row in [restore_controls, real_shifted, real_shuffled, restore_wrong])
            if passed or score > float(best_restore.get("score", -999999.0)):
                best_restore = {
                    "condition": condition,
                    "top_k": top_k,
                    "passed": passed,
                    "score": score,
                    "restore_source_minus_controls": restore_controls,
                    "restore_real_minus_shifted": real_shifted,
                    "restore_real_minus_shuffled": real_shuffled,
                    "restore_correct_minus_wrong": restore_wrong,
                    "restore_target_rank_effect": restore_rank,
                }
            if passed:
                break
        if best_restore.get("passed"):
            break
    return {
        "clean_zeroing_source_minus_controls": zero_controls,
        "clean_zeroing_correct_minus_wrong": zero_wrong,
        "clean_zeroing_target_rank_effect": zero_rank,
        "clean_zeroing_passed": _positive(zero_controls) and _positive(zero_wrong) and float(zero_rank.get("mean", 0.0)) > 0,
        "restore_gate": best_restore,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    summary: list[dict[str, Any]] = []
    spec: list[dict[str, Any]] = []
    for pack in ["primary", "strict"]:
        prefix = _prefix(pack, args.mode, args.layer, args.tag)
        manifest = _read_csv(args.cross_dir / f"{prefix}_manifest.csv")
        zeroing = _read_csv(args.cross_dir / f"{prefix}_zeroing_raw.csv")
        group = _read_csv(args.cross_dir / f"{prefix}_group_raw.csv")
        if manifest:
            summary.extend(_candidate_summary(manifest, pack))
        if zeroing:
            spec.extend(_zeroing_specificity(zeroing, pack))
        if group:
            spec.extend(_group_specificity(group, pack))
    primary_candidates = next((row for row in summary if row.get("record_type") == "candidate_manifest" and row.get("pack") == "primary" and row.get("group") == "all"), {})
    strict_candidates = next((row for row in summary if row.get("record_type") == "candidate_manifest" and row.get("pack") == "strict" and row.get("group") == "all"), {})
    primary_gate = _gate(spec, "primary")
    strict_gate = _gate(spec, "strict")
    if args.mode == "smoke":
        status = "smoke_completed_not_decisive" if summary or spec else "blocked"
    elif int(primary_candidates.get("main_rows", 0) or 0) < args.min_candidates:
        status = "qwen_hidden_route_supported_plt_unresolved"
    elif primary_gate["clean_zeroing_passed"] and primary_gate["restore_gate"].get("passed") and strict_gate["clean_zeroing_passed"] and strict_gate["restore_gate"].get("passed"):
        status = "qwen_adapter_v4_route_supported"
    elif summary or spec:
        status = "qwen_not_gemma_style_under_v4_adapter"
    else:
        status = "blocked"
    decision = {
        "created_at": _now(),
        "status": status,
        "mode": args.mode,
        "layer": args.layer,
        "tag": args.tag,
        "candidate_counts": {
            "primary": primary_candidates,
            "strict": strict_candidates,
        },
        "gates": {
            "primary": primary_gate,
            "strict": strict_gate,
        },
        "claim_boundary": "Adapter V4 uses Qwen layer-14 artifacts only. Failure is not evidence that Qwen lacks cross-modal mechanisms.",
    }
    _write_csv(
        args.summary_csv,
        summary,
        [
            "record_type",
            "pack",
            "group",
            "rows",
            "main_rows",
            "prompt_runs",
            "mean_adapter_v4_score",
            "mean_zeroing_damage",
            "exact_match_rows",
            "same_feature_rows",
        ],
    )
    _write_csv(
        args.specificity_csv,
        spec,
        ["metric", "pack", "n", "mean", "ci95_low", "ci95_high", "positive_frac", "mask_condition", "top_k"],
    )
    _write_json(args.decision_json, decision)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage4 Qwen Adapter V4 route validation.")
    parser.add_argument("--cross-dir", type=Path, default=CROSS)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--tag", default="layer14")
    parser.add_argument("--min-candidates", type=int, default=20)
    parser.add_argument("--summary-csv", type=Path, default=CROSS / "stage4_qwen_adapter_v4_summary.csv")
    parser.add_argument("--specificity-csv", type=Path, default=CROSS / "stage4_qwen_adapter_v4_specificity.csv")
    parser.add_argument("--decision-json", type=Path, default=CROSS / "stage4_qwen_adapter_v4_decision.json")
    args = parser.parse_args()
    decision = analyze(args)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
