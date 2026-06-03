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


def _f(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _ci(values: list[float], seed: int = 20260526, n: int = 2000) -> tuple[float, float]:
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


def _metric(metric: str, pack: str, values: list[float], **extra: Any) -> dict[str, Any]:
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
    row.update(extra)
    return row


def _prefix(pack: str) -> str:
    return f"stage4_qwen_adapter_v4_{pack}_full_L14_layer14"


def _candidate_summary(pack: str, manifest: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for level in ["all", "exact_pos_feature", "same_feature"]:
        rows = manifest if level == "all" else [row for row in manifest if row.get("v4_match_level") == level]
        out.append(
            {
                "record_type": "candidate_summary",
                "pack": pack,
                "slice": level,
                "rows": len(rows),
                "main_rows": sum(row.get("include_main") == "1" for row in rows),
                "prompt_runs": len({(row.get("sample_id"), row.get("prompt_name")) for row in rows}),
                "mean_adapter_v4_score": _mean([_f(row.get("adapter_v4_score")) for row in rows]),
                "mean_zeroing_damage": _mean([_f(row.get("zeroing_damage")) for row in rows]),
            }
        )
    return out


def _zeroing_metrics(pack: str, zeroing: list[dict[str, str]], candidate_ids: set[str] | None = None, slice_name: str = "all") -> list[dict[str, Any]]:
    target = [
        row for row in zeroing
        if row.get("intervention_kind") == "clean_zeroing"
        and row.get("token_scored") == "target"
        and row.get("status", "ok") in {"", "ok"}
        and (candidate_ids is None or row.get("candidate_id") in candidate_ids)
    ]
    wrong = [
        row for row in zeroing
        if row.get("intervention_kind") == "clean_zeroing"
        and row.get("token_scored") == "wrong"
        and row.get("status", "ok") in {"", "ok"}
        and (candidate_ids is None or row.get("candidate_id") in candidate_ids)
    ]
    by_candidate: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in target:
        by_candidate[row.get("candidate_id", "")][row.get("control_group", "")] = row
    wrong_by_candidate = {row.get("candidate_id", ""): row for row in wrong if row.get("control_group") == "source"}
    source_minus_controls = []
    correct_minus_wrong = []
    rank_effects = []
    for candidate_id, groups in by_candidate.items():
        source = groups.get("source")
        controls = [groups[name] for name in CONTROL_GROUPS if name in groups]
        if source and controls:
            source_minus_controls.append(_f(source.get("logit_effect")) - _mean([_f(row.get("logit_effect")) for row in controls]))
            rank_effects.append(_f(source.get("rank_effect")))
        if source and candidate_id in wrong_by_candidate:
            correct_minus_wrong.append(_f(source.get("logit_effect")) - _f(wrong_by_candidate[candidate_id].get("logit_effect")))
    return [
        _metric("clean_zeroing_source_minus_controls", pack, source_minus_controls, slice=slice_name),
        _metric("clean_zeroing_correct_minus_wrong", pack, correct_minus_wrong, slice=slice_name),
        _metric("clean_zeroing_rank_effect", pack, rank_effects, slice=slice_name),
    ]


def _group_metrics(pack: str, group_rows: list[dict[str, str]], candidate_ids: set[str] | None = None, slice_name: str = "all") -> list[dict[str, Any]]:
    if candidate_ids is not None:
        def _row_intersects(row: dict[str, str]) -> bool:
            return bool(set(row.get("candidate_ids", "").split("|")) & candidate_ids)
        group_rows = [row for row in group_rows if _row_intersects(row)]
    restore = [row for row in group_rows if row.get("record_type") == "group_restore"]
    drops = [row for row in group_rows if row.get("record_type") == "activation_drop"]
    key = lambda row: (row.get("sample_id", "") + "::" + row.get("prompt_name", ""), row.get("mask_condition", ""), row.get("control_group", ""), row.get("top_k", ""))
    restore_idx = {key(row): row for row in restore}
    drop_idx = {key(row): row for row in drops}
    run_ids = sorted({row.get("sample_id", "") + "::" + row.get("prompt_name", "") for row in group_rows})
    topks = sorted({row.get("top_k", "") for row in group_rows if row.get("top_k", "")}, key=lambda raw: int(float(raw)))
    out: list[dict[str, Any]] = []
    for top_k in topks:
        for condition in ["answer_mask", "union_mask"]:
            restore_controls = []
            restore_shifted = []
            restore_shuffled = []
            restore_wrong = []
            restore_rank = []
            drop_shifted = []
            drop_shuffled = []
            for run_id in run_ids:
                source = restore_idx.get((run_id, condition, "source", top_k))
                controls = [restore_idx[(run_id, condition, name, top_k)] for name in CONTROL_GROUPS if (run_id, condition, name, top_k) in restore_idx]
                shifted = restore_idx.get((run_id, "shifted_mask", "source", top_k))
                shuffled = restore_idx.get((run_id, "shuffled_mask", "source", top_k))
                if source and controls:
                    restore_controls.append(_f(source.get("target_logit_effect")) - _mean([_f(row.get("target_logit_effect")) for row in controls]))
                    restore_wrong.append(_f(source.get("target_logit_effect")) - _f(source.get("wrong_logit_effect")))
                    restore_rank.append(_f(source.get("target_rank_effect")))
                if source and shifted:
                    restore_shifted.append(_f(source.get("target_logit_effect")) - _f(shifted.get("target_logit_effect")))
                if source and shuffled:
                    restore_shuffled.append(_f(source.get("target_logit_effect")) - _f(shuffled.get("target_logit_effect")))
                source_drop = drop_idx.get((run_id, condition, "source", top_k))
                shifted_drop = drop_idx.get((run_id, "shifted_mask", "source", top_k))
                shuffled_drop = drop_idx.get((run_id, "shuffled_mask", "source", top_k))
                if source_drop and shifted_drop:
                    drop_shifted.append(_f(source_drop.get("activation_drop_sum")) - _f(shifted_drop.get("activation_drop_sum")))
                if source_drop and shuffled_drop:
                    drop_shuffled.append(_f(source_drop.get("activation_drop_sum")) - _f(shuffled_drop.get("activation_drop_sum")))
            extra = {"slice": slice_name, "mask_condition": condition, "top_k": top_k}
            out.extend(
                [
                    _metric("activation_drop_real_minus_shifted", pack, drop_shifted, **extra),
                    _metric("activation_drop_real_minus_shuffled", pack, drop_shuffled, **extra),
                    _metric("restore_source_minus_controls", pack, restore_controls, **extra),
                    _metric("restore_real_minus_shifted", pack, restore_shifted, **extra),
                    _metric("restore_real_minus_shuffled", pack, restore_shuffled, **extra),
                    _metric("restore_correct_minus_wrong", pack, restore_wrong, **extra),
                    _metric("restore_rank_effect", pack, restore_rank, **extra),
                ]
            )
    return out


def _positive(row: dict[str, Any], min_n: int = 8, min_frac: float = 0.55) -> bool:
    return int(row.get("n", 0)) >= min_n and float(row.get("ci95_low", 0.0)) > 0 and float(row.get("positive_frac", 0.0)) >= min_frac


def _best(rows: list[dict[str, Any]], metric: str, pack: str, slice_name: str = "all") -> dict[str, Any]:
    subset = [row for row in rows if row.get("metric") == metric and row.get("pack") == pack and row.get("slice") == slice_name]
    if not subset:
        return {"n": 0, "mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "positive_frac": 0.0}
    return max(subset, key=lambda row: (float(row.get("ci95_low", 0.0)), float(row.get("mean", 0.0))))


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for pack in ["primary", "strict"]:
        prefix = _prefix(pack)
        manifest = _read_csv(args.cross_dir / f"{prefix}_manifest.csv")
        zeroing = _read_csv(args.cross_dir / f"{prefix}_zeroing_raw.csv")
        group = _read_csv(args.cross_dir / f"{prefix}_group_raw.csv")
        summaries.extend(_candidate_summary(pack, manifest))
        for slice_name, ids in [
            ("all", None),
            ("exact_pos_feature", {row.get("candidate_id", "") for row in manifest if row.get("v4_match_level") == "exact_pos_feature"}),
            ("same_feature", {row.get("candidate_id", "") for row in manifest if row.get("v4_match_level") == "same_feature"}),
        ]:
            metrics.extend(_zeroing_metrics(pack, zeroing, ids, slice_name))
            metrics.extend(_group_metrics(pack, group, ids, slice_name))

    primary_activation = _best(metrics, "activation_drop_real_minus_shifted", "primary")
    strict_activation = _best(metrics, "activation_drop_real_minus_shifted", "strict")
    primary_zeroing = _best(metrics, "clean_zeroing_source_minus_controls", "primary")
    strict_zeroing = _best(metrics, "clean_zeroing_source_minus_controls", "strict")
    primary_restore = _best(metrics, "restore_real_minus_shifted", "primary")
    strict_restore = _best(metrics, "restore_real_minus_shifted", "strict")
    primary_exact = next((row for row in summaries if row["pack"] == "primary" and row["slice"] == "exact_pos_feature"), {})
    strict_exact = next((row for row in summaries if row["pack"] == "strict" and row["slice"] == "exact_pos_feature"), {})

    activation_ok = _positive(primary_activation) and _positive(strict_activation)
    zeroing_ok = _positive(primary_zeroing) and _positive(strict_zeroing)
    restore_ok = _positive(primary_restore) and _positive(strict_restore)
    exact_sparse = int(primary_exact.get("main_rows", 0) or 0) < args.min_exact_candidates or int(strict_exact.get("main_rows", 0) or 0) < args.min_exact_candidates

    if activation_ok and zeroing_ok and not restore_ok:
        failure_type = "operator_or_feature_not_sufficient"
    elif exact_sparse:
        failure_type = "exact_position_candidate_sparse"
    elif not activation_ok:
        failure_type = "evidence_activation_not_specific"
    elif not zeroing_ok:
        failure_type = "answer_support_zeroing_not_specific"
    else:
        failure_type = "mixed_or_unresolved"

    decision = {
        "created_at": _now(),
        "status": "qwen_native_route_decomposition_complete",
        "failure_type": failure_type,
        "key_gates": {
            "activation_specificity_primary": primary_activation,
            "activation_specificity_strict": strict_activation,
            "clean_zeroing_primary": primary_zeroing,
            "clean_zeroing_strict": strict_zeroing,
            "restore_specificity_primary": primary_restore,
            "restore_specificity_strict": strict_restore,
            "exact_position_primary": primary_exact,
            "exact_position_strict": strict_exact,
        },
        "claim_boundary": "This decomposition explains current Adapter V4 failure. It is not a negative claim about Qwen mechanisms.",
    }
    _write_csv(
        args.summary_csv,
        summaries,
        ["record_type", "pack", "slice", "rows", "main_rows", "prompt_runs", "mean_adapter_v4_score", "mean_zeroing_damage"],
    )
    _write_csv(
        args.metrics_csv,
        metrics,
        ["metric", "pack", "slice", "n", "mean", "ci95_low", "ci95_high", "positive_frac", "mask_condition", "top_k"],
    )
    _write_json(args.decision_json, decision)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage4-038 Qwen native route decomposition.")
    parser.add_argument("--cross-dir", type=Path, default=CROSS)
    parser.add_argument("--min-exact-candidates", type=int, default=20)
    parser.add_argument("--summary-csv", type=Path, default=CROSS / "stage4_qwen_native_route_v4_failure_decomposition_summary.csv")
    parser.add_argument("--metrics-csv", type=Path, default=CROSS / "stage4_qwen_native_route_v4_failure_decomposition_metrics.csv")
    parser.add_argument("--decision-json", type=Path, default=CROSS / "stage4_qwen_native_route_v4_failure_decomposition_decision.json")
    args = parser.parse_args()
    decision = analyze(args)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
