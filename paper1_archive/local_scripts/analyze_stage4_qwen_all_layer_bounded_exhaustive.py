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
PREFIX = "stage4_qwen_all_layer_bounded_exhaustive"
CONTROL_GROUPS = {
    "same_position_matched_feature_control",
    "same_feature_random_position_control",
    "random_active_feature_control",
}


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


def _ci(values: list[float], n: int = 2000, seed: int = 44044) -> tuple[float, float]:
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


def _metric(metric: str, values: list[float], extra: dict[str, Any]) -> dict[str, Any]:
    lo, hi = _ci(values)
    row = {
        "metric": metric,
        "n": len(values),
        "mean": _mean(values),
        "ci95_low": lo,
        "ci95_high": hi,
        "positive_frac": sum(1 for value in values if value > 0) / len(values) if values else 0.0,
    }
    row.update(extra)
    return row


def _positive(row: dict[str, Any], min_n: int = 12) -> bool:
    return int(row.get("n", 0)) >= min_n and float(row.get("ci95_low", 0.0)) > 0 and float(row.get("positive_frac", 0.0)) >= 0.6


def _files(mode: str) -> list[tuple[str, int, Path, Path, Path]]:
    out = []
    for cand in CROSS.glob(f"{PREFIX}_*_{mode}_L*_candidates.csv"):
        rest = cand.name[len(PREFIX) + 1 : -len("_candidates.csv")]
        pack_part, _mode_part, layer_part = rest.rsplit("_", 2)
        layer = int(layer_part.removeprefix("L"))
        zero = CROSS / f"{PREFIX}_{pack_part}_{mode}_L{layer}_zeroing_raw.csv"
        group = CROSS / f"{PREFIX}_{pack_part}_{mode}_L{layer}_group_raw.csv"
        out.append((pack_part, layer, cand, zero, group))
    return sorted(out, key=lambda item: (item[0], item[1]))


def _candidate_summary(pack: str, layer: int, rows: list[dict[str, str]]) -> dict[str, Any]:
    exact = [row for row in rows if row.get("position_group") in {"visual_answer", "exact"}]
    return {
        "stage": "candidate_screen",
        "metric": "candidate_manifest",
        "pack": pack,
        "layer": layer,
        "n": len(rows),
        "prompt_runs": len({(row.get("sample_id", ""), row.get("prompt_name", "")) for row in rows}),
        "main_rows": sum(1 for row in rows if row.get("include_main") == "1"),
        "exact_or_visual_answer_rows": len(exact),
        "mean": _mean([_f(row, "evidence_first_score") for row in rows]),
        "ci95_low": "",
        "ci95_high": "",
        "positive_frac": "",
    }


def _zeroing_metrics(pack: str, layer: int, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    target = [row for row in rows if row.get("status", "ok") in {"", "ok"} and row.get("intervention_kind") == "clean_zeroing" and row.get("token_scored") == "target"]
    wrong = [row for row in rows if row.get("status", "ok") in {"", "ok"} and row.get("intervention_kind") == "clean_zeroing" and row.get("token_scored") == "wrong"]
    by_candidate: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in target:
        by_candidate[row.get("candidate_id", "")][row.get("control_group", "")] = row
    wrong_source = {row.get("candidate_id", ""): row for row in wrong if row.get("control_group") == "source"}
    source_minus_controls: list[float] = []
    correct_minus_wrong: list[float] = []
    source_rank: list[float] = []
    for candidate_id, groups in by_candidate.items():
        source = groups.get("source")
        controls = [groups[name] for name in CONTROL_GROUPS if name in groups]
        if source and controls:
            source_minus_controls.append(_f(source, "logit_effect") - _mean([_f(row, "logit_effect") for row in controls]))
        if source and candidate_id in wrong_source:
            correct_minus_wrong.append(_f(source, "logit_effect") - _f(wrong_source[candidate_id], "logit_effect"))
        if source:
            source_rank.append(_f(source, "rank_effect"))
    extra = {"stage": "zeroing", "pack": pack, "layer": layer, "mask_condition": "clean", "top_k": ""}
    return [
        _metric("zeroing_source_minus_controls", source_minus_controls, extra),
        _metric("zeroing_correct_minus_wrong", correct_minus_wrong, extra),
        _metric("zeroing_source_rank_effect", source_rank, extra),
    ]


def _group_metrics(pack: str, layer: int, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("record_type") == "activation_drop":
            continue
        grouped[(row.get("top_k", ""), row.get("mask_condition", ""), row.get("control_group", ""))].append(row)
    out: list[dict[str, Any]] = []
    topks = sorted({key[0] for key in grouped}, key=lambda raw: int(float(raw or 0)))
    for topk in topks:
        for real in ["answer_mask", "union_mask"]:
            source = grouped.get((topk, real, "source"), [])
            shifted = grouped.get((topk, "shifted_mask", "source"), [])
            shuffled = grouped.get((topk, "shuffled_mask", "source"), [])
            controls = [row for control in CONTROL_GROUPS for row in grouped.get((topk, real, control), [])]
            by_run = lambda row: (row.get("sample_id", ""), row.get("prompt_name", ""))
            shifted_idx = {by_run(row): row for row in shifted}
            shuffled_idx = {by_run(row): row for row in shuffled}
            control_idx: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
            for row in controls:
                control_idx[by_run(row)].append(row)
            source_effect = [_f(row, "target_logit_effect") for row in source]
            real_shifted = [_f(row, "target_logit_effect") - _f(shifted_idx[by_run(row)], "target_logit_effect") for row in source if by_run(row) in shifted_idx]
            real_shuffled = [_f(row, "target_logit_effect") - _f(shuffled_idx[by_run(row)], "target_logit_effect") for row in source if by_run(row) in shuffled_idx]
            source_controls = [
                _f(row, "target_logit_effect") - _mean([_f(control, "target_logit_effect") for control in control_idx[by_run(row)]])
                for row in source
                if control_idx.get(by_run(row))
            ]
            rank_effect = [_f(row, "target_rank_effect") for row in source]
            extra = {"stage": "grouped_restore", "pack": pack, "layer": layer, "mask_condition": real, "top_k": topk}
            out.append(_metric("restore_source_effect", source_effect, extra))
            out.append(_metric("restore_source_minus_controls", source_controls, extra))
            out.append(_metric("restore_real_minus_shifted", real_shifted, extra))
            out.append(_metric("restore_real_minus_shuffled", real_shuffled, extra))
            out.append(_metric("restore_rank_effect", rank_effect, extra))
    return out


def _load_stage4_038_context() -> dict[str, Any]:
    decision = CROSS / "stage4_qwen_native_route_hidden_to_plt_decision.json"
    metrics = CROSS / "stage4_qwen_native_route_hidden_to_plt_metrics.csv"
    return {
        "stage4_038_decision": json.loads(decision.read_text(encoding="utf-8")) if decision.exists() else {},
        "stage4_038_metrics_rows": len(_read_csv(metrics)),
    }


def analyze(mode: str) -> dict[str, Any]:
    summary: list[dict[str, Any]] = []
    specificity: list[dict[str, Any]] = []
    for pack, layer, cand_path, zero_path, group_path in _files(mode):
        candidates = _read_csv(cand_path)
        zero = _read_csv(zero_path)
        group = _read_csv(group_path)
        summary.append(_candidate_summary(pack, layer, candidates))
        summary.append({"stage": "zeroing", "metric": "raw_rows", "pack": pack, "layer": layer, "n": len(zero), "mean": "", "ci95_low": "", "ci95_high": "", "positive_frac": ""})
        summary.append({"stage": "grouped_restore", "metric": "raw_rows", "pack": pack, "layer": layer, "n": len(group), "mean": "", "ci95_low": "", "ci95_high": "", "positive_frac": ""})
        specificity.extend(_zeroing_metrics(pack, layer, zero))
        specificity.extend(_group_metrics(pack, layer, group))
    summary_fields = sorted({key for row in summary for key in row.keys()})
    spec_fields = sorted({key for row in specificity for key in row.keys()})
    _write_csv(CROSS / f"{PREFIX}_summary.csv", summary, summary_fields)
    _write_csv(CROSS / f"{PREFIX}_specificity.csv", specificity, spec_fields)

    primary_sparse = [
        row for row in specificity
        if row.get("pack") == "primary"
        and row.get("metric") in {"zeroing_source_minus_controls", "restore_source_minus_controls"}
        and _positive(row)
    ]
    primary_mask = [
        row for row in specificity
        if row.get("pack") == "primary"
        and row.get("metric") in {"restore_real_minus_shifted", "restore_real_minus_shuffled"}
        and _positive(row)
    ]
    strict_sparse = [
        row for row in specificity
        if row.get("pack") == "strict"
        and row.get("metric") in {"zeroing_source_minus_controls", "restore_source_minus_controls"}
        and _positive(row)
    ]
    strict_mask = [
        row for row in specificity
        if row.get("pack") == "strict"
        and row.get("metric") in {"restore_real_minus_shifted", "restore_real_minus_shuffled"}
        and _positive(row)
    ]
    topk_pass = [
        row for row in specificity
        if row.get("metric") == "restore_source_minus_controls"
        and str(row.get("top_k", "")) not in {"", "1"}
        and _positive(row)
    ]

    status = "blocked"
    if summary:
        status = "qwen_sparse_plt_route_not_supported_under_bounded_exhaustive"
    if primary_sparse and primary_mask and strict_sparse and strict_mask:
        status = "qwen_sparse_plt_route_supported"
    elif topk_pass and primary_mask:
        status = "qwen_distributed_plt_route_supported"
    elif summary and _load_stage4_038_context().get("stage4_038_decision", {}).get("status") == "qwen_route_may_live_in_plt_error":
        status = "qwen_route_may_live_in_plt_error_confirmed"

    decision = {
        "status": status,
        "mode": mode,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "layers_seen": sorted({int(row["layer"]) for row in summary if row.get("layer", "") != ""}),
        "primary_positive_sparse_metrics": len(primary_sparse),
        "primary_positive_mask_metrics": len(primary_mask),
        "strict_positive_sparse_metrics": len(strict_sparse),
        "strict_positive_mask_metrics": len(strict_mask),
        "topk_positive_restore_metrics": len(topk_pass),
        "stage4_038_context": _load_stage4_038_context(),
        "claim_boundary": "Bounded exhaustive over active Qwen-PLT candidates and top causal validations; not literal all-node forward intervention.",
    }
    _write_json(CROSS / f"{PREFIX}_decision.json", decision)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage4-044 Qwen all-layer bounded exhaustive PLT route search.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    print(json.dumps(analyze(args.mode), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
