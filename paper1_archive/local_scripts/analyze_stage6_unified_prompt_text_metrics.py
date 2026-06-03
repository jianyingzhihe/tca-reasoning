#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
STAGE6_CROSS = ROOT / "doc" / "experiments" / "stage6" / "cross_model"
PREFIX = "stage6_unified"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _f(raw: Any, default: float = 0.0) -> float:
    try:
        value = float(raw) if raw not in (None, "") else default
        if math.isnan(value):
            return default
        return value
    except ValueError:
        return default


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _ci_low_sem(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    mean = _mean(values)
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean - 1.96 * math.sqrt(var / len(values))


def _agg(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": _mean(values),
        "ci_low_sem": _ci_low_sem(values),
        "positive_frac": sum(1 for value in values if value > 0) / len(values) if values else 0.0,
    }


def _summary(rows: list[dict[str, Any]], keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key, "")) for key in keys)].append(row)
    out = []
    for key, part in sorted(grouped.items()):
        item = {name: value for name, value in zip(keys, key, strict=False)}
        item["rows"] = len(part)
        item["unique_samples"] = len({row.get("stage6_original_sample_id") or row.get("sample_id", "") for row in part})
        for metric in metrics:
            stat = _agg([_f(row.get(metric)) for row in part if row.get(metric) not in (None, "")])
            item[f"{metric}_mean"] = stat["mean"]
            item[f"{metric}_ci_low_sem"] = stat["ci_low_sem"]
            item[f"{metric}_positive_frac"] = stat["positive_frac"]
        out.append(item)
    return out


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _gemma_fixednode_coverage_rows(
    planned_rows: list[dict[str, str]],
    raw_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    raw_by_candidate = {row.get("candidate_id", ""): row for row in raw_rows if row.get("candidate_id")}
    out = []
    for planned in planned_rows:
        candidate_id = planned.get("candidate_id", "")
        raw = raw_by_candidate.get(candidate_id, {})
        prefix_ok = planned.get("prefix_ok") == "1"
        target_same = planned.get("target_token_same") == "1"
        aligned_planned = prefix_ok and target_same
        attempted = bool(raw)
        status = raw.get("status", "not_attempted")
        error_message = raw.get("error_message", "")
        out_of_range = "source_pos_out_of_range" in error_message
        usable_exact = aligned_planned and status == "ok"
        out.append(
            {
                "candidate_id": candidate_id,
                "sample_id": planned.get("sample_id", ""),
                "stage6_original_sample_id": planned.get("stage6_original_sample_id", ""),
                "stage6_question_variant": planned.get("stage6_question_variant", ""),
                "stage6_prompt_family": planned.get("stage6_prompt_family", ""),
                "stage6_sample_type": planned.get("stage6_sample_type", ""),
                "source_layer": planned.get("source_layer", ""),
                "source_pos": planned.get("source_pos", ""),
                "source_feature_id": planned.get("source_feature_id", ""),
                "planned": 1,
                "prefix_ok": 1 if prefix_ok else 0,
                "target_token_same": 1 if target_same else 0,
                "aligned_planned": 1 if aligned_planned else 0,
                "attempted": 1 if attempted else 0,
                "usable_exact": 1 if usable_exact else 0,
                "skipped_unaligned": 1 if status == "skipped_unaligned" else 0,
                "error": 1 if status == "error" else 0,
                "source_pos_out_of_range": 1 if out_of_range else 0,
                "not_attempted": 1 if not attempted else 0,
                "status": status,
                "error_message": error_message,
            }
        )
    return out


def _gemma_fixednode_coverage_summary(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key, "")) for key in keys)].append(row)
    out = []
    for key, part in sorted(grouped.items()):
        planned = len(part)
        aligned = sum(int(row.get("aligned_planned", 0)) for row in part)
        attempted = sum(int(row.get("attempted", 0)) for row in part)
        usable = sum(int(row.get("usable_exact", 0)) for row in part)
        skipped = sum(int(row.get("skipped_unaligned", 0)) for row in part)
        errors = sum(int(row.get("error", 0)) for row in part)
        out_of_range = sum(int(row.get("source_pos_out_of_range", 0)) for row in part)
        item = {name: value for name, value in zip(keys, key, strict=False)}
        item.update(
            {
                "planned_rows": planned,
                "planned_aligned_rows": aligned,
                "attempted_rows": attempted,
                "usable_exact_rows": usable,
                "skipped_unaligned_rows": skipped,
                "error_rows": errors,
                "source_pos_out_of_range_rows": out_of_range,
                "not_attempted_rows": planned - attempted,
                "usable_exact_over_planned": _fraction(usable, planned),
                "usable_exact_over_planned_aligned": _fraction(usable, aligned),
                "out_of_range_over_attempted": _fraction(out_of_range, attempted),
                "attempted_over_planned": _fraction(attempted, planned),
                "unique_samples": len({row.get("stage6_original_sample_id", "") for row in part}),
            }
        )
        out.append(item)
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _old_qwen_reanalysis(tag: str) -> dict[str, Any]:
    path = STAGE6_CROSS / "stage6_prompt_text_cot_full_prompttext_v1_candidate_metrics.csv"
    rows = _read_csv(path)
    out = []
    for row in rows:
        pos234 = (
            _f(row.get("restore_source_minus_controls")) > 0
            and _f(row.get("real_minus_shifted")) > 0
            and _f(row.get("real_minus_shuffled")) > 0
        )
        item = dict(row)
        item["pos234"] = "1" if pos234 else "0"
        out.append(item)
    metrics = [
        "restore_source_minus_controls",
        "real_minus_shifted",
        "real_minus_shuffled",
        "restore_correct_minus_wrong",
        "clean_source_minus_controls",
        "pos234_numeric",
    ]
    for row in out:
        row["pos234_numeric"] = 1.0 if row.get("pos234") == "1" else 0.0
    by_prompt = _summary(out, ["stage6_prompt_family"], metrics)
    by_variant = _summary(out, ["stage6_question_variant"], metrics)
    by_prompt_variant = _summary(out, ["stage6_prompt_family", "stage6_question_variant"], metrics)
    _write_csv(STAGE6_CROSS / f"{PREFIX}_crossmodel_{tag}_old_qwen_summary_by_prompt.csv", by_prompt)
    _write_csv(STAGE6_CROSS / f"{PREFIX}_crossmodel_{tag}_old_qwen_summary_by_variant.csv", by_variant)
    _write_csv(STAGE6_CROSS / f"{PREFIX}_crossmodel_{tag}_old_qwen_summary_by_prompt_variant.csv", by_prompt_variant)
    return {
        "source": str(path),
        "rows": len(rows),
        "summary_by_prompt": str(STAGE6_CROSS / f"{PREFIX}_crossmodel_{tag}_old_qwen_summary_by_prompt.csv"),
        "interpretation": "Old Qwen Stage6 reanalysis reports fixed-node causal metrics and pos234; evidence_specificity is not used as the primary stability claim.",
    }


def _old_gemma_reanalysis(tag: str, source_tag: str) -> dict[str, Any]:
    prefix = f"stage6_gemma_prompt_text_cot_full_{source_tag}"
    compare = _read_csv(STAGE6_CROSS / f"{prefix}_sample_compare_controlled.csv")
    meta_a = {row["sample_id"]: row for row in _read_csv(STAGE6_CROSS / f"{prefix}_meta_a.csv") if row.get("sample_id")}
    rows = []
    for row in compare:
        sample_id = row.get("sample_id", "")
        meta = meta_a.get(sample_id, {})
        item = dict(row)
        item["prefix_ok"] = "1" if (meta.get("assistant_prefix") or "").strip() else "0"
        item["format_ok"] = "1" if "the answer is" in (meta.get("generated_text") or "").lower() else "0"
        item["target_token_same"] = "1" if row.get("a_target_token_id") and row.get("a_target_token_id") == row.get("b_target_token_id") else "0"
        item["stage6_original_sample_id"] = sample_id.split("__", 1)[0]
        parts = sample_id.split("__")
        item["stage6_question_variant"] = parts[1] if len(parts) > 1 else ""
        item["stage6_prompt_family"] = parts[2] if len(parts) > 2 else ""
        item["aligned_main"] = "1" if item["prefix_ok"] == "1" and item["target_token_same"] == "1" else "0"
        rows.append(item)
    aligned = [row for row in rows if row["aligned_main"] == "1"]
    metrics = [
        "node_overlap_jaccard",
        "edge_overlap_jaccard",
        "delta_target_total_in_abs",
        "delta_target_feature_ratio",
        "delta_target_error_ratio",
    ]
    by_prompt_all = _summary(rows, ["stage6_prompt_family"], metrics)
    by_prompt_aligned = _summary(aligned, ["stage6_prompt_family"], metrics)
    by_variant_aligned = _summary(aligned, ["stage6_question_variant"], metrics)
    _write_csv(STAGE6_CROSS / f"{PREFIX}_crossmodel_{tag}_old_gemma_alignment_rows.csv", rows)
    _write_csv(STAGE6_CROSS / f"{PREFIX}_crossmodel_{tag}_old_gemma_summary_by_prompt_all.csv", by_prompt_all)
    _write_csv(STAGE6_CROSS / f"{PREFIX}_crossmodel_{tag}_old_gemma_summary_by_prompt_aligned.csv", by_prompt_aligned)
    _write_csv(STAGE6_CROSS / f"{PREFIX}_crossmodel_{tag}_old_gemma_summary_by_variant_aligned.csv", by_variant_aligned)
    return {
        "source_prefix": prefix,
        "rows": len(rows),
        "aligned_rows": len(aligned),
        "prefix_fail_rows": sum(1 for row in rows if row["prefix_ok"] != "1"),
        "target_mismatch_rows": sum(1 for row in rows if row["target_token_same"] != "1"),
        "summary_by_prompt_aligned": str(STAGE6_CROSS / f"{PREFIX}_crossmodel_{tag}_old_gemma_summary_by_prompt_aligned.csv"),
        "interpretation": "Old Gemma graph-overlap results are split into aligned and format/target-token failure rows before any mechanism interpretation.",
    }


def _source_control_metric(rows: list[dict[str, str]], intervention: str, condition: str, token: str) -> float:
    matched = [
        row
        for row in rows
        if row.get("intervention_kind") == intervention
        and row.get("mask_condition") == condition
        and row.get("token_scored") == token
    ]
    source = [_f(row.get("logit_effect")) for row in matched if row.get("control_group") == "source"]
    controls = [_f(row.get("logit_effect")) for row in matched if row.get("control_group") != "source"]
    return (source[0] if source else 0.0) - _mean(controls)


def _source_metric(rows: list[dict[str, str]], intervention: str, condition: str, token: str, field: str) -> float:
    for row in rows:
        if (
            row.get("intervention_kind") == intervention
            and row.get("mask_condition") == condition
            and row.get("token_scored") == token
            and row.get("control_group") == "source"
        ):
            return _f(row.get(field))
    return 0.0


def _qwen_routeidentity_metrics(tag: str, mode: str) -> list[dict[str, Any]]:
    manifest_path = STAGE6_CROSS / f"{PREFIX}_routeidentity_{tag}_manifest.csv"
    manifest = {row["candidate_id"]: row for row in _read_csv(manifest_path) if row.get("candidate_id")}
    raw_rows: list[dict[str, str]] = []
    for path in sorted(STAGE6_CROSS.glob(f"{PREFIX}_routeidentity_{mode}_L*_{tag}_raw.csv")):
        raw_rows.extend(_read_csv(path))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        grouped[row.get("candidate_id", "")].append(row)

    out = []
    for cid, rows in grouped.items():
        meta = manifest.get(cid, {})
        answer_restore = _source_control_metric(rows, "mask_restore", "answer_mask", "target")
        union_restore = _source_control_metric(rows, "mask_restore", "union_mask", "target")
        real = "answer_mask" if answer_restore >= union_restore else "union_mask"
        real_restore = max(answer_restore, union_restore)
        shifted = _source_control_metric(rows, "mask_restore", "shifted_mask", "target")
        shuffled = _source_control_metric(rows, "mask_restore", "shuffled_mask", "target")
        clean = _source_control_metric(rows, "clean_zeroing", "clean", "target")
        restore_wrong = _source_control_metric(rows, "mask_restore", real, "wrong")
        answer_drop = _source_metric(rows, "mask_restore", "answer_mask", "target", "activation_drop")
        union_drop = _source_metric(rows, "mask_restore", "union_mask", "target", "activation_drop")
        shifted_drop = _source_metric(rows, "mask_restore", "shifted_mask", "target", "activation_drop")
        shuffled_drop = _source_metric(rows, "mask_restore", "shuffled_mask", "target", "activation_drop")
        evidence_specificity = max(answer_drop, union_drop) - max(shifted_drop, shuffled_drop)
        route_score = (
            max(0.0, clean)
            + max(0.0, real_restore)
            + max(0.0, real_restore - shifted)
            + max(0.0, real_restore - shuffled)
            + max(0.0, real_restore - restore_wrong)
        )
        out.append(
            {
                "candidate_id": cid,
                "stage6_base_candidate_id": meta.get("stage6_base_candidate_id", ""),
                "stage6_condition_id": meta.get("stage6_condition_id", ""),
                "sample_id": meta.get("sample_id") or (rows[0].get("sample_id", "") if rows else ""),
                "stage6_question_variant": meta.get("stage6_question_variant", ""),
                "stage6_prompt_family": meta.get("stage6_prompt_family", ""),
                "stage6_sample_type": meta.get("stage6_sample_type", ""),
                "layer": meta.get("layer", ""),
                "source_pos": meta.get("source_pos", ""),
                "source_feature_id": meta.get("source_feature_id", ""),
                "node_key_layer_pos_feature": f"{meta.get('layer', '')}:{meta.get('source_pos', '')}:{meta.get('source_feature_id', '')}",
                "feature_id_key": meta.get("source_feature_id", ""),
                "real_condition": real,
                "clean_source_minus_controls": clean,
                "restore_source_minus_controls": real_restore,
                "real_minus_shifted": real_restore - shifted,
                "real_minus_shuffled": real_restore - shuffled,
                "restore_correct_minus_wrong": real_restore - restore_wrong,
                "evidence_specificity": evidence_specificity,
                "route_identity_score": route_score,
                "pos234": "1" if real_restore > 0 and real_restore - shifted > 0 and real_restore - shuffled > 0 else "0",
            }
        )
    return out


def _topk_overlap_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        grouped[(row.get("sample_id", ""), row.get("stage6_question_variant", ""), row.get("stage6_prompt_family", ""))].append(row)
    baselines = {
        sample: sorted(rows, key=lambda row: _f(row.get("route_identity_score")), reverse=True)
        for (sample, variant, prompt), rows in grouped.items()
        if variant == "original" and prompt == "B_direct"
    }
    out = []
    for (sample, variant, prompt), rows in sorted(grouped.items()):
        base = baselines.get(sample)
        if not base:
            continue
        ranked = sorted(rows, key=lambda row: _f(row.get("route_identity_score")), reverse=True)
        for k in [4, 8, 16, 32]:
            base_top = base[:k]
            cond_top = ranked[:k]
            out.append(
                {
                    "sample_id": sample,
                    "stage6_question_variant": variant,
                    "stage6_prompt_family": prompt,
                    "topk": k,
                    "base_n": len(base_top),
                    "condition_n": len(cond_top),
                    "base_candidate_overlap": _jaccard(
                        {row.get("stage6_base_candidate_id", "") for row in base_top},
                        {row.get("stage6_base_candidate_id", "") for row in cond_top},
                    ),
                    "feature_id_overlap": _jaccard(
                        {row.get("feature_id_key", "") for row in base_top},
                        {row.get("feature_id_key", "") for row in cond_top},
                    ),
                    "layer_pos_feature_overlap": _jaccard(
                        {row.get("node_key_layer_pos_feature", "") for row in base_top},
                        {row.get("node_key_layer_pos_feature", "") for row in cond_top},
                    ),
                    "condition_score_mean": _mean([_f(row.get("route_identity_score")) for row in cond_top]),
                    "baseline_score_mean": _mean([_f(row.get("route_identity_score")) for row in base_top]),
                }
            )
    return out


def _analyze_qwen_routeidentity(tag: str, mode: str) -> dict[str, Any]:
    metrics = _qwen_routeidentity_metrics(tag, mode)
    overlap = _topk_overlap_rows(metrics)
    for row in metrics:
        row["pos234_numeric"] = 1.0 if row.get("pos234") == "1" else 0.0
    metric_names = [
        "route_identity_score",
        "restore_source_minus_controls",
        "real_minus_shifted",
        "real_minus_shuffled",
        "restore_correct_minus_wrong",
        "pos234_numeric",
    ]
    _write_csv(STAGE6_CROSS / f"{PREFIX}_routeidentity_{mode}_{tag}_candidate_metrics.csv", metrics)
    _write_csv(STAGE6_CROSS / f"{PREFIX}_routeidentity_{mode}_{tag}_topk_overlap.csv", overlap)
    _write_csv(
        STAGE6_CROSS / f"{PREFIX}_routeidentity_{mode}_{tag}_summary_by_prompt.csv",
        _summary(metrics, ["stage6_prompt_family"], metric_names),
    )
    _write_csv(
        STAGE6_CROSS / f"{PREFIX}_routeidentity_{mode}_{tag}_summary_overlap_by_prompt_topk.csv",
        _summary(overlap, ["stage6_prompt_family", "topk"], ["base_candidate_overlap", "feature_id_overlap", "layer_pos_feature_overlap"]),
    )
    return {
        "mode": mode,
        "candidate_metrics": len(metrics),
        "topk_overlap_rows": len(overlap),
        "candidate_metrics_csv": str(STAGE6_CROSS / f"{PREFIX}_routeidentity_{mode}_{tag}_candidate_metrics.csv"),
        "topk_overlap_csv": str(STAGE6_CROSS / f"{PREFIX}_routeidentity_{mode}_{tag}_topk_overlap.csv"),
    }


def _analyze_gemma_fixednode(tag: str, mode: str) -> dict[str, Any]:
    raw_path = STAGE6_CROSS / f"{PREFIX}_fixednode_{mode}_{tag}_raw.csv"
    manifest_path = STAGE6_CROSS / f"{PREFIX}_fixednode_{tag}_manifest.csv"
    rows = _read_csv(raw_path)
    planned_rows = [row for row in _read_csv(manifest_path) if row.get(f"stage6_include_{mode}", "1") == "1"]
    if not rows:
        coverage = _gemma_fixednode_coverage_rows(planned_rows, [])
        _write_csv(STAGE6_CROSS / f"{PREFIX}_fixednode_{mode}_{tag}_coverage_rows.csv", coverage)
        _write_csv(
            STAGE6_CROSS / f"{PREFIX}_fixednode_{mode}_{tag}_coverage_summary_overall.csv",
            _gemma_fixednode_coverage_summary(coverage, []),
        )
        return {
            "mode": mode,
            "planned_rows": len(planned_rows),
            "rows": 0,
            "status": "not_available",
            "coverage_rows_csv": str(STAGE6_CROSS / f"{PREFIX}_fixednode_{mode}_{tag}_coverage_rows.csv"),
        }
    for row in rows:
        row["aligned_main"] = "1" if row.get("prefix_ok") == "1" and row.get("target_token_same") == "1" else "0"
        row["usable_exact"] = "1" if row["aligned_main"] == "1" and row.get("status", "ok") == "ok" else "0"
        row["source_pos_out_of_range"] = "1" if "source_pos_out_of_range" in row.get("error_message", "") else "0"
        row["source_minus_controls_positive"] = 1.0 if _f(row.get("source_minus_controls")) > 0 else 0.0
    aligned = [row for row in rows if row["aligned_main"] == "1"]
    usable = [row for row in rows if row["usable_exact"] == "1"]
    coverage = _gemma_fixednode_coverage_rows(planned_rows, rows)
    metrics = ["source_damage_target_logit", "source_minus_controls", "correct_minus_wrong", "source_minus_controls_positive"]
    _write_csv(STAGE6_CROSS / f"{PREFIX}_fixednode_{mode}_{tag}_aligned_rows.csv", aligned)
    _write_csv(STAGE6_CROSS / f"{PREFIX}_fixednode_{mode}_{tag}_usable_exact_rows.csv", usable)
    _write_csv(STAGE6_CROSS / f"{PREFIX}_fixednode_{mode}_{tag}_coverage_rows.csv", coverage)
    _write_csv(
        STAGE6_CROSS / f"{PREFIX}_fixednode_{mode}_{tag}_coverage_summary_overall.csv",
        _gemma_fixednode_coverage_summary(coverage, []),
    )
    _write_csv(
        STAGE6_CROSS / f"{PREFIX}_fixednode_{mode}_{tag}_coverage_summary_by_prompt.csv",
        _gemma_fixednode_coverage_summary(coverage, ["stage6_prompt_family"]),
    )
    _write_csv(
        STAGE6_CROSS / f"{PREFIX}_fixednode_{mode}_{tag}_coverage_summary_by_variant.csv",
        _gemma_fixednode_coverage_summary(coverage, ["stage6_question_variant"]),
    )
    _write_csv(
        STAGE6_CROSS / f"{PREFIX}_fixednode_{mode}_{tag}_coverage_summary_by_sample.csv",
        _gemma_fixednode_coverage_summary(coverage, ["stage6_original_sample_id"]),
    )
    _write_csv(
        STAGE6_CROSS / f"{PREFIX}_fixednode_{mode}_{tag}_summary_by_prompt_aligned.csv",
        _summary(usable, ["stage6_prompt_family"], metrics),
    )
    _write_csv(
        STAGE6_CROSS / f"{PREFIX}_fixednode_{mode}_{tag}_summary_by_variant_aligned.csv",
        _summary(usable, ["stage6_question_variant"], metrics),
    )
    planned_aligned = sum(1 for row in coverage if row.get("aligned_planned") == 1)
    out_of_range = sum(1 for row in coverage if row.get("source_pos_out_of_range") == 1)
    attempted = sum(1 for row in coverage if row.get("attempted") == 1)
    return {
        "mode": mode,
        "planned_rows": len(planned_rows),
        "rows": len(rows),
        "aligned_rows": len(aligned),
        "usable_exact_rows": len(usable),
        "planned_aligned_rows": planned_aligned,
        "source_pos_out_of_range_rows": out_of_range,
        "source_pos_out_of_range_over_attempted": _fraction(out_of_range, attempted),
        "usable_exact_over_planned": _fraction(len(usable), len(planned_rows)),
        "usable_exact_over_planned_aligned": _fraction(len(usable), planned_aligned),
        "raw_csv": str(raw_path),
        "aligned_csv": str(STAGE6_CROSS / f"{PREFIX}_fixednode_{mode}_{tag}_aligned_rows.csv"),
        "usable_exact_csv": str(STAGE6_CROSS / f"{PREFIX}_fixednode_{mode}_{tag}_usable_exact_rows.csv"),
        "coverage_rows_csv": str(STAGE6_CROSS / f"{PREFIX}_fixednode_{mode}_{tag}_coverage_rows.csv"),
        "coverage_summary_overall_csv": str(STAGE6_CROSS / f"{PREFIX}_fixednode_{mode}_{tag}_coverage_summary_overall.csv"),
    }


def _baseline_condition(row: dict[str, Any]) -> bool:
    return row.get("stage6_question_variant") == "original" and row.get("stage6_prompt_family") == "B_direct"


def _ratio(value: float, baseline: float) -> float | str:
    if abs(baseline) < 1e-9:
        return ""
    return value / baseline


def _fixed_node_key(row: dict[str, Any], model: str) -> tuple[str, str, str, str, str]:
    if model == "qwen":
        return (
            str(row.get("sample_id", "")),
            str(row.get("stage6_base_candidate_id", "")),
            str(row.get("layer", "")),
            str(row.get("source_pos", "")),
            str(row.get("source_feature_id", "")),
        )
    return (
        str(row.get("stage6_original_sample_id") or str(row.get("sample_id", "")).split("__", 1)[0]),
        "",
        str(row.get("source_layer", "")),
        str(row.get("source_pos", "")),
        str(row.get("source_feature_id", "")),
    )


def _unified_fixednode_outputs(tag: str, mode: str) -> dict[str, Any]:
    qwen_rows = _read_csv(STAGE6_CROSS / "stage6_prompt_text_cot_full_prompttext_v1_candidate_metrics.csv")
    gemma_rows = _read_csv(STAGE6_CROSS / f"{PREFIX}_fixednode_{mode}_{tag}_usable_exact_rows.csv")
    rows: list[dict[str, Any]] = []

    sources = [
        {
            "model": "qwen",
            "source_rows": qwen_rows,
            "metric": "clean_source_minus_controls",
            "correct_metric": "clean_correct_minus_wrong",
            "condition_id": lambda row: f"{row.get('sample_id', '')}__{row.get('stage6_question_variant', '')}__{row.get('stage6_prompt_family', '')}",
        },
        {
            "model": "gemma",
            "source_rows": [row for row in gemma_rows if row.get("usable_exact", "1") == "1"],
            "metric": "source_minus_controls",
            "correct_metric": "correct_minus_wrong",
            "condition_id": lambda row: row.get("sample_id", ""),
        },
    ]

    for source in sources:
        model = str(source["model"])
        source_rows = list(source["source_rows"])
        metric = str(source["metric"])
        correct_metric = str(source["correct_metric"])
        baseline: dict[tuple[str, str, str, str, str], float] = {}
        for row in source_rows:
            if _baseline_condition(row):
                baseline[_fixed_node_key(row, model)] = _f(row.get(metric))
        for row in source_rows:
            key = _fixed_node_key(row, model)
            value = _f(row.get(metric))
            base_value = baseline.get(key)
            item = {
                "model": model,
                "metric_lens": "fixed_node_causal_strength",
                "condition_id": source["condition_id"](row),
                "sample_id": row.get("sample_id", ""),
                "stage6_original_sample_id": row.get("stage6_original_sample_id") or row.get("sample_id", ""),
                "stage6_question_variant": row.get("stage6_question_variant", ""),
                "stage6_prompt_family": row.get("stage6_prompt_family", ""),
                "stage6_sample_type": row.get("stage6_sample_type", ""),
                "node_key": ":".join(key),
                "metric_name": metric,
                "fixed_node_strength": value,
                "fixed_node_positive": 1.0 if value > 0 else 0.0,
                "correct_minus_wrong": _f(row.get(correct_metric)),
                "baseline_available": 1.0 if base_value is not None else 0.0,
                "baseline_strength": base_value if base_value is not None else "",
                "delta_from_baseline": value - base_value if base_value is not None else "",
                "retention_ratio": _ratio(value, base_value) if base_value is not None else "",
                "main_filter": "usable_exact_prefix_target_position" if model == "gemma" else "qwen_stage6_old_fixed_node",
            }
            rows.append(item)

    out_path = STAGE6_CROSS / f"{PREFIX}_crossmodel_{tag}_fixednode_unified_rows.csv"
    by_prompt = STAGE6_CROSS / f"{PREFIX}_crossmodel_{tag}_fixednode_unified_summary_by_model_prompt.csv"
    by_variant = STAGE6_CROSS / f"{PREFIX}_crossmodel_{tag}_fixednode_unified_summary_by_model_variant.csv"
    metrics = ["fixed_node_strength", "fixed_node_positive", "correct_minus_wrong", "delta_from_baseline", "retention_ratio"]
    _write_csv(out_path, rows)
    _write_csv(by_prompt, _summary(rows, ["model", "stage6_prompt_family"], metrics))
    _write_csv(by_variant, _summary(rows, ["model", "stage6_question_variant"], metrics))
    return {
        "rows": len(rows),
        "models": sorted({row.get("model", "") for row in rows}),
        "rows_csv": str(out_path),
        "summary_by_model_prompt": str(by_prompt),
        "summary_by_model_variant": str(by_variant),
        "interpretation": "Fixed-node rows are comparable by direction, positive fraction, and within-model delta/retention; raw logit magnitudes are not compared across models.",
    }


def _gemma_aligned_route_rows(source_tag: str) -> list[dict[str, Any]]:
    prefix = f"stage6_gemma_prompt_text_cot_full_{source_tag}"
    compare = _read_csv(STAGE6_CROSS / f"{prefix}_sample_compare_controlled.csv")
    meta_a = {row["sample_id"]: row for row in _read_csv(STAGE6_CROSS / f"{prefix}_meta_a.csv") if row.get("sample_id")}
    rows = []
    for row in compare:
        sample_id = row.get("sample_id", "")
        meta = meta_a.get(sample_id, {})
        parts = sample_id.split("__")
        target_same = row.get("a_target_token_id") and row.get("a_target_token_id") == row.get("b_target_token_id")
        prefix_ok = bool((meta.get("assistant_prefix") or "").strip())
        if not (target_same and prefix_ok):
            continue
        item = dict(row)
        item["stage6_original_sample_id"] = sample_id.split("__", 1)[0]
        item["stage6_question_variant"] = parts[1] if len(parts) > 1 else ""
        item["stage6_prompt_family"] = parts[2] if len(parts) > 2 else ""
        rows.append(item)
    return rows


def _unified_routeidentity_outputs(tag: str, mode: str, gemma_source_tag: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    qwen_overlap = _read_csv(STAGE6_CROSS / f"{PREFIX}_routeidentity_{mode}_{tag}_topk_overlap.csv")
    for row in qwen_overlap:
        topk = str(row.get("topk", ""))
        constructed_full_pool = topk == "32" and tag.endswith("focus4")
        for metric in ["feature_id_overlap", "layer_pos_feature_overlap"]:
            rows.append(
                {
                    "model": "qwen",
                    "route_object": "qwen_route_first_feature_set",
                    "overlap_metric": metric,
                    "overlap_value": _f(row.get(metric)),
                    "sample_id": row.get("sample_id", ""),
                    "stage6_question_variant": row.get("stage6_question_variant", ""),
                    "stage6_prompt_family": row.get("stage6_prompt_family", ""),
                    "topk": topk,
                    "constructed_full_pool": 1.0 if constructed_full_pool else 0.0,
                    "main_filter": "exclude_top32_for_focus4_identity_claims" if constructed_full_pool else "main",
                }
            )

    for row in _gemma_aligned_route_rows(gemma_source_tag):
        for metric in ["node_overlap_jaccard", "edge_overlap_jaccard"]:
            rows.append(
                {
                    "model": "gemma",
                    "route_object": "gemma_source_tracing_graph",
                    "overlap_metric": metric,
                    "overlap_value": _f(row.get(metric)),
                    "sample_id": row.get("sample_id", ""),
                    "stage6_question_variant": row.get("stage6_question_variant", ""),
                    "stage6_prompt_family": row.get("stage6_prompt_family", ""),
                    "topk": "graph_mf8",
                    "constructed_full_pool": 0.0,
                    "main_filter": "aligned_prefix_and_target",
                }
            )

    out_path = STAGE6_CROSS / f"{PREFIX}_crossmodel_{tag}_routeidentity_unified_rows.csv"
    by_prompt = STAGE6_CROSS / f"{PREFIX}_crossmodel_{tag}_routeidentity_unified_summary_by_model_prompt.csv"
    by_metric = STAGE6_CROSS / f"{PREFIX}_crossmodel_{tag}_routeidentity_unified_summary_by_model_metric.csv"
    main_rows = [row for row in rows if row.get("main_filter") != "exclude_top32_for_focus4_identity_claims"]
    _write_csv(out_path, rows)
    _write_csv(by_prompt, _summary(main_rows, ["model", "route_object", "overlap_metric", "stage6_prompt_family", "topk"], ["overlap_value"]))
    _write_csv(by_metric, _summary(main_rows, ["model", "route_object", "overlap_metric", "topk"], ["overlap_value"]))
    return {
        "rows": len(rows),
        "main_rows": len(main_rows),
        "rows_csv": str(out_path),
        "summary_by_model_prompt": str(by_prompt),
        "summary_by_model_metric": str(by_metric),
        "interpretation": "Route-identity rows use Jaccard-style overlap. Qwen focus4 Top32 is marked constructed_full_pool and excluded from main identity claims.",
    }


def analyze(tag: str, mode: str, gemma_source_tag: str) -> dict[str, Any]:
    old_qwen = _old_qwen_reanalysis(tag)
    old_gemma = _old_gemma_reanalysis(tag, gemma_source_tag)
    qwen = _analyze_qwen_routeidentity(tag, mode)
    gemma_fixed = _analyze_gemma_fixednode(tag, mode)
    unified_fixednode = _unified_fixednode_outputs(tag, mode)
    unified_routeidentity = _unified_routeidentity_outputs(tag, mode, gemma_source_tag)
    decision = {
        "updated_at": _now(),
        "tag": tag,
        "mode": mode,
        "status": "partial_or_full_unified_analysis",
        "old_qwen_reanalysis": old_qwen,
        "old_gemma_reanalysis": old_gemma,
        "qwen_routeidentity": qwen,
        "gemma_fixednode": gemma_fixed,
        "unified_fixednode": unified_fixednode,
        "unified_routeidentity": unified_routeidentity,
        "claim_boundary": (
            "Stage6 unified analysis separates fixed-node causal strength from route identity/topology stability. "
            "Format failures and target-token mismatches are diagnostics, not mechanism failures."
        ),
    }
    _write_json(STAGE6_CROSS / f"{PREFIX}_crossmodel_{tag}_decision.json", decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage6 unified prompt/text metrics.")
    parser.add_argument("--tag", default="unified_v1")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--gemma-source-tag", default="gemmaprompt_v1_mf8_sharded")
    args = parser.parse_args()
    analyze(args.tag, args.mode, args.gemma_source_tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
