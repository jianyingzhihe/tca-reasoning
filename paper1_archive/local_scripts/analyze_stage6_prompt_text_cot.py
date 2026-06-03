#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
STAGE6 = ROOT / "doc" / "experiments" / "stage6" / "cross_model"
PREFIX = "stage6_prompt_text_cot"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
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


def _f(raw: Any, default: float = 0.0) -> float:
    try:
        value = float(raw) if raw not in (None, "") else default
        if math.isnan(value):
            return default
        return value
    except ValueError:
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sem_ci_low(values: list[float]) -> float:
    if len(values) < 2:
        return values[0] if values else 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean - 1.96 * math.sqrt(variance / len(values))


def _aggregate(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": _mean(values),
        "ci_low_sem": _sem_ci_low(values),
        "positive_frac": sum(1 for value in values if value > 0) / len(values) if values else 0.0,
    }


def _raw_paths(mode: str, tag: str) -> list[Path]:
    return sorted(STAGE6.glob(f"{PREFIX}_{mode}_L*_{tag}_raw.csv"))


def _manifest_index(tag: str) -> dict[str, dict[str, str]]:
    path = STAGE6 / f"{PREFIX}_{tag}_manifest.csv"
    return {row.get("candidate_id", ""): row for row in _read_csv(path) if row.get("candidate_id")}


def _combined_raw(mode: str, tag: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in _raw_paths(mode, tag):
        rows.extend(_read_csv(path))
    return rows


def _source_control_metric(rows: list[dict[str, str]], intervention: str, mask_condition: str, token: str) -> float:
    matched = [
        row
        for row in rows
        if row.get("intervention_kind") == intervention
        and row.get("mask_condition") == mask_condition
        and row.get("token_scored") == token
    ]
    source = [_f(row.get("logit_effect")) for row in matched if row.get("control_group") == "source"]
    controls = [_f(row.get("logit_effect")) for row in matched if row.get("control_group") != "source"]
    return (source[0] if source else 0.0) - _mean(controls)


def _source_metric(rows: list[dict[str, str]], intervention: str, mask_condition: str, token: str, field: str) -> float:
    for row in rows:
        if (
            row.get("intervention_kind") == intervention
            and row.get("mask_condition") == mask_condition
            and row.get("token_scored") == token
            and row.get("control_group") == "source"
        ):
            return _f(row.get(field))
    return 0.0


def _candidate_metrics(rows: list[dict[str, str]], manifest: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("candidate_id", "")].append(row)
    metrics: list[dict[str, Any]] = []
    for candidate_id, cand_rows in grouped.items():
        meta = manifest.get(candidate_id, {})
        answer_restore = _source_control_metric(cand_rows, "mask_restore", "answer_mask", "target")
        union_restore = _source_control_metric(cand_rows, "mask_restore", "union_mask", "target")
        real_condition = "answer_mask" if answer_restore >= union_restore else "union_mask"
        real_restore = max(answer_restore, union_restore)
        shifted_restore = _source_control_metric(cand_rows, "mask_restore", "shifted_mask", "target")
        shuffled_restore = _source_control_metric(cand_rows, "mask_restore", "shuffled_mask", "target")
        clean_zeroing = _source_control_metric(cand_rows, "clean_zeroing", "clean", "target")
        restore_wrong = _source_control_metric(cand_rows, "mask_restore", real_condition, "wrong")
        clean_wrong = _source_control_metric(cand_rows, "clean_zeroing", "clean", "wrong")
        answer_drop = _source_metric(cand_rows, "mask_restore", "answer_mask", "target", "activation_drop")
        union_drop = _source_metric(cand_rows, "mask_restore", "union_mask", "target", "activation_drop")
        shifted_drop = _source_metric(cand_rows, "mask_restore", "shifted_mask", "target", "activation_drop")
        shuffled_drop = _source_metric(cand_rows, "mask_restore", "shuffled_mask", "target", "activation_drop")
        evidence_specificity = max(answer_drop, union_drop) - max(shifted_drop, shuffled_drop)
        metrics.append(
            {
                "candidate_id": candidate_id,
                "sample_id": meta.get("sample_id") or cand_rows[0].get("sample_id", ""),
                "stage6_sample_type": meta.get("stage6_sample_type", ""),
                "stage6_question_variant": meta.get("stage6_question_variant", ""),
                "stage6_prompt_family": meta.get("stage6_prompt_family") or cand_rows[0].get("prompt_name", ""),
                "stage6_base_candidate_id": meta.get("stage6_base_candidate_id", ""),
                "stage6_source_prompt_name": meta.get("stage6_source_prompt_name", ""),
                "layer": meta.get("layer", ""),
                "source_pos": meta.get("source_pos", ""),
                "source_feature_id": meta.get("source_feature_id", ""),
                "clean_source_minus_controls": clean_zeroing,
                "restore_source_minus_controls": real_restore,
                "real_condition": real_condition,
                "real_minus_shifted": real_restore - shifted_restore,
                "real_minus_shuffled": real_restore - shuffled_restore,
                "restore_correct_minus_wrong": real_restore - restore_wrong,
                "clean_correct_minus_wrong": clean_zeroing - clean_wrong,
                "evidence_specificity": evidence_specificity,
                "answer_activation_drop": answer_drop,
                "union_activation_drop": union_drop,
                "shifted_activation_drop": shifted_drop,
                "shuffled_activation_drop": shuffled_drop,
                "row_count": len(cand_rows),
            }
        )
    return metrics


def _summary_rows(metrics: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        grouped[tuple(str(row.get(key, "")) for key in group_keys)].append(row)
    out: list[dict[str, Any]] = []
    metric_names = [
        "clean_source_minus_controls",
        "restore_source_minus_controls",
        "real_minus_shifted",
        "real_minus_shuffled",
        "restore_correct_minus_wrong",
        "clean_correct_minus_wrong",
        "evidence_specificity",
    ]
    for key, rows in sorted(grouped.items()):
        base = {group_key: value for group_key, value in zip(group_keys, key, strict=False)}
        base["candidates"] = len(rows)
        base["unique_samples"] = len({row.get("sample_id", "") for row in rows})
        for name in metric_names:
            agg = _aggregate([_f(row.get(name)) for row in rows])
            base[f"{name}_mean"] = agg["mean"]
            base[f"{name}_ci_low_sem"] = agg["ci_low_sem"]
            base[f"{name}_positive_frac"] = agg["positive_frac"]
        out.append(base)
    return out


def _route_stability(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        grouped[
            (
                str(row.get("sample_id", "")),
                str(row.get("stage6_base_candidate_id", "")),
                str(row.get("stage6_prompt_family", "")),
            )
        ].append(row)
    out: list[dict[str, Any]] = []
    for (sample_id, base_candidate_id, prompt_family), rows in sorted(grouped.items()):
        by_variant = {row.get("stage6_question_variant", ""): row for row in rows}
        if "original" not in by_variant:
            continue
        original = by_variant["original"]
        for variant, row in sorted(by_variant.items()):
            if variant == "original":
                continue
            out.append(
                {
                    "sample_id": sample_id,
                    "stage6_base_candidate_id": base_candidate_id,
                    "stage6_prompt_family": prompt_family,
                    "stage6_question_variant": variant,
                    "restore_source_delta_vs_original": _f(row.get("restore_source_minus_controls"))
                    - _f(original.get("restore_source_minus_controls")),
                    "evidence_specificity_delta_vs_original": _f(row.get("evidence_specificity"))
                    - _f(original.get("evidence_specificity")),
                    "real_minus_shifted_delta_vs_original": _f(row.get("real_minus_shifted"))
                    - _f(original.get("real_minus_shifted")),
                    "real_minus_shuffled_delta_vs_original": _f(row.get("real_minus_shuffled"))
                    - _f(original.get("real_minus_shuffled")),
                }
            )
    return out


def analyze(mode: str, tag: str) -> dict[str, Any]:
    manifest = _manifest_index(tag)
    raw = _combined_raw(mode, tag)
    if not raw:
        raise FileNotFoundError(f"no raw Stage6 files found for mode={mode} tag={tag}")
    metrics = _candidate_metrics(raw, manifest)
    by_prompt = _summary_rows(metrics, ["stage6_prompt_family"])
    by_variant = _summary_rows(metrics, ["stage6_question_variant"])
    by_type = _summary_rows(metrics, ["stage6_sample_type"])
    by_prompt_variant = _summary_rows(metrics, ["stage6_prompt_family", "stage6_question_variant"])
    stability = _route_stability(metrics)

    metric_fields = sorted({key for row in metrics for key in row})
    summary_fields = sorted({key for rows in [by_prompt, by_variant, by_type, by_prompt_variant] for row in rows for key in row})
    stability_fields = sorted({key for row in stability for key in row})
    _write_csv(STAGE6 / f"{PREFIX}_{mode}_{tag}_candidate_metrics.csv", metrics, metric_fields)
    _write_csv(STAGE6 / f"{PREFIX}_{mode}_{tag}_summary_by_prompt.csv", by_prompt, summary_fields)
    _write_csv(STAGE6 / f"{PREFIX}_{mode}_{tag}_summary_by_variant.csv", by_variant, summary_fields)
    _write_csv(STAGE6 / f"{PREFIX}_{mode}_{tag}_summary_by_type.csv", by_type, summary_fields)
    _write_csv(STAGE6 / f"{PREFIX}_{mode}_{tag}_summary_by_prompt_variant.csv", by_prompt_variant, summary_fields)
    if stability:
        _write_csv(STAGE6 / f"{PREFIX}_{mode}_{tag}_rewrite_stability.csv", stability, stability_fields)

    prompt_map = {row.get("stage6_prompt_family", ""): row for row in by_prompt}
    variant_map = {row.get("stage6_question_variant", ""): row for row in by_variant}
    original = variant_map.get("original", {})
    paraphrases = [row for key, row in variant_map.items() if key.startswith("paraphrase")]
    text_stability_hint = (
        bool(original)
        and bool(paraphrases)
        and all(_f(row.get("evidence_specificity_mean")) > 0 for row in paraphrases)
    )
    prompt_modulation_hint = False
    if len(prompt_map) >= 2:
        evidence_means = [_f(row.get("evidence_specificity_mean")) for row in prompt_map.values()]
        prompt_modulation_hint = max(evidence_means) - min(evidence_means) > 0.5

    decision = {
        "updated_at": _now(),
        "status": "exploratory_analyzed",
        "mode": mode,
        "tag": tag,
        "raw_rows": len(raw),
        "candidate_metrics": len(metrics),
        "unique_samples": len({row.get("sample_id", "") for row in metrics}),
        "prompt_counts": dict(Counter(row.get("stage6_prompt_family", "") for row in metrics)),
        "variant_counts": dict(Counter(row.get("stage6_question_variant", "") for row in metrics)),
        "type_counts": dict(Counter(row.get("stage6_sample_type", "") for row in metrics)),
        "text_stability_hint": text_stability_hint,
        "prompt_modulation_hint": prompt_modulation_hint,
        "claim_boundary": (
            "Exploratory secondary-claim analysis only. Positive signs motivate paper-side claims "
            "after manual prompt/format audit; they do not replace Stage4 causal route evidence."
        ),
    }
    _write_json(STAGE6 / f"{PREFIX}_{mode}_{tag}_decision.json", decision)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage6 prompt/text/CoT exploratory probe.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--tag", default="prompttext_v1")
    args = parser.parse_args()
    decision = analyze(args.mode, args.tag)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
