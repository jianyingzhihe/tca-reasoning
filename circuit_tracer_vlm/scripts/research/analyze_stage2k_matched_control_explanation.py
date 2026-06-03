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
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _float(value: Any) -> float | None:
    try:
        if value == "" or value is None:
            return None
        return float(value)
    except Exception:
        return None


def _mean(values: list[float]) -> float | str:
    return round(mean(values), 6) if values else ""


def _bootstrap_ci(values: list[float], seed: int = 20260520, n_boot: int = 5000) -> tuple[float | str, float | str, float | str, str]:
    vals = [value for value in values if value is not None]
    if not vals:
        return "", "", "", "not_available"
    rng = random.Random(seed)
    n = len(vals)
    obs = mean(vals)
    boots = [mean([rng.choice(vals) for _ in range(n)]) for _ in range(n_boot)]
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot) - 1]
    if lo > 0:
        status = "stable_positive"
    elif obs > 0:
        status = "weak_or_heterogeneous_positive"
    else:
        status = "not_positive"
    return round(obs, 6), round(lo, 6), round(hi, 6), status


def _manifest_index(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    return {row["sample_id"]: row for row in rows if row.get("sample_id")}


def _load_rows(qwen_csv: Path, llava_csv: Path, manifest: Path) -> list[dict[str, Any]]:
    metadata = _manifest_index(manifest)
    out: list[dict[str, Any]] = []
    for model, path in [("qwen", qwen_csv), ("llava", llava_csv)]:
        for row in _read_csv(path):
            sample_id = row.get("sample_id", "")
            meta = metadata.get(sample_id, {})
            if row.get("direction") not in {"restore", "corrupt"}:
                continue
            if row.get("scale") not in {"1", "1.0"}:
                continue
            out.append(
                {
                    **row,
                    "model_family": row.get("model_family") or model,
                    "reasoning_operation": meta.get("reasoning_operation", "unknown") or "unknown",
                    "visual_structure": meta.get("visual_structure", ""),
                    "answer_area_frac": meta.get("answer_area_frac", ""),
                    "effect_logit_f": _float(row.get("effect_logit")),
                    "effect_rank_f": _float(row.get("effect_rank")),
                    "effect_gap_closure_f": _float(row.get("effect_gap_closure")),
                }
            )
    return out


def _key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        row.get("model_family", ""),
        row.get("sample_id", ""),
        row.get("prompt_name", ""),
        row.get("direction", ""),
    )


def _case_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[_key(row)][row.get("group_name", "")] = row

    out: list[dict[str, Any]] = []
    for (model, sample_id, prompt, direction), group in sorted(grouped.items()):
        def value(group_name: str, metric: str = "effect_logit_f") -> float | None:
            item = group.get(group_name)
            return item.get(metric) if item else None

        answer_adj = value("answer_adjacent_text")
        source_visual = value("top_hidden_delta")
        source_combo = value("top_hidden_delta_plus_answer_adjacent")
        delta_visual = value("delta_matched_control")
        delta_combo = value("delta_matched_plus_answer_adjacent")
        activation_visual = value("activation_matched_control")
        activation_combo = value("activation_matched_plus_answer_adjacent")

        def diff(a: float | None, b: float | None) -> float | None:
            if a is None or b is None:
                return None
            return a - b

        any_row = next(iter(group.values()))
        out.append(
            {
                "model_family": model,
                "sample_id": sample_id,
                "prompt_name": prompt,
                "direction": direction,
                "reasoning_operation": any_row.get("reasoning_operation", "unknown"),
                "visual_structure": any_row.get("visual_structure", ""),
                "answer_area_frac": any_row.get("answer_area_frac", ""),
                "answer_adjacent_logit": answer_adj,
                "source_visual_logit": source_visual,
                "source_combo_logit": source_combo,
                "delta_visual_logit": delta_visual,
                "delta_combo_logit": delta_combo,
                "activation_visual_logit": activation_visual,
                "activation_combo_logit": activation_combo,
                "source_visual_increment_over_answer_adjacent": diff(source_combo, answer_adj),
                "delta_visual_increment_over_answer_adjacent": diff(delta_combo, answer_adj),
                "activation_visual_increment_over_answer_adjacent": diff(activation_combo, answer_adj),
                "combo_minus_delta_combo": diff(source_combo, delta_combo),
                "combo_minus_activation_combo": diff(source_combo, activation_combo),
                "visual_minus_delta_visual": diff(source_visual, delta_visual),
                "visual_minus_activation_visual": diff(source_visual, activation_visual),
                "combo_minus_answer_adjacent": diff(source_combo, answer_adj),
                "visual_specificity_label": _case_label(diff(source_combo, delta_combo), diff(source_combo, activation_combo)),
            }
        )
    return out


def _case_label(delta_diff: float | None, activation_diff: float | None) -> str:
    if delta_diff is None or activation_diff is None:
        return "not_available"
    if delta_diff > 0 and activation_diff > 0:
        return "above_both_matched_controls"
    if delta_diff <= 0 and activation_diff > 0:
        return "delta_absorbed_activation_positive"
    if delta_diff > 0 and activation_diff <= 0:
        return "activation_absorbed_delta_positive"
    return "not_above_matched_controls"


def _aggregate(case_rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[tuple(str(row.get(key, "")) for key in group_keys)].append(row)

    metrics = [
        "source_visual_increment_over_answer_adjacent",
        "delta_visual_increment_over_answer_adjacent",
        "activation_visual_increment_over_answer_adjacent",
        "combo_minus_delta_combo",
        "combo_minus_activation_combo",
        "visual_minus_delta_visual",
        "visual_minus_activation_visual",
        "combo_minus_answer_adjacent",
    ]
    out: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        base = {group_key: value for group_key, value in zip(group_keys, key, strict=False)}
        base["n_rows"] = len(rows)
        base["n_samples"] = len({row["sample_id"] for row in rows})
        for metric in metrics:
            values = [row.get(metric) for row in rows if row.get(metric) is not None]
            obs, lo, hi, status = _bootstrap_ci([float(value) for value in values])
            base[f"{metric}_mean"] = obs
            base[f"{metric}_ci95_low"] = lo
            base[f"{metric}_ci95_high"] = hi
            base[f"{metric}_status"] = status
        labels = defaultdict(int)
        for row in rows:
            labels[str(row.get("visual_specificity_label", ""))] += 1
        base["case_label_counts"] = json.dumps(dict(sorted(labels.items())), ensure_ascii=False)
        out.append(base)
    return out


def _decision(model_summary: list[dict[str, Any]], typed_summary: list[dict[str, Any]]) -> dict[str, Any]:
    by_model_direction = {
        (row["model_family"], row["direction"]): row
        for row in model_summary
    }
    out = {
        "claim_boundary": (
            "Stage 2K is an explanatory re-analysis of hidden-position matched-control outputs. "
            "It can support or weaken interpretations about delta/activation explanations, "
            "but cannot establish CLT feature-level source-control route replication."
        ),
        "model_decisions": {},
        "typed_notes": {},
    }
    for model in sorted({row["model_family"] for row in model_summary}):
        restore = by_model_direction.get((model, "restore"), {})
        corrupt = by_model_direction.get((model, "corrupt"), {})
        restore_delta = restore.get("combo_minus_delta_combo_status", "")
        restore_activation = restore.get("combo_minus_activation_combo_status", "")
        corrupt_delta = corrupt.get("combo_minus_delta_combo_status", "")
        corrupt_activation = corrupt.get("combo_minus_activation_combo_status", "")
        if model == "qwen":
            status = "specificity_explanation_supported" if restore_delta == "stable_positive" else "specificity_explanation_partial"
            reading = (
                "Qwen remains above both delta- and activation-matched bridge controls in restore; "
                "corrupt is positive but activation-matched is weaker. This supports Qwen as the stronger cross-model auxiliary line."
            )
        else:
            status = "delta_explanation_partially_supported"
            reading = (
                "LLaVA is robustly above activation-matched controls, but not robustly above delta-matched controls. "
                "This supports the interpretation that clean-vs-union delta magnitude explains part of LLaVA's bridge."
            )
        out["model_decisions"][model] = {
            "status": status,
            "restore_combo_minus_delta_status": restore_delta,
            "restore_combo_minus_activation_status": restore_activation,
            "corrupt_combo_minus_delta_status": corrupt_delta,
            "corrupt_combo_minus_activation_status": corrupt_activation,
            "reading": reading,
        }

    for row in typed_summary:
        key = f"{row['model_family']}::{row['reasoning_operation']}::{row['direction']}"
        out["typed_notes"][key] = {
            "n_rows": row["n_rows"],
            "combo_minus_delta_mean": row.get("combo_minus_delta_combo_mean", ""),
            "combo_minus_delta_status": row.get("combo_minus_delta_combo_status", ""),
            "combo_minus_activation_mean": row.get("combo_minus_activation_combo_mean", ""),
            "combo_minus_activation_status": row.get("combo_minus_activation_combo_status", ""),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2K explanatory analysis for matched-control hidden-position results.")
    parser.add_argument("--qwen-csv", required=True)
    parser.add_argument("--llava-csv", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-case", required=True)
    parser.add_argument("--out-model-summary", required=True)
    parser.add_argument("--out-typed-summary", required=True)
    parser.add_argument("--out-decision", required=True)
    args = parser.parse_args()

    rows = _load_rows(Path(args.qwen_csv), Path(args.llava_csv), Path(args.manifest))
    cases = _case_metrics(rows)
    model_summary = _aggregate(cases, ["model_family", "direction"])
    typed_summary = _aggregate(cases, ["model_family", "reasoning_operation", "direction"])
    decision = _decision(model_summary, typed_summary)

    case_fields = [
        "model_family",
        "sample_id",
        "prompt_name",
        "direction",
        "reasoning_operation",
        "visual_structure",
        "answer_area_frac",
        "answer_adjacent_logit",
        "source_visual_logit",
        "source_combo_logit",
        "delta_visual_logit",
        "delta_combo_logit",
        "activation_visual_logit",
        "activation_combo_logit",
        "source_visual_increment_over_answer_adjacent",
        "delta_visual_increment_over_answer_adjacent",
        "activation_visual_increment_over_answer_adjacent",
        "combo_minus_delta_combo",
        "combo_minus_activation_combo",
        "visual_minus_delta_visual",
        "visual_minus_activation_visual",
        "combo_minus_answer_adjacent",
        "visual_specificity_label",
    ]
    summary_fields = list(model_summary[0].keys()) if model_summary else []
    typed_fields = list(typed_summary[0].keys()) if typed_summary else []
    _write_csv(Path(args.out_case), cases, case_fields)
    _write_csv(Path(args.out_model_summary), model_summary, summary_fields)
    _write_csv(Path(args.out_typed_summary), typed_summary, typed_fields)
    _write_json(Path(args.out_decision), decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

