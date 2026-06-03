#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(r"E:\Bridging")
CROSS_DIR = ROOT / "doc" / "experiments" / "stage2" / "cross_model"
SELECTED_MANIFEST = CROSS_DIR / "stage2m_selected_24_manifest.csv"
MATCHED_CASE = CROSS_DIR / "stage2m_matched_control_case.csv"
HIDDEN_CASE = CROSS_DIR / "stage2m_hidden_position_patch_case_table.csv"
OUT_PROMPT_ROWS = CROSS_DIR / "stage2m_decoded_candidate_prompt_rows.csv"
OUT_MANIFEST = CROSS_DIR / "stage2m_decoded_candidate_manifest.csv"
OUT_DECISION = CROSS_DIR / "stage2m_decoded_candidate_decision.json"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "" or value is None:
            return default
        return float(value)
    except Exception:
        return default


def _positive_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return mean(values)


def _group_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row.get("model_family", ""), row.get("sample_id", ""), row.get("prompt_name", "")


def _load_hidden_scores() -> dict[tuple[str, str, str], dict[str, float]]:
    rows = _read_csv(HIDDEN_CASE)
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("group_name") != "top_hidden_delta_plus_answer_adjacent":
            continue
        if row.get("direction") not in {"restore", "corrupt"}:
            continue
        key = _group_key(row)
        direction = row.get("direction", "")
        grouped[key][f"{direction}_effect_logit"].append(_float(row.get("effect_logit")))
        grouped[key][f"{direction}_gap_closure"].append(_float(row.get("effect_gap_closure")))
        grouped[key][f"{direction}_rank_effect"].append(_float(row.get("effect_rank")))

    out: dict[tuple[str, str, str], dict[str, float]] = {}
    for key, metrics in grouped.items():
        out[key] = {name: _positive_mean(values) for name, values in metrics.items()}
        out[key]["hidden_effect_score"] = (
            max(0.0, out[key].get("restore_effect_logit", 0.0))
            + max(0.0, out[key].get("corrupt_effect_logit", 0.0))
            + 0.25 * max(0.0, out[key].get("restore_gap_closure", 0.0))
            + 0.25 * max(0.0, out[key].get("corrupt_gap_closure", 0.0))
        )
    return out


def _load_matched_scores() -> dict[tuple[str, str, str], dict[str, float | str]]:
    rows = _read_csv(MATCHED_CASE)
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    labels: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    metadata: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = _group_key(row)
        direction = row.get("direction", "")
        if direction not in {"restore", "corrupt"}:
            continue
        grouped[key][f"{direction}_source_combo"].append(_float(row.get("source_combo_logit")))
        grouped[key][f"{direction}_minus_delta"].append(_float(row.get("combo_minus_delta_combo")))
        grouped[key][f"{direction}_minus_activation"].append(_float(row.get("combo_minus_activation_combo")))
        grouped[key][f"{direction}_minus_answer_adjacent"].append(_float(row.get("combo_minus_answer_adjacent")))
        labels[key].add(row.get("visual_specificity_label", ""))
        metadata[key] = {
            "reasoning_operation": row.get("reasoning_operation", ""),
            "visual_structure": row.get("visual_structure", ""),
            "answer_area_frac": row.get("answer_area_frac", ""),
        }

    out: dict[tuple[str, str, str], dict[str, float | str]] = {}
    for key, metrics in grouped.items():
        row: dict[str, float | str] = {name: _positive_mean(values) for name, values in metrics.items()}
        row.update(metadata.get(key, {}))
        row["specificity_labels"] = ";".join(sorted(labels.get(key, set())))
        matched_bonus = 0.0
        for name in [
            "restore_minus_delta",
            "restore_minus_activation",
            "corrupt_minus_delta",
            "corrupt_minus_activation",
        ]:
            matched_bonus += max(0.0, float(row.get(name, 0.0)))
        source_score = max(0.0, float(row.get("restore_source_combo", 0.0))) + max(
            0.0, float(row.get("corrupt_source_combo", 0.0))
        )
        row["matched_specificity_score"] = source_score + matched_bonus
        out[key] = row
    return out


def _type_bonus(model: str, reasoning_operation: str) -> float:
    if model == "qwen":
        if reasoning_operation == "symbol_text_reading":
            return 1.0
        if reasoning_operation == "visual_readout":
            return 0.4
    if model == "llava":
        if reasoning_operation == "visual_readout":
            return 0.8
        if reasoning_operation == "symbol_text_reading":
            return 0.6
    return 0.0


def _select_prompt_rows() -> list[dict[str, Any]]:
    hidden = _load_hidden_scores()
    matched = _load_matched_scores()
    manifest_by_sample = {row["sample_id"]: row for row in _read_csv(SELECTED_MANIFEST)}
    candidates: list[dict[str, Any]] = []
    for key, matched_metrics in matched.items():
        model, sample_id, prompt_name = key
        if model not in {"qwen", "llava"}:
            continue
        sample_meta = manifest_by_sample.get(sample_id, {})
        hidden_metrics = hidden.get(key, {})
        reasoning_operation = str(matched_metrics.get("reasoning_operation") or sample_meta.get("reasoning_operation", ""))
        score = (
            float(matched_metrics.get("matched_specificity_score", 0.0))
            + 0.75 * float(hidden_metrics.get("hidden_effect_score", 0.0))
            + _type_bonus(model, reasoning_operation)
        )
        has_bidirectional = (
            float(hidden_metrics.get("restore_effect_logit", 0.0)) > 0.0
            and float(hidden_metrics.get("corrupt_effect_logit", 0.0)) > 0.0
        )
        has_specificity = (
            float(matched_metrics.get("restore_minus_delta", 0.0)) > 0.0
            and float(matched_metrics.get("restore_minus_activation", 0.0)) > 0.0
        )
        candidates.append(
            {
                "model_family": model,
                "sample_id": sample_id,
                "prompt_name": prompt_name,
                "selection_score": round(score, 6),
                "reasoning_operation": reasoning_operation,
                "visual_structure": matched_metrics.get("visual_structure") or sample_meta.get("visual_structure", ""),
                "answer_area_frac": matched_metrics.get("answer_area_frac") or sample_meta.get("answer_area_frac", ""),
                "restore_effect_logit": round(float(hidden_metrics.get("restore_effect_logit", 0.0)), 6),
                "corrupt_effect_logit": round(float(hidden_metrics.get("corrupt_effect_logit", 0.0)), 6),
                "restore_gap_closure": round(float(hidden_metrics.get("restore_gap_closure", 0.0)), 6),
                "corrupt_gap_closure": round(float(hidden_metrics.get("corrupt_gap_closure", 0.0)), 6),
                "restore_source_combo": round(float(matched_metrics.get("restore_source_combo", 0.0)), 6),
                "corrupt_source_combo": round(float(matched_metrics.get("corrupt_source_combo", 0.0)), 6),
                "restore_minus_delta": round(float(matched_metrics.get("restore_minus_delta", 0.0)), 6),
                "restore_minus_activation": round(float(matched_metrics.get("restore_minus_activation", 0.0)), 6),
                "corrupt_minus_delta": round(float(matched_metrics.get("corrupt_minus_delta", 0.0)), 6),
                "corrupt_minus_activation": round(float(matched_metrics.get("corrupt_minus_activation", 0.0)), 6),
                "has_bidirectional_hidden_effect": has_bidirectional,
                "has_restore_specificity": has_specificity,
                "specificity_labels": matched_metrics.get("specificity_labels", ""),
            }
        )

    selected: list[dict[str, Any]] = []
    sample_quotas = {"qwen": 8, "llava": 6}
    for model, limit in sample_quotas.items():
        model_rows = [row for row in candidates if row["model_family"] == model]
        model_rows.sort(
            key=lambda row: (
                bool(row["has_bidirectional_hidden_effect"]),
                bool(row["has_restore_specificity"]),
                float(row["selection_score"]),
            ),
            reverse=True,
        )
        seen_samples: set[str] = set()
        for row in model_rows:
            sample_id = str(row["sample_id"])
            if sample_id in seen_samples:
                continue
            selected.append(row)
            seen_samples.add(sample_id)
            if len(seen_samples) >= limit:
                break
    selected.sort(key=lambda row: (row["model_family"], -float(row["selection_score"]), row["sample_id"], row["prompt_name"]))
    return selected


def _build_manifest(prompt_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_rows = _read_csv(SELECTED_MANIFEST)
    selected_ids = {row["sample_id"] for row in prompt_rows}
    model_map: dict[str, list[str]] = defaultdict(list)
    prompts_map: dict[str, list[str]] = defaultdict(list)
    for row in prompt_rows:
        model_map[row["sample_id"]].append(row["model_family"])
        prompts_map[row["sample_id"]].append(row["prompt_name"])

    out: list[dict[str, Any]] = []
    for row in source_rows:
        if row["sample_id"] not in selected_ids:
            continue
        item = dict(row)
        item["stage2m_decoded_models"] = ",".join(sorted(set(model_map[row["sample_id"]])))
        item["stage2m_decoded_prompts"] = ",".join(sorted(set(prompts_map[row["sample_id"]])))
        item["stage2m_decoded_selection_note"] = "passing_case_for_decoded_bridge"
        out.append(item)
    return out


def main() -> int:
    prompt_rows = _select_prompt_rows()
    manifest_rows = _build_manifest(prompt_rows)
    prompt_fields = [
        "model_family",
        "sample_id",
        "prompt_name",
        "selection_score",
        "reasoning_operation",
        "visual_structure",
        "answer_area_frac",
        "restore_effect_logit",
        "corrupt_effect_logit",
        "restore_gap_closure",
        "corrupt_gap_closure",
        "restore_source_combo",
        "corrupt_source_combo",
        "restore_minus_delta",
        "restore_minus_activation",
        "corrupt_minus_delta",
        "corrupt_minus_activation",
        "has_bidirectional_hidden_effect",
        "has_restore_specificity",
        "specificity_labels",
    ]
    manifest_fields = list(_read_csv(SELECTED_MANIFEST)[0].keys()) + [
        "stage2m_decoded_models",
        "stage2m_decoded_prompts",
        "stage2m_decoded_selection_note",
    ]
    _write_csv(OUT_PROMPT_ROWS, prompt_rows, prompt_fields)
    _write_csv(OUT_MANIFEST, manifest_rows, manifest_fields)
    decision = {
        "status": "pass",
        "prompt_rows": len(prompt_rows),
        "unique_samples": len({row["sample_id"] for row in prompt_rows}),
        "model_prompt_rows": {
            model: sum(1 for row in prompt_rows if row["model_family"] == model) for model in ["qwen", "llava"]
        },
        "model_unique_samples": {
            model: len({row["sample_id"] for row in prompt_rows if row["model_family"] == model})
            for model in ["qwen", "llava"]
        },
        "claim_boundary": (
            "This manifest only selects passing cases for decoded bridge smoke. It does not establish "
            "generation-level replication by itself."
        ),
    }
    _write_json(OUT_DECISION, decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
