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


PRIMARY_GROUP = "top_hidden_delta_plus_answer_adjacent"
SOURCE_GROUPS = {
    "top_hidden_delta",
    "top_hidden_delta_plus_answer_adjacent",
    "evidence_region",
    "evidence_region_plus_answer_adjacent",
}
MATCHED_GROUPS = ["delta_matched_plus_answer_adjacent", "activation_matched_plus_answer_adjacent"]


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


def _float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except Exception:
        return None


def _mean(values: list[float]) -> float | str:
    return round(mean(values), 6) if values else ""


def _bootstrap(values: list[float], seed: int = 20260521, n_boot: int = 2000) -> tuple[float | str, float | str, float | str, str]:
    vals = [value for value in values if value is not None]
    if not vals:
        return "", "", "", "not_available"
    obs = mean(vals)
    if len(vals) == 1:
        low = high = obs
    else:
        rng = random.Random(seed)
        boots = []
        for _ in range(n_boot):
            boots.append(mean([vals[rng.randrange(len(vals))] for _ in vals]))
        boots.sort()
        low = boots[int(0.025 * len(boots))]
        high = boots[int(0.975 * len(boots)) - 1]
    if low > 0:
        status = "stable_positive"
    elif obs > 0:
        status = "weak_or_heterogeneous_positive"
    else:
        status = "not_positive"
    return round(obs, 6), round(low, 6), round(high, 6), status


def _manifest_index(path: Path) -> dict[str, dict[str, str]]:
    return {row["sample_id"]: row for row in _read_csv(path) if row.get("sample_id")}


def _stage2m_ids(path: Path) -> set[str]:
    return {row["sample_id"] for row in _read_csv(path) if row.get("sample_id")}


def _normalize_row(row: dict[str, str], model: str, source_pack: str, manifest: dict[str, dict[str, str]], stage2m_ids: set[str]) -> dict[str, Any]:
    sample_id = row.get("sample_id", "")
    meta = manifest.get(sample_id, {})
    mask_condition = row.get("mask_condition") or "union_mask"
    effect_logit = _float(row.get("effect_logit"))
    effect_rank = _float(row.get("effect_rank"))
    clean_mask_gap = _float(row.get("clean_mask_logit_gap"))
    if clean_mask_gap is None:
        clean_mask_gap = _float(row.get("clean_union_logit_gap"))
    clean_rank_gap = _float(row.get("clean_mask_rank_gap"))
    if clean_rank_gap is None:
        clean_rank_gap = _float(row.get("clean_union_rank_gap"))
    return {
        **row,
        "model_family": row.get("model_family") or model,
        "source_pack": source_pack,
        "mask_condition": mask_condition,
        "reasoning_operation": meta.get("reasoning_operation", "unknown") or "unknown",
        "visual_structure": meta.get("visual_structure", ""),
        "answer_area_frac": meta.get("answer_area_frac", ""),
        "is_stage2m": sample_id in stage2m_ids,
        "is_stage2n_heldout": sample_id not in stage2m_ids,
        "effect_logit_f": effect_logit,
        "effect_rank_f": effect_rank,
        "effect_gap_closure_f": _float(row.get("effect_gap_closure")),
        "clean_mask_logit_gap_f": clean_mask_gap,
        "clean_mask_rank_gap_f": clean_rank_gap,
    }


def _load_rows(
    *,
    qwen_stage2n: Path,
    llava_stage2n: Path,
    qwen_stage2m: Path,
    llava_stage2m: Path,
    manifest: Path,
    stage2m_manifest: Path,
) -> list[dict[str, Any]]:
    meta = _manifest_index(manifest)
    stage2m_ids = _stage2m_ids(stage2m_manifest)
    out: list[dict[str, Any]] = []
    for model, source_pack, path in [
        ("qwen", "stage2n_heldout", qwen_stage2n),
        ("llava", "stage2n_heldout", llava_stage2n),
        ("qwen", "stage2m", qwen_stage2m),
        ("llava", "stage2m", llava_stage2m),
    ]:
        for row in _read_csv(path):
            out.append(_normalize_row(row, model, source_pack, meta, stage2m_ids))
    return out


def _group(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    out: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[tuple(str(row.get(key, "")) for key in keys)].append(row)
    return out


def _summary(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    out = []
    for key, items in sorted(_group(rows, group_keys).items()):
        active = [row for row in items if row.get("direction") in {"restore", "corrupt"}]
        if not active:
            continue
        base = {field: value for field, value in zip(group_keys, key, strict=False)}
        effects = [row["effect_logit_f"] for row in active if row["effect_logit_f"] is not None]
        rank_effects = [row["effect_rank_f"] for row in active if row["effect_rank_f"] is not None]
        gap_effects = [row["effect_gap_closure_f"] for row in active if row["effect_gap_closure_f"] is not None]
        obs, low, high, status = _bootstrap(effects)
        base.update(
            {
                "n_rows": len(active),
                "n_samples": len({row.get("sample_id", "") for row in active}),
                "mean_effect_logit": obs,
                "ci95_low": low,
                "ci95_high": high,
                "effect_status": status,
                "positive_effect_logit_n": sum(1 for value in effects if value > 0),
                "mean_effect_rank": _mean(rank_effects),
                "positive_effect_rank_n": sum(1 for value in rank_effects if value > 0),
                "mean_effect_gap_closure": _mean(gap_effects),
                "mean_clean_mask_logit_gap": _mean([row["clean_mask_logit_gap_f"] for row in active if row["clean_mask_logit_gap_f"] is not None]),
            }
        )
        out.append(base)
    return out


def _case_specificity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active = [row for row in rows if row.get("direction") in {"restore", "corrupt"}]
    grouped = _group(active, ["source_pack", "model_family", "sample_id", "prompt_name", "mask_condition", "direction"])
    out: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        source_pack, model, sample_id, prompt, mask_condition, direction = key
        by_group = {row.get("group_name", ""): row for row in items}
        source = by_group.get(PRIMARY_GROUP)
        if source is None:
            continue
        random_rows = [row for name, row in by_group.items() if name.startswith("random_control_")]
        random_mean = _mean([row["effect_logit_f"] for row in random_rows if row["effect_logit_f"] is not None])
        random_rank_mean = _mean([row["effect_rank_f"] for row in random_rows if row["effect_rank_f"] is not None])
        low = by_group.get("low_delta_control")
        delta = by_group.get("delta_matched_plus_answer_adjacent")
        activation = by_group.get("activation_matched_plus_answer_adjacent")
        source_effect = source["effect_logit_f"]
        source_rank = source["effect_rank_f"]

        def diff(other: float | str | None) -> float | str:
            if source_effect is None or other in ("", None):
                return ""
            return round(source_effect - float(other), 6)

        def diff_rank(other: float | str | None) -> float | str:
            if source_rank is None or other in ("", None):
                return ""
            return round(source_rank - float(other), 6)

        out.append(
            {
                "source_pack": source_pack,
                "model_family": model,
                "sample_id": sample_id,
                "prompt_name": prompt,
                "mask_condition": mask_condition,
                "direction": direction,
                "reasoning_operation": source.get("reasoning_operation", "unknown"),
                "source_effect_logit": source_effect,
                "source_effect_rank": source_rank,
                "random_mean_effect_logit": random_mean,
                "source_minus_random_logit": diff(random_mean),
                "random_mean_effect_rank": random_rank_mean,
                "source_minus_random_rank": diff_rank(random_rank_mean),
                "low_delta_effect_logit": "" if low is None else low["effect_logit_f"],
                "source_minus_low_delta_logit": "" if low is None else diff(low["effect_logit_f"]),
                "delta_matched_effect_logit": "" if delta is None else delta["effect_logit_f"],
                "source_minus_delta_matched_logit": "" if delta is None else diff(delta["effect_logit_f"]),
                "activation_matched_effect_logit": "" if activation is None else activation["effect_logit_f"],
                "source_minus_activation_matched_logit": "" if activation is None else diff(activation["effect_logit_f"]),
            }
        )
    return out


def _specificity_summary(cases: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    metrics = [
        "source_effect_logit",
        "source_minus_random_logit",
        "source_minus_low_delta_logit",
        "source_minus_delta_matched_logit",
        "source_minus_activation_matched_logit",
    ]
    out = []
    for key, items in sorted(_group(cases, keys).items()):
        base = {field: value for field, value in zip(keys, key, strict=False)}
        base["n_rows"] = len(items)
        base["n_samples"] = len({row["sample_id"] for row in items})
        for metric in metrics:
            values = [_float(row.get(metric)) for row in items]
            valid = [value for value in values if value is not None]
            obs, low, high, status = _bootstrap(valid)
            base[f"{metric}_mean"] = obs
            base[f"{metric}_ci95_low"] = low
            base[f"{metric}_ci95_high"] = high
            base[f"{metric}_status"] = status
            base[f"{metric}_positive_n"] = sum(1 for value in valid if value > 0)
        out.append(base)
    return out


def _decision(spec_summary: list[dict[str, Any]]) -> dict[str, Any]:
    decision: dict[str, Any] = {
        "claim_boundary": (
            "Stage 2N evaluates hidden-state heldout replication. It does not establish CLT feature-level "
            "or Gemma-style source-control route replication."
        ),
        "model_decisions": {},
    }
    for model in sorted({row["model_family"] for row in spec_summary if row.get("source_pack") == "stage2n_heldout"}):
        model_rows = [
            row
            for row in spec_summary
            if row.get("source_pack") == "stage2n_heldout"
            and row.get("model_family") == model
            and row.get("direction") in {"restore", "corrupt"}
        ]
        primary = [
            row
            for row in model_rows
            if row.get("mask_condition") in {"answer_mask", "union_mask"}
        ]
        source_random_positive = [
            row for row in primary if row.get("source_minus_random_logit_status") in {"stable_positive", "weak_or_heterogeneous_positive"}
        ]
        source_random_stable = [row for row in primary if row.get("source_minus_random_logit_status") == "stable_positive"]
        matched_positive = [
            row
            for row in primary
            if row.get("source_minus_delta_matched_logit_status") in {"stable_positive", "weak_or_heterogeneous_positive"}
            and row.get("source_minus_activation_matched_logit_status") in {"stable_positive", "weak_or_heterogeneous_positive"}
        ]
        if len(source_random_stable) >= 2 and len(matched_positive) >= 1:
            status = "heldout_hidden_replication_supported"
        elif len(source_random_positive) >= 2:
            status = "heldout_hidden_replication_partial"
        else:
            status = "heldout_hidden_replication_not_supported"
        decision["model_decisions"][model] = {
            "status": status,
            "source_random_positive_rows": len(source_random_positive),
            "source_random_stable_rows": len(source_random_stable),
            "matched_positive_rows": len(matched_positive),
            "primary_rows": primary,
        }
    statuses = [item["status"] for item in decision["model_decisions"].values()]
    if statuses and all(status == "heldout_hidden_replication_supported" for status in statuses):
        decision["overall_status"] = "cross_model_heldout_hidden_replication_supported"
    elif any(status in {"heldout_hidden_replication_supported", "heldout_hidden_replication_partial"} for status in statuses):
        decision["overall_status"] = "cross_model_heldout_hidden_replication_partial"
    else:
        decision["overall_status"] = "cross_model_heldout_hidden_replication_not_supported"
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage 2N hidden-state heldout replication.")
    parser.add_argument("--qwen-stage2n", required=True)
    parser.add_argument("--llava-stage2n", required=True)
    parser.add_argument("--qwen-stage2m", required=True)
    parser.add_argument("--llava-stage2m", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stage2m-manifest", required=True)
    parser.add_argument("--out-heldout-summary", required=True)
    parser.add_argument("--out-all52-summary", required=True)
    parser.add_argument("--out-specificity-case", required=True)
    parser.add_argument("--out-specificity-summary", required=True)
    parser.add_argument("--out-typed-summary", required=True)
    parser.add_argument("--out-mask-summary", required=True)
    parser.add_argument("--out-prompt-summary", required=True)
    parser.add_argument("--out-decision", required=True)
    args = parser.parse_args()

    rows = _load_rows(
        qwen_stage2n=Path(args.qwen_stage2n),
        llava_stage2n=Path(args.llava_stage2n),
        qwen_stage2m=Path(args.qwen_stage2m),
        llava_stage2m=Path(args.llava_stage2m),
        manifest=Path(args.manifest),
        stage2m_manifest=Path(args.stage2m_manifest),
    )
    heldout_rows = [row for row in rows if row["source_pack"] == "stage2n_heldout"]
    all52_rows = [row for row in rows if row.get("source_pack") in {"stage2n_heldout", "stage2m"}]
    heldout_summary = _summary(heldout_rows, ["source_pack", "model_family", "mask_condition", "direction", "group_name", "group_kind"])
    all52_summary = _summary(all52_rows, ["model_family", "mask_condition", "direction", "group_name", "group_kind"])
    specificity_cases = _case_specificity(rows)
    specificity_summary = _specificity_summary(specificity_cases, ["source_pack", "model_family", "mask_condition", "direction"])
    typed_summary = _specificity_summary(specificity_cases, ["source_pack", "model_family", "reasoning_operation", "mask_condition", "direction"])
    mask_summary = _specificity_summary(specificity_cases, ["source_pack", "model_family", "mask_condition"])
    prompt_summary = _specificity_summary(specificity_cases, ["source_pack", "model_family", "prompt_name", "mask_condition"])
    decision = _decision(specificity_summary)

    summary_fields = list(heldout_summary[0].keys()) if heldout_summary else []
    all52_fields = list(all52_summary[0].keys()) if all52_summary else []
    case_fields = list(specificity_cases[0].keys()) if specificity_cases else []
    spec_fields = list(specificity_summary[0].keys()) if specificity_summary else []
    typed_fields = list(typed_summary[0].keys()) if typed_summary else []
    mask_fields = list(mask_summary[0].keys()) if mask_summary else []
    prompt_fields = list(prompt_summary[0].keys()) if prompt_summary else []
    _write_csv(Path(args.out_heldout_summary), heldout_summary, summary_fields)
    _write_csv(Path(args.out_all52_summary), all52_summary, all52_fields)
    _write_csv(Path(args.out_specificity_case), specificity_cases, case_fields)
    _write_csv(Path(args.out_specificity_summary), specificity_summary, spec_fields)
    _write_csv(Path(args.out_typed_summary), typed_summary, typed_fields)
    _write_csv(Path(args.out_mask_summary), mask_summary, mask_fields)
    _write_csv(Path(args.out_prompt_summary), prompt_summary, prompt_fields)
    _write_json(Path(args.out_decision), decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
