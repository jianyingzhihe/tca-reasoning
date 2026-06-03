#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
STAGE6_CROSS = ROOT / "doc" / "experiments" / "stage6" / "cross_model"
GEMMA_PREFIX = "stage6_gemma_hidden_lattice_primary_smoke_gemmamask"


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
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _key(row: dict[str, str]) -> tuple[str, str]:
    return row.get("sample_id", ""), row.get("prompt_name", "")


def _gemma_variant_summary(variant: str, tag: str) -> dict[str, Any]:
    path = STAGE6_CROSS / f"{GEMMA_PREFIX}_{variant}_{tag}_raw.csv"
    rows = [
        row
        for row in _read_csv(path)
        if row.get("layer") == "1"
        and row.get("direction") == "restore"
        and row.get("position_group") == "visual+answer"
    ]
    by_condition_token: dict[tuple[str, str], dict[tuple[str, str], dict[str, str]]] = {}
    for row in rows:
        by_condition_token.setdefault((row.get("mask_condition", ""), row.get("token_scored", "")), {})[_key(row)] = row

    target_union = by_condition_token.get(("union_mask", "target"), {})
    target_answer = by_condition_token.get(("answer_mask", "target"), {})
    target_shifted = by_condition_token.get(("shifted_mask", "target"), {})
    target_shuffled = by_condition_token.get(("shuffled_mask", "target"), {})
    wrong_union = by_condition_token.get(("union_mask", "wrong"), {})

    common_shifted = sorted(set(target_union) & set(target_shifted))
    common_shuffled = sorted(set(target_union) & set(target_shuffled))
    common_wrong = sorted(set(target_union) & set(wrong_union))

    return {
        "model": "gemma",
        "variant": variant,
        "n": len(target_union),
        "lens": "hidden_lattice_layer1_visual+answer_restore",
        "target_effect_union_mean": _mean([_f(row.get("logit_effect")) for row in target_union.values()]),
        "target_effect_answer_mean": _mean([_f(row.get("logit_effect")) for row in target_answer.values()]),
        "target_rank_union_mean": _mean([_f(row.get("rank_effect")) for row in target_union.values()]),
        "real_minus_shifted_mean": _mean(
            [_f(target_union[key].get("logit_effect")) - _f(target_shifted[key].get("logit_effect")) for key in common_shifted]
        ),
        "real_minus_shuffled_mean": _mean(
            [_f(target_union[key].get("logit_effect")) - _f(target_shuffled[key].get("logit_effect")) for key in common_shuffled]
        ),
        "correct_minus_wrong_mean": _mean(
            [_f(target_union[key].get("logit_effect")) - _f(wrong_union[key].get("logit_effect")) for key in common_wrong]
        ),
        "evidence_specificity_mean": _mean(
            [
                _f(target_union[key].get("logit_effect"))
                - max(_f(target_shifted[key].get("logit_effect")), _f(target_shuffled[key].get("logit_effect")))
                for key in sorted(set(target_union) & set(target_shifted) & set(target_shuffled))
            ]
        ),
    }


def _qwen_rows(tag: str) -> list[dict[str, Any]]:
    path = STAGE6_CROSS / f"stage6_mask_robustness_smoke_{tag}_summary.csv"
    rows: list[dict[str, Any]] = []
    for row in _read_csv(path):
        rows.append(
            {
                "model": "qwen",
                "variant": row.get("variant", ""),
                "n": row.get("n", ""),
                "lens": "route_first_layer14",
                "target_effect_union_mean": "",
                "target_effect_answer_mean": "",
                "target_rank_union_mean": "",
                "real_minus_shifted_mean": row.get("real_minus_shifted_mean", ""),
                "real_minus_shuffled_mean": row.get("real_minus_shuffled_mean", ""),
                "correct_minus_wrong_mean": "",
                "evidence_specificity_mean": row.get("evidence_specificity_mean", ""),
                "route_first_234_frac": row.get("route_first_234_frac", ""),
                "route_first_evidence_gold_frac": row.get("route_first_evidence_gold_frac", ""),
                "restore_source_minus_controls_mean": row.get("restore_source_minus_controls_mean", ""),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="mask16_defensive_v1")
    args = parser.parse_args()

    rows = _qwen_rows(args.tag)
    rows.extend(_gemma_variant_summary(variant, args.tag) for variant in ["original", "dilate", "erode"])

    stem = f"stage6_crossmodel_mask_robustness_smoke_{args.tag}"
    fieldnames = [
        "model",
        "variant",
        "n",
        "lens",
        "target_effect_union_mean",
        "target_effect_answer_mean",
        "target_rank_union_mean",
        "real_minus_shifted_mean",
        "real_minus_shuffled_mean",
        "correct_minus_wrong_mean",
        "evidence_specificity_mean",
        "route_first_234_frac",
        "route_first_evidence_gold_frac",
        "restore_source_minus_controls_mean",
    ]
    _write_csv(STAGE6_CROSS / f"{stem}_summary.csv", rows, fieldnames)

    qwen_ready = all(_f(row.get("n")) > 0 for row in rows if row.get("model") == "qwen")
    gemma_ready = all(_f(row.get("n")) > 0 for row in rows if row.get("model") == "gemma")
    decision = {
        "tag": args.tag,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "crossmodel_mask_morphology_ready" if qwen_ready and gemma_ready else "blocked_missing_model_variant",
        "qwen_ready": qwen_ready,
        "gemma_ready": gemma_ready,
        "interpretation_boundary": (
            "Qwen route-first and Gemma hidden-lattice morphology are comparable as defensive mask-artifact lenses, "
            "but their raw effect magnitudes are not directly compared."
        ),
    }
    _write_json(STAGE6_CROSS / f"{stem}_decision.json", decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
