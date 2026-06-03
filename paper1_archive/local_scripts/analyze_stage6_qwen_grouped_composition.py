#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("STAGE6_DEFENSIVE_ROOT", r"E:\Bridging"))
STAGE4_CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"
STAGE6_CROSS = Path(
    os.environ.get(
        "STAGE6_DEFENSIVE_OUT_DIR",
        str(ROOT / "doc" / "experiments" / "stage6" / "cross_model"),
    )
)


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


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value not in ("", None) else default
    except (TypeError, ValueError):
        return default


def _i(value: Any, default: int = 0) -> int:
    try:
        return int(float(value)) if value not in ("", None) else default
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _layer_band(layer: int) -> str:
    if 10 <= layer <= 12:
        return "L10-L12"
    if 13 <= layer <= 15:
        return "L13-L15"
    if 16 <= layer <= 17:
        return "L16-L17"
    if layer == 14:
        return "L14-near"
    return "other"


def _limit_smoke(rows: list[dict[str, str]], max_samples: int, max_prompts: int, topks: set[int]) -> list[dict[str, str]]:
    sample_order: list[str] = []
    prompt_order: dict[str, list[str]] = {}
    kept: list[dict[str, str]] = []
    for row in rows:
        sample = row.get("sample_id", "")
        prompt = row.get("prompt_name", "")
        topk = _i(row.get("topk"))
        if topk not in topks:
            continue
        if sample not in sample_order:
            if len(sample_order) >= max_samples:
                continue
            sample_order.append(sample)
            prompt_order[sample] = []
        if prompt not in prompt_order.setdefault(sample, []):
            if len(prompt_order[sample]) >= max_prompts:
                continue
            prompt_order[sample].append(prompt)
        if sample in sample_order and prompt in prompt_order[sample]:
            kept.append(row)
    return kept


def summarize_grouped(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (row.get("pack", ""), row.get("mode", ""), _i(row.get("topk")))
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (pack, mode, topk), items in sorted(grouped.items()):
        out.append(
            {
                "pack": pack,
                "mode": mode,
                "topk": topk,
                "n": len(items),
                "route_restore_source_minus_controls_mean": _mean([_f(r.get("route_restore_source_minus_controls")) for r in items]),
                "route_real_minus_shifted_mean": _mean([_f(r.get("route_real_minus_shifted")) for r in items]),
                "route_real_minus_shuffled_mean": _mean([_f(r.get("route_real_minus_shuffled")) for r in items]),
                "route_correct_minus_wrong_mean": _mean([_f(r.get("route_correct_minus_wrong")) for r in items]),
                "gate1_frac": _mean([_f(r.get("gate1_evidence_sensitive")) for r in items]),
                "gate2_frac": _mean([_f(r.get("gate2_clean_source_gt_controls")) for r in items]),
                "gate3_frac": _mean([_f(r.get("gate3_restore_source_gt_controls")) for r in items]),
                "gate4_frac": _mean([_f(r.get("gate4_real_gt_shifted_shuffled")) for r in items]),
                "gate5_frac": _mean([_f(r.get("gate5_correct_gt_wrong")) for r in items]),
                "all_gates_frac": _mean([_f(r.get("route_first_all_gates")) for r in items]),
            }
        )
    return out


def summarize_layer_bands(candidates: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in candidates:
        layer = _i(row.get("layer"))
        band = _layer_band(layer)
        if band == "other":
            continue
        grouped.setdefault((row.get("pack", ""), band), []).append(row)

    out: list[dict[str, Any]] = []
    for (pack, band), items in sorted(grouped.items()):
        out.append(
            {
                "pack": pack,
                "layer_band": band,
                "n": len(items),
                "restore_source_minus_controls_mean": _mean([_f(r.get("restore_source_minus_controls")) for r in items]),
                "real_minus_shifted_mean": _mean([_f(r.get("real_minus_shifted")) for r in items]),
                "real_minus_shuffled_mean": _mean([_f(r.get("real_minus_shuffled")) for r in items]),
                "restore_correct_minus_wrong_mean": _mean([_f(r.get("restore_correct_minus_wrong")) for r in items]),
                "route_first_234_frac": _mean([_f(r.get("route_first_234")) for r in items]),
                "route_first_gold_frac": _mean([_f(r.get("route_first_gold")) for r in items]),
                "route_first_evidence_gold_frac": _mean([_f(r.get("route_first_evidence_gold")) for r in items]),
            }
        )
    return out


def summarize_topk_nonmonotonic(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("route_base_id", ""), []).append(row)

    out: list[dict[str, Any]] = []
    for base, items in grouped.items():
        by_topk = {_i(row.get("topk")): _f(row.get("route_restore_source_minus_controls")) for row in items}
        if len(by_topk) < 2:
            continue
        ordered = sorted(by_topk.items())
        best_topk, best_value = max(ordered, key=lambda pair: pair[1])
        largest_topk, largest_value = ordered[-1]
        out.append(
            {
                "route_base_id": base,
                "best_topk": best_topk,
                "best_restore_source_minus_controls": best_value,
                "largest_topk": largest_topk,
                "largest_restore_source_minus_controls": largest_value,
                "nonmonotonic_drop": best_value - largest_value,
                "is_nonmonotonic": int(best_topk != largest_topk and best_value > largest_value),
            }
        )
    out.sort(key=lambda row: _f(row["nonmonotonic_drop"]), reverse=True)
    return out


def run(mode: str, tag: str, max_samples: int, max_prompts: int) -> dict[str, Any]:
    route_metrics = _read_csv(STAGE4_CROSS / "stage4_qwen_feature_route_featureroute_v1_route_metrics.csv")
    candidates = _read_csv(STAGE4_CROSS / "stage4_qwen_route_first_routefirst_v1_route_candidates.csv")
    if mode == "smoke":
        route_metrics = _limit_smoke(route_metrics, max_samples=max_samples, max_prompts=max_prompts, topks={1, 4, 8})
        sample_prompt = {(row.get("sample_id", ""), row.get("prompt_name", "")) for row in route_metrics}
        candidates = [row for row in candidates if (row.get("sample_id", ""), row.get("prompt_name", "")) in sample_prompt]

    topk_rows = summarize_grouped(route_metrics)
    band_rows = summarize_layer_bands(candidates)
    nonmono_rows = summarize_topk_nonmonotonic(route_metrics)

    stem = f"stage6_defensive_qwen_grouped_composition_{mode}_{tag}"
    _write_csv(
        STAGE6_CROSS / f"{stem}_topk_summary.csv",
        topk_rows,
        [
            "pack",
            "mode",
            "topk",
            "n",
            "route_restore_source_minus_controls_mean",
            "route_real_minus_shifted_mean",
            "route_real_minus_shuffled_mean",
            "route_correct_minus_wrong_mean",
            "gate1_frac",
            "gate2_frac",
            "gate3_frac",
            "gate4_frac",
            "gate5_frac",
            "all_gates_frac",
        ],
    )
    _write_csv(
        STAGE6_CROSS / f"{stem}_layer_band_summary.csv",
        band_rows,
        [
            "pack",
            "layer_band",
            "n",
            "restore_source_minus_controls_mean",
            "real_minus_shifted_mean",
            "real_minus_shuffled_mean",
            "restore_correct_minus_wrong_mean",
            "route_first_234_frac",
            "route_first_gold_frac",
            "route_first_evidence_gold_frac",
        ],
    )
    _write_csv(
        STAGE6_CROSS / f"{stem}_topk_nonmonotonic.csv",
        nonmono_rows[:50],
        [
            "route_base_id",
            "best_topk",
            "best_restore_source_minus_controls",
            "largest_topk",
            "largest_restore_source_minus_controls",
            "nonmonotonic_drop",
            "is_nonmonotonic",
        ],
    )

    status = "qwen_grouped_composition_artifact_smoke_ready" if topk_rows and band_rows else "blocked_missing_artifact"
    nonmono_frac = _mean([_f(row["is_nonmonotonic"]) for row in nonmono_rows])
    decision = {
        "tag": tag,
        "mode": mode,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "route_metric_rows": len(route_metrics),
        "candidate_rows": len(candidates),
        "topk_summary_rows": len(topk_rows),
        "layer_band_rows": len(band_rows),
        "nonmonotonic_routes": sum(1 for row in nonmono_rows if _i(row["is_nonmonotonic"])),
        "nonmonotonic_frac": nonmono_frac,
        "interpretation": "artifact-level diagnostic; true support-only regrouped intervention remains a follow-up if needed",
    }
    _write_json(STAGE6_CROSS / f"{stem}_decision.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--tag", default="defensive_v1")
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--max-prompts", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args.mode, args.tag, args.max_samples, args.max_prompts), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
