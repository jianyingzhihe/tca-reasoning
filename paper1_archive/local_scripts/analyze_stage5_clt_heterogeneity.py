#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
CROSS = ROOT / "doc" / "experiments" / "stage5" / "cross_model"
PREFIX = "stage5_clt_heterogeneity"
ASSET_LABEL = {"qwen_clt": "qwen2p5vl_clt", "llava_clt": "llava15_clt"}
CONTROL_GROUPS = [
    "activation_matched_topk",
    "drop_matched_topk",
    "attribution_matched_mask_insensitive_topk",
    "random_active_topk",
]


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _f(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return math.nan


def _mean(values: list[float]) -> float:
    vals = [value for value in values if not math.isnan(value)]
    return statistics.fmean(vals) if vals else math.nan


def _ci(values: list[float], seed: int = 5050, n: int = 1000) -> tuple[float, float]:
    vals = [value for value in values if not math.isnan(value)]
    if not vals:
        return math.nan, math.nan
    if len(vals) == 1:
        return vals[0], vals[0]
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        means.append(statistics.fmean(sample))
    means.sort()
    return means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]


def _metric(metric: str, values: list[float], extra: dict[str, Any]) -> dict[str, Any]:
    vals = [value for value in values if not math.isnan(value)]
    lo, hi = _ci(vals)
    row = {
        "metric": metric,
        "n": len(vals),
        "mean": _mean(vals),
        "ci95_low": lo,
        "ci95_high": hi,
        "positive_frac": sum(1 for value in vals if value > 0) / len(vals) if vals else math.nan,
    }
    row.update(extra)
    return row


def _feature_specificity(rows: list[dict[str, str]], extra: dict[str, Any]) -> list[dict[str, Any]]:
    if not rows:
        return [_metric("feature_source_minus_controls", [], extra)]
    by_key: dict[tuple[str, ...], dict[str, float]] = defaultdict(dict)
    for row in rows:
        direction = row.get("direction", "")
        effect = _f(row, "logit_restore_vs_mask") if direction == "restore" else _f(row, "logit_damage_vs_clean")
        key = (
            row.get("sample_id", ""),
            row.get("prompt_name", ""),
            row.get("position_group", ""),
            direction,
            row.get("mask_condition", ""),
        )
        by_key[key][row.get("feature_group", "")] = effect
    diffs = []
    for values in by_key.values():
        source = values.get("evidence_attribution_topk")
        if source is None or math.isnan(source):
            continue
        controls = [values[name] for name in CONTROL_GROUPS if name in values and not math.isnan(values[name])]
        if controls:
            diffs.append(source - statistics.fmean(controls))
    return [_metric("feature_source_minus_controls", diffs, extra)]


def _source_specificity(rows: list[dict[str, str]], extra: dict[str, Any]) -> list[dict[str, Any]]:
    if not rows:
        return [
            _metric("source_minus_control", [], extra),
            _metric("real_minus_shuffled", [], extra),
        ]
    by_key: dict[tuple[str, ...], dict[str, float]] = defaultdict(dict)
    real_idx: dict[tuple[str, ...], dict[str, float]] = defaultdict(dict)
    for row in rows:
        effect = _f(row, "effect_logit")
        key = (
            row.get("sample_id", ""),
            row.get("prompt_name", ""),
            row.get("mask_condition", ""),
            row.get("mask_variant", ""),
            row.get("intervention", ""),
        )
        by_key[key][row.get("feature_role", "")] = effect
        if row.get("feature_role") == "source":
            real_key = (
                row.get("sample_id", ""),
                row.get("prompt_name", ""),
                row.get("mask_condition", ""),
                row.get("intervention", ""),
            )
            real_idx[real_key][row.get("mask_variant", "")] = effect
    source_control = []
    for values in by_key.values():
        if "source" in values and "matched_control" in values:
            source_control.append(values["source"] - values["matched_control"])
    real_shuffled = []
    for values in real_idx.values():
        if "real_mask" in values and "mask_shuffled" in values:
            real_shuffled.append(values["real_mask"] - values["mask_shuffled"])
    return [
        _metric("source_minus_control", source_control, extra),
        _metric("real_minus_shuffled", real_shuffled, extra),
    ]


def _positive(row: dict[str, Any], min_n: int = 24) -> bool:
    try:
        return int(row.get("n", 0)) >= min_n and float(row.get("ci95_low", math.nan)) > 0 and float(row.get("positive_frac", 0)) >= 0.55
    except (TypeError, ValueError):
        return False


def analyze(asset: str, pack: str, mode: str) -> dict[str, Any]:
    label = ASSET_LABEL[asset]
    summary: list[dict[str, Any]] = []
    specificity: list[dict[str, Any]] = []
    for feature_path in sorted(CROSS.glob(f"{PREFIX}_{label}_{pack}_{mode}_L*_topK*_feature_union.csv")):
        stem = feature_path.name[:-len("_feature_union.csv")]
        parts = stem.split("_L")
        layer_s, topk_s = parts[1].split("_topK")
        layer = int(layer_s)
        topk = int(topk_s)
        source_path = CROSS / f"{stem}_source_control.csv"
        feature = _read(feature_path)
        source = _read(source_path)
        extra = {"asset": asset, "pack": pack, "mode": mode, "layer": layer, "topk": topk}
        summary.append(
            {
                **extra,
                "feature_rows": len(feature),
                "feature_prompt_runs": len({(row.get("sample_id", ""), row.get("prompt_name", "")) for row in feature}),
                "source_rows": len(source),
                "source_prompt_runs": len({(row.get("sample_id", ""), row.get("prompt_name", "")) for row in source}),
            }
        )
        specificity.extend(_feature_specificity(feature, extra))
        specificity.extend(_source_specificity(source, extra))
    _write_csv(CROSS / f"{PREFIX}_{asset}_{pack}_{mode}_summary.csv", summary)
    _write_csv(CROSS / f"{PREFIX}_{asset}_{pack}_{mode}_specificity.csv", specificity)
    near_pass = [
        row for row in specificity
        if row.get("metric") in {"feature_source_minus_controls", "source_minus_control", "real_minus_shuffled"}
        and _positive(row)
    ]
    usable = [row for row in summary if row.get("feature_rows", 0) or row.get("source_rows", 0)]
    if not usable:
        status = "blocked"
    elif asset == "qwen_clt" and near_pass:
        status = "qwen_clt_route_screen_near_pass"
    elif asset == "llava_clt" and near_pass:
        status = "llava_clt_layer_dependent_weak_route_screen"
    else:
        status = "clt_not_established_under_tested_assets"
    decision = {
        "status": status,
        "asset": asset,
        "pack": pack,
        "mode": mode,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "usable_configs": len(usable),
        "near_pass_metrics": len(near_pass),
        "near_pass_configs": sorted({f"L{row['layer']}_topK{row['topk']}" for row in near_pass}),
        "claim_boundary": "Stage5 CLT is heterogeneity/representation mapping; failure does not imply absence of a cross-modal mechanism.",
    }
    _write_json(CROSS / f"{PREFIX}_{asset}_{pack}_{mode}_decision.json", decision)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage5 CLT heterogeneity outputs.")
    parser.add_argument("--asset", choices=["qwen_clt", "llava_clt"], required=True)
    parser.add_argument("--pack", choices=["primary", "strict"], default="primary")
    parser.add_argument("--mode", choices=["smoke", "screen", "full", "strict-confirm"], default="smoke")
    args = parser.parse_args()
    print(json.dumps(analyze(args.asset, args.pack, args.mode), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
