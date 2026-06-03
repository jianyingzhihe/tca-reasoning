#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(r"E:\Bridging")
STAGE3 = ROOT / "doc" / "experiments" / "stage3"
STAGE3_CROSS = STAGE3 / "cross_model"

STAGE3_MANIFEST = STAGE3_CROSS / "stage3_aligned24_manifest.csv"
GEMMA_RANDOM16 = (
    ROOT
    / "remote_sync"
    / "2026-05-19_core24_prefixfix_random16"
    / "region_mask_mainline_prefixfix_random16.csv"
)
GEMMA_SUPPORT_PAIRS = (
    ROOT
    / "doc"
    / "5.16"
    / "core24_prefixfix_region_analysis_2026-05-19"
    / "random16_analysis"
    / "support_source_nearest_pairs_iou0p06.csv"
)

OUT_OVERLAP_MANIFEST = STAGE3_CROSS / "stage3_gemma_overlap_manifest.csv"
OUT_RAW = STAGE3_CROSS / "stage3_gemma_overlap_raw_rows.csv"
OUT_SUPPORT_PAIRS = STAGE3_CROSS / "stage3_gemma_overlap_support_pairs.csv"
OUT_SUMMARY = STAGE3_CROSS / "stage3_gemma_overlap_summary.csv"
OUT_MISSING = STAGE3_CROSS / "stage3_gemma_missing_source_samples.csv"
OUT_DECISION = STAGE3_CROSS / "stage3_gemma_overlap_decision.json"

MASK_CONDITIONS = ["answer_mask", "union_mask"]
RANDOM_PREFIX = "random_control_"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def to_float(value: str | float | int | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def bootstrap_ci(values: list[float], *, seed: int = 1729, rounds: int = 2000) -> tuple[float | str, float | str]:
    if not values:
        return "", ""
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed + len(values))
    means: list[float] = []
    for _ in range(rounds):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(mean(sample))
    means.sort()
    return means[int(0.025 * (rounds - 1))], means[int(0.975 * (rounds - 1))]


def stable_seed(*parts: str) -> int:
    value = 1729
    for part in parts:
        for char in part:
            value = (value * 131 + ord(char)) % (2**31)
    return value


def summarize_values(rows: list[dict[str, Any]], metric: str, group_fields: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = to_float(row.get(metric))
        if value is None:
            continue
        key = tuple(str(row.get(field, "")) for field in group_fields)
        grouped[key].append(value)

    out: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        ci_low, ci_high = bootstrap_ci(values, seed=stable_seed(metric, *key))
        record = {field: value for field, value in zip(group_fields, key)}
        record.update(
            {
                "metric": metric,
                "n": len(values),
                "mean": mean(values),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "positive_count": sum(value > 0 for value in values),
                "positive_rate": sum(value > 0 for value in values) / len(values),
            }
        )
        out.append(record)
    return out


def build_overlap_manifest(stage3_rows: list[dict[str, str]], gemma_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in gemma_rows:
        by_sample[row.get("sample_id", "")].append(row)

    out: list[dict[str, Any]] = []
    for row in stage3_rows:
        sample_id = row.get("sample_id", "")
        rows = by_sample.get(sample_id, [])
        support_sources = [
            item
            for item in rows
            if item.get("node_role") == "support"
            and item.get("node_source") == "source"
            and item.get("condition") == "clean"
        ]
        support_nearest = [
            item
            for item in rows
            if item.get("node_role") == "support"
            and item.get("node_source") == "nearest_control"
            and item.get("condition") == "clean"
        ]
        status = "available" if support_sources and support_nearest else "missing_gemma_source_control"
        out.append(
            {
                "sample_id": sample_id,
                "stage3_selection_rank": row.get("stage3_selection_rank", ""),
                "reasoning_operation": row.get("reasoning_operation", ""),
                "image_dependence": row.get("image_dependence", ""),
                "mask_pack": row.get("mask_pack", ""),
                "gemma_overlap_status": status,
                "gemma_raw_rows": len(rows),
                "support_source_clean_rows": len(support_sources),
                "support_nearest_clean_rows": len(support_nearest),
                "available_runs": ",".join(sorted({item.get("run", "") for item in support_sources if item.get("run", "")})),
                "available_prompt_names": ",".join(
                    sorted({item.get("prompt_name", "") for item in support_sources if item.get("prompt_name", "")})
                ),
            }
        )
    return out


def build_condition_effects(gemma_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_key_condition: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for row in gemma_rows:
        key = (
            row.get("sample_id", ""),
            row.get("run", ""),
            row.get("prompt_name", ""),
            row.get("node_role", ""),
            row.get("node_source", ""),
            row.get("condition", ""),
        )
        by_key_condition[key] = row

    effects: list[dict[str, Any]] = []
    run_keys = sorted(
        {
            (
                row.get("sample_id", ""),
                row.get("run", ""),
                row.get("prompt_name", ""),
                row.get("node_role", ""),
                row.get("node_source", ""),
            )
            for row in gemma_rows
        }
    )
    for sample_id, run, prompt_name, node_role, node_source in run_keys:
        clean = by_key_condition.get((sample_id, run, prompt_name, node_role, node_source, "clean"))
        if clean is None:
            continue
        clean_delta = to_float(clean.get("delta_target_logit"))
        if clean_delta is None:
            continue
        random_values: list[float] = []
        for idx in range(1, 17):
            random_row = by_key_condition.get(
                (sample_id, run, prompt_name, node_role, node_source, f"{RANDOM_PREFIX}{idx}")
            )
            random_delta = to_float(random_row.get("delta_target_logit") if random_row else None)
            if random_delta is not None:
                random_values.append(random_delta - clean_delta)
        random16_mean = mean(random_values) if random_values else None

        for condition in MASK_CONDITIONS:
            cond = by_key_condition.get((sample_id, run, prompt_name, node_role, node_source, condition))
            cond_delta = to_float(cond.get("delta_target_logit") if cond else None)
            if cond_delta is None:
                continue
            weakening = cond_delta - clean_delta
            effects.append(
                {
                    "sample_id": sample_id,
                    "run": run,
                    "prompt_name": prompt_name,
                    "node_role": node_role,
                    "node_source": node_source,
                    "condition": condition,
                    "clean_delta_target_logit": clean_delta,
                    "condition_delta_target_logit": cond_delta,
                    "weakening": weakening,
                    "random16_weakening_mean": random16_mean if random16_mean is not None else "",
                    "mask_minus_random16": weakening - random16_mean if random16_mean is not None else "",
                    "reasoning_operation": clean.get("reasoning_operation", ""),
                    "visual_structure": clean.get("visual_structure", ""),
                    "image_dependence": clean.get("image_dependence", ""),
                }
            )
    return effects


def build_source_nearest_comparisons(effects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in effects:
        if row.get("node_role") != "support":
            continue
        key = (
            str(row.get("sample_id", "")),
            str(row.get("run", "")),
            str(row.get("prompt_name", "")),
            str(row.get("condition", "")),
            str(row.get("reasoning_operation", "")),
        )
        by_key[key][str(row.get("node_source", ""))] = row

    out: list[dict[str, Any]] = []
    for (sample_id, run, prompt_name, condition, reasoning_operation), values in sorted(by_key.items()):
        source = values.get("source")
        nearest = values.get("nearest_control")
        if source is None or nearest is None:
            continue
        source_weakening = to_float(source.get("weakening"))
        nearest_weakening = to_float(nearest.get("weakening"))
        source_minus_random16 = to_float(source.get("mask_minus_random16"))
        if source_weakening is None or nearest_weakening is None:
            continue
        out.append(
            {
                "sample_id": sample_id,
                "run": run,
                "prompt_name": prompt_name,
                "condition": condition,
                "reasoning_operation": reasoning_operation,
                "source_weakening": source_weakening,
                "nearest_weakening": nearest_weakening,
                "source_minus_nearest": source_weakening - nearest_weakening,
                "source_random16_weakening_mean": source.get("random16_weakening_mean", ""),
                "source_minus_random16": source_minus_random16 if source_minus_random16 is not None else "",
            }
        )
    return out


def main() -> None:
    stage3_rows = read_csv(STAGE3_MANIFEST)
    gemma_all_rows = read_csv(GEMMA_RANDOM16)
    support_pair_rows = read_csv(GEMMA_SUPPORT_PAIRS) if GEMMA_SUPPORT_PAIRS.exists() else []

    stage3_ids = {row.get("sample_id", "") for row in stage3_rows}
    gemma_overlap_rows = [row for row in gemma_all_rows if row.get("sample_id", "") in stage3_ids]
    support_pair_overlap = [row for row in support_pair_rows if row.get("sample_id", "") in stage3_ids]
    overlap_manifest = build_overlap_manifest(stage3_rows, gemma_overlap_rows)
    missing = [row for row in overlap_manifest if row.get("gemma_overlap_status") != "available"]
    effects = build_condition_effects(gemma_overlap_rows)
    comparisons = build_source_nearest_comparisons(effects)

    summary_rows: list[dict[str, Any]] = []
    summary_rows.extend(summarize_values(comparisons, "source_minus_nearest", ["condition"]))
    summary_rows.extend(summarize_values(comparisons, "source_minus_random16", ["condition"]))
    summary_rows.extend(
        summarize_values(
            comparisons,
            "source_minus_nearest",
            ["condition", "reasoning_operation"],
        )
    )
    summary_rows.extend(
        summarize_values(
            comparisons,
            "source_minus_random16",
            ["condition", "reasoning_operation"],
        )
    )

    write_csv(
        OUT_OVERLAP_MANIFEST,
        overlap_manifest,
        [
            "sample_id",
            "stage3_selection_rank",
            "reasoning_operation",
            "image_dependence",
            "mask_pack",
            "gemma_overlap_status",
            "gemma_raw_rows",
            "support_source_clean_rows",
            "support_nearest_clean_rows",
            "available_runs",
            "available_prompt_names",
        ],
    )
    write_csv(OUT_RAW, gemma_overlap_rows, list(gemma_all_rows[0].keys()) if gemma_all_rows else [])
    write_csv(OUT_SUPPORT_PAIRS, support_pair_overlap, list(support_pair_rows[0].keys()) if support_pair_rows else [])
    write_csv(
        OUT_SUMMARY,
        summary_rows,
        [
            "condition",
            "reasoning_operation",
            "metric",
            "n",
            "mean",
            "ci95_low",
            "ci95_high",
            "positive_count",
            "positive_rate",
        ],
    )
    write_csv(
        OUT_MISSING,
        missing,
        [
            "sample_id",
            "stage3_selection_rank",
            "reasoning_operation",
            "image_dependence",
            "mask_pack",
            "gemma_overlap_status",
            "gemma_raw_rows",
            "support_source_clean_rows",
            "support_nearest_clean_rows",
            "available_runs",
            "available_prompt_names",
        ],
    )

    available_samples = sorted(
        row["sample_id"] for row in overlap_manifest if row.get("gemma_overlap_status") == "available"
    )
    decision = {
        "status": "partial_calibration_only",
        "stage3_manifest_samples": len(stage3_rows),
        "gemma_overlap_available_samples": len(available_samples),
        "gemma_overlap_available_sample_ids": available_samples,
        "missing_source_control_samples": len(missing),
        "missing_source_control_sample_ids": [row["sample_id"] for row in missing],
        "support_source_nearest_comparison_rows": len(comparisons),
        "summary_artifacts": {
            "overlap_manifest": str(OUT_OVERLAP_MANIFEST),
            "raw_rows": str(OUT_RAW),
            "support_pairs": str(OUT_SUPPORT_PAIRS),
            "summary": str(OUT_SUMMARY),
            "missing": str(OUT_MISSING),
        },
        "interpretation": (
            "Gemma3-PLT has overlapping historical prefix-fix source/control evidence for a subset of "
            "Stage3 aligned24, so it can be used as an overlap calibration baseline. It is not a full "
            "24-sample Stage3 rerun. A full Gemma Stage3 baseline requires new source tracing or a "
            "run-ready manifest for the remaining Stage3 samples."
        ),
    }
    OUT_DECISION.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
