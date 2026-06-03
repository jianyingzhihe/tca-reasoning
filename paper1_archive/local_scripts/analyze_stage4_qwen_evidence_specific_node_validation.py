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
PREFIX = "stage4_qwen_evidence_specific_nodes"
CONTROL_GROUPS = [
    "same_position_matched_feature_control",
    "same_feature_random_position_control",
    "random_active_feature_control",
]


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


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "")) if row.get(key, "") != "" else default
    except ValueError:
        return default


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _bootstrap_ci(values: list[float], seed: int = 20260528, n: int = 2000) -> tuple[float, float]:
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


def _paths(tag: str) -> tuple[Path, Path, Path, Path]:
    suffix = f"_{tag}" if tag else ""
    return (
        CROSS / f"{PREFIX}{suffix}_summary.csv",
        CROSS / f"{PREFIX}{suffix}_specificity.csv",
        CROSS / f"{PREFIX}{suffix}_case_table.csv",
        CROSS / f"{PREFIX}{suffix}_decision.json",
    )


def _raw_files(tag: str) -> list[tuple[str, str, int, str, Path]]:
    out = []
    for path in sorted(CROSS.glob(f"{PREFIX}_*_L*_raw.csv")):
        stem = path.stem
        # stage4_qwen_evidence_specific_nodes_primary_full_L10_raw
        rest = stem.removeprefix(f"{PREFIX}_")
        parts = rest.split("_")
        if len(parts) < 4:
            continue
        pack = parts[0]
        mode = parts[1]
        layer_idx = next((idx for idx, part in enumerate(parts) if part.startswith("L") and part[1:].isdigit()), -1)
        layer_part = parts[layer_idx] if layer_idx >= 0 else ""
        if not layer_part:
            continue
        extra = "_".join(parts[layer_idx + 1 : -1])
        if extra != tag:
            continue
        out.append((pack, mode, int(layer_part[1:]), extra, path))
    return out


def _index(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str, str], float]:
    out = {}
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        key = (
            row.get("candidate_id", ""),
            row.get("intervention_kind", ""),
            row.get("mask_condition", ""),
            row.get("control_group", ""),
            row.get("token_scored", ""),
        )
        out[key] = _f(row, "logit_effect")
    return out


def _metric_row(pack: str, mode: str, layer: int | str, metric: str, values: list[float]) -> dict[str, Any]:
    lo, hi = _bootstrap_ci(values)
    return {
        "pack": pack,
        "mode": mode,
        "layer": layer,
        "metric": metric,
        "paired_n": len(values),
        "mean_diff": _mean(values),
        "ci95_low": lo,
        "ci95_high": hi,
        "positive_frac": sum(1 for value in values if value > 0) / len(values) if values else 0.0,
    }


def _specificity_for(pack: str, mode: str, layer: int | str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    effects = _index(rows)
    candidates = sorted({row.get("candidate_id", "") for row in rows if row.get("candidate_id")})
    out: list[dict[str, Any]] = []

    clean_source_vs_controls = []
    clean_correct_minus_wrong = []
    for cid in candidates:
        source = effects.get((cid, "clean_zeroing", "clean", "source", "target"))
        wrong = effects.get((cid, "clean_zeroing", "clean", "source", "wrong"))
        controls = [
            effects[(cid, "clean_zeroing", "clean", control, "target")]
            for control in CONTROL_GROUPS
            if (cid, "clean_zeroing", "clean", control, "target") in effects
        ]
        if source is not None and controls:
            clean_source_vs_controls.append(source - _mean(controls))
        if source is not None and wrong is not None:
            clean_correct_minus_wrong.append(source - wrong)
    out.append(_metric_row(pack, mode, layer, "clean_source_minus_controls", clean_source_vs_controls))
    out.append(_metric_row(pack, mode, layer, "clean_correct_minus_wrong", clean_correct_minus_wrong))

    for real in ["answer_mask", "union_mask"]:
        source_vs_controls = []
        real_minus_shifted = []
        real_minus_shuffled = []
        restore_correct_wrong = []
        for cid in candidates:
            source = effects.get((cid, "mask_restore", real, "source", "target"))
            wrong = effects.get((cid, "mask_restore", real, "source", "wrong"))
            controls = [
                effects[(cid, "mask_restore", real, control, "target")]
                for control in CONTROL_GROUPS
                if (cid, "mask_restore", real, control, "target") in effects
            ]
            shifted = effects.get((cid, "mask_restore", "shifted_mask", "source", "target"))
            shuffled = effects.get((cid, "mask_restore", "shuffled_mask", "source", "target"))
            if source is not None and controls:
                source_vs_controls.append(source - _mean(controls))
            if source is not None and shifted is not None:
                real_minus_shifted.append(source - shifted)
            if source is not None and shuffled is not None:
                real_minus_shuffled.append(source - shuffled)
            if source is not None and wrong is not None:
                restore_correct_wrong.append(source - wrong)
        out.append(_metric_row(pack, mode, layer, f"{real}_restore_source_minus_controls", source_vs_controls))
        out.append(_metric_row(pack, mode, layer, f"{real}_restore_real_minus_shifted", real_minus_shifted))
        out.append(_metric_row(pack, mode, layer, f"{real}_restore_real_minus_shuffled", real_minus_shuffled))
        out.append(_metric_row(pack, mode, layer, f"{real}_restore_correct_minus_wrong", restore_correct_wrong))
    return out


def _case_rows(pack: str, mode: str, layer: int, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    effects = _index(rows)
    first = {}
    for row in rows:
        first.setdefault(row.get("candidate_id", ""), row)
    out = []
    for cid, row in first.items():
        controls = [
            effects[(cid, "clean_zeroing", "clean", control, "target")]
            for control in CONTROL_GROUPS
            if (cid, "clean_zeroing", "clean", control, "target") in effects
        ]
        source_clean = effects.get((cid, "clean_zeroing", "clean", "source", "target"), "")
        wrong_clean = effects.get((cid, "clean_zeroing", "clean", "source", "wrong"), "")
        answer_restore = effects.get((cid, "mask_restore", "answer_mask", "source", "target"), "")
        union_restore = effects.get((cid, "mask_restore", "union_mask", "source", "target"), "")
        shifted_restore = effects.get((cid, "mask_restore", "shifted_mask", "source", "target"), "")
        shuffled_restore = effects.get((cid, "mask_restore", "shuffled_mask", "source", "target"), "")
        out.append(
            {
                "pack": pack,
                "mode": mode,
                "layer": layer,
                "candidate_id": cid,
                "sample_id": row.get("sample_id", ""),
                "prompt_name": row.get("prompt_name", ""),
                "source_node_id": row.get("source_node_id", ""),
                "source_clean_logit_effect": source_clean,
                "clean_control_mean": _mean(controls),
                "clean_source_minus_controls": float(source_clean) - _mean(controls) if source_clean != "" and controls else "",
                "clean_correct_minus_wrong": float(source_clean) - float(wrong_clean) if source_clean != "" and wrong_clean != "" else "",
                "answer_restore": answer_restore,
                "union_restore": union_restore,
                "shifted_restore": shifted_restore,
                "shuffled_restore": shuffled_restore,
                "target_answer": row.get("target_answer", ""),
                "target_token": row.get("target_token", ""),
            }
        )
    return out


def analyze(tag: str) -> dict[str, Any]:
    summary_path, specificity_path, case_table_path, decision_path = _paths(tag)
    summary: list[dict[str, Any]] = []
    specificity: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    combined: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for pack, mode, layer, _tag, path in _raw_files(tag):
        rows = _read_csv(path)
        combined[(pack, mode)].extend(rows)
        candidates = {row.get("candidate_id", "") for row in rows if row.get("candidate_id")}
        summary.append(
            {
                "pack": pack,
                "mode": mode,
                "layer": layer,
                "raw_rows": len(rows),
                "candidate_count": len(candidates),
                "ok_rows": sum(1 for row in rows if row.get("status", "ok") == "ok"),
                "source_rows": sum(1 for row in rows if row.get("control_group") == "source"),
            }
        )
        specificity.extend(_specificity_for(pack, mode, layer, rows))
        cases.extend(_case_rows(pack, mode, layer, rows))
    for (pack, mode), rows in sorted(combined.items()):
        specificity.extend(_specificity_for(pack, mode, "all", rows))

    if summary:
        _write_csv(summary_path, summary, list(summary[0].keys()))
    if specificity:
        _write_csv(specificity_path, specificity, list(specificity[0].keys()))
    if cases:
        _write_csv(case_table_path, cases, list(cases[0].keys()))

    def has_gate(pack: str, metric_names: set[str]) -> bool:
        rows = [
            row
            for row in specificity
            if row["pack"] == pack
            and row["mode"] == "full"
            and row["metric"] in metric_names
            and int(row["paired_n"]) > 0
        ]
        return any(float(row["ci95_low"]) > 0 and float(row["positive_frac"]) >= 0.5 for row in rows)

    primary_source = has_gate("primary", {"clean_source_minus_controls"}) and has_gate("primary", {"clean_correct_minus_wrong"})
    strict_source = has_gate("strict", {"clean_source_minus_controls"}) and has_gate("strict", {"clean_correct_minus_wrong"})
    primary_mask = has_gate("primary", {"answer_mask_restore_real_minus_shifted", "answer_mask_restore_real_minus_shuffled", "union_mask_restore_real_minus_shifted", "union_mask_restore_real_minus_shuffled"})
    strict_mask = has_gate("strict", {"answer_mask_restore_real_minus_shifted", "answer_mask_restore_real_minus_shuffled", "union_mask_restore_real_minus_shifted", "union_mask_restore_real_minus_shuffled"})

    packs_seen = sorted({row["pack"] for row in summary})
    if not summary:
        status = "blocked"
    elif primary_source and strict_source and primary_mask and strict_mask:
        status = "qwen_evidence_linked_route_supported"
    elif primary_source and strict_source:
        status = "qwen_evidence_specific_causal_nodes_supported"
    elif primary_source and "strict" not in packs_seen:
        status = "primary_targeted_near_pass_pending_strict"
    elif not primary_source:
        status = "qwen_evidence_sensitive_but_not_causal"
    else:
        status = "qwen_plt_localization_unresolved"

    decision = {
        "status": status,
        "updated_at": _now(),
        "tag": tag,
        "packs_seen": packs_seen,
        "layers_seen": sorted({int(row["layer"]) for row in summary}),
        "primary_source_gate": primary_source,
        "strict_source_gate": strict_source,
        "primary_mask_gate": primary_mask,
        "strict_mask_gate": strict_mask,
        "summary_csv": str(summary_path),
        "specificity_csv": str(specificity_path),
        "case_table_csv": str(case_table_path),
        "claim_boundary": "Targeted validation tests Qwen-native evidence-specific PLT candidates; it is not Gemma-style automatic source tracing.",
    }
    _write_json(decision_path, decision)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage4-052 Qwen evidence-specific node validation.")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    decision = analyze(args.tag)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
