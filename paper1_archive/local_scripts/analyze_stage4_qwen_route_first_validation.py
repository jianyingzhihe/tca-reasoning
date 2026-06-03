#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"
PREFIX = "stage4_qwen_route_first"
REAL_MASKS = ["answer_mask", "union_mask"]
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


def _f(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw) if raw not in (None, "") else default
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


def _suffix(tag: str) -> str:
    return f"_{tag}" if tag else ""


def _out_paths(tag: str) -> dict[str, Path]:
    suffix = _suffix(tag)
    return {
        "summary": CROSS / f"{PREFIX}{suffix}_summary.csv",
        "specificity": CROSS / f"{PREFIX}{suffix}_specificity.csv",
        "route_candidates": CROSS / f"{PREFIX}{suffix}_route_candidates.csv",
        "concentration": CROSS / f"{PREFIX}{suffix}_concentration.csv",
        "decision": CROSS / f"{PREFIX}{suffix}_decision.json",
    }


def _raw_files(tag: str, pack_filter: str, mode_filter: str) -> list[tuple[str, str, int, Path]]:
    pattern = re.compile(rf"^{re.escape(PREFIX)}_(primary|strict)_(smoke|full)_L(\d+)(?:_(.*))?_raw\.csv$")
    out: list[tuple[str, str, int, Path]] = []
    for path in sorted(CROSS.glob(f"{PREFIX}_*_raw.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        pack, mode, layer_raw, extra = match.groups()
        extra = extra or ""
        if extra != tag:
            continue
        if pack_filter and pack != pack_filter:
            continue
        if mode_filter and mode != mode_filter:
            continue
        out.append((pack, mode, int(layer_raw), path))
    return out


def _effect_index(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
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
        out[key] = row
    return out


def _row_effect(effects: dict[tuple[str, str, str, str, str], dict[str, str]], *key: str) -> float | None:
    row = effects.get(tuple(key))
    if row is None:
        return None
    return _f(row.get("logit_effect"), 0.0)


def _activation_drop(effects: dict[tuple[str, str, str, str, str], dict[str, str]], cid: str, condition: str) -> float | None:
    row = effects.get((cid, "mask_restore", condition, "source", "target"))
    if row is None:
        row = effects.get((cid, "mask_restore", condition, "source", "wrong"))
    if row is None or row.get("activation_drop", "") == "":
        return None
    return _f(row.get("activation_drop"), 0.0)


def _control_effects(
    effects: dict[tuple[str, str, str, str, str], dict[str, str]],
    cid: str,
    intervention: str,
    condition: str,
    token: str = "target",
) -> list[float]:
    values = []
    for control in CONTROL_GROUPS:
        value = _row_effect(effects, cid, intervention, condition, control, token)
        if value is not None:
            values.append(value)
    return values


def _load_primary_conditions(tag: str) -> dict[str, str]:
    path = CROSS / f"{PREFIX}{_suffix(tag)}_route_candidates.csv"
    out: dict[str, str] = {}
    for row in _read_csv(path):
        if row.get("pack") == "primary" and row.get("mode") == "full" and row.get("route_first_234") == "1":
            cid = row.get("candidate_id", "")
            condition = row.get("best_real_condition", "")
            if cid and condition:
                out[cid] = condition
    return out


def _candidate_rows(
    *,
    pack: str,
    mode: str,
    layer: int,
    rows: list[dict[str, str]],
    primary_conditions: dict[str, str],
) -> list[dict[str, Any]]:
    effects = _effect_index(rows)
    first: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("candidate_id"):
            first.setdefault(row["candidate_id"], row)

    out: list[dict[str, Any]] = []
    for cid, row in sorted(first.items()):
        clean_source = _row_effect(effects, cid, "clean_zeroing", "clean", "source", "target")
        clean_wrong = _row_effect(effects, cid, "clean_zeroing", "clean", "source", "wrong")
        clean_controls = _control_effects(effects, cid, "clean_zeroing", "clean", "target")
        clean_source_minus_controls = clean_source - _mean(clean_controls) if clean_source is not None and clean_controls else None
        clean_correct_minus_wrong = clean_source - clean_wrong if clean_source is not None and clean_wrong is not None else None

        fixed_condition = primary_conditions.get(cid) if pack == "strict" and mode == "full" else ""
        candidate_metrics: list[dict[str, Any]] = []
        for real in REAL_MASKS:
            source_restore = _row_effect(effects, cid, "mask_restore", real, "source", "target")
            wrong_restore = _row_effect(effects, cid, "mask_restore", real, "source", "wrong")
            restore_controls = _control_effects(effects, cid, "mask_restore", real, "target")
            shifted = _row_effect(effects, cid, "mask_restore", "shifted_mask", "source", "target")
            shuffled = _row_effect(effects, cid, "mask_restore", "shuffled_mask", "source", "target")
            restore_source_minus_controls = source_restore - _mean(restore_controls) if source_restore is not None and restore_controls else None
            real_minus_shifted = source_restore - shifted if source_restore is not None and shifted is not None else None
            real_minus_shuffled = source_restore - shuffled if source_restore is not None and shuffled is not None else None
            restore_correct_minus_wrong = source_restore - wrong_restore if source_restore is not None and wrong_restore is not None else None
            real_drop = _activation_drop(effects, cid, real)
            shifted_drop = _activation_drop(effects, cid, "shifted_mask")
            shuffled_drop = _activation_drop(effects, cid, "shuffled_mask")
            control_drop = max([drop for drop in [shifted_drop, shuffled_drop] if drop is not None], default=None)
            evidence_specificity = real_drop - control_drop if real_drop is not None and control_drop is not None else None
            gate3 = restore_source_minus_controls is not None and restore_source_minus_controls > 0
            gate4a = real_minus_shifted is not None and real_minus_shifted > 0
            gate4b = real_minus_shuffled is not None and real_minus_shuffled > 0
            candidate_metrics.append(
                {
                    "best_real_condition": real,
                    "source_restore": source_restore,
                    "restore_source_minus_controls": restore_source_minus_controls,
                    "real_minus_shifted": real_minus_shifted,
                    "real_minus_shuffled": real_minus_shuffled,
                    "restore_correct_minus_wrong": restore_correct_minus_wrong,
                    "real_activation_drop": real_drop,
                    "shifted_activation_drop": shifted_drop,
                    "shuffled_activation_drop": shuffled_drop,
                    "evidence_specificity": evidence_specificity,
                    "real_gate_score": sum(1 for value in [gate3, gate4a, gate4b] if value)
                    + (restore_source_minus_controls or 0.0)
                    + (real_minus_shifted or 0.0)
                    + (real_minus_shuffled or 0.0),
                }
            )
        if fixed_condition:
            selected = next((item for item in candidate_metrics if item["best_real_condition"] == fixed_condition), candidate_metrics[0])
            condition_source = "primary_frozen"
        else:
            selected = sorted(candidate_metrics, key=lambda item: item["real_gate_score"], reverse=True)[0]
            condition_source = "best_of_answer_union"

        gate2 = clean_source_minus_controls is not None and clean_source_minus_controls > 0
        gate3 = selected["restore_source_minus_controls"] is not None and selected["restore_source_minus_controls"] > 0
        gate4 = (
            selected["real_minus_shifted"] is not None
            and selected["real_minus_shifted"] > 0
            and selected["real_minus_shuffled"] is not None
            and selected["real_minus_shuffled"] > 0
        )
        gate1 = selected["evidence_specificity"] is not None and selected["evidence_specificity"] > 0
        gate5_clean = clean_correct_minus_wrong is not None and clean_correct_minus_wrong > 0
        gate5_restore = selected["restore_correct_minus_wrong"] is not None and selected["restore_correct_minus_wrong"] > 0
        gate5 = gate5_clean or gate5_restore
        route_first_234 = gate2 and gate3 and gate4
        route_first_gold = route_first_234 and gate5
        route_first_evidence_gold = route_first_234 and gate1 and gate5

        out.append(
            {
                "pack": pack,
                "mode": mode,
                "layer": layer,
                "candidate_id": cid,
                "sample_id": row.get("sample_id", ""),
                "prompt_name": row.get("prompt_name", ""),
                "source_node_id": row.get("source_node_id", ""),
                "source_feature_id": row.get("source_feature_id", ""),
                "source_pos": row.get("source_pos", ""),
                "target_answer": row.get("target_answer", ""),
                "target_token_id": row.get("target_token_id", ""),
                "target_token": row.get("target_token", ""),
                "wrong_token_id": row.get("wrong_token_id", ""),
                "wrong_token": row.get("wrong_token", ""),
                "clean_target_logit": row.get("clean_target_logit", ""),
                "clean_target_rank": row.get("clean_target_rank", ""),
                "best_real_condition": selected["best_real_condition"],
                "condition_source": condition_source,
                "clean_source_effect": clean_source,
                "clean_control_mean": _mean(clean_controls) if clean_controls else "",
                "clean_source_minus_controls": clean_source_minus_controls,
                "restore_source_effect": selected["source_restore"],
                "restore_source_minus_controls": selected["restore_source_minus_controls"],
                "real_minus_shifted": selected["real_minus_shifted"],
                "real_minus_shuffled": selected["real_minus_shuffled"],
                "clean_correct_minus_wrong": clean_correct_minus_wrong,
                "restore_correct_minus_wrong": selected["restore_correct_minus_wrong"],
                "real_activation_drop": selected["real_activation_drop"],
                "shifted_activation_drop": selected["shifted_activation_drop"],
                "shuffled_activation_drop": selected["shuffled_activation_drop"],
                "evidence_specificity": selected["evidence_specificity"],
                "gate1_evidence_sensitive": "1" if gate1 else "0",
                "gate2_clean_source_gt_controls": "1" if gate2 else "0",
                "gate3_restore_source_gt_controls": "1" if gate3 else "0",
                "gate4_real_gt_shifted_shuffled": "1" if gate4 else "0",
                "gate5_correct_gt_wrong": "1" if gate5 else "0",
                "gate5_clean_correct_gt_wrong": "1" if gate5_clean else "0",
                "gate5_restore_correct_gt_wrong": "1" if gate5_restore else "0",
                "route_first_234": "1" if route_first_234 else "0",
                "route_first_gold": "1" if route_first_gold else "0",
                "route_first_evidence_gold": "1" if route_first_evidence_gold else "0",
            }
        )
    return out


def _metric_row(pack: str, mode: str, layer: str | int, candidate_set: str, metric: str, values: list[float]) -> dict[str, Any]:
    lo, hi = _bootstrap_ci(values)
    return {
        "pack": pack,
        "mode": mode,
        "layer": layer,
        "candidate_set": candidate_set,
        "metric": metric,
        "paired_n": len(values),
        "mean_diff": _mean(values),
        "ci95_low": lo,
        "ci95_high": hi,
        "positive_frac": sum(1 for value in values if value > 0) / len(values) if values else 0.0,
    }


def _metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    out = []
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            out.append(float(value))
    return out


def _specificity_rows(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str | int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        groups[(row["pack"], row["mode"], row["layer"])].append(row)
        groups[(row["pack"], row["mode"], "all")].append(row)

    set_filters = {
        "all_candidates": lambda row: True,
        "route_first_234": lambda row: row.get("route_first_234") == "1",
        "route_first_gold": lambda row: row.get("route_first_gold") == "1",
        "route_first_evidence_gold": lambda row: row.get("route_first_evidence_gold") == "1",
    }
    metric_keys = [
        "clean_source_minus_controls",
        "restore_source_minus_controls",
        "real_minus_shifted",
        "real_minus_shuffled",
        "evidence_specificity",
        "clean_correct_minus_wrong",
        "restore_correct_minus_wrong",
    ]
    for (pack, mode, layer), rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], str(item[0][2]))):
        for set_name, predicate in set_filters.items():
            subset = [row for row in rows if predicate(row)]
            for metric in metric_keys:
                out.append(_metric_row(pack, mode, layer, set_name, metric, _metric_values(subset, metric)))
    return out


def _summary_rows(raw_meta: list[tuple[str, str, int, Path]], raw_by_file: dict[Path, list[dict[str, str]]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_file_candidate_counts = {
        path: len({row.get("candidate_id", "") for row in raw_by_file[path] if row.get("candidate_id")})
        for _, _, _, path in raw_meta
    }
    by_pack_mode_layer_counts: Counter[tuple[str, str, int, str]] = Counter()
    for row in candidates:
        for key in ["route_first_234", "route_first_gold", "route_first_evidence_gold"]:
            if row.get(key) == "1":
                by_pack_mode_layer_counts[(row["pack"], row["mode"], int(row["layer"]), key)] += 1
    out = []
    for pack, mode, layer, path in raw_meta:
        rows = raw_by_file[path]
        out.append(
            {
                "pack": pack,
                "mode": mode,
                "layer": layer,
                "raw_file": path.name,
                "raw_rows": len(rows),
                "candidate_count": by_file_candidate_counts[path],
                "ok_rows": sum(1 for row in rows if row.get("status", "ok") == "ok"),
                "source_rows": sum(1 for row in rows if row.get("control_group") == "source"),
                "route_first_234_count": by_pack_mode_layer_counts[(pack, mode, layer, "route_first_234")],
                "route_first_gold_count": by_pack_mode_layer_counts[(pack, mode, layer, "route_first_gold")],
                "route_first_evidence_gold_count": by_pack_mode_layer_counts[(pack, mode, layer, "route_first_evidence_gold")],
            }
        )
    return out


def _concentration_rows(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    set_filters = {
        "all_candidates": lambda row: True,
        "route_first_234": lambda row: row.get("route_first_234") == "1",
        "route_first_gold": lambda row: row.get("route_first_gold") == "1",
        "route_first_evidence_gold": lambda row: row.get("route_first_evidence_gold") == "1",
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[(row["pack"], row["mode"])].append(row)
    for (pack, mode), rows in sorted(grouped.items()):
        for set_name, predicate in set_filters.items():
            subset = [row for row in rows if predicate(row)]
            sample_counts = Counter(row.get("sample_id", "") for row in subset)
            layer_counts = Counter(str(row.get("layer", "")) for row in subset)
            top_counts = [count for _, count in sample_counts.most_common()]
            n = len(subset)
            out.append(
                {
                    "pack": pack,
                    "mode": mode,
                    "candidate_set": set_name,
                    "candidate_count": n,
                    "unique_sample_count": len(sample_counts),
                    "top1_sample_share": top_counts[0] / n if n and top_counts else 0.0,
                    "top5_sample_share": sum(top_counts[:5]) / n if n else 0.0,
                    "layer_distribution_json": json.dumps(dict(layer_counts), sort_keys=True),
                    "top_samples_json": json.dumps(dict(sample_counts.most_common(10)), ensure_ascii=False),
                }
            )
    return out


def _aggregate_pass(specificity: list[dict[str, Any]], pack: str, mode: str, candidate_set: str, metrics: set[str]) -> bool:
    rows = [
        row
        for row in specificity
        if row["pack"] == pack
        and row["mode"] == mode
        and row["layer"] == "all"
        and row["candidate_set"] == candidate_set
        and row["metric"] in metrics
        and int(row["paired_n"]) > 0
    ]
    return any(float(row["ci95_low"]) > 0 and float(row["positive_frac"]) >= 0.5 for row in rows)


def _decide(
    *,
    tag: str,
    pack_filter: str,
    mode_filter: str,
    summary: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    specificity: list[dict[str, Any]],
    paths: dict[str, Path],
) -> dict[str, Any]:
    packs_seen = sorted({row["pack"] for row in summary})
    modes_seen = sorted({row["mode"] for row in summary})
    primary_full = [row for row in candidates if row["pack"] == "primary" and row["mode"] == "full"]
    strict_full = [row for row in candidates if row["pack"] == "strict" and row["mode"] == "full"]
    primary_234 = [row for row in primary_full if row.get("route_first_234") == "1"]
    strict_234 = [row for row in strict_full if row.get("route_first_234") == "1"]
    strict_n = len(strict_full)

    primary_gate2 = _aggregate_pass(specificity, "primary", "full", "route_first_234", {"clean_source_minus_controls"})
    primary_gate3 = _aggregate_pass(specificity, "primary", "full", "route_first_234", {"restore_source_minus_controls"})
    primary_gate4_shifted = _aggregate_pass(specificity, "primary", "full", "route_first_234", {"real_minus_shifted"})
    primary_gate4_shuffled = _aggregate_pass(specificity, "primary", "full", "route_first_234", {"real_minus_shuffled"})
    primary_gate1 = _aggregate_pass(specificity, "primary", "full", "route_first_234", {"evidence_specificity"})
    primary_gate5 = _aggregate_pass(
        specificity,
        "primary",
        "full",
        "route_first_234",
        {"clean_correct_minus_wrong", "restore_correct_minus_wrong"},
    )
    strict_gate2 = _aggregate_pass(specificity, "strict", "full", "all_candidates", {"clean_source_minus_controls"})
    strict_gate3 = _aggregate_pass(specificity, "strict", "full", "all_candidates", {"restore_source_minus_controls"})
    strict_gate4_shifted = _aggregate_pass(specificity, "strict", "full", "all_candidates", {"real_minus_shifted"})
    strict_gate4_shuffled = _aggregate_pass(specificity, "strict", "full", "all_candidates", {"real_minus_shuffled"})
    strict_gate1 = _aggregate_pass(specificity, "strict", "full", "route_first_234", {"evidence_specificity"})
    strict_gate5 = _aggregate_pass(
        specificity,
        "strict",
        "full",
        "route_first_234",
        {"clean_correct_minus_wrong", "restore_correct_minus_wrong"},
    )

    primary_234_aggregate = primary_gate2 and primary_gate3 and primary_gate4_shifted and primary_gate4_shuffled
    strict_234_aggregate = strict_gate2 and strict_gate3 and strict_gate4_shifted and strict_gate4_shuffled

    if not summary:
        status = "blocked"
    elif mode_filter == "smoke" or "full" not in modes_seen:
        status = "smoke_ok" if candidates else "blocked"
    elif primary_234 and "strict" not in packs_seen:
        status = "qwen_route_first_candidates_exist"
    elif primary_234 and strict_n < 20:
        status = "qwen_route_first_candidates_exist"
    elif primary_234 and not strict_234_aggregate:
        status = "qwen_route_first_not_supported"
    elif primary_234_aggregate and strict_234_aggregate and primary_gate1 and strict_gate1 and primary_gate5 and strict_gate5:
        status = "qwen_route_first_full_supported"
    elif primary_234_aggregate and strict_234_aggregate and primary_gate1 and strict_gate1:
        status = "qwen_route_first_evidence_linked_supported"
    elif primary_234_aggregate and strict_234_aggregate and primary_gate5 and strict_gate5:
        status = "qwen_route_first_gold_answer_supported"
    elif primary_234_aggregate and strict_234_aggregate and not (primary_gate1 and strict_gate1):
        status = "qwen_answer_route_not_evidence_linked"
    elif primary_234_aggregate and strict_234_aggregate and not (primary_gate5 and strict_gate5):
        status = "qwen_evidence_route_not_gold_specific"
    else:
        status = "qwen_route_first_not_supported"

    return {
        "status": status,
        "updated_at": _now(),
        "tag": tag,
        "pack_filter": pack_filter,
        "mode_filter": mode_filter,
        "packs_seen": packs_seen,
        "modes_seen": modes_seen,
        "layers_seen": sorted({int(row["layer"]) for row in summary}),
        "primary_full_candidate_count": len(primary_full),
        "primary_route_first_234_count": len(primary_234),
        "primary_route_first_gold_count": sum(1 for row in primary_full if row.get("route_first_gold") == "1"),
        "primary_route_first_evidence_gold_count": sum(1 for row in primary_full if row.get("route_first_evidence_gold") == "1"),
        "strict_full_candidate_count": strict_n,
        "strict_route_first_234_count": len(strict_234),
        "strict_route_first_gold_count": sum(1 for row in strict_full if row.get("route_first_gold") == "1"),
        "strict_route_first_evidence_gold_count": sum(1 for row in strict_full if row.get("route_first_evidence_gold") == "1"),
        "primary_aggregate_gates": {
            "gate2_clean_source_gt_controls": primary_gate2,
            "gate3_restore_source_gt_controls": primary_gate3,
            "gate4_real_gt_shifted": primary_gate4_shifted,
            "gate4_real_gt_shuffled": primary_gate4_shuffled,
            "gate1_evidence_sensitive_on_route_first": primary_gate1,
            "gate5_correct_gt_wrong_on_route_first": primary_gate5,
        },
        "strict_aggregate_gates": {
            "gate2_clean_source_gt_controls": strict_gate2,
            "gate3_restore_source_gt_controls": strict_gate3,
            "gate4_real_gt_shifted": strict_gate4_shifted,
            "gate4_real_gt_shuffled": strict_gate4_shuffled,
            "gate1_evidence_sensitive_on_route_first": strict_gate1,
            "gate5_correct_gt_wrong_on_route_first": strict_gate5,
        },
        "small_n_exception": strict_n < 20 if "strict" in packs_seen else False,
        "summary_csv": str(paths["summary"]),
        "specificity_csv": str(paths["specificity"]),
        "route_candidates_csv": str(paths["route_candidates"]),
        "concentration_csv": str(paths["concentration"]),
        "claim_boundary": "Stage4-060 tests Qwen-native route-first causal candidates. It does not establish Gemma-style automatic source tracing without an automatic graph closure.",
    }


def analyze(tag: str, pack_filter: str, mode_filter: str) -> dict[str, Any]:
    paths = _out_paths(tag)
    raw_meta = _raw_files(tag, pack_filter, mode_filter)
    primary_conditions = _load_primary_conditions(tag)
    raw_by_file: dict[Path, list[dict[str, str]]] = {}
    candidate_rows: list[dict[str, Any]] = []
    for pack, mode, layer, path in raw_meta:
        rows = _read_csv(path)
        raw_by_file[path] = rows
        candidate_rows.extend(
            _candidate_rows(
                pack=pack,
                mode=mode,
                layer=layer,
                rows=rows,
                primary_conditions=primary_conditions,
            )
        )

    summary = _summary_rows(raw_meta, raw_by_file, candidate_rows)
    specificity = _specificity_rows(candidate_rows)
    concentration = _concentration_rows(candidate_rows)

    if summary:
        _write_csv(paths["summary"], summary, list(summary[0].keys()))
    else:
        _write_csv(paths["summary"], [], ["pack", "mode", "layer", "raw_file"])
    if specificity:
        _write_csv(paths["specificity"], specificity, list(specificity[0].keys()))
    else:
        _write_csv(paths["specificity"], [], ["pack", "mode", "layer", "candidate_set", "metric"])
    if candidate_rows:
        _write_csv(paths["route_candidates"], candidate_rows, list(candidate_rows[0].keys()))
    else:
        _write_csv(paths["route_candidates"], [], ["candidate_id"])
    if concentration:
        _write_csv(paths["concentration"], concentration, list(concentration[0].keys()))
    else:
        _write_csv(paths["concentration"], [], ["pack", "mode", "candidate_set"])

    decision = _decide(
        tag=tag,
        pack_filter=pack_filter,
        mode_filter=mode_filter,
        summary=summary,
        candidates=candidate_rows,
        specificity=specificity,
        paths=paths,
    )
    _write_json(paths["decision"], decision)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage4-060 Qwen route-first validation.")
    parser.add_argument("--pack", choices=["primary", "strict", "all"], default="all")
    parser.add_argument("--mode", choices=["smoke", "full", "all"], default="all")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    decision = analyze(
        tag=args.tag,
        pack_filter="" if args.pack == "all" else args.pack,
        mode_filter="" if args.mode == "all" else args.mode,
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
