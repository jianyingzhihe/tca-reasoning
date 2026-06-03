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
PREFIX = "stage4_qwen_feature_route"
REAL_MASKS = ["answer_mask", "union_mask"]
CONTROL_GROUPS = [
    "same_size_matched_feature_route_control",
    "same_feature_random_position_route_control",
    "random_active_route_control",
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


def _bootstrap_ci(values: list[float], seed: int = 20260529, n: int = 2000) -> tuple[float, float]:
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
        "route_metrics": CROSS / f"{PREFIX}{suffix}_route_metrics.csv",
        "concentration": CROSS / f"{PREFIX}{suffix}_concentration.csv",
        "decision": CROSS / f"{PREFIX}{suffix}_decision.json",
    }


def _raw_files(tag: str, pack_filter: str, mode_filter: str) -> list[tuple[str, str, Path]]:
    pattern = re.compile(rf"^{re.escape(PREFIX)}_(primary|strict)_(smoke|full)(?:_(.*))?_raw\.csv$")
    out: list[tuple[str, str, Path]] = []
    for path in sorted(CROSS.glob(f"{PREFIX}_*_raw.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        pack, mode, extra = match.groups()
        extra = extra or ""
        if extra != tag:
            continue
        if pack_filter and pack != pack_filter:
            continue
        if mode_filter and mode != mode_filter:
            continue
        out.append((pack, mode, path))
    return out


def _effect_index(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        key = (
            row.get("route_id", ""),
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


def _row_rank_effect(effects: dict[tuple[str, str, str, str, str], dict[str, str]], *key: str) -> float | None:
    row = effects.get(tuple(key))
    if row is None:
        return None
    return _f(row.get("rank_effect"), 0.0)


def _activation_drop(effects: dict[tuple[str, str, str, str, str], dict[str, str]], route_id: str, condition: str) -> float | None:
    row = effects.get((route_id, "mask_route_restore", condition, "source", "target"))
    if row is None or row.get("route_activation_drop", "") == "":
        return None
    return _f(row.get("route_activation_drop"), 0.0)


def _control_effects(
    effects: dict[tuple[str, str, str, str, str], dict[str, str]],
    route_id: str,
    intervention: str,
    condition: str,
    token: str = "target",
) -> list[float]:
    values = []
    for control in CONTROL_GROUPS:
        value = _row_effect(effects, route_id, intervention, condition, control, token)
        if value is not None:
            values.append(value)
    return values


def _first_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        route_id = row.get("route_id", "")
        if route_id:
            out.setdefault(route_id, row)
    return out


def _select_real_condition(
    route_id: str,
    row: dict[str, str],
    effects: dict[tuple[str, str, str, str, str], dict[str, str]],
) -> tuple[str, str]:
    frozen = row.get("best_real_condition", "")
    if frozen in REAL_MASKS:
        return frozen, "manifest_frozen"
    scored: list[tuple[float, str]] = []
    for real in REAL_MASKS:
        source = _row_effect(effects, route_id, "mask_route_restore", real, "source", "target")
        controls = _control_effects(effects, route_id, "mask_route_restore", real, "target")
        shifted = _row_effect(effects, route_id, "mask_route_restore", "shifted_mask", "source", "target")
        shuffled = _row_effect(effects, route_id, "mask_route_restore", "shuffled_mask", "source", "target")
        if source is None:
            continue
        score = 0.0
        if controls:
            score += source - _mean(controls)
        if shifted is not None:
            score += source - shifted
        if shuffled is not None:
            score += source - shuffled
        scored.append((score, real))
    if not scored:
        return "answer_mask", "fallback"
    scored.sort(reverse=True)
    return scored[0][1], "best_available"


def _route_metric_rows(pack: str, mode: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    effects = _effect_index(rows)
    first = _first_rows(rows)
    out: list[dict[str, Any]] = []
    for route_id, row in sorted(first.items()):
        topk = int(_f(row.get("topk"), 0))
        real_condition, condition_source = _select_real_condition(route_id, row, effects)
        zero_source = _row_effect(effects, route_id, "clean_route_zeroing", "clean", "source", "target")
        zero_wrong = _row_effect(effects, route_id, "clean_route_zeroing", "clean", "source", "wrong")
        zero_controls = _control_effects(effects, route_id, "clean_route_zeroing", "clean", "target")
        zero_rank = _row_rank_effect(effects, route_id, "clean_route_zeroing", "clean", "source", "target")
        restore_source = _row_effect(effects, route_id, "mask_route_restore", real_condition, "source", "target")
        restore_wrong = _row_effect(effects, route_id, "mask_route_restore", real_condition, "source", "wrong")
        restore_controls = _control_effects(effects, route_id, "mask_route_restore", real_condition, "target")
        restore_rank = _row_rank_effect(effects, route_id, "mask_route_restore", real_condition, "source", "target")
        shifted = _row_effect(effects, route_id, "mask_route_restore", "shifted_mask", "source", "target")
        shuffled = _row_effect(effects, route_id, "mask_route_restore", "shuffled_mask", "source", "target")

        route_zeroing_source_minus_controls = zero_source - _mean(zero_controls) if zero_source is not None and zero_controls else None
        route_restore_source_minus_controls = restore_source - _mean(restore_controls) if restore_source is not None and restore_controls else None
        route_real_minus_shifted = restore_source - shifted if restore_source is not None and shifted is not None else None
        route_real_minus_shuffled = restore_source - shuffled if restore_source is not None and shuffled is not None else None
        route_clean_correct_minus_wrong = zero_source - zero_wrong if zero_source is not None and zero_wrong is not None else None
        route_restore_correct_minus_wrong = restore_source - restore_wrong if restore_source is not None and restore_wrong is not None else None
        correct_values = [v for v in [route_clean_correct_minus_wrong, route_restore_correct_minus_wrong] if v is not None]
        route_correct_minus_wrong = max(correct_values) if correct_values else None
        rank_values = [v for v in [zero_rank, restore_rank] if v is not None]
        route_rank_effect = max(rank_values) if rank_values else None
        real_drop = _activation_drop(effects, route_id, real_condition)
        shifted_drop = _activation_drop(effects, route_id, "shifted_mask")
        shuffled_drop = _activation_drop(effects, route_id, "shuffled_mask")
        control_drop = max([drop for drop in [shifted_drop, shuffled_drop] if drop is not None], default=None)
        route_evidence_specificity = real_drop - control_drop if real_drop is not None and control_drop is not None else None

        gate1 = route_evidence_specificity is not None and route_evidence_specificity > 0
        gate2 = route_zeroing_source_minus_controls is not None and route_zeroing_source_minus_controls > 0
        gate3 = route_restore_source_minus_controls is not None and route_restore_source_minus_controls > 0
        gate4 = (
            route_real_minus_shifted is not None
            and route_real_minus_shifted > 0
            and route_real_minus_shuffled is not None
            and route_real_minus_shuffled > 0
        )
        gate5 = route_correct_minus_wrong is not None and route_correct_minus_wrong > 0
        out.append(
            {
                "pack": pack,
                "mode": mode,
                "route_id": route_id,
                "route_base_id": row.get("route_base_id", ""),
                "sample_id": row.get("sample_id", ""),
                "prompt_name": row.get("prompt_name", ""),
                "topk": topk,
                "route_node_count": row.get("route_node_count", ""),
                "primary_node_count": row.get("primary_node_count", ""),
                "strict_missing_fraction": row.get("strict_missing_fraction", "0"),
                "best_real_condition": real_condition,
                "condition_source": condition_source,
                "target_answer": row.get("target_answer", ""),
                "target_token_id": row.get("target_token_id", ""),
                "wrong_token_id": row.get("wrong_token_id", ""),
                "route_zeroing_source_minus_controls": route_zeroing_source_minus_controls,
                "route_restore_source_minus_controls": route_restore_source_minus_controls,
                "route_real_minus_shifted": route_real_minus_shifted,
                "route_real_minus_shuffled": route_real_minus_shuffled,
                "route_evidence_specificity": route_evidence_specificity,
                "route_clean_correct_minus_wrong": route_clean_correct_minus_wrong,
                "route_restore_correct_minus_wrong": route_restore_correct_minus_wrong,
                "route_correct_minus_wrong": route_correct_minus_wrong,
                "route_zeroing_rank_effect": zero_rank,
                "route_restore_rank_effect": restore_rank,
                "route_rank_effect": route_rank_effect,
                "route_sequence_score_effect": "",
                "route_real_activation_drop": real_drop,
                "route_shifted_activation_drop": shifted_drop,
                "route_shuffled_activation_drop": shuffled_drop,
                "gate1_evidence_sensitive": "1" if gate1 else "0",
                "gate2_clean_source_gt_controls": "1" if gate2 else "0",
                "gate3_restore_source_gt_controls": "1" if gate3 else "0",
                "gate4_real_gt_shifted_shuffled": "1" if gate4 else "0",
                "gate5_correct_gt_wrong": "1" if gate5 else "0",
                "route_first_all_gates": "1" if (gate1 and gate2 and gate3 and gate4 and gate5) else "0",
            }
        )
    return out


def _metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    out = []
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            out.append(float(value))
    return out


def _metric_row(pack: str, mode: str, topk: str | int, metric: str, values: list[float]) -> dict[str, Any]:
    lo, hi = _bootstrap_ci(values)
    return {
        "pack": pack,
        "mode": mode,
        "topk": topk,
        "metric": metric,
        "usable_routes": len(values),
        "mean_diff": _mean(values),
        "ci95_low": lo,
        "ci95_high": hi,
        "positive_frac": sum(1 for value in values if value > 0) / len(values) if values else 0.0,
    }


def _specificity_rows(route_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str | int], list[dict[str, Any]]] = defaultdict(list)
    for row in route_metrics:
        groups[(row["pack"], row["mode"], row["topk"])].append(row)
        groups[(row["pack"], row["mode"], "all")].append(row)
    metric_keys = [
        "route_zeroing_source_minus_controls",
        "route_restore_source_minus_controls",
        "route_real_minus_shifted",
        "route_real_minus_shuffled",
        "route_evidence_specificity",
        "route_correct_minus_wrong",
        "route_rank_effect",
    ]
    for (pack, mode, topk), rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], str(item[0][2]))):
        for metric in metric_keys:
            out.append(_metric_row(pack, mode, topk, metric, _metric_values(rows, metric)))
    return out


def _summary_rows(raw_meta: list[tuple[str, str, Path]], raw_by_file: dict[Path, list[dict[str, str]]]) -> list[dict[str, Any]]:
    out = []
    for pack, mode, path in raw_meta:
        rows = raw_by_file[path]
        run_json = path.with_name(path.name.replace("_raw.csv", "_run.json"))
        run_payload: dict[str, Any] = {}
        if run_json.exists():
            try:
                run_payload = json.loads(run_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                run_payload = {}
        out.append(
            {
                "pack": pack,
                "mode": mode,
                "raw_file": path.name,
                "raw_rows": len(rows),
                "route_count": len({row.get("route_id", "") for row in rows if row.get("route_id")}),
                "ok_rows": sum(1 for row in rows if row.get("status", "ok") == "ok"),
                "source_rows": sum(1 for row in rows if row.get("control_group") == "source"),
                "topk_distribution_json": json.dumps(dict(Counter(row.get("topk", "") for row in rows if row.get("route_id"))), sort_keys=True),
                "run_status": (run_payload.get("decision") or {}).get("status", ""),
                "usable_routes": (run_payload.get("decision") or {}).get("usable_routes", ""),
                "requested_routes": (run_payload.get("decision") or {}).get("requested_routes", ""),
                "skipped_count": (run_payload.get("decision") or {}).get("skipped_count", ""),
            }
        )
    return out


def _concentration_rows(route_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    grouped: dict[tuple[str, str, str | int], list[dict[str, Any]]] = defaultdict(list)
    for row in route_metrics:
        grouped[(row["pack"], row["mode"], row["topk"])].append(row)
        grouped[(row["pack"], row["mode"], "all")].append(row)
    for (pack, mode, topk), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], str(item[0][2]))):
        sample_counts = Counter(row.get("sample_id", "") for row in rows)
        top_counts = [count for _, count in sample_counts.most_common()]
        n = len(rows)
        out.append(
            {
                "pack": pack,
                "mode": mode,
                "topk": topk,
                "route_count": n,
                "unique_sample_count": len(sample_counts),
                "top1_sample_share": top_counts[0] / n if n and top_counts else 0.0,
                "top5_sample_share": sum(top_counts[:5]) / n if n else 0.0,
                "top_samples_json": json.dumps(dict(sample_counts.most_common(10)), ensure_ascii=False),
            }
        )
    return out


def _metric_pass(
    specificity: list[dict[str, Any]],
    pack: str,
    mode: str,
    topk: int,
    metric: str,
    min_routes: int = 20,
) -> bool:
    for row in specificity:
        if (
            row["pack"] == pack
            and row["mode"] == mode
            and str(row["topk"]) == str(topk)
            and row["metric"] == metric
            and int(row["usable_routes"]) >= min_routes
            and float(row["ci95_low"]) > 0
            and float(row["positive_frac"]) >= 0.5
        ):
            return True
    return False


def _topk_gates(specificity: list[dict[str, Any]], pack: str, mode: str, topk: int, min_routes: int = 20) -> dict[str, bool]:
    return {
        "gate1_evidence_sensitive": _metric_pass(specificity, pack, mode, topk, "route_evidence_specificity", min_routes),
        "gate2_clean_source_gt_controls": _metric_pass(specificity, pack, mode, topk, "route_zeroing_source_minus_controls", min_routes),
        "gate3_restore_source_gt_controls": _metric_pass(specificity, pack, mode, topk, "route_restore_source_minus_controls", min_routes),
        "gate4_real_gt_shifted": _metric_pass(specificity, pack, mode, topk, "route_real_minus_shifted", min_routes),
        "gate4_real_gt_shuffled": _metric_pass(specificity, pack, mode, topk, "route_real_minus_shuffled", min_routes),
        "gate5_correct_gt_wrong": _metric_pass(specificity, pack, mode, topk, "route_correct_minus_wrong", min_routes),
        "rank_bridge": _metric_pass(specificity, pack, mode, topk, "route_rank_effect", min_routes),
    }


def _all_main_gates(gates: dict[str, bool]) -> bool:
    return all(
        gates[key]
        for key in [
            "gate1_evidence_sensitive",
            "gate2_clean_source_gt_controls",
            "gate3_restore_source_gt_controls",
            "gate4_real_gt_shifted",
            "gate4_real_gt_shuffled",
            "gate5_correct_gt_wrong",
        ]
    )


def _decide(
    *,
    tag: str,
    pack_filter: str,
    mode_filter: str,
    summary: list[dict[str, Any]],
    route_metrics: list[dict[str, Any]],
    specificity: list[dict[str, Any]],
    paths: dict[str, Path],
) -> dict[str, Any]:
    packs_seen = sorted({row["pack"] for row in summary})
    modes_seen = sorted({row["mode"] for row in summary})
    topks_seen = sorted({int(row["topk"]) for row in route_metrics if str(row.get("topk", "")).isdigit()})
    primary_full = [row for row in route_metrics if row["pack"] == "primary" and row["mode"] == "full"]
    strict_full = [row for row in route_metrics if row["pack"] == "strict" and row["mode"] == "full"]
    strict_missing_values = [_f(row.get("strict_missing_fraction")) for row in strict_full if row.get("strict_missing_fraction") not in (None, "")]
    topk_gate_report: dict[str, Any] = {}
    passing_topks: list[int] = []
    for topk in topks_seen:
        primary_gates = _topk_gates(specificity, "primary", "full", topk)
        strict_gates = _topk_gates(specificity, "strict", "full", topk)
        topk_gate_report[str(topk)] = {"primary": primary_gates, "strict": strict_gates}
        if _all_main_gates(primary_gates) and _all_main_gates(strict_gates):
            passing_topks.append(topk)

    if not summary:
        status = "blocked"
    elif mode_filter == "smoke" or "full" not in modes_seen:
        status = "smoke_ok" if route_metrics else "blocked"
    elif "strict" not in packs_seen:
        status = "qwen_feature_route_case_clustered" if primary_full else "blocked"
    elif strict_full and len({row["route_id"] for row in strict_full}) < 20:
        status = "qwen_feature_route_case_clustered"
    elif passing_topks:
        min_passing = min(passing_topks)
        status = "qwen_distributed_feature_route_supported" if min_passing > min(topks_seen or [min_passing]) else "qwen_feature_route_supported"
    elif primary_full and strict_full:
        status = "qwen_route_first_nodes_supported_route_unresolved"
    else:
        status = "qwen_hidden_route_feature_route_not_supported"

    return {
        "status": status,
        "updated_at": _now(),
        "tag": tag,
        "pack_filter": pack_filter,
        "mode_filter": mode_filter,
        "packs_seen": packs_seen,
        "modes_seen": modes_seen,
        "topks_seen": topks_seen,
        "passing_topks": passing_topks,
        "primary_full_route_count": len({row["route_id"] for row in primary_full}),
        "strict_full_route_count": len({row["route_id"] for row in strict_full}),
        "primary_unique_sample_count": len({row["sample_id"] for row in primary_full}),
        "strict_unique_sample_count": len({row["sample_id"] for row in strict_full}),
        "strict_missing_fraction_mean": _mean(strict_missing_values),
        "topk_gate_report": topk_gate_report,
        "summary_csv": str(paths["summary"]),
        "specificity_csv": str(paths["specificity"]),
        "route_metrics_csv": str(paths["route_metrics"]),
        "concentration_csv": str(paths["concentration"]),
        "claim_boundary": (
            "Stage4-066 tests Qwen-native grouped feature routes under route-first discovery. "
            "It does not establish Gemma-style automatic source-tracing graph replication."
        ),
    }


def analyze(tag: str, pack_filter: str, mode_filter: str) -> dict[str, Any]:
    paths = _out_paths(tag)
    raw_meta = _raw_files(tag, pack_filter, mode_filter)
    raw_by_file: dict[Path, list[dict[str, str]]] = {}
    route_metrics: list[dict[str, Any]] = []
    for pack, mode, path in raw_meta:
        rows = _read_csv(path)
        raw_by_file[path] = rows
        route_metrics.extend(_route_metric_rows(pack, mode, rows))

    summary = _summary_rows(raw_meta, raw_by_file)
    specificity = _specificity_rows(route_metrics)
    concentration = _concentration_rows(route_metrics)

    _write_csv(paths["summary"], summary, list(summary[0].keys()) if summary else ["pack", "mode", "raw_file"])
    _write_csv(
        paths["route_metrics"],
        route_metrics,
        list(route_metrics[0].keys()) if route_metrics else ["pack", "mode", "route_id"],
    )
    _write_csv(
        paths["specificity"],
        specificity,
        list(specificity[0].keys()) if specificity else ["pack", "mode", "topk", "metric"],
    )
    _write_csv(
        paths["concentration"],
        concentration,
        list(concentration[0].keys()) if concentration else ["pack", "mode", "topk"],
    )
    decision = _decide(
        tag=tag,
        pack_filter=pack_filter,
        mode_filter=mode_filter,
        summary=summary,
        route_metrics=route_metrics,
        specificity=specificity,
        paths=paths,
    )
    _write_json(paths["decision"], decision)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage4-066 Qwen grouped feature-route validation.")
    parser.add_argument("--pack", choices=["primary", "strict", "all"], default="all")
    parser.add_argument("--mode", choices=["smoke", "full", "all"], default="all")
    parser.add_argument("--tag", default="featureroute_v1")
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
