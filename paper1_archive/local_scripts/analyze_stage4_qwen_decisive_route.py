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


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value != "" else default
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _ci(values: list[float], n: int = 2000, seed: int = 24024) -> tuple[float, float]:
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


def _metric_row(metric: str, values: list[float], extra: dict[str, Any]) -> dict[str, Any]:
    lo, hi = _ci(values)
    row = {
        "metric": metric,
        "n": len(values),
        "mean": _mean(values),
        "ci95_low": lo,
        "ci95_high": hi,
        "positive_frac": sum(1 for value in values if value > 0) / len(values) if values else 0.0,
    }
    row.update(extra)
    return row


def _positive(row: dict[str, Any], min_n: int = 8) -> bool:
    return int(row.get("n", 0)) >= min_n and float(row.get("ci95_low", 0.0)) > 0 and float(row.get("positive_frac", 0.0)) >= 0.6


def _hidden_rows(cross_dir: Path, mode: str, hidden_tag: str = "") -> list[dict[str, str]]:
    rows = []
    suffix = f"_{hidden_tag}" if hidden_tag else ""
    for pack in ["primary", "strict"]:
        path = cross_dir / f"stage4_qwen_decisive_route_hidden_{pack}_{mode}{suffix}_raw.csv"
        for row in _read_csv(path):
            row["_pack"] = pack
            rows.append(row)
    return rows


def _plt_pack_layer_files(cross_dir: Path, mode: str) -> list[tuple[str, str, Path, Path, Path]]:
    out: list[tuple[str, str, Path, Path, Path]] = []
    for cand in cross_dir.glob(f"stage4_qwen_decisive_route_plt_*_{mode}_L*_candidates.csv"):
        name = cand.name
        prefix = "stage4_qwen_decisive_route_plt_"
        rest = name[len(prefix) : -len("_candidates.csv")]
        parts = rest.split("_")
        if len(parts) < 3:
            continue
        pack = parts[0]
        layer = parts[-1].removeprefix("L")
        zero = cross_dir / f"stage4_qwen_decisive_route_plt_{pack}_{mode}_L{layer}_zeroing_raw.csv"
        group = cross_dir / f"stage4_qwen_decisive_route_plt_{pack}_{mode}_L{layer}_group_raw.csv"
        out.append((pack, layer, cand, zero, group))
    return sorted(out, key=lambda item: (item[0], int(item[1])))


def _hidden_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("token_scored") != "target":
            continue
        grouped[
            (
                row.get("_pack", ""),
                row.get("layer", ""),
                row.get("direction", ""),
                row.get("position_group", ""),
                row.get("mask_condition", ""),
                row.get("scale", ""),
            )
        ].append(row)
    out = []
    for key, group in sorted(grouped.items()):
        pack, layer, direction, position_group, mask_condition, scale = key
        vals = [_f(row, "logit_effect") for row in group]
        ranks = [_f(row, "rank_effect") for row in group]
        row = _metric_row(
            "hidden_target_logit_effect",
            vals,
            {
                "stage": "hidden_lattice",
                "pack": pack,
                "layer": layer,
                "direction": direction,
                "position_group": position_group,
                "mask_condition": mask_condition,
                "scale": scale,
                "prompt_runs": len({(item.get("sample_id", ""), item.get("prompt_name", "")) for item in group}),
                "rank_mean": _mean(ranks),
            },
        )
        out.append(row)
    return out


def _hidden_specificity(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    idx: dict[tuple[str, str, str, str, str, str, str, str], dict[str, str]] = {}
    run_groups: set[tuple[str, str, str, str, str, str]] = set()
    for row in rows:
        run_id = row.get("sample_id", "") + "::" + row.get("prompt_name", "")
        key = (
            row.get("_pack", ""),
            run_id,
            row.get("layer", ""),
            row.get("direction", ""),
            row.get("position_group", ""),
            row.get("scale", ""),
            row.get("mask_condition", ""),
            row.get("token_scored", ""),
        )
        idx[key] = row
        run_groups.add(key[:6])
    out = []
    for pack in ["primary", "strict"]:
        for layer in sorted({row.get("layer", "") for row in rows if row.get("_pack") == pack}, key=lambda x: int(x or 0)):
            for direction in ["restore", "corrupt"]:
                for position_group in sorted({row.get("position_group", "") for row in rows if row.get("_pack") == pack and row.get("layer") == layer}):
                    for scale in sorted({row.get("scale", "") for row in rows if row.get("_pack") == pack and row.get("layer") == layer}):
                        subset = [key for key in run_groups if key[0] == pack and key[2] == layer and key[3] == direction and key[4] == position_group and key[5] == scale]
                        for real in ["answer_mask", "union_mask"]:
                            effect = []
                            real_shifted = []
                            real_shuffled = []
                            correct_wrong = []
                            rank_effect = []
                            for base in subset:
                                target_real = idx.get((*base, real, "target"))
                                wrong_real = idx.get((*base, real, "wrong"))
                                target_shifted = idx.get((*base, "shifted_mask", "target"))
                                target_shuffled = idx.get((*base, "shuffled_mask", "target"))
                                if target_real:
                                    effect.append(_f(target_real, "logit_effect"))
                                    rank_effect.append(_f(target_real, "rank_effect"))
                                if target_real and target_shifted:
                                    real_shifted.append(_f(target_real, "logit_effect") - _f(target_shifted, "logit_effect"))
                                if target_real and target_shuffled:
                                    real_shuffled.append(_f(target_real, "logit_effect") - _f(target_shuffled, "logit_effect"))
                                if target_real and wrong_real:
                                    correct_wrong.append(_f(target_real, "logit_effect") - _f(wrong_real, "logit_effect"))
                            extra = {
                                "stage": "hidden_lattice",
                                "pack": pack,
                                "layer": layer,
                                "direction": direction,
                                "position_group": position_group,
                                "mask_condition": real,
                                "scale": scale,
                            }
                            out.append(_metric_row("hidden_effect", effect, extra))
                            out.append(_metric_row("hidden_real_minus_shifted", real_shifted, extra))
                            out.append(_metric_row("hidden_real_minus_shuffled", real_shuffled, extra))
                            out.append(_metric_row("hidden_correct_minus_wrong", correct_wrong, extra))
                            out.append(_metric_row("hidden_rank_effect", rank_effect, extra))
    return out


def _plt_summary(cross_dir: Path, mode: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pack, layer, cand_path, zero_path, group_path in _plt_pack_layer_files(cross_dir, mode):
        candidates = _read_csv(cand_path)
        zeroing = _read_csv(zero_path)
        grouped = _read_csv(group_path)
        out.append(
            {
                "stage": "plt_layer_sweep",
                "metric": "plt_candidate_manifest",
                "pack": pack,
                "layer": layer,
                "n": len(candidates),
                "prompt_runs": len({(row.get("sample_id", ""), row.get("prompt_name", "")) for row in candidates}),
                "main_rows": sum(1 for row in candidates if row.get("include_main") == "1"),
                "zeroing_rows": len(zeroing),
                "group_rows": len(grouped),
                "mean": _mean([_f(row, "evidence_first_score") for row in candidates]),
                "ci95_low": "",
                "ci95_high": "",
                "positive_frac": "",
            }
        )
    return out


def _plt_zeroing_specificity(rows: list[dict[str, str]], pack: str, layer: str) -> list[dict[str, Any]]:
    target = [row for row in rows if row.get("status", "ok") in {"", "ok"} and row.get("intervention_kind") == "clean_zeroing" and row.get("token_scored") == "target"]
    wrong = [row for row in rows if row.get("status", "ok") in {"", "ok"} and row.get("intervention_kind") == "clean_zeroing" and row.get("token_scored") == "wrong"]
    by_candidate: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in target:
        by_candidate[row.get("candidate_id", "")][row.get("control_group", "")] = row
    wrong_by_candidate: dict[str, dict[str, str]] = {}
    for row in wrong:
        if row.get("control_group") == "source":
            wrong_by_candidate[row.get("candidate_id", "")] = row
    source_minus_controls = []
    correct_minus_wrong = []
    source_rank_effect = []
    for candidate_id, group in by_candidate.items():
        source_row = group.get("source")
        controls = [group[name] for name in CONTROL_GROUPS if name in group]
        if source_row and controls:
            source_minus_controls.append(_f(source_row, "logit_effect") - _mean([_f(row, "logit_effect") for row in controls]))
            source_rank_effect.append(_f(source_row, "rank_effect"))
        wrong_row = wrong_by_candidate.get(candidate_id)
        if source_row and wrong_row:
            correct_minus_wrong.append(_f(source_row, "logit_effect") - _f(wrong_row, "logit_effect"))
    extra = {"stage": "plt_layer_sweep", "pack": pack, "layer": layer}
    return [
        _metric_row("plt_clean_zeroing_source_minus_controls", source_minus_controls, extra),
        _metric_row("plt_clean_zeroing_correct_minus_wrong", correct_minus_wrong, extra),
        _metric_row("plt_clean_zeroing_target_rank_effect", source_rank_effect, extra),
    ]


def _plt_group_specificity(rows: list[dict[str, str]], pack: str, layer: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    restore = [row for row in rows if row.get("record_type") == "group_restore"]
    drops = [row for row in rows if row.get("record_type") == "activation_drop"]

    def key(row: dict[str, str]) -> tuple[str, str, str, str]:
        run_id = row.get("sample_id", "") + "::" + row.get("prompt_name", "")
        return run_id, row.get("mask_condition", ""), row.get("control_group", ""), row.get("top_k", "")

    restore_idx = {key(row): row for row in restore}
    drop_idx = {key(row): row for row in drops}
    run_ids = sorted({row.get("sample_id", "") + "::" + row.get("prompt_name", "") for row in rows})
    topks = sorted({row.get("top_k", "") for row in rows if row.get("top_k", "")}, key=lambda raw: int(raw))
    for top_k in topks:
        for condition in ["answer_mask", "union_mask"]:
            restore_source_controls = []
            restore_real_shifted = []
            restore_real_shuffled = []
            restore_correct_wrong = []
            restore_rank = []
            drop_real_shifted = []
            drop_real_shuffled = []
            for run_id in run_ids:
                source_row = restore_idx.get((run_id, condition, "source", top_k))
                controls = [restore_idx[(run_id, condition, name, top_k)] for name in CONTROL_GROUPS if (run_id, condition, name, top_k) in restore_idx]
                shifted = restore_idx.get((run_id, "shifted_mask", "source", top_k))
                shuffled = restore_idx.get((run_id, "shuffled_mask", "source", top_k))
                if source_row and controls:
                    restore_source_controls.append(_f(source_row, "target_logit_effect") - _mean([_f(row, "target_logit_effect") for row in controls]))
                    restore_correct_wrong.append(_f(source_row, "target_logit_effect") - _f(source_row, "wrong_logit_effect"))
                    restore_rank.append(_f(source_row, "target_rank_effect"))
                if source_row and shifted:
                    restore_real_shifted.append(_f(source_row, "target_logit_effect") - _f(shifted, "target_logit_effect"))
                if source_row and shuffled:
                    restore_real_shuffled.append(_f(source_row, "target_logit_effect") - _f(shuffled, "target_logit_effect"))
                source_drop = drop_idx.get((run_id, condition, "source", top_k))
                shifted_drop = drop_idx.get((run_id, "shifted_mask", "source", top_k))
                shuffled_drop = drop_idx.get((run_id, "shuffled_mask", "source", top_k))
                if source_drop and shifted_drop:
                    drop_real_shifted.append(_f(source_drop, "activation_drop_sum") - _f(shifted_drop, "activation_drop_sum"))
                if source_drop and shuffled_drop:
                    drop_real_shuffled.append(_f(source_drop, "activation_drop_sum") - _f(shuffled_drop, "activation_drop_sum"))
            extra = {"stage": "plt_layer_sweep", "pack": pack, "layer": layer, "mask_condition": condition, "top_k": top_k}
            out.append(_metric_row("plt_restore_source_minus_controls", restore_source_controls, extra))
            out.append(_metric_row("plt_restore_real_minus_shifted", restore_real_shifted, extra))
            out.append(_metric_row("plt_restore_real_minus_shuffled", restore_real_shuffled, extra))
            out.append(_metric_row("plt_restore_correct_minus_wrong", restore_correct_wrong, extra))
            out.append(_metric_row("plt_restore_target_rank_effect", restore_rank, extra))
            out.append(_metric_row("plt_activation_drop_real_minus_shifted", drop_real_shifted, extra))
            out.append(_metric_row("plt_activation_drop_real_minus_shuffled", drop_real_shuffled, extra))
    return out


def _plt_specificity(cross_dir: Path, mode: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pack, layer, _cand_path, zero_path, group_path in _plt_pack_layer_files(cross_dir, mode):
        out.extend(_plt_zeroing_specificity(_read_csv(zero_path), pack, layer))
        out.extend(_plt_group_specificity(_read_csv(group_path), pack, layer))
    return out


def _best_hidden_gate(spec: list[dict[str, Any]], pack: str) -> dict[str, Any]:
    candidates = []
    for row in spec:
        if row.get("pack") != pack or row.get("metric") != "hidden_effect":
            continue
        attrs = {key: row.get(key, "") for key in ["layer", "direction", "position_group", "mask_condition", "scale"]}
        matching = {
            item["metric"]: item
            for item in spec
            if item.get("pack") == pack
            and all(item.get(key, "") == value for key, value in attrs.items())
        }
        gate = {
            **attrs,
            "effect": row,
            "real_minus_shifted": matching.get("hidden_real_minus_shifted", {}),
            "real_minus_shuffled": matching.get("hidden_real_minus_shuffled", {}),
            "correct_minus_wrong": matching.get("hidden_correct_minus_wrong", {}),
            "rank_effect": matching.get("hidden_rank_effect", {}),
        }
        gate["passed"] = (
            _positive(gate["effect"])
            and _positive(gate["real_minus_shifted"])
            and _positive(gate["real_minus_shuffled"])
            and _positive(gate["correct_minus_wrong"])
            and float(gate["rank_effect"].get("mean", 0.0)) > 0
        )
        gate["score"] = sum(
            float(gate[name].get("ci95_low", 0.0))
            for name in ["effect", "real_minus_shifted", "real_minus_shuffled", "correct_minus_wrong"]
        ) + 0.01 * float(gate["rank_effect"].get("mean", 0.0))
        candidates.append(gate)
    if not candidates:
        return {"passed": False, "reason": "no_hidden_rows"}
    return max(candidates, key=lambda item: (bool(item["passed"]), float(item["score"])))


def _hidden_gate_for_attrs(spec: list[dict[str, Any]], pack: str, attrs: dict[str, Any]) -> dict[str, Any]:
    if not attrs or attrs.get("reason"):
        return {"passed": False, "reason": "no_primary_gate_to_confirm"}
    wanted = {key: str(attrs.get(key, "")) for key in ["layer", "direction", "position_group", "mask_condition", "scale"]}
    matching = {
        item["metric"]: item
        for item in spec
        if item.get("pack") == pack
        and item.get("stage") == "hidden_lattice"
        and all(str(item.get(key, "")) == value for key, value in wanted.items())
    }
    if "hidden_effect" not in matching:
        return {**wanted, "passed": False, "reason": "matching_hidden_gate_missing"}
    gate = {
        **wanted,
        "effect": matching.get("hidden_effect", {}),
        "real_minus_shifted": matching.get("hidden_real_minus_shifted", {}),
        "real_minus_shuffled": matching.get("hidden_real_minus_shuffled", {}),
        "correct_minus_wrong": matching.get("hidden_correct_minus_wrong", {}),
        "rank_effect": matching.get("hidden_rank_effect", {}),
    }
    gate["passed"] = (
        _positive(gate["effect"])
        and _positive(gate["real_minus_shifted"])
        and _positive(gate["real_minus_shuffled"])
        and _positive(gate["correct_minus_wrong"])
        and float(gate["rank_effect"].get("mean", 0.0)) > 0
    )
    gate["score"] = sum(
        float(gate[name].get("ci95_low", 0.0))
        for name in ["effect", "real_minus_shifted", "real_minus_shuffled", "correct_minus_wrong"]
    ) + 0.01 * float(gate["rank_effect"].get("mean", 0.0))
    return gate


def _best_plt_gate(spec: list[dict[str, Any]], pack: str) -> dict[str, Any]:
    layers = sorted({row.get("layer", "") for row in spec if row.get("stage") == "plt_layer_sweep" and row.get("pack") == pack}, key=lambda raw: int(raw or 0))
    candidates = []
    for layer in layers:
        layer_rows = [row for row in spec if row.get("stage") == "plt_layer_sweep" and row.get("pack") == pack and row.get("layer") == layer]
        by_metric = {row["metric"]: row for row in layer_rows if row.get("top_k", "") == "" and row.get("mask_condition", "") == ""}
        zero_passed = (
            _positive(by_metric.get("plt_clean_zeroing_source_minus_controls", {}))
            and _positive(by_metric.get("plt_clean_zeroing_correct_minus_wrong", {}))
            and float(by_metric.get("plt_clean_zeroing_target_rank_effect", {}).get("mean", 0.0)) > 0
        )
        best_restore: dict[str, Any] = {"passed": False, "score": -999999.0}
        for condition in ["answer_mask", "union_mask"]:
            for top_k in ["1", "4", "8", "16", "32", "64"]:
                matching = {
                    row["metric"]: row
                    for row in layer_rows
                    if row.get("mask_condition") == condition and row.get("top_k") == top_k
                }
                restore_passed = (
                    _positive(matching.get("plt_restore_source_minus_controls", {}))
                    and _positive(matching.get("plt_restore_real_minus_shifted", {}))
                    and _positive(matching.get("plt_restore_real_minus_shuffled", {}))
                    and _positive(matching.get("plt_restore_correct_minus_wrong", {}))
                    and float(matching.get("plt_restore_target_rank_effect", {}).get("mean", 0.0)) > 0
                )
                score = sum(
                    float(matching.get(metric, {}).get("ci95_low", 0.0))
                    for metric in [
                        "plt_restore_source_minus_controls",
                        "plt_restore_real_minus_shifted",
                        "plt_restore_real_minus_shuffled",
                        "plt_restore_correct_minus_wrong",
                    ]
                )
                if restore_passed or score > float(best_restore.get("score", -999999.0)):
                    best_restore = {
                        "passed": restore_passed,
                        "condition": condition,
                        "top_k": top_k,
                        "score": score,
                        "metrics": matching,
                    }
        gate = {
            "layer": layer,
            "zeroing_passed": zero_passed,
            "zeroing": by_metric,
            "restore_gate": best_restore,
            "passed": zero_passed and bool(best_restore.get("passed")),
            "score": (10.0 if zero_passed else 0.0) + (10.0 if best_restore.get("passed") else 0.0) + float(best_restore.get("score", 0.0)),
        }
        candidates.append(gate)
    if not candidates:
        return {"passed": False, "reason": "no_plt_rows"}
    return max(candidates, key=lambda item: (bool(item["passed"]), float(item["score"])))


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    hidden = _hidden_rows(args.cross_dir, args.mode, args.hidden_tag)
    summary = _hidden_summary(hidden) + _plt_summary(args.cross_dir, args.mode)
    spec = _hidden_specificity(hidden) + _plt_specificity(args.cross_dir, args.mode)
    primary_gate = _best_hidden_gate(spec, "primary")
    strict_best_gate = _best_hidden_gate(spec, "strict")
    strict_gate = _hidden_gate_for_attrs(spec, "strict", primary_gate)
    primary_plt_gate = _best_plt_gate(spec, "primary")
    strict_plt_gate = _best_plt_gate(spec, "strict")
    if args.mode == "smoke":
        status = "blocked" if not hidden else "smoke_completed_not_decisive"
    elif primary_plt_gate.get("passed") and (strict_plt_gate.get("passed") or strict_plt_gate.get("reason") == "no_plt_rows"):
        status = "qwen_plt_layer_dependent_supported"
    elif primary_gate.get("passed") and (strict_gate.get("passed") or not any(row.get("_pack") == "strict" for row in hidden)):
        status = "qwen_hidden_route_supported"
    elif primary_gate.get("passed") and primary_plt_gate.get("reason") != "no_plt_rows":
        status = "qwen_hidden_only_plt_failed"
    elif hidden:
        status = "hidden_lattice_not_supported"
    else:
        status = "blocked"
    decision = {
        "created_at": _now(),
        "status": status,
        "mode": args.mode,
        "hidden_tag": args.hidden_tag,
        "hidden_rows": len(hidden),
        "hidden_prompt_runs": len({(row.get("_pack"), row.get("sample_id"), row.get("prompt_name")) for row in hidden}),
        "primary_hidden_gate": primary_gate,
        "strict_hidden_gate": strict_gate,
        "strict_best_hidden_gate_diagnostic": strict_best_gate,
        "primary_plt_gate": primary_plt_gate,
        "strict_plt_gate": strict_plt_gate,
        "claim_boundary": "This analyzer only decides the hidden causal lattice stage. A final sparse-route negative verdict requires PLT layer sweep, multilayer patch, and Adapter V4 artifacts too.",
    }
    summary_fields = [
        "stage",
        "metric",
        "pack",
        "layer",
        "direction",
        "position_group",
        "mask_condition",
        "scale",
        "top_k",
        "n",
        "prompt_runs",
        "main_rows",
        "zeroing_rows",
        "group_rows",
        "mean",
        "ci95_low",
        "ci95_high",
        "positive_frac",
        "rank_mean",
    ]
    specificity_fields = [
        "stage",
        "metric",
        "pack",
        "layer",
        "direction",
        "position_group",
        "mask_condition",
        "scale",
        "top_k",
        "n",
        "mean",
        "ci95_low",
        "ci95_high",
        "positive_frac",
    ]
    _write_csv(
        args.summary_csv,
        summary,
        summary_fields,
    )
    _write_csv(
        args.specificity_csv,
        spec,
        specificity_fields,
    )
    _write_json(args.decision_json, decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage4-024 Qwen decisive route artifacts.")
    parser.add_argument("--cross-dir", type=Path, default=CROSS)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--hidden-tag", default="", help="Optional hidden artifact suffix, e.g. alllayers.")
    parser.add_argument("--summary-csv", type=Path, default=CROSS / "stage4_qwen_decisive_route_summary.csv")
    parser.add_argument("--specificity-csv", type=Path, default=CROSS / "stage4_qwen_decisive_route_specificity.csv")
    parser.add_argument("--decision-json", type=Path, default=CROSS / "stage4_qwen_decisive_route_decision.json")
    args = parser.parse_args()
    analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
