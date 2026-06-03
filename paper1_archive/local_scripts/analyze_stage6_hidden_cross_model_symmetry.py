#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
STAGE6_CROSS = ROOT / "doc" / "experiments" / "stage6" / "cross_model"
STAGE4_CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"
PREFIX = "stage6_hidden_crossmodel_symmetric"
GEMMA_PREFIX = "stage6_gemma_hidden_lattice"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row}) if rows else ["status"]
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


def _ci(values: list[float], n: int = 2000, seed: int = 6014) -> tuple[float, float]:
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
    return int(row.get("n", 0) or 0) >= min_n and float(row.get("ci95_low", 0.0) or 0.0) > 0 and float(row.get("positive_frac", 0.0) or 0.0) >= 0.6


def _hidden_rows(mode: str, tag: str) -> list[dict[str, str]]:
    rows = []
    suffix = f"_{tag}" if tag else ""
    for pack in ["primary", "strict"]:
        path = STAGE6_CROSS / f"{GEMMA_PREFIX}_{pack}_{mode}{suffix}_raw.csv"
        for row in _read_csv(path):
            row["_pack"] = pack
            rows.append(row)
    return rows


def _run_json(pack: str, mode: str, tag: str) -> dict[str, Any]:
    suffix = f"_{tag}" if tag else ""
    path = STAGE6_CROSS / f"{GEMMA_PREFIX}_{pack}_{mode}{suffix}_run.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _hidden_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("token_scored") != "target":
            continue
        key = (
            row.get("_pack", ""),
            row.get("layer", ""),
            row.get("direction", ""),
            row.get("position_group", ""),
            row.get("mask_condition", ""),
            row.get("scale", ""),
        )
        grouped.setdefault(key, []).append(row)
    out = []
    for key, group in sorted(grouped.items(), key=lambda item: (item[0][0], int(item[0][1] or 0), item[0][2:])):
        pack, layer, direction, position_group, mask_condition, scale = key
        vals = [_f(row, "logit_effect") for row in group]
        ranks = [_f(row, "rank_effect") for row in group]
        out.append(
            _metric_row(
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
        )
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
    packs = sorted({row.get("_pack", "") for row in rows})
    layers = sorted({row.get("layer", "") for row in rows if row.get("layer", "")}, key=lambda raw: int(raw))
    for pack in packs:
        for layer in layers:
            for direction in ["restore", "corrupt"]:
                groups = sorted({row.get("position_group", "") for row in rows if row.get("_pack") == pack and row.get("layer") == layer})
                for position_group in groups:
                    scales = sorted({row.get("scale", "") for row in rows if row.get("_pack") == pack and row.get("layer") == layer})
                    for scale in scales:
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


def _best_hidden_gate(spec: list[dict[str, Any]], pack: str) -> dict[str, Any]:
    candidates = []
    for row in spec:
        if row.get("pack") != pack or row.get("metric") != "hidden_effect":
            continue
        attrs = {key: row.get(key, "") for key in ["layer", "direction", "position_group", "mask_condition", "scale"]}
        matching = {
            item["metric"]: item
            for item in spec
            if item.get("pack") == pack and all(item.get(key, "") == value for key, value in attrs.items())
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
            and float(gate["rank_effect"].get("mean", 0.0) or 0.0) > 0
        )
        gate["score"] = sum(
            float(gate[name].get("ci95_low", 0.0) or 0.0)
            for name in ["effect", "real_minus_shifted", "real_minus_shuffled", "correct_minus_wrong"]
        ) + 0.01 * float(gate["rank_effect"].get("mean", 0.0) or 0.0)
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
        and float(gate["rank_effect"].get("mean", 0.0) or 0.0) > 0
    )
    gate["score"] = sum(
        float(gate[name].get("ci95_low", 0.0) or 0.0)
        for name in ["effect", "real_minus_shifted", "real_minus_shuffled", "correct_minus_wrong"]
    ) + 0.01 * float(gate["rank_effect"].get("mean", 0.0) or 0.0)
    return gate


def _gate_table_row(model: str, pack: str, gate: dict[str, Any], layer_count: int | None) -> dict[str, Any]:
    layer = gate.get("layer", "")
    denom = (layer_count - 1) if layer_count and layer_count > 1 else ""
    layer_fraction = float(layer) / denom if denom != "" and layer != "" else ""
    return {
        "model": model,
        "pack": pack,
        "status": "passed" if gate.get("passed") else gate.get("reason", "not_passed"),
        "layer": layer,
        "normalized_layer_fraction": layer_fraction,
        "direction": gate.get("direction", ""),
        "position_group": gate.get("position_group", ""),
        "mask_condition": gate.get("mask_condition", ""),
        "effect_n": gate.get("effect", {}).get("n", ""),
        "effect_mean": gate.get("effect", {}).get("mean", ""),
        "effect_ci95_low": gate.get("effect", {}).get("ci95_low", ""),
        "effect_positive_frac": gate.get("effect", {}).get("positive_frac", ""),
        "real_minus_shifted_ci95_low": gate.get("real_minus_shifted", {}).get("ci95_low", ""),
        "real_minus_shuffled_ci95_low": gate.get("real_minus_shuffled", {}).get("ci95_low", ""),
        "correct_minus_wrong_ci95_low": gate.get("correct_minus_wrong", {}).get("ci95_low", ""),
        "rank_effect_mean": gate.get("rank_effect", {}).get("mean", ""),
    }


def _load_qwen_rows() -> list[dict[str, Any]]:
    path = STAGE4_CROSS / "stage4_qwen_decisive_route_decision.json"
    if not path.exists():
        return [{"model": "qwen2.5-vl", "pack": "primary", "status": "missing_qwen_decision"}]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        _gate_table_row("qwen2.5-vl", "primary", payload.get("primary_hidden_gate", {}), 28),
        _gate_table_row("qwen2.5-vl", "strict", payload.get("strict_hidden_gate", {}), 28),
    ]


def _write_verdict_markdown(path: Path, decision: dict[str, Any]) -> None:
    gemma_primary = decision.get("primary_hidden_gate", {})
    gemma_strict = decision.get("strict_hidden_gate", {})
    qwen_status = decision.get("qwen_decision_status", "")
    lines = [
        "# Stage6-015 Hidden Cross-Model Symmetric Verdict",
        "",
        f"Updated: {decision.get('created_at', '')}",
        "",
        "## Status",
        "",
        f"- Gemma hidden symmetric status: `{decision.get('status', '')}`",
        f"- Qwen existing hidden status: `{qwen_status}`",
        "",
        "## Gemma Primary Gate",
        "",
        f"- layer: `{gemma_primary.get('layer', '')}`",
        f"- direction: `{gemma_primary.get('direction', '')}`",
        f"- position group: `{gemma_primary.get('position_group', '')}`",
        f"- mask condition: `{gemma_primary.get('mask_condition', '')}`",
        f"- passed: `{gemma_primary.get('passed', False)}`",
        "",
        "## Gemma Strict Confirmation",
        "",
        f"- layer: `{gemma_strict.get('layer', '')}`",
        f"- direction: `{gemma_strict.get('direction', '')}`",
        f"- position group: `{gemma_strict.get('position_group', '')}`",
        f"- mask condition: `{gemma_strict.get('mask_condition', '')}`",
        f"- passed: `{gemma_strict.get('passed', False)}`",
        "",
        "## Boundary",
        "",
        "This hidden residual lens is symmetric with the Qwen Stage4 hidden lattice. It does not replace Gemma sparse PLT source tracing and does not claim Gemma/Qwen graph topology is identical.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage6-014 Gemma/Qwen hidden cross-model symmetry.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--tag", default="symmetric_v1")
    args = parser.parse_args()

    rows = _hidden_rows(args.mode, args.tag)
    summary = _hidden_summary(rows)
    specificity = _hidden_specificity(rows)
    _write_csv(STAGE6_CROSS / f"{PREFIX}_{args.mode}_{args.tag}_gemma_summary.csv", summary)
    _write_csv(STAGE6_CROSS / f"{PREFIX}_{args.mode}_{args.tag}_gemma_specificity.csv", specificity)

    primary_gate = _best_hidden_gate(specificity, "primary")
    strict_best_gate = _best_hidden_gate(specificity, "strict")
    strict_gate = _hidden_gate_for_attrs(specificity, "strict", primary_gate)
    primary_run = _run_json("primary", args.mode, args.tag)
    strict_run = _run_json("strict", args.mode, args.tag)
    layer_count = primary_run.get("available_layer_count") or strict_run.get("available_layer_count")

    if not rows:
        status = "blocked_no_hidden_rows"
    elif primary_gate.get("passed") and strict_gate.get("passed"):
        status = "gemma_hidden_route_supported"
    elif primary_gate.get("passed") and not any(row.get("_pack") == "strict" for row in rows):
        status = "gemma_hidden_route_primary_only"
    elif primary_gate.get("passed"):
        status = "gemma_hidden_route_strict_not_confirmed"
    else:
        status = "gemma_hidden_route_not_supported"

    qwen_decision = {}
    qwen_path = STAGE4_CROSS / "stage4_qwen_decisive_route_decision.json"
    if qwen_path.exists():
        qwen_decision = json.loads(qwen_path.read_text(encoding="utf-8"))

    cross_rows = [
        _gate_table_row("gemma3", "primary", primary_gate, int(layer_count) if layer_count else None),
        _gate_table_row("gemma3", "strict", strict_gate, int(layer_count) if layer_count else None),
        *_load_qwen_rows(),
    ]
    _write_csv(STAGE6_CROSS / f"{PREFIX}_{args.mode}_{args.tag}_crossmodel_table.csv", cross_rows)

    decision = {
        "created_at": _now(),
        "status": status,
        "mode": args.mode,
        "tag": args.tag,
        "hidden_rows": len(rows),
        "hidden_prompt_runs": len({(row.get("sample_id", ""), row.get("prompt_name", ""), row.get("_pack", "")) for row in rows}),
        "gemma_available_layer_count": layer_count,
        "primary_hidden_gate": primary_gate,
        "strict_hidden_gate": strict_gate,
        "strict_best_hidden_gate_diagnostic": strict_best_gate,
        "qwen_decision_status": qwen_decision.get("status", "missing"),
        "qwen_decision_source": str(qwen_path),
        "summary_csv": str(STAGE6_CROSS / f"{PREFIX}_{args.mode}_{args.tag}_gemma_summary.csv"),
        "specificity_csv": str(STAGE6_CROSS / f"{PREFIX}_{args.mode}_{args.tag}_gemma_specificity.csv"),
        "crossmodel_table_csv": str(STAGE6_CROSS / f"{PREFIX}_{args.mode}_{args.tag}_crossmodel_table.csv"),
        "claim_boundary": "Symmetric hidden residual lens only; raw logit magnitudes are not compared across models.",
    }
    _write_json(STAGE6_CROSS / f"{PREFIX}_{args.mode}_{args.tag}_decision.json", decision)
    _write_verdict_markdown(ROOT / "doc" / "experiments" / "stage6" / "015_stage6_hidden_cross_model_symmetric_verdict.md", decision)
    print(json.dumps({"status": status, "rows": len(rows), "decision": str(STAGE6_CROSS / f"{PREFIX}_{args.mode}_{args.tag}_decision.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
