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
STAGE3_CROSS = ROOT / "doc" / "experiments" / "stage3" / "cross_model"
PREFIX = "stage6_gemma_hidden_to_plt_decomp"
OUT_PREFIX = "stage6_hidden_to_plt_crossmodel"


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


def _f(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _ci(values: list[float], seed: int = 6016, n: int = 2000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        means.append(_mean([values[rng.randrange(len(values))] for _ in values]))
    means.sort()
    return means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]


def _metric(metric: str, values: list[float], extra: dict[str, Any]) -> dict[str, Any]:
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


def _stem(pack: str, mode: str, tag: str) -> str:
    suffix = f"_{tag}" if tag else ""
    return f"{PREFIX}_{pack}_{mode}{suffix}"


def _load_rows(mode: str, tag: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for pack in ["primary", "strict"]:
        path = STAGE6_CROSS / f"{_stem(pack, mode, tag)}_raw.csv"
        for row in _read_csv(path):
            row["_pack"] = pack
            rows.append(row)
    return rows


def _summarize(rows: list[dict[str, str]], mode: str) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str, str, str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = (
            row.get("_pack", ""),
            f"{row.get('sample_id', '')}::{row.get('prompt_name', '')}",
            row.get("operator", ""),
            row.get("top_k", ""),
            row.get("mask_condition", ""),
            row.get("position_group", ""),
            row.get("layer", ""),
        )
        by_key.setdefault(key, {})[row.get("token_scored", "")] = row

    grouped: dict[tuple[str, str, str, str, str], dict[str, list[float]]] = {}
    for (pack, _run_id, operator, top_k, condition, position_group, layer), scored in by_key.items():
        target = scored.get("target")
        wrong = scored.get("wrong")
        if not target:
            continue
        out = grouped.setdefault(
            (pack, operator, top_k, condition, position_group),
            {"target_effect": [], "rank_effect": [], "correct_minus_wrong": []},
        )
        out["target_effect"].append(_f(target.get("logit_effect")))
        out["rank_effect"].append(_f(target.get("rank_effect")))
        if wrong:
            out["correct_minus_wrong"].append(_f(target.get("logit_effect")) - _f(wrong.get("logit_effect")))

    metrics: list[dict[str, Any]] = []
    for (pack, operator, top_k, condition, position_group), values in grouped.items():
        extra = {
            "model": "gemma3",
            "pack": pack,
            "mode": mode,
            "operator": operator,
            "top_k": top_k,
            "mask_condition": condition,
            "position_group": position_group,
        }
        for metric_name, metric_values in values.items():
            metrics.append(_metric(metric_name, metric_values, extra))

    for pack, operator, top_k, position_group in sorted({(r.get("_pack", ""), r.get("operator", ""), r.get("top_k", ""), r.get("position_group", "")) for r in rows}):
        for real in ["answer_mask", "union_mask"]:
            for control in ["shifted_mask", "shuffled_mask"]:
                real_pairs = []
                control_pairs = []
                for key, scored in by_key.items():
                    key_pack, run_id, key_operator, key_top_k, key_condition, key_position_group, _layer = key
                    if key_pack != pack or key_operator != operator or key_top_k != top_k or key_position_group != position_group or "target" not in scored:
                        continue
                    if key_condition == real:
                        real_pairs.append((run_id, _f(scored["target"].get("logit_effect"))))
                    if key_condition == control:
                        control_pairs.append((run_id, _f(scored["target"].get("logit_effect"))))
                control_idx = dict(control_pairs)
                diffs = [value - control_idx[run_id] for run_id, value in real_pairs if run_id in control_idx]
                metrics.append(
                    _metric(
                        f"{real}_minus_{control}",
                        diffs,
                        {
                            "model": "gemma3",
                            "pack": pack,
                            "mode": mode,
                            "operator": operator,
                            "top_k": top_k,
                            "mask_condition": real,
                            "position_group": position_group,
                        },
                    )
                )
    return metrics


def _ratio_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (
            row.get("pack", ""),
            row.get("mask_condition", ""),
            row.get("position_group", ""),
            row.get("metric", ""),
            row.get("operator", ""),
            row.get("top_k", ""),
        ): row
        for row in metrics
    }
    out = []
    for row in metrics:
        if row.get("operator") not in {"plt_topk_reconstruction", "plt_reconstruction_error"}:
            continue
        if row.get("metric") != "target_effect":
            continue
        hidden = index.get((row.get("pack", ""), row.get("mask_condition", ""), row.get("position_group", ""), "target_effect", "hidden_residual", "full"))
        if not hidden:
            continue
        hidden_mean = float(hidden.get("mean", 0.0) or 0.0)
        row_mean = float(row.get("mean", 0.0) or 0.0)
        out.append(
            {
                "model": "gemma3",
                "pack": row.get("pack", ""),
                "mask_condition": row.get("mask_condition", ""),
                "position_group": row.get("position_group", ""),
                "operator": row.get("operator", ""),
                "top_k": row.get("top_k", ""),
                "operator_mean": row_mean,
                "hidden_residual_mean": hidden_mean,
                "retention_over_hidden": row_mean / hidden_mean if abs(hidden_mean) > 1e-9 else "",
                "operator_ci95_low": row.get("ci95_low", ""),
                "hidden_ci95_low": hidden.get("ci95_low", ""),
            }
        )
    return out


def _source_route_rows() -> list[dict[str, Any]]:
    out = []
    for pack in ["primary", "strict"]:
        analysis = STAGE3_CROSS / f"stage3_gemma_source_tracing_{pack}_full_analysis.json"
        compare = STAGE3_CROSS / f"stage3_gemma_source_tracing_{pack}_full_sample_compare_controlled.csv"
        payload = json.loads(analysis.read_text(encoding="utf-8")) if analysis.exists() else {}
        rows = _read_csv(compare)
        deltas = [_f(row.get("delta_target_total_in_abs")) for row in rows]
        out.append(
            {
                "model": "gemma3",
                "pack": pack,
                "operator": "source_tracing_route",
                "status": payload.get("status", "missing"),
                "n": len(rows),
                "node_overlap_jaccard_mean": payload.get("compare_summary", {}).get("node_overlap_jaccard_mean", ""),
                "edge_overlap_jaccard_mean": payload.get("compare_summary", {}).get("edge_overlap_jaccard_mean", ""),
                "delta_target_total_in_abs_mean": _mean(deltas),
                "graph_success_rate_vs_valid": payload.get("counts", {}).get("graph_success_rate_vs_valid", ""),
                "claim_boundary": "Imported existing Gemma Stage3 source-tracing graph compare; no graph rerun.",
            }
        )
    return out


def _qwen_rows() -> list[dict[str, Any]]:
    path = STAGE4_CROSS / "stage4_qwen_native_route_hidden_to_plt_metrics.csv"
    out = []
    for row in _read_csv(path):
        if row.get("metric") != "target_effect" or row.get("mask_condition") not in {"answer_mask", "union_mask"}:
            continue
        out.append(
            {
                "model": "qwen2.5-vl",
                "pack": row.get("pack", ""),
                "operator": row.get("operator", ""),
                "top_k": row.get("top_k", ""),
                "mask_condition": row.get("mask_condition", ""),
                "mean": row.get("mean", ""),
                "ci95_low": row.get("ci95_low", ""),
                "positive_frac": row.get("positive_frac", ""),
                "source": "stage4_qwen_native_route_hidden_to_plt_metrics.csv",
            }
        )
    return out


def _best_pass(metrics: list[dict[str, Any]], pack: str, operator: str) -> dict[str, Any] | None:
    candidates = [
        row
        for row in metrics
        if row.get("pack") == pack
        and row.get("operator") == operator
        and row.get("metric") == "target_effect"
        and row.get("mask_condition") in {"answer_mask", "union_mask"}
        and _positive(row)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row.get("ci95_low", 0.0) or 0.0))


def _decide(metrics: list[dict[str, Any]], ratios: list[dict[str, Any]], source_routes: list[dict[str, Any]], mode: str, tag: str) -> dict[str, Any]:
    primary_hidden = _best_pass(metrics, "primary", "hidden_residual")
    strict_hidden = _best_pass(metrics, "strict", "hidden_residual")
    primary_topk = _best_pass(metrics, "primary", "plt_topk_reconstruction")
    strict_topk = _best_pass(metrics, "strict", "plt_topk_reconstruction")
    primary_error = _best_pass(metrics, "primary", "plt_reconstruction_error")
    strict_error = _best_pass(metrics, "strict", "plt_reconstruction_error")

    source_ok = all(row.get("status", "").startswith(("primary_full_passed", "strict_full_graph_compare_passed", "strict_full")) for row in source_routes)
    topk_ok = primary_topk is not None and strict_topk is not None
    error_ok = primary_error is not None and strict_error is not None
    hidden_ok = primary_hidden is not None and strict_hidden is not None

    topk_retention = [
        _f(row.get("retention_over_hidden"))
        for row in ratios
        if row.get("operator") == "plt_topk_reconstruction"
        and row.get("pack") in {"primary", "strict"}
        and row.get("mask_condition") in {"answer_mask", "union_mask"}
    ]
    error_retention = [
        _f(row.get("retention_over_hidden"))
        for row in ratios
        if row.get("operator") == "plt_reconstruction_error"
        and row.get("pack") in {"primary", "strict"}
        and row.get("mask_condition") in {"answer_mask", "union_mask"}
    ]
    topk_retention_mean = _mean(topk_retention)
    error_retention_mean = _mean(error_retention)

    if not hidden_ok:
        status = "gemma_hidden_to_plt_decomp_blocked" if not metrics else "gemma_hidden_to_plt_decomp_hidden_not_replicated"
    elif topk_ok and topk_retention_mean >= 0.35 and source_ok:
        status = "gemma_hidden_flow_sparse_plt_captured"
    elif source_ok and (topk_ok or topk_retention_mean > 0.1):
        status = "gemma_hidden_flow_route_captured_but_topk_partial"
    elif error_ok and error_retention_mean >= max(0.35, topk_retention_mean):
        status = "gemma_hidden_flow_error_heavy_like_qwen"
    else:
        status = "gemma_hidden_flow_route_captured_but_topk_partial" if source_ok else "gemma_hidden_to_plt_decomp_mixed"

    return {
        "created_at": _now(),
        "status": status,
        "mode": mode,
        "tag": tag,
        "gemma_metrics_rows": len(metrics),
        "primary_hidden_gate": primary_hidden,
        "strict_hidden_gate": strict_hidden,
        "primary_topk_gate": primary_topk,
        "strict_topk_gate": strict_topk,
        "primary_error_gate": primary_error,
        "strict_error_gate": strict_error,
        "source_route_imported": source_routes,
        "source_route_ok": source_ok,
        "topk_retention_mean": topk_retention_mean,
        "error_retention_mean": error_retention_mean,
        "claim_boundary": "Within-model normalized decomposition only; no cross-model raw logit magnitude comparison.",
    }


def _write_verdict(decision: dict[str, Any], mode: str, tag: str) -> None:
    path = ROOT / "doc" / "experiments" / "stage6" / "017_stage6_hidden_to_plt_cross_model_verdict.md"
    lines = [
        "# Stage6-017 Hidden-to-PLT Cross-Model Verdict",
        "",
        f"Updated: {decision.get('created_at', '')}",
        "",
        "## Status",
        "",
        f"- Gemma decomposition status: `{decision.get('status', '')}`",
        "- Qwen decomposition status: `qwen_route_may_live_in_plt_error`",
        "",
        "## Gemma Gates",
        "",
        f"- hidden primary: `{bool(decision.get('primary_hidden_gate'))}`",
        f"- hidden strict: `{bool(decision.get('strict_hidden_gate'))}`",
        f"- PLT topK primary: `{bool(decision.get('primary_topk_gate'))}`",
        f"- PLT topK strict: `{bool(decision.get('strict_topk_gate'))}`",
        f"- PLT error primary: `{bool(decision.get('primary_error_gate'))}`",
        f"- PLT error strict: `{bool(decision.get('strict_error_gate'))}`",
        "",
        "## Boundary",
        "",
        "This compares hidden-to-PLT decomposition lenses. It does not compare raw logit magnitudes across models and does not claim Gemma/Qwen sparse topology is identical.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage6-016 Gemma hidden-to-PLT decomposition.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--tag", default="decomp_v1")
    args = parser.parse_args()

    rows = _load_rows(args.mode, args.tag)
    metrics = _summarize(rows, args.mode)
    ratios = _ratio_rows(metrics)
    source_routes = _source_route_rows()
    qwen = _qwen_rows()
    decision = _decide(metrics, ratios, source_routes, args.mode, args.tag)

    suffix = f"_{args.tag}" if args.tag else ""
    metrics_path = STAGE6_CROSS / f"{OUT_PREFIX}_{args.mode}{suffix}_gemma_metrics.csv"
    ratio_path = STAGE6_CROSS / f"{OUT_PREFIX}_{args.mode}{suffix}_retention_ratios.csv"
    route_path = STAGE6_CROSS / f"{OUT_PREFIX}_{args.mode}{suffix}_source_route_import.csv"
    qwen_path = STAGE6_CROSS / f"{OUT_PREFIX}_{args.mode}{suffix}_qwen_reference_metrics.csv"
    decision_path = STAGE6_CROSS / f"{OUT_PREFIX}_{args.mode}{suffix}_decision.json"
    _write_csv(metrics_path, metrics)
    _write_csv(ratio_path, ratios)
    _write_csv(route_path, source_routes)
    _write_csv(qwen_path, qwen)
    decision.update(
        {
            "metrics_csv": str(metrics_path),
            "ratios_csv": str(ratio_path),
            "source_route_csv": str(route_path),
            "qwen_reference_csv": str(qwen_path),
        }
    )
    _write_json(decision_path, decision)
    _write_verdict(decision, args.mode, args.tag)
    print(json.dumps({"status": decision["status"], "rows": len(rows), "decision": str(decision_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
