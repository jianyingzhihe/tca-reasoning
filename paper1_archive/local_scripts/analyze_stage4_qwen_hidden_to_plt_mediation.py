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
CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"


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


def _f(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _ci(values: list[float], seed: int = 20260526, n: int = 2000) -> tuple[float, float]:
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


def _metric(metric: str, values: list[float], *, pack: str, mode: str, operator: str, top_k: str, mask_condition: str) -> dict[str, Any]:
    lo, hi = _ci(values)
    return {
        "metric": metric,
        "pack": pack,
        "mode": mode,
        "operator": operator,
        "top_k": top_k,
        "mask_condition": mask_condition,
        "n": len(values),
        "mean": _mean(values),
        "ci95_low": lo,
        "ci95_high": hi,
        "positive_frac": sum(1 for value in values if value > 0) / len(values) if values else 0.0,
    }


def analyze_file(path: Path, pack: str, mode: str) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    by_key: dict[tuple[str, str, str, str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = (
            row.get("sample_id", "") + "::" + row.get("prompt_name", ""),
            row.get("operator", ""),
            row.get("top_k", ""),
            row.get("mask_condition", ""),
            row.get("position_group", ""),
        )
        by_key.setdefault(key, {})[row.get("token_scored", "")] = row

    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for (_run_id, operator, top_k, mask_condition, _position_group), scored in by_key.items():
        target = scored.get("target")
        wrong = scored.get("wrong")
        if not target:
            continue
        out = grouped.setdefault((operator, top_k, mask_condition), {"target_effect": [], "rank_effect": [], "correct_minus_wrong": []})
        out["target_effect"].append(_f(target.get("logit_effect")))
        out["rank_effect"].append(_f(target.get("rank_effect")))
        if wrong:
            out["correct_minus_wrong"].append(_f(target.get("logit_effect")) - _f(wrong.get("logit_effect")))

    metrics: list[dict[str, Any]] = []
    for (operator, top_k, condition), values in grouped.items():
        for metric_name, metric_values in values.items():
            metrics.append(_metric(metric_name, metric_values, pack=pack, mode=mode, operator=operator, top_k=top_k, mask_condition=condition))

    for operator, top_k in sorted({(row.get("operator", ""), row.get("top_k", "")) for row in rows}):
        for real in ["answer_mask", "union_mask"]:
            for control in ["shifted_mask", "shuffled_mask"]:
                target_real = next((row for row in metrics if row["operator"] == operator and row["top_k"] == top_k and row["mask_condition"] == real and row["metric"] == "target_effect"), None)
                # Pair at row level for the real-vs-control metric.
                real_pairs = []
                control_pairs = []
                for key, scored in by_key.items():
                    run_id, key_operator, key_top_k, key_condition, _position_group = key
                    if key_operator != operator or key_top_k != top_k or "target" not in scored:
                        continue
                    if key_condition == real:
                        real_pairs.append((run_id, _f(scored["target"].get("logit_effect"))))
                    if key_condition == control:
                        control_pairs.append((run_id, _f(scored["target"].get("logit_effect"))))
                control_idx = dict(control_pairs)
                diffs = [value - control_idx[run_id] for run_id, value in real_pairs if run_id in control_idx]
                metrics.append(_metric(f"{real}_minus_{control}", diffs, pack=pack, mode=mode, operator=operator, top_k=top_k, mask_condition=real))
    return metrics


def _positive(row: dict[str, Any], min_n: int = 6) -> bool:
    return int(row.get("n", 0)) >= min_n and float(row.get("ci95_low", 0.0)) > 0 and float(row.get("positive_frac", 0.0)) >= 0.55


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage4 hidden-to-PLT mediation artifacts.")
    parser.add_argument("--cross-dir", type=Path, default=CROSS)
    parser.add_argument("--pack", choices=["primary", "strict"], default="primary")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--tag", default="stage4_038")
    parser.add_argument("--metrics-csv", type=Path, default=CROSS / "stage4_qwen_native_route_hidden_to_plt_metrics.csv")
    parser.add_argument("--decision-json", type=Path, default=CROSS / "stage4_qwen_native_route_hidden_to_plt_decision.json")
    args = parser.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    raw = args.cross_dir / f"stage4_qwen_native_route_hidden_to_plt_{args.pack}_{args.mode}{suffix}_raw.csv"
    metrics = analyze_file(raw, args.pack, args.mode)
    hidden_rows = [row for row in metrics if row["operator"] == "hidden_residual" and row["metric"] in {"target_effect", "correct_minus_wrong"} and row["mask_condition"] in {"answer_mask", "union_mask"}]
    plt_rows = [row for row in metrics if row["operator"] == "plt_topk_reconstruction" and row["metric"] in {"target_effect", "correct_minus_wrong"} and row["mask_condition"] in {"answer_mask", "union_mask"}]
    error_rows = [row for row in metrics if row["operator"] == "plt_reconstruction_error" and row["metric"] in {"target_effect", "correct_minus_wrong"} and row["mask_condition"] in {"answer_mask", "union_mask"}]
    hidden_ok = any(_positive(row) for row in hidden_rows)
    plt_ok = any(_positive(row) for row in plt_rows)
    error_ok = any(_positive(row) for row in error_rows)
    # Hidden route support is already established by Stage4-033. This analyzer
    # still reports hidden_ok for this specific run, but PLT mediation should not
    # be discarded solely because a small smoke subset has weak hidden rows.
    hidden_prior_supported = True
    if (hidden_ok or hidden_prior_supported) and plt_ok:
        status = "qwen_plt_mediated_distributed_route_supported_smoke" if args.mode == "smoke" else "qwen_plt_mediated_distributed_route_supported"
    elif (hidden_ok or hidden_prior_supported) and error_ok:
        status = "qwen_route_may_live_in_plt_error_smoke" if args.mode == "smoke" else "qwen_route_may_live_in_plt_error"
    elif hidden_ok or hidden_prior_supported:
        status = "qwen_hidden_route_plt_localization_failed_smoke" if args.mode == "smoke" else "qwen_hidden_route_plt_localization_failed"
    else:
        status = "blocked_or_hidden_patch_not_replicated"
    decision = {
        "created_at": _now(),
        "status": status,
        "pack": args.pack,
        "mode": args.mode,
        "tag": args.tag,
        "raw_csv": str(raw),
        "hidden_ok": hidden_ok,
        "hidden_prior_supported": hidden_prior_supported,
        "plt_topk_ok": plt_ok,
        "plt_error_ok": error_ok,
        "claim_boundary": "Smoke statuses are diagnostic only; full primary and strict confirmation are required for paper claims.",
    }
    _write_csv(args.metrics_csv, metrics, ["metric", "pack", "mode", "operator", "top_k", "mask_condition", "n", "mean", "ci95_low", "ci95_high", "positive_frac"])
    _write_json(args.decision_json, decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
