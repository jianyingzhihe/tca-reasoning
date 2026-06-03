#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
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


def _safe_float(raw: Any) -> float:
    try:
        if raw in ("", None):
            return math.nan
        return float(raw)
    except Exception:  # noqa: BLE001
        return math.nan


def _mean(vals: list[float]) -> float:
    vals = [v for v in vals if not math.isnan(v)]
    return float(sum(vals) / len(vals)) if vals else math.nan


def _frac(vals: list[bool]) -> float:
    return float(sum(1 for v in vals if v) / len(vals)) if vals else math.nan


def _fmt(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.6g}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage4 Qwen source-tracing artifacts.")
    parser.add_argument("--cross-dir", default="doc/experiments/stage4/cross_model")
    parser.add_argument("--pack", choices=["primary", "strict"], required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--out-prefix", default="")
    args = parser.parse_args()

    cross_dir = Path(args.cross_dir)
    prefix = args.out_prefix or f"stage4_qwen_source_tracing_{args.pack}_{args.mode}"
    meta_a = _read_csv(cross_dir / f"{prefix}_meta_a.csv")
    meta_b = _read_csv(cross_dir / f"{prefix}_meta_b.csv")
    valid = _read_csv(cross_dir / f"{prefix}_valid_samples.csv")
    compare = _read_csv(cross_dir / f"{prefix}_sample_compare_controlled.csv")
    nodes = _read_csv(cross_dir / f"{prefix}_nodes_detailed_controlled.csv")
    edges = _read_csv(cross_dir / f"{prefix}_edges_detailed_controlled.csv")
    intervention = _read_csv(cross_dir / f"{prefix}_intervention.csv")

    ok_a = [row for row in meta_a if row.get("status") == "ok"]
    ok_b = [row for row in meta_b if row.get("status") == "ok"]
    graph_success = (len(ok_a) + len(ok_b)) / (len(meta_a) + len(meta_b)) if (meta_a or meta_b) else 0.0
    prompt_runs_ok = len(ok_a) + len(ok_b)
    feature_nodes = [row for row in nodes if row.get("node_type") == "feature"]
    deltas = [_safe_float(row.get("delta_target_logit")) for row in intervention]
    rank_deltas = [_safe_float(row.get("delta_target_rank")) for row in intervention]
    negative_frac = _frac([v < 0 for v in deltas if not math.isnan(v)])
    rank_hurt_frac = _frac([v > 0 for v in rank_deltas if not math.isnan(v)])
    by_mode: dict[str, dict[str, float]] = {}
    for mode_name in sorted({row.get("zeroing_mode", "subtract") or "subtract" for row in intervention}):
        part = [row for row in intervention if (row.get("zeroing_mode", "subtract") or "subtract") == mode_name]
        mode_deltas = [_safe_float(row.get("delta_target_logit")) for row in part]
        mode_ranks = [_safe_float(row.get("delta_target_rank")) for row in part]
        by_mode[mode_name] = {
            "rows": float(len(part)),
            "mean_delta_target_logit": _mean(mode_deltas),
            "frac_negative_delta_target_logit": _frac([v < 0 for v in mode_deltas if not math.isnan(v)]),
            "frac_rank_hurt": _frac([v > 0 for v in mode_ranks if not math.isnan(v)]),
        }
    best_mode = ""
    best_negative_frac = negative_frac
    if by_mode:
        best_mode, best_payload = max(
            by_mode.items(),
            key=lambda item: (
                -1 if math.isnan(item[1]["frac_negative_delta_target_logit"]) else item[1]["frac_negative_delta_target_logit"]
            ),
        )
        best_negative_frac = best_payload["frac_negative_delta_target_logit"]

    summary_rows = [
        {
            "pack": args.pack,
            "mode": args.mode,
            "meta_a_rows": len(meta_a),
            "meta_b_rows": len(meta_b),
            "graph_ok_a": len(ok_a),
            "graph_ok_b": len(ok_b),
            "prompt_runs_ok": prompt_runs_ok,
            "graph_success_rate": _fmt(graph_success),
            "valid_samples": len(valid),
            "compare_rows": len(compare),
            "node_rows": len(nodes),
            "feature_node_rows": len(feature_nodes),
            "edge_rows": len(edges),
            "intervention_rows": len(intervention),
            "mean_delta_target_logit": _fmt(_mean(deltas)),
            "frac_negative_delta_target_logit": _fmt(negative_frac),
            "frac_rank_hurt": _fmt(rank_hurt_frac),
            "best_zeroing_mode": best_mode,
            "best_mode_frac_negative_delta_target_logit": _fmt(best_negative_frac),
        }
    ]
    _write_csv(cross_dir / f"{prefix}_analysis_summary.csv", summary_rows, list(summary_rows[0].keys()))

    if not meta_a and not meta_b:
        status = "qwen_adapter_blocked"
        reason = "missing_meta_outputs"
    elif graph_success < 0.8 and args.mode == "full":
        status = "qwen_adapter_blocked"
        reason = "graph_success_below_80pct"
    elif not compare or not nodes or not edges:
        status = "qwen_adapter_blocked"
        reason = "missing_compare_node_or_edge_outputs"
    elif args.mode == "smoke" and intervention and best_negative_frac >= 0.5:
        status = "qwen_smoke_passed"
        reason = "smoke_artifacts_present_and_zeroing_direction_supportive"
    elif args.mode == "smoke" and intervention:
        status = "qwen_smoke_artifacts_passed_direction_not_supported"
        reason = "smoke_artifacts_present_but_feature_zeroing_did_not_hurt_target"
    elif not intervention:
        status = "qwen_approx_only"
        reason = "graph_compare_present_but_intervention_missing"
    elif best_negative_frac >= 0.55 and prompt_runs_ok >= (120 if args.mode == "full" else 4):
        status = "qwen_full_source_tracing_supported"
        reason = "graph_compare_and_feature_zeroing_direction_positive"
    elif args.mode == "full":
        status = "qwen_source_tracing_not_supported"
        reason = "adapter_ran_but_intervention_direction_not_sufficient"
    else:
        status = "qwen_approx_only"
        reason = "insufficient_smoke_direction"

    decision = {
        "status": status,
        "reason": reason,
        "pack": args.pack,
        "mode": args.mode,
        "metrics": summary_rows[0],
        "zeroing_mode_summary": {
            mode_name: {key: _fmt(value) if isinstance(value, float) else value for key, value in payload.items()}
            for mode_name, payload in by_mode.items()
        },
        "claim_boundary": (
            "Qwen full source-tracing verdict is only supported if adapter graph, compare, "
            "intervention, and controls all pass. Blocked adapter is not negative mechanism evidence."
        ),
    }
    _write_json(cross_dir / f"{prefix}_decision.json", decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
