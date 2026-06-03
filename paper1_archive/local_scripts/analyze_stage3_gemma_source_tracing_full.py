#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(r"E:\Bridging")
CROSS = ROOT / "doc" / "experiments" / "stage3" / "cross_model"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def numeric_mean(rows: list[dict[str, str]], field: str) -> float | str:
    values = [v for row in rows if (v := to_float(row.get(field))) is not None]
    return mean(values) if values else ""


def positive_rate(rows: list[dict[str, str]], field: str) -> float | str:
    values = [v for row in rows if (v := to_float(row.get(field))) is not None]
    return sum(value > 0 for value in values) / len(values) if values else ""


def summarize_by(rows: list[dict[str, str]], group_fields: list[str], numeric_fields: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in group_fields)].append(row)
    out: list[dict[str, Any]] = []
    for key, part in sorted(grouped.items()):
        record: dict[str, Any] = {field: value for field, value in zip(group_fields, key)}
        record["n"] = len(part)
        for field in numeric_fields:
            record[f"{field}_mean"] = numeric_mean(part, field)
            record[f"{field}_positive_rate"] = positive_rate(part, field)
        out.append(record)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage3 Gemma source-tracing full/smoke outputs.")
    parser.add_argument("--pack", choices=["primary", "strict"], default="primary")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = parser.parse_args()

    suffix = f"{args.pack}_{args.mode}"
    decision_path = CROSS / f"stage3_gemma_source_tracing_{suffix}_decision.json"
    eval_a = read_csv(CROSS / f"stage3_gemma_source_tracing_{suffix}_eval_A_D_visual_only.csv")
    eval_b = read_csv(CROSS / f"stage3_gemma_source_tracing_{suffix}_eval_B_B_direct.csv")
    valid = read_csv(CROSS / f"stage3_gemma_source_tracing_{suffix}_valid_samples.csv")
    failures = read_csv(CROSS / f"stage3_gemma_source_tracing_{suffix}_failure_manifest.csv")
    meta_a = read_csv(CROSS / f"stage3_gemma_source_tracing_{suffix}_meta_a.csv")
    meta_b = read_csv(CROSS / f"stage3_gemma_source_tracing_{suffix}_meta_b.csv")
    sample_compare = read_csv(CROSS / f"stage3_gemma_source_tracing_{suffix}_sample_compare_controlled.csv")
    bucket_summary = read_csv(CROSS / f"stage3_gemma_source_tracing_{suffix}_bucket_summary_controlled.csv")
    nodes = read_csv(CROSS / f"stage3_gemma_source_tracing_{suffix}_nodes_detailed_controlled.csv")
    edges = read_csv(CROSS / f"stage3_gemma_source_tracing_{suffix}_edges_detailed_controlled.csv")
    intervention = read_csv(CROSS / f"stage3_gemma_source_tracing_{suffix}_intervention.csv")

    decision: dict[str, Any] = {}
    if decision_path.exists():
        try:
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            decision = {"status": "missing_or_incomplete_decision_json"}

    valid_count = len(valid)
    graph_a_count = len({row.get("sample_id", "") for row in meta_a if row.get("status", "ok") != "error"})
    graph_b_count = len({row.get("sample_id", "") for row in meta_b if row.get("status", "ok") != "error"})
    graph_success_rate = min(graph_a_count, graph_b_count) / valid_count if valid_count else 0.0

    eval_summary_rows = [
        {
            "split": "A_D_visual_only",
            "rows": len(eval_a),
            "empty_generated": sum(1 for row in eval_a if not (row.get("generated_text") or "").strip()),
            "error_rows": sum(1 for row in eval_a if (row.get("error_message") or "").strip()),
            "strict_gold_correct": sum(1 for row in eval_a if row.get("strict_gold_correct") == "1"),
        },
        {
            "split": "B_B_direct",
            "rows": len(eval_b),
            "empty_generated": sum(1 for row in eval_b if not (row.get("generated_text") or "").strip()),
            "error_rows": sum(1 for row in eval_b if (row.get("error_message") or "").strip()),
            "strict_gold_correct": sum(1 for row in eval_b if row.get("strict_gold_correct") == "1"),
        },
    ]

    node_summary = summarize_by(nodes, ["run", "node_type"], ["path_mass_best", "depth_from_target"])
    intervention_summary = summarize_by(intervention, ["run"], ["delta_target_logit", "delta_target_prob"])
    compare_summary = [
        {
            "metric_group": "sample_compare",
            "n": len(sample_compare),
            "node_overlap_jaccard_mean": numeric_mean(sample_compare, "node_overlap_jaccard"),
            "edge_overlap_jaccard_mean": numeric_mean(sample_compare, "edge_overlap_jaccard"),
            "delta_target_total_in_abs_mean": numeric_mean(sample_compare, "delta_target_total_in_abs"),
            "a_traced_nodes_mean": numeric_mean(sample_compare, "a_traced_nodes"),
            "b_traced_nodes_mean": numeric_mean(sample_compare, "b_traced_nodes"),
            "a_traced_edges_mean": numeric_mean(sample_compare, "a_traced_edges"),
            "b_traced_edges_mean": numeric_mean(sample_compare, "b_traced_edges"),
        }
    ]

    failure_reason_counts = Counter()
    for row in failures:
        failure_reason_counts[row.get("failure_a", "") or "a_ok"] += 1
        failure_reason_counts[row.get("failure_b", "") or "b_ok"] += 1

    if args.mode == "full" and valid_count >= 48 and graph_success_rate >= 0.8 and sample_compare and nodes and intervention:
        status = f"{args.pack}_full_passed"
    elif args.mode == "full" and valid_count >= 48 and graph_success_rate >= 0.8 and sample_compare and nodes:
        status = f"{args.pack}_full_graph_compare_passed_intervention_blocked"
    elif valid_count >= 2 and sample_compare and nodes:
        status = f"{args.pack}_{args.mode}_partial_or_smoke_passed"
    else:
        status = f"{args.pack}_{args.mode}_blocked_or_insufficient"

    payload = {
        "status": status,
        "pack": args.pack,
        "mode": args.mode,
        "remote_decision_status": decision.get("status", ""),
        "counts": {
            "eval_a_rows": len(eval_a),
            "eval_b_rows": len(eval_b),
            "valid_samples": valid_count,
            "failure_rows": len(failures),
            "meta_a_rows": len(meta_a),
            "meta_b_rows": len(meta_b),
            "graph_a_sample_count": graph_a_count,
            "graph_b_sample_count": graph_b_count,
            "graph_success_rate_vs_valid": graph_success_rate,
            "sample_compare_rows": len(sample_compare),
            "bucket_summary_rows": len(bucket_summary),
            "node_rows": len(nodes),
            "edge_rows": len(edges),
            "intervention_rows": len(intervention),
        },
        "failure_reason_counts": dict(sorted(failure_reason_counts.items())),
        "compare_summary": compare_summary[0],
        "interpretation": (
            "This analyzer checks source-tracing pipeline completeness. It does not by itself establish "
            "region-mask causal specificity; region-mask/random/nearest analyses remain separate endpoints."
        ),
        "artifacts": {
            "summary_csv": str(CROSS / f"stage3_gemma_source_tracing_{suffix}_analysis_summary.csv"),
            "node_summary_csv": str(CROSS / f"stage3_gemma_source_tracing_{suffix}_node_summary.csv"),
            "intervention_summary_csv": str(CROSS / f"stage3_gemma_source_tracing_{suffix}_intervention_summary.csv"),
            "analysis_json": str(CROSS / f"stage3_gemma_source_tracing_{suffix}_analysis.json"),
        },
    }

    write_csv(
        CROSS / f"stage3_gemma_source_tracing_{suffix}_analysis_summary.csv",
        eval_summary_rows + compare_summary,
        [
            "split",
            "metric_group",
            "rows",
            "empty_generated",
            "error_rows",
            "strict_gold_correct",
            "n",
            "node_overlap_jaccard_mean",
            "edge_overlap_jaccard_mean",
            "delta_target_total_in_abs_mean",
            "a_traced_nodes_mean",
            "b_traced_nodes_mean",
            "a_traced_edges_mean",
            "b_traced_edges_mean",
        ],
    )
    write_csv(
        CROSS / f"stage3_gemma_source_tracing_{suffix}_node_summary.csv",
        node_summary,
        [
            "run",
            "node_type",
            "n",
            "path_mass_best_mean",
            "path_mass_best_positive_rate",
            "depth_from_target_mean",
            "depth_from_target_positive_rate",
        ],
    )
    write_csv(
        CROSS / f"stage3_gemma_source_tracing_{suffix}_intervention_summary.csv",
        intervention_summary,
        [
            "run",
            "n",
            "delta_target_logit_mean",
            "delta_target_logit_positive_rate",
            "delta_target_prob_mean",
            "delta_target_prob_positive_rate",
        ],
    )
    (CROSS / f"stage3_gemma_source_tracing_{suffix}_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
