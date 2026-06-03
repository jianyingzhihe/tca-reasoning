#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("STAGE6_DEFENSIVE_ROOT", r"E:\Bridging"))
STAGE3_CROSS = ROOT / "doc" / "experiments" / "stage3" / "cross_model"
STAGE6_CROSS = Path(
    os.environ.get(
        "STAGE6_DEFENSIVE_OUT_DIR",
        str(ROOT / "doc" / "experiments" / "stage6" / "cross_model"),
    )
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value not in ("", None) else default
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def pack_paths(pack: str) -> tuple[Path, Path]:
    stem = f"stage3_gemma_source_tracing_{pack}_full"
    return (
        STAGE3_CROSS / f"{stem}_edges_detailed_controlled.csv",
        STAGE3_CROSS / f"{stem}_sample_compare_controlled.csv",
    )


def route_rows_for_pack(pack: str, topks: list[int]) -> list[dict[str, Any]]:
    edges_path, _ = pack_paths(pack)
    edges = read_csv(edges_path)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in edges:
        sample = row.get("sample_id", "")
        run = row.get("run", "")
        if not sample or run not in {"A", "B"}:
            continue
        grouped.setdefault((sample, run), []).append(row)

    out: list[dict[str, Any]] = []
    for (sample, run), items in sorted(grouped.items()):
        items = sorted(items, key=lambda r: f(r.get("path_mass")), reverse=True)
        total_path_mass = sum(f(r.get("path_mass")) for r in items)
        total_abs_weight = sum(f(r.get("abs_weight")) for r in items)
        total_signed_path_mass = sum((1.0 if f(r.get("weight")) >= 0 else -1.0) * f(r.get("path_mass")) for r in items)

        for topk in topks:
            part = items[:topk]
            if not part:
                continue
            path_mass = sum(f(r.get("path_mass")) for r in part)
            abs_weight = sum(f(r.get("abs_weight")) for r in part)
            signed_path_mass = sum((1.0 if f(r.get("weight")) >= 0 else -1.0) * f(r.get("path_mass")) for r in part)
            positive_path_mass = sum(f(r.get("path_mass")) for r in part if f(r.get("weight")) > 0)
            negative_path_mass = sum(f(r.get("path_mass")) for r in part if f(r.get("weight")) < 0)
            nodes = {r.get("src_node", "") for r in part} | {r.get("dst_node", "") for r in part}
            layers = [int(f(r.get("src_layer"))) for r in part] + [int(f(r.get("dst_layer"))) for r in part]
            top1_mass = f(part[0].get("path_mass"))
            out.append(
                {
                    "pack": pack,
                    "sample_id": sample,
                    "run": run,
                    "route_base_id": f"{sample}::{run}",
                    "topk": topk,
                    "edge_count": len(part),
                    "node_count": len([node for node in nodes if node]),
                    "path_mass_sum": path_mass,
                    "path_mass_retention": path_mass / total_path_mass if total_path_mass else 0.0,
                    "abs_weight_retention": abs_weight / total_abs_weight if total_abs_weight else 0.0,
                    "positive_path_mass_frac": positive_path_mass / path_mass if path_mass else 0.0,
                    "negative_path_mass_frac": negative_path_mass / path_mass if path_mass else 0.0,
                    "signed_path_mass_balance": abs(signed_path_mass) / path_mass if path_mass else 0.0,
                    "signed_path_mass_retention": signed_path_mass / total_signed_path_mass if total_signed_path_mass else 0.0,
                    "leave_one_top_edge_mass_frac": top1_mass / path_mass if path_mass else 0.0,
                    "layer_min": min(layers) if layers else "",
                    "layer_max": max(layers) if layers else "",
                    "layer_span": (max(layers) - min(layers)) if layers else "",
                }
            )
    return out


def summarize_topk(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["pack"], row["run"], int(row["topk"])), []).append(row)

    out: list[dict[str, Any]] = []
    for (pack, run, topk), part in sorted(grouped.items()):
        out.append(
            {
                "pack": pack,
                "run": run,
                "topk": topk,
                "n": len(part),
                "path_mass_retention_mean": mean([f(r["path_mass_retention"]) for r in part]),
                "positive_path_mass_frac_mean": mean([f(r["positive_path_mass_frac"]) for r in part]),
                "signed_path_mass_balance_mean": mean([f(r["signed_path_mass_balance"]) for r in part]),
                "leave_one_top_edge_mass_frac_mean": mean([f(r["leave_one_top_edge_mass_frac"]) for r in part]),
                "layer_span_mean": mean([f(r["layer_span"]) for r in part]),
            }
        )
    return out


def summarize_stability(tag: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_route: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_route.setdefault(f"{row['pack']}::{row['route_base_id']}", []).append(row)

    out: list[dict[str, Any]] = []
    for route, part in sorted(by_route.items()):
        by_topk = {int(row["topk"]): row for row in part}
        largest_k = max(by_topk)
        largest = by_topk[largest_k]
        best_balance = max(f(row["signed_path_mass_balance"]) for row in part)
        best_support = max(f(row["positive_path_mass_frac"]) for row in part)
        out.append(
            {
                "tag": tag,
                "route_base_id": route,
                "largest_topk": largest_k,
                "largest_positive_path_mass_frac": largest["positive_path_mass_frac"],
                "largest_signed_path_mass_balance": largest["signed_path_mass_balance"],
                "largest_leave_one_top_edge_mass_frac": largest["leave_one_top_edge_mass_frac"],
                "support_frac_drop_to_largest": best_support - f(largest["positive_path_mass_frac"]),
                "balance_drop_to_largest": best_balance - f(largest["signed_path_mass_balance"]),
                "is_composition_mixed_at_largest": int(0.25 < f(largest["positive_path_mass_frac"]) < 0.75),
                "is_top_edge_dominated_at_largest": int(f(largest["leave_one_top_edge_mass_frac"]) >= 0.5),
            }
        )
    return out


def compare_summary() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pack in ["primary", "strict"]:
        _, compare_path = pack_paths(pack)
        rows = read_csv(compare_path)
        out.append(
            {
                "pack": pack,
                "n": len(rows),
                "node_overlap_jaccard_mean": mean([f(r.get("node_overlap_jaccard")) for r in rows]),
                "edge_overlap_jaccard_mean": mean([f(r.get("edge_overlap_jaccard")) for r in rows]),
                "a_traced_nodes_mean": mean([f(r.get("a_traced_nodes")) for r in rows]),
                "b_traced_nodes_mean": mean([f(r.get("b_traced_nodes")) for r in rows]),
                "a_traced_edges_mean": mean([f(r.get("a_traced_edges")) for r in rows]),
                "b_traced_edges_mean": mean([f(r.get("b_traced_edges")) for r in rows]),
                "a_total_path_mass_mean": mean([f(r.get("a_traced_total_path_mass")) for r in rows]),
                "b_total_path_mass_mean": mean([f(r.get("b_traced_total_path_mass")) for r in rows]),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Gemma grouped source-tracing route composition from existing artifacts.")
    parser.add_argument("--tag", default="defensive_v1")
    parser.add_argument("--topks", default="1,2,4,8,16,32")
    args = parser.parse_args()

    topks = [int(x) for x in args.topks.split(",") if x.strip()]
    route_rows = route_rows_for_pack("primary", topks) + route_rows_for_pack("strict", topks)
    topk_rows = summarize_topk(route_rows)
    stability_rows = summarize_stability(args.tag, route_rows)
    compare_rows = compare_summary()

    stem = f"stage6_defensive_gemma_grouped_composition_full_{args.tag}"
    write_csv(
        STAGE6_CROSS / f"{stem}_route_rows.csv",
        route_rows,
        [
            "pack",
            "sample_id",
            "run",
            "route_base_id",
            "topk",
            "edge_count",
            "node_count",
            "path_mass_sum",
            "path_mass_retention",
            "abs_weight_retention",
            "positive_path_mass_frac",
            "negative_path_mass_frac",
            "signed_path_mass_balance",
            "signed_path_mass_retention",
            "leave_one_top_edge_mass_frac",
            "layer_min",
            "layer_max",
            "layer_span",
        ],
    )
    write_csv(
        STAGE6_CROSS / f"{stem}_topk_summary.csv",
        topk_rows,
        [
            "pack",
            "run",
            "topk",
            "n",
            "path_mass_retention_mean",
            "positive_path_mass_frac_mean",
            "signed_path_mass_balance_mean",
            "leave_one_top_edge_mass_frac_mean",
            "layer_span_mean",
        ],
    )
    write_csv(
        STAGE6_CROSS / f"{stem}_route_stability.csv",
        stability_rows,
        [
            "tag",
            "route_base_id",
            "largest_topk",
            "largest_positive_path_mass_frac",
            "largest_signed_path_mass_balance",
            "largest_leave_one_top_edge_mass_frac",
            "support_frac_drop_to_largest",
            "balance_drop_to_largest",
            "is_composition_mixed_at_largest",
            "is_top_edge_dominated_at_largest",
        ],
    )
    write_csv(
        STAGE6_CROSS / f"{stem}_compare_summary.csv",
        compare_rows,
        [
            "pack",
            "n",
            "node_overlap_jaccard_mean",
            "edge_overlap_jaccard_mean",
            "a_traced_nodes_mean",
            "b_traced_nodes_mean",
            "a_traced_edges_mean",
            "b_traced_edges_mean",
            "a_total_path_mass_mean",
            "b_total_path_mass_mean",
        ],
    )

    mixed_frac = mean([f(r["is_composition_mixed_at_largest"]) for r in stability_rows])
    top_edge_frac = mean([f(r["is_top_edge_dominated_at_largest"]) for r in stability_rows])
    status = "gemma_grouped_composition_artifact_ready" if route_rows and topk_rows else "blocked_missing_artifact"
    decision = {
        "tag": args.tag,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "route_rows": len(route_rows),
        "topk_summary_rows": len(topk_rows),
        "route_stability_rows": len(stability_rows),
        "compare_summary_rows": len(compare_rows),
        "mixed_composition_frac_at_largest_topk": mixed_frac,
        "top_edge_dominated_frac_at_largest_topk": top_edge_frac,
        "interpretation_boundary": (
            "Artifact-level Gemma source-tracing composition diagnostic. "
            "It mirrors Qwen grouped-composition analysis at the route-bundle stability question, "
            "but does not rerun new graph interventions."
        ),
    }
    write_json(STAGE6_CROSS / f"{stem}_decision.json", decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
