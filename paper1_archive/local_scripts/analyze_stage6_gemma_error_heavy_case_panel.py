#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
STAGE3_CROSS = ROOT / "doc" / "experiments" / "stage3" / "cross_model"
STAGE6_CROSS = ROOT / "doc" / "experiments" / "stage6" / "cross_model"


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


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value not in ("", None) else default
    except (TypeError, ValueError):
        return default


def _best_by_operator(raw: list[dict[str, str]], pack: str) -> dict[tuple[str, str], dict[str, dict[str, str]]]:
    rows: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in raw:
        if row.get("token_scored") != "target":
            continue
        if row.get("operator") not in {"hidden_residual", "plt_topk_reconstruction", "plt_reconstruction_error"}:
            continue
        if row.get("position_group") != "visual+answer":
            continue
        if row.get("mask_condition") != "union_mask":
            continue
        key = (row.get("sample_id", ""), row.get("prompt_name", ""))
        op = row.get("operator", "")
        current = rows.setdefault(key, {}).get(op)
        if current is None:
            rows[key][op] = row
            continue
        # Prefer K=32 for PLT operators because the Stage6-017 gate reports K=32.
        if op.startswith("plt") and row.get("top_k") == "32":
            rows[key][op] = row
        elif op == "hidden_residual" and row.get("top_k") == "full":
            rows[key][op] = row
    return rows


def _source_by_sample(pack: str) -> dict[str, dict[str, str]]:
    path = STAGE3_CROSS / f"stage3_gemma_source_tracing_{pack}_full_sample_compare_controlled.csv"
    rows = _read_csv(path)
    return {row.get("sample_id", ""): row for row in rows if row.get("sample_id")}


def build_case_panel(mode: str, tag: str, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows = []
    for pack in ["primary", "strict"]:
        path = STAGE6_CROSS / f"stage6_gemma_hidden_to_plt_decomp_{pack}_{mode}_decomp_v1_raw.csv"
        for row in _read_csv(path):
            row["_pack"] = pack
            raw_rows.append(row)

    rows: list[dict[str, Any]] = []
    for pack in ["primary", "strict"]:
        pack_rows = [row for row in raw_rows if row.get("_pack") == pack]
        by_op = _best_by_operator(pack_rows, pack)
        source_rows = _source_by_sample(pack)
        for (sample_id, prompt_name), ops in by_op.items():
            hidden = ops.get("hidden_residual")
            topk = ops.get("plt_topk_reconstruction")
            error = ops.get("plt_reconstruction_error")
            if not hidden or not topk or not error:
                continue
            source = source_rows.get(sample_id, {})
            hidden_effect = _f(hidden.get("logit_effect"))
            error_effect = _f(error.get("logit_effect"))
            topk_effect = _f(topk.get("logit_effect"))
            rows.append(
                {
                    "pack": pack,
                    "sample_id": sample_id,
                    "prompt_name": prompt_name,
                    "target_answer": hidden.get("target_answer", ""),
                    "hidden_effect": hidden_effect,
                    "plt_topk_k": topk.get("top_k", ""),
                    "plt_topk_effect": topk_effect,
                    "plt_error_k": error.get("top_k", ""),
                    "plt_error_effect": error_effect,
                    "error_over_hidden": error_effect / hidden_effect if hidden_effect else "",
                    "topk_over_hidden": topk_effect / hidden_effect if hidden_effect else "",
                    "source_node_overlap_jaccard": source.get("node_overlap_jaccard", ""),
                    "source_edge_overlap_jaccard": source.get("edge_overlap_jaccard", ""),
                    "source_delta_target_total_in_abs": source.get("delta_target_total_in_abs", ""),
                    "interpretation": "error_heavy_but_source_route_artifact_available",
                }
            )

    rows.sort(key=lambda row: (_f(row["hidden_effect"]) + _f(row["plt_error_effect"])) / 2, reverse=True)
    selected = rows[:limit]
    status = "case_panel_ready" if selected else "blocked_missing_case_alignment"
    decision = {
        "tag": tag,
        "mode": mode,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "raw_candidate_rows": len(rows),
        "selected_rows": len(selected),
        "interpretation": (
            "Gemma local PLT topK reconstruction can be weak while reconstruction error "
            "retains hidden effect and source-tracing graph artifacts remain available."
        ),
    }
    return selected, decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="full")
    parser.add_argument("--tag", default="defensive_v1")
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()

    rows, decision = build_case_panel(args.mode, args.tag, args.limit)
    stem = f"stage6_defensive_gemma_case_panel_{args.mode}_{args.tag}"
    fields = [
        "pack",
        "sample_id",
        "prompt_name",
        "target_answer",
        "hidden_effect",
        "plt_topk_k",
        "plt_topk_effect",
        "plt_error_k",
        "plt_error_effect",
        "error_over_hidden",
        "topk_over_hidden",
        "source_node_overlap_jaccard",
        "source_edge_overlap_jaccard",
        "source_delta_target_total_in_abs",
        "interpretation",
    ]
    _write_csv(STAGE6_CROSS / f"{stem}.csv", rows, fields)
    _write_json(STAGE6_CROSS / f"{stem}_decision.json", decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
