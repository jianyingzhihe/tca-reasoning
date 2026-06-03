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
STAGE3_PAPERPACK = ROOT / "doc" / "experiments" / "stage3" / "paperpack72"
STAGE6_CROSS = ROOT / "doc" / "experiments" / "stage6" / "cross_model"
OUT_PREFIX = "stage6_node_generation_bridge"

GEMMA_CASE_ORDER = [
    ("primary", "okvqa_val_2100995", "B_direct", "wicker"),
    ("primary", "okvqa_val_2100995", "D_visual_only", "wicker"),
    ("primary", "okvqa_val_3955315", "D_visual_only", "buddhism"),
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


def _index(rows: list[dict[str, str]], *keys: str) -> dict[tuple[str, ...], dict[str, str]]:
    return {tuple(row.get(key, "") for key in keys): row for row in rows}


def _build_gemma_rows(mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case_panel = _read_csv(STAGE6_CROSS / "stage6_defensive_gemma_case_panel_full_defensive_v1.csv")
    panel_by_key = _index(case_panel, "pack", "sample_id", "prompt_name")
    runs_by_pack = {
        "primary": _index(_read_csv(STAGE3_PAPERPACK / "paperpack72_primary_prompt_runs.csv"), "sample_id", "prompt_name"),
        "strict": _index(_read_csv(STAGE3_PAPERPACK / "paperpack72_strict_sensitivity_prompt_runs.csv"), "sample_id", "prompt_name"),
    }
    wanted = GEMMA_CASE_ORDER[:1] if mode == "smoke" else GEMMA_CASE_ORDER
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for pack, sample_id, prompt_name, target_answer in wanted:
        run = runs_by_pack.get(pack, {}).get((sample_id, prompt_name))
        panel = panel_by_key.get((pack, sample_id, prompt_name))
        if run is None:
            missing.append({"model": "gemma3", "sample_id": sample_id, "prompt_name": prompt_name, "reason": "prompt_run_missing"})
            continue
        if panel is None:
            missing.append({"model": "gemma3", "sample_id": sample_id, "prompt_name": prompt_name, "reason": "case_panel_missing"})
        row = dict(run)
        row.update(
            {
                "model_family": "gemma3",
                "pack": pack,
                "target_answer": target_answer or run.get("answer_text", ""),
                "mask_condition": "answer_mask",
                "layer": "1",
                "position_group": "top_hidden_delta_32",
                "bridge_operator": "hidden_residual_generation_bridge",
                "selection_source": "stage6_019_case_panel",
                "hidden_effect": (panel or {}).get("hidden_effect", ""),
                "plt_topk_effect": (panel or {}).get("plt_topk_effect", ""),
                "plt_error_effect": (panel or {}).get("plt_error_effect", ""),
            }
        )
        rows.append(row)
    return rows, missing


def _build_qwen_rows(mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = _read_csv(STAGE3_CROSS / "stage3_qwen_generation_bridge_v2_manifest.csv")
    if not source:
        return [], [{"model": "qwen2.5-vl", "reason": "stage3_qwen_generation_bridge_v2_manifest_missing"}]

    def score(row: dict[str, str]) -> tuple[int, float]:
        is_plt = 1 if row.get("asset_id") == "qwen2p5vl_plt" else 0
        try:
            return is_plt, float(row.get("selection_score", "0") or 0)
        except ValueError:
            return is_plt, 0.0

    rows = sorted(source, key=score, reverse=True)
    if mode == "smoke":
        rows = [next((row for row in rows if row.get("asset_id") == "qwen2p5vl_plt"), rows[0])]
    else:
        plt = [row for row in rows if row.get("asset_id") == "qwen2p5vl_plt"][:3]
        clt = [row for row in rows if row.get("asset_id") == "qwen2p5vl_clt"][:3]
        rows = plt + clt
    for row in rows:
        row["bridge_operator"] = "qwen_multifeature_sequence_bridge"
        row["selection_source"] = "stage3_qwen_generation_bridge_v2_manifest"
    return rows, []


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage6-022 node-to-generation bridge manifests.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--tag", default="defensive_v1")
    args = parser.parse_args()

    stem = f"{OUT_PREFIX}_{args.mode}_{args.tag}"
    gemma_rows, gemma_missing = _build_gemma_rows(args.mode)
    qwen_rows, qwen_missing = _build_qwen_rows(args.mode)
    missing = gemma_missing + qwen_missing

    gemma_fields = [
        "model_family",
        "pack",
        "sample_id",
        "prompt_name",
        "image_filename",
        "local_image_path",
        "question_text",
        "answer_text",
        "target_answer",
        "reasoning_operation",
        "image_dependence_tier",
        "paperpack_source",
        "mask_dir",
        "mask_condition",
        "layer",
        "position_group",
        "bridge_operator",
        "selection_source",
        "hidden_effect",
        "plt_topk_effect",
        "plt_error_effect",
    ]
    qwen_fields = sorted({key for row in qwen_rows for key in row}) if qwen_rows else ["asset_id"]
    _write_csv(STAGE6_CROSS / f"{stem}_gemma_manifest.csv", gemma_rows, gemma_fields)
    _write_csv(STAGE6_CROSS / f"{stem}_qwen_manifest.csv", qwen_rows, qwen_fields)

    status = "ready" if gemma_rows and qwen_rows else "blocked_missing_artifact"
    decision = {
        "created_at": _now(),
        "mode": args.mode,
        "tag": args.tag,
        "status": status,
        "gemma_rows": len(gemma_rows),
        "qwen_rows": len(qwen_rows),
        "missing": missing,
        "claim_boundary": (
            "Stage6-022 is a case-level node-to-generation bridge. "
            "Gemma rows use hidden-residual generation bridge unless a graph-route generation operator is added later."
        ),
    }
    _write_json(STAGE6_CROSS / f"{stem}_manifest_decision.json", decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
