#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
STAGE2_CROSS = ROOT / "doc" / "experiments" / "stage2" / "cross_model"
STAGE3_CROSS = ROOT / "doc" / "experiments" / "stage3" / "cross_model"
SOURCE_MANIFEST = STAGE2_CROSS / "stage2n_all52_manifest.csv"

DEFAULT_QUOTAS = {
    "symbol_text_reading": 12,
    "visual_readout": 8,
    "scene_inference": 4,
}

PROMPTS = ("B_direct", "D_visual_only")

ASSETS = [
    {
        "asset_id": "gemma3_plt",
        "model_axis": "Gemma3",
        "asset_axis": "PLT",
        "base_model": "google/gemma-3-4b-it",
        "transcoder_ref": "tianhux2/gemma3-4b-it-plt",
        "stage3_role": "plt_baseline",
        "default_layer": "",
        "risk_level": "low",
        "notes": "Current full-chain Gemma baseline.",
    },
    {
        "asset_id": "qwen2p5vl_plt",
        "model_axis": "Qwen2.5-VL",
        "asset_axis": "PLT",
        "base_model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "transcoder_ref": "KokosDev/qwen2p5vl-7b-plt",
        "stage3_role": "plt_main_cross_model",
        "default_layer": "26",
        "risk_level": "medium",
        "notes": "Primary PLT-aligned cross-model target.",
    },
    {
        "asset_id": "qwen35_plt",
        "model_axis": "Qwen35",
        "asset_axis": "PLT",
        "base_model": "Qwen/Qwen3.5-4B",
        "transcoder_ref": "KokosDev/qwen35-4b-plt",
        "stage3_role": "plt_high_risk_third_model",
        "default_layer": "",
        "risk_level": "high",
        "notes": "Missing L1 and VLM capability must be checked before mainline.",
    },
    {
        "asset_id": "qwen2p5vl_clt",
        "model_axis": "Qwen2.5-VL",
        "asset_axis": "CLT",
        "base_model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "transcoder_ref": "KokosDev/qwen2p5vl-7b-clt",
        "stage3_role": "clt_auxiliary_positive_or_heterogeneity",
        "default_layer": "26",
        "risk_level": "medium",
        "notes": "Stage2 positive auxiliary line; compare against Qwen PLT.",
    },
    {
        "asset_id": "llava15_clt",
        "model_axis": "LLaVA",
        "asset_axis": "CLT",
        "base_model": "llava-hf/llava-1.5-7b-hf",
        "transcoder_ref": "KokosDev/llava15-7b-clt",
        "stage3_role": "clt_appendix_diagnostic",
        "default_layer": "15",
        "risk_level": "high",
        "notes": "No public PLT found; keep as CLT diagnostic only.",
    },
]


def _read_csv(path: Path) -> list[dict[str, str]]:
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


def _float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("-inf")


def _eligible(row: dict[str, str]) -> bool:
    if row.get("image_dependence") != "strong":
        return False
    if row.get("eligibility") not in {"primary_eligible", "secondary_large_localized"}:
        return False
    if not row.get("sample_id") or not row.get("question_text") or not row.get("answer_text"):
        return False
    if not Path(row.get("local_image_path", "")).exists():
        return False
    mask_dir = Path(row.get("mask_dir", ""))
    if not (mask_dir / "answer.png").exists():
        return False
    if not (mask_dir / "relate.png").exists():
        return False
    return True


def _dedupe_best(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["sample_id"]].append(row)
    best = [
        sorted(items, key=lambda item: _float(item.get("selection_score", "")), reverse=True)[0]
        for items in grouped.values()
    ]
    return sorted(best, key=lambda item: _float(item.get("selection_score", "")), reverse=True)


def _parse_quotas(raw: str) -> dict[str, int]:
    if not raw:
        return dict(DEFAULT_QUOTAS)
    out: dict[str, int] = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        key, value = part.split("=", 1)
        out[key.strip()] = int(value.strip())
    return out


def _select(rows: list[dict[str, str]], quotas: dict[str, int], target_count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_type[row.get("reasoning_operation", "")].append(row)

    deficits: dict[str, int] = {}
    for reasoning_type, quota in quotas.items():
        candidates = by_type.get(reasoning_type, [])
        take = min(quota, len(candidates), target_count - len(selected))
        for row in candidates[:take]:
            selected.append(dict(row))
            selected_ids.add(row["sample_id"])
        if take < quota:
            deficits[reasoning_type] = quota - take

    if len(selected) < target_count:
        for row in rows:
            if row["sample_id"] in selected_ids:
                continue
            selected.append(dict(row))
            selected_ids.add(row["sample_id"])
            if len(selected) >= target_count:
                break

    for rank, row in enumerate(selected, start=1):
        row["stage3_selection_rank"] = rank
        row["stage3_selection_note"] = "quota_selected"

    counts = Counter(row.get("reasoning_operation", "") for row in selected)
    summary = {
        "source_manifest": str(SOURCE_MANIFEST),
        "target_count": target_count,
        "selected_count": len(selected),
        "available_eligible_count": len(rows),
        "type_quotas": quotas,
        "selected_type_counts": dict(counts),
        "quota_deficits": deficits,
        "prompt_names": list(PROMPTS),
        "prompt_run_count": len(selected) * len(PROMPTS),
        "usable_status": "pass_min20" if len(selected) >= 20 else "blocked_need_more_annotations",
        "selected_samples": [row["sample_id"] for row in selected],
        "claim_boundary": "Manifest construction only; it does not establish cross-model evidence.",
    }
    return selected, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage3 dual-track cross-model manifests.")
    parser.add_argument("--source", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=STAGE3_CROSS)
    parser.add_argument("--target-count", type=int, default=24)
    parser.add_argument("--quotas", default="")
    args = parser.parse_args()

    quotas = _parse_quotas(args.quotas)
    raw_rows = _read_csv(args.source)
    eligible = _dedupe_best([row for row in raw_rows if _eligible(row)])
    selected, summary = _select(eligible, quotas, args.target_count)

    selected_fields = [
        "stage3_selection_rank",
        "sample_id",
        "image_filename",
        "question_text",
        "answer_text",
        "reasoning_operation",
        "visual_structure",
        "image_dependence",
        "mask_pack",
        "mask_dir",
        "local_image_path",
        "width",
        "height",
        "answer_area_px",
        "relate_area_px",
        "union_area_px",
        "answer_area_frac",
        "relate_area_frac",
        "union_area_frac",
        "compactness_label",
        "eligibility",
        "selection_score",
        "metadata_source",
        "annotation_priority",
        "stage3_selection_note",
    ]
    prompt_rows: list[dict[str, Any]] = []
    for row in selected:
        for prompt_name in PROMPTS:
            prompt_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "prompt_name": prompt_name,
                    "image_filename": row["image_filename"],
                    "question_text": row["question_text"],
                    "answer_text": row["answer_text"],
                    "reasoning_operation": row["reasoning_operation"],
                    "mask_dir": row["mask_dir"],
                    "local_image_path": row["local_image_path"],
                    "stage3_selection_rank": row["stage3_selection_rank"],
                }
            )

    _write_csv(args.out_dir / "stage3_aligned24_manifest.csv", selected, selected_fields)
    _write_csv(
        args.out_dir / "stage3_aligned48_prompt_runs.csv",
        prompt_rows,
        [
            "sample_id",
            "prompt_name",
            "image_filename",
            "question_text",
            "answer_text",
            "reasoning_operation",
            "mask_dir",
            "local_image_path",
            "stage3_selection_rank",
        ],
    )
    _write_csv(
        args.out_dir / "stage3_asset_table.csv",
        ASSETS,
        [
            "asset_id",
            "model_axis",
            "asset_axis",
            "base_model",
            "transcoder_ref",
            "stage3_role",
            "default_layer",
            "risk_level",
            "notes",
        ],
    )
    _write_json(args.out_dir / "stage3_manifest_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

