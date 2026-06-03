#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
PAPER2 = ROOT / "doc" / "experiments" / "paper2"
STAGE1 = PAPER2 / "stage1"
STAGE2 = PAPER2 / "stage2"
STAGE3 = PAPER2 / "stage3_mechanism"
ASSETS = ROOT / "paper2_compositional_evidence_routing" / "annotation" / "stage3_mechanism_assets_v1"
SOURCE_RUNS = STAGE2 / "paper2_stage2_hidden_union_strong_prompt_runs.csv"
LABELS = STAGE1 / "composition_effect_by_sample_mask_full_v1.csv"


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
            writer.writerow({key: row.get(key, "") for key in fields})


def _copy(src: Path, dst: Path) -> None:
    if not src.exists() or src.stat().st_size == 0:
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _label_index() -> dict[tuple[str, str], str]:
    return {(row["model_family"], row["sample_id"]): row["stage1_label"] for row in _read_csv(LABELS)}


def _make_prompt_rows(model: str, sample_ids: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    source = {row["sample_id"]: row for row in _read_csv(SOURCE_RUNS)}
    labels = _label_index()
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for sample_id in sample_ids:
        row = source.get(sample_id)
        if row is None:
            missing.append(f"{sample_id}:missing_source_prompt_run")
            continue
        if labels.get((model, sample_id)) != "multi_evidence_supported":
            missing.append(f"{sample_id}:not_multi_evidence_supported_for_{model}")
            continue
        image_name = Path(row["image_filename"]).name
        stem = Path(image_name).stem
        src_image = Path(row["local_image_path"])
        src_mask_dir = Path(row["mask_dir"])
        out_image = ASSETS / "images" / image_name
        out_mask_dir = ASSETS / "exported_masks" / stem
        try:
            _copy(src_image, out_image)
            for name in ["answer.png", "union.png", "shifted.png", "shuffled.png"]:
                _copy(src_mask_dir / name, out_mask_dir / name)
        except FileNotFoundError as exc:
            missing.append(f"{sample_id}:missing_asset:{exc}")
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "prompt_name": "B_direct",
                "image_filename": image_name,
                "local_image_path": str(out_image),
                "question_text": row["question_text"],
                "answer_text": row["answer_text"],
                "reasoning_operation": row.get("reasoning_operation", "paper2_composition"),
                "image_dependence_tier": row.get("image_dependence_tier", "paper2"),
                "paperpack_source": "paper2_stage3_multi_evidence_mechanism",
                "mask_dir": str(out_mask_dir),
            }
        )
    return rows, missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Paper2 Stage3 mechanism prompt-runs from multi-evidence behavior cases.")
    parser.add_argument("--tag", default="paper2_mechanism_v1")
    args = parser.parse_args()

    qwen_ids = ["okvqa_val_00310", "okvqa_val_00893", "okvqa_val_258605"]
    gemma_ids = ["okvqa_val_1295955"]
    fields = [
        "sample_id",
        "prompt_name",
        "image_filename",
        "local_image_path",
        "question_text",
        "answer_text",
        "reasoning_operation",
        "image_dependence_tier",
        "paperpack_source",
        "mask_dir",
    ]
    qwen_rows, qwen_missing = _make_prompt_rows("qwen", qwen_ids)
    gemma_rows, gemma_missing = _make_prompt_rows("gemma", gemma_ids)
    all_rows = qwen_rows + [row for row in gemma_rows if row["sample_id"] not in {r["sample_id"] for r in qwen_rows}]

    _write_csv(STAGE3 / f"paper2_stage3_mechanism_qwen_prompt_runs_{args.tag}.csv", qwen_rows, fields)
    _write_csv(STAGE3 / f"paper2_stage3_mechanism_gemma_prompt_runs_{args.tag}.csv", gemma_rows, fields)
    _write_csv(STAGE3 / f"paper2_stage3_mechanism_all_prompt_runs_{args.tag}.csv", all_rows, fields)

    decision = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": args.tag,
        "status": "paper2_stage3_mechanism_pack_ready" if qwen_rows and gemma_rows else "blocked_missing_artifact",
        "qwen_rows": len(qwen_rows),
        "gemma_rows": len(gemma_rows),
        "qwen_samples": [row["sample_id"] for row in qwen_rows],
        "gemma_samples": [row["sample_id"] for row in gemma_rows],
        "missing": qwen_missing + gemma_missing,
        "asset_root": str(ASSETS),
        "mask_mapping": {
            "answer.png": "A/B union evidence mask from Paper2 Stage2 hidden assets",
            "union.png": "same union evidence mask",
            "shifted.png": "random/same-area control from Paper2 Stage2 hidden assets",
            "shuffled.png": "random/same-area control from Paper2 Stage2 hidden assets",
        },
    }
    (STAGE3 / f"paper2_stage3_mechanism_pack_{args.tag}_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0 if decision["status"] == "paper2_stage3_mechanism_pack_ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
