#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw


ROOT = Path(r"E:\Bridging")
STAGE1 = ROOT / "doc" / "experiments" / "paper2" / "stage1"
STAGE2 = ROOT / "doc" / "experiments" / "paper2" / "stage2"
ASSETS = ROOT / "paper2_compositional_evidence_routing" / "annotation" / "stage2_hidden_lens_assets_v1"
MANIFEST = STAGE1 / "stage1_composition_screen_viability_v1_manifest.csv"
STRONG = STAGE1 / "paper2_stage1_behavior_viability_v1_strong_cases.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _copy_mask(src: Path, dst: Path) -> None:
    if not src.exists() or src.stat().st_size == 0:
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _labelme_candidates(stem: str) -> list[Path]:
    return [
        ROOT / "annotation" / "stage3_paperpack72_labelme" / "annotator_a" / "images" / f"{stem}.json",
        ROOT / "annotation" / "stage3_paperpack72_labelme" / "single_pass1_primary72" / "images" / f"{stem}.json",
        ROOT / "annotation" / "stage3_paperpack72_labelme" / "single_pass1_localized" / "images" / f"{stem}.json",
        ROOT / "annotation" / "okvqa_evidence_labelme_round3_core" / "images" / f"{stem}.json",
    ]


def _render_labelme_union(stem: str, image_path: Path, out_mask_dir: Path) -> bool:
    json_path = next((p for p in _labelme_candidates(stem) if p.exists()), None)
    if json_path is None:
        return False
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for shape in payload.get("shapes", []):
        points = shape.get("points") or []
        if len(points) < 2:
            continue
        shape_type = shape.get("shape_type", "polygon")
        xy = [(float(x), float(y)) for x, y in points]
        if shape_type == "rectangle" and len(xy) >= 2:
            x0, y0 = xy[0]
            x1, y1 = xy[1]
            draw.rectangle([x0, y0, x1, y1], fill=255)
        else:
            draw.polygon(xy, fill=255)
    if mask.getbbox() is None:
        return False
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    mask.save(out_mask_dir / "answer.png")
    mask.save(out_mask_dir / "union.png")
    ImageChops.offset(mask, max(1, width // 3), 0).save(out_mask_dir / "shifted.png")
    ImageChops.offset(mask, 0, max(1, height // 3)).save(out_mask_dir / "shuffled.png")
    return True


def _usable(row: dict[str, str]) -> bool:
    mask_dir = Path(row["mask_dir"])
    stem = Path(row["image_filename"]).stem
    has_png_masks = (
        (mask_dir / "region_A_union_B.png").exists()
        and (mask_dir / "region_A_union_B.png").stat().st_size > 0
        and (mask_dir / "random_union_size.png").exists()
        and (mask_dir / "random_union_size.png").stat().st_size > 0
    )
    has_labelme_json = any(p.exists() for p in _labelme_candidates(stem))
    return has_png_masks or has_labelme_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Paper2 Stage2 hidden-lens prompt-runs from Stage1 strong cases.")
    parser.add_argument("--selection", choices=["common", "union"], default="common")
    parser.add_argument("--max-per-model", type=int, default=4)
    parser.add_argument("--out-name", default="paper2_stage2_hidden_common_strong_prompt_runs.csv")
    args = parser.parse_args()

    manifest = {row["sample_id"]: row for row in _read_csv(MANIFEST)}
    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    strong_rows = _read_csv(STRONG)
    for row in strong_rows:
        by_sample[row["sample_id"]].append(row)

    selected: list[str]
    if args.selection == "common":
        common: list[tuple[float, str]] = []
        for sample_id, rows in by_sample.items():
            models = {row["model_family"] for row in rows}
            if {"gemma", "qwen"}.issubset(models) and sample_id in manifest and _usable(manifest[sample_id]):
                scores = [float(row.get("strong_case_score") or 0.0) for row in rows]
                common.append((sum(scores) / len(scores), sample_id))
        common.sort(reverse=True)
        selected = [sample_id for _score, sample_id in common[:2]]
        if len(selected) < 2:
            raise RuntimeError(f"expected at least 2 common strong cases, got {selected}")
    else:
        selected_set: set[str] = set()
        for model in ["gemma", "qwen"]:
            usable = [
                row
                for row in strong_rows
                if row["model_family"] == model
                and row["sample_id"] in manifest
                and _usable(manifest[row["sample_id"]])
            ]
            usable.sort(key=lambda row: float(row.get("strong_case_score") or 0.0), reverse=True)
            for row in usable[: args.max_per_model]:
                selected_set.add(row["sample_id"])
        selected = sorted(selected_set)
        if not selected:
            raise RuntimeError("no usable union strong cases")

    rows_out: list[dict[str, str]] = []
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
    for sample_id in selected:
        row = manifest[sample_id]
        image_filename = Path(row["image_filename"]).name
        stem = Path(image_filename).stem
        local_image = Path(row["local_image_path"])
        if not local_image.exists():
            local_image = ASSETS / "images" / image_filename
            local_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(row["original_image_path"]), local_image)

        src_mask_dir = Path(row["mask_dir"])
        out_mask_dir = ASSETS / "exported_masks" / stem
        try:
            _copy_mask(src_mask_dir / "region_A_union_B.png", out_mask_dir / "answer.png")
            _copy_mask(src_mask_dir / "region_A_union_B.png", out_mask_dir / "union.png")
            _copy_mask(src_mask_dir / "random_union_size.png", out_mask_dir / "shifted.png")
            _copy_mask(src_mask_dir / "random_union_size.png", out_mask_dir / "shuffled.png")
        except FileNotFoundError:
            if not _render_labelme_union(stem, local_image, out_mask_dir):
                raise

        rows_out.append(
            {
                "sample_id": sample_id,
                "prompt_name": "B_direct",
                "image_filename": image_filename,
                "local_image_path": str(local_image),
                "question_text": row["question_text"],
                "answer_text": row["answer_text"],
                "reasoning_operation": row.get("sample_type", "paper2_composition"),
                "image_dependence_tier": row.get("legacy_knowledge_level_label", "paper2"),
                "paperpack_source": "paper2_stage1_common_strong",
                "mask_dir": str(out_mask_dir),
            }
        )

    out_csv = STAGE2 / args.out_name
    _write_csv(out_csv, rows_out, fields)
    decision = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "paper2_stage2_hidden_prompt_runs_ready",
        "selection": args.selection,
        "selected_samples": selected,
        "rows": len(rows_out),
        "prompt_runs": str(out_csv),
        "asset_root": str(ASSETS),
        "mask_mapping": {
            "answer.png": "region_A_union_B.png",
            "union.png": "region_A_union_B.png",
            "shifted.png": "random_union_size.png",
            "shuffled.png": "random_union_size.png",
        },
    }
    (STAGE2 / "paper2_stage2_hidden_common_strong_prompt_runs_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
