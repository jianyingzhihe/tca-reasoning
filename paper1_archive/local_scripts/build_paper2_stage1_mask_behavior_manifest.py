#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw


ROOT = Path(r"E:\Bridging")
PAPER2_ROOT = ROOT / "paper2_compositional_evidence_routing"
LOCAL_RESULTS = PAPER2_ROOT / "doc" / "experiments" / "stage1" / "results"
SOURCE_MANIFEST = ROOT / "doc" / "experiments" / "paper2" / "stage1" / "stage1_composition_screen_viability_v1_manifest.csv"
PROMPT_RUNS = ROOT / "doc" / "experiments" / "paper2" / "stage2" / "paper2_stage2_hidden_union_strong_prompt_runs.csv"
ASSET_ROOT = PAPER2_ROOT / "annotation" / "stage1_mask_behavior_assets_v1"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _labelme_candidates(stem: str) -> list[Path]:
    return [
        ROOT / "annotation" / "stage3_paperpack72_labelme" / "annotator_a" / "images" / f"{stem}.json",
        ROOT / "annotation" / "stage3_paperpack72_labelme" / "single_pass1_primary72" / "images" / f"{stem}.json",
        ROOT / "annotation" / "stage3_paperpack72_labelme" / "single_pass1_localized" / "images" / f"{stem}.json",
        ROOT / "annotation" / "okvqa_evidence_labelme_round3_core" / "images" / f"{stem}.json",
    ]


def _shape_mask(payload: dict, image_size: tuple[int, int], labels: set[str]) -> Image.Image:
    width, height = image_size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for shape in payload.get("shapes", []):
        label = str(shape.get("label", "")).strip().lower()
        if label not in labels:
            continue
        points = shape.get("points") or []
        if len(points) < 2:
            continue
        xy = [(float(x), float(y)) for x, y in points]
        if shape.get("shape_type", "polygon") == "rectangle" and len(xy) >= 2:
            x0, y0 = xy[0]
            x1, y1 = xy[1]
            draw.rectangle([x0, y0, x1, y1], fill=255)
        else:
            draw.polygon(xy, fill=255)
    return mask


def _save_masks(stem: str, image_path: Path, out_dir: Path) -> bool:
    json_path = next((p for p in _labelme_candidates(stem) if p.exists()), None)
    if json_path is None:
        return False
    image = Image.open(image_path).convert("RGB")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    answer = _shape_mask(payload, image.size, {"answer", "region_a", "a"})
    relate = _shape_mask(payload, image.size, {"relate", "related", "region_b", "b"})
    if answer.getbbox() is None:
        return False
    if relate.getbbox() is None:
        relate = answer.copy()
    union = ImageChops.lighter(answer, relate)
    width, height = image.size
    out_dir.mkdir(parents=True, exist_ok=True)
    answer.save(out_dir / "region_A.png")
    relate.save(out_dir / "region_B.png")
    union.save(out_dir / "region_A_union_B.png")
    ImageChops.offset(answer, max(1, width // 3), 0).save(out_dir / "random_A_size.png")
    ImageChops.offset(relate, 0, max(1, height // 3)).save(out_dir / "random_B_size.png")
    ImageChops.offset(union, max(1, width // 4), max(1, height // 4)).save(out_dir / "random_union_size.png")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Paper2 Stage1 mask behavior manifest.")
    parser.add_argument("--selection", choices=["hidden-union-strong", "all-usable"], default="hidden-union-strong")
    parser.add_argument("--tag", default="mask_v1")
    args = parser.parse_args()

    source_rows = _read_csv(SOURCE_MANIFEST)
    source = {row["sample_id"]: row for row in source_rows}
    if args.selection == "hidden-union-strong":
        selected = [row["sample_id"] for row in _read_csv(PROMPT_RUNS)]
    else:
        selected = []
        for row in source_rows:
            stem = Path(row["image_filename"]).stem
            if any(p.exists() for p in _labelme_candidates(stem)):
                selected.append(row["sample_id"])
    rows: list[dict[str, str]] = []
    for sample_id in selected:
        row = dict(source[sample_id])
        image_name = Path(row["image_filename"]).name
        stem = Path(image_name).stem
        image_src = Path(row["local_image_path"])
        image_dst = ASSET_ROOT / "images" / image_name
        image_dst.parent.mkdir(parents=True, exist_ok=True)
        if image_src.exists():
            shutil.copy2(image_src, image_dst)
        wrong_src = Path(row.get("wrong_image_path", ""))
        if wrong_src.exists():
            wrong_dst = ASSET_ROOT / "images" / wrong_src.name
            shutil.copy2(wrong_src, wrong_dst)
            row["wrong_image_path"] = str(wrong_dst)
        out_mask_dir = ASSET_ROOT / "exported_masks" / stem
        if not _save_masks(stem, image_dst, out_mask_dir):
            raise RuntimeError(f"could not render masks for {sample_id} {stem}")
        row["local_image_path"] = str(image_dst)
        row["mask_dir"] = str(out_mask_dir)
        row["region_A_mask_path"] = str(out_mask_dir / "region_A.png")
        row["region_B_mask_path"] = str(out_mask_dir / "region_B.png")
        row["region_A_union_B_mask_path"] = str(out_mask_dir / "region_A_union_B.png")
        row["random_A_size_mask_path"] = str(out_mask_dir / "random_A_size.png")
        row["random_B_size_mask_path"] = str(out_mask_dir / "random_B_size.png")
        row["random_union_size_mask_path"] = str(out_mask_dir / "random_union_size.png")
        rows.append(row)

    fields = list(_read_csv(SOURCE_MANIFEST)[0].keys())
    out = LOCAL_RESULTS / f"stage1_behavior_manifest_{args.tag}.csv"
    _write_csv(out, rows, fields)
    print(json.dumps({"status": "ready", "selection": args.selection, "tag": args.tag, "rows": len(rows), "manifest": str(out)}, indent=2))


if __name__ == "__main__":
    main()
