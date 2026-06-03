#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter


ROOT = Path(r"E:\Bridging")
STAGE3_PAPERPACK = ROOT / "doc" / "experiments" / "stage3" / "paperpack72"
STAGE4_CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"
STAGE6_CROSS = ROOT / "doc" / "experiments" / "stage6" / "cross_model"
VARIANT_ROOT = ROOT / "annotation" / "stage6_defensive_mask_variants"

MASK_SUMMARY = STAGE3_PAPERPACK / "paperpack81_mask_export_summary.csv"
PROMPT_RUNS = STAGE3_PAPERPACK / "paperpack72_primary_prompt_runs.csv"
ROUTE_FIRST_MANIFEST = STAGE4_CROSS / "stage4_qwen_route_first_primary_manifest.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
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


def _f(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _load_selected_samples(limit: int) -> list[dict[str, str]]:
    mask_rows = {row["sample_id"]: row for row in _read_csv(MASK_SUMMARY) if row.get("sample_id")}
    prompt_rows = _read_csv(PROMPT_RUNS)
    manifest_rows = _read_csv(ROUTE_FIRST_MANIFEST)

    prompt_index: dict[str, set[str]] = {}
    for row in prompt_rows:
        prompt_index.setdefault(row.get("sample_id", ""), set()).add(row.get("prompt_name", ""))

    manifest_index: dict[str, set[str]] = {}
    for row in manifest_rows:
        if row.get("layer") == "14":
            manifest_index.setdefault(row.get("sample_id", ""), set()).add(row.get("prompt_name", ""))

    candidates: list[dict[str, str]] = []
    for sample_id, row in mask_rows.items():
        prompts = prompt_index.get(sample_id, set())
        manifest_prompts = manifest_index.get(sample_id, set())
        if not {"B_direct", "D_visual_only"}.issubset(prompts):
            continue
        if not {"B_direct", "D_visual_only"}.intersection(manifest_prompts):
            continue
        if row.get("mask_export_status") != "ok":
            continue
        union_frac = _f(row.get("union_area_frac"))
        if union_frac <= 0.0 or union_frac > 0.35:
            continue
        if _f(row.get("random16_valid_count")) < 8:
            continue
        candidates.append(row)

    candidates.sort(
        key=lambda row: (
            _f(row.get("union_area_frac")),
            -_f(row.get("random16_valid_count")),
            row.get("sample_id", ""),
        )
    )
    return candidates[:limit]


def _binary_mask(path: Path) -> Image.Image:
    image = Image.open(path).convert("L")
    return image.point(lambda x: 255 if x > 0 else 0)


def _save_variant(src_dir: Path, dst_dir: Path, variant: str) -> dict[str, str]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    answer = _binary_mask(src_dir / "answer.png")
    union = _binary_mask(src_dir / "union.png")

    if variant == "dilate":
        answer = answer.filter(ImageFilter.MaxFilter(9))
        union = union.filter(ImageFilter.MaxFilter(9))
    elif variant == "erode":
        answer = answer.filter(ImageFilter.MinFilter(9))
        union = union.filter(ImageFilter.MinFilter(9))
    elif variant != "original":
        raise ValueError(f"unknown variant: {variant}")

    answer_path = dst_dir / "answer.png"
    union_path = dst_dir / "union.png"
    answer.save(answer_path)
    union.save(union_path)

    shifted_src = src_dir / "shifted.png"
    shuffled_src = src_dir / "shuffled.png"
    shifted_dst = dst_dir / "shifted.png"
    shuffled_dst = dst_dir / "shuffled.png"
    shutil.copy2(shifted_src, shifted_dst)
    shutil.copy2(shuffled_src, shuffled_dst)
    return {
        "mask_dir": str(dst_dir),
        "answer_mask_path": str(answer_path),
        "union_mask_path": str(union_path),
        "shifted_mask_path": str(shifted_dst),
        "shuffled_mask_path": str(shuffled_dst),
    }


def _variant_manifest_rows(base_rows: list[dict[str, str]], sample_ids: set[str], variant_map: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in base_rows:
        sample_id = row.get("sample_id", "")
        prompt_name = row.get("prompt_name", "")
        if sample_id not in sample_ids:
            continue
        if prompt_name not in {"B_direct", "D_visual_only"}:
            continue
        if sample_id not in variant_map:
            continue
        merged = dict(row)
        merged.update(variant_map[sample_id])
        out.append(merged)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="defensive_v1")
    parser.add_argument("--smoke-samples", type=int, default=3)
    args = parser.parse_args()

    selected = _load_selected_samples(args.smoke_samples)
    base_rows = _read_csv(ROUTE_FIRST_MANIFEST)
    sample_ids = {row["sample_id"] for row in selected}

    variant_maps: dict[str, dict[str, dict[str, str]]] = {"original": {}, "dilate": {}, "erode": {}}
    asset_rows: list[dict[str, Any]] = []
    for row in selected:
        sample_id = row["sample_id"]
        src_dir = Path(row["mask_dir"])
        stem = src_dir.name
        for variant in ["original", "dilate", "erode"]:
            dst_dir = VARIANT_ROOT / args.tag / variant / stem
            variant_paths = _save_variant(src_dir, dst_dir, variant)
            variant_maps[variant][sample_id] = variant_paths
        asset_rows.append(
            {
                "sample_id": sample_id,
                "image_filename": row.get("image_filename", ""),
                "union_area_frac": row.get("union_area_frac", ""),
                "random16_valid_count": row.get("random16_valid_count", ""),
                "mask_dir": row.get("mask_dir", ""),
            }
        )

    manifests: dict[str, str] = {}
    for variant in ["original", "dilate", "erode"]:
        manifest_rows = _variant_manifest_rows(base_rows, sample_ids, variant_maps[variant])
        manifest_path = STAGE6_CROSS / f"stage6_mask_robustness_{variant}_smoke_{args.tag}_manifest.csv"
        manifests[variant] = str(manifest_path)
        if manifest_rows:
            _write_csv(manifest_path, manifest_rows, list(manifest_rows[0].keys()))
        else:
            _write_csv(manifest_path, [], ["sample_id"])

    asset_summary_path = STAGE6_CROSS / f"stage6_mask_robustness_assets_smoke_{args.tag}.csv"
    _write_csv(
        asset_summary_path,
        asset_rows,
        ["sample_id", "image_filename", "union_area_frac", "random16_valid_count", "mask_dir"],
    )

    decision = {
        "tag": args.tag,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "mask_assets_ready" if selected else "blocked_no_smoke_samples",
        "smoke_samples": len(selected),
        "sample_ids": sorted(sample_ids),
        "variant_manifests": manifests,
    }
    _write_json(STAGE6_CROSS / f"stage6_mask_robustness_assets_smoke_{args.tag}_decision.json", decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
