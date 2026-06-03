#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(r"E:\Bridging")
STAGE3 = ROOT / "doc" / "experiments" / "stage3"
PAPERPACK = STAGE3 / "paperpack72"
LABELME_ROOT = ROOT / "annotation" / "stage3_paperpack72_labelme"
ORIGINAL_PACK = LABELME_ROOT / "single_pass1_primary72_clean"
REPLACEMENT_PACK = LABELME_ROOT / "replacement_pass1_candidates12"
FINAL_ASSET_ROOT = LABELME_ROOT / "paperpack81_final_assets"

OUT_POOL = PAPERPACK / "paperpack81_annotated_pool.csv"
OUT_PRIMARY = PAPERPACK / "paperpack72_primary_manifest.csv"
OUT_STRICT = PAPERPACK / "paperpack72_strict_sensitivity_manifest.csv"
OUT_PRIMARY_PROMPTS = PAPERPACK / "paperpack72_primary_prompt_runs.csv"
OUT_STRICT_PROMPTS = PAPERPACK / "paperpack72_strict_sensitivity_prompt_runs.csv"
OUT_EXCLUSION = PAPERPACK / "paperpack72_final_exclusion_manifest.csv"
OUT_BUILD_REPORT = PAPERPACK / "paperpack72_manifest_build_report.json"
OUT_MASK_SUMMARY = PAPERPACK / "paperpack81_mask_export_summary.csv"
OUT_CONTROL_SUMMARY = PAPERPACK / "paperpack81_control_mask_summary.csv"
OUT_MASK_REPORT = PAPERPACK / "paperpack81_mask_export_report.json"
OUT_GEOMETRY_WARNINGS = PAPERPACK / "paperpack81_mask_geometry_warnings.csv"

ALLOWED_LABELS = {"answer", "relate"}
EXCLUDE_RECOMMENDATION = "exclude_or_replace"
MODERATE_RECOMMENDATION = "manual_review_image_dependence"
RANDOM_CONTROL_COUNT = 16
RANDOM_MAX_TRIES = 64
RANDOM_IOU_TARGET = 0.10
PROMPTS = ("B_direct", "D_visual_only")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_labelme(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_seed(*parts: str) -> int:
    raw = "||".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**32)


def sanitize_shape_type(shape: dict[str, Any]) -> str:
    return str(shape.get("shape_type") or "polygon").strip().lower()


def draw_shape(mask: Image.Image, shape: dict[str, Any]) -> None:
    points = [(float(x), float(y)) for x, y in shape.get("points", [])]
    if not points:
        return
    draw = ImageDraw.Draw(mask)
    shape_type = sanitize_shape_type(shape)
    if shape_type == "rectangle" and len(points) >= 2:
        (x1, y1), (x2, y2) = points[:2]
        draw.rectangle((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), fill=255)
    elif shape_type == "circle" and len(points) >= 2:
        (x1, y1), (x2, y2) = points[:2]
        r = math.hypot(x2 - x1, y2 - y1)
        draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill=255)
    elif shape_type in {"line", "linestrip"} and len(points) >= 2:
        draw.line(points, fill=255, width=3)
    elif shape_type == "point":
        x, y = points[0]
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=255)
    elif len(points) >= 3:
        draw.polygon(points, fill=255)


def rasterize_label(data: dict[str, Any], label: str, size: tuple[int, int]) -> tuple[Image.Image, int]:
    mask = Image.new("L", size, 0)
    count = 0
    for shape in data.get("shapes", []):
        if str(shape.get("label", "")).strip().lower() != label:
            continue
        draw_shape(mask, shape)
        count += 1
    return mask, count


def mask_bool(mask: Image.Image) -> np.ndarray:
    return np.asarray(mask.convert("L"), dtype=np.uint8) > 0


def save_bool_mask(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8) * 255).save(path)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = float(np.logical_or(mask_a, mask_b).sum())
    if union <= 0:
        return 0.0
    return float(np.logical_and(mask_a, mask_b).sum()) / union


def translated_shape_control(reference: np.ndarray, avoid: np.ndarray) -> tuple[np.ndarray, float, bool]:
    bbox = bbox_from_mask(reference)
    if bbox is None:
        return np.zeros_like(reference, dtype=bool), 0.0, False
    h, w = reference.shape
    x0, y0, x1, y1 = bbox
    crop = reference[y0:y1, x0:x1]
    crop_h, crop_w = crop.shape
    candidates = [
        (0, 0),
        (w - crop_w, 0),
        (0, h - crop_h),
        (w - crop_w, h - crop_h),
        (max(0, (w - crop_w) // 2), 0),
        (max(0, (w - crop_w) // 2), h - crop_h),
        (0, max(0, (h - crop_h) // 2)),
        (w - crop_w, max(0, (h - crop_h) // 2)),
    ]
    best: tuple[float, np.ndarray] | None = None
    for tx, ty in candidates:
        tx = max(0, min(tx, w - crop_w))
        ty = max(0, min(ty, h - crop_h))
        candidate = np.zeros_like(reference, dtype=bool)
        candidate[ty : ty + crop_h, tx : tx + crop_w] = crop
        iou = mask_iou(candidate, avoid)
        if best is None or iou < best[0]:
            best = (iou, candidate)
    if best is None:
        return np.zeros_like(reference, dtype=bool), 0.0, False
    return best[1], float(best[0]), bool(best[0] <= RANDOM_IOU_TARGET)


def area_matched_rectangle(reference: np.ndarray, x0: int, y0: int) -> np.ndarray | None:
    target_area = int(reference.sum())
    if target_area <= 0:
        return np.zeros_like(reference, dtype=bool)
    bbox = bbox_from_mask(reference)
    if bbox is None:
        return np.zeros_like(reference, dtype=bool)
    h, w = reference.shape
    bx0, by0, bx1, by1 = bbox
    bbox_w = max(1, bx1 - bx0)
    bbox_h = max(1, by1 - by0)
    aspect = bbox_w / bbox_h
    rect_w = max(1, int(round((target_area * aspect) ** 0.5)))
    rect_h = max(1, int(math.ceil(target_area / rect_w)))
    if rect_w > w or rect_h > h:
        scale = max(rect_w / max(1, w), rect_h / max(1, h))
        rect_w = max(1, int(rect_w / scale))
        rect_h = max(1, int(math.ceil(target_area / rect_w)))
    if x0 < 0 or y0 < 0 or x0 + rect_w > w or y0 + rect_h > h:
        return None
    out = np.zeros_like(reference, dtype=bool)
    full_rows = target_area // rect_w
    rem = target_area % rect_w
    used_rows = min(full_rows, rect_h)
    if used_rows > 0:
        out[y0 : y0 + used_rows, x0 : x0 + rect_w] = True
    if rem > 0 and used_rows < rect_h:
        out[y0 + used_rows, x0 : x0 + rem] = True
    return out


def random_area_control(reference: np.ndarray, avoid: np.ndarray, seed: int) -> tuple[np.ndarray, float, bool]:
    target_area = int(reference.sum())
    if target_area <= 0:
        return np.zeros_like(reference, dtype=bool), 0.0, False
    bbox = bbox_from_mask(reference)
    if bbox is None:
        return np.zeros_like(reference, dtype=bool), 0.0, False
    h, w = reference.shape
    bx0, by0, bx1, by1 = bbox
    aspect = max(1, bx1 - bx0) / max(1, by1 - by0)
    rect_w = min(w, max(1, int(round((target_area * aspect) ** 0.5))))
    rect_h = min(h, max(1, int(math.ceil(target_area / rect_w))))
    rng = np.random.default_rng(seed)
    best_mask: np.ndarray | None = None
    best_iou: float | None = None
    for _ in range(RANDOM_MAX_TRIES):
        x0 = int(rng.integers(0, max(1, w - rect_w + 1)))
        y0 = int(rng.integers(0, max(1, h - rect_h + 1)))
        candidate = area_matched_rectangle(reference, x0, y0)
        if candidate is None:
            continue
        iou = mask_iou(candidate, avoid)
        if best_iou is None or iou < best_iou:
            best_iou = iou
            best_mask = candidate
        if iou <= RANDOM_IOU_TARGET:
            return candidate, float(iou), True
    if best_mask is None:
        return np.zeros_like(reference, dtype=bool), 0.0, False
    return best_mask, float(best_iou or 0.0), False


def copy_asset(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or src.stat().st_size != dst.stat().st_size:
        shutil.copy2(src, dst)


def normalize_original_row(row: dict[str, str], audit: dict[str, dict[str, str]]) -> dict[str, Any]:
    sample_id = row["sample_id"]
    recommendation = audit.get(sample_id, {}).get("recommendation", "keep_pending_export")
    if recommendation == EXCLUDE_RECOMMENDATION:
        tier = "exclude"
    elif recommendation == MODERATE_RECOMMENDATION:
        tier = "moderate"
    else:
        tier = "strong"
    return {
        "sample_id": sample_id,
        "image_filename": row["image_filename"],
        "source_image_path": row["image_path"],
        "question_text": row["question_text"],
        "answer_text": row["answer_text"],
        "quota_slot": row.get("quota_slot", ""),
        "reasoning_operation": row.get("proposed_reasoning_operation", ""),
        "proposed_reasoning_operation": row.get("proposed_reasoning_operation", ""),
        "paperpack_source": "original",
        "original_rank": row.get("paperpack72_rank", ""),
        "replacement_rank": "",
        "image_dependence_tier": tier,
        "manual_review_flag": "1" if recommendation == MODERATE_RECOMMENDATION else "0",
        "audit_recommendation": recommendation,
        "audit_flags": audit.get(sample_id, {}).get("flags", ""),
        "exclusion_reason": audit.get(sample_id, {}).get("user_flag_reason", ""),
        "source_labelme_json_path": row["labelme_json_expected"],
    }


def normalize_replacement_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "image_filename": row["image_filename"],
        "source_image_path": row["image_path"],
        "question_text": row["question_text"],
        "answer_text": row["answer_text"],
        "quota_slot": row.get("quota_slot", ""),
        "reasoning_operation": row.get("proposed_reasoning_operation", ""),
        "proposed_reasoning_operation": row.get("proposed_reasoning_operation", ""),
        "paperpack_source": "replacement",
        "original_rank": "",
        "replacement_rank": row.get("replacement_rank", ""),
        "image_dependence_tier": "replacement",
        "manual_review_flag": "0",
        "audit_recommendation": "replacement_candidate",
        "audit_flags": "",
        "exclusion_reason": "",
        "source_labelme_json_path": row["labelme_json_expected"],
    }


def build_manifests() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    audit_rows = {row["sample_id"]: row for row in read_csv(PAPERPACK / "paperpack72_pass1_audit.csv")}
    original = [normalize_original_row(row, audit_rows) for row in read_csv(ORIGINAL_PACK / "manifest.csv")]
    replacement = [normalize_replacement_row(row) for row in read_csv(REPLACEMENT_PACK / "manifest.csv")]

    exclusions = [row for row in original if row["image_dependence_tier"] == "exclude"]
    original_pool = [row for row in original if row["image_dependence_tier"] != "exclude"]
    replacements_sorted = sorted(replacement, key=lambda row: int(row["replacement_rank"]))
    primary_replacement_ids = {row["sample_id"] for row in replacements_sorted[:3]}
    strict_replacement_ids = {row["sample_id"] for row in replacements_sorted[:8]}
    strict_excluded_ids = {row["sample_id"] for row in original if row["image_dependence_tier"] in {"exclude", "moderate"}}

    pool = original_pool + replacements_sorted
    if len(pool) != 81:
        raise RuntimeError(f"expected paperpack81 pool to have 81 rows, got {len(pool)}")

    for row in pool:
        sid = row["sample_id"]
        row["include_annotated_pool"] = "1"
        row["include_primary"] = "1" if row["paperpack_source"] == "original" or sid in primary_replacement_ids else "0"
        row["include_strict_sensitivity"] = (
            "1"
            if (row["paperpack_source"] == "original" and sid not in strict_excluded_ids)
            or sid in strict_replacement_ids
            else "0"
        )
        row["analysis_image_dependence_tier"] = (
            "moderate" if row["image_dependence_tier"] == "moderate" else "strong_or_replacement"
        )

    primary = [row for row in pool if row["include_primary"] == "1"]
    strict = [row for row in pool if row["include_strict_sensitivity"] == "1"]
    if len(primary) != 72:
        raise RuntimeError(f"expected paperpack72 primary to have 72 rows, got {len(primary)}")
    if len(strict) != 72:
        raise RuntimeError(f"expected paperpack72 strict sensitivity to have 72 rows, got {len(strict)}")

    seen_samples = [row["sample_id"] for row in pool]
    if len(seen_samples) != len(set(seen_samples)):
        raise RuntimeError("duplicate sample_id in paperpack81 pool")
    image_counts = Counter(row["image_filename"] for row in pool)
    duplicate_images = [name for name, count in image_counts.items() if count > 1]
    if duplicate_images:
        raise RuntimeError(f"duplicate image filenames would collide in mask dirs: {duplicate_images}")

    return pool, primary, strict, exclusions


def attach_asset_paths(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        image_src = Path(row["source_image_path"])
        json_src = Path(row["source_labelme_json_path"])
        image_dst = FINAL_ASSET_ROOT / "images" / row["image_filename"]
        json_dst = FINAL_ASSET_ROOT / "labelme_json" / f"{Path(row['image_filename']).stem}.json"
        mask_dir = FINAL_ASSET_ROOT / "exported_masks" / Path(row["image_filename"]).stem
        row["local_image_path"] = str(image_dst)
        row["image_path"] = str(image_dst)
        row["labelme_json_path"] = str(json_dst)
        row["mask_dir"] = str(mask_dir)
        row["answer_mask_path"] = str(mask_dir / "answer.png")
        row["relate_mask_path"] = str(mask_dir / "relate.png")
        row["union_mask_path"] = str(mask_dir / "union.png")
        row["shifted_mask_path"] = str(mask_dir / "shifted.png")
        row["shuffled_mask_path"] = str(mask_dir / "shuffled.png")
        row["random16_mask_dir"] = str(mask_dir / "random16")
        copy_asset(image_src, image_dst)
        copy_asset(json_src, json_dst)


def export_masks(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mask_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for row in rows:
        image_path = Path(row["local_image_path"])
        json_path = Path(row["labelme_json_path"])
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        data = load_labelme(json_path)
        labels = [str(shape.get("label", "")).strip().lower() for shape in data.get("shapes", [])]
        invalid = sorted({label for label in labels if label not in ALLOWED_LABELS})
        answer, answer_shapes = rasterize_label(data, "answer", image.size)
        relate, relate_shapes = rasterize_label(data, "relate", image.size)
        answer_arr = mask_bool(answer)
        relate_arr = mask_bool(relate)
        union_arr = np.logical_or(answer_arr, relate_arr)
        mask_dir = Path(row["mask_dir"])
        mask_dir.mkdir(parents=True, exist_ok=True)
        answer.save(mask_dir / "answer.png")
        relate.save(mask_dir / "relate.png")
        save_bool_mask(mask_dir / "union.png", union_arr)

        shifted, shifted_iou, shifted_valid = translated_shape_control(union_arr, union_arr)
        save_bool_mask(mask_dir / "shifted.png", shifted)
        shuffled, shuffled_iou, shuffled_valid = random_area_control(
            union_arr, union_arr, stable_seed(row["sample_id"], "shuffled")
        )
        save_bool_mask(mask_dir / "shuffled.png", shuffled)

        random_dir = mask_dir / "random16"
        random_valid_count = 0
        for idx in range(1, RANDOM_CONTROL_COUNT + 1):
            seed = stable_seed(row["sample_id"], f"random16_{idx}")
            random_mask, random_iou, random_valid = random_area_control(union_arr, union_arr, seed)
            random_valid_count += int(random_valid)
            random_path = random_dir / f"random_control_{idx:02d}.png"
            save_bool_mask(random_path, random_mask)
            control_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "image_filename": row["image_filename"],
                    "condition": f"random_control_{idx:02d}",
                    "mask_path": str(random_path),
                    "seed": seed,
                    "actual_iou_with_union": f"{random_iou:.6g}",
                    "valid_iou_le_0p10": "1" if random_valid else "0",
                    "area_px": int(random_mask.sum()),
                    "reference_union_area_px": int(union_arr.sum()),
                }
            )

        total_px = width * height
        union_frac = float(union_arr.sum() / total_px)
        answer_frac = float(answer_arr.sum() / total_px)
        if union_frac > 0.75:
            geometry_tier = "very_broad_union"
        elif union_frac > 0.50:
            geometry_tier = "broad_union"
        elif union_frac > 0.20:
            geometry_tier = "medium_union"
        else:
            geometry_tier = "compact_union"
        geometry_warnings = []
        if answer_frac > 0.50:
            geometry_warnings.append("answer_area_gt_50pct")
        if union_frac > 0.50:
            geometry_warnings.append("union_area_gt_50pct")
        if random_valid_count < 4:
            geometry_warnings.append("random16_low_valid_count")

        row.update(
            {
                "width": width,
                "height": height,
                "answer_area_px": int(answer_arr.sum()),
                "relate_area_px": int(relate_arr.sum()),
                "union_area_px": int(union_arr.sum()),
                "answer_area_frac": f"{answer_frac:.8f}",
                "relate_area_frac": f"{relate_arr.sum() / total_px:.8f}",
                "union_area_frac": f"{union_frac:.8f}",
                "mask_geometry_tier": geometry_tier,
                "mask_geometry_warnings": "|".join(geometry_warnings),
                "answer_shape_count": answer_shapes,
                "relate_shape_count": relate_shapes,
                "invalid_labels": "|".join(invalid),
                "mask_export_status": "ok" if answer_arr.sum() > 0 and relate_arr.sum() > 0 and not invalid else "needs_fix",
                "shifted_mask_iou_with_union": f"{shifted_iou:.6g}",
                "shifted_mask_valid": "1" if shifted_valid else "0",
                "shuffled_mask_iou_with_union": f"{shuffled_iou:.6g}",
                "shuffled_mask_valid": "1" if shuffled_valid else "0",
                "random16_valid_count": random_valid_count,
            }
        )
        mask_rows.append(
            {
                "sample_id": row["sample_id"],
                "image_filename": row["image_filename"],
                "width": width,
                "height": height,
                "labels": "|".join(labels),
                "invalid_labels": "|".join(invalid),
                "answer_shape_count": answer_shapes,
                "relate_shape_count": relate_shapes,
                "answer_area_px": int(answer_arr.sum()),
                "relate_area_px": int(relate_arr.sum()),
                "union_area_px": int(union_arr.sum()),
                "answer_area_frac": row["answer_area_frac"],
                "relate_area_frac": row["relate_area_frac"],
                "union_area_frac": row["union_area_frac"],
                "mask_geometry_tier": row["mask_geometry_tier"],
                "mask_geometry_warnings": row["mask_geometry_warnings"],
                "mask_export_status": row["mask_export_status"],
                "shifted_mask_iou_with_union": row["shifted_mask_iou_with_union"],
                "shuffled_mask_iou_with_union": row["shuffled_mask_iou_with_union"],
                "random16_valid_count": random_valid_count,
                "mask_dir": row["mask_dir"],
            }
        )
        control_rows.extend(
            [
                {
                    "sample_id": row["sample_id"],
                    "image_filename": row["image_filename"],
                    "condition": "shifted_mask",
                    "mask_path": row["shifted_mask_path"],
                    "seed": "",
                    "actual_iou_with_union": f"{shifted_iou:.6g}",
                    "valid_iou_le_0p10": "1" if shifted_valid else "0",
                    "area_px": int(shifted.sum()),
                    "reference_union_area_px": int(union_arr.sum()),
                },
                {
                    "sample_id": row["sample_id"],
                    "image_filename": row["image_filename"],
                    "condition": "shuffled_mask",
                    "mask_path": row["shuffled_mask_path"],
                    "seed": stable_seed(row["sample_id"], "shuffled"),
                    "actual_iou_with_union": f"{shuffled_iou:.6g}",
                    "valid_iou_le_0p10": "1" if shuffled_valid else "0",
                    "area_px": int(shuffled.sum()),
                    "reference_union_area_px": int(union_arr.sum()),
                },
            ]
        )
    return mask_rows, control_rows


def build_prompt_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        for prompt_name in PROMPTS:
            out.append(
                {
                    "sample_id": row["sample_id"],
                    "prompt_name": prompt_name,
                    "image_filename": row["image_filename"],
                    "local_image_path": row["local_image_path"],
                    "question_text": row["question_text"],
                    "answer_text": row["answer_text"],
                    "reasoning_operation": row["reasoning_operation"],
                    "image_dependence_tier": row["image_dependence_tier"],
                    "paperpack_source": row["paperpack_source"],
                    "mask_dir": row["mask_dir"],
                }
            )
    return out


FIELDNAMES = [
    "sample_id",
    "image_filename",
    "local_image_path",
    "image_path",
    "question_text",
    "answer_text",
    "quota_slot",
    "reasoning_operation",
    "proposed_reasoning_operation",
    "paperpack_source",
    "original_rank",
    "replacement_rank",
    "image_dependence_tier",
    "analysis_image_dependence_tier",
    "manual_review_flag",
    "audit_recommendation",
    "audit_flags",
    "include_annotated_pool",
    "include_primary",
    "include_strict_sensitivity",
    "source_labelme_json_path",
    "labelme_json_path",
    "mask_dir",
    "answer_mask_path",
    "relate_mask_path",
    "union_mask_path",
    "shifted_mask_path",
    "shuffled_mask_path",
    "random16_mask_dir",
    "width",
    "height",
    "answer_shape_count",
    "relate_shape_count",
    "answer_area_px",
    "relate_area_px",
    "union_area_px",
    "answer_area_frac",
    "relate_area_frac",
    "union_area_frac",
    "mask_geometry_tier",
    "mask_geometry_warnings",
    "invalid_labels",
    "mask_export_status",
    "shifted_mask_iou_with_union",
    "shifted_mask_valid",
    "shuffled_mask_iou_with_union",
    "shuffled_mask_valid",
    "random16_valid_count",
    "exclusion_reason",
]


def main() -> int:
    pool, primary, strict, exclusions = build_manifests()
    attach_asset_paths(pool)
    mask_rows, control_rows = export_masks(pool)

    primary_ids = {row["sample_id"] for row in primary}
    strict_ids = {row["sample_id"] for row in strict}
    primary_out = [row for row in pool if row["sample_id"] in primary_ids]
    strict_out = [row for row in pool if row["sample_id"] in strict_ids]

    write_csv(OUT_POOL, pool, FIELDNAMES)
    write_csv(OUT_PRIMARY, primary_out, FIELDNAMES)
    write_csv(OUT_STRICT, strict_out, FIELDNAMES)
    prompt_fields = [
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
    write_csv(OUT_PRIMARY_PROMPTS, build_prompt_runs(primary_out), prompt_fields)
    write_csv(OUT_STRICT_PROMPTS, build_prompt_runs(strict_out), prompt_fields)
    write_csv(OUT_EXCLUSION, exclusions, FIELDNAMES)
    write_csv(
        OUT_MASK_SUMMARY,
        mask_rows,
        [
            "sample_id",
            "image_filename",
            "width",
            "height",
            "labels",
            "invalid_labels",
            "answer_shape_count",
            "relate_shape_count",
            "answer_area_px",
            "relate_area_px",
            "union_area_px",
            "answer_area_frac",
            "relate_area_frac",
            "union_area_frac",
            "mask_geometry_tier",
            "mask_geometry_warnings",
            "mask_export_status",
            "shifted_mask_iou_with_union",
            "shuffled_mask_iou_with_union",
            "random16_valid_count",
            "mask_dir",
        ],
    )
    geometry_warnings = [row for row in mask_rows if row.get("mask_geometry_warnings")]
    write_csv(
        OUT_GEOMETRY_WARNINGS,
        geometry_warnings,
        [
            "sample_id",
            "image_filename",
            "answer_area_frac",
            "relate_area_frac",
            "union_area_frac",
            "mask_geometry_tier",
            "mask_geometry_warnings",
            "random16_valid_count",
            "mask_dir",
        ],
    )
    write_csv(
        OUT_CONTROL_SUMMARY,
        control_rows,
        [
            "sample_id",
            "image_filename",
            "condition",
            "mask_path",
            "seed",
            "actual_iou_with_union",
            "valid_iou_le_0p10",
            "area_px",
            "reference_union_area_px",
        ],
    )

    mask_status = Counter(row["mask_export_status"] for row in mask_rows)
    random_valid_counts = [int(row["random16_valid_count"]) for row in mask_rows]
    report = {
        "status": "completed" if mask_status.get("needs_fix", 0) == 0 else "needs_fix",
        "asset_root": str(FINAL_ASSET_ROOT),
        "pool_count": len(pool),
        "primary_count": len(primary_out),
        "strict_sensitivity_count": len(strict_out),
        "exclusion_count": len(exclusions),
        "moderate_count_in_pool": sum(1 for row in pool if row["image_dependence_tier"] == "moderate"),
        "replacement_count_in_pool": sum(1 for row in pool if row["paperpack_source"] == "replacement"),
        "primary_replacement_count": sum(1 for row in primary_out if row["paperpack_source"] == "replacement"),
        "strict_replacement_count": sum(1 for row in strict_out if row["paperpack_source"] == "replacement"),
        "mask_status_counts": dict(mask_status),
        "random16_min_valid_count": min(random_valid_counts) if random_valid_counts else 0,
        "random16_mean_valid_count": sum(random_valid_counts) / len(random_valid_counts) if random_valid_counts else 0,
        "geometry_warning_count": len(geometry_warnings),
        "geometry_tier_counts": dict(Counter(row["mask_geometry_tier"] for row in mask_rows)),
        "outputs": {
            "pool": str(OUT_POOL),
            "primary": str(OUT_PRIMARY),
            "primary_prompt_runs": str(OUT_PRIMARY_PROMPTS),
            "strict_sensitivity": str(OUT_STRICT),
            "strict_sensitivity_prompt_runs": str(OUT_STRICT_PROMPTS),
            "exclusion": str(OUT_EXCLUSION),
            "mask_summary": str(OUT_MASK_SUMMARY),
            "control_summary": str(OUT_CONTROL_SUMMARY),
            "geometry_warnings": str(OUT_GEOMETRY_WARNINGS),
        },
    }
    write_json(OUT_BUILD_REPORT, report)
    write_json(OUT_MASK_REPORT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
