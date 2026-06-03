#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw


def _find_image_for_json(json_path: Path) -> Path | None:
    stem = json_path.stem
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    for candidate in json_path.parent.glob(stem + ".*"):
        if candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return candidate
    return None


def _sanitize_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in label.strip()).strip("_") or "unlabeled"


def _shape_points(shape: dict) -> list[tuple[float, float]]:
    pts = shape.get("points", [])
    return [(float(x), float(y)) for x, y in pts]


def _draw_shape(mask: Image.Image, shape: dict) -> None:
    draw = ImageDraw.Draw(mask)
    pts = _shape_points(shape)
    if not pts:
        return
    shape_type = (shape.get("shape_type") or "polygon").lower()
    if shape_type == "rectangle" and len(pts) >= 2:
        (x1, y1), (x2, y2) = pts[:2]
        draw.rectangle((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), fill=255)
    elif shape_type == "circle" and len(pts) >= 2:
        (x1, y1), (x2, y2) = pts[:2]
        r = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill=255)
    elif shape_type == "point" and len(pts) >= 1:
        x, y = pts[0]
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=255)
    elif shape_type == "line" and len(pts) >= 2:
        draw.line(pts, fill=255, width=3)
    elif shape_type == "linestrip" and len(pts) >= 2:
        draw.line(pts, fill=255, width=3)
    else:
        draw.polygon(pts, fill=255)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Labelme polygons into per-label binary masks.")
    parser.add_argument("--annotation-dir", required=True, help="Directory containing images and Labelme JSON files.")
    parser.add_argument("--out-dir", required=True, help="Directory to write exported masks and summary CSV.")
    args = parser.parse_args()

    annotation_dir = Path(args.annotation_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(annotation_dir.glob("*.json"))
    if not json_paths:
        raise ValueError(f"no Labelme json files found in {annotation_dir}")

    summary_rows: list[dict[str, str]] = []

    for json_path in json_paths:
        image_path = _find_image_for_json(json_path)
        if image_path is None:
            raise FileNotFoundError(f"could not find image corresponding to {json_path}")

        data = json.loads(json_path.read_text(encoding="utf-8"))
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        shapes = data.get("shapes", [])

        by_label: dict[str, list[dict]] = {}
        for shape in shapes:
            label = _sanitize_label(shape.get("label", ""))
            by_label.setdefault(label, []).append(shape)

        image_stem = image_path.stem
        image_out_dir = out_dir / image_stem
        image_out_dir.mkdir(parents=True, exist_ok=True)

        label_areas: dict[str, int] = {}
        for label, label_shapes in sorted(by_label.items()):
            mask = Image.new("L", (width, height), 0)
            for shape in label_shapes:
                _draw_shape(mask, shape)
            mask_path = image_out_dir / f"{label}.png"
            mask.save(mask_path)
            label_areas[label] = sum(1 for px in mask.getdata() if px > 0)

        summary_rows.append(
            {
                "image_name": image_path.name,
                "json_name": json_path.name,
                "width": str(width),
                "height": str(height),
                "labels": ",".join(sorted(by_label.keys())),
                "shape_count": str(len(shapes)),
                "answer_area_px": str(label_areas.get("answer", 0)),
                "relate_area_px": str(label_areas.get("relate", 0)),
                "mask_dir": str(image_out_dir),
            }
        )

    summary_csv = out_dir / "mask_export_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_name",
                "json_name",
                "width",
                "height",
                "labels",
                "shape_count",
                "answer_area_px",
                "relate_area_px",
                "mask_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[done] json_files={len(json_paths)}")
    print(f"[done] out_dir={out_dir}")
    print(f"[done] summary_csv={summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
