#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname


ROOT = Path(r"E:\Bridging")
SOURCE_DIR = ROOT / "annotation" / "okvqa_type_label_round4_400"
SOURCE_HTML = SOURCE_DIR / "round4_400_five_axis_annotation_ui.html"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.csv"
SOURCE_BLANK = SOURCE_DIR / "round4_400_five_axis_blank.csv"
SOURCE_README = SOURCE_DIR / "round4_400_five_axis_README.txt"
SOURCE_SCHEMA = ROOT / "annotation" / "okvqa_type_label_round3_320" / "five_axis_annotation_schema_v3_zh.md"
SOURCE_HISTORY = ROOT / "annotation" / "okvqa_type_label_round3_320" / "annotation_scheme_evolution_history.md"

OUT_DIR = ROOT / "annotation" / "okvqa_type_label_round4_400_mobile_package"
OUT_IMAGES = OUT_DIR / "images"
OUT_HTML = OUT_DIR / "round4_400_five_axis_annotation_ui_mobile.html"
OUT_MANIFEST = OUT_DIR / "manifest.csv"
OUT_BLANK = OUT_DIR / "round4_400_five_axis_blank.csv"
OUT_README = OUT_DIR / "README.txt"
OUT_ZIP = ROOT / "annotation" / "okvqa_type_label_round4_400_mobile_package.zip"


def file_uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    return Path(url2pathname(parsed.path))


def read_manifest_rows() -> list[dict[str, str]]:
    with SOURCE_MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_mobile_html(rows: list[dict[str, str]]) -> None:
    html_text = SOURCE_HTML.read_text(encoding="utf-8")
    for row in rows:
        image_filename = row["image_filename"]
        old_uri = row["image_url"]
        new_rel = f"images/{image_filename}"
        html_text = html_text.replace(old_uri, new_rel)
    html_text = html_text.replace("okvqa_round4_400_five_axis", "okvqa_round4_400_five_axis_mobile")
    html_text = html_text.replace(
        "round4_400_five_axis_annotation_ui.html",
        "round4_400_five_axis_annotation_ui_mobile.html",
    )
    OUT_HTML.write_text(html_text, encoding="utf-8")


def copy_images(rows: list[dict[str, str]]) -> None:
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for row in rows:
        image_filename = row["image_filename"]
        if image_filename in seen:
            continue
        seen.add(image_filename)
        src_path = file_uri_to_path(row["image_url"])
        dst_path = OUT_IMAGES / image_filename
        shutil.copy2(src_path, dst_path)


def build_manifest(rows: list[dict[str, str]]) -> None:
    mobile_rows = []
    for row in rows:
        mobile_row = dict(row)
        mobile_row["image_url"] = f"images/{row['image_filename']}"
        mobile_rows.append(mobile_row)
    write_manifest(OUT_MANIFEST, mobile_rows, list(mobile_rows[0].keys()))


def build_readme() -> None:
    readme = """Round4 400 Mobile Annotation Package
================================

This package is prepared for mobile use.

What is inside
--------------
- `round4_400_five_axis_annotation_ui_mobile.html`
- `round4_400_five_axis_blank.csv`
- `manifest.csv`
- `images/`
- `five_axis_annotation_schema_v3_zh.md`
- `annotation_scheme_evolution_history.md`

How to use
----------
1. Extract the whole folder on the phone.
2. Open `round4_400_five_axis_annotation_ui_mobile.html`.
3. Annotate all five axes.
4. Export CSV when finished.

Important
---------
- Keep the `images` folder in the same package.
- The old80 samples include legacy labels only as reference.
- Please annotate using the new 5-axis schema.
"""
    OUT_README.write_text(readme, encoding="utf-8")


def zip_package() -> None:
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(OUT_DIR.parent))


def main() -> None:
    rows = read_manifest_rows()

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    build_mobile_html(rows)
    build_manifest(rows)
    copy_images(rows)
    shutil.copy2(SOURCE_BLANK, OUT_BLANK)
    shutil.copy2(SOURCE_SCHEMA, OUT_DIR / SOURCE_SCHEMA.name)
    shutil.copy2(SOURCE_HISTORY, OUT_DIR / SOURCE_HISTORY.name)
    shutil.copy2(SOURCE_README, OUT_DIR / SOURCE_README.name)
    build_readme()
    zip_package()

    print(f"[done] out_dir={OUT_DIR}")
    print(f"[done] html={OUT_HTML}")
    print(f"[done] zip={OUT_ZIP}")


if __name__ == "__main__":
    main()
