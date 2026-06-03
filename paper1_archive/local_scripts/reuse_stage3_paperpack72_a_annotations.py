#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
from pathlib import Path


ROOT = Path(r"E:\Bridging")
PAPERPACK = ROOT / "doc" / "experiments" / "stage3" / "paperpack72"
SOURCE_DIR = ROOT / "annotation" / "stage3_paperpack72_labelme" / "annotator_a"
TARGET_DIR = ROOT / "annotation" / "stage3_paperpack72_labelme" / "single_pass1_primary72_clean"
OUT_REPORT = PAPERPACK / "paperpack72_a_annotation_reuse_report.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    source_manifest = read_csv(SOURCE_DIR / "manifest.csv") if (SOURCE_DIR / "manifest.csv").exists() else []
    target_manifest = read_csv(TARGET_DIR / "manifest.csv")
    source_by_sample = {row["sample_id"]: row for row in source_manifest}
    source_by_image = {row["image_filename"]: row for row in source_manifest}
    report: list[dict[str, str]] = []

    copied = 0
    for target in target_manifest:
        sample_id = target["sample_id"]
        image_filename = target["image_filename"]
        source = source_by_sample.get(sample_id) or source_by_image.get(image_filename)
        target_json = Path(target["labelme_json_expected"])
        source_json = None
        status = "not_found_in_old_annotator_a"
        if source:
            source_json = Path(source["labelme_json_expected"])
            if source_json.exists():
                if target_json.exists():
                    status = "target_already_has_json_skipped"
                else:
                    shutil.copy2(source_json, target_json)
                    copied += 1
                    status = "copied_to_single_pass1_localized"
            else:
                status = "old_annotator_a_row_without_json"
        report.append(
            {
                "sample_id": sample_id,
                "image_filename": image_filename,
                "target_json": str(target_json),
                "source_json": str(source_json) if source_json else "",
                "reuse_status": status,
            }
        )

    write_csv(
        OUT_REPORT,
        report,
        ["sample_id", "image_filename", "target_json", "source_json", "reuse_status"],
    )
    print(f"copied={copied}")
    print(f"report={OUT_REPORT}")


if __name__ == "__main__":
    main()
