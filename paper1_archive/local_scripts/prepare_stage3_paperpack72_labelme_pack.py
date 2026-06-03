#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(r"E:\Bridging")
PAPERPACK = ROOT / "doc" / "experiments" / "stage3" / "paperpack72"
OUT_ROOT = ROOT / "annotation" / "stage3_paperpack72_labelme"
PYTHON_EXE = Path(r"E:\code\conda\python.exe")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_image_path(row: dict[str, str]) -> Path:
    if row.get("local_image_path"):
        return Path(row["local_image_path"])
    image_url = row.get("image_url", "")
    if image_url.startswith("file:///"):
        parsed = urlparse(image_url)
        return Path(unquote(parsed.path).lstrip("/"))
    raise ValueError(f"Missing local image path for {row.get('sample_id')}")


def image_stem(row: dict[str, str]) -> str:
    return Path(row["image_filename"]).stem


def prepare_pack(out_name: str, rows: list[dict[str, str]], title: str) -> None:
    out_dir = OUT_ROOT / out_name
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    question_lines = [
        f"# Paperpack72 {title} Question Sheet",
        "",
        "LabelMe 里只画两个 label：`answer` 和 `relate`。",
        "",
        "`answer` = 最直接支持答案的核心区域；`relate` = 必要上下文或支持区域。",
        "",
    ]

    for row in rows:
        src = resolve_image_path(row)
        if not src.exists():
            raise FileNotFoundError(f"Image not found for {row.get('sample_id')}: {src}")
        dst = images_dir / row["image_filename"]
        if not dst.exists():
            shutil.copy2(src, dst)
        clean_row = {
            "paperpack72_rank": row.get("paperpack72_rank", ""),
            "sample_id": row.get("sample_id", ""),
            "image_filename": row.get("image_filename", ""),
            "image_path": str(dst),
            "question_text": row.get("question_text", ""),
            "answer_text": row.get("answer_text", ""),
            "quota_slot": row.get("quota_slot", ""),
            "proposed_reasoning_operation": row.get("proposed_reasoning_operation", ""),
            "selection_note": row.get("selection_note", ""),
            "labelme_json_expected": str(images_dir / f"{image_stem(row)}.json"),
            "annotator_notes": "",
        }
        manifest_rows.append(clean_row)
        question_lines.extend(
            [
                f"## {row.get('paperpack72_rank')} / {row.get('sample_id')}",
                "",
                f"- image: `{row.get('image_filename')}`",
                f"- question: {row.get('question_text')}",
                f"- answer: `{row.get('answer_text')}`",
                f"- quota_slot: `{row.get('quota_slot')}`",
                f"- proposed_type: `{row.get('proposed_reasoning_operation')}`",
                "",
            ]
        )

    write_csv(
        out_dir / "manifest.csv",
        manifest_rows,
        [
            "paperpack72_rank",
            "sample_id",
            "image_filename",
            "image_path",
            "question_text",
            "answer_text",
            "quota_slot",
            "proposed_reasoning_operation",
            "selection_note",
            "labelme_json_expected",
            "annotator_notes",
        ],
    )
    (out_dir / "QUESTION_SHEET.md").write_text("\n".join(question_lines), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                f"# Paperpack72 {title}",
                "",
                "## Start LabelMe",
                "",
                "```powershell",
                f"powershell -ExecutionPolicy Bypass -File {out_dir / 'launch_labelme.ps1'}",
                "```",
                "",
                "## Labels",
                "",
                "- `answer`: 最直接支持答案的核心区域",
                "- `relate`: 必要上下文或支持区域",
                "",
                "标完后，每张图片旁边应有同名 `.json` 文件。",
            ]
        ),
        encoding="utf-8",
    )
    (out_dir / "launch_labelme.ps1").write_text(
        "\n".join(
            [
                f'$imagesDir = "{images_dir}"',
                f'$pythonExe = "{PYTHON_EXE}"',
                "",
                'Write-Host "Launching LabelMe on $imagesDir"',
                'Start-Process -FilePath $pythonExe -ArgumentList @("-m", "labelme", $imagesDir)',
            ]
        ),
        encoding="utf-8",
    )
    print(f"Prepared {title} pack: {out_dir}")


def prepare_one(annotator: str) -> None:
    assignment = PAPERPACK / f"annotator_{annotator}_assignment.csv"
    rows = read_csv(assignment)
    prepare_pack(f"annotator_{annotator}", rows, f"Annotator {annotator.upper()}")


def prepare_single() -> None:
    rows = read_csv(PAPERPACK / "paperpack72_manifest_template.csv")
    prepare_pack("single_pass1_primary72_clean", rows, "Single Pass 1 Primary72 Clean")
    shuffled = list(rows)
    random.Random(1729).shuffle(shuffled)
    for idx, row in enumerate(shuffled, start=1):
        row["paperpack72_rank"] = str(idx)
    prepare_pack("single_pass2_primary72_reliability_clean", shuffled, "Single Pass 2 Primary72 Reliability Clean")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator", choices=["a", "b", "both", "single"], default="both")
    args = parser.parse_args()
    if args.annotator == "single":
        prepare_single()
    else:
        annotators = ["a", "b"] if args.annotator == "both" else [args.annotator]
        for annotator in annotators:
            prepare_one(annotator)


if __name__ == "__main__":
    main()
