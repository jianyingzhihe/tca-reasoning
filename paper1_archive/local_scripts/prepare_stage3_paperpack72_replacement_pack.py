#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(r"E:\Bridging")
PAPERPACK = ROOT / "doc" / "experiments" / "stage3" / "paperpack72"
OUT_ROOT = ROOT / "annotation" / "stage3_paperpack72_labelme"
OUT_NAME = "replacement_pass1_candidates12"
PYTHON_EXE = Path(r"E:\code\conda\python.exe")

RESERVE = PAPERPACK / "paperpack72_reserve_candidates.csv"
OUT_SELECTION = PAPERPACK / "paperpack72_replacement_candidates12.csv"

# Hand-picked for stronger image dependence than the three user-flagged
# generic-knowledge cases. This is intentionally conservative: annotate 12,
# then admit only the needed replacements after audit.
REPLACEMENT_IDS = [
    "okvqa_val_03506",   # dog breed visible in image
    "okvqa_val_03609",   # shower material visible in image
    "okvqa_val_03979",   # visible beverage/brand cue
    "okvqa_val_04593",   # visible animal relationship
    "okvqa_val_04177",   # visible human activity
    "okvqa_val_1286755", # visible place/setting
    "okvqa_val_2357845", # visible skiing subtype
    "okvqa_val_355265",  # visible bathroom setting
    "okvqa_val_4513245", # visible computer brand
    "okvqa_val_5179385", # visible umpire/person role
    "okvqa_val_5201095", # visible cloud type
    "okvqa_val_579175",  # visible fence type
]


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
    local = row.get("local_image_path", "")
    if local:
        return Path(local)
    image_url = row.get("image_url", "")
    if image_url.startswith("file:///"):
        parsed = urlparse(image_url)
        return Path(unquote(parsed.path).lstrip("/"))
    raise ValueError(f"Missing local image path for {row.get('sample_id')}")


def image_stem(row: dict[str, str]) -> str:
    return Path(row["image_filename"]).stem


def main() -> None:
    reserve_rows = {row["sample_id"]: row for row in read_csv(RESERVE)}
    selected: list[dict[str, str]] = []
    missing_ids: list[str] = []
    for idx, sample_id in enumerate(REPLACEMENT_IDS, start=1):
        row = reserve_rows.get(sample_id)
        if row is None:
            missing_ids.append(sample_id)
            continue
        out = dict(row)
        out["replacement_rank"] = str(idx)
        out["replacement_reason"] = "stronger_image_dependence_prescreen"
        selected.append(out)

    if missing_ids:
        raise RuntimeError(f"Replacement ids not found in reserve CSV: {missing_ids}")

    out_dir = OUT_ROOT / OUT_NAME
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    question_lines = [
        "# Stage3 Paperpack72 Replacement Pass1 Candidates12",
        "",
        "只使用两个 LabelMe 标签：`answer` 和 `relate`。",
        "",
        "`answer` = 最直接支持答案的核心证据区域；`relate` = 必要上下文区域。",
        "",
        "这些是替换候选，不会自动进入最终 paperpack；标完后还要审计 image-dependence。",
        "",
    ]

    for row in selected:
        src = resolve_image_path(row)
        if not src.exists():
            raise FileNotFoundError(f"Image not found for {row['sample_id']}: {src}")
        dst = images_dir / row["image_filename"]
        if not dst.exists():
            shutil.copy2(src, dst)
        manifest_row = {
            "replacement_rank": row["replacement_rank"],
            "sample_id": row["sample_id"],
            "image_filename": row["image_filename"],
            "image_path": str(dst),
            "question_text": row["question_text"],
            "answer_text": row["answer_text"],
            "visual_tier": row.get("visual_tier", ""),
            "quota_slot": row.get("quota_slot", ""),
            "proposed_reasoning_operation": row.get("proposed_reasoning_operation", ""),
            "replacement_reason": row["replacement_reason"],
            "labelme_json_expected": str(images_dir / f"{image_stem(row)}.json"),
            "annotator_notes": "",
        }
        manifest_rows.append(manifest_row)
        question_lines.extend(
            [
                f"## R{row['replacement_rank']} / {row['sample_id']}",
                "",
                f"- image: `{row['image_filename']}`",
                f"- question: {row['question_text']}",
                f"- answer: `{row['answer_text']}`",
                f"- visual_tier: `{row.get('visual_tier', '')}`",
                f"- quota_slot: `{row.get('quota_slot', '')}`",
                f"- proposed_type: `{row.get('proposed_reasoning_operation', '')}`",
                "",
            ]
        )

    fields = [
        "replacement_rank",
        "sample_id",
        "image_filename",
        "image_path",
        "question_text",
        "answer_text",
        "visual_tier",
        "quota_slot",
        "proposed_reasoning_operation",
        "replacement_reason",
        "labelme_json_expected",
        "annotator_notes",
    ]
    write_csv(out_dir / "manifest.csv", manifest_rows, fields)
    write_csv(OUT_SELECTION, manifest_rows, fields)
    (out_dir / "QUESTION_SHEET.md").write_text("\n".join(question_lines), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Stage3 Paperpack72 Replacement Pass1 Candidates12",
                "",
                "## Start LabelMe",
                "",
                "```powershell",
                f"powershell -ExecutionPolicy Bypass -File {out_dir / 'launch_labelme.ps1'}",
                "```",
                "",
                "## Labels",
                "",
                "- `answer`: 最直接支持答案的核心证据区域",
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
    print(f"Prepared replacement pack: {out_dir}")
    print(f"Selection manifest: {OUT_SELECTION}")


if __name__ == "__main__":
    main()
