#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
READOUT_DIR = ROOT / "doc" / "5.16" / "expanded_localized_discovery_screen20_readout_2026-05-18"
CANDIDATE_CSV = READOUT_DIR / "annotation_candidate_sheet.csv"
OUT_DIR = ROOT / "annotation" / "okvqa_evidence_labelme_round5_expanded20_route7"

IMAGE_ROOTS = [
    ROOT / "annotation" / "okvqa_type_label_round3_320" / "images",
    ROOT / "annotation" / "okvqa_type_label_round4_400_mobile_package" / "images",
]

PRIMARY_ORDER = [
    "okvqa_val_1882965",
    "okvqa_val_3774865",
    "okvqa_val_5606265",
    "okvqa_val_1251075",
    "okvqa_val_512035",
]

BACKUP_ORDER = [
    "okvqa_val_136595",
    "okvqa_val_3608785",
]

CASE_HINTS = {
    "okvqa_val_1882965": {
        "answer_hint": "Mark the ball as the smallest answer evidence.",
        "relate_hint": "Mark the nearby sport context/equipment if it helps disambiguate the sport.",
    },
    "okvqa_val_3774865": {
        "answer_hint": "Mark the man's hat as the smallest answer evidence.",
        "relate_hint": "Mark the head/upper body context if needed.",
    },
    "okvqa_val_5606265": {
        "answer_hint": "Mark all visible white legwear regions. Multiple answer shapes are OK.",
        "relate_hint": "Mark the girl's legs/body context if it supports the answer.",
    },
    "okvqa_val_1251075": {
        "answer_hint": "Mark the dog, especially the face/body used for breed recognition.",
        "relate_hint": "Mark the surrounding dog context or water-bottle interaction if useful.",
    },
    "okvqa_val_512035": {
        "answer_hint": "Mark the pink garment/color region directly supporting the answer.",
        "relate_hint": "Mark the full garment/person context if useful.",
    },
    "okvqa_val_136595": {
        "answer_hint": "Backup case: mark the visible logo/text region on the box.",
        "relate_hint": "Mark the surrounding box/package region if useful.",
    },
    "okvqa_val_3608785": {
        "answer_hint": "Backup case: mark the clearest equipment/ball/table region that determines the game.",
        "relate_hint": "Mark player/table context if the core cue alone is ambiguous.",
    },
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _find_image(image_filename: str) -> Path:
    for root in IMAGE_ROOTS:
        candidate = root / image_filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing image {image_filename} in {IMAGE_ROOTS}")


def _ordered_candidates(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_sample = {row["sample_id"]: row for row in rows}
    out: list[dict[str, str]] = []
    for sample_id in PRIMARY_ORDER:
        if sample_id in by_sample:
            row = dict(by_sample[sample_id])
            row["annotation_priority"] = "primary"
            out.append(row)
    for sample_id in BACKUP_ORDER:
        if sample_id in by_sample:
            row = dict(by_sample[sample_id])
            row["annotation_priority"] = "backup"
            out.append(row)
    return out


def main() -> int:
    rows = _ordered_candidates(_read_csv(CANDIDATE_CSV))
    if not rows:
        raise ValueError(f"no candidates loaded from {CANDIDATE_CSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = OUT_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    question_lines = [
        "# Expanded20 Route Evidence Annotation Sheet",
        "",
        "Purpose: annotate answer/relate regions for the strongest localized route candidates from expanded_localized_discovery_screen20.",
        "",
        "Labels:",
        "",
        "- `answer`: the smallest direct visual evidence region for the answer.",
        "- `relate`: broader support/context. If `answer` is already split across regions, use `relate` for the next useful context region.",
        "- Multiple `answer` shapes are allowed. The experiment will union all answer shapes.",
        "",
        "Priority:",
        "",
        "- Primary cases have support source nodes beating both nearest and random4 controls under `wrong_image`.",
        "- Backup cases are useful if a primary case turns out too awkward to annotate cleanly.",
        "",
    ]

    for idx, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        image_filename = row["image_filename"]
        src_image = _find_image(image_filename)
        dst_image = images_dir / image_filename
        if not dst_image.exists():
            shutil.copy2(src_image, dst_image)

        hints = CASE_HINTS.get(sample_id, {})
        manifest_row = {
            "selection_rank": str(idx),
            "sample_id": sample_id,
            "annotation_priority": row.get("annotation_priority", ""),
            "question_text": row.get("orig_question", ""),
            "answer_text": row.get("gold_answer", ""),
            "image_filename": image_filename,
            "local_image_path": str(dst_image),
            "source_local_image_path": str(src_image),
            "reasoning_operation": row.get("reasoning_operation", ""),
            "visual_structure": row.get("visual_structure", ""),
            "image_dependence": row.get("image_dependence", ""),
            "best_source_effect": row.get("best_source_effect", ""),
            "best_nearest_gap": row.get("best_nearest_gap", ""),
            "best_random4_gap": row.get("best_random4_gap", ""),
            "both_control_positive_rows": row.get("both_control_positive_rows", ""),
            "target_logit_drop": row.get("target_logit_drop", ""),
            "answer_hint": hints.get("answer_hint", ""),
            "relate_hint": hints.get("relate_hint", ""),
            "annotation_goal": "Mark answer and relate regions. Use multiple answer shapes when the answer evidence has multiple visible parts.",
        }
        manifest_rows.append(manifest_row)

        question_lines.extend(
            [
                f"## {idx}. {sample_id}",
                "",
                f"- priority: `{manifest_row['annotation_priority']}`",
                f"- image: `{image_filename}`",
                f"- question: {manifest_row['question_text']}",
                f"- answer: `{manifest_row['answer_text']}`",
                f"- visual_structure: `{manifest_row['visual_structure']}`",
                f"- reasoning_operation: `{manifest_row['reasoning_operation']}`",
                f"- best_source_effect: `{manifest_row['best_source_effect']}`",
                f"- nearest_gap: `{manifest_row['best_nearest_gap']}`",
                f"- random4_gap: `{manifest_row['best_random4_gap']}`",
                f"- answer hint: {manifest_row['answer_hint']}",
                f"- relate hint: {manifest_row['relate_hint']}",
                "",
            ]
        )

    fieldnames = list(manifest_rows[0].keys())
    _write_csv(OUT_DIR / "manifest.csv", manifest_rows, fieldnames)
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "QUESTION_SHEET.md").write_text("\n".join(question_lines) + "\n", encoding="utf-8")

    (OUT_DIR / "launch_labelme.ps1").write_text(
        "\n".join(
            [
                f'$imagesDir = "{images_dir}"',
                '$pythonExe = "E:\\code\\conda\\python.exe"',
                "",
                'Write-Host "Launching Labelme on $imagesDir"',
                'Start-Process -FilePath $pythonExe -ArgumentList @("-m", "labelme", $imagesDir)',
                "",
            ]
        ),
        encoding="utf-8",
    )

    quickstart = [
        "# Expanded20 Route Evidence Annotation Quickstart",
        "",
        "Open the question sheet while labeling:",
        "",
        f"- QUESTION_SHEET: `{OUT_DIR / 'QUESTION_SHEET.md'}`",
        f"- manifest: `{OUT_DIR / 'manifest.csv'}`",
        "",
        "Launch:",
        "",
        "```powershell",
        f"powershell -ExecutionPolicy Bypass -File {OUT_DIR / 'launch_labelme.ps1'}",
        "```",
        "",
        "Use only two labels: `answer` and `relate`.",
        "If one image has multiple answer regions, draw multiple `answer` polygons/boxes; the downstream mask exporter will union them.",
        "",
    ]
    (OUT_DIR / "ANNOTATION_QUICKSTART.md").write_text("\n".join(quickstart), encoding="utf-8")

    summary_lines = [
        "# Expanded20 Route Evidence Annotation Pack Summary",
        "",
        f"- total selected: `{len(manifest_rows)}`",
        f"- primary: `{sum(row['annotation_priority'] == 'primary' for row in manifest_rows)}`",
        f"- backup: `{sum(row['annotation_priority'] == 'backup' for row in manifest_rows)}`",
        "",
        "## Samples",
        "",
    ]
    for row in manifest_rows:
        summary_lines.append(
            f"- `{row['selection_rank']}` | `{row['annotation_priority']}` | `{row['sample_id']}` | "
            f"answer=`{row['answer_text']}` | image=`{row['image_filename']}`"
        )
    (OUT_DIR / "selection_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[done] out_dir={OUT_DIR}")
    print(f"[done] selected={len(manifest_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
