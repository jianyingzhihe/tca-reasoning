#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "annotation" / "okvqa_evidence_labelme_round4_ultraeasy16_fresh"

FRESH_SAMPLE_IDS = [
    "okvqa_val_1521035",
    "okvqa_val_1994425",
    "okvqa_val_80655",
    "okvqa_val_667695",
    "okvqa_val_3918255",
    "okvqa_val_2131565",
    "okvqa_val_3064265",
    "okvqa_val_1083925",
    "okvqa_val_2496585",
    "okvqa_val_2373185",
    "okvqa_val_1172375",
    "okvqa_val_1740705",
    "okvqa_val_3326275",
    "okvqa_val_3265105",
    "okvqa_val_407295",
    "okvqa_val_4662635",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def main() -> int:
    manifest_rows = {r["sample_id"]: r for r in _read_csv(ROOT / "annotation" / "okvqa_type_label_round3_320" / "manifest.csv")}
    legacy_labels = {}
    for fp in (ROOT / "annotation" / "okvqa_type_label_round3_320_static_splits").glob("*_labels.csv"):
        for r in _read_csv(fp):
            if r.get("item_id"):
                legacy_labels[r["item_id"]] = r

    used = set()
    for dirname in [
        "okvqa_evidence_labelme_round4_core24_easy",
        "okvqa_evidence_labelme_round4_core16_extra",
        "okvqa_evidence_labelme_round4_mainline16",
    ]:
        p = ROOT / "annotation" / dirname / "manifest.csv"
        if p.exists():
            used |= {r["sample_id"] for r in _read_csv(p)}

    overlap = [sid for sid in FRESH_SAMPLE_IDS if sid in used]
    if overlap:
        raise ValueError(f"fresh pack still overlaps with previous packs: {overlap}")

    out_rows = []
    sheet_lines = [
        "# Ultra-Easy Fresh16 Question Sheet",
        "",
        "这批是重新生成的、和前两批不重叠的新样本。",
        "目标仍然是尽量挑更容易画紧凑 core 的题。",
        "",
    ]

    for idx, sample_id in enumerate(FRESH_SAMPLE_IDS, start=1):
        row = manifest_rows[sample_id]
        image_src = Path(row["local_image_path"])
        image_filename = row["image_filename"]
        image_dst = OUT_DIR / "images" / image_filename
        _copy_image(image_src, image_dst)
        legacy = legacy_labels.get(row["item_id"], {})

        out_rows.append(
            {
                "selection_rank": str(idx),
                "sample_id": sample_id,
                "question_text": row.get("display_question", ""),
                "answer_text": row.get("answer_text", ""),
                "image_filename": image_filename,
                "local_image_path": str(image_dst),
                "source_local_image_path": row.get("local_image_path", ""),
                "remote_image_path": row.get("remote_image_path", ""),
                "legacy_visual_type_label": legacy.get("visual_type_label", ""),
                "legacy_knowledge_level_label": legacy.get("knowledge_level_label", ""),
                "compact_core_tier": "ultraeasy_fresh",
                "annotation_goal": "Mark the most direct core answer region as 'answer', and the broader supporting evidence or context as 'relate'.",
            }
        )

        sheet_lines.extend(
            [
                f"## {idx}. {sample_id}",
                "",
                f"- image: `{image_filename}`",
                f"- question: {row.get('display_question','')}",
                f"- answer: `{row.get('answer_text','')}`",
                f"- legacy_visual_type_label: `{legacy.get('visual_type_label','')}`",
                f"- legacy_knowledge_level_label: `{legacy.get('knowledge_level_label','')}`",
                "",
            ]
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in out_rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    _write_csv(OUT_DIR / "manifest.csv", out_rows, fieldnames)
    (OUT_DIR / "manifest.json").write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "QUESTION_SHEET.md").write_text("\n".join(sheet_lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# UltraEasy Fresh16 Summary",
        "",
        "- zero overlap with previous annotation packs",
        "",
        "## Samples",
        "",
    ]
    for row in out_rows:
        summary_lines.append(
            f"- `{row['selection_rank']}` | `{row['sample_id']}` | answer=`{row['answer_text']}` | question={row['question_text']}"
        )
    (OUT_DIR / "selection_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[done] out_dir={OUT_DIR}")
    print(f"[done] selected={len(out_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
