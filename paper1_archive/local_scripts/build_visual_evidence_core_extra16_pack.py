#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "annotation" / "okvqa_evidence_labelme_round4_core16_extra"

MANUAL_SAMPLE_IDS = [
    "okvqa_val_4214575",
    "okvqa_val_1407975",
    "okvqa_val_343215",
    "okvqa_val_136595",
    "okvqa_val_4700705",
    "okvqa_val_1927165",
    "okvqa_val_3794755",
    "okvqa_val_5735275",
    "okvqa_val_602025",
    "okvqa_val_4033335",
    "okvqa_val_3975875",
    "okvqa_val_2683965",
    "okvqa_val_1996815",
    "okvqa_val_3313665",
    "okvqa_val_1083155",
    "okvqa_val_2802115",
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
            item_id = r.get("item_id", "")
            if item_id:
                legacy_labels[item_id] = r

    out_rows = []
    sheet_lines = [
        "# Core Extra16 Annotation Question Sheet",
        "",
        "这批是从旧人工确认过的 `localized` 池里手动再挑出来的补充样本。",
        "它们整体上比 `diffuse_global` 更适合画 core region，但稳定性略低于第一批主包。",
        "",
    ]

    for idx, sample_id in enumerate(MANUAL_SAMPLE_IDS, start=1):
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
    _write_csv(OUT_DIR / "manifest.csv", out_rows, list(out_rows[0].keys()))
    (OUT_DIR / "manifest.json").write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "QUESTION_SHEET.md").write_text("\n".join(sheet_lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# Core Extra16 Summary",
        "",
        "- source: legacy manually labeled `localized` candidates",
        "",
        "## Samples",
        "",
    ]
    for row in out_rows:
        summary_lines.append(
            f"- `{row['selection_rank']}` | `{row['sample_id']}` | answer=`{row['answer_text']}` | "
            f"`legacy={row['legacy_visual_type_label']}` | `knowledge={row['legacy_knowledge_level_label']}`"
        )
    (OUT_DIR / "selection_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[done] out_dir={OUT_DIR}")
    print(f"[done] selected={len(out_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
