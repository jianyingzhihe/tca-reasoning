#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "annotation" / "okvqa_evidence_labelme_round4_ultraeasy20"

SOURCE_MANIFESTS = [
    ROOT / "annotation" / "okvqa_evidence_labelme_round4_core24_easy" / "manifest.csv",
    ROOT / "annotation" / "okvqa_evidence_labelme_round4_core16_extra" / "manifest.csv",
]

ULTRA_SAMPLE_IDS = [
    "okvqa_val_4043385",
    "okvqa_val_4502065",
    "okvqa_val_3959785",
    "okvqa_val_1729795",
    "okvqa_val_2847255",
    "okvqa_val_3658865",
    "okvqa_val_4739195",
    "okvqa_val_4938465",
    "okvqa_val_136595",
    "okvqa_val_1927165",
    "okvqa_val_5735275",
    "okvqa_val_2683965",
    "okvqa_val_343215",
    "okvqa_val_3794755",
    "okvqa_val_1996815",
    "okvqa_val_3313665",
    "okvqa_val_1083155",
    "okvqa_val_1058855",
    "okvqa_val_4157235",
    "okvqa_val_340155",
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


def _union_fieldnames(rows: list[dict[str, str]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    return ordered


def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def main() -> int:
    by_sample: dict[str, dict[str, str]] = {}
    for path in SOURCE_MANIFESTS:
        for row in _read_csv(path):
            sample_id = row.get("sample_id", "")
            if sample_id and sample_id not in by_sample:
                by_sample[sample_id] = row

    missing = [sample_id for sample_id in ULTRA_SAMPLE_IDS if sample_id not in by_sample]
    if missing:
        raise ValueError(f"missing sample ids in source manifests: {missing}")

    out_rows = []
    question_lines = [
        "# Ultra-Easy Compact-Core Question Sheet",
        "",
        "这批是更严格的 compact-core 候选。",
        "优先放：文字 / logo / 标志 / 品牌 / 局部对象 / 局部符号。",
        "尽量避开整体天气、整体色调、整体地形、整体场景氛围。",
        "",
    ]

    for idx, sample_id in enumerate(ULTRA_SAMPLE_IDS, start=1):
        row = by_sample[sample_id]
        image_src = Path(row["local_image_path"])
        image_filename = row["image_filename"]
        image_dst = OUT_DIR / "images" / image_filename
        _copy_image(image_src, image_dst)

        out_row = dict(row)
        out_row["selection_rank"] = str(idx)
        out_row["local_image_path"] = str(image_dst)
        out_row["compact_core_tier"] = "ultra_easy"
        out_rows.append(out_row)

        question_lines.extend(
            [
                f"## {idx}. {sample_id}",
                "",
                f"- image: `{image_filename}`",
                f"- question: {row.get('question_text','')}",
                f"- answer: `{row.get('answer_text','')}`",
                "",
            ]
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUT_DIR / "manifest.csv", out_rows, _union_fieldnames(out_rows))
    (OUT_DIR / "manifest.json").write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "QUESTION_SHEET.md").write_text("\n".join(question_lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# UltraEasy20 Summary",
        "",
        "- focus: compact-core examples only",
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
