#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "annotation" / "okvqa_evidence_labelme_round4_core24_easy"

SELECTED_CSVS = [
    ROOT / "doc" / "5.13" / "bd_visual_only_mech_pack" / "visual_positive_strong12_selected.csv",
    ROOT / "doc" / "5.13" / "bd_visual_only_mech_pack" / "visual_positive_strong18_next_selected.csv",
]

IMAGE_MANIFEST_CSVS = [
    ROOT / "annotation" / "okvqa_type_label_round4_400_mobile_package" / "manifest.csv",
    ROOT / "annotation" / "okvqa_type_label_round3_320" / "manifest.csv",
]

ALLOWED_VISUAL_STRUCTURES = {"single_core", "split_cores", "core_plus_context"}
VISUAL_ORDER = {"single_core": 0, "split_cores": 1, "core_plus_context": 2}
REASONING_ORDER = {"symbol_text_reading": 0, "visual_readout": 1, "scene_inference": 2}
PRIORITY_ORDER = {"high": 0, "medium": 1, "new": 2, "low": 3}
EXCLUDED_BY_ANNOTATOR = {
    "okvqa_val_1235705",
    "okvqa_val_02033",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_selected_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in SELECTED_CSVS:
        subset_name = "strong12" if "strong12" in path.name else "strong18_next"
        for row in _read_csv(path):
            if (row.get("sample_id") or "").strip() in EXCLUDED_BY_ANNOTATOR:
                continue
            if (row.get("image_dependence") or "").strip() != "strong":
                continue
            visual_structure = (row.get("visual_structure") or "").strip()
            if visual_structure not in ALLOWED_VISUAL_STRUCTURES:
                continue
            row = dict(row)
            row["subset_name"] = subset_name
            rows.append(row)
    return rows


def _load_image_meta() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for path in IMAGE_MANIFEST_CSVS:
        for row in _read_csv(path):
            sample_id = (row.get("sample_id") or "").strip()
            if not sample_id or sample_id in out:
                continue
            row = dict(row)
            if not row.get("local_image_path"):
                image_url = (row.get("image_url") or "").strip()
                if image_url:
                    row["local_image_path"] = str(path.parent / image_url)
            out[sample_id] = row
    return out


def _sort_key(row: dict[str, str]) -> tuple[int, int, int, int, str]:
    return (
        REASONING_ORDER.get((row.get("reasoning_operation") or "").strip(), 99),
        VISUAL_ORDER.get((row.get("visual_structure") or "").strip(), 99),
        PRIORITY_ORDER.get((row.get("priority") or "").strip(), 99),
        0 if (row.get("subset_name") or "") == "strong12" else 1,
        row.get("sample_id", ""),
    )


def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def main() -> int:
    selected_rows = sorted(_load_selected_rows(), key=_sort_key)
    image_meta = _load_image_meta()

    manifest_rows: list[dict[str, str]] = []
    question_lines = [
        "# Core-Structured Easy Annotation Question Sheet",
        "",
        "这批样本优先挑的是更容易画 `core answer region` 的例子。",
        "建议先标 `answer`，再补 `relate`。",
        "",
    ]

    for idx, row in enumerate(selected_rows, start=1):
        sample_id = row["sample_id"]
        if sample_id not in image_meta:
            raise ValueError(f"missing image metadata for {sample_id}")
        image = image_meta[sample_id]
        image_filename = image.get("image_filename") or Path(image.get("local_image_path", "")).name
        dst_image = OUT_DIR / "images" / image_filename
        _copy_image(Path(image["local_image_path"]), dst_image)

        manifest_rows.append(
            {
                "selection_rank": str(idx),
                "sample_id": sample_id,
                "subset_name": row.get("subset_name", ""),
                "priority": row.get("priority", ""),
                "question_text": row.get("question_text", ""),
                "answer_text": row.get("answer_text", ""),
                "reasoning_operation": row.get("reasoning_operation", ""),
                "visual_structure": row.get("visual_structure", ""),
                "image_dependence": row.get("image_dependence", ""),
                "image_filename": image_filename,
                "local_image_path": str(dst_image),
                "source_local_image_path": image.get("local_image_path", ""),
                "remote_image_path": image.get("remote_image_path", ""),
                "annotation_goal": "Mark the most direct core answer region as 'answer', and the broader supporting evidence or context as 'relate'.",
            }
        )

        question_lines.extend(
            [
                f"## {idx}. {sample_id}",
                "",
                f"- image: `{image_filename}`",
                f"- subset: `{row.get('subset_name','')}`",
                f"- question: {row.get('question_text','')}",
                f"- answer: `{row.get('answer_text','')}`",
                f"- visual_structure: `{row.get('visual_structure','')}`",
                f"- reasoning_operation: `{row.get('reasoning_operation','')}`",
                "",
            ]
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUT_DIR / "manifest.csv", manifest_rows, list(manifest_rows[0].keys()))
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "QUESTION_SHEET.md").write_text("\n".join(question_lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# Core24 Easy Pack Summary",
        "",
        f"- total selected: `{len(manifest_rows)}`",
        "",
        "## Samples",
        "",
    ]
    for row in manifest_rows:
        summary_lines.append(
            f"- `{row['selection_rank']}` | `{row['sample_id']}` | `{row['visual_structure']}` | "
            f"`{row['reasoning_operation']}` | answer=`{row['answer_text']}`"
        )
    (OUT_DIR / "selection_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[done] out_dir={OUT_DIR}")
    print(f"[done] selected={len(manifest_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
