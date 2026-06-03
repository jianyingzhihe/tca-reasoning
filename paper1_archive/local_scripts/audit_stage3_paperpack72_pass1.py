#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
STAGE3 = ROOT / "doc" / "experiments" / "stage3"
PAPERPACK = STAGE3 / "paperpack72"
PACK = ROOT / "annotation" / "stage3_paperpack72_labelme" / "single_pass1_primary72_clean"

OUT_AUDIT = PAPERPACK / "paperpack72_pass1_audit.csv"
OUT_REVIEW = PAPERPACK / "paperpack72_pass1_review_or_exclude.csv"
OUT_SUMMARY = PAPERPACK / "paperpack72_pass1_audit_summary.json"

USER_FLAGGED_NON_IMAGE = {
    "okvqa_val_3508555": "question asks generic birthday-candle custom; image not needed",
    "okvqa_val_4536495": "question asks generic zebra habitat; image not needed",
    "okvqa_val_03189": "question text already identifies surfboard fin; answer likely generic function",
}

GENERIC_TEXT_PATTERNS = [
    "why do people",
    "what kind of environments do",
    "what type of degree do you need",
    "what is the purpose of",
    "what kind of climate",
    "where do these animals go",
    "how many calories",
    "how long does this animal",
    "how high can",
]

ALLOWED_LABELS = {"answer", "relate"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    manifest = read_csv(PACK / "manifest.csv")
    audit_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    label_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    flag_counter: Counter[str] = Counter()

    for row in manifest:
        sample_id = row["sample_id"]
        json_path = Path(row["labelme_json_expected"])
        flags: list[str] = []
        labels: list[str] = []
        shape_count = 0
        answer_shapes = 0
        relate_shapes = 0
        invalid_labels: list[str] = []
        if not json_path.exists():
            flags.append("missing_json")
        else:
            data = load_json(json_path)
            shapes = data.get("shapes", [])
            shape_count = len(shapes)
            labels = [str(shape.get("label", "")).strip() for shape in shapes]
            label_counter.update(labels)
            answer_shapes = sum(label == "answer" for label in labels)
            relate_shapes = sum(label == "relate" for label in labels)
            invalid_labels = sorted({label for label in labels if label not in ALLOWED_LABELS})
            if not shapes:
                flags.append("empty_shapes")
            if answer_shapes == 0:
                flags.append("missing_answer_label")
            if invalid_labels:
                flags.append("invalid_label")

        question = row.get("question_text", "")
        q_lower = question.lower()
        if sample_id in USER_FLAGGED_NON_IMAGE:
            flags.append("user_flag_non_image_dependent")
        if any(pattern in q_lower for pattern in GENERIC_TEXT_PATTERNS):
            flags.append("heuristic_possible_question_only_or_knowledge")

        if "user_flag_non_image_dependent" in flags:
            recommendation = "exclude_or_replace"
        elif "missing_json" in flags or "empty_shapes" in flags or "missing_answer_label" in flags or "invalid_label" in flags:
            recommendation = "fix_annotation"
        elif "heuristic_possible_question_only_or_knowledge" in flags:
            recommendation = "manual_review_image_dependence"
        else:
            recommendation = "keep_pending_export"

        out = {
            "paperpack72_rank": row.get("paperpack72_rank", ""),
            "sample_id": sample_id,
            "image_filename": row.get("image_filename", ""),
            "question_text": question,
            "answer_text": row.get("answer_text", ""),
            "proposed_reasoning_operation": row.get("proposed_reasoning_operation", ""),
            "shape_count": shape_count,
            "answer_shapes": answer_shapes,
            "relate_shapes": relate_shapes,
            "labels": "|".join(labels),
            "invalid_labels": "|".join(invalid_labels),
            "flags": "|".join(flags),
            "user_flag_reason": USER_FLAGGED_NON_IMAGE.get(sample_id, ""),
            "recommendation": recommendation,
            "json_path": str(json_path),
        }
        audit_rows.append(out)
        status_counter[recommendation] += 1
        for flag in flags:
            flag_counter[flag] += 1
        if recommendation != "keep_pending_export":
            review_rows.append(out)

    fields = [
        "paperpack72_rank",
        "sample_id",
        "image_filename",
        "question_text",
        "answer_text",
        "proposed_reasoning_operation",
        "shape_count",
        "answer_shapes",
        "relate_shapes",
        "labels",
        "invalid_labels",
        "flags",
        "user_flag_reason",
        "recommendation",
        "json_path",
    ]
    write_csv(OUT_AUDIT, audit_rows, fields)
    write_csv(OUT_REVIEW, review_rows, fields)
    summary = {
        "status": "completed",
        "pack": str(PACK),
        "rows": len(audit_rows),
        "label_counts": dict(label_counter),
        "recommendation_counts": dict(status_counter),
        "flag_counts": dict(flag_counter),
        "review_or_exclude_count": len(review_rows),
        "outputs": {
            "audit_csv": str(OUT_AUDIT),
            "review_csv": str(OUT_REVIEW),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
