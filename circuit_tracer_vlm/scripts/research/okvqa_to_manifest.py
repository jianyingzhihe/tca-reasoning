#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def load_questions(path: Path) -> dict[int, dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions", [])
    return {int(q["question_id"]): q for q in questions}


def load_answers(path: Path) -> dict[int, list[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    annos = data.get("annotations", [])
    out: dict[int, list[str]] = {}
    for a in annos:
        qid = int(a["question_id"])
        answers = [x.get("answer", "").strip() for x in a.get("answers", [])]
        answers = [x for x in answers if x]
        out[qid] = answers
    return out


def coco_image_path(image_root: Path, split: str, image_id: int) -> str:
    if split == "train":
        subdir = "train2014"
        prefix = "COCO_train2014_"
    else:
        subdir = "val2014"
        prefix = "COCO_val2014_"
    file_name = f"{prefix}{image_id:012d}.jpg"
    return str(image_root / subdir / file_name)


def majority_answer(answers: list[str]) -> str:
    if not answers:
        return ""
    counts: dict[str, int] = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert OK-VQA JSON to sample_manifest.csv format.")
    parser.add_argument("--questions", required=True, help="Path to OpenEnded_mscoco_*_questions.json")
    parser.add_argument("--annotations", required=True, help="Path to mscoco_*_annotations.json")
    parser.add_argument("--image-root", required=True, help="COCO image root, e.g. ~/data/okvqa/images")
    parser.add_argument("--split", choices=["train", "val"], required=True, help="Dataset split")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--limit", type=int, default=200, help="Max rows to export")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--id-prefix", default="okvqa", help="Prefix for sample_id")
    args = parser.parse_args()

    questions_path = Path(args.questions).expanduser().resolve()
    annotations_path = Path(args.annotations).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    q_map = load_questions(questions_path)
    a_map = load_answers(annotations_path)

    rows = []
    for qid, q in q_map.items():
        image_id = int(q["image_id"])
        question = q.get("question", "").strip()
        if not question:
            continue
        answers = a_map.get(qid, [])
        row = {
            "sample_id": "",
            "image_path": coco_image_path(image_root, args.split, image_id),
            "question": question,
            "split": args.split,
            "trace_source": "none",
            "question_type": "okvqa",
            "notes": f"qid={qid};image_id={image_id};answer={majority_answer(answers)}",
        }
        rows.append(row)

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.limit > 0:
        rows = rows[: args.limit]

    for i, row in enumerate(rows, start=1):
        row["sample_id"] = f"{args.id_prefix}_{args.split}_{i:05d}"

    fields = ["sample_id", "image_path", "question", "split", "trace_source", "question_type", "notes"]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[ok] wrote {len(rows)} rows -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
