from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PROMPTS = {
    "B_direct": "{question} Reply with only one short sentence in exactly this format: The answer is <short answer>.",
    "C_step_only": "{question} Think step by step internally, then reply with only one short sentence in exactly this format: The answer is <short answer>.",
    "D_visual_only": "{question} Use visual evidence, then reply with only one short sentence in exactly this format: The answer is <short answer>.",
    "A_step_visual": "{question} Think step by step from visual evidence internally, then reply with only one short sentence in exactly this format: The answer is <short answer>.",
    "E_reply_step_only": "{question} Reply step by step, then give the final answer in exactly this format: The answer is <short answer>.",
    "F_reply_step_visual": "{question} Reply step by step from visual evidence, then give the final answer in exactly this format: The answer is <short answer>.",
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def majority_answer(answers: list[str]) -> str:
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build four OK-VQA factorization manifests.")
    parser.add_argument("--questions-json", required=True)
    parser.add_argument("--annotations-json", required=True)
    parser.add_argument("--image-root", required=True, help="Directory containing val2014/")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    questions_obj = read_json(Path(args.questions_json).expanduser().resolve())
    ann_obj = read_json(Path(args.annotations_json).expanduser().resolve())
    image_root = Path(args.image_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    ann_map: dict[int, list[str]] = {}
    for ann in ann_obj.get("annotations", []):
        qid = int(ann["question_id"])
        answers = [str(x.get("answer", "")).strip() for x in ann.get("answers", [])]
        answers = [x for x in answers if x]
        ann_map[qid] = answers

    base_rows: list[dict[str, str]] = []
    for q in questions_obj.get("questions", []):
        qid = int(q["question_id"])
        image_id = int(q["image_id"])
        question = str(q.get("question", "")).strip()
        if not question:
            continue
        answers = ann_map.get(qid, [])
        gold_answer = majority_answer(answers)
        image_filename = f"COCO_val2014_{image_id:012d}.jpg"
        image_path = image_root / "val2014" / image_filename
        base_rows.append(
            {
                "sample_id": f"okvqa_val_{qid}",
                "question_id": str(qid),
                "image_id": str(image_id),
                "orig_question": question,
                "image_path": str(image_path),
                "gold_answer": gold_answer,
                "notes": f"qid={qid};image_id={image_id};answer={gold_answer};sample_id=okvqa_val_{qid}",
            }
        )

    base_rows.sort(key=lambda r: int(r["question_id"]))

    fieldnames = [
        "sample_id",
        "question",
        "image_path",
        "gold_answer",
        "question_id",
        "image_id",
        "notes",
        "prompt_key",
        "orig_question",
    ]

    for prompt_key, template in PROMPTS.items():
        rows: list[dict[str, str]] = []
        for row in base_rows:
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "question": template.format(question=row["orig_question"]),
                    "image_path": row["image_path"],
                    "gold_answer": row["gold_answer"],
                    "question_id": row["question_id"],
                    "image_id": row["image_id"],
                    "notes": row["notes"],
                    "prompt_key": prompt_key,
                    "orig_question": row["orig_question"],
                }
            )
        write_csv(out_dir / f"{prompt_key}.csv", rows, fieldnames)
        print(f"[done] {prompt_key}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
