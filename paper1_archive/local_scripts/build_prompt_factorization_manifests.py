from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PROMPTS = {
    "B_direct": "{question} Reply with only one short sentence in exactly this format: The answer is <short answer>.",
    "C_step_only": "{question} Think step by step internally, then reply with only one short sentence in exactly this format: The answer is <short answer>.",
    "D_visual_only": "{question} Use visual evidence, then reply with only one short sentence in exactly this format: The answer is <short answer>.",
    "A_step_visual": "{question} Think step by step from visual evidence internally, then reply with only one short sentence in exactly this format: The answer is <short answer>.",
}

DEFAULT_SERVER_IMAGE_ROOT = "/root/autodl-tmp/tca-reasoning/data/okvqa/images/val2014"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def infer_question_id(sample_id: str, item_id: str) -> str:
    for text in (sample_id, item_id):
        m = re.search(r"okvqa_val_(\d+)", text or "")
        if m:
            return m.group(1)
    return ""


def infer_image_id(image_filename: str) -> str:
    m = re.search(r"COCO_val2014_(\d+)\.jpg$", image_filename or "")
    if not m:
        return ""
    return str(int(m.group(1)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build four factorized prompt manifests from a base pack csv."
    )
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for smoke runs.")
    parser.add_argument(
        "--server-image-root",
        default=DEFAULT_SERVER_IMAGE_ROOT,
        help="Root directory used to construct image_path when base csv only has image_filename.",
    )
    args = parser.parse_args()

    base_rows = read_csv(Path(args.base_csv).expanduser().resolve())
    if args.limit > 0:
        base_rows = base_rows[: args.limit]
    if not base_rows:
        raise SystemExit("empty base csv")

    out_dir = Path(args.out_dir).expanduser().resolve()
    fieldnames = [
        "sample_id",
        "question",
        "image_path",
        "gold_answer",
        "question_id",
        "image_id",
        "notes",
        "question_type",
        "visual_structure",
        "image_dependence",
        "reasoning_operation",
        "ambiguity_flag",
        "extra_evidence",
        "prompt_key",
        "orig_question",
    ]

    for prompt_key, template in PROMPTS.items():
        out_rows: list[dict[str, str]] = []
        for row in base_rows:
            q = (row.get("question_text") or row.get("question") or "").strip()
            image_path = (row.get("image_path") or "").strip()
            image_filename = (row.get("image_filename") or "").strip()
            if not image_path:
                if image_filename:
                    image_path = f"{args.server_image_root.rstrip('/')}/{image_filename}"
            sample_id = (row.get("sample_id") or "").strip()
            item_id = (row.get("item_id") or "").strip()
            question_id = (row.get("question_id") or "").strip() or infer_question_id(sample_id, item_id)
            image_id = (row.get("image_id") or "").strip() or infer_image_id(image_filename)
            out_rows.append(
                {
                    "sample_id": sample_id,
                    "question": template.format(question=q),
                    "image_path": image_path,
                    "gold_answer": (row.get("answer_text") or row.get("gold_answer") or "").strip(),
                    "question_id": question_id,
                    "image_id": image_id,
                    "notes": (row.get("notes") or "").strip(),
                    "question_type": (row.get("question_type") or "").strip(),
                    "visual_structure": (row.get("visual_structure") or "").strip(),
                    "image_dependence": (row.get("image_dependence") or "").strip(),
                    "reasoning_operation": (row.get("reasoning_operation") or "").strip(),
                    "ambiguity_flag": (row.get("ambiguity_flag") or "").strip(),
                    "extra_evidence": (row.get("extra_evidence") or "").strip(),
                    "prompt_key": prompt_key,
                    "orig_question": q,
                }
            )
        write_csv(out_dir / f"{prompt_key}.csv", out_rows, fieldnames)
        print(f"[done] {prompt_key}: {len(out_rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
