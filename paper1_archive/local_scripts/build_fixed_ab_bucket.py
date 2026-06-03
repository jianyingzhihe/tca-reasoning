from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def base_sample_id(sample_id: str) -> str:
    sid = (sample_id or "").strip()
    if sid.endswith("_A") or sid.endswith("_B"):
        return sid[:-2]
    return sid


def to_bool01(value: str) -> int:
    return 1 if str(value).strip() == "1" else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an A/B bucket csv while normalizing trailing _A/_B sample suffixes."
    )
    parser.add_argument("--eval-a-csv", required=True)
    parser.add_argument("--eval-b-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    eval_a_rows = read_csv(Path(args.eval_a_csv).expanduser().resolve())
    eval_b_rows = read_csv(Path(args.eval_b_csv).expanduser().resolve())

    eval_a = {base_sample_id(row.get("sample_id", "")): row for row in eval_a_rows}
    eval_b = {base_sample_id(row.get("sample_id", "")): row for row in eval_b_rows}

    shared_ids = sorted(set(eval_a) & set(eval_b))
    if not shared_ids:
        raise SystemExit("no overlapping base sample ids after normalization")

    out_rows: list[dict[str, str]] = []
    for sid in shared_ids:
        a_row = eval_a[sid]
        b_row = eval_b[sid]
        a_correct = to_bool01(a_row.get("correct", "0"))
        b_correct = to_bool01(b_row.get("correct", "0"))
        if a_correct and b_correct:
            bucket = "A1_B1"
        elif a_correct and not b_correct:
            bucket = "A1_B0"
        elif (not a_correct) and b_correct:
            bucket = "A0_B1"
        else:
            bucket = "A0_B0"
        out_rows.append(
            {
                "sample_id": sid,
                "a_sample_id": a_row.get("sample_id", ""),
                "b_sample_id": b_row.get("sample_id", ""),
                "question_id": a_row.get("question_id", "") or b_row.get("question_id", ""),
                "image_id": a_row.get("image_id", "") or b_row.get("image_id", ""),
                "gold_answer": a_row.get("gold_answer", "") or b_row.get("gold_answer", ""),
                "a_correct": str(a_correct),
                "b_correct": str(b_correct),
                "a_vqa_score": a_row.get("vqa_score", ""),
                "b_vqa_score": b_row.get("vqa_score", ""),
                "bucket": bucket,
                "a_predicted_answer": a_row.get("predicted_answer", ""),
                "b_predicted_answer": b_row.get("predicted_answer", ""),
            }
        )

    fieldnames = [
        "sample_id",
        "a_sample_id",
        "b_sample_id",
        "question_id",
        "image_id",
        "gold_answer",
        "a_correct",
        "b_correct",
        "a_vqa_score",
        "b_vqa_score",
        "bucket",
        "a_predicted_answer",
        "b_predicted_answer",
    ]
    write_csv(Path(args.out_csv).expanduser().resolve(), out_rows, fieldnames)
    print(f"[done] wrote {len(out_rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
