#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import string
from collections import Counter
from pathlib import Path


def _normalize_no_punc_ws(text: str) -> str:
    s = (text or "").lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", "", s)
    return s


def _calc_score_dict(answers: list[str]) -> dict[str, float]:
    c = Counter(answers)
    return {k: min(v / 3.0, 1.0) for k, v in c.items()}


def _load_annotations(path: Path) -> dict[str, list[str]]:
    obj = json.load(path.open("r", encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for ann in obj.get("annotations", []):
        qid = str(ann.get("question_id", "")).strip()
        vals = []
        for a in ann.get("answers", []):
            s = str(a.get("answer", "")).strip()
            if s:
                vals.append(s)
        if qid and vals:
            out[qid] = vals
    return out


def _pick_prediction(row: dict, pred_cols: list[str]) -> str:
    for c in pred_cols:
        v = (row.get(c) or "").strip()
        if v:
            return v
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an OKVQA run csv with StaR-KVQA-style loose metrics: "
            "hit rate + substring-based ACC."
        )
    )
    parser.add_argument("--eval-csv", required=True, help="Run output csv")
    parser.add_argument("--annotations-json", required=True, help="OKVQA annotations json")
    parser.add_argument("--qid-col", default="question_id")
    parser.add_argument(
        "--pred-cols",
        default="predicted_answer,generated_text,prediction,answer",
        help="Comma-separated prediction column fallback order",
    )
    args = parser.parse_args()

    eval_csv = Path(args.eval_csv).expanduser().resolve()
    ann_json = Path(args.annotations_json).expanduser().resolve()

    ann_map = _load_annotations(ann_json)
    rows = list(csv.DictReader(eval_csv.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise ValueError(f"empty csv: {eval_csv}")

    pred_cols = [x.strip() for x in args.pred_cols.split(",") if x.strip()]

    total = 0
    hit = 0
    acc_sum = 0.0
    missing_qid = 0
    missing_pred = 0

    for r in rows:
        qid = (r.get(args.qid_col) or "").strip()
        if not qid or qid not in ann_map:
            missing_qid += 1
            continue

        pred = _pick_prediction(r, pred_cols)
        if not pred:
            missing_pred += 1
        pred_n = _normalize_no_punc_ws(pred)

        gt_raw = ann_map[qid]
        gt_n = [_normalize_no_punc_ws(x) for x in gt_raw if _normalize_no_punc_ws(x)]
        if not gt_n:
            missing_qid += 1
            continue

        # StaR-KVQA-style hit: any gt substring appears in prediction.
        if any(g in pred_n for g in set(gt_n)):
            hit += 1

        # StaR-KVQA-style acc: sum matched answer scores, clipped at 1.
        score_dict = _calc_score_dict(gt_n)
        sc = 0.0
        for g, s in score_dict.items():
            if g and g in pred_n:
                sc += s
        acc_sum += min(1.0, sc)
        total += 1

    hit_rate = (hit * 100.0 / total) if total else 0.0
    acc_rate = (acc_sum * 100.0 / total) if total else 0.0

    print(f"rows_in_csv={len(rows)}")
    print(f"rows_scored={total}")
    print(f"missing_qid_or_ann={missing_qid}")
    print(f"missing_prediction={missing_pred}")
    print(f"starkvqa_hit_rate={hit_rate:.2f}% ({hit}/{total if total else 1})")
    print(f"starkvqa_acc={acc_rate:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

