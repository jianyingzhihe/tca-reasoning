#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


TRUE_SET = {"1", "true", "t", "yes", "y", "ok"}
FALSE_SET = {"0", "false", "f", "no", "n"}


def _auto_find_col(existing: set[str], candidates: list[str]) -> str:
    for c in candidates:
        if c in existing:
            return c
    return ""


def _parse_bool(v: str) -> bool | None:
    s = (v or "").strip().lower()
    if s in TRUE_SET:
        return True
    if s in FALSE_SET:
        return False
    return None


def _normalize_answer(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_bool01(v: bool) -> str:
    return "1" if v else "0"


def _load_run(
    path: Path,
    id_col: str,
    correct_col: str,
    pred_col: str,
    gold_col: str,
    no_normalize: bool,
) -> dict[str, dict]:
    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"empty csv: {path}")

    cols = set(rows[0].keys())
    chosen_correct_col = correct_col or _auto_find_col(
        cols, ["correct", "is_correct", "status_correct", "a_correct", "b_correct"]
    )
    chosen_pred_col = pred_col or _auto_find_col(
        cols, ["predicted_answer", "pred_answer", "prediction", "model_answer", "answer"]
    )
    chosen_gold_col = gold_col or _auto_find_col(
        cols, ["gold_answer", "gt_answer", "label_answer", "target_answer"]
    )

    out: dict[str, dict] = {}
    for r in rows:
        sid = (r.get(id_col) or "").strip()
        if not sid:
            continue

        c_val: bool | None = None
        pred = ""
        gold = ""

        if chosen_correct_col:
            c_val = _parse_bool(r.get(chosen_correct_col, ""))
            if c_val is None:
                raise ValueError(
                    f"cannot parse boolean in {path} col={chosen_correct_col} sample_id={sid}"
                )
            if chosen_pred_col:
                pred = (r.get(chosen_pred_col) or "").strip()
            if chosen_gold_col:
                gold = (r.get(chosen_gold_col) or "").strip()
        else:
            if not chosen_pred_col or not chosen_gold_col:
                raise ValueError(
                    f"{path} needs either correct column or both pred/gold columns."
                )
            if chosen_pred_col == chosen_gold_col:
                raise ValueError(
                    f"{path} uses same column for pred/gold ({chosen_pred_col}). "
                    "Pass --pred-col/--gold-col explicitly."
                )
            pred = (r.get(chosen_pred_col) or "").strip()
            gold = (r.get(chosen_gold_col) or "").strip()
            if no_normalize:
                c_val = pred == gold
            else:
                c_val = _normalize_answer(pred) == _normalize_answer(gold)

        out[sid] = {
            "correct": bool(c_val),
            "pred": pred,
            "pred_norm": _normalize_answer(pred),
            "gold": gold,
        }
    return out


def _bucket(a_ok: bool, b_ok: bool) -> str:
    return f"A{int(a_ok)}_B{int(b_ok)}"


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build A/B bucket CSV (A1_B1/A1_B0/A0_B1/A0_B0) from two run result CSVs. "
            "Each run CSV needs sample_id and either a boolean correct column, "
            "or pred+gold answer columns."
        )
    )
    parser.add_argument("--run-a-csv", required=True, help="CSV for prompt/run A")
    parser.add_argument("--run-b-csv", required=True, help="CSV for prompt/run B")
    parser.add_argument("--out-bucket-csv", required=True, help="Output bucket CSV")
    parser.add_argument("--out-summary-txt", default="", help="Optional summary text file")
    parser.add_argument("--id-col", default="sample_id", help="ID column name")
    parser.add_argument(
        "--correct-col",
        default="",
        help="Boolean correctness column name; leave empty for auto-detect.",
    )
    parser.add_argument(
        "--pred-col",
        default="",
        help="Predicted answer column; used when no correctness column.",
    )
    parser.add_argument(
        "--gold-col",
        default="",
        help="Gold answer column; used when no correctness column.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="If deriving correctness from pred/gold, do strict exact match.",
    )
    parser.add_argument(
        "--require-same-pred",
        action="store_true",
        help="Keep only rows where normalized predicted answers are identical in A and B.",
    )
    args = parser.parse_args()

    run_a_csv = Path(args.run_a_csv).expanduser().resolve()
    run_b_csv = Path(args.run_b_csv).expanduser().resolve()
    out_bucket_csv = Path(args.out_bucket_csv).expanduser().resolve()
    out_summary_txt = Path(args.out_summary_txt).expanduser().resolve() if args.out_summary_txt else None

    a_map = _load_run(
        path=run_a_csv,
        id_col=args.id_col,
        correct_col=args.correct_col,
        pred_col=args.pred_col,
        gold_col=args.gold_col,
        no_normalize=args.no_normalize,
    )
    b_map = _load_run(
        path=run_b_csv,
        id_col=args.id_col,
        correct_col=args.correct_col,
        pred_col=args.pred_col,
        gold_col=args.gold_col,
        no_normalize=args.no_normalize,
    )

    common_ids = sorted(set(a_map.keys()) & set(b_map.keys()))
    if not common_ids:
        raise ValueError("no overlapping sample_id between A and B csv")

    rows: list[dict] = []
    counts = {"A1_B1": 0, "A1_B0": 0, "A0_B1": 0, "A0_B0": 0}
    a_ok_count = 0
    b_ok_count = 0
    dropped_diff_pred = 0
    for sid in common_ids:
        a_ok = bool(a_map[sid]["correct"])
        b_ok = bool(b_map[sid]["correct"])
        a_pred_norm = a_map[sid].get("pred_norm", "")
        b_pred_norm = b_map[sid].get("pred_norm", "")
        same_pred = bool(a_pred_norm and b_pred_norm and a_pred_norm == b_pred_norm)
        if args.require_same_pred and not same_pred:
            dropped_diff_pred += 1
            continue
        bkt = _bucket(a_ok, b_ok)
        counts[bkt] += 1
        a_ok_count += int(a_ok)
        b_ok_count += int(b_ok)
        rows.append(
            {
                "sample_id": sid,
                "a_correct": _to_bool01(a_ok),
                "b_correct": _to_bool01(b_ok),
                "bucket": bkt,
                "a_pred": a_map[sid]["pred"],
                "a_pred_norm": a_pred_norm,
                "a_gold": a_map[sid]["gold"],
                "b_pred": b_map[sid]["pred"],
                "b_pred_norm": b_pred_norm,
                "b_gold": b_map[sid]["gold"],
                "same_pred": _to_bool01(same_pred),
            }
        )

    if not rows:
        raise ValueError("no rows left after filtering; try disabling --require-same-pred")

    _write_csv(
        out_bucket_csv,
        rows,
        [
            "sample_id",
            "a_correct",
            "b_correct",
            "bucket",
            "a_pred",
            "a_pred_norm",
            "a_gold",
            "b_pred",
            "b_pred_norm",
            "b_gold",
            "same_pred",
        ],
    )

    total = len(rows)
    a_acc = a_ok_count / total
    b_acc = b_ok_count / total
    lines = [
        f"rows={total}",
        f"a_accuracy={a_acc:.6f}",
        f"b_accuracy={b_acc:.6f}",
        f"A1_B1={counts['A1_B1']}",
        f"A1_B0={counts['A1_B0']}",
        f"A0_B1={counts['A0_B1']}",
        f"A0_B0={counts['A0_B0']}",
        f"require_same_pred={int(args.require_same_pred)}",
        f"dropped_diff_pred={dropped_diff_pred}",
        f"only_in_a={len(set(a_map.keys()) - set(b_map.keys()))}",
        f"only_in_b={len(set(b_map.keys()) - set(a_map.keys()))}",
    ]
    summary_text = "\n".join(lines) + "\n"
    print(summary_text, end="")
    if out_summary_txt is not None:
        out_summary_txt.parent.mkdir(parents=True, exist_ok=True)
        out_summary_txt.write_text(summary_text, encoding="utf-8")
        print(f"[done] summary: {out_summary_txt}")
    print(f"[done] bucket csv: {out_bucket_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
