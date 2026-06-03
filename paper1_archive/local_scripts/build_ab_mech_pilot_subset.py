from __future__ import annotations

import argparse
import csv
import json
import re
import string
from collections import Counter
from pathlib import Path


PROMPT_MAP = {
    "A": "A_step_visual",
    "B": "B_direct",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _norm(text: str) -> str:
    s = (text or "").lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", "", s)
    return s


def _calc_score_dict(answers: list[str]) -> dict[str, float]:
    c = Counter(answers)
    return {k: min(v / 3.0, 1.0) for k, v in c.items()}


def _score_fulltext(sample_id: str, text: str, ann_map: dict[str, list[str]]) -> float:
    qid = sample_id.split("_")[-1].lstrip("0") or "0"
    gt = [_norm(x) for x in ann_map.get(qid, []) if _norm(x)]
    if not gt:
        return 0.0
    pred = _norm(text)
    score_dict = _calc_score_dict(gt)
    sc = 0.0
    for g, s in score_dict.items():
        if g and g in pred:
            sc += s
    return min(1.0, sc)


def _ambiguity_rank(v: str) -> int:
    order = {"low": 0, "medium": 1, "high": 2}
    return order.get((v or "").strip(), 3)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a small A/B mechanism pilot subset from labeled 395 rows.")
    parser.add_argument("--merged-csv", required=True)
    parser.add_argument("--manifest-a-csv", required=True)
    parser.add_argument("--manifest-b-csv", required=True)
    parser.add_argument("--annotations-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--a-better-target", type=int, default=12)
    parser.add_argument("--b-better-target", type=int, default=8)
    args = parser.parse_args()

    merged_rows = _read_csv(Path(args.merged_csv).expanduser().resolve())
    manifest_a_rows = _read_csv(Path(args.manifest_a_csv).expanduser().resolve())
    manifest_b_rows = _read_csv(Path(args.manifest_b_csv).expanduser().resolve())
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_obj = json.loads(Path(args.annotations_json).expanduser().resolve().read_text(encoding="utf-8"))
    ann_map: dict[str, list[str]] = {}
    for ann in ann_obj.get("annotations", []):
        qid = str(ann.get("question_id", "")).strip()
        vals = []
        for a in ann.get("answers", []):
            s = str(a.get("answer", "")).strip()
            if s:
                vals.append(s)
        if qid and vals:
            ann_map[qid] = vals

    candidates: list[dict[str, str | float | int]] = []
    for row in merged_rows:
        if (row.get("visual_structure") or "").strip() != "single_core":
            continue
        if (row.get("reasoning_operation") or "").strip() not in {"visual_readout", "world_fact_retrieval"}:
            continue
        sid = (row.get("sample_id") or "").strip()
        a_full = _score_fulltext(sid, row.get("A_step_visual__generated_text", ""), ann_map)
        b_full = _score_fulltext(sid, row.get("B_direct__generated_text", ""), ann_map)
        a_strict = int((row.get("A_step_visual__correct") or "0").strip() or "0")
        b_strict = int((row.get("B_direct__correct") or "0").strip() or "0")
        candidates.append(
            {
                "sample_id": sid,
                "question_text": row.get("question_text", ""),
                "answer_text": row.get("answer_text", ""),
                "question_type": row.get("question_type", ""),
                "visual_structure": row.get("visual_structure", ""),
                "image_dependence": row.get("image_dependence", ""),
                "reasoning_operation": row.get("reasoning_operation", ""),
                "ambiguity_flag": row.get("ambiguity_flag", ""),
                "extra_evidence": row.get("extra_evidence", ""),
                "A_fulltext_acc": a_full,
                "B_fulltext_acc": b_full,
                "A_strict": a_strict,
                "B_strict": b_strict,
                "delta_fulltext": a_full - b_full,
                "delta_strict": a_strict - b_strict,
            }
        )

    ranked_a = sorted(
        candidates,
        key=lambda x: (
            _ambiguity_rank(str(x["ambiguity_flag"])),
            -float(x["delta_fulltext"]),
            -int(x["delta_strict"]),
            str(x["sample_id"]),
        ),
    )
    ranked_b = sorted(
        candidates,
        key=lambda x: (
            _ambiguity_rank(str(x["ambiguity_flag"])),
            float(x["delta_fulltext"]),
            int(x["delta_strict"]),
            str(x["sample_id"]),
        ),
    )

    selected: list[dict[str, str | float | int]] = []
    used: set[str] = set()

    for row in ranked_a:
        if float(row["delta_fulltext"]) <= 0:
            continue
        sid = str(row["sample_id"])
        if sid in used:
            continue
        row = dict(row)
        row["pilot_side"] = "A_better"
        selected.append(row)
        used.add(sid)
        if sum(1 for x in selected if x["pilot_side"] == "A_better") >= args.a_better_target:
            break

    for row in ranked_b:
        if float(row["delta_fulltext"]) >= 0:
            continue
        sid = str(row["sample_id"])
        if sid in used:
            continue
        row = dict(row)
        row["pilot_side"] = "B_better"
        selected.append(row)
        used.add(sid)
        if sum(1 for x in selected if x["pilot_side"] == "B_better") >= args.b_better_target:
            break

    selected_ids = {str(r["sample_id"]) for r in selected}
    manifest_a_by_id = {(r.get("sample_id") or "").strip(): r for r in manifest_a_rows}
    manifest_b_by_id = {(r.get("sample_id") or "").strip(): r for r in manifest_b_rows}

    manifest_a_out = [manifest_a_by_id[sid] for sid in selected_ids if sid in manifest_a_by_id]
    manifest_b_out = [manifest_b_by_id[sid] for sid in selected_ids if sid in manifest_b_by_id]
    manifest_a_out.sort(key=lambda r: (r.get("sample_id") or ""))
    manifest_b_out.sort(key=lambda r: (r.get("sample_id") or ""))

    selected.sort(key=lambda r: (str(r["pilot_side"]), str(r["sample_id"])))

    selected_fieldnames = [
        "pilot_side",
        "sample_id",
        "question_text",
        "answer_text",
        "question_type",
        "visual_structure",
        "image_dependence",
        "reasoning_operation",
        "ambiguity_flag",
        "extra_evidence",
        "A_strict",
        "B_strict",
        "A_fulltext_acc",
        "B_fulltext_acc",
        "delta_fulltext",
        "delta_strict",
    ]
    _write_csv(out_dir / "selected_cases.csv", [{k: r.get(k, "") for k in selected_fieldnames} for r in selected], selected_fieldnames)

    if manifest_a_out:
        _write_csv(out_dir / "manifest_A.csv", manifest_a_out, list(manifest_a_out[0].keys()))
    if manifest_b_out:
        _write_csv(out_dir / "manifest_B.csv", manifest_b_out, list(manifest_b_out[0].keys()))

    eval_a_rows = []
    eval_b_rows = []
    for row in selected:
        sid = str(row["sample_id"])
        eval_a_rows.append(
            {
                "sample_id": sid,
                "correct": str(int(row["A_strict"])),
                "predicted_answer": next(r for r in merged_rows if (r.get("sample_id") or "").strip() == sid).get("A_step_visual__predicted_answer", ""),
                "gold_answer": next(r for r in merged_rows if (r.get("sample_id") or "").strip() == sid).get("answer_text", ""),
            }
        )
        eval_b_rows.append(
            {
                "sample_id": sid,
                "correct": str(int(row["B_strict"])),
                "predicted_answer": next(r for r in merged_rows if (r.get("sample_id") or "").strip() == sid).get("B_direct__predicted_answer", ""),
                "gold_answer": next(r for r in merged_rows if (r.get("sample_id") or "").strip() == sid).get("answer_text", ""),
            }
        )
    _write_csv(out_dir / "eval_A_subset.csv", eval_a_rows, ["sample_id", "correct", "predicted_answer", "gold_answer"])
    _write_csv(out_dir / "eval_B_subset.csv", eval_b_rows, ["sample_id", "correct", "predicted_answer", "gold_answer"])

    md_lines = [
        "# A/B Mechanism Pilot Subset",
        "",
        f"- selected total: `{len(selected)}`",
        f"- A_better: `{sum(1 for x in selected if x['pilot_side'] == 'A_better')}`",
        f"- B_better: `{sum(1 for x in selected if x['pilot_side'] == 'B_better')}`",
        "",
        "## Selected cases",
        "",
    ]
    for row in selected:
        md_lines.append(
            f"- `{row['pilot_side']}` | `{row['sample_id']}` | "
            f"`{row['reasoning_operation']}` | "
            f"`A_full={float(row['A_fulltext_acc']):.3f}` | "
            f"`B_full={float(row['B_fulltext_acc']):.3f}` | "
            f"`A_strict={row['A_strict']}` | "
            f"`B_strict={row['B_strict']}` | "
            f"{row['question_text']}"
        )
    (out_dir / "selection_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[done] selected={len(selected)}")
    print(f"[done] out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
