#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


IMAGE_ROOT = "/root/autodl-tmp/tca-reasoning/data/okvqa/images/val2014"
B_SUFFIX = "Reply with only one short sentence in exactly this format: The answer is <short answer>."
D_SUFFIX = "Use visual evidence, then reply with only one short sentence in exactly this format: The answer is <short answer>."


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _question_id(sample_id: str) -> str:
    return sample_id.replace("okvqa_val_", "")


def _image_id(image_filename: str) -> str:
    stem = Path(image_filename).stem
    return stem.replace("COCO_val2014_", "").lstrip("0") or "0"


def _image_path(image_filename: str) -> str:
    return f"{IMAGE_ROOT}/{image_filename}"


def _prompt(question: str, prompt_key: str) -> str:
    question = question.strip().rstrip()
    suffix = B_SUFFIX if prompt_key == "B_direct" else D_SUFFIX
    return f"{question} {suffix}"


def _manifest_row(row: dict[str, str], prompt_key: str) -> dict[str, str]:
    question = (row.get("question_text") or "").strip()
    image_filename = (row.get("image_filename") or "").strip()
    operation = (row.get("heuristic_reasoning_operation") or "").strip() or "visual_readout"
    return {
        "sample_id": (row.get("sample_id") or "").strip(),
        "question": _prompt(question, prompt_key),
        "image_path": _image_path(image_filename),
        "gold_answer": (row.get("answer_text") or "").strip(),
        "question_id": _question_id((row.get("sample_id") or "").strip()),
        "image_id": _image_id(image_filename),
        "notes": "round400_localized_discovery_screen",
        "question_type": "direct_visual",
        "visual_structure": "single_core",
        "image_dependence": "strong",
        "reasoning_operation": operation,
        "ambiguity_flag": "low",
        "extra_evidence": "none",
        "prompt_key": prompt_key,
        "orig_question": question,
    }


def _existing_eval_rows(path: Path, selected_ids: set[str]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows = [row for row in _read_csv(path) if (row.get("sample_id") or "").strip() in selected_ids]
    rows.sort(key=lambda row: row.get("sample_id", ""))
    return rows


def _bucket_row(sample_id: str, a_row: dict[str, str] | None, b_row: dict[str, str] | None) -> dict[str, str]:
    a_correct = (a_row or {}).get("correct", "")
    b_correct = (b_row or {}).get("correct", "")
    bucket = ""
    if a_correct != "" and b_correct != "":
        bucket = f"A{a_correct}_B{b_correct}"
    return {
        "sample_id": sample_id,
        "a_correct": a_correct,
        "b_correct": b_correct,
        "bucket": bucket,
        "a_pred": (a_row or {}).get("predicted_answer", ""),
        "a_pred_norm": (a_row or {}).get("predicted_answer_norm", ""),
        "a_gold": (a_row or {}).get("gold_answer", ""),
        "b_pred": (b_row or {}).get("predicted_answer", ""),
        "b_pred_norm": (b_row or {}).get("predicted_answer_norm", ""),
        "b_gold": (b_row or {}).get("gold_answer", ""),
        "same_pred": "1"
        if (a_row or {}).get("predicted_answer_norm", "") and (a_row or {}).get("predicted_answer_norm", "") == (b_row or {}).get("predicted_answer_norm", "")
        else "0",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the round400 localized discovery clean-screen package.")
    parser.add_argument("--pool-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", default="round400_localized_discovery_screen7")
    parser.add_argument("--existing-b-eval-csv", default="")
    parser.add_argument("--existing-d-eval-csv", default="")
    args = parser.parse_args()

    pool_csv = Path(args.pool_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(pool_csv)
    rows = [row for row in rows if (row.get("sample_id") or "").strip()]
    rows.sort(key=lambda row: (-float(row.get("discovery_score") or 0), int(row.get("rank") or 999999), row.get("sample_id", "")))
    selected_ids = {(row.get("sample_id") or "").strip() for row in rows}

    sample_rows = [
        {
            "sample_id": row["sample_id"],
            "question_text": row.get("question_text", ""),
            "answer_text": row.get("answer_text", ""),
            "image_filename": row.get("image_filename", ""),
            "heuristic_reasoning_operation": row.get("heuristic_reasoning_operation", ""),
            "legacy_visual_type_label": row.get("legacy_visual_type_label", ""),
            "legacy_knowledge_level_label": row.get("legacy_knowledge_level_label", ""),
            "discovery_score": row.get("discovery_score", ""),
            "priority": row.get("priority", ""),
            "bucket": row.get("bucket", ""),
            "source_round": row.get("source_round", ""),
        }
        for row in rows
    ]
    sample_fieldnames = list(sample_rows[0].keys()) if sample_rows else ["sample_id"]
    _write_csv(out_dir / f"{args.tag}_sample_ids.csv", sample_rows, sample_fieldnames)

    manifest_fieldnames = [
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
    b_manifest = [_manifest_row(row, "B_direct") for row in rows]
    d_manifest = [_manifest_row(row, "D_visual_only") for row in rows]
    _write_csv(out_dir / f"manifest_B_direct_{args.tag}.csv", b_manifest, manifest_fieldnames)
    _write_csv(out_dir / f"manifest_D_visual_only_{args.tag}.csv", d_manifest, manifest_fieldnames)

    existing_b = _existing_eval_rows(Path(args.existing_b_eval_csv).expanduser().resolve(), selected_ids) if args.existing_b_eval_csv else []
    existing_d = _existing_eval_rows(Path(args.existing_d_eval_csv).expanduser().resolve(), selected_ids) if args.existing_d_eval_csv else []
    if existing_b:
        _write_csv(out_dir / f"{args.tag}_B_direct_existing_eval_subset.csv", existing_b, list(existing_b[0].keys()))
    if existing_d:
        _write_csv(out_dir / f"{args.tag}_D_visual_only_existing_eval_subset.csv", existing_d, list(existing_d[0].keys()))
    b_by_id = {(row.get("sample_id") or "").strip(): row for row in existing_b}
    d_by_id = {(row.get("sample_id") or "").strip(): row for row in existing_d}
    bucket_rows = [_bucket_row(row["sample_id"], d_by_id.get(row["sample_id"]), b_by_id.get(row["sample_id"])) for row in sample_rows]
    _write_csv(
        out_dir / f"{args.tag}_existing_bucket_partial.csv",
        bucket_rows,
        ["sample_id", "a_correct", "b_correct", "bucket", "a_pred", "a_pred_norm", "a_gold", "b_pred", "b_pred_norm", "b_gold", "same_pred"],
    )

    existing_ids = set(b_by_id) & set(d_by_id)
    missing_eval_ids = sorted(selected_ids - existing_ids)
    lines = [
        f"# {args.tag}",
        "",
        "Purpose: cheap clean discovery screen before asking for more evidence-region annotation.",
        "",
        f"- candidates: `{len(rows)}`",
        f"- existing B/D eval coverage: `{len(existing_ids)}/{len(rows)}`",
        f"- missing eval ids: `{', '.join(missing_eval_ids) if missing_eval_ids else 'none'}`",
        "",
        "## Candidate Table",
        "",
        "| sample_id | op | question | answer | image | existing_eval |",
        "|---|---|---|---|---|---|",
    ]
    for row in sample_rows:
        sid = row["sample_id"]
        lines.append(
            f"| `{sid}` | `{row.get('heuristic_reasoning_operation', '')}` | "
            f"{row.get('question_text', '')} | `{row.get('answer_text', '')}` | "
            f"`{row.get('image_filename', '')}` | `{sid in existing_ids}` |"
        )
    lines.extend(
        [
            "",
            "## Next Run",
            "",
            "1. Run deterministic B/D eval for the full candidate set on the server.",
            "2. Build the A-slot `D_visual_only` vs B-slot `B_direct` bucket CSV from fresh evals.",
            "3. Run answer-aligned trace at `main64`.",
            "4. Build alignment-clean subset.",
            "5. Run clean source intervention plus nearest/random controls only if clean-core yield is usable.",
            "",
            "Decision rule: do not request new human masks until the clean intervention/control screen finds enough localized support-source candidates.",
        ]
    )
    (out_dir / f"{args.tag}_RUNPLAN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {out_dir}")
    print(f"[summary] candidates={len(rows)} existing_eval={len(existing_ids)} missing_eval={len(missing_eval_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
