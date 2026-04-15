#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _format_prompt(template: str, question: str, extra_vars: dict[str, str]) -> str:
    q = (question or "").strip()
    fmt_vars = {
        "question": q,
        "instruction": q,
    }
    fmt_vars.update(extra_vars)
    has_placeholder = any("{" + k + "}" in template for k in fmt_vars)
    if has_placeholder:
        return template.format(**fmt_vars).strip()
    # Keep behavior forgiving for quick iteration.
    return f"{template.strip()} {q}".strip()


def _parse_gold_answer(notes: str) -> str:
    # Typical notes pattern:
    # qid=123;image_id=456;answer=bench
    if not notes:
        return ""
    for part in notes.split(";"):
        part = part.strip()
        if part.startswith("answer="):
            return part.split("=", 1)[1].strip()
    return ""


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build two or three manifests (A/B[/C]) from one base manifest by applying prompt templates. "
            "This supports controlled prompt comparison with identical sample ids."
        )
    )
    parser.add_argument("--base-manifest", required=True, help="Input manifest csv")
    parser.add_argument("--out-manifest-a", required=True, help="Output manifest for prompt A")
    parser.add_argument("--out-manifest-b", required=True, help="Output manifest for prompt B")
    parser.add_argument(
        "--out-manifest-c",
        default="",
        help="Optional output manifest for prompt C",
    )
    parser.add_argument(
        "--prompt-a-template",
        required=True,
        help='Prompt template A. You may use "{question}" or "{instruction}".',
    )
    parser.add_argument(
        "--prompt-b-template",
        required=True,
        help='Prompt template B. You may use "{question}" or "{instruction}".',
    )
    parser.add_argument(
        "--prompt-c-template",
        default="",
        help='Optional prompt template C. You may use "{question}" / "{instruction}" and shot vars.',
    )
    parser.add_argument(
        "--example-instruction",
        default="What color is the bus?",
        help='Optional shot variable "{example_instruction}" for templates.',
    )
    parser.add_argument(
        "--example-response",
        default="The answer is yellow.",
        help='Optional shot variable "{example_response}" for templates.',
    )
    parser.add_argument(
        "--copy-gold-from-notes",
        action="store_true",
        help='If set, parse "answer=..." from notes and write gold_answer column.',
    )
    args = parser.parse_args()

    base_manifest = Path(args.base_manifest).expanduser().resolve()
    out_a = Path(args.out_manifest_a).expanduser().resolve()
    out_b = Path(args.out_manifest_b).expanduser().resolve()
    out_c = Path(args.out_manifest_c).expanduser().resolve() if args.out_manifest_c else None

    with base_manifest.open("r", encoding="utf-8", newline="") as f:
        base_rows = list(csv.DictReader(f))

    if not base_rows:
        raise ValueError(f"empty manifest: {base_manifest}")

    # Preserve all original columns, then add experiment columns.
    base_fields = list(base_rows[0].keys())
    extra_fields = [
        "orig_question",
        "prompt_variant",
        "pair_id",
        "prompt_template",
        "gold_answer",
    ]
    fieldnames = base_fields + [x for x in extra_fields if x not in base_fields]

    rows_a: list[dict] = []
    rows_b: list[dict] = []
    rows_c: list[dict] = []
    extra_vars = {
        "example_instruction": args.example_instruction,
        "example_response": args.example_response,
        "original_response": args.example_response,
    }
    for row in base_rows:
        sid = (row.get("sample_id") or "").strip()
        question = (row.get("question") or "").strip()
        notes = (row.get("notes") or "").strip()
        if not sid:
            continue

        gold = _parse_gold_answer(notes) if args.copy_gold_from_notes else ""

        row_a = dict(row)
        row_a["orig_question"] = question
        row_a["question"] = _format_prompt(args.prompt_a_template, question, extra_vars)
        row_a["prompt_variant"] = "A"
        row_a["pair_id"] = sid
        row_a["prompt_template"] = args.prompt_a_template
        row_a["gold_answer"] = gold
        rows_a.append(row_a)

        row_b = dict(row)
        row_b["orig_question"] = question
        row_b["question"] = _format_prompt(args.prompt_b_template, question, extra_vars)
        row_b["prompt_variant"] = "B"
        row_b["pair_id"] = sid
        row_b["prompt_template"] = args.prompt_b_template
        row_b["gold_answer"] = gold
        rows_b.append(row_b)

        if out_c is not None:
            if not args.prompt_c_template:
                raise ValueError("--prompt-c-template is required when --out-manifest-c is set")
            row_c = dict(row)
            row_c["orig_question"] = question
            row_c["question"] = _format_prompt(args.prompt_c_template, question, extra_vars)
            row_c["prompt_variant"] = "C"
            row_c["pair_id"] = sid
            row_c["prompt_template"] = args.prompt_c_template
            row_c["gold_answer"] = gold
            rows_c.append(row_c)

    _write_csv(out_a, rows_a, fieldnames)
    _write_csv(out_b, rows_b, fieldnames)
    if out_c is not None:
        _write_csv(out_c, rows_c, fieldnames)

    print(f"[OK] base rows: {len(base_rows)}")
    print(f"[OK] A manifest: {out_a} rows={len(rows_a)}")
    print(f"[OK] B manifest: {out_b} rows={len(rows_b)}")
    if out_c is not None:
        print(f"[OK] C manifest: {out_c} rows={len(rows_c)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
