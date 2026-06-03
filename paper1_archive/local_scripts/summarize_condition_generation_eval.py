#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ANSWER_PREFIX = "the answer is "


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_text(value: object) -> str:
    return str(value or "").strip()


def _format_prefix_ok(text: str) -> bool:
    return _normalize_text(text).lower().startswith(ANSWER_PREFIX)


def _extra_newline_spillover(text: str) -> bool:
    stripped = _normalize_text(text)
    if not stripped:
        return False
    first_line, *rest = stripped.splitlines()
    return bool(rest) and bool("".join(rest).strip()) and _format_prefix_ok(first_line)


def _empty_or_error(row: pd.Series) -> bool:
    generated = _normalize_text(row.get("generated_text", ""))
    predicted = _normalize_text(row.get("predicted_answer", ""))
    err = _normalize_text(row.get("error_message", ""))
    return (not generated) or (not predicted) or bool(err)


def _summarize_one(df: pd.DataFrame, run_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["generated_text"] = work["generated_text"].fillna("")
    work["predicted_answer"] = work["predicted_answer"].fillna("")
    work["error_message"] = work["error_message"].fillna("")
    work["vqa_score"] = pd.to_numeric(work["vqa_score"], errors="coerce").fillna(0.0)
    work["correct"] = pd.to_numeric(work["correct"], errors="coerce").fillna(0).astype(int)
    work["strict_gold_correct"] = pd.to_numeric(work["strict_gold_correct"], errors="coerce").fillna(0).astype(int)
    work["format_prefix_ok"] = work["generated_text"].apply(_format_prefix_ok)
    work["extra_newline_spillover"] = work["generated_text"].apply(_extra_newline_spillover)
    work["empty_or_error"] = work.apply(_empty_or_error, axis=1)

    clean_pred = (
        work[work["condition"] == "clean"][["sample_id", "predicted_answer"]]
        .rename(columns={"predicted_answer": "clean_predicted_answer"})
        .drop_duplicates()
    )
    work = work.merge(clean_pred, on="sample_id", how="left")
    work["answer_changed_from_clean"] = (
        work["condition"] != "clean"
    ) & (work["predicted_answer"].fillna("") != work["clean_predicted_answer"].fillna(""))

    summary = (
        work.groupby("condition", dropna=False)
        .agg(
            n_rows=("sample_id", "size"),
            n_samples=("sample_id", "nunique"),
            mean_vqa_score=("vqa_score", "mean"),
            correct_count=("correct", "sum"),
            strict_gold_correct_count=("strict_gold_correct", "sum"),
            format_prefix_ok_count=("format_prefix_ok", "sum"),
            extra_newline_spillover_count=("extra_newline_spillover", "sum"),
            empty_or_error_count=("empty_or_error", "sum"),
            answer_changed_from_clean_count=("answer_changed_from_clean", "sum"),
        )
        .reset_index()
    )
    summary.insert(0, "run", run_name)
    return work, summary


def _write_markdown(
    out_path: Path,
    *,
    subset_name: str,
    summaries: list[pd.DataFrame],
) -> None:
    lines = [
        "# Condition Generation Summary",
        "",
        f"- subset: `{subset_name}`",
        "",
    ]

    for df in summaries:
        run_name = df["run"].iloc[0]
        lines.extend([f"## {run_name}", ""])
        for _, row in df.iterrows():
            n = int(row["n_samples"])
            changed = int(row.get("answer_changed_from_clean_count", 0))
            denom = n if row["condition"] != "clean" else 0
            lines.extend(
                [
                    f"- `{row['condition']}`",
                    f"  mean `vqa_score = {float(row['mean_vqa_score']):.4f}`",
                    f"  `correct = {int(row['correct_count'])}/{n}`",
                    f"  `strict_gold_correct = {int(row['strict_gold_correct_count'])}/{n}`",
                    f"  `format_prefix_ok = {int(row['format_prefix_ok_count'])}/{n}`",
                    f"  spillover `{int(row['extra_newline_spillover_count'])}/{n}`",
                    f"  empty/error `{int(row['empty_or_error_count'])}/{n}`",
                    (
                        f"  answer changed from clean `{changed}/{denom}`"
                        if row["condition"] != "clean"
                        else "  answer changed from clean `0/0`"
                    ),
                ]
            )
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize generation-side condition evals for B vs D subsets.")
    parser.add_argument("--b-csv", required=True)
    parser.add_argument("--d-csv", required=True)
    parser.add_argument("--subset-name", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    _ensure_dir(out_dir)

    b_df = pd.read_csv(Path(args.b_csv).expanduser().resolve())
    d_df = pd.read_csv(Path(args.d_csv).expanduser().resolve())

    b_detail, b_summary = _summarize_one(b_df, "B_direct")
    d_detail, d_summary = _summarize_one(d_df, "D_visual_only")

    pd.concat([b_detail.assign(run="B_direct"), d_detail.assign(run="D_visual_only")], ignore_index=True).to_csv(
        out_dir / "condition_generation_detail.csv", index=False
    )
    pd.concat([b_summary, d_summary], ignore_index=True).to_csv(
        out_dir / "condition_generation_summary.csv", index=False
    )
    _write_markdown(
        out_dir / "condition_generation_summary.md",
        subset_name=args.subset_name,
        summaries=[b_summary, d_summary],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
