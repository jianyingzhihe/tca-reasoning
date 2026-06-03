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


def _load_selected_ids(path: Path) -> set[str]:
    df = pd.read_csv(path)
    if "sample_id" not in df.columns:
        raise ValueError("sample ids csv must include sample_id")
    return set(df["sample_id"].astype(str))


def _summarize_one(df: pd.DataFrame, run_name: str) -> tuple[pd.DataFrame, dict[str, object]]:
    work = df.copy()
    work["generated_text"] = work["generated_text"].fillna("")
    work["predicted_answer"] = work["predicted_answer"].fillna("")
    work["error_message"] = work["error_message"].fillna("")
    work["format_prefix_ok"] = work["generated_text"].apply(_format_prefix_ok)
    work["extra_newline_spillover"] = work["generated_text"].apply(_extra_newline_spillover)
    work["empty_or_error"] = work.apply(_empty_or_error, axis=1)

    summary = {
        "run": run_name,
        "n_samples": int(len(work)),
        "mean_vqa_score": float(work["vqa_score"].mean()) if len(work) else 0.0,
        "correct_count": int(work["correct"].fillna(0).astype(int).sum()),
        "strict_gold_correct_count": int(work["strict_gold_correct"].fillna(0).astype(int).sum()),
        "format_prefix_ok_count": int(work["format_prefix_ok"].astype(int).sum()),
        "extra_newline_spillover_count": int(work["extra_newline_spillover"].astype(int).sum()),
        "empty_or_error_count": int(work["empty_or_error"].astype(int).sum()),
    }

    detail_cols = [
        "sample_id",
        "gold_answer",
        "predicted_answer",
        "vqa_score",
        "correct",
        "strict_gold_correct",
        "format_prefix_ok",
        "extra_newline_spillover",
        "empty_or_error",
        "generated_text",
    ]
    return work[detail_cols].copy(), summary


def _write_markdown(
    out_path: Path,
    *,
    subset_name: str,
    subset_count: int,
    summaries: list[dict[str, object]],
    examples: pd.DataFrame,
) -> None:
    lines = [
        "# B/D Behavior Summary",
        "",
        f"- subset: `{subset_name}`",
        f"- sample count: `{subset_count}`",
        "",
        "## Aggregate",
        "",
    ]

    for row in summaries:
        n = int(row["n_samples"])
        lines.extend(
            [
                f"- `{row['run']}`",
                f"  mean `vqa_score = {row['mean_vqa_score']:.4f}`",
                f"  `correct = {row['correct_count']}/{n}`",
                f"  `strict_gold_correct = {row['strict_gold_correct_count']}/{n}`",
                f"  `format_prefix_ok = {row['format_prefix_ok_count']}/{n}`",
                f"  extra newline spillover: `{row['extra_newline_spillover_count']}/{n}`",
                f"  empty/error cases: `{row['empty_or_error_count']}/{n}`",
            ]
        )

    if not examples.empty:
        lines.extend(["", "## Notable Rows", ""])
        for _, row in examples.iterrows():
            lines.append(
                f"- `{row['sample_id']}` / `{row['run']}` / "
                f"`vqa={float(row['vqa_score']):.4f}` / "
                f"`spillover={int(bool(row['extra_newline_spillover']))}` / "
                f"`strict={int(row['strict_gold_correct'])}`"
            )
            lines.append(f"  gold: `{row['gold_answer']}`")
            lines.append(f"  pred: `{row['predicted_answer']}`")
            lines.append(f"  text: `{_normalize_text(row['generated_text'])}`")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize B_direct vs D_visual_only behavior on a selected sample subset."
    )
    parser.add_argument("--b-eval-csv", required=True)
    parser.add_argument("--d-eval-csv", required=True)
    parser.add_argument("--sample-ids-csv", required=True)
    parser.add_argument("--subset-name", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    _ensure_dir(out_dir)

    selected_ids = _load_selected_ids(Path(args.sample_ids_csv).expanduser().resolve())
    b_df = pd.read_csv(Path(args.b_eval_csv).expanduser().resolve())
    d_df = pd.read_csv(Path(args.d_eval_csv).expanduser().resolve())

    b_subset = b_df[b_df["sample_id"].astype(str).isin(selected_ids)].copy()
    d_subset = d_df[d_df["sample_id"].astype(str).isin(selected_ids)].copy()

    b_detail, b_summary = _summarize_one(b_subset, "B_direct")
    d_detail, d_summary = _summarize_one(d_subset, "D_visual_only")

    b_detail.insert(0, "run", "B_direct")
    d_detail.insert(0, "run", "D_visual_only")
    detail = pd.concat([b_detail, d_detail], ignore_index=True)
    detail.to_csv(out_dir / "behavior_detail.csv", index=False)

    summary_df = pd.DataFrame([b_summary, d_summary])
    summary_df.to_csv(out_dir / "behavior_summary.csv", index=False)

    examples = detail[
        detail["extra_newline_spillover"] | detail["empty_or_error"] | (detail["strict_gold_correct"] == 0)
    ].copy()
    examples = examples.sort_values(
        ["run", "extra_newline_spillover", "vqa_score", "sample_id"],
        ascending=[True, False, True, True],
    ).head(8)

    _write_markdown(
        out_dir / "behavior_summary.md",
        subset_name=args.subset_name,
        subset_count=len(selected_ids),
        summaries=[b_summary, d_summary],
        examples=examples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
