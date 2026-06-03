#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ANSWER_PREFIX = "the answer is "


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


def _condition_bucket(condition: str) -> str:
    if str(condition).startswith("random_control_"):
        return "random4"
    return str(condition)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize decoded generation under evidence-region masks.")
    parser.add_argument("--generation-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--random-iou-threshold", type=float, default=0.06)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Path(args.generation_csv).expanduser().resolve())

    work = df.copy()
    for col in ["generated_text", "predicted_answer", "error_message"]:
        work[col] = work[col].fillna("")
    work["condition_bucket"] = work["condition"].apply(_condition_bucket)
    work["format_prefix_ok"] = work["generated_text"].apply(_format_prefix_ok)
    work["extra_newline_spillover"] = work["generated_text"].apply(_extra_newline_spillover)
    work["empty_or_error"] = work.apply(_empty_or_error, axis=1)
    work["random_control_actual_iou"] = pd.to_numeric(work["random_control_actual_iou"], errors="coerce")
    work["random_control_valid_bool"] = work["random_control_valid"].astype(str).str.lower().eq("true")
    work["valid_random_for_summary"] = (
        work["condition"].astype(str).str.startswith("random_control_")
        & work["random_control_valid_bool"]
        & (work["random_control_actual_iou"] <= args.random_iou_threshold)
    )

    keys = ["sample_id", "run", "prompt_name"]
    clean = (
        work[work["condition"] == "clean"][keys + ["predicted_answer"]]
        .rename(columns={"predicted_answer": "clean_predicted_answer"})
        .drop_duplicates(keys)
    )
    work = work.merge(clean, on=keys, how="left")
    work["answer_changed_from_clean"] = (
        work["condition"] != "clean"
    ) & (work["predicted_answer"].fillna("") != work["clean_predicted_answer"].fillna(""))

    non_random = work[~work["condition"].astype(str).str.startswith("random_control_")].copy()
    random_valid = work[work["valid_random_for_summary"]].copy()

    condition_summary = (
        non_random.groupby(["prompt_name", "condition"], dropna=False)
        .agg(
            n_rows=("sample_id", "size"),
            n_samples=("sample_id", "nunique"),
            format_prefix_ok_count=("format_prefix_ok", "sum"),
            empty_or_error_count=("empty_or_error", "sum"),
            answer_changed_count=("answer_changed_from_clean", "sum"),
            unique_predictions=("predicted_answer", "nunique"),
        )
        .reset_index()
    )
    random_summary = (
        random_valid.groupby(["prompt_name"], dropna=False)
        .agg(
            n_rows=("sample_id", "size"),
            n_samples=("sample_id", "nunique"),
            format_prefix_ok_count=("format_prefix_ok", "sum"),
            empty_or_error_count=("empty_or_error", "sum"),
            answer_changed_count=("answer_changed_from_clean", "sum"),
            unique_predictions=("predicted_answer", "nunique"),
        )
        .reset_index()
    )
    random_summary.insert(1, "condition", "random4_valid")
    summary = pd.concat([condition_summary, random_summary], ignore_index=True)
    summary["answer_changed_rate"] = summary.apply(
        lambda row: float(row["answer_changed_count"]) / float(row["n_rows"]) if row["condition"] != "clean" and row["n_rows"] else 0.0,
        axis=1,
    )
    summary["format_prefix_ok_rate"] = summary.apply(
        lambda row: float(row["format_prefix_ok_count"]) / float(row["n_rows"]) if row["n_rows"] else 0.0,
        axis=1,
    )
    summary["empty_or_error_rate"] = summary.apply(
        lambda row: float(row["empty_or_error_count"]) / float(row["n_rows"]) if row["n_rows"] else 0.0,
        axis=1,
    )

    type_summary = (
        non_random[non_random["condition"].isin(["answer_mask", "union_mask"])]
        .groupby(["reasoning_operation", "condition"], dropna=False)
        .agg(
            n_rows=("sample_id", "size"),
            n_samples=("sample_id", "nunique"),
            answer_changed_count=("answer_changed_from_clean", "sum"),
            format_prefix_ok_count=("format_prefix_ok", "sum"),
            empty_or_error_count=("empty_or_error", "sum"),
        )
        .reset_index()
    )
    type_summary["answer_changed_rate"] = type_summary.apply(
        lambda row: float(row["answer_changed_count"]) / float(row["n_rows"]) if row["n_rows"] else 0.0,
        axis=1,
    )

    case_table = non_random[
        [
            "sample_id",
            "run",
            "prompt_name",
            "reasoning_operation",
            "condition",
            "clean_predicted_answer",
            "predicted_answer",
            "answer_changed_from_clean",
            "format_prefix_ok",
            "empty_or_error",
            "generated_text",
        ]
    ].sort_values(["sample_id", "run", "condition"])

    detail_path = out_dir / "region_mask_generation_detail.csv"
    summary_path = out_dir / "region_mask_generation_summary.csv"
    type_path = out_dir / "region_mask_generation_typed_summary.csv"
    case_path = out_dir / "region_mask_generation_case_table.csv"
    md_path = out_dir / "region_mask_generation_summary.md"
    work.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    type_summary.to_csv(type_path, index=False)
    case_table.to_csv(case_path, index=False)

    lines = [
        "# Region Mask Generation Summary",
        "",
        f"- generation csv: `{Path(args.generation_csv).expanduser().resolve()}`",
        f"- random IoU threshold: `{args.random_iou_threshold}`",
        "",
        "## Condition Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Typed Summary",
        "",
        type_summary.to_markdown(index=False),
        "",
        "## Notes",
        "",
        "- `generated_text` is reconstructed as `assistant_prefix + generated_continuation`, so prefix-format checks remain meaningful.",
        "- `random4_valid` includes only random controls marked valid and with actual IoU below the configured threshold.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {detail_path}")
    print(f"[done] wrote {summary_path}")
    print(f"[done] wrote {type_path}")
    print(f"[done] wrote {case_path}")
    print(f"[done] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
