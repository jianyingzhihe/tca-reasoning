#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _summarize_one(df: pd.DataFrame, run_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["error_message"] = work["error_message"].fillna("")
    work = _to_numeric(
        work,
        [
            "target_logit",
            "target_prob",
            "target_rank",
            "target_in_top1",
            "target_in_top5",
            "competitor_logit",
            "competitor_prob",
            "target_vs_competitor_margin",
        ],
    )
    work["has_error"] = work["error_message"].astype(str).str.len() > 0

    clean = (
        work[work["condition"] == "clean"][
            ["sample_id", "target_rank", "target_prob", "target_vs_competitor_margin"]
        ]
        .rename(
            columns={
                "target_rank": "clean_target_rank",
                "target_prob": "clean_target_prob",
                "target_vs_competitor_margin": "clean_target_vs_competitor_margin",
            }
        )
        .drop_duplicates()
    )
    work = work.merge(clean, on="sample_id", how="left")
    work["rank_worsened_vs_clean"] = (work["condition"] != "clean") & (work["target_rank"] > work["clean_target_rank"])
    work["prob_drop_vs_clean"] = work["clean_target_prob"] - work["target_prob"]
    work["margin_drop_vs_clean"] = work["clean_target_vs_competitor_margin"] - work["target_vs_competitor_margin"]

    summary = (
        work.groupby("condition", dropna=False)
        .agg(
            n_rows=("sample_id", "size"),
            n_samples=("sample_id", "nunique"),
            mean_target_rank=("target_rank", "mean"),
            median_target_rank=("target_rank", "median"),
            mean_target_prob=("target_prob", "mean"),
            mean_margin=("target_vs_competitor_margin", "mean"),
            top1_count=("target_in_top1", "sum"),
            top5_count=("target_in_top5", "sum"),
            rank_worsened_count=("rank_worsened_vs_clean", "sum"),
            mean_prob_drop_vs_clean=("prob_drop_vs_clean", "mean"),
            mean_margin_drop_vs_clean=("margin_drop_vs_clean", "mean"),
            error_count=("has_error", "sum"),
        )
        .reset_index()
    )
    summary.insert(0, "run", run_name)
    return work, summary


def _write_markdown(out_path: Path, *, subset_name: str, summaries: list[pd.DataFrame]) -> None:
    lines = [
        "# Condition Target-Rank Summary",
        "",
        f"- subset: `{subset_name}`",
        "",
    ]
    for df in summaries:
        run_name = df["run"].iloc[0]
        lines.extend([f"## {run_name}", ""])
        for _, row in df.iterrows():
            n = int(row["n_samples"])
            lines.extend(
                [
                    f"- `{row['condition']}`",
                    f"  mean `target_rank = {float(row['mean_target_rank']):.4f}`",
                    f"  median `target_rank = {float(row['median_target_rank']):.4f}`",
                    f"  mean `target_prob = {float(row['mean_target_prob']):.6f}`",
                    f"  mean `target_vs_competitor_margin = {float(row['mean_margin']):.6f}`",
                    f"  `target top1 = {int(row['top1_count'])}/{n}`",
                    f"  `target top5 = {int(row['top5_count'])}/{n}`",
                    (
                        f"  rank worsened vs clean `{int(row['rank_worsened_count'])}/{n}`"
                        if row["condition"] != "clean"
                        else "  rank worsened vs clean `0/0`"
                    ),
                    (
                        f"  mean prob drop vs clean `= {float(row['mean_prob_drop_vs_clean']):.6f}`"
                        if row["condition"] != "clean"
                        else "  mean prob drop vs clean `= 0.000000`"
                    ),
                    (
                        f"  mean margin drop vs clean `= {float(row['mean_margin_drop_vs_clean']):.6f}`"
                        if row["condition"] != "clean"
                        else "  mean margin drop vs clean `= 0.000000`"
                    ),
                    f"  error `{int(row['error_count'])}/{n}`",
                ]
            )
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize target-rank condition evals for B vs D subsets.")
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
        out_dir / "condition_target_rank_detail.csv", index=False
    )
    pd.concat([b_summary, d_summary], ignore_index=True).to_csv(
        out_dir / "condition_target_rank_summary.csv", index=False
    )
    _write_markdown(
        out_dir / "condition_target_rank_summary.md",
        subset_name=args.subset_name,
        summaries=[b_summary, d_summary],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
