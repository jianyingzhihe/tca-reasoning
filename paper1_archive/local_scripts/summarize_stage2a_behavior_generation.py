#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _norm_answer(value: object) -> str:
    text = str(value or "").lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(the|a|an) ", "", text)
    return text


def _bootstrap_ci(values: pd.Series, seed: int = 20260520, n_boot: int = 10000) -> tuple[float, float]:
    vals = _num(values).dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boots[i] = rng.choice(vals, size=len(vals), replace=True).mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def _metric_row(name: str, values: pd.Series) -> dict[str, object]:
    vals = _num(values).dropna()
    lo, hi = _bootstrap_ci(vals)
    return {
        "metric": name,
        "n": int(len(vals)),
        "mean": float(vals.mean()) if len(vals) else math.nan,
        "median": float(vals.median()) if len(vals) else math.nan,
        "positive_rate": float((vals > 0).mean()) if len(vals) else math.nan,
        "bootstrap_ci_low": lo,
        "bootstrap_ci_high": hi,
    }


def _prepare_behavior(behavior: pd.DataFrame, threshold: float) -> pd.DataFrame:
    behavior = behavior.copy()
    for col in ["target_rank", "target_vs_competitor_margin", "random_control_actual_iou"]:
        behavior[col] = _num(behavior[col])
    idx = ["sample_id", "run", "prompt_name", "reasoning_operation"]
    fixed = behavior[behavior["condition"].isin(["clean", "answer_mask", "relate_mask", "union_mask"])]
    wide = fixed.pivot_table(index=idx, columns="condition", values=["target_rank", "target_vs_competitor_margin"], aggfunc="first")
    wide.columns = [f"{metric}_{condition}" for metric, condition in wide.columns]
    wide = wide.reset_index()
    for condition in ["answer_mask", "relate_mask", "union_mask"]:
        wide[f"{condition}_rank_damage"] = wide[f"target_rank_{condition}"] - wide["target_rank_clean"]
        wide[f"{condition}_margin_drop"] = (
            wide["target_vs_competitor_margin_clean"] - wide[f"target_vs_competitor_margin_{condition}"]
        )

    rand = behavior[
        behavior["condition"].astype(str).str.startswith("random_control_")
        & behavior["random_control_actual_iou"].notna()
        & (behavior["random_control_actual_iou"] <= threshold)
    ]
    if not rand.empty:
        random_mean = (
            rand.groupby(idx, as_index=False)
            .agg(
                random4_rank=("target_rank", "mean"),
                random4_margin=("target_vs_competitor_margin", "mean"),
                valid_random_controls=("condition", "nunique"),
                random4_max_iou=("random_control_actual_iou", "max"),
            )
        )
        wide = wide.merge(random_mean, on=idx, how="left")
    else:
        wide["random4_rank"] = math.nan
        wide["random4_margin"] = math.nan
        wide["valid_random_controls"] = math.nan
        wide["random4_max_iou"] = math.nan
    for condition in ["answer_mask", "union_mask"]:
        wide[f"{condition}_rank_over_random4"] = wide[f"target_rank_{condition}"] - wide["random4_rank"]
        wide[f"{condition}_margin_drop_over_random4"] = (
            wide["target_vs_competitor_margin_clean"] - wide[f"target_vs_competitor_margin_{condition}"]
        ) - (wide["target_vs_competitor_margin_clean"] - wide["random4_margin"])
    return wide


def _prepare_generation(generation: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    generation = generation.copy()
    generation["random_control_actual_iou"] = _num(generation["random_control_actual_iou"])
    generation["has_error"] = generation["error_message"].fillna("").astype(str).str.len() > 0
    generation["is_empty"] = generation["predicted_answer"].fillna("").astype(str).str.strip().str.len() == 0
    clean = generation[generation["condition"].eq("clean")][["sample_id", "run", "predicted_answer"]].rename(
        columns={"predicted_answer": "clean_answer"}
    )
    merged = generation.merge(clean, on=["sample_id", "run"], how="left")
    merged["norm_predicted_answer"] = merged["predicted_answer"].map(_norm_answer)
    merged["norm_clean_answer"] = merged["clean_answer"].map(_norm_answer)
    merged["answer_changed_from_clean"] = merged["norm_predicted_answer"] != merged["norm_clean_answer"]

    fixed = merged[merged["condition"].isin(["answer_mask", "relate_mask", "union_mask"])]
    random = merged[
        merged["condition"].astype(str).str.startswith("random_control_")
        & merged["random_control_actual_iou"].notna()
        & (merged["random_control_actual_iou"] <= threshold)
    ]
    rows = []
    for condition, sub in fixed.groupby("condition"):
        rows.append(
            {
                "condition": condition,
                "rows": int(len(sub)),
                "sample_runs": int(sub[["sample_id", "run"]].drop_duplicates().shape[0]),
                "changed_rows": int(sub["answer_changed_from_clean"].sum()),
                "changed_rate": float(sub["answer_changed_from_clean"].mean()) if len(sub) else math.nan,
                "empty_rows": int(sub["is_empty"].sum()),
                "error_rows": int(sub["has_error"].sum()),
            }
        )
    if not random.empty:
        rows.append(
            {
                "condition": "random_controls_valid_rows",
                "rows": int(len(random)),
                "sample_runs": int(random[["sample_id", "run"]].drop_duplicates().shape[0]),
                "changed_rows": int(random["answer_changed_from_clean"].sum()),
                "changed_rate": float(random["answer_changed_from_clean"].mean()),
                "empty_rows": int(random["is_empty"].sum()),
                "error_rows": int(random["has_error"].sum()),
            }
        )
        random_any = random.groupby(["sample_id", "run"])["answer_changed_from_clean"].any()
        rows.append(
            {
                "condition": "random_controls_any_changed_by_sample_run",
                "rows": int(len(random_any)),
                "sample_runs": int(len(random_any)),
                "changed_rows": int(random_any.sum()),
                "changed_rate": float(random_any.mean()),
                "empty_rows": 0,
                "error_rows": 0,
            }
        )
    return merged, pd.DataFrame(rows)


def _correlation_table(linkage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = [
        ("answer_weakening", "answer_mask_rank_damage"),
        ("union_weakening", "union_mask_rank_damage"),
        ("answer_weakening", "answer_mask_margin_drop"),
        ("union_weakening", "union_mask_margin_drop"),
    ]
    for x, y in pairs:
        sub = linkage[[x, y]].dropna()
        rows.append(
            {
                "x": x,
                "y": y,
                "n": int(len(sub)),
                "pearson": float(sub[x].corr(sub[y])) if len(sub) > 1 else math.nan,
                "spearman": float(sub[x].corr(sub[y], method="spearman")) if len(sub) > 1 else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _write_readout(
    out_path: Path,
    *,
    threshold: float,
    behavior_summary: pd.DataFrame,
    generation_summary: pd.DataFrame,
    behavior_wide: pd.DataFrame,
    generation_cases: pd.DataFrame,
    linkage: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    behavior_case_cols = [
        "sample_id",
        "run",
        "target_rank_clean",
        "target_rank_answer_mask",
        "target_rank_union_mask",
        "answer_mask_rank_damage",
        "union_mask_rank_damage",
        "answer_mask_margin_drop",
        "union_mask_margin_drop",
        "valid_random_controls",
        "answer_mask_rank_over_random4",
        "union_mask_rank_over_random4",
    ]
    gen_case_cols = [
        "sample_id",
        "run",
        "condition",
        "clean_answer",
        "predicted_answer",
        "answer_changed_from_clean",
        "has_error",
        "is_empty",
    ]
    link_cols = [
        "sample_id",
        "run",
        "answer_weakening",
        "union_weakening",
        "answer_mask_rank_damage",
        "union_mask_rank_damage",
        "answer_mask_margin_drop",
        "union_mask_margin_drop",
        "answer_mask_changed",
        "union_mask_changed",
    ]
    lines = [
        "# Stage 2A Behavior And Generation Readout",
        "",
        f"- random IoU threshold: `{threshold}`",
        "",
        "## Behavior Metric Summary",
        "",
        *behavior_summary.to_markdown(index=False).splitlines(),
        "",
        "## Generation Summary",
        "",
        *generation_summary.to_markdown(index=False).splitlines(),
        "",
        "## Behavior Cases",
        "",
        *behavior_wide[behavior_case_cols].to_markdown(index=False).splitlines(),
        "",
        "## Generation Cases",
        "",
        *generation_cases[generation_cases["condition"].isin(["clean", "answer_mask", "union_mask"])][gen_case_cols]
        .to_markdown(index=False)
        .splitlines(),
        "",
        "## Route-Behavior Linkage",
        "",
        *linkage[link_cols].to_markdown(index=False).splitlines(),
        "",
        "## Correlations",
        "",
        *correlations.to_markdown(index=False).splitlines(),
        "",
        "## Interpretation Note",
        "",
        "- `rank_damage > 0` 表示 target token 排名变差。",
        "- `margin_drop > 0` 表示 target-vs-competitor margin 变差。",
        "- `answer_changed_from_clean=True` 表示归一化后的生成答案与 clean condition 不同。",
        "- 相关性只用于方向性描述，不做复杂显著性叙事。",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Stage 2A behavior/generation evidence and route-behavior linkage.")
    parser.add_argument("--behavior-csv", required=True)
    parser.add_argument("--generation-csv", required=True)
    parser.add_argument("--route-wide-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--random-iou-thresholds", default="0.05,0.20,1.0")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    behavior = pd.read_csv(Path(args.behavior_csv).expanduser().resolve())
    generation = pd.read_csv(Path(args.generation_csv).expanduser().resolve())
    route = pd.read_csv(Path(args.route_wide_csv).expanduser().resolve())
    route_support = (
        route[(route["node_role"].eq("support")) & (route["node_source"].eq("source"))]
        .groupby(["sample_id", "run"], as_index=False)
        .agg(
            answer_weakening=("answer_mask_weakening", "mean"),
            union_weakening=("union_mask_weakening", "mean"),
        )
    )

    for threshold_text in [x.strip() for x in args.random_iou_thresholds.split(",") if x.strip()]:
        threshold = float(threshold_text)
        label = threshold_text.replace(".", "p")
        behavior_wide = _prepare_behavior(behavior, threshold)
        behavior_summary = pd.DataFrame(
            [
                _metric_row("answer_mask_rank_damage", behavior_wide["answer_mask_rank_damage"]),
                _metric_row("union_mask_rank_damage", behavior_wide["union_mask_rank_damage"]),
                _metric_row("answer_mask_margin_drop", behavior_wide["answer_mask_margin_drop"]),
                _metric_row("union_mask_margin_drop", behavior_wide["union_mask_margin_drop"]),
                _metric_row("answer_mask_rank_over_random4", behavior_wide["answer_mask_rank_over_random4"]),
                _metric_row("union_mask_rank_over_random4", behavior_wide["union_mask_rank_over_random4"]),
                _metric_row("answer_mask_margin_drop_over_random4", behavior_wide["answer_mask_margin_drop_over_random4"]),
                _metric_row("union_mask_margin_drop_over_random4", behavior_wide["union_mask_margin_drop_over_random4"]),
            ]
        )
        generation_cases, generation_summary = _prepare_generation(generation, threshold)
        gen_flags = generation_cases[generation_cases["condition"].isin(["answer_mask", "union_mask"])].pivot_table(
            index=["sample_id", "run"],
            columns="condition",
            values="answer_changed_from_clean",
            aggfunc="first",
        )
        gen_flags = gen_flags.reset_index().rename(
            columns={
                "answer_mask": "answer_mask_changed",
                "union_mask": "union_mask_changed",
            }
        )
        linkage = behavior_wide.merge(route_support, on=["sample_id", "run"], how="left").merge(
            gen_flags, on=["sample_id", "run"], how="left"
        )
        correlations = _correlation_table(linkage)

        behavior_wide.to_csv(out_dir / f"behavior_wide_iou{label}.csv", index=False)
        generation_cases.to_csv(out_dir / f"generation_cases_iou{label}.csv", index=False)
        generation_summary.to_csv(out_dir / f"generation_summary_iou{label}.csv", index=False)
        behavior_summary.to_csv(out_dir / f"behavior_metric_summary_iou{label}.csv", index=False)
        linkage.to_csv(out_dir / f"route_behavior_linkage_iou{label}.csv", index=False)
        correlations.to_csv(out_dir / f"route_behavior_correlations_iou{label}.csv", index=False)
        _write_readout(
            out_dir / f"STAGE2A_BEHAVIOR_GENERATION_READOUT_iou{label}.md",
            threshold=threshold,
            behavior_summary=behavior_summary,
            generation_summary=generation_summary,
            behavior_wide=behavior_wide,
            generation_cases=generation_cases,
            linkage=linkage,
            correlations=correlations,
        )

    print(f"[done] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
