#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = [
    "sample_id",
    "run",
    "prompt_name",
    "node_role",
    "node_source",
    "feature_layer",
    "feature_pos",
    "feature_id",
    "reasoning_operation",
    "visual_structure",
    "image_dependence",
]


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path).expanduser().resolve())


def _to_num(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [
        "delta_target_logit",
        "reference_clean_delta_target_logit",
        "target_rank",
        "target_vs_competitor_margin",
        "random_control_actual_iou",
    ]:
        if col in df.columns:
            df[col] = _num(df[col])
    return df


def _calibration(region: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    ref_keys = ["sample_id", "run", "node_role", "node_source", "feature_layer", "feature_pos", "feature_id"]
    ref_cols = ref_keys + ["reference_clean_delta_target_logit"]
    ref = manifest[ref_cols].copy()
    clean = region[region["condition"].eq("clean")].merge(ref, on=ref_keys, how="left")
    clean["abs_error"] = (clean["delta_target_logit"] - clean["reference_clean_delta_target_logit"]).abs()
    return (
        clean.groupby(["node_role", "node_source"], dropna=False)
        .agg(
            rows=("sample_id", "count"),
            samples=("sample_id", "nunique"),
            mean_abs_error=("abs_error", "mean"),
            max_abs_error=("abs_error", "max"),
            exact_match_rate=("abs_error", lambda s: float((s < 1e-9).mean())),
        )
        .reset_index()
    )


def _random_mask(df: pd.DataFrame, threshold: float) -> pd.Series:
    cond = df["condition"].astype(str).str.startswith("random_control_")
    iou = _num(df.get("random_control_actual_iou", pd.Series(index=df.index, dtype=float)))
    return cond & iou.notna() & (iou <= threshold)


def _prepare_behavior(behavior: pd.DataFrame, threshold: float) -> pd.DataFrame:
    random_rows = behavior[_random_mask(behavior, threshold)].copy()
    rand = (
        random_rows.groupby(["sample_id", "run"], as_index=False)
        .agg(
            valid_random_controls=("condition", "nunique"),
            random4_rank=("target_rank", "mean"),
            random4_margin=("target_vs_competitor_margin", "mean"),
            random4_max_iou=("random_control_actual_iou", "max"),
        )
        if not random_rows.empty
        else pd.DataFrame(columns=["sample_id", "run", "valid_random_controls", "random4_rank", "random4_margin", "random4_max_iou"])
    )
    fixed = behavior[behavior["condition"].isin(["clean", "answer_mask", "relate_mask", "union_mask"])]
    wide = fixed.pivot_table(
        index=["sample_id", "run", "prompt_name", "question", "reasoning_operation", "visual_structure", "image_dependence"],
        columns="condition",
        values=["target_rank", "target_vs_competitor_margin"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{condition}" for metric, condition in wide.columns]
    wide = wide.reset_index().merge(rand, on=["sample_id", "run"], how="left")
    wide["answer_rank_damage"] = wide["target_rank_answer_mask"] - wide["target_rank_clean"]
    wide["union_rank_damage"] = wide["target_rank_union_mask"] - wide["target_rank_clean"]
    wide["answer_margin_drop"] = wide["target_vs_competitor_margin_clean"] - wide["target_vs_competitor_margin_answer_mask"]
    wide["union_margin_drop"] = wide["target_vs_competitor_margin_clean"] - wide["target_vs_competitor_margin_union_mask"]
    wide["answer_rank_over_random4"] = wide["target_rank_answer_mask"] - wide["random4_rank"]
    wide["union_rank_over_random4"] = wide["target_rank_union_mask"] - wide["random4_rank"]
    wide["answer_margin_drop_over_random4"] = (
        wide["target_vs_competitor_margin_clean"] - wide["target_vs_competitor_margin_answer_mask"]
    ) - (wide["target_vs_competitor_margin_clean"] - wide["random4_margin"])
    return wide


def _prepare_region(region: pd.DataFrame, threshold: float) -> pd.DataFrame:
    random_rows = region[_random_mask(region, threshold)].copy()
    rand = (
        random_rows.groupby(KEYS, dropna=False, as_index=False)
        .agg(
            random4_delta=("delta_target_logit", "mean"),
            valid_random_controls=("condition", "nunique"),
            random4_max_iou=("random_control_actual_iou", "max"),
        )
        if not random_rows.empty
        else pd.DataFrame(columns=KEYS + ["random4_delta", "valid_random_controls", "random4_max_iou"])
    )
    fixed = region[region["condition"].isin(["clean", "answer_mask", "relate_mask", "union_mask"])]
    wide = fixed.pivot_table(index=KEYS, columns="condition", values="delta_target_logit", aggfunc="first")
    wide = wide.reset_index().merge(rand, on=KEYS, how="left")
    for condition in ["answer_mask", "relate_mask", "union_mask"]:
        wide[f"{condition}_weakening"] = np.where(
            wide["node_role"].eq("support"),
            wide[condition] - wide["clean"],
            wide["clean"] - wide[condition],
        )
    wide["random4_weakening"] = np.where(
        wide["node_role"].eq("support"),
        wide["random4_delta"] - wide["clean"],
        wide["clean"] - wide["random4_delta"],
    )
    return wide


def _support_pairs(region_wide: pd.DataFrame) -> pd.DataFrame:
    support = region_wide[region_wide["node_role"].eq("support")].copy()
    pair_cols = ["sample_id", "run", "prompt_name", "reasoning_operation", "visual_structure", "image_dependence"]
    source = support[support["node_source"].eq("source")].set_index(pair_cols)
    nearest = support[support["node_source"].eq("nearest_control")].set_index(pair_cols)
    rows = []
    for idx in source.index.intersection(nearest.index):
        s = source.loc[idx]
        n = nearest.loc[idx]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[0]
        if isinstance(n, pd.DataFrame):
            n = n.iloc[0]
        idx_tuple = idx if isinstance(idx, tuple) else (idx,)
        row = dict(zip(pair_cols, idx_tuple))
        for condition in ["answer_mask", "relate_mask", "union_mask", "random4"]:
            key = f"{condition}_weakening" if condition != "random4" else "random4_weakening"
            source_value = s.get(key)
            nearest_value = n.get(key)
            row[f"source_{condition}_weakening"] = source_value
            row[f"nearest_{condition}_weakening"] = nearest_value
            row[f"source_minus_nearest_{condition}_weakening"] = (
                source_value - nearest_value if pd.notna(source_value) and pd.notna(nearest_value) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _summary(region_wide: pd.DataFrame, behavior_wide: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(name: str, values: pd.Series) -> None:
        vals = _num(values).dropna()
        rows.append(
            {
                "metric": name,
                "n": int(len(vals)),
                "mean": float(vals.mean()) if len(vals) else math.nan,
                "median": float(vals.median()) if len(vals) else math.nan,
                "positive_rate": float((vals > 0).mean()) if len(vals) else math.nan,
            }
        )

    support_source = region_wide[region_wide["node_role"].eq("support") & region_wide["node_source"].eq("source")]
    for col in ["answer_mask_weakening", "union_mask_weakening", "random4_weakening"]:
        add(f"support_source_{col}", support_source[col])
    for col in [
        "source_minus_nearest_answer_mask_weakening",
        "source_minus_nearest_union_mask_weakening",
        "source_minus_nearest_random4_weakening",
    ]:
        if col in pairs.columns:
            add(col, pairs[col])
    for col in [
        "answer_rank_damage",
        "union_rank_damage",
        "answer_rank_over_random4",
        "union_rank_over_random4",
        "answer_margin_drop",
        "union_margin_drop",
    ]:
        add(f"behavior_{col}", behavior_wide[col])
    return pd.DataFrame(rows)


def _write_readout(
    path: Path,
    *,
    threshold: float,
    calibration: pd.DataFrame,
    summary: pd.DataFrame,
    behavior_wide: pd.DataFrame,
    pairs: pd.DataFrame,
) -> None:
    case_cols = [
        "sample_id",
        "run",
        "prompt_name",
        "valid_random_controls",
        "target_rank_clean",
        "target_rank_answer_mask",
        "target_rank_union_mask",
        "random4_rank",
        "answer_rank_over_random4",
        "union_rank_over_random4",
        "answer_margin_drop",
        "union_margin_drop",
    ]
    pair_cols = [
        "sample_id",
        "run",
        "prompt_name",
        "source_minus_nearest_answer_mask_weakening",
        "source_minus_nearest_union_mask_weakening",
        "source_minus_nearest_random4_weakening",
    ]
    lines = [
        "# Prefix-Fixed Region Evidence Summary",
        "",
        f"- random IoU threshold: `{threshold}`",
        "",
        "## Clean Calibration",
        "",
        *calibration.to_markdown(index=False).splitlines(),
        "",
        "## Primary Metric Summary",
        "",
        *summary.to_markdown(index=False).splitlines(),
        "",
        "## Behavior Cases",
        "",
        *behavior_wide[case_cols].sort_values(
            ["answer_rank_over_random4", "union_rank_over_random4"], ascending=False, na_position="last"
        ).to_markdown(index=False).splitlines(),
        "",
        "## Support Source-Nearest Pairs",
        "",
        *(pairs[pair_cols].to_markdown(index=False).splitlines() if not pairs.empty else ["No paired source-nearest support rows."]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize prefix-fixed evidence-region experiments with strict/relaxed random4 IoU thresholds.")
    parser.add_argument("--region-csv", required=True)
    parser.add_argument("--behavior-csv", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--random-iou-thresholds", default="0.05,0.06")
    args = parser.parse_args()

    region = _to_num(_read_csv(args.region_csv))
    behavior = _to_num(_read_csv(args.behavior_csv))
    manifest = _to_num(_read_csv(args.manifest_csv))
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    calibration = _calibration(region, manifest)
    calibration.to_csv(out_dir / "prefixfix_clean_calibration_summary.csv", index=False)

    for threshold_text in [x.strip() for x in args.random_iou_thresholds.split(",") if x.strip()]:
        threshold = float(threshold_text)
        label = str(threshold_text).replace(".", "p")
        behavior_wide = _prepare_behavior(behavior, threshold)
        region_wide = _prepare_region(region, threshold)
        pairs = _support_pairs(region_wide)
        summary = _summary(region_wide, behavior_wide, pairs)

        behavior_wide.to_csv(out_dir / f"behavior_case_table_iou{label}.csv", index=False)
        region_wide.to_csv(out_dir / f"region_route_weakening_iou{label}.csv", index=False)
        pairs.to_csv(out_dir / f"support_source_nearest_pairs_iou{label}.csv", index=False)
        summary.to_csv(out_dir / f"primary_metric_summary_iou{label}.csv", index=False)
        _write_readout(
            out_dir / f"PREFIXFIX_REGION_EVIDENCE_READOUT_iou{label}.md",
            threshold=threshold,
            calibration=calibration,
            summary=summary,
            behavior_wide=behavior_wide,
            pairs=pairs,
        )

    print(f"[done] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
