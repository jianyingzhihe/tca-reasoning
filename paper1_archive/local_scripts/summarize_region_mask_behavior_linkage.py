#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MASK_CONDITIONS = ["answer_mask", "relate_mask", "union_mask", "random4_mean"]


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _valid_random_rows(df: pd.DataFrame) -> pd.DataFrame:
    random_rows = df[df["condition"].str.startswith("random_control_")].copy()
    if "random_control_valid" not in random_rows.columns:
        return random_rows
    valid = random_rows["random_control_valid"].astype(str).str.lower().isin({"true", "1", "yes"})
    return random_rows[valid].copy()


def _bootstrap_ci(values: np.ndarray, reducer, n_boot: int, seed: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        stats.append(reducer(sample))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3:
        return np.nan
    if np.nanstd(x[valid]) == 0 or np.nanstd(y[valid]) == 0:
        return np.nan
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def _bootstrap_corr_ci(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 4:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    stats = []
    idx = np.arange(len(x))
    for _ in range(n_boot):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        val = _corr(x[sample_idx], y[sample_idx])
        if np.isfinite(val):
            stats.append(val)
    if not stats:
        return np.nan, np.nan
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def _prepare_behavior(behavior: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    behavior = behavior.copy()
    behavior["target_rank"] = _num(behavior["target_rank"])
    behavior["target_vs_competitor_margin"] = _num(behavior["target_vs_competitor_margin"])

    random_rows = _valid_random_rows(behavior)
    if not random_rows.empty:
        random_mean = (
            random_rows.groupby(["sample_id", "run"], as_index=False)
            .agg(
                target_rank=("target_rank", "mean"),
                target_vs_competitor_margin=("target_vs_competitor_margin", "mean"),
                target_logit=("target_logit", lambda s: _num(s).mean()),
                target_prob=("target_prob", lambda s: _num(s).mean()),
            )
        )
        random_mean["condition"] = "random4_mean"
        keep_cols = [
            "sample_id",
            "run",
            "prompt_name",
            "question",
            "image_path",
            "reasoning_operation",
            "visual_structure",
            "image_dependence",
        ]
        meta = behavior.drop_duplicates(["sample_id", "run"])[keep_cols]
        random_mean = random_mean.merge(meta, on=["sample_id", "run"], how="left")
        behavior_aug = pd.concat([behavior, random_mean[behavior.columns.intersection(random_mean.columns)]], ignore_index=True)
    else:
        behavior_aug = behavior

    clean = behavior_aug[behavior_aug["condition"] == "clean"][
        ["sample_id", "run", "target_rank", "target_vs_competitor_margin", "top5_token_ids"]
    ].rename(
        columns={
            "target_rank": "clean_target_rank",
            "target_vs_competitor_margin": "clean_margin",
            "top5_token_ids": "clean_top5_token_ids",
        }
    )
    joined = behavior_aug.merge(clean, on=["sample_id", "run"], how="left")
    joined["rank_damage_vs_clean"] = joined["target_rank"] - joined["clean_target_rank"]
    joined["margin_drop_vs_clean"] = joined["clean_margin"] - joined["target_vs_competitor_margin"]
    if "top5_token_ids" in joined.columns:
        has_top5 = joined["top5_token_ids"].notna() & joined["clean_top5_token_ids"].notna()
        joined["top5_changed_from_clean"] = np.where(
            has_top5,
            joined["top5_token_ids"].astype(str) != joined["clean_top5_token_ids"].astype(str),
            np.nan,
        )
    else:
        joined["top5_changed_from_clean"] = np.nan

    condition_summary = (
        joined.groupby(["condition"], as_index=False)
        .agg(
            rows=("sample_id", "count"),
            samples=("sample_id", "nunique"),
            mean_rank_damage=("rank_damage_vs_clean", "mean"),
            median_rank_damage=("rank_damage_vs_clean", "median"),
            mean_margin_drop=("margin_drop_vs_clean", "mean"),
            median_margin_drop=("margin_drop_vs_clean", "median"),
            top5_changed_rate=("top5_changed_from_clean", "mean"),
        )
        .sort_values("condition")
    )
    return joined, condition_summary


def _prepare_region(region: pd.DataFrame) -> pd.DataFrame:
    region = region.copy()
    region["delta_target_logit"] = _num(region["delta_target_logit"])
    keys = [
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
    random_rows = _valid_random_rows(region)
    random_mean = pd.DataFrame()
    if not random_rows.empty:
        random_mean = random_rows.groupby(keys, dropna=False, as_index=False).agg(delta_target_logit=("delta_target_logit", "mean"))
        random_mean["condition"] = "random4_mean"
    region_aug = pd.concat([region, random_mean], ignore_index=True)
    clean = region_aug[region_aug["condition"] == "clean"][keys + ["delta_target_logit"]].rename(
        columns={"delta_target_logit": "clean_delta_target_logit"}
    )
    joined = region_aug.merge(clean, on=keys, how="left")
    support = joined["node_role"].eq("support")
    joined["path_weakening"] = np.where(
        support,
        joined["delta_target_logit"] - joined["clean_delta_target_logit"],
        joined["clean_delta_target_logit"] - joined["delta_target_logit"],
    )
    return joined


def _linkage(region_joined: pd.DataFrame, behavior_joined: pd.DataFrame, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    behavior_small = behavior_joined[
        [
            "sample_id",
            "run",
            "condition",
            "target_rank",
            "rank_damage_vs_clean",
            "margin_drop_vs_clean",
            "top5_changed_from_clean",
        ]
    ]
    merged = region_joined.merge(behavior_small, on=["sample_id", "run", "condition"], how="left")
    merged = merged[merged["condition"].isin(MASK_CONDITIONS)].copy()

    stats = []
    for (role, source, condition), g in merged.groupby(["node_role", "node_source", "condition"], dropna=False):
        weakening = _num(g["path_weakening"]).to_numpy(dtype=float)
        rank_damage = _num(g["rank_damage_vs_clean"]).to_numpy(dtype=float)
        margin_drop = _num(g["margin_drop_vs_clean"]).to_numpy(dtype=float)
        mean_lo, mean_hi = _bootstrap_ci(weakening, np.nanmean, n_boot, seed=17)
        rank_corr = _corr(weakening, rank_damage)
        margin_corr = _corr(weakening, margin_drop)
        rank_lo, rank_hi = _bootstrap_corr_ci(weakening, rank_damage, n_boot, seed=23)
        margin_lo, margin_hi = _bootstrap_corr_ci(weakening, margin_drop, n_boot, seed=29)
        stats.append(
            {
                "node_role": role,
                "node_source": source,
                "condition": condition,
                "rows": int(len(g)),
                "samples": int(g["sample_id"].nunique()),
                "mean_path_weakening": float(np.nanmean(weakening)) if len(weakening) else np.nan,
                "mean_path_weakening_ci_low": mean_lo,
                "mean_path_weakening_ci_high": mean_hi,
                "mean_rank_damage": float(np.nanmean(rank_damage)) if len(rank_damage) else np.nan,
                "mean_margin_drop": float(np.nanmean(margin_drop)) if len(margin_drop) else np.nan,
                "corr_weakening_rank_damage": rank_corr,
                "corr_weakening_rank_damage_ci_low": rank_lo,
                "corr_weakening_rank_damage_ci_high": rank_hi,
                "corr_weakening_margin_drop": margin_corr,
                "corr_weakening_margin_drop_ci_low": margin_lo,
                "corr_weakening_margin_drop_ci_high": margin_hi,
                "top5_changed_rate": float(g["top5_changed_from_clean"].mean()) if len(g) else np.nan,
            }
        )
    return merged, pd.DataFrame(stats)


def _case_shortlist(region_joined: pd.DataFrame, behavior_joined: pd.DataFrame) -> pd.DataFrame:
    behavior_wide = (
        behavior_joined[
            [
                "sample_id",
                "run",
                "prompt_name",
                "question",
                "image_path",
                "reasoning_operation",
                "visual_structure",
                "image_dependence",
                "condition",
                "target_rank",
                "rank_damage_vs_clean",
                "margin_drop_vs_clean",
                "top5_changed_from_clean",
            ]
        ]
        .pivot_table(
            index=[
                "sample_id",
                "run",
                "prompt_name",
                "question",
                "image_path",
                "reasoning_operation",
                "visual_structure",
                "image_dependence",
            ],
            columns="condition",
            values=["target_rank", "rank_damage_vs_clean", "margin_drop_vs_clean", "top5_changed_from_clean"],
            aggfunc="first",
        )
    )
    behavior_wide.columns = [f"{metric}_{condition}" for metric, condition in behavior_wide.columns]
    behavior_wide = behavior_wide.reset_index()

    route = region_joined[region_joined["condition"].isin(MASK_CONDITIONS)].copy()
    route["role_source"] = route["node_role"].astype(str) + "_" + route["node_source"].astype(str)
    route_wide = (
        route.pivot_table(
            index=["sample_id", "run"],
            columns=["role_source", "condition"],
            values="path_weakening",
            aggfunc="mean",
        )
    )
    route_wide.columns = [f"weakening_{role_source}_{condition}" for role_source, condition in route_wide.columns]
    route_wide = route_wide.reset_index()

    out = behavior_wide.merge(route_wide, on=["sample_id", "run"], how="left")
    out["answer_rank_damage_over_random4"] = out.get("rank_damage_vs_clean_answer_mask") - out.get(
        "rank_damage_vs_clean_random4_mean"
    )
    out["union_rank_damage_over_random4"] = out.get("rank_damage_vs_clean_union_mask") - out.get(
        "rank_damage_vs_clean_random4_mean"
    )
    out["answer_margin_drop_over_random4"] = out.get("margin_drop_vs_clean_answer_mask") - out.get(
        "margin_drop_vs_clean_random4_mean"
    )
    out["union_margin_drop_over_random4"] = out.get("margin_drop_vs_clean_union_mask") - out.get(
        "margin_drop_vs_clean_random4_mean"
    )

    profile_parts = []
    for _, row in out.iterrows():
        parts = []
        if row.get("answer_rank_damage_over_random4", 0) > 0:
            parts.append("answer-behavior-sensitive")
        if row.get("union_rank_damage_over_random4", 0) > 0:
            parts.append("union-behavior-sensitive")
        if row.get("weakening_support_source_answer_mask", 0) > 0:
            parts.append("support-source-answer-weakening")
        if row.get("weakening_suppressor_source_answer_mask", 0) > 0:
            parts.append("suppressor-source-answer-weakening")
        if row.get("weakening_suppressor_source_union_mask", 0) > 0:
            parts.append("suppressor-source-union-weakening")
        profile_parts.append(";".join(parts) if parts else "low-information")
    out["case_profile"] = profile_parts
    sort_cols = ["answer_rank_damage_over_random4", "union_rank_damage_over_random4", "answer_margin_drop_over_random4"]
    return out.sort_values(sort_cols, ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize behavior linkage for evidence-region mask experiments.")
    parser.add_argument("--region-csv", required=True)
    parser.add_argument("--behavior-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    region = pd.read_csv(args.region_csv)
    behavior = pd.read_csv(args.behavior_csv)

    behavior_joined, behavior_summary = _prepare_behavior(behavior)
    region_joined = _prepare_region(region)
    linkage_rows, linkage_summary = _linkage(region_joined, behavior_joined, args.n_boot)
    case_shortlist = _case_shortlist(region_joined, behavior_joined)

    behavior_path = out_dir / "region_mask_behavior_condition_summary.csv"
    joined_path = out_dir / "region_mask_path_behavior_joined.csv"
    linkage_path = out_dir / "region_mask_path_behavior_linkage.csv"
    behavior_joined_path = out_dir / "region_mask_behavior_per_condition.csv"
    case_path = out_dir / "region_mask_behavior_case_shortlist.csv"
    behavior_summary.to_csv(behavior_path, index=False)
    linkage_rows.to_csv(joined_path, index=False)
    linkage_summary.to_csv(linkage_path, index=False)
    behavior_joined.to_csv(behavior_joined_path, index=False)
    case_shortlist.to_csv(case_path, index=False)

    md_path = out_dir / "region_mask_behavior_linkage_readout.md"
    lines = [
        "# Region-Mask Behavior Linkage Readout",
        "",
        "This readout links evidence-region masking to target rank/margin behavior and then joins those behavior deltas to route weakening.",
        "",
        "Behavior by condition:",
        "",
        *behavior_summary.to_markdown(index=False).splitlines(),
        "",
        "Path-behavior linkage:",
        "",
        *linkage_summary.to_markdown(index=False).splitlines(),
        "",
        "Top case shortlist:",
        "",
        *case_shortlist[
            [
                "sample_id",
                "run",
                "prompt_name",
                "answer_rank_damage_over_random4",
                "union_rank_damage_over_random4",
                "answer_margin_drop_over_random4",
                "case_profile",
            ]
        ]
        .head(12)
        .to_markdown(index=False)
        .splitlines(),
        "",
        "Interpretation note: top-5 token change is a lightweight distributional proxy, not a natural-generation answer-change metric.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] wrote {behavior_path}")
    print(f"[done] wrote {joined_path}")
    print(f"[done] wrote {linkage_path}")
    print(f"[done] wrote {behavior_joined_path}")
    print(f"[done] wrote {case_path}")
    print(f"[done] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
