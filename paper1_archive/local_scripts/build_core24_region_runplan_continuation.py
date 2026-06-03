#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _mean(series: pd.Series) -> float:
    vals = _num(series)
    return float(vals.mean()) if vals.notna().any() else np.nan


def _safe_rate(series: pd.Series) -> float:
    vals = series.dropna()
    return float(vals.mean()) if len(vals) else np.nan


def _prompt_to_run_name(prompt: str) -> str:
    if prompt == "B_direct":
        return "B_direct"
    if prompt == "D_visual_only":
        return "D_visual_only"
    return prompt


def _status_from_values(
    behavior_delta: float,
    source_minus_random: float,
    source_minus_nearest: float,
    union_minus_answer: float,
    min_rows_ok: bool,
) -> str:
    if not min_rows_ok:
        return "too_few_rows"
    if behavior_delta > 0 and source_minus_random > 0 and source_minus_nearest > 0 and union_minus_answer >= 0:
        return "typed_region_success_candidate"
    if behavior_delta > 0 and (source_minus_random > 0 or source_minus_nearest > 0):
        return "typed_partial_route_behavior"
    if behavior_delta > 0:
        return "behavior_only"
    return "not_supported"


def build_typed_behavior(case_df: pd.DataFrame) -> pd.DataFrame:
    df = case_df.copy()
    bool_cols = [
        "answer_rank_damage_over_random4",
        "union_rank_damage_over_random4",
        "answer_margin_drop_over_random4",
        "union_margin_drop_over_random4",
    ]
    for col in bool_cols:
        df[col] = _num(df[col])
    df["answer_behavior_sensitive"] = df["answer_rank_damage_over_random4"] > 0
    df["union_behavior_sensitive"] = df["union_rank_damage_over_random4"] > 0
    rows = []
    for typ, g in df.groupby("reasoning_operation", dropna=False):
        rows.append(
            {
                "reasoning_operation": typ,
                "sample_runs": len(g),
                "samples": g["sample_id"].nunique(),
                "mean_answer_rank_damage_over_random4": _mean(g["answer_rank_damage_over_random4"]),
                "median_answer_rank_damage_over_random4": float(_num(g["answer_rank_damage_over_random4"]).median()),
                "mean_union_rank_damage_over_random4": _mean(g["union_rank_damage_over_random4"]),
                "mean_answer_margin_drop_over_random4": _mean(g["answer_margin_drop_over_random4"]),
                "mean_union_margin_drop_over_random4": _mean(g["union_margin_drop_over_random4"]),
                "answer_behavior_sensitive_rate": _safe_rate(g["answer_behavior_sensitive"]),
                "union_behavior_sensitive_rate": _safe_rate(g["union_behavior_sensitive"]),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_answer_rank_damage_over_random4", ascending=False)


def build_typed_route(region_joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = region_joined.copy()
    df["path_weakening"] = _num(df["path_weakening"])
    condition_order = ["answer_mask", "relate_mask", "union_mask", "random4_mean"]
    route_rows = []
    for keys, g in df[df["condition"].isin(condition_order)].groupby(
        ["reasoning_operation", "node_role", "node_source", "condition"], dropna=False
    ):
        typ, role, source, condition = keys
        route_rows.append(
            {
                "reasoning_operation": typ,
                "node_role": role,
                "node_source": source,
                "condition": condition,
                "rows": len(g),
                "samples": g["sample_id"].nunique(),
                "mean_path_weakening": _mean(g["path_weakening"]),
                "median_path_weakening": float(_num(g["path_weakening"]).median()),
                "positive_weakening_rate": _safe_rate(g["path_weakening"] > 0),
            }
        )
    route_summary = pd.DataFrame(route_rows)

    status_rows = []
    for typ, g in df.groupby("reasoning_operation", dropna=False):
        support = g[g["node_role"].eq("support")]
        source = support[support["node_source"].eq("source")]
        nearest = support[support["node_source"].eq("nearest_control")]
        src_answer = _mean(source[source["condition"].eq("answer_mask")]["path_weakening"])
        src_random = _mean(source[source["condition"].eq("random4_mean")]["path_weakening"])
        src_union = _mean(source[source["condition"].eq("union_mask")]["path_weakening"])
        near_answer = _mean(nearest[nearest["condition"].eq("answer_mask")]["path_weakening"])
        behavior_g = g.drop_duplicates(["sample_id", "run"])
        behavior_delta = _mean(behavior_g["rank_damage_vs_clean"])
        source_minus_random = src_answer - src_random
        source_minus_nearest = src_answer - near_answer
        union_minus_answer = src_union - src_answer
        status_rows.append(
            {
                "reasoning_operation": typ,
                "support_source_answer_rows": int(len(source[source["condition"].eq("answer_mask")])),
                "support_nearest_answer_rows": int(len(nearest[nearest["condition"].eq("answer_mask")])),
                "mean_behavior_rank_damage_joined": behavior_delta,
                "support_source_answer_weakening": src_answer,
                "support_source_random4_weakening": src_random,
                "support_source_answer_minus_random4": source_minus_random,
                "support_nearest_answer_weakening": near_answer,
                "support_source_minus_nearest_answer": source_minus_nearest,
                "support_source_union_weakening": src_union,
                "support_union_minus_answer": union_minus_answer,
                "typed_status": _status_from_values(
                    behavior_delta,
                    source_minus_random,
                    source_minus_nearest,
                    union_minus_answer,
                    min_rows_ok=int(len(source[source["condition"].eq("answer_mask")])) >= 2,
                ),
            }
        )
    status = pd.DataFrame(status_rows).sort_values("typed_status")
    return route_summary, status


def build_wrong_region_bridge(case_df: pd.DataFrame, wrong_df: pd.DataFrame) -> pd.DataFrame:
    cases = case_df.copy()
    cases["run_name"] = cases["prompt_name"].map(_prompt_to_run_name)
    bridge = cases.merge(wrong_df, on=["sample_id", "run_name"], how="left", suffixes=("_region", "_wrong"))
    for col in [
        "answer_rank_damage_over_random4",
        "union_rank_damage_over_random4",
        "answer_margin_drop_over_random4",
        "support_route_weakening",
        "suppressor_route_weakening",
        "weakening_support_source_answer_mask",
        "weakening_suppressor_source_answer_mask",
        "weakening_suppressor_source_union_mask",
        "wrong_image_target_rank",
        "margin_drop_vs_clean",
    ]:
        if col in bridge.columns:
            bridge[col] = _num(bridge[col])
    profiles = []
    for _, row in bridge.iterrows():
        parts = []
        if pd.notna(row.get("support_route_weakening")) and row.get("support_route_weakening") > 0:
            parts.append("wrong-support-weakens")
        if pd.notna(row.get("suppressor_route_weakening")) and row.get("suppressor_route_weakening") > 0:
            parts.append("wrong-suppressor-weakens")
        if row.get("answer_rank_damage_over_random4", 0) > 0:
            parts.append("answer-region-behavior")
        if row.get("union_rank_damage_over_random4", 0) > 0:
            parts.append("union-region-behavior")
        if row.get("weakening_support_source_answer_mask", 0) > 0:
            parts.append("region-support-source")
        if row.get("weakening_suppressor_source_answer_mask", 0) > 0 or row.get("weakening_suppressor_source_union_mask", 0) > 0:
            parts.append("region-suppressor-source")
        if pd.isna(row.get("wrong_image_target_rank")):
            parts.append("missing-wrong-image-row")
        profiles.append(";".join(parts) if parts else "low-information")
    bridge["bridge_profile"] = profiles
    score = (
        _num(bridge.get("answer_rank_damage_over_random4", pd.Series(dtype=float))).fillna(0) / 1000.0
        + _num(bridge.get("union_rank_damage_over_random4", pd.Series(dtype=float))).fillna(0) / 1000.0
        + _num(bridge.get("support_route_weakening", pd.Series(dtype=float))).fillna(0)
        + _num(bridge.get("suppressor_route_weakening", pd.Series(dtype=float))).fillna(0)
        + _num(bridge.get("weakening_support_source_answer_mask", pd.Series(dtype=float))).fillna(0) * 5
        + _num(bridge.get("weakening_suppressor_source_answer_mask", pd.Series(dtype=float))).fillna(0) * 5
    )
    bridge["bridge_score"] = score
    keep = [
        "sample_id",
        "run",
        "run_name",
        "prompt_name",
        "reasoning_operation",
        "visual_structure",
        "answer_rank_damage_over_random4",
        "union_rank_damage_over_random4",
        "answer_margin_drop_over_random4",
        "wrong_image_target_rank",
        "margin_drop_vs_clean",
        "support_route_weakening",
        "suppressor_route_weakening",
        "weakening_support_source_answer_mask",
        "weakening_support_source_union_mask",
        "weakening_suppressor_source_answer_mask",
        "weakening_suppressor_source_union_mask",
        "bridge_profile",
        "bridge_score",
        "question",
        "image_path",
    ]
    keep = [c for c in keep if c in bridge.columns]
    return bridge[keep].sort_values("bridge_score", ascending=False)


def build_next_candidates(
    wrong_top: pd.DataFrame,
    selected_meta: pd.DataFrame,
    current_manifest: pd.DataFrame,
    current_case: pd.DataFrame,
) -> pd.DataFrame:
    current_samples = set(current_manifest["sample_id"].dropna().astype(str))
    current_run_samples = set(zip(current_case["sample_id"].astype(str), current_case["prompt_name"].map(_prompt_to_run_name)))
    out = wrong_top.copy()
    meta_cols = [
        "sample_id",
        "question_text",
        "answer_text",
        "reasoning_operation",
        "visual_structure",
        "image_dependence",
        "priority",
    ]
    meta = selected_meta[[c for c in meta_cols if c in selected_meta.columns]].drop_duplicates("sample_id")
    out = out.merge(meta, on="sample_id", how="left")
    out["already_in_current_manifest"] = out["sample_id"].isin(current_samples)
    out["already_has_region_behavior_run"] = [
        (sid, run) in current_run_samples for sid, run in zip(out["sample_id"].astype(str), out["run_name"].astype(str))
    ]
    out["candidate_reason"] = np.where(
        out["already_has_region_behavior_run"],
        "already_analyzed",
        np.where(
            out["already_in_current_manifest"],
            "manifest_present_but_no_region_run_or_unusable",
            "candidate_for_future_annotation_or_region_run",
        ),
    )
    out["localized_preference"] = out["visual_structure"].isin(["single_core", "split_cores"]) & out[
        "reasoning_operation"
    ].isin(["symbol_text_reading", "visual_readout"])
    out["next_score"] = (
        _num(out["figure_score"]).fillna(0)
        + out["localized_preference"].astype(float) * 25.0
        - out["already_has_region_behavior_run"].astype(float) * 50.0
    )
    cols = [
        "sample_id",
        "run_name",
        "next_score",
        "candidate_reason",
        "localized_preference",
        "question_text",
        "answer_text",
        "reasoning_operation",
        "visual_structure",
        "image_dependence",
        "figure_score",
        "wrong_image_target_rank",
        "margin_drop_vs_clean",
        "support_route_weakening",
        "suppressor_route_weakening",
        "already_in_current_manifest",
        "already_has_region_behavior_run",
    ]
    return out[[c for c in cols if c in out.columns]].sort_values("next_score", ascending=False)


def write_markdown(
    out_path: Path,
    typed_behavior: pd.DataFrame,
    typed_status: pd.DataFrame,
    bridge: pd.DataFrame,
    next_candidates: pd.DataFrame,
) -> None:
    lines = [
        "# Core24 Run-Plan Continuation Readout",
        "",
        "This continues the region-evidence run plan after the offset sweep and behavior-linkage supplement.",
        "",
        "## Typed Behavior Summary",
        "",
        *typed_behavior.to_markdown(index=False).splitlines(),
        "",
        "## Typed Support-Route Criteria",
        "",
        *typed_status.to_markdown(index=False).splitlines(),
        "",
        "Read: the behavior side is strongest in `visual_readout` and `symbol_text_reading`, but the support-source route criteria are still weak or incomplete under region masks.",
        "",
        "## Wrong-Image vs Region Bridge",
        "",
        *bridge.head(12).to_markdown(index=False).splitlines(),
        "",
        "Read: this bridge tells us whether a case that is strong under whole-image corruption is also strong under evidence-region masking. Missing wrong-image rows or zero region-route rows should be treated as non-evidence, not negative proof.",
        "",
        "## Next Candidate Queue",
        "",
        *next_candidates.head(16).to_markdown(index=False).splitlines(),
        "",
        "Recommendation: do not request broad new labeling yet. If we do ask for labels, use the candidate queue and prefer localized `symbol_text_reading` / `visual_readout` rows with nonzero clean source intervention and available controls.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build typed summaries, wrong-image bridge, and next candidate queue for core24 region run plan.")
    parser.add_argument("--region-joined-csv", required=True)
    parser.add_argument("--behavior-case-csv", required=True)
    parser.add_argument("--wrong-image-detail-csv", required=True)
    parser.add_argument("--wrong-image-top-csv", required=True)
    parser.add_argument("--current-manifest-csv", required=True)
    parser.add_argument("--selected-meta-csv", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    region = pd.read_csv(args.region_joined_csv)
    cases = pd.read_csv(args.behavior_case_csv)
    wrong_detail = pd.read_csv(args.wrong_image_detail_csv)
    wrong_top = pd.read_csv(args.wrong_image_top_csv)
    manifest = pd.read_csv(args.current_manifest_csv)
    selected_meta = pd.concat([pd.read_csv(p) for p in args.selected_meta_csv], ignore_index=True)

    typed_behavior = build_typed_behavior(cases)
    route_summary, typed_status = build_typed_route(region)
    bridge = build_wrong_region_bridge(cases, wrong_detail)
    next_candidates = build_next_candidates(wrong_top, selected_meta, manifest, cases)

    typed_behavior.to_csv(out_dir / "core24_typed_behavior_summary.csv", index=False)
    route_summary.to_csv(out_dir / "core24_typed_route_summary.csv", index=False)
    typed_status.to_csv(out_dir / "core24_typed_success_status.csv", index=False)
    bridge.to_csv(out_dir / "core24_wrong_image_region_bridge.csv", index=False)
    next_candidates.to_csv(out_dir / "core24_next_candidate_queue.csv", index=False)
    write_markdown(out_dir / "CORE24_RUNPLAN_CONTINUATION_READOUT.md", typed_behavior, typed_status, bridge, next_candidates)

    print(f"[done] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
