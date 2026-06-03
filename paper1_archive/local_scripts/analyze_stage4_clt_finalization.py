#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(r"E:\Bridging")
DEFAULT_CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"

ASSETS = {
    "qwen_clt": {"label": "qwen2p5vl_clt", "layers": [26]},
    "llava_clt": {"label": "llava15_clt", "layers": [12, 15, 18, 21]},
}
CONTROL_GROUPS = [
    "activation_matched_topk",
    "drop_matched_topk",
    "attribution_matched_mask_insensitive_topk",
    "random_active_topk",
]


def _as_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _mean(series: pd.Series) -> float:
    vals = _as_num(series).dropna()
    return float(vals.mean()) if len(vals) else math.nan


def _fmt(value: float) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.6g}"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _feature_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "feature_rows": 0,
            "feature_prompt_runs": 0,
            "feature_positive_controls": 0,
            "feature_control_tests": 0,
            "feature_mean_source_minus_control": math.nan,
        }
    df = df.copy()
    restore = df["direction"].eq("restore")
    df["effect"] = pd.NA
    df.loc[restore, "effect"] = _as_num(df.loc[restore, "logit_restore_vs_mask"])
    df.loc[~restore, "effect"] = _as_num(df.loc[~restore, "logit_damage_vs_clean"])
    key = ["sample_id", "prompt_name", "position_group", "direction", "mask_condition"]
    diffs = []
    for control in CONTROL_GROUPS:
        sub = df[df["feature_group"].isin(["evidence_attribution_topk", control])]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index=key, columns="feature_group", values="effect", aggfunc="mean")
        if {"evidence_attribution_topk", control}.issubset(pivot.columns):
            diffs.extend((pivot["evidence_attribution_topk"] - pivot[control]).dropna().tolist())
    return {
        "feature_rows": int(len(df)),
        "feature_prompt_runs": int(df[["sample_id", "prompt_name"]].drop_duplicates().shape[0]),
        "feature_positive_controls": int(sum(1 for value in diffs if value > 0)),
        "feature_control_tests": int(len(diffs)),
        "feature_mean_source_minus_control": float(pd.Series(diffs).mean()) if diffs else math.nan,
    }


def _source_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "source_rows": 0,
            "source_prompt_runs": 0,
            "source_pairs": 0,
            "source_positive_pairs": 0,
            "source_mean_minus_control": math.nan,
            "real_minus_shuffled_mean": math.nan,
        }
    df = df.copy()
    df["effect"] = _as_num(df["effect_logit"])
    key = ["sample_id", "prompt_name", "mask_condition", "mask_variant", "intervention"]
    pivot = df.pivot_table(index=key, columns="feature_role", values="effect", aggfunc="mean")
    diffs = []
    if {"source", "matched_control"}.issubset(pivot.columns):
        diffs = (pivot["source"] - pivot["matched_control"]).dropna().tolist()
    real = df[df["feature_role"].eq("source") & df["mask_variant"].isin(["real_mask", "mask_shuffled"])]
    rv = real.pivot_table(
        index=["sample_id", "prompt_name", "mask_condition", "intervention"],
        columns="mask_variant",
        values="effect",
        aggfunc="mean",
    )
    real_shuffled = []
    if {"real_mask", "mask_shuffled"}.issubset(rv.columns):
        real_shuffled = (rv["real_mask"] - rv["mask_shuffled"]).dropna().tolist()
    return {
        "source_rows": int(len(df)),
        "source_prompt_runs": int(df[["sample_id", "prompt_name"]].drop_duplicates().shape[0]),
        "source_pairs": int(len(diffs)),
        "source_positive_pairs": int(sum(1 for value in diffs if value > 0)),
        "source_mean_minus_control": float(pd.Series(diffs).mean()) if diffs else math.nan,
        "real_minus_shuffled_mean": float(pd.Series(real_shuffled).mean()) if real_shuffled else math.nan,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage4 CLT finalization outputs.")
    parser.add_argument("--cross-dir", default=str(DEFAULT_CROSS))
    parser.add_argument("--asset", choices=["qwen_clt", "llava_clt", "both"], default="both")
    parser.add_argument("--pack", choices=["primary", "strict"], default="primary")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--topks", default="1,4,8,16,32")
    args = parser.parse_args()

    cross = Path(args.cross_dir)
    topks = [int(x.strip()) for x in args.topks.split(",") if x.strip()]
    assets = ["qwen_clt", "llava_clt"] if args.asset == "both" else [args.asset]
    rows: list[dict[str, Any]] = []
    for asset in assets:
        cfg = ASSETS[asset]
        for layer in cfg["layers"]:
            for topk in topks:
                stem = f"stage4_{cfg['label']}_{args.pack}_{args.mode}_L{layer}_topK{topk}"
                feature = _read(cross / f"{stem}_feature_union.csv")
                source = _read(cross / f"{stem}_source_control.csv")
                feature_s = _feature_summary(feature)
                source_s = _source_summary(source)
                rows.append(
                    {
                        "asset": asset,
                        "pack": args.pack,
                        "mode": args.mode,
                        "layer": layer,
                        "topk": topk,
                        **feature_s,
                        **source_s,
                        "feature_mean_source_minus_control": _fmt(feature_s["feature_mean_source_minus_control"]),
                        "source_mean_minus_control": _fmt(source_s["source_mean_minus_control"]),
                        "real_minus_shuffled_mean": _fmt(source_s["real_minus_shuffled_mean"]),
                    }
                )

    out_summary = cross / f"stage4_clt_finalization_{args.asset}_{args.pack}_{args.mode}_summary.csv"
    pd.DataFrame(rows).to_csv(out_summary, index=False)

    decision_by_asset: dict[str, Any] = {}
    for asset in assets:
        part = [row for row in rows if row["asset"] == asset]
        usable = [row for row in part if int(row["source_pairs"]) > 0 or int(row["feature_control_tests"]) > 0]
        best_source = max((float(row["source_mean_minus_control"] or "nan") for row in usable), default=math.nan)
        best_feature = max((float(row["feature_mean_source_minus_control"] or "nan") for row in usable), default=math.nan)
        any_real_positive = any(
            row["real_minus_shuffled_mean"] not in ("", None) and float(row["real_minus_shuffled_mean"]) > 0
            for row in usable
        )
        if not usable:
            status = "blocked"
            reason = "no_usable_stage4_clt_outputs"
        elif asset == "qwen_clt" and best_source > 0 and best_feature > 0:
            status = "qwen_clt_robustness_final"
            reason = "positive_feature_and_source_control_under_topk_sensitivity"
        elif asset == "llava_clt" and best_source > 0 and best_feature > 0 and any_real_positive:
            status = "llava_clt_layer_dependent_weak_bridge"
            reason = "at_least_one_layer_topk_has_positive_feature_source_and_real_control"
        elif asset == "llava_clt":
            status = "llava_clt_feature_source_not_established"
            reason = "tested_layers_topks_do_not_jointly_establish_feature_source_route"
        else:
            status = "partial_or_representation_dependent"
            reason = "some_positive_direction_but_not_all_final_criteria"
        decision_by_asset[asset] = {
            "status": status,
            "reason": reason,
            "best_source_mean_minus_control": _fmt(best_source),
            "best_feature_mean_source_minus_control": _fmt(best_feature),
            "usable_configs": len(usable),
        }

    decision = {
        "status": "stage4_clt_finalization_analyzed",
        "pack": args.pack,
        "mode": args.mode,
        "summary_csv": str(out_summary),
        "assets": decision_by_asset,
        "claim_boundary": (
            "CLT is an auxiliary robustness/diagnostic line. Qwen CLT success supports robustness; "
            "LLaVA CLT failure does not imply absence of cross-modal mechanisms."
        ),
    }
    out_decision = cross / f"stage4_clt_finalization_{args.asset}_{args.pack}_{args.mode}_decision.json"
    out_decision.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

