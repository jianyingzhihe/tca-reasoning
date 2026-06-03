#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\Bridging")
STAGE3 = ROOT / "doc" / "experiments" / "stage3"
CROSS = STAGE3 / "cross_model"
PAPERPACK = STAGE3 / "paperpack72"


ASSETS = {
    "qwen_clt": {
        "label": "qwen2p5vl_clt",
        "display": "Qwen2.5-VL-CLT",
        "positive_status": "qwen_clt_robustness_support",
        "partial_status": "partial_qwen_clt_support",
        "boundary": (
            "Qwen2.5-VL-CLT is an auxiliary robustness line. Success supports "
            "transcoder-type robustness/representation comparison, not full Gemma-style source tracing."
        ),
    },
    "llava_clt": {
        "label": "llava15_clt",
        "display": "LLaVA-1.5-CLT",
        "positive_status": "llava_clt_feature_diagnostic_support",
        "partial_status": "partial_llava_clt_or_hidden_only_support",
        "boundary": (
            "LLaVA-CLT is diagnostic/auxiliary. Weak feature localization is not evidence that "
            "LLaVA has no cross-modal mechanism."
        ),
    },
}

CONTROL_GROUPS = [
    "activation_matched_topk",
    "drop_matched_topk",
    "attribution_matched_mask_insensitive_topk",
    "random_active_topk",
]


def _as_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _bootstrap_ci(values: pd.Series | np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    arr = np.asarray(pd.Series(values).dropna(), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _manifest_path(pack: str) -> Path:
    if pack == "primary":
        return PAPERPACK / "paperpack72_primary_manifest.csv"
    return PAPERPACK / "paperpack72_strict_sensitivity_manifest.csv"


def _input_paths(asset: str, pack: str) -> tuple[Path, Path]:
    label = ASSETS[asset]["label"]
    suffix = f"{pack}_full"
    return (
        CROSS / f"stage3_{label}_feature_union_{suffix}.csv",
        CROSS / f"stage3_{label}_source_control_{suffix}.csv",
    )


def _write_blocked(asset: str, pack: str, reason: str, feature_path: Path, source_path: Path) -> None:
    label = ASSETS[asset]["label"]
    prefix = CROSS / f"stage3_{label}_{pack}_full"
    payload = {
        "status": "blocked",
        "asset": ASSETS[asset]["display"],
        "pack": pack,
        "reason": reason,
        "feature_path": str(feature_path),
        "source_path": str(source_path),
        "claim_boundary": ASSETS[asset]["boundary"],
    }
    with open(f"{prefix}_decision.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _add_manifest_columns(df: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "sample_id",
        "quota_slot",
        "reasoning_operation",
        "paperpack_source",
        "image_dependence_tier",
        "manual_review_flag",
        "mask_geometry_tier",
        "union_area_frac",
    ]
    return df.merge(manifest[[c for c in keep if c in manifest.columns]], on="sample_id", how="left")


def _prepare_feature(feature: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    feature = _add_manifest_columns(feature.copy(), manifest)
    restore = feature["direction"].eq("restore")
    feature["effect_logit_signed"] = np.where(
        restore,
        _as_num(feature["logit_restore_vs_mask"]),
        _as_num(feature["logit_damage_vs_clean"]),
    )
    feature["effect_rank_signed"] = np.where(
        restore,
        _as_num(feature["rank_restore_vs_mask"]),
        _as_num(feature["rank_damage_vs_clean"]),
    )
    return feature


def _prepare_source(source: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    source = _add_manifest_columns(source.copy(), manifest)
    source["effect_logit_signed"] = _as_num(source["effect_logit"])
    source["effect_rank_signed"] = _as_num(source["effect_rank"])
    source["correct_minus_wrong_logit"] = _as_num(source["correct_minus_wrong_logit"])
    return source


def _summarize_values(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        values = _as_num(group[value_col])
        ci_low, ci_high = _bootstrap_ci(values)
        rank_values = _as_num(group["effect_rank_signed"]) if "effect_rank_signed" in group.columns else pd.Series(dtype=float)
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "n_rows": int(values.notna().sum()),
                "n_prompt_runs": int(group[["sample_id", "prompt_name"]].drop_duplicates().shape[0])
                if {"sample_id", "prompt_name"}.issubset(group.columns)
                else int(values.notna().sum()),
                f"mean_{value_col}": float(values.mean()) if values.notna().any() else float("nan"),
                f"ci95_low_{value_col}": ci_low,
                f"ci95_high_{value_col}": ci_high,
                f"positive_frac_{value_col}": float((values > 0).mean()) if values.notna().any() else float("nan"),
                "mean_gap_closure": float(_as_num(group.get("gap_closure", pd.Series(dtype=float))).mean()),
                "mean_effect_rank": float(rank_values.mean()) if rank_values.notna().any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _feature_specificity(feature: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["sample_id", "prompt_name", "position_group", "direction", "mask_condition"]
    rows: list[dict[str, object]] = []
    for control in CONTROL_GROUPS:
        sub = feature[feature["feature_group"].isin(["evidence_attribution_topk", control])].copy()
        pivot = sub.pivot_table(
            index=key_cols,
            columns="feature_group",
            values=["effect_logit_signed", "effect_rank_signed", "gap_closure"],
            aggfunc="mean",
        )
        required = [("effect_logit_signed", "evidence_attribution_topk"), ("effect_logit_signed", control)]
        if not all(col in pivot.columns for col in required):
            continue
        tmp = pivot.reset_index()
        tmp["diff_logit"] = (
            pivot[("effect_logit_signed", "evidence_attribution_topk")] - pivot[("effect_logit_signed", control)]
        ).to_numpy()
        tmp["diff_rank"] = (
            pivot[("effect_rank_signed", "evidence_attribution_topk")] - pivot[("effect_rank_signed", control)]
        ).to_numpy()
        tmp["diff_gap_closure"] = (
            pivot[("gap_closure", "evidence_attribution_topk")] - pivot[("gap_closure", control)]
        ).to_numpy()
        for key, group in tmp.groupby(["position_group", "direction", "mask_condition"], dropna=False):
            ci_low, ci_high = _bootstrap_ci(group["diff_logit"])
            rows.append(
                {
                    "control_group": control,
                    "position_group": key[0],
                    "direction": key[1],
                    "mask_condition": key[2],
                    "n_pairs": int(group["diff_logit"].notna().sum()),
                    "mean_evidence_minus_control_logit": float(group["diff_logit"].mean()),
                    "ci95_low_logit": ci_low,
                    "ci95_high_logit": ci_high,
                    "positive_frac_logit": float((group["diff_logit"] > 0).mean()),
                    "mean_evidence_minus_control_rank": float(group["diff_rank"].mean()),
                    "mean_evidence_minus_control_gap_closure": float(group["diff_gap_closure"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _source_specificity(source: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["sample_id", "prompt_name", "mask_condition", "mask_variant", "intervention"]
    pivot = source.pivot_table(
        index=key_cols,
        columns="feature_role",
        values=["effect_logit_signed", "effect_rank_signed", "gap_closure", "correct_minus_wrong_logit"],
        aggfunc="mean",
    )
    if not {("effect_logit_signed", "source"), ("effect_logit_signed", "matched_control")}.issubset(pivot.columns):
        return pd.DataFrame()
    tmp = pivot.reset_index()
    tmp["source_minus_control_logit"] = (
        tmp[("effect_logit_signed", "source")] - tmp[("effect_logit_signed", "matched_control")]
    )
    tmp["source_minus_control_rank"] = (
        tmp[("effect_rank_signed", "source")] - tmp[("effect_rank_signed", "matched_control")]
    )
    tmp["source_minus_control_gap_closure"] = tmp[("gap_closure", "source")] - tmp[("gap_closure", "matched_control")]
    tmp["source_minus_control_correct_wrong"] = (
        tmp[("correct_minus_wrong_logit", "source")] - tmp[("correct_minus_wrong_logit", "matched_control")]
    )

    rows: list[dict[str, object]] = []
    for key, group in tmp.groupby(["mask_condition", "mask_variant", "intervention"], dropna=False):
        ci_low, ci_high = _bootstrap_ci(group["source_minus_control_logit"])
        rows.append(
            {
                "comparison": "source_minus_matched_control",
                "mask_condition": key[0],
                "mask_variant": key[1],
                "intervention": key[2],
                "n_pairs": int(group["source_minus_control_logit"].notna().sum()),
                "mean_logit": float(group["source_minus_control_logit"].mean()),
                "ci95_low_logit": ci_low,
                "ci95_high_logit": ci_high,
                "positive_frac_logit": float((group["source_minus_control_logit"] > 0).mean()),
                "mean_rank": float(group["source_minus_control_rank"].mean()),
                "mean_gap_closure": float(group["source_minus_control_gap_closure"].mean()),
                "mean_correct_minus_wrong_logit": float(group["source_minus_control_correct_wrong"].mean()),
            }
        )

    for role in ["source", "matched_control"]:
        sub = source[source["feature_role"].eq(role) & source["mask_variant"].isin(["real_mask", "mask_shuffled"])]
        p = sub.pivot_table(
            index=["sample_id", "prompt_name", "mask_condition", "intervention"],
            columns="mask_variant",
            values="effect_logit_signed",
            aggfunc="mean",
        )
        if {"real_mask", "mask_shuffled"}.issubset(p.columns):
            p["real_minus_shuffled_logit"] = p["real_mask"] - p["mask_shuffled"]
            for key, group in p.reset_index().groupby(["mask_condition", "intervention"], dropna=False):
                ci_low, ci_high = _bootstrap_ci(group["real_minus_shuffled_logit"])
                rows.append(
                    {
                        "comparison": f"{role}_real_minus_shuffled",
                        "mask_condition": key[0],
                        "mask_variant": "real_vs_shuffled",
                        "intervention": key[1],
                        "n_pairs": int(group["real_minus_shuffled_logit"].notna().sum()),
                        "mean_logit": float(group["real_minus_shuffled_logit"].mean()),
                        "ci95_low_logit": ci_low,
                        "ci95_high_logit": ci_high,
                        "positive_frac_logit": float((group["real_minus_shuffled_logit"] > 0).mean()),
                        "mean_rank": float("nan"),
                        "mean_gap_closure": float("nan"),
                        "mean_correct_minus_wrong_logit": float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def _case_table(feature: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    primary_feature = feature[
        feature["position_group"].eq("top_hidden_delta_plus_answer_adjacent")
        & feature["feature_group"].eq("evidence_attribution_topk")
    ].copy()
    f = (
        primary_feature.groupby(["sample_id", "prompt_name", "direction"], dropna=False)
        .agg(
            feature_mean_logit=("effect_logit_signed", "mean"),
            feature_mean_rank=("effect_rank_signed", "mean"),
            feature_mean_gap_closure=("gap_closure", "mean"),
            clean_rank=("clean_target_rank", "mean"),
            mask_rank=("mask_target_rank", "mean"),
            clean_mask_logit_gap=("clean_mask_logit_gap", "mean"),
            target_answer=("target_answer", "first"),
            quota_slot=("quota_slot", "first"),
            image_dependence_tier=("image_dependence_tier", "first"),
            manual_review_flag=("manual_review_flag", "first"),
        )
        .reset_index()
    )
    if f.empty:
        return pd.DataFrame()
    f_pivot = f.pivot_table(
        index=["sample_id", "prompt_name", "target_answer", "quota_slot", "image_dependence_tier", "manual_review_flag"],
        columns="direction",
        values=["feature_mean_logit", "feature_mean_rank", "feature_mean_gap_closure", "clean_rank", "mask_rank", "clean_mask_logit_gap"],
        aggfunc="mean",
    ).reset_index()
    f_pivot.columns = ["_".join([str(x) for x in col if str(x) != ""]) for col in f_pivot.columns]

    src = source[source["mask_variant"].eq("real_mask")].copy()
    p = src.pivot_table(
        index=["sample_id", "prompt_name", "mask_condition", "intervention"],
        columns="feature_role",
        values="effect_logit_signed",
        aggfunc="mean",
    )
    if {"source", "matched_control"}.issubset(p.columns):
        p["source_minus_control"] = p["source"] - p["matched_control"]
        s = p.reset_index().pivot_table(
            index=["sample_id", "prompt_name"],
            columns=["mask_condition", "intervention"],
            values="source_minus_control",
            aggfunc="mean",
        ).reset_index()
        s.columns = ["_".join([str(x) for x in col if str(x) != ""]) for col in s.columns]
        return f_pivot.merge(s, on=["sample_id", "prompt_name"], how="left")
    return f_pivot


def _decision(asset: str, pack: str, feature: pd.DataFrame, source: pd.DataFrame, feature_spec: pd.DataFrame, source_spec: pd.DataFrame) -> dict[str, object]:
    cfg = ASSETS[asset]
    primary_feature = feature_spec[
        feature_spec["position_group"].eq("top_hidden_delta_plus_answer_adjacent")
        & feature_spec["direction"].isin(["restore", "corrupt"])
    ].copy()
    source_real = source_spec[
        source_spec["comparison"].eq("source_minus_matched_control")
        & source_spec["mask_variant"].eq("real_mask")
    ].copy()
    source_shuf = source_spec[source_spec["comparison"].str.endswith("real_minus_shuffled", na=False)].copy()

    feature_positive = int((primary_feature["mean_evidence_minus_control_logit"] > 0).sum()) if not primary_feature.empty else 0
    feature_total = int(primary_feature.shape[0])
    source_positive = int((source_real["mean_logit"] > 0).sum()) if not source_real.empty else 0
    source_total = int(source_real.shape[0])
    shuffled_positive = int((source_shuf["mean_logit"] > 0).sum()) if not source_shuf.empty else 0
    shuffled_total = int(source_shuf.shape[0])

    status = "not_supported"
    if feature_total and source_total and feature_positive >= max(1, feature_total // 2) and source_positive >= max(1, source_total // 2):
        status = f"{cfg['partial_status']}_{pack}"
    if (
        feature_total
        and feature_positive == feature_total
        and source_total
        and source_positive == source_total
        and shuffled_total
        and shuffled_positive >= max(1, shuffled_total // 2)
    ):
        status = f"{cfg['positive_status']}_{pack}"

    return {
        "status": status,
        "asset": cfg["display"],
        "pack": pack,
        "claim_boundary": cfg["boundary"],
        "inputs": {
            "feature_rows": int(feature.shape[0]),
            "feature_prompt_runs": int(feature[["sample_id", "prompt_name"]].drop_duplicates().shape[0]) if not feature.empty else 0,
            "source_rows": int(source.shape[0]),
            "source_prompt_runs": int(source[["sample_id", "prompt_name"]].drop_duplicates().shape[0]) if not source.empty else 0,
            "source_usable_pairs": int(source[source["feature_role"].eq("source")].shape[0] / 4) if not source.empty else 0,
        },
        "primary_feature_specificity_positive": {"positive": feature_positive, "total": feature_total},
        "source_control_positive": {"positive": source_positive, "total": source_total},
        "real_vs_shuffled_positive": {"positive": shuffled_positive, "total": shuffled_total},
        "known_gaps": [
            "This CLT line is auxiliary and should not alter the PLT-only verdict by itself.",
            "Source-control is approximate feature probing, not Gemma ReplacementModel source tracing.",
            "Decoded generation-level bridge is not part of this paperpack CLT analyzer.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage3 paperpack CLT full outputs.")
    parser.add_argument("--asset", choices=sorted(ASSETS), required=True)
    parser.add_argument("--pack", choices=["primary", "strict"], default="primary")
    args = parser.parse_args()

    feature_path, source_path = _input_paths(args.asset, args.pack)
    if not feature_path.exists() or not source_path.exists():
        _write_blocked(args.asset, args.pack, "missing_feature_or_source_csv", feature_path, source_path)
        return 1

    feature_raw = pd.read_csv(feature_path)
    source_raw = pd.read_csv(source_path)
    manifest = pd.read_csv(_manifest_path(args.pack))
    feature = _prepare_feature(feature_raw, manifest)
    source = _prepare_source(source_raw, manifest)

    label = ASSETS[args.asset]["label"]
    prefix = CROSS / f"stage3_{label}_{args.pack}_full"
    feature_summary = _summarize_values(
        feature,
        ["position_group", "direction", "feature_group", "mask_condition"],
        "effect_logit_signed",
    )
    feature_spec = _feature_specificity(feature)
    source_summary = _summarize_values(
        source,
        ["mask_condition", "mask_variant", "intervention", "feature_role"],
        "effect_logit_signed",
    )
    source_spec = _source_specificity(source)
    cases = _case_table(feature, source)
    decision = _decision(args.asset, args.pack, feature, source, feature_spec, source_spec)

    feature_summary.to_csv(f"{prefix}_feature_summary.csv", index=False)
    feature_spec.to_csv(f"{prefix}_feature_specificity.csv", index=False)
    source_summary.to_csv(f"{prefix}_source_control_summary.csv", index=False)
    source_spec.to_csv(f"{prefix}_source_control_specificity.csv", index=False)
    cases.to_csv(f"{prefix}_case_table.csv", index=False)
    with open(f"{prefix}_decision.json", "w", encoding="utf-8") as handle:
        json.dump(decision, handle, ensure_ascii=False, indent=2)

    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(f"wrote outputs with prefix {prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
