#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\Bridging")
CROSS = ROOT / "doc" / "experiments" / "stage3" / "cross_model"


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


def _feature_path(asset_label: str, pack: str) -> Path:
    return CROSS / f"stage3_{asset_label}_feature_union_{pack}_full.csv"


def _source_path(asset_label: str, pack: str) -> Path:
    return CROSS / f"stage3_{asset_label}_source_control_{pack}_full.csv"


def _prepare_feature(path: Path, asset: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    restore = df["direction"].eq("restore")
    df["effect_logit_signed"] = np.where(
        restore,
        _as_num(df["logit_restore_vs_mask"]),
        _as_num(df["logit_damage_vs_clean"]),
    )
    df["effect_rank_signed"] = np.where(
        restore,
        _as_num(df["rank_restore_vs_mask"]),
        _as_num(df["rank_damage_vs_clean"]),
    )
    key_cols = ["sample_id", "prompt_name", "mask_condition", "position_group", "direction", "feature_group"]
    out = (
        df.groupby(key_cols, dropna=False)
        .agg(
            effect_logit=("effect_logit_signed", "mean"),
            effect_rank=("effect_rank_signed", "mean"),
            gap_closure=("gap_closure", "mean"),
            clean_mask_logit_gap=("clean_mask_logit_gap", "mean"),
        )
        .reset_index()
    )
    return out.rename(
        columns={
            "effect_logit": f"{asset}_effect_logit",
            "effect_rank": f"{asset}_effect_rank",
            "gap_closure": f"{asset}_gap_closure",
            "clean_mask_logit_gap": f"{asset}_clean_mask_logit_gap",
        }
    )


def _prepare_source(path: Path, asset: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["effect_logit"] = _as_num(df["effect_logit"])
    df["effect_rank"] = _as_num(df["effect_rank"])
    pivot = df.pivot_table(
        index=["sample_id", "prompt_name", "mask_condition", "mask_variant", "intervention"],
        columns="feature_role",
        values=["effect_logit", "effect_rank", "gap_closure", "correct_minus_wrong_logit"],
        aggfunc="mean",
    )
    if not {("effect_logit", "source"), ("effect_logit", "matched_control")}.issubset(pivot.columns):
        return pd.DataFrame()
    out = pivot.reset_index()
    out[f"{asset}_source_minus_control_logit"] = out[("effect_logit", "source")] - out[("effect_logit", "matched_control")]
    out[f"{asset}_source_minus_control_rank"] = out[("effect_rank", "source")] - out[("effect_rank", "matched_control")]
    out[f"{asset}_source_minus_control_gap_closure"] = out[("gap_closure", "source")] - out[("gap_closure", "matched_control")]
    out[f"{asset}_source_correct_minus_wrong"] = out[("correct_minus_wrong_logit", "source")]
    keep = [
        "sample_id",
        "prompt_name",
        "mask_condition",
        "mask_variant",
        "intervention",
        f"{asset}_source_minus_control_logit",
        f"{asset}_source_minus_control_rank",
        f"{asset}_source_minus_control_gap_closure",
        f"{asset}_source_correct_minus_wrong",
    ]
    out.columns = ["_".join([str(x) for x in col if str(x) != ""]) if isinstance(col, tuple) else str(col) for col in out.columns]
    keep_flat = [col.replace("__", "_") for col in keep]
    rename = {
        f"{asset}_source_minus_control_logit": f"{asset}_source_minus_control_logit",
        f"{asset}_source_minus_control_rank": f"{asset}_source_minus_control_rank",
        f"{asset}_source_minus_control_gap_closure": f"{asset}_source_minus_control_gap_closure",
        f"{asset}_source_correct_minus_wrong": f"{asset}_source_correct_minus_wrong",
    }
    out = out.rename(columns=rename)
    return out[[col for col in keep_flat if col in out.columns]]


def _summarize_paired(df: pd.DataFrame, group_cols: list[str], plt_col: str, clt_col: str, diff_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        values = _as_num(group[diff_col])
        ci_low, ci_high = _bootstrap_ci(values)
        plt = _as_num(group[plt_col])
        clt = _as_num(group[clt_col])
        sign_agree = ((plt > 0) & (clt > 0)) | ((plt < 0) & (clt < 0)) | ((plt == 0) & (clt == 0))
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "n_pairs": int(values.notna().sum()),
                "mean_plt": float(plt.mean()) if plt.notna().any() else float("nan"),
                "mean_clt": float(clt.mean()) if clt.notna().any() else float("nan"),
                "mean_clt_minus_plt": float(values.mean()) if values.notna().any() else float("nan"),
                "ci95_low_clt_minus_plt": ci_low,
                "ci95_high_clt_minus_plt": ci_high,
                "plt_positive_frac": float((plt > 0).mean()) if plt.notna().any() else float("nan"),
                "clt_positive_frac": float((clt > 0).mean()) if clt.notna().any() else float("nan"),
                "same_positive_frac": float(((plt > 0) & (clt > 0)).mean()) if plt.notna().any() else float("nan"),
                "sign_agreement_frac": float(sign_agree.mean()) if plt.notna().any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _decision(pack: str, feature_summary: pd.DataFrame, source_summary: pd.DataFrame, paired_feature: pd.DataFrame, paired_source: pd.DataFrame) -> dict[str, object]:
    primary_feature = feature_summary[
        feature_summary["position_group"].eq("top_hidden_delta_plus_answer_adjacent")
        & feature_summary["feature_group"].eq("evidence_attribution_topk")
    ].copy()
    primary_source = source_summary[
        source_summary["mask_variant"].eq("real_mask")
    ].copy()
    clt_feature_positive = int((primary_feature["clt_positive_frac"] > 0.5).sum()) if not primary_feature.empty else 0
    clt_feature_total = int(primary_feature.shape[0])
    clt_source_positive = int((primary_source["clt_positive_frac"] > 0.5).sum()) if not primary_source.empty else 0
    clt_source_total = int(primary_source.shape[0])
    agreement = float(
        pd.concat(
            [
                primary_feature.get("sign_agreement_frac", pd.Series(dtype=float)),
                primary_source.get("sign_agreement_frac", pd.Series(dtype=float)),
            ]
        ).mean()
    )

    status = "representation_dependent_or_not_established"
    if clt_feature_total and clt_source_total and clt_feature_positive >= max(1, clt_feature_total // 2) and clt_source_positive >= max(1, clt_source_total // 2):
        status = "qwen_clt_partial_robustness"
    if status == "qwen_clt_partial_robustness" and agreement >= 0.6:
        status = "qwen_plt_clt_same_direction_robustness_support"

    return {
        "status": status,
        "pack": pack,
        "paired_feature_rows": int(paired_feature.shape[0]),
        "paired_source_rows": int(paired_source.shape[0]),
        "primary_feature_clt_positive": {"positive": clt_feature_positive, "total": clt_feature_total},
        "primary_source_clt_positive": {"positive": clt_source_positive, "total": clt_source_total},
        "primary_sign_agreement_mean": agreement,
        "claim_boundary": (
            "This compares Qwen2.5-VL PLT vs CLT on the same paperpack sample/prompt/mask/control schema. "
            "It supports robustness or representation-dependence claims only; it is not Gemma-style source tracing."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired Qwen2.5-VL PLT-vs-CLT paperpack comparison.")
    parser.add_argument("--pack", choices=["primary", "strict"], default="primary")
    args = parser.parse_args()

    paths = {
        "plt_feature": _feature_path("qwen2p5vl_plt", args.pack),
        "clt_feature": _feature_path("qwen2p5vl_clt", args.pack),
        "plt_source": _source_path("qwen2p5vl_plt", args.pack),
        "clt_source": _source_path("qwen2p5vl_clt", args.pack),
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    prefix = CROSS / f"stage3_qwen_plt_vs_clt_{args.pack}"
    if missing:
        payload = {
            "status": "blocked",
            "pack": args.pack,
            "missing_inputs": {name: str(paths[name]) for name in missing},
            "claim_boundary": "PLT-vs-CLT comparison requires both PLT and CLT paperpack full outputs.",
        }
        with open(f"{prefix}_decision.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    plt_feature = _prepare_feature(paths["plt_feature"], "plt")
    clt_feature = _prepare_feature(paths["clt_feature"], "clt")
    feature_keys = ["sample_id", "prompt_name", "mask_condition", "position_group", "direction", "feature_group"]
    paired_feature = plt_feature.merge(clt_feature, on=feature_keys, how="inner")
    paired_feature["clt_minus_plt_logit"] = paired_feature["clt_effect_logit"] - paired_feature["plt_effect_logit"]
    paired_feature["clt_minus_plt_rank"] = paired_feature["clt_effect_rank"] - paired_feature["plt_effect_rank"]

    plt_source = _prepare_source(paths["plt_source"], "plt")
    clt_source = _prepare_source(paths["clt_source"], "clt")
    source_keys = ["sample_id", "prompt_name", "mask_condition", "mask_variant", "intervention"]
    paired_source = plt_source.merge(clt_source, on=source_keys, how="inner") if not plt_source.empty and not clt_source.empty else pd.DataFrame()
    if not paired_source.empty:
        paired_source["clt_minus_plt_logit"] = (
            paired_source["clt_source_minus_control_logit"] - paired_source["plt_source_minus_control_logit"]
        )
        paired_source["clt_minus_plt_rank"] = (
            paired_source["clt_source_minus_control_rank"] - paired_source["plt_source_minus_control_rank"]
        )

    feature_summary = _summarize_paired(
        paired_feature,
        ["position_group", "direction", "feature_group", "mask_condition"],
        "plt_effect_logit",
        "clt_effect_logit",
        "clt_minus_plt_logit",
    )
    source_summary = (
        _summarize_paired(
            paired_source,
            ["mask_condition", "mask_variant", "intervention"],
            "plt_source_minus_control_logit",
            "clt_source_minus_control_logit",
            "clt_minus_plt_logit",
        )
        if not paired_source.empty
        else pd.DataFrame()
    )
    decision = _decision(args.pack, feature_summary, source_summary, paired_feature, paired_source)

    paired_feature.to_csv(f"{prefix}_paired_feature.csv", index=False)
    paired_source.to_csv(f"{prefix}_paired_source_control.csv", index=False)
    feature_summary.to_csv(f"{prefix}_feature_summary.csv", index=False)
    source_summary.to_csv(f"{prefix}_source_control_summary.csv", index=False)
    pd.concat(
        [
            feature_summary.assign(kind="feature"),
            source_summary.assign(kind="source_control"),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(f"{prefix}_summary.csv", index=False)
    with open(f"{prefix}_decision.json", "w", encoding="utf-8") as handle:
        json.dump(decision, handle, ensure_ascii=False, indent=2)

    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(f"wrote outputs with prefix {prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
