#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_ORDER = ["m5", "m4", "m3", "m2", "m1", "0"]


def _safe_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _sign(values: pd.Series) -> pd.Series:
    arr = np.sign(_safe_float(values).fillna(0.0).to_numpy())
    return pd.Series(arr, index=values.index)


def _summarize_group(df: pd.DataFrame) -> dict[str, float | int | str]:
    ref = _safe_float(df["reference_clean_delta_target_logit"])
    obs = _safe_float(df["delta_target_logit"])
    valid = ref.notna() & obs.notna()
    nz_ref = valid & (ref.abs() > 1e-9)
    nz_obs = valid & (obs.abs() > 1e-9)
    both_nz = nz_ref & nz_obs
    err = (obs - ref).abs()
    signed_match = (_sign(ref[nz_ref]) == _sign(obs[nz_ref])).mean() if int(nz_ref.sum()) else np.nan
    corr = ref[valid].corr(obs[valid]) if int(valid.sum()) >= 3 else np.nan
    corr_nz = ref[both_nz].corr(obs[both_nz]) if int(both_nz.sum()) >= 3 else np.nan
    return {
        "rows": int(len(df)),
        "valid_rows": int(valid.sum()),
        "nonzero_reference_rows": int(nz_ref.sum()),
        "nonzero_observed_rows": int(nz_obs.sum()),
        "both_nonzero_rows": int(both_nz.sum()),
        "sign_agreement_on_nonzero_reference": float(signed_match) if not np.isnan(signed_match) else np.nan,
        "mean_abs_error_all": float(err[valid].mean()) if int(valid.sum()) else np.nan,
        "median_abs_error_all": float(err[valid].median()) if int(valid.sum()) else np.nan,
        "mean_abs_error_nonzero_reference": float(err[nz_ref].mean()) if int(nz_ref.sum()) else np.nan,
        "pearson_all": float(corr) if not np.isnan(corr) else np.nan,
        "pearson_both_nonzero": float(corr_nz) if not np.isnan(corr_nz) else np.nan,
    }


def _rank_offsets(summary: pd.DataFrame, min_valid_rows: int) -> pd.DataFrame:
    ranked = summary.copy()
    ranked["coverage_eligible"] = ranked["valid_rows"].fillna(0).astype(float) >= float(min_valid_rows)
    ranked["rank_score"] = (
        ranked["coverage_eligible"].astype(float) * 1000.0
        + ranked["sign_agreement_on_nonzero_reference"].fillna(0.0) * 100.0
        + ranked["both_nonzero_rows"].fillna(0.0)
        - ranked["mean_abs_error_nonzero_reference"].fillna(ranked["mean_abs_error_all"]).fillna(999.0)
    )
    return ranked.sort_values(
        ["coverage_eligible", "rank_score", "sign_agreement_on_nonzero_reference", "both_nonzero_rows"],
        ascending=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize clean-only feature-position offset sweep for the core24 region-mask pack.")
    parser.add_argument("--sweep-dir", required=True, help="Directory containing region_mask_clean_offset_<label>.csv files.")
    parser.add_argument(
        "--manifest-dir",
        default="",
        help="Directory containing region_experiment_manifest_remote_offset_<label>.csv files. Defaults to --sweep-dir.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--labels", default=",".join(LABEL_ORDER))
    parser.add_argument("--min-valid-rows", type=int, default=30)
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir).expanduser().resolve()
    manifest_dir = Path(args.manifest_dir).expanduser().resolve() if args.manifest_dir else sweep_dir
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]

    rows: list[dict[str, object]] = []
    role_rows: list[dict[str, object]] = []
    per_row_frames: list[pd.DataFrame] = []
    for label in labels:
        path = sweep_dir / f"region_mask_clean_offset_{label}.csv"
        if not path.exists():
            rows.append({"offset_label": label, "status": "missing"})
            continue
        df = pd.read_csv(path)
        if "reference_clean_delta_target_logit" not in df.columns:
            manifest_path = manifest_dir / f"region_experiment_manifest_remote_offset_{label}.csv"
            if not manifest_path.exists():
                rows.append({"offset_label": label, "status": "missing_manifest", "path": str(path)})
                continue
            manifest = pd.read_csv(manifest_path)
            merge_keys = [
                "sample_id",
                "run",
                "prompt_name",
                "node_role",
                "node_source",
                "feature_layer",
                "feature_pos",
                "feature_id",
                "target_token_id",
            ]
            ref_cols = merge_keys + ["reference_clean_delta_target_logit"]
            manifest = manifest[ref_cols].drop_duplicates(merge_keys)
            df = df.merge(manifest, on=merge_keys, how="left")
        df["offset_label"] = label
        df["reference_clean_delta_target_logit"] = _safe_float(df["reference_clean_delta_target_logit"])
        df["delta_target_logit"] = _safe_float(df["delta_target_logit"])
        df["clean_abs_error"] = (df["delta_target_logit"] - df["reference_clean_delta_target_logit"]).abs()
        df["clean_sign_match"] = (
            np.sign(df["reference_clean_delta_target_logit"].fillna(0.0))
            == np.sign(df["delta_target_logit"].fillna(0.0))
        )
        per_row_frames.append(df)

        summary = _summarize_group(df)
        summary.update({"offset_label": label, "status": "ok", "path": str(path)})
        rows.append(summary)

        for (role, source), g in df.groupby(["node_role", "node_source"], dropna=False):
            sub = _summarize_group(g)
            sub.update({"offset_label": label, "node_role": role, "node_source": source, "status": "ok"})
            role_rows.append(sub)

    summary_df = pd.DataFrame(rows)
    role_df = pd.DataFrame(role_rows)
    per_row_df = pd.concat(per_row_frames, ignore_index=True) if per_row_frames else pd.DataFrame()

    summary_path = out_dir / "core24_offset_sweep_summary.csv"
    role_path = out_dir / "core24_offset_sweep_by_role_source.csv"
    row_path = out_dir / "core24_offset_sweep_per_row.csv"
    summary_df.to_csv(summary_path, index=False)
    role_df.to_csv(role_path, index=False)
    per_row_df.to_csv(row_path, index=False)

    ok = summary_df[summary_df.get("status", "") == "ok"].copy()
    ranked = _rank_offsets(ok, args.min_valid_rows) if not ok.empty else pd.DataFrame()
    ranked_path = out_dir / "core24_offset_sweep_ranked.csv"
    ranked.to_csv(ranked_path, index=False)

    best_label = ranked.iloc[0]["offset_label"] if not ranked.empty else "NA"
    md_path = out_dir / "core24_offset_sweep_decision.md"
    lines = [
        "# Core24 Offset Sweep Decision",
        "",
        "Purpose: choose the most faithful global feature-position offset before spending a full region-mask run.",
        "",
        f"- best offset by frozen ranking rule: `{best_label}`",
        f"- minimum coverage for primary selection: `{args.min_valid_rows}` valid rows",
        f"- summary table: `{summary_path.name}`",
        f"- role/source table: `{role_path.name}`",
        f"- per-row diagnostics: `{row_path.name}`",
        "",
        "Ranking rule: first require sufficient coverage, then maximize sign agreement against the reference clean intervention, retain more nonzero observed rows, and penalize mean absolute error.",
        "",
        "Top offsets:",
        "",
    ]
    if not ranked.empty:
        cols = [
            "offset_label",
            "valid_rows",
            "nonzero_reference_rows",
            "nonzero_observed_rows",
            "both_nonzero_rows",
            "sign_agreement_on_nonzero_reference",
            "mean_abs_error_nonzero_reference",
            "coverage_eligible",
            "rank_score",
        ]
        lines.extend(ranked[cols].head(6).to_markdown(index=False).splitlines())
    else:
        lines.append("No completed offset files were found.")
    lines.append("")
    lines.append("Interpretation note: this is a calibration diagnostic, not evidence for the mechanism claim itself.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] wrote {summary_path}")
    print(f"[done] wrote {role_path}")
    print(f"[done] wrote {row_path}")
    print(f"[done] wrote {ranked_path}")
    print(f"[done] wrote {md_path}")
    print(f"[decision] best_offset={best_label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
