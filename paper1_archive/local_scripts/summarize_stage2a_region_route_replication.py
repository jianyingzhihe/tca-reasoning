#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = [
    "pair_id",
    "sample_id",
    "run",
    "prompt_name",
    "node_role",
    "node_source",
    "feature_layer",
    "feature_pos",
    "feature_id",
    "reasoning_operation",
]


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


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


def _metric_row(name: str, values: pd.Series, *, unit: str) -> dict[str, object]:
    vals = _num(values).dropna()
    lo, hi = _bootstrap_ci(vals)
    return {
        "metric": name,
        "unit": unit,
        "n": int(vals.shape[0]),
        "mean": float(vals.mean()) if len(vals) else math.nan,
        "median": float(vals.median()) if len(vals) else math.nan,
        "positive_rate": float((vals > 0).mean()) if len(vals) else math.nan,
        "bootstrap_ci_low": lo,
        "bootstrap_ci_high": hi,
    }


def _read_region(region_csv: Path, manifest_csv: Path) -> pd.DataFrame:
    region = pd.read_csv(region_csv)
    manifest = pd.read_csv(manifest_csv)
    keys = ["sample_id", "run", "node_role", "node_source", "feature_layer", "feature_pos", "feature_id"]
    for df in (region, manifest):
        for col in ["feature_layer", "feature_pos", "feature_id"]:
            df[col] = _num(df[col]).astype("Int64").astype(str)
    manifest_cols = keys + ["pair_id", "reference_clean_delta_target_logit"]
    merged = region.merge(manifest[manifest_cols], on=keys, how="left")
    missing = int(merged["pair_id"].isna().sum())
    if missing:
        raise ValueError(f"missing pair_id for {missing} region rows")
    merged["delta_target_logit"] = _num(merged["delta_target_logit"])
    merged["reference_clean_delta_target_logit"] = _num(merged["reference_clean_delta_target_logit"])
    merged["random_control_actual_iou"] = _num(merged.get("random_control_actual_iou", pd.Series(index=merged.index)))
    return merged


def _clean_calibration(region: pd.DataFrame) -> pd.DataFrame:
    clean = region[region["condition"].eq("clean")].copy()
    clean["abs_error"] = (clean["delta_target_logit"] - clean["reference_clean_delta_target_logit"]).abs()
    return (
        clean.groupby(["node_role", "node_source"], dropna=False)
        .agg(
            rows=("sample_id", "size"),
            samples=("sample_id", "nunique"),
            mean_abs_error=("abs_error", "mean"),
            max_abs_error=("abs_error", "max"),
            exact_match_rate=("abs_error", lambda s: float((s < 1e-9).mean())),
        )
        .reset_index()
    )


def _route_wide(region: pd.DataFrame, random_iou_threshold: float) -> pd.DataFrame:
    fixed = region[region["condition"].isin(["clean", "answer_mask", "relate_mask", "union_mask"])]
    wide = fixed.pivot_table(index=KEYS, columns="condition", values="delta_target_logit", aggfunc="first").reset_index()
    rand = region[
        region["condition"].astype(str).str.startswith("random_control_")
        & region["random_control_actual_iou"].notna()
        & (region["random_control_actual_iou"] <= random_iou_threshold)
    ]
    if rand.empty:
        rand_mean = pd.DataFrame(columns=KEYS + ["random4_delta", "valid_random_controls", "random4_max_iou"])
    else:
        rand_mean = (
            rand.groupby(KEYS, as_index=False, dropna=False)
            .agg(
                random4_delta=("delta_target_logit", "mean"),
                valid_random_controls=("condition", "nunique"),
                random4_max_iou=("random_control_actual_iou", "max"),
            )
        )
    wide = wide.merge(rand_mean, on=KEYS, how="left")
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
    for condition in ["answer_mask", "relate_mask", "union_mask"]:
        wide[f"{condition}_minus_random4"] = wide[f"{condition}_weakening"] - wide["random4_weakening"]
    return wide


def _support_pairs(wide: pd.DataFrame) -> pd.DataFrame:
    support = wide[wide["node_role"].eq("support")]
    source = support[support["node_source"].eq("source")].set_index("pair_id")
    nearest = support[support["node_source"].eq("nearest_control")].set_index("pair_id")
    rows = []
    for pair_id in source.index.intersection(nearest.index):
        s = source.loc[pair_id]
        n = nearest.loc[pair_id]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[0]
        if isinstance(n, pd.DataFrame):
            n = n.iloc[0]
        row = {
            "pair_id": pair_id,
            "sample_id": s["sample_id"],
            "run": s["run"],
            "prompt_name": s["prompt_name"],
            "reasoning_operation": s.get("reasoning_operation", ""),
        }
        for condition in ["answer_mask", "relate_mask", "union_mask", "random4"]:
            key = f"{condition}_weakening" if condition != "random4" else "random4_weakening"
            row[f"source_{condition}_weakening"] = s.get(key, math.nan)
            row[f"nearest_{condition}_weakening"] = n.get(key, math.nan)
            row[f"source_minus_nearest_{condition}_weakening"] = (
                s.get(key, math.nan) - n.get(key, math.nan)
                if pd.notna(s.get(key, math.nan)) and pd.notna(n.get(key, math.nan))
                else math.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _random_coverage(wide: pd.DataFrame) -> pd.DataFrame:
    df = wide.copy()
    df["has_random4"] = df["valid_random_controls"].notna()
    return (
        df
        .groupby(["node_role", "node_source"], dropna=False)
        .agg(
            rows=("pair_id", "size"),
            samples=("sample_id", "nunique"),
            rows_with_random4=("has_random4", "sum"),
            samples_with_random4=("sample_id", lambda s: df.loc[s.index[df.loc[s.index, "has_random4"]], "sample_id"].nunique()),
            mean_valid_random_controls=("valid_random_controls", "mean"),
            max_random4_iou=("random4_max_iou", "max"),
        )
        .reset_index()
    )


def _summary_for_threshold(wide: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for role in ["support", "suppressor"]:
        for source in ["source", "nearest_control"]:
            sub = wide[(wide["node_role"].eq(role)) & (wide["node_source"].eq(source))]
            if sub.empty:
                continue
            for col in [
                "answer_mask_weakening",
                "relate_mask_weakening",
                "union_mask_weakening",
                "random4_weakening",
                "answer_mask_minus_random4",
                "union_mask_minus_random4",
            ]:
                if col in sub:
                    rows.append(_metric_row(f"{role}_{source}_{col}", sub[col], unit="pair"))
                    sample_values = sub.groupby("sample_id", dropna=False)[col].mean()
                    rows.append(_metric_row(f"{role}_{source}_{col}", sample_values, unit="sample_mean"))
    if not pairs.empty:
        for col in [
            "source_minus_nearest_answer_mask_weakening",
            "source_minus_nearest_relate_mask_weakening",
            "source_minus_nearest_union_mask_weakening",
            "source_minus_nearest_random4_weakening",
        ]:
            rows.append(_metric_row(col, pairs[col], unit="pair"))
            sample_values = pairs.groupby("sample_id", dropna=False)[col].mean()
            rows.append(_metric_row(col, sample_values, unit="sample_mean"))
    return pd.DataFrame(rows)


def _write_readout(
    out_path: Path,
    *,
    threshold: float,
    calibration: pd.DataFrame,
    coverage: pd.DataFrame,
    summary: pd.DataFrame,
    pairs: pd.DataFrame,
    wide: pd.DataFrame,
) -> None:
    key_metrics = summary[
        summary["metric"].isin(
            [
                "support_source_answer_mask_weakening",
                "support_source_union_mask_weakening",
                "support_source_answer_mask_minus_random4",
                "support_source_union_mask_minus_random4",
                "source_minus_nearest_answer_mask_weakening",
                "source_minus_nearest_union_mask_weakening",
            ]
        )
    ]
    support_sample = (
        wide[(wide["node_role"].eq("support")) & (wide["node_source"].eq("source"))]
        .groupby("sample_id", as_index=False)
        .agg(
            support_pairs=("pair_id", "size"),
            answer_weakening=("answer_mask_weakening", "mean"),
            union_weakening=("union_mask_weakening", "mean"),
            random4_weakening=("random4_weakening", "mean"),
            answer_minus_random4=("answer_mask_minus_random4", "mean"),
            union_minus_random4=("union_mask_minus_random4", "mean"),
            valid_random_controls=("valid_random_controls", "mean"),
        )
    )
    pair_cols = [
        "pair_id",
        "sample_id",
        "run",
        "source_answer_mask_weakening",
        "nearest_answer_mask_weakening",
        "source_minus_nearest_answer_mask_weakening",
        "source_union_mask_weakening",
        "nearest_union_mask_weakening",
        "source_minus_nearest_union_mask_weakening",
        "source_minus_nearest_random4_weakening",
    ]
    lines = [
        "# Stage 2A Region Route Replication Readout",
        "",
        f"- random IoU threshold: `{threshold}`",
        "",
        "## Clean Calibration",
        "",
        *calibration.to_markdown(index=False).splitlines(),
        "",
        "## Random-Control Coverage",
        "",
        *coverage.to_markdown(index=False).splitlines(),
        "",
        "## Key Metrics",
        "",
        *key_metrics.to_markdown(index=False).splitlines(),
        "",
        "## Support Source Sample Table",
        "",
        *support_sample.to_markdown(index=False).splitlines(),
        "",
        "## Support Source-Nearest Pair Table",
        "",
        *(pairs[pair_cols].to_markdown(index=False).splitlines() if not pairs.empty else ["No pairs."]),
        "",
        "## Interpretation Note",
        "",
        "- `answer_mask_weakening` 和 `union_mask_weakening` 为正，表示遮挡证据区域后对应 route 变弱。",
        "- `source_minus_nearest_*_weakening` 为正，表示 source route 的区域敏感性强于 nearest non-source control。",
        "- `random4` 只应在 `valid_random_controls` 覆盖足够时作为主对照；如果覆盖很低，只能作为 diagnostic。",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Stage 2A nearest8 region route replication.")
    parser.add_argument("--region-csv", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--random-iou-thresholds", default="0.05,0.10,0.20,1.0")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    region = _read_region(Path(args.region_csv).expanduser().resolve(), Path(args.manifest_csv).expanduser().resolve())
    calibration = _clean_calibration(region)
    calibration.to_csv(out_dir / "clean_calibration_summary.csv", index=False)

    for threshold_text in [x.strip() for x in args.random_iou_thresholds.split(",") if x.strip()]:
        threshold = float(threshold_text)
        label = threshold_text.replace(".", "p")
        wide = _route_wide(region, threshold)
        pairs = _support_pairs(wide)
        coverage = _random_coverage(wide)
        summary = _summary_for_threshold(wide, pairs)

        wide.to_csv(out_dir / f"route_weakening_iou{label}.csv", index=False)
        pairs.to_csv(out_dir / f"support_source_nearest_pairs_iou{label}.csv", index=False)
        coverage.to_csv(out_dir / f"random_coverage_iou{label}.csv", index=False)
        summary.to_csv(out_dir / f"metric_summary_iou{label}.csv", index=False)
        _write_readout(
            out_dir / f"STAGE2A_REGION_ROUTE_READOUT_iou{label}.md",
            threshold=threshold,
            calibration=calibration,
            coverage=coverage,
            summary=summary,
            pairs=pairs,
            wide=wide,
        )

    print(f"[done] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
