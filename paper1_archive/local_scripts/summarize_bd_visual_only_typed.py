#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_LABEL_COLS = [
    "question_type",
    "visual_structure",
    "image_dependence",
    "image_dependence_group",
    "reasoning_operation",
    "ambiguity_flag",
    "priority",
]

DEFAULT_SLICE_COLS = [
    "image_dependence_group",
    "reasoning_operation",
    "visual_structure",
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _maybe_read_csv(path_str: str | None) -> pd.DataFrame | None:
    if not path_str:
        return None
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        return None
    return pd.read_csv(path)


def _role_from_delta(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: "support" if pd.notna(x) and x < 0 else "suppressor")


def _attach_labels(
    df: pd.DataFrame,
    labels_df: pd.DataFrame,
    label_cols: list[str],
) -> pd.DataFrame:
    keep_cols = ["sample_id"] + [col for col in label_cols if col in labels_df.columns]
    return df.merge(labels_df[keep_cols].drop_duplicates(), on="sample_id", how="left")


def _slice_summary(
    df: pd.DataFrame,
    slice_cols: list[str],
    group_cols: list[str],
    metric_cols: list[str],
    value_name_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    overall = df.groupby(group_cols, dropna=False).agg(
        n_rows=("sample_id", "size"),
        n_samples=("sample_id", "nunique"),
        **{
            out_name: (src_name, "mean")
            for src_name, out_name in (value_name_map or {c: c for c in metric_cols}).items()
            if src_name in df.columns
        },
    ).reset_index()
    overall.insert(0, "slice_value", "all")
    overall.insert(0, "slice_col", "overall")
    rows.append(overall)

    for slice_col in slice_cols:
        if slice_col not in df.columns:
            continue
        part = df.groupby([slice_col] + group_cols, dropna=False).agg(
            n_rows=("sample_id", "size"),
            n_samples=("sample_id", "nunique"),
            **{
                out_name: (src_name, "mean")
                for src_name, out_name in (value_name_map or {c: c for c in metric_cols}).items()
                if src_name in df.columns
            },
        ).reset_index()
        part = part.rename(columns={slice_col: "slice_value"})
        part.insert(0, "slice_col", slice_col)
        rows.append(part)

    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["slice_col", "slice_value"] + group_cols).reset_index(drop=True)


def _format_float(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize B_direct vs D_visual_only alignment-clean results by typed slices."
    )
    parser.add_argument("--clean-subset-csv", required=True)
    parser.add_argument("--sample-ids-csv")
    parser.add_argument("--intervention-csv")
    parser.add_argument("--modality-csv")
    parser.add_argument("--matched-control-csv")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--label-cols",
        default=",".join(DEFAULT_LABEL_COLS),
        help="Comma-separated label columns to carry from the clean subset table.",
    )
    parser.add_argument(
        "--slice-cols",
        default=",".join(DEFAULT_SLICE_COLS),
        help="Comma-separated slice columns for one-dimensional typed summaries.",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    _ensure_dir(out_dir)

    label_cols = [c.strip() for c in args.label_cols.split(",") if c.strip()]
    slice_cols = [c.strip() for c in args.slice_cols.split(",") if c.strip()]

    clean_subset = pd.read_csv(Path(args.clean_subset_csv).expanduser().resolve())
    selected_sample_ids: set[str] | None = None
    if args.sample_ids_csv:
        sample_ids_df = pd.read_csv(Path(args.sample_ids_csv).expanduser().resolve())
        if "sample_id" not in sample_ids_df.columns:
            raise ValueError("sample_ids_csv must include a sample_id column")
        selected_sample_ids = set(sample_ids_df["sample_id"].astype(str))
        clean_subset = clean_subset[clean_subset["sample_id"].astype(str).isin(selected_sample_ids)].copy()

    sample_summary_rows: list[dict[str, object]] = []
    sample_summary_rows.append(
        {"slice_col": "overall", "slice_value": "all", "n_samples": int(clean_subset["sample_id"].nunique())}
    )
    for slice_col in slice_cols:
        if slice_col not in clean_subset.columns:
            continue
        counts = clean_subset.groupby(slice_col, dropna=False)["sample_id"].nunique().reset_index(name="n_samples")
        for _, row in counts.iterrows():
            sample_summary_rows.append(
                {"slice_col": slice_col, "slice_value": row[slice_col], "n_samples": int(row["n_samples"])}
            )
    sample_summary = pd.DataFrame(sample_summary_rows)
    sample_summary.to_csv(out_dir / "typed_sample_summary.csv", index=False)

    intervention = _maybe_read_csv(args.intervention_csv)
    modality = _maybe_read_csv(args.modality_csv)
    matched = _maybe_read_csv(args.matched_control_csv)

    generated_files = ["typed_sample_summary.csv"]
    markdown_lines = [
        "# B/D Typed Summary",
        "",
        f"- sample count: `{clean_subset['sample_id'].nunique()}`",
        f"- slice cols: `{', '.join(slice_cols)}`",
        "",
        "## Sample composition",
        "",
    ]
    for _, row in sample_summary.iterrows():
        markdown_lines.append(f"- `{row['slice_col']}` / `{row['slice_value']}`: `{row['n_samples']}`")

    if intervention is not None:
        if selected_sample_ids is not None and "sample_id" in intervention.columns:
            intervention = intervention[intervention["sample_id"].astype(str).isin(selected_sample_ids)].copy()
        intervention = _attach_labels(intervention, clean_subset, label_cols)
        if "node_role" not in intervention.columns and "delta_target_logit" in intervention.columns:
            intervention["node_role"] = _role_from_delta(intervention["delta_target_logit"])
        intervention["frac_negative"] = (intervention["delta_target_logit"] < 0).astype(float)
        intervention_summary = _slice_summary(
            intervention,
            slice_cols=slice_cols,
            group_cols=["run", "node_role"],
            metric_cols=["delta_target_logit", "frac_negative"],
            value_name_map={
                "delta_target_logit": "mean_delta_target_logit",
                "frac_negative": "frac_negative",
            },
        )
        intervention_summary.to_csv(out_dir / "typed_intervention_summary.csv", index=False)
        generated_files.append("typed_intervention_summary.csv")

        markdown_lines.extend(
            [
                "",
                "## Intervention summary",
                "",
            ]
        )
        for _, row in intervention_summary.iterrows():
            markdown_lines.append(
                "- "
                f"`{row['slice_col']}` / `{row['slice_value']}` / run `{row['run']}` / role `{row['node_role']}`: "
                f"`n_rows={row['n_rows']}`, `n_samples={row['n_samples']}`, "
                f"`mean_dlogit={_format_float(row.get('mean_delta_target_logit'))}`, "
                f"`frac_negative={_format_float(row.get('frac_negative'))}`"
            )

    if modality is not None:
        if selected_sample_ids is not None and "sample_id" in modality.columns:
            modality = modality[modality["sample_id"].astype(str).isin(selected_sample_ids)].copy()
        modality = _attach_labels(modality, clean_subset, label_cols)
        modality["frac_negative"] = (modality["delta_target_logit"] < 0).astype(float)
        modality_summary = _slice_summary(
            modality,
            slice_cols=slice_cols,
            group_cols=["run", "condition", "node_role"],
            metric_cols=["delta_target_logit", "frac_negative"],
            value_name_map={
                "delta_target_logit": "mean_delta_target_logit",
                "frac_negative": "frac_negative",
            },
        )
        modality_summary.to_csv(out_dir / "typed_modality_summary.csv", index=False)
        generated_files.append("typed_modality_summary.csv")

        markdown_lines.extend(
            [
                "",
                "## Modality summary",
                "",
            ]
        )
        for _, row in modality_summary.iterrows():
            markdown_lines.append(
                "- "
                f"`{row['slice_col']}` / `{row['slice_value']}` / run `{row['run']}` / "
                f"condition `{row['condition']}` / role `{row['node_role']}`: "
                f"`n_rows={row['n_rows']}`, `n_samples={row['n_samples']}`, "
                f"`mean_dlogit={_format_float(row.get('mean_delta_target_logit'))}`, "
                f"`frac_negative={_format_float(row.get('frac_negative'))}`"
            )

    if modality is not None and matched is not None:
        if selected_sample_ids is not None and "sample_id" in matched.columns:
            matched = matched[matched["sample_id"].astype(str).isin(selected_sample_ids)].copy()
        source_cols = [
            "sample_id",
            "run",
            "condition",
            "node_role",
            "feature_layer",
            "feature_pos",
            "feature_id",
            "delta_target_logit",
        ]
        source_df = modality[[c for c in source_cols if c in modality.columns]].copy()
        source_df = source_df.rename(
            columns={
                "feature_layer": "source_feature_layer",
                "feature_pos": "source_feature_pos",
                "feature_id": "source_feature_id",
                "delta_target_logit": "source_delta_target_logit",
            }
        )
        matched_join = matched.merge(
            source_df,
            on=[
                "sample_id",
                "run",
                "condition",
                "node_role",
                "source_feature_layer",
                "source_feature_pos",
                "source_feature_id",
            ],
            how="inner",
        )
        matched_join = _attach_labels(matched_join, clean_subset, label_cols)
        matched_join["source_minus_control_dlogit"] = (
            matched_join["source_delta_target_logit"] - matched_join["delta_target_logit"]
        )
        matched_join["source_abs_ge_control_abs"] = (
            matched_join["source_delta_target_logit"].abs() >= matched_join["delta_target_logit"].abs()
        ).astype(float)
        matched_summary = _slice_summary(
            matched_join,
            slice_cols=slice_cols,
            group_cols=["run", "condition", "node_role"],
            metric_cols=[
                "source_delta_target_logit",
                "delta_target_logit",
                "source_minus_control_dlogit",
                "source_abs_ge_control_abs",
            ],
            value_name_map={
                "source_delta_target_logit": "mean_source_delta_target_logit",
                "delta_target_logit": "mean_control_delta_target_logit",
                "source_minus_control_dlogit": "mean_source_minus_control_dlogit",
                "source_abs_ge_control_abs": "frac_source_abs_ge_control_abs",
            },
        )
        matched_summary.to_csv(out_dir / "typed_matched_control_summary.csv", index=False)
        generated_files.append("typed_matched_control_summary.csv")

        markdown_lines.extend(
            [
                "",
                "## Matched-control summary",
                "",
            ]
        )
        for _, row in matched_summary.iterrows():
            markdown_lines.append(
                "- "
                f"`{row['slice_col']}` / `{row['slice_value']}` / run `{row['run']}` / "
                f"condition `{row['condition']}` / role `{row['node_role']}`: "
                f"`n_rows={row['n_rows']}`, `n_samples={row['n_samples']}`, "
                f"`source={_format_float(row.get('mean_source_delta_target_logit'))}`, "
                f"`control={_format_float(row.get('mean_control_delta_target_logit'))}`, "
                f"`gap={_format_float(row.get('mean_source_minus_control_dlogit'))}`, "
                f"`frac_source_abs_ge_control_abs={_format_float(row.get('frac_source_abs_ge_control_abs'))}`"
            )

    markdown_lines.extend(
        [
            "",
            "## Files",
            "",
        ]
    )
    for name in generated_files:
        markdown_lines.append(f"- `{name}`")

    (out_dir / "typed_summary.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
