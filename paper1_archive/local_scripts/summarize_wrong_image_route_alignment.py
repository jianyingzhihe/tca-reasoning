#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RUN_MAP = {
    "A": "D_visual_only",
    "B": "B_direct",
    "D_visual_only": "D_visual_only",
    "B_direct": "B_direct",
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _map_run_name(value: object) -> str:
    return RUN_MAP.get(str(value), str(value))


def _compute_weakening(row: pd.Series) -> float:
    if row["node_role"] == "support":
        return float(row["wrong_delta"] - row["clean_delta"])
    return float(row["clean_delta"] - row["wrong_delta"])


def build_joined_detail(modality_csv: Path, condition_detail_csv: Path) -> pd.DataFrame:
    modality = pd.read_csv(modality_csv)
    condition = pd.read_csv(condition_detail_csv)

    modality = modality[modality["condition"].isin(["clean", "wrong_image"])].copy()
    modality["run_name"] = modality["run"].map(_map_run_name)

    sample_role_condition = (
        modality.groupby(["sample_id", "run_name", "node_role", "condition"], as_index=False)["delta_target_logit"]
        .mean()
    )
    clean = sample_role_condition[sample_role_condition["condition"] == "clean"][
        ["sample_id", "run_name", "node_role", "delta_target_logit"]
    ].rename(columns={"delta_target_logit": "clean_delta"})
    wrong = sample_role_condition[sample_role_condition["condition"] == "wrong_image"][
        ["sample_id", "run_name", "node_role", "delta_target_logit"]
    ].rename(columns={"delta_target_logit": "wrong_delta"})
    joined = clean.merge(wrong, on=["sample_id", "run_name", "node_role"], how="inner")
    joined["weakening"] = joined.apply(_compute_weakening, axis=1)

    wrong_condition = condition[condition["condition"] == "wrong_image"].copy()
    wrong_condition["run_name"] = wrong_condition["run"].map(_map_run_name)
    wrong_condition = wrong_condition[
        [
            "sample_id",
            "run_name",
            "predicted_answer",
            "clean_predicted_answer",
            "answer_changed_from_clean",
            "format_prefix_ok",
            "extra_newline_spillover",
            "empty_or_error",
            "vqa_score",
        ]
    ]
    joined = joined.merge(wrong_condition, on=["sample_id", "run_name"], how="left")
    joined["prefix_fail"] = ~joined["format_prefix_ok"].fillna(False)
    return joined


def summarize_joined(detail: pd.DataFrame) -> pd.DataFrame:
    return (
        detail.groupby(["run_name", "node_role", "answer_changed_from_clean"], dropna=False)
        .agg(
            n_rows=("sample_id", "size"),
            mean_clean_delta=("clean_delta", "mean"),
            mean_wrong_delta=("wrong_delta", "mean"),
            mean_weakening=("weakening", "mean"),
            mean_wrong_vqa=("vqa_score", "mean"),
            empty_or_error_count=("empty_or_error", "sum"),
            prefix_fail_count=("prefix_fail", "sum"),
            spillover_count=("extra_newline_spillover", "sum"),
        )
        .reset_index()
    )


def write_markdown(out_path: Path, subset_name: str, summary: pd.DataFrame) -> None:
    lines = [
        "# Wrong-Image Route Alignment Summary",
        "",
        f"- subset: `{subset_name}`",
        "",
    ]
    for run_name in ["B_direct", "D_visual_only"]:
        run_df = summary[summary["run_name"] == run_name]
        if run_df.empty:
            continue
        lines.extend([f"## {run_name}", ""])
        for node_role in ["support", "suppressor"]:
            role_df = run_df[run_df["node_role"] == node_role]
            if role_df.empty:
                continue
            lines.extend([f"### {node_role}", ""])
            for _, row in role_df.iterrows():
                label = "changed" if bool(row["answer_changed_from_clean"]) else "unchanged"
                n = int(row["n_rows"])
                lines.extend(
                    [
                        f"- `{label}` rows: `{n}`",
                        f"  mean clean `delta_target_logit = {float(row['mean_clean_delta']):.4f}`",
                        f"  mean wrong-image `delta_target_logit = {float(row['mean_wrong_delta']):.4f}`",
                        f"  mean weakening `= {float(row['mean_weakening']):.4f}`",
                        f"  mean wrong-image `vqa_score = {float(row['mean_wrong_vqa']):.4f}`",
                        f"  empty/error `{int(row['empty_or_error_count'])}/{n}`",
                        f"  prefix fail `{int(row['prefix_fail_count'])}/{n}`",
                        f"  spillover `{int(row['spillover_count'])}/{n}`",
                    ]
                )
            lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Join wrong-image generation changes with per-sample route weakening from modality pilot outputs."
    )
    parser.add_argument("--modality-csv", required=True)
    parser.add_argument("--condition-detail-csv", required=True)
    parser.add_argument("--subset-name", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    _ensure_dir(out_dir)

    detail = build_joined_detail(
        Path(args.modality_csv).expanduser().resolve(),
        Path(args.condition_detail_csv).expanduser().resolve(),
    )
    summary = summarize_joined(detail)

    detail.to_csv(out_dir / "wrong_image_route_alignment_detail.csv", index=False)
    summary.to_csv(out_dir / "wrong_image_route_alignment_summary.csv", index=False)
    write_markdown(out_dir / "wrong_image_route_alignment_summary.md", args.subset_name, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
