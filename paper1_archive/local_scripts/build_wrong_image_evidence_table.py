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


def _strength_from_delta(node_role: str, delta: float) -> float:
    if pd.isna(delta):
        return float("nan")
    if node_role == "support":
        return -float(delta)
    return float(delta)


def _route_agg(modality_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(modality_csv)
    df["run_name"] = df["run"].map(_map_run_name)
    grouped = (
        df[df["condition"].isin(["clean", "wrong_image"])]
        .groupby(["sample_id", "run_name", "node_role", "condition"], as_index=False)["delta_target_logit"]
        .mean()
    )
    grouped["strength"] = grouped.apply(lambda r: _strength_from_delta(r["node_role"], r["delta_target_logit"]), axis=1)
    wide = (
        grouped.pivot_table(
            index=["sample_id", "run_name", "node_role"],
            columns="condition",
            values=["delta_target_logit", "strength"],
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns = [
        "sample_id",
        "run_name",
        "node_role",
        "clean_delta_target_logit",
        "wrong_image_delta_target_logit",
        "clean_strength",
        "wrong_image_strength",
    ]
    wide["route_weakening"] = wide["clean_strength"] - wide["wrong_image_strength"]
    return wide


def _control_agg(matched_control_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(matched_control_csv)
    df["run_name"] = df["run"].map(_map_run_name)
    grouped = (
        df[df["condition"].isin(["clean", "wrong_image"])]
        .groupby(["sample_id", "run_name", "node_role", "condition"], as_index=False)["delta_target_logit"]
        .mean()
    )
    grouped["control_strength"] = grouped.apply(lambda r: _strength_from_delta(r["node_role"], r["delta_target_logit"]), axis=1)
    wide = (
        grouped.pivot_table(
            index=["sample_id", "run_name", "node_role"],
            columns="condition",
            values=["delta_target_logit", "control_strength"],
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns = [
        "sample_id",
        "run_name",
        "node_role",
        "clean_control_delta_target_logit",
        "wrong_image_control_delta_target_logit",
        "clean_control_strength",
        "wrong_image_control_strength",
    ]
    return wide


def _generation_wrong(gen_detail_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(gen_detail_csv)
    out = df[df["condition"] == "wrong_image"].copy()
    out = out[
        [
            "sample_id",
            "run",
            "predicted_answer",
            "clean_predicted_answer",
            "answer_changed_from_clean",
            "format_prefix_ok",
            "extra_newline_spillover",
            "empty_or_error",
            "vqa_score",
        ]
    ].copy()
    out["run_name"] = out["run"].map(_map_run_name)
    out = out.drop(columns=["run"])
    return out


def _rank_wrong(rank_detail_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(rank_detail_csv)
    out = df[df["condition"] == "wrong_image"].copy()
    out = out[
        [
            "sample_id",
            "run",
            "target_rank",
            "clean_target_rank",
            "target_prob",
            "clean_target_prob",
            "target_vs_competitor_margin",
            "clean_target_vs_competitor_margin",
            "margin_drop_vs_clean",
            "prob_drop_vs_clean",
            "target_in_top1",
            "target_in_top5",
            "rank_worsened_vs_clean",
        ]
    ].copy()
    return out


def _shape_wide(route_df: pd.DataFrame, control_df: pd.DataFrame) -> pd.DataFrame:
    merged = route_df.merge(control_df, on=["sample_id", "run_name", "node_role"], how="left")
    merged["clean_strength_gap"] = merged["clean_strength"] - merged["clean_control_strength"]
    merged["wrong_image_strength_gap"] = merged["wrong_image_strength"] - merged["wrong_image_control_strength"]
    merged["gap_drop"] = merged["clean_strength_gap"] - merged["wrong_image_strength_gap"]

    rows = []
    for _, row in merged.iterrows():
        prefix = str(row["node_role"])
        item = {
            "sample_id": row["sample_id"],
            "run_name": row["run_name"],
            f"{prefix}_clean_delta_target_logit": row["clean_delta_target_logit"],
            f"{prefix}_wrong_image_delta_target_logit": row["wrong_image_delta_target_logit"],
            f"{prefix}_clean_strength": row["clean_strength"],
            f"{prefix}_wrong_image_strength": row["wrong_image_strength"],
            f"{prefix}_route_weakening": row["route_weakening"],
            f"{prefix}_clean_control_delta_target_logit": row["clean_control_delta_target_logit"],
            f"{prefix}_wrong_image_control_delta_target_logit": row["wrong_image_control_delta_target_logit"],
            f"{prefix}_clean_control_strength": row["clean_control_strength"],
            f"{prefix}_wrong_image_control_strength": row["wrong_image_control_strength"],
            f"{prefix}_clean_strength_gap": row["clean_strength_gap"],
            f"{prefix}_wrong_image_strength_gap": row["wrong_image_strength_gap"],
            f"{prefix}_gap_drop": row["gap_drop"],
        }
        rows.append(item)
    wide = pd.DataFrame(rows)
    wide = wide.groupby(["sample_id", "run_name"], as_index=False).first()
    return wide


def _write_markdown(out_path: Path, subset_name: str, detail: pd.DataFrame) -> None:
    lines = [
        "# Wrong-Image Evidence Table",
        "",
        f"- subset: `{subset_name}`",
        "",
        "This table joins generation change, answer-token rank damage, source-route weakening, and matched-control gap change on the same sample-runs.",
        "",
    ]

    for run_name in ["B_direct", "D_visual_only"]:
        run_df = detail[detail["run_name"] == run_name].copy()
        if run_df.empty:
            continue
        run_df = run_df.sort_values(
            by=["answer_changed_from_clean", "wrong_image_vqa_score", "wrong_image_target_rank"],
            ascending=[False, True, False],
        )
        lines.extend([f"## {run_name}", ""])
        lines.append(
            "| sample | changed | wrong VQA | wrong rank | clean rank | margin drop | support weaken | suppressor weaken | support gap drop | suppressor gap drop |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in run_df.iterrows():
            lines.append(
                "| {sample} | {changed} | {vqa:.4f} | {wrong_rank:.0f} | {clean_rank:.0f} | {margin_drop:.4f} | {support_w:.4f} | {suppressor_w:.4f} | {support_gap:.4f} | {suppressor_gap:.4f} |".format(
                    sample=row["sample_id"],
                    changed=1 if bool(row["answer_changed_from_clean"]) else 0,
                    vqa=float(row["wrong_image_vqa_score"]),
                    wrong_rank=float(row["wrong_image_target_rank"]),
                    clean_rank=float(row["clean_target_rank"]),
                    margin_drop=float(row["margin_drop_vs_clean"]),
                    support_w=float(row.get("support_route_weakening", float("nan"))),
                    suppressor_w=float(row.get("suppressor_route_weakening", float("nan"))),
                    support_gap=float(row.get("support_gap_drop", float("nan"))),
                    suppressor_gap=float(row.get("suppressor_gap_drop", float("nan"))),
                )
            )
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a per-sample wrong-image evidence table by joining behavior, rank, route, and matched-control summaries.")
    parser.add_argument("--modality-csv", required=True)
    parser.add_argument("--matched-control-csv", required=True)
    parser.add_argument("--generation-detail-csv", required=True)
    parser.add_argument("--target-rank-detail-csv", required=True)
    parser.add_argument("--subset-name", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    _ensure_dir(out_dir)

    route_df = _route_agg(Path(args.modality_csv).expanduser().resolve())
    control_df = _control_agg(Path(args.matched_control_csv).expanduser().resolve())
    wide_df = _shape_wide(route_df, control_df)
    gen_df = _generation_wrong(Path(args.generation_detail_csv).expanduser().resolve())
    rank_df = _rank_wrong(Path(args.target_rank_detail_csv).expanduser().resolve())

    detail = wide_df.merge(gen_df, on=["sample_id", "run_name"], how="left").merge(
        rank_df,
        left_on=["sample_id", "run_name"],
        right_on=["sample_id", "run"],
        how="left",
    )
    if "run" in detail.columns:
        detail = detail.drop(columns=["run"])
    detail = detail.rename(
        columns={
            "predicted_answer": "wrong_image_predicted_answer",
            "clean_predicted_answer": "clean_predicted_answer",
            "vqa_score": "wrong_image_vqa_score",
            "target_rank": "wrong_image_target_rank",
            "target_prob": "wrong_image_target_prob",
            "target_vs_competitor_margin": "wrong_image_target_vs_competitor_margin",
        }
    )

    detail.to_csv(out_dir / "wrong_image_evidence_detail.csv", index=False)
    _write_markdown(out_dir / "wrong_image_evidence_table.md", args.subset_name, detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
