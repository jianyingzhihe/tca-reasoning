#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LOCAL_REASONING = {"symbol_text_reading", "visual_readout"}
LOCAL_STRUCTURES = {"single_core", "split_cores"}


def _read_many(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        p = Path(path).expanduser().resolve()
        if p.exists():
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _feature_key(layer, pos, feature_id) -> str:
    try:
        return f"L{int(layer)}:P{int(pos)}:F{int(feature_id)}"
    except Exception:
        return ""


def _role_strength(role: str, delta: float) -> float:
    if not np.isfinite(delta):
        return np.nan
    if role == "support":
        return -delta
    if role == "suppressor":
        return delta
    return np.nan


def _role_weakening(role: str, clean_delta: float, corrupt_delta: float) -> float:
    clean_strength = _role_strength(role, clean_delta)
    corrupt_strength = _role_strength(role, corrupt_delta)
    if not np.isfinite(clean_strength) or not np.isfinite(corrupt_strength):
        return np.nan
    return clean_strength - corrupt_strength


def _mask_stems(mask_root: str) -> set[str]:
    if not mask_root:
        return set()
    root = Path(mask_root).expanduser()
    if not root.exists():
        return set()
    return {p.name for p in root.iterdir() if p.is_dir()}


def _sample_to_image_stem(meta: pd.DataFrame) -> dict[str, str]:
    out = {}
    if "image_filename" in meta.columns:
        for _, row in meta.dropna(subset=["sample_id", "image_filename"]).iterrows():
            out[str(row["sample_id"])] = Path(str(row["image_filename"])).stem
    return out


def _build_nearest_rows(nearest: pd.DataFrame) -> pd.DataFrame:
    if nearest.empty:
        return pd.DataFrame()
    df = nearest.copy()
    df = df[df["condition"].isin(["clean", "wrong_image"])].copy()
    df["source_feature"] = [
        _feature_key(l, p, f)
        for l, p, f in zip(df["source_feature_layer"], df["source_feature_pos"], df["source_feature_id"])
    ]
    key = ["sample_id", "run", "node_role", "source_feature"]
    clean = df[df["condition"] == "clean"][key + ["delta_target_logit", "match_mode", "sampled_match_label"]].rename(
        columns={
            "delta_target_logit": "nearest_clean_control_delta",
            "match_mode": "nearest_match_mode",
            "sampled_match_label": "nearest_match_label",
        }
    )
    wrong = df[df["condition"] == "wrong_image"][key + ["delta_target_logit"]].rename(
        columns={"delta_target_logit": "nearest_wrong_control_delta"}
    )
    return clean.merge(wrong, on=key, how="left").drop_duplicates(key)


def _build_random_rows(random4: pd.DataFrame) -> pd.DataFrame:
    if random4.empty:
        return pd.DataFrame()
    df = random4.copy()
    df = df[df["condition"].isin(["clean", "wrong_image"])].copy()
    key = ["sample_id", "run", "node_role", "source_feature"]
    clean = df[df["condition"] == "clean"][
        key
        + [
            "n_control_draws",
            "mean_source_delta_target_logit",
            "mean_control_delta_target_logit",
            "mean_source_minus_control_dlogit",
            "frac_source_abs_ge_control_abs",
        ]
    ].rename(
        columns={
            "n_control_draws": "random4_clean_n_control_draws",
            "mean_source_delta_target_logit": "random4_clean_source_delta",
            "mean_control_delta_target_logit": "random4_clean_control_delta",
            "mean_source_minus_control_dlogit": "random4_clean_source_minus_control",
            "frac_source_abs_ge_control_abs": "random4_clean_frac_source_abs_ge_control_abs",
        }
    )
    wrong = df[df["condition"] == "wrong_image"][
        key
        + [
            "n_control_draws",
            "mean_source_delta_target_logit",
            "mean_control_delta_target_logit",
            "mean_source_minus_control_dlogit",
            "frac_source_abs_ge_control_abs",
        ]
    ].rename(
        columns={
            "n_control_draws": "random4_wrong_n_control_draws",
            "mean_source_delta_target_logit": "random4_wrong_source_delta",
            "mean_control_delta_target_logit": "random4_wrong_control_delta",
            "mean_source_minus_control_dlogit": "random4_wrong_source_minus_control",
            "frac_source_abs_ge_control_abs": "random4_wrong_frac_source_abs_ge_control_abs",
        }
    )
    return clean.merge(wrong, on=key, how="left").drop_duplicates(key)


def _build_source_rows(nearest: pd.DataFrame, random4: pd.DataFrame) -> pd.DataFrame:
    if nearest.empty:
        return pd.DataFrame()
    df = nearest.copy()
    df = df[df["condition"].isin(["clean", "wrong_image"])].copy()
    df["source_feature"] = [
        _feature_key(l, p, f)
        for l, p, f in zip(df["source_feature_layer"], df["source_feature_pos"], df["source_feature_id"])
    ]
    key = ["bucket", "sample_id", "run", "node_role", "source_feature"]
    meta_cols = [
        "question",
        "image_path",
        "target_token_id",
        "source_feature_layer",
        "source_feature_pos",
        "source_feature_id",
        "source_path_mass_best",
        "source_depth_from_target",
    ]
    clean = df[df["condition"] == "clean"][key + meta_cols + ["original_target_logit", "delta_target_logit"]].rename(
        columns={
            "original_target_logit": "clean_original_target_logit",
            "delta_target_logit": "clean_source_delta",
        }
    )
    wrong = df[df["condition"] == "wrong_image"][
        ["sample_id", "run", "node_role", "source_feature", "original_target_logit", "delta_target_logit"]
    ].rename(
        columns={
            "original_target_logit": "wrong_original_target_logit",
            "delta_target_logit": "wrong_source_delta",
        }
    )
    out = clean.merge(wrong, on=["sample_id", "run", "node_role", "source_feature"], how="left")
    out = out.merge(_build_nearest_rows(nearest), on=["sample_id", "run", "node_role", "source_feature"], how="left")
    out = out.merge(_build_random_rows(random4), on=["sample_id", "run", "node_role", "source_feature"], how="left")
    return out.drop_duplicates(["sample_id", "run", "node_role", "source_feature"])


def _add_meta(rows: pd.DataFrame, meta: pd.DataFrame, mask_root: str, already_region_samples: set[tuple[str, str]]) -> pd.DataFrame:
    if rows.empty:
        return rows
    meta_cols = [
        "sample_id",
        "question_text",
        "answer_text",
        "question_type",
        "visual_structure",
        "image_dependence",
        "image_dependence_group",
        "reasoning_operation",
        "ambiguity_flag",
        "priority",
        "B_predicted_answer",
        "D_predicted_answer",
    ]
    m = meta[[c for c in meta_cols if c in meta.columns]].drop_duplicates("sample_id")
    out = rows.merge(m, on="sample_id", how="left")
    stems = _mask_stems(mask_root)
    sample_stems = _sample_to_image_stem(meta)
    out["image_stem"] = out["sample_id"].map(sample_stems).fillna(out["image_path"].astype(str).map(lambda x: Path(x).stem))
    out["has_exported_mask"] = out["image_stem"].isin(stems)
    out["already_region_run"] = [(sid, run) in already_region_samples for sid, run in zip(out["sample_id"], out["run"])]
    return out


def _score(rows: pd.DataFrame, diffuse_samples: set[str], clean_threshold: float) -> pd.DataFrame:
    out = rows.copy()
    for col in [
        "clean_source_delta",
        "wrong_source_delta",
        "nearest_clean_control_delta",
        "nearest_wrong_control_delta",
        "random4_clean_control_delta",
        "random4_wrong_control_delta",
        "source_path_mass_best",
    ]:
        if col in out.columns:
            out[col] = _num(out[col])
    out["clean_source_strength"] = [_role_strength(r, d) for r, d in zip(out["node_role"], out["clean_source_delta"])]
    out["wrong_source_strength"] = [_role_strength(r, d) for r, d in zip(out["node_role"], out["wrong_source_delta"])]
    out["wrong_image_route_weakening"] = [
        _role_weakening(r, c, w) for r, c, w in zip(out["node_role"], out["clean_source_delta"], out["wrong_source_delta"])
    ]
    out["nearest_clean_control_strength"] = [
        _role_strength(r, d) for r, d in zip(out["node_role"], out["nearest_clean_control_delta"])
    ]
    out["random4_clean_control_strength"] = [
        _role_strength(r, d) for r, d in zip(out["node_role"], out["random4_clean_control_delta"])
    ]
    out["clean_source_minus_nearest_strength"] = out["clean_source_strength"] - out["nearest_clean_control_strength"]
    out["clean_source_minus_random4_strength"] = out["clean_source_strength"] - out["random4_clean_control_strength"]

    out["localized_type_ok"] = out["reasoning_operation"].isin(LOCAL_REASONING) & out["visual_structure"].isin(LOCAL_STRUCTURES)
    out["not_user_or_meta_diffuse"] = ~out["sample_id"].isin(diffuse_samples) & ~out["visual_structure"].astype(str).str.contains(
        "diffuse", case=False, na=False
    )
    out["clean_source_nonzero"] = out["clean_source_strength"].abs() >= clean_threshold
    out["clean_source_role_consistent"] = out["clean_source_strength"] >= clean_threshold
    out["has_nearest_control"] = out["nearest_clean_control_delta"].notna()
    out["has_random4_control"] = out["random4_clean_control_delta"].notna()
    out["wrong_image_sensitive"] = out["wrong_image_route_weakening"] > 0
    out["source_beats_nearest_clean"] = out["clean_source_minus_nearest_strength"] > 0
    out["source_beats_random4_clean"] = out["clean_source_minus_random4_strength"] > 0

    out["primary_support_ready_existing_mask"] = (
        out["node_role"].eq("support")
        & out["localized_type_ok"]
        & out["not_user_or_meta_diffuse"]
        & out["clean_source_role_consistent"]
        & out["has_nearest_control"]
        & out["has_random4_control"]
        & out["wrong_image_sensitive"]
        & out["has_exported_mask"]
    )
    out["primary_support_needs_mask"] = (
        out["node_role"].eq("support")
        & out["localized_type_ok"]
        & out["not_user_or_meta_diffuse"]
        & out["clean_source_role_consistent"]
        & out["has_nearest_control"]
        & out["has_random4_control"]
        & out["wrong_image_sensitive"]
        & ~out["has_exported_mask"]
    )
    out["secondary_suppressor_ready_existing_mask"] = (
        out["node_role"].eq("suppressor")
        & out["localized_type_ok"]
        & out["not_user_or_meta_diffuse"]
        & out["clean_source_role_consistent"]
        & out["has_nearest_control"]
        & out["has_random4_control"]
        & out["wrong_image_sensitive"]
        & out["has_exported_mask"]
    )
    out["candidate_score"] = (
        out["localized_type_ok"].astype(float) * 25
        + out["not_user_or_meta_diffuse"].astype(float) * 20
        + out["clean_source_role_consistent"].astype(float) * 20
        + out["has_nearest_control"].astype(float) * 10
        + out["has_random4_control"].astype(float) * 10
        + out["wrong_image_sensitive"].astype(float) * 15
        + out["has_exported_mask"].astype(float) * 8
        + out["source_beats_nearest_clean"].astype(float) * 8
        + out["source_beats_random4_clean"].astype(float) * 8
        + out["clean_source_strength"].fillna(0).clip(lower=0, upper=5)
        + out["wrong_image_route_weakening"].fillna(0).clip(lower=0, upper=5)
    )
    reason_parts = []
    for _, row in out.iterrows():
        parts = []
        for flag, label in [
            ("localized_type_ok", "localized-type"),
            ("not_user_or_meta_diffuse", "not-diffuse"),
            ("clean_source_nonzero", "clean-source-nonzero"),
            ("clean_source_role_consistent", "clean-source-role-consistent"),
            ("has_nearest_control", "nearest-control"),
            ("has_random4_control", "random4-control"),
            ("wrong_image_sensitive", "wrong-image-sensitive"),
            ("has_exported_mask", "has-mask"),
            ("source_beats_nearest_clean", "source>nearest-clean"),
            ("source_beats_random4_clean", "source>random4-clean"),
        ]:
            if bool(row.get(flag, False)):
                parts.append(label)
        reason_parts.append(";".join(parts))
    out["candidate_flags"] = reason_parts
    return out.sort_values("candidate_score", ascending=False)


def _sample_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    agg = rows.groupby("sample_id", as_index=False).agg(
        question_text=("question_text", "first"),
        answer_text=("answer_text", "first"),
        reasoning_operation=("reasoning_operation", "first"),
        visual_structure=("visual_structure", "first"),
        image_dependence=("image_dependence", "first"),
        has_exported_mask=("has_exported_mask", "max"),
        already_region_run=("already_region_run", "max"),
        n_source_rows=("source_feature", "count"),
        n_support_rows=("node_role", lambda s: int((s == "support").sum())),
        n_suppressor_rows=("node_role", lambda s: int((s == "suppressor").sum())),
        n_primary_support_ready_existing_mask=("primary_support_ready_existing_mask", "sum"),
        n_primary_support_needs_mask=("primary_support_needs_mask", "sum"),
        n_secondary_suppressor_ready_existing_mask=("secondary_suppressor_ready_existing_mask", "sum"),
        max_candidate_score=("candidate_score", "max"),
        max_clean_source_strength=("clean_source_strength", "max"),
        max_wrong_image_route_weakening=("wrong_image_route_weakening", "max"),
    )
    return agg.sort_values(["n_primary_support_ready_existing_mask", "n_primary_support_needs_mask", "max_candidate_score"], ascending=False)


def _write_markdown(path: Path, rows: pd.DataFrame, samples: pd.DataFrame, clean_threshold: float) -> None:
    ready_support = int(rows["primary_support_ready_existing_mask"].sum()) if not rows.empty else 0
    needs_support = int(rows["primary_support_needs_mask"].sum()) if not rows.empty else 0
    ready_suppressor = int(rows["secondary_suppressor_ready_existing_mask"].sum()) if not rows.empty else 0
    ready_support_samples = int(rows.loc[rows["primary_support_ready_existing_mask"], "sample_id"].nunique()) if ready_support else 0
    needs_support_samples = int(rows.loc[rows["primary_support_needs_mask"], "sample_id"].nunique()) if needs_support else 0
    lines = [
        "# Strict Region Candidate Miner",
        "",
        "Purpose: decide whether we can run another region-evidence pack now, or whether we need a targeted annotation round.",
        "",
        f"- clean source threshold: `{clean_threshold}` role-strength units",
        f"- primary support rows ready with existing masks: `{ready_support}` rows / `{ready_support_samples}` samples",
        f"- primary support rows needing masks: `{needs_support}` rows / `{needs_support_samples}` samples",
        f"- secondary suppressor rows ready with existing masks: `{ready_suppressor}` rows",
        "",
        "Primary support-ready rows:",
        "",
    ]
    view_cols = [
        "sample_id",
        "run",
        "node_role",
        "reasoning_operation",
        "visual_structure",
        "clean_source_strength",
        "wrong_image_route_weakening",
        "clean_source_minus_nearest_strength",
        "clean_source_minus_random4_strength",
        "has_exported_mask",
        "already_region_run",
        "candidate_flags",
        "question_text",
        "answer_text",
    ]
    support_ready = rows[rows["primary_support_ready_existing_mask"]]
    if support_ready.empty:
        lines.append("No primary support rows are fully ready under the strict filter.")
    else:
        lines.extend(support_ready[view_cols].head(20).to_markdown(index=False).splitlines())
    lines.extend(["", "Primary support rows that would need masks:", ""])
    support_needs = rows[rows["primary_support_needs_mask"]]
    if support_needs.empty:
        lines.append("No unmasked primary support rows passed the strict filter.")
    else:
        lines.extend(support_needs[view_cols].head(20).to_markdown(index=False).splitlines())
    lines.extend(["", "Secondary suppressor-ready rows:", ""])
    supp_ready = rows[rows["secondary_suppressor_ready_existing_mask"]]
    if supp_ready.empty:
        lines.append("No secondary suppressor rows are fully ready under the strict filter.")
    else:
        lines.extend(supp_ready[view_cols].head(20).to_markdown(index=False).splitlines())
    lines.extend(
        [
            "",
            "Sample-level summary:",
            "",
            *samples.head(20).to_markdown(index=False).splitlines(),
            "",
            "Decision rule:",
            "",
            "- If strict primary support ready samples are at least 8, run another region experiment without new labeling.",
            "- If strict primary support needing-mask samples are at least 8, build a targeted annotation pack.",
            "- If both are below 8, do not spend annotation time yet; the bottleneck is candidate quality, not masks.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine strict localized region-evidence candidates from existing visual-positive experiments.")
    parser.add_argument("--nearest-csv", action="append", required=True)
    parser.add_argument("--random4-per-source-csv", action="append", required=True)
    parser.add_argument("--selected-meta-csv", action="append", required=True)
    parser.add_argument("--mask-root", default="")
    parser.add_argument("--region-case-csv", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--diffuse-sample-id", action="append", default=[])
    parser.add_argument("--clean-threshold", type=float, default=0.1)
    args = parser.parse_args()

    nearest = _read_many(args.nearest_csv)
    random4 = _read_many(args.random4_per_source_csv)
    meta = _read_many(args.selected_meta_csv)
    already_region = set()
    if args.region_case_csv and Path(args.region_case_csv).exists():
        rc = pd.read_csv(args.region_case_csv)
        already_region = set(zip(rc["sample_id"].astype(str), rc["run"].astype(str)))

    rows = _build_source_rows(nearest, random4)
    rows = _add_meta(rows, meta, args.mask_root, already_region)
    rows = _score(rows, set(args.diffuse_sample_id), args.clean_threshold)
    samples = _sample_summary(rows)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_dir / "strict_region_candidate_source_rows.csv", index=False)
    samples.to_csv(out_dir / "strict_region_candidate_samples.csv", index=False)
    _write_markdown(out_dir / "STRICT_REGION_CANDIDATE_MINER_READOUT.md", rows, samples, args.clean_threshold)

    print(f"[done] wrote {out_dir}")
    print(f"[summary] primary_support_ready_rows={int(rows['primary_support_ready_existing_mask'].sum())}")
    print(f"[summary] primary_support_needs_mask_rows={int(rows['primary_support_needs_mask'].sum())}")
    print(f"[summary] secondary_suppressor_ready_rows={int(rows['secondary_suppressor_ready_existing_mask'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
