#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fmt(value: float | int | str) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _truthy_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _gap_table(source: pd.DataFrame, control: pd.DataFrame, control_name: str) -> pd.DataFrame:
    src = (
        source.groupby(["sample_id", "run", "node_role", "condition"], dropna=False)
        .agg(source_n=("delta_target_logit", "size"), source_mean=("delta_target_logit", "mean"))
        .reset_index()
    )
    ctrl = (
        control.groupby(["sample_id", "run", "node_role", "condition"], dropna=False)
        .agg(control_n=("delta_target_logit", "size"), control_mean=("delta_target_logit", "mean"))
        .reset_index()
    )
    out = src.merge(ctrl, on=["sample_id", "run", "node_role", "condition"], how="inner")
    out["control"] = control_name
    out["source_minus_control"] = out["source_mean"] - out["control_mean"]
    return out[
        [
            "control",
            "sample_id",
            "run",
            "node_role",
            "condition",
            "source_n",
            "source_mean",
            "control_n",
            "control_mean",
            "source_minus_control",
        ]
    ]


def _weakening_table(source: pd.DataFrame) -> pd.DataFrame:
    src = (
        source.groupby(["sample_id", "run", "node_role", "condition"], dropna=False)
        .agg(n=("delta_target_logit", "size"), mean_delta=("delta_target_logit", "mean"))
        .reset_index()
    )
    wide = src.pivot_table(
        index=["sample_id", "run", "node_role"],
        columns="condition",
        values="mean_delta",
        aggfunc="mean",
    ).reset_index()
    if "clean" not in wide.columns:
        wide["clean"] = pd.NA
    if "wrong_image" not in wide.columns:
        wide["wrong_image"] = pd.NA
    # Positive support_weakening means support becomes less negative under wrong_image.
    wide["support_style_weakening"] = wide["wrong_image"] - wide["clean"]
    # Positive suppressor_weakening means suppressor becomes less positive under wrong_image.
    wide["suppressor_style_weakening"] = wide["clean"] - wide["wrong_image"]
    return wide


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize the round400 localized discovery clean-screen run.")
    parser.add_argument("--sync-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    sync_dir = Path(args.sync_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    clean = _read(sync_dir / "clean_subset" / "alignment_clean_round400_localized_discovery_screen7.csv")
    smoke = _read(sync_dir / "intervention_smoke_round400_localized_discovery_screen7_clean_core_thresh4_top3.csv")
    source = _read(sync_dir / "modality_pilot_round400_localized_discovery_screen7_clean_core_thresh4_top3_clean_wrong.csv")
    nearest = _read(sync_dir / "modality_matched_control_nearest_round400_localized_discovery_screen7_clean_core_thresh4_top3_clean_wrong.csv")
    random4 = _read(sync_dir / "modality_matched_control_random4_round400_localized_discovery_screen7_clean_core_thresh4_top3_clean_wrong.csv")

    clean_core = _truthy_series(clean["clean_core"])
    clean_prompt = _truthy_series(clean["clean_prompt"])
    clean["feature_pair"] = clean_core & (clean["feature_count_a"].astype(int) > 0) & (clean["feature_count_b"].astype(int) > 0)
    clean_summary = {
        "total": len(clean),
        "clean_core": int(clean_core.sum()),
        "clean_prompt": int(clean_prompt.sum()),
        "feature_pair": int(clean["feature_pair"].sum()),
    }

    smoke_summary = (
        smoke.groupby(["sample_id", "run"], dropna=False)
        .agg(n=("delta_target_logit", "size"), mean=("delta_target_logit", "mean"), min=("delta_target_logit", "min"), max=("delta_target_logit", "max"))
        .reset_index()
    )
    source_summary = (
        source.groupby(["sample_id", "run", "node_role", "condition"], dropna=False)
        .agg(n=("delta_target_logit", "size"), source_mean=("delta_target_logit", "mean"))
        .reset_index()
    )
    gap = pd.concat([_gap_table(source, nearest, "nearest"), _gap_table(source, random4, "random4")], ignore_index=True)
    weak = _weakening_table(source)

    clean.to_csv(out_dir / "clean_alignment_table.csv", index=False)
    smoke_summary.to_csv(out_dir / "smoke_summary.csv", index=False)
    source_summary.to_csv(out_dir / "source_modality_summary.csv", index=False)
    gap.to_csv(out_dir / "source_control_gap_summary.csv", index=False)
    weak.to_csv(out_dir / "source_wrong_image_weakening_summary.csv", index=False)

    support_wrong_gap = gap[(gap["node_role"] == "support") & (gap["condition"] == "wrong_image")]
    support_wrong_gap_md = support_wrong_gap.to_markdown(index=False)
    source_summary_md = source_summary.to_markdown(index=False)
    smoke_summary_md = smoke_summary.to_markdown(index=False)
    clean_rows_md = clean[
        [
            "sample_id",
            "clean_core",
            "clean_prompt",
            "feature_count_a",
            "feature_count_b",
            "a_assistant_prefix_canonical",
            "b_assistant_prefix_canonical",
        ]
    ].to_markdown(index=False)
    weak_md = weak.to_markdown(index=False)

    lines = [
        "# Round400 Localized Discovery Screen Readout",
        "",
        "## Status",
        "",
        "This run was a discovery screen, not a new main experiment branch.",
        "",
        f"- total traced candidates: `{clean_summary['total']}`",
        f"- clean-core samples: `{clean_summary['clean_core']}`",
        f"- clean-prompt samples: `{clean_summary['clean_prompt']}`",
        f"- clean-core samples with feature nodes in both A/B: `{clean_summary['feature_pair']}`",
        f"- intervention smoke rows: `{len(smoke)}`",
        f"- modality source rows: `{len(source)}`",
        f"- nearest-control rows: `{len(nearest)}`",
        f"- random4-control rows: `{len(random4)}`",
        "",
        "## Clean Alignment",
        "",
        clean_rows_md,
        "",
        "Read: `okvqa_val_03358`, `okvqa_val_04424`, and `okvqa_val_03435` were clean-core feature-pair candidates, but only `okvqa_val_03358` produced a balanced signed source-route screen. `okvqa_val_03435` also has a non-canonical D-side assistant prefix.",
        "",
        "## Intervention Smoke",
        "",
        smoke_summary_md,
        "",
        "Read: the successful source rows concentrate on `okvqa_val_03358`; `okvqa_val_03435` contributes only one B-side suppressor-like row, and `okvqa_val_04424` does not survive the non-generic source screen.",
        "",
        "## Source Modality Summary",
        "",
        source_summary_md,
        "",
        "Read: for `okvqa_val_03358`, both support and suppressor routes are strong. Aggregate support becomes less negative under `wrong_image`, but individual support nodes are heterogeneous. Suppressor routes become stronger rather than collapsing in this case.",
        "",
        "## Wrong-Image Weakening",
        "",
        weak_md,
        "",
        "For support rows, positive `support_style_weakening` means source support is weaker under `wrong_image`. For suppressor rows, positive `suppressor_style_weakening` means suppressor is weaker under `wrong_image`.",
        "",
        "## Support Source-Control Gap Under Wrong Image",
        "",
        support_wrong_gap_md,
        "",
        "Read: under `wrong_image`, the `okvqa_val_03358` support source rows remain stronger than both nearest and random4 controls in both prompts. However, because this rests on a single sample, it should be treated as a discovery case, not a new aggregate claim.",
        "",
        "## Decision",
        "",
        "Do not request a new annotation batch from this screen yet.",
        "",
        "The run is useful because it identified `okvqa_val_03358` as a figure/case-study candidate with signed routes and source-over-control support specificity under `wrong_image`. It is not sufficient as a multi-sample localized support pool. The next productive step is either to use `okvqa_val_03358` as a small case panel, or mine a broader localized candidate pool before asking for more masks.",
    ]
    _write_md(out_dir / "ROUND400_LOCALIZED_DISCOVERY_SCREEN_READOUT.md", "\n".join(lines) + "\n")
    print(f"[done] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
