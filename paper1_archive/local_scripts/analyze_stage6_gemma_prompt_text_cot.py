#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(r"E:\Bridging")
CROSS = ROOT / "doc" / "experiments" / "stage6" / "cross_model"
PREFIX = "stage6_gemma_prompt_text_cot"

METRICS = [
    "node_overlap_jaccard",
    "edge_overlap_jaccard",
    "delta_target_total_in_abs",
    "delta_target_feature_ratio",
    "delta_target_error_ratio",
    "delta_traced_nodes",
    "delta_traced_edges",
    "a_traced_nodes",
    "b_traced_nodes",
    "a_traced_edges",
    "b_traced_edges",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _ci95(series: pd.Series) -> tuple[float, float]:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    n = len(vals)
    if n == 0:
        return (math.nan, math.nan)
    mean = float(vals.mean())
    if n == 1:
        return (mean, mean)
    se = float(vals.std(ddof=1)) / math.sqrt(n)
    return (mean - 1.96 * se, mean + 1.96 * se)


def _summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row: dict[str, Any] = {col: key[i] for i, col in enumerate(group_cols)}
        row["n"] = len(sub)
        row["unique_original_samples"] = sub["stage6_original_sample_id"].nunique() if "stage6_original_sample_id" in sub else ""
        for metric in METRICS:
            if metric not in sub:
                continue
            vals = pd.to_numeric(sub[metric], errors="coerce")
            low, high = _ci95(vals)
            row[f"{metric}_mean"] = float(vals.mean()) if vals.notna().any() else math.nan
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
            if metric.startswith("delta_"):
                row[f"{metric}_positive_frac"] = float((vals > 0).mean()) if vals.notna().any() else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _rewrite_stability(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    keys = ["stage6_original_sample_id", "stage6_prompt_family"]
    rows: list[dict[str, Any]] = []
    for key, sub in df.groupby(keys, dropna=False):
        original = sub[sub["stage6_question_variant"] == "original"]
        if original.empty:
            continue
        base = original.iloc[0]
        for variant in ["paraphrase_1", "paraphrase_2"]:
            other = sub[sub["stage6_question_variant"] == variant]
            if other.empty:
                continue
            row: dict[str, Any] = {
                "stage6_original_sample_id": key[0],
                "stage6_prompt_family": key[1],
                "stage6_question_variant": variant,
            }
            other_row = other.iloc[0]
            for metric in METRICS:
                if metric in sub:
                    row[f"{metric}_delta_vs_original"] = pd.to_numeric(pd.Series([other_row[metric]]), errors="coerce").iloc[0] - pd.to_numeric(pd.Series([base[metric]]), errors="coerce").iloc[0]
            rows.append(row)
    return pd.DataFrame(rows)


def _summarize_rewrite_deltas(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    metric_cols = [col for col in df.columns if col.endswith("_delta_vs_original")]
    rows: list[dict[str, Any]] = []
    for key, sub in df.groupby(["stage6_prompt_family", "stage6_question_variant"], dropna=False):
        row: dict[str, Any] = {
            "stage6_prompt_family": key[0],
            "stage6_question_variant": key[1],
            "n": len(sub),
            "unique_original_samples": sub["stage6_original_sample_id"].nunique(),
        }
        for metric in metric_cols:
            vals = pd.to_numeric(sub[metric], errors="coerce")
            low, high = _ci95(vals)
            row[f"{metric}_mean"] = float(vals.mean()) if vals.notna().any() else math.nan
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
            row[f"{metric}_positive_frac"] = float((vals > 0).mean()) if vals.notna().any() else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def analyze(mode: str, tag: str) -> dict[str, Any]:
    prefix = f"{PREFIX}_{mode}_{tag}"
    manifest = _read_csv(CROSS / f"{PREFIX}_{tag}_manifest.csv")
    decision = _read_json(CROSS / f"{prefix}_decision.json")
    sample_compare = _read_csv(CROSS / f"{prefix}_sample_compare_controlled.csv")
    failures = _read_csv(CROSS / f"{prefix}_failure_manifest.csv")

    if sample_compare.empty:
        payload = {
            "status": "blocked",
            "reason": "sample_compare_controlled missing or empty",
            "mode": mode,
            "tag": tag,
            "remote_decision": decision,
        }
        (CROSS / f"{prefix}_analysis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return payload

    if not manifest.empty:
        meta_cols = [
            "sample_id",
            "stage6_original_sample_id",
            "stage6_question_variant",
            "stage6_prompt_family",
            "stage6_sample_type",
            "stage6_include_smoke",
            "stage6_include_full",
            "answer_text",
        ]
        meta = manifest[[col for col in meta_cols if col in manifest.columns]].drop_duplicates("sample_id")
        sample_compare = sample_compare.merge(meta, on="sample_id", how="left")

    overall = _summarize(sample_compare, [])
    by_prompt = _summarize(sample_compare, ["stage6_prompt_family"])
    by_variant = _summarize(sample_compare, ["stage6_question_variant"])
    by_prompt_variant = _summarize(sample_compare, ["stage6_prompt_family", "stage6_question_variant"])
    by_type = _summarize(sample_compare, ["stage6_sample_type"])
    rewrite = _rewrite_stability(sample_compare)
    rewrite_summary = _summarize_rewrite_deltas(rewrite)

    overall.to_csv(CROSS / f"{prefix}_summary_overall.csv", index=False)
    by_prompt.to_csv(CROSS / f"{prefix}_summary_by_prompt.csv", index=False)
    by_variant.to_csv(CROSS / f"{prefix}_summary_by_variant.csv", index=False)
    by_prompt_variant.to_csv(CROSS / f"{prefix}_summary_by_prompt_variant.csv", index=False)
    by_type.to_csv(CROSS / f"{prefix}_summary_by_type.csv", index=False)
    rewrite.to_csv(CROSS / f"{prefix}_rewrite_deltas.csv", index=False)
    rewrite_summary.to_csv(CROSS / f"{prefix}_rewrite_stability_summary.csv", index=False)

    node_mean = float(overall.get("node_overlap_jaccard_mean", pd.Series([math.nan])).iloc[0])
    edge_mean = float(overall.get("edge_overlap_jaccard_mean", pd.Series([math.nan])).iloc[0])
    rows = int(len(sample_compare))
    unique_samples = int(sample_compare["stage6_original_sample_id"].nunique()) if "stage6_original_sample_id" in sample_compare else 0
    valid_rate = rows / max(1, rows + len(failures))
    text_stability_hint = False
    prompt_modulation_hint = False
    rewrite_node_delta_col = "node_overlap_jaccard_delta_vs_original_mean"
    if not rewrite_summary.empty and rewrite_node_delta_col in rewrite_summary:
        text_stability_hint = bool((rewrite_summary[rewrite_node_delta_col].abs() < 0.08).all())
    if not by_prompt.empty and "node_overlap_jaccard_mean" in by_prompt:
        vals = by_prompt.set_index("stage6_prompt_family")["node_overlap_jaccard_mean"].dropna()
        if "B_direct" in vals.index and len(vals) > 1:
            prompt_modulation_hint = bool((vals.max() - vals.min()) > 0.10)

    sharded_streaming = bool(decision.get("sharded_streaming")) or tag.endswith("_sharded")
    status = "gemma_stage6_full_sharded_streaming" if sharded_streaming and mode == "full" else "gemma_stage6_source_tracing_prompt_text_cot_analyzed"
    if valid_rate < 0.7 or rows < 10:
        status = "gemma_stage6_diagnostic_small_or_noisy"

    payload = {
        "status": status,
        "mode": mode,
        "tag": tag,
        "sharded_streaming": sharded_streaming,
        "rows": rows,
        "unique_original_samples": unique_samples,
        "failure_rows": int(len(failures)),
        "valid_rate": valid_rate,
        "overall_node_overlap_jaccard_mean": node_mean,
        "overall_edge_overlap_jaccard_mean": edge_mean,
        "text_stability_hint": text_stability_hint,
        "prompt_modulation_hint": prompt_modulation_hint,
        "interpretation_boundary": (
            "This is Gemma source-tracing route robustness under Stage6 prompt/text/CoT conditions. "
            "It complements, but does not replace, Stage3 causal source-tracing evidence."
        ),
        "artifacts": {
            "summary_overall": str(CROSS / f"{prefix}_summary_overall.csv"),
            "summary_by_prompt": str(CROSS / f"{prefix}_summary_by_prompt.csv"),
            "summary_by_variant": str(CROSS / f"{prefix}_summary_by_variant.csv"),
            "summary_by_prompt_variant": str(CROSS / f"{prefix}_summary_by_prompt_variant.csv"),
            "summary_by_type": str(CROSS / f"{prefix}_summary_by_type.csv"),
            "rewrite_deltas": str(CROSS / f"{prefix}_rewrite_deltas.csv"),
            "rewrite_stability_summary": str(CROSS / f"{prefix}_rewrite_stability_summary.csv"),
        },
        "remote_decision": decision,
    }
    (CROSS / f"{prefix}_analysis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage6 Gemma prompt/text/CoT source-tracing counterpart.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--tag", default="gemmaprompt_v1")
    args = parser.parse_args()
    analyze(args.mode, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
