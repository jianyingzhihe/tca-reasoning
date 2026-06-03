#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _bool(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().eq("true")


def _summary(group: pd.DataFrame) -> pd.Series:
    rank_damage = _num(group["rank_damage_by_intervention"])
    delta_logit = _num(group["delta_target_logit"])
    top1_changed = _bool(group["top1_changed_by_intervention"])
    return pd.Series(
        {
            "n": int(len(group)),
            "mean_rank_damage": float(rank_damage.mean()) if len(group) else np.nan,
            "median_rank_damage": float(rank_damage.median()) if len(group) else np.nan,
            "positive_rank_damage_rate": float((rank_damage > 0).mean()) if len(group) else np.nan,
            "mean_delta_target_logit": float(delta_logit.mean()) if len(group) else np.nan,
            "median_delta_target_logit": float(delta_logit.median()) if len(group) else np.nan,
            "negative_delta_logit_rate": float((delta_logit < 0).mean()) if len(group) else np.nan,
            "top1_changed_rate": float(top1_changed.mean()) if len(group) else np.nan,
            "top1_changed_count": int(top1_changed.sum()),
        }
    )


def _write_md(
    out_path: Path,
    csv_path: Path,
    source_summary: pd.DataFrame,
    gap_summary: pd.DataFrame,
    pair_gap: pd.DataFrame,
    case_table: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.extend(
        [
            "# Stage 2B first-token node-intervention smoke summary",
            "",
            "## 目的",
            "",
            "本分析读取 4-case source+nearest first-token smoke 结果，检查 support source node 清零是否会比 nearest-control node 更明显伤害下一答案 token 的目标 logit / rank，作为 `node intervention -> generation-side distribution` 的最小桥接证据。",
            "",
            "注意：这不是完整 decoded generation。它只回答“在已经固定 `assistant_prefix = The answer is` 的条件下，清零节点是否改变下一 token 分布”。",
            "",
            "## 输入",
            "",
            f"```text\n{csv_path}\n```",
            "",
            "## Node-source summary",
            "",
            *source_summary.to_markdown(index=False).splitlines(),
            "",
            "## Source-minus-nearest gap summary",
            "",
            "这里 `rank_damage_gap = source_rank_damage - nearest_rank_damage`，正值表示 source 清零比 nearest 更伤 target rank；`delta_logit_gap = source_delta_logit - nearest_delta_logit`，负值表示 source 清零比 nearest 更降低目标 token logit。",
            "",
            *gap_summary.to_markdown(index=False).splitlines(),
            "",
            "## Pair-level source-minus-nearest gaps",
            "",
            *pair_gap.to_markdown(index=False).splitlines(),
            "",
            "## Case table",
            "",
            *case_table.to_markdown(index=False).splitlines(),
            "",
            "## 结论",
            "",
            "这轮 smoke 证明工程上已经可以在多模态 batch 上做 `feature_intervention -> first answer token distribution`。机制上，结果是 mixed but informative：",
            "",
            "- `okvqa_val_2683965` 在 `answer_mask` 下出现强 source-specific rank damage，source 为 `+96`，nearest 为 `-107`，是最强正向桥接 case。",
            "- `okvqa_val_1927165` 的 source 清零会降低目标 token logit，并在 `answer_mask` 下改变 top1 token，但 rank damage 不是单调正向。",
            "- `okvqa_val_80655` 在 `union_mask` 下 source 和 nearest 都改变 top1，说明它不是干净的 source-specific case。",
            "- `okvqa_val_3794755` 的 clean 条件下 source 和 nearest 都改变 top1，说明该 case 更像非特异或格式/竞争答案敏感 case。",
            "",
            "因此 Stage 2B 当前读法应是：`first-token node-to-generation bridge is feasible, with one strong positive case and several heterogeneous / non-specific cases`。不能把它写成强统计结论，但可以作为下一步扩展 node-to-generation 的工程和 case 选择依据。",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Stage 2B first-token node-intervention smoke.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df["rank_damage_by_intervention"] = _num(df["rank_damage_by_intervention"])
    df["delta_target_logit"] = _num(df["delta_target_logit"])
    df["delta_target_prob"] = _num(df["delta_target_prob"])
    df["top1_changed_bool"] = _bool(df["top1_changed_by_intervention"])

    source_summary = (
        df.groupby(["node_source", "condition"], dropna=False)
        .apply(_summary, include_groups=False)
        .reset_index()
        .sort_values(["condition", "node_source"])
    )

    source = df[df["node_source"].eq("source")].copy()
    nearest = df[df["node_source"].eq("nearest_control")].copy()
    merge_cols = ["pair_id", "sample_id", "run", "condition"]
    pair_gap = source.merge(
        nearest,
        on=merge_cols,
        suffixes=("_source", "_nearest"),
        how="inner",
    )
    pair_gap["rank_damage_gap"] = pair_gap["rank_damage_by_intervention_source"] - pair_gap["rank_damage_by_intervention_nearest"]
    pair_gap["delta_logit_gap"] = pair_gap["delta_target_logit_source"] - pair_gap["delta_target_logit_nearest"]
    pair_gap["source_top1_changed_only"] = pair_gap["top1_changed_bool_source"] & ~pair_gap["top1_changed_bool_nearest"]
    pair_gap["nearest_top1_changed_only"] = ~pair_gap["top1_changed_bool_source"] & pair_gap["top1_changed_bool_nearest"]

    gap_summary = (
        pair_gap.groupby(["condition"], dropna=False)
        .agg(
            n=("pair_id", "count"),
            mean_rank_damage_gap=("rank_damage_gap", "mean"),
            median_rank_damage_gap=("rank_damage_gap", "median"),
            positive_rank_damage_gap_rate=("rank_damage_gap", lambda s: float((s > 0).mean())),
            mean_delta_logit_gap=("delta_logit_gap", "mean"),
            median_delta_logit_gap=("delta_logit_gap", "median"),
            negative_delta_logit_gap_rate=("delta_logit_gap", lambda s: float((s < 0).mean())),
            source_top1_changed_only_count=("source_top1_changed_only", "sum"),
            nearest_top1_changed_only_count=("nearest_top1_changed_only", "sum"),
        )
        .reset_index()
    )

    pair_gap_view = pair_gap[
        [
            "sample_id",
            "run",
            "condition",
            "target_token_source",
            "rank_damage_by_intervention_source",
            "rank_damage_by_intervention_nearest",
            "rank_damage_gap",
            "delta_target_logit_source",
            "delta_target_logit_nearest",
            "delta_logit_gap",
            "baseline_top1_token_source",
            "intervention_top1_token_source",
            "baseline_top1_token_nearest",
            "intervention_top1_token_nearest",
        ]
    ].copy()

    case_table = df[
        [
            "sample_id",
            "run",
            "node_source",
            "condition",
            "target_token",
            "baseline_target_rank",
            "intervention_target_rank",
            "rank_damage_by_intervention",
            "delta_target_logit",
            "baseline_top1_token",
            "intervention_top1_token",
            "top1_changed_by_intervention",
        ]
    ].copy()

    source_summary.to_csv(out_dir / "first_token_node_source_summary.csv", index=False)
    gap_summary.to_csv(out_dir / "first_token_source_nearest_gap_summary.csv", index=False)
    pair_gap_view.to_csv(out_dir / "first_token_pair_gap_table.csv", index=False)
    case_table.to_csv(out_dir / "first_token_case_table.csv", index=False)
    _write_md(
        out_dir / "STAGE2B_FIRST_TOKEN_SMOKE_SUMMARY.md",
        csv_path,
        source_summary,
        gap_summary,
        pair_gap_view,
        case_table,
    )
    print(f"[done] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
