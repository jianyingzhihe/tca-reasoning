#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(r"E:\Bridging")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.10g}"


def _mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _run_label(slot: str) -> str:
    return "D_visual_only" if slot == "A" else "B_direct"


def _load_overall_modality(path: Path, subset: str) -> list[dict[str, str]]:
    rows = _read_csv(path)
    out = []
    for row in rows:
        if row.get("slice_col") != "overall" or row.get("slice_value") != "all":
            continue
        out.append(
            {
                "subset": subset,
                "prompt": _run_label(row["run"]),
                "condition": row["condition"],
                "node_role": row["node_role"],
                "n_units": row["n_samples"],
                "source_effect": row["mean_delta_target_logit"],
            }
        )
    return out


def _load_overall_nearest(path: Path, subset: str) -> list[dict[str, str]]:
    rows = _read_csv(path)
    out = []
    for row in rows:
        if row.get("slice_col") != "overall" or row.get("slice_value") != "all":
            continue
        out.append(
            {
                "subset": subset,
                "prompt": _run_label(row["run"]),
                "condition": row["condition"],
                "node_role": row["node_role"],
                "nearest_n_units": row["n_samples"],
                "nearest_gap": row["mean_source_minus_control_dlogit"],
                "nearest_source_abs_ge_control_abs": row["frac_source_abs_ge_control_abs"],
            }
        )
    return out


def _load_random(path: Path, subset: str) -> list[dict[str, str]]:
    rows = _read_csv(path)
    out = []
    for row in rows:
        out.append(
            {
                "subset": subset,
                "prompt": "pooled",
                "condition": row["condition"],
                "node_role": row["node_role"],
                "random4_n_units": row["n_sources"],
                "random4_gap": row["mean_source_minus_control_dlogit"],
                "random4_source_abs_ge_control_abs": row["frac_source_abs_ge_control_abs"],
            }
        )
    return out


def _build_table1(out_dir: Path) -> None:
    modality_rows = []
    modality_rows.extend(
        _load_overall_modality(
            ROOT / "analysis_cache" / "visual_positive_strong12_full7_cap0_summary" / "typed_modality_summary.csv",
            "strong12",
        )
    )
    modality_rows.extend(
        _load_overall_modality(
            ROOT / "analysis_cache" / "visual_positive_strong18_next_full10_summary" / "typed_modality_summary.csv",
            "strong18_next",
        )
    )
    nearest_rows = []
    nearest_rows.extend(
        _load_overall_nearest(
            ROOT / "analysis_cache" / "visual_positive_strong12_full7_cap0_summary" / "typed_matched_control_summary.csv",
            "strong12",
        )
    )
    nearest_rows.extend(
        _load_overall_nearest(
            ROOT / "analysis_cache" / "visual_positive_strong18_next_full10_summary" / "typed_matched_control_summary.csv",
            "strong18_next",
        )
    )
    random_rows = []
    random_rows.extend(
        _load_random(
            ROOT / "analysis_cache" / "visual_positive_strong12_random4_summary" / "modality_matched_control_per_source_summary.csv",
            "strong12",
        )
    )
    random_rows.extend(
        _load_random(
            ROOT / "analysis_cache" / "visual_positive_strong18_next_random4_summary" / "modality_matched_control_per_source_summary.csv",
            "strong18_next",
        )
    )

    by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in modality_rows + nearest_rows:
        key = (row["subset"], row["prompt"], row["condition"], row["node_role"])
        by_key.setdefault(key, {}).update(row)

    pooled_random: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in random_rows:
        key = (row["subset"], row["condition"], row["node_role"])
        pooled_random[key] = row

    out_rows: list[dict[str, str]] = []
    for subset in ("strong12", "strong18_next"):
        for prompt in ("D_visual_only", "B_direct"):
            for condition in ("clean", "wrong_image"):
                for role in ("support", "suppressor"):
                    key = (subset, prompt, condition, role)
                    base = by_key.get(key, {})
                    pooled = pooled_random.get((subset, condition, role), {})
                    out_rows.append(
                        {
                            "subset": subset,
                            "prompt": prompt,
                            "condition": condition,
                            "node_role": role,
                            "source_effect": base.get("source_effect", ""),
                            "nearest_gap": base.get("nearest_gap", ""),
                            "nearest_abs_ge_control": base.get("nearest_source_abs_ge_control_abs", ""),
                            "random4_gap_pooled": pooled.get("random4_gap", ""),
                            "random4_abs_ge_control_pooled": pooled.get("random4_source_abs_ge_control_abs", ""),
                        }
                    )

    csv_path = out_dir / "table1_mechanism_clean_wrong_image.csv"
    _write_csv(csv_path, out_rows, list(out_rows[0].keys()))
    md_path = out_dir / "table1_mechanism_clean_wrong_image.md"
    with md_path.open("w", encoding="utf-8", newline="") as f:
        f.write("| subset | prompt | condition | role | source effect | nearest gap | nearest abs>=control | random4 gap (subset-pooled) | random4 abs>=control (subset-pooled) |\n")
        f.write("|---|---|---|---|---:|---:|---:|---:|---:|\n")
        for row in out_rows:
            f.write(
                f"| {row['subset']} | {row['prompt']} | {row['condition']} | {row['node_role']} | {row['source_effect']} | {row['nearest_gap']} | {row['nearest_abs_ge_control']} | {row['random4_gap_pooled']} | {row['random4_abs_ge_control_pooled']} |\n"
            )


def _load_behavior_detail(path: Path, subset: str) -> list[dict[str, str]]:
    rows = _read_csv(path)
    out = []
    for row in rows:
        if row.get("condition") != "wrong_image":
            continue
        out.append(
            {
                "subset": subset,
                "sample_id": row["sample_id"],
                "prompt": row["run"],
                "answer_changed": row["answer_changed_from_clean"],
                "format_prefix_ok": row["format_prefix_ok"],
                "empty_or_error": row["empty_or_error"],
                "wrong_image_vqa_score": row["vqa_score"],
            }
        )
    return out


def _load_rank_detail(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path)
    out = []
    for row in rows:
        if row.get("condition") != "wrong_image":
            continue
        out.append(
            {
                "sample_id": row["sample_id"],
                "prompt": row["run"],
                "rank_worsened": row["rank_worsened_vs_clean"],
                "margin_drop_vs_clean": row["margin_drop_vs_clean"],
                "wrong_image_target_rank": row["target_rank"],
            }
        )
    return out


def _build_table2(out_dir: Path) -> None:
    behavior_rows = []
    behavior_rows.extend(
        _load_behavior_detail(
            ROOT / "analysis_cache" / "visual_positive_strong12_clean_core7_condition_eval" / "condition_generation_detail.csv",
            "strong12",
        )
    )
    behavior_rows.extend(
        _load_behavior_detail(
            ROOT / "analysis_cache" / "visual_positive_strong18_next_clean_core10_condition_eval" / "condition_generation_detail.csv",
            "strong18_next",
        )
    )
    rank_rows = []
    rank_rows.extend(
        _load_rank_detail(
            ROOT / "analysis_cache" / "visual_positive_strong12_clean_core7_target_rank" / "condition_target_rank_detail.csv"
        )
    )
    rank_rows.extend(
        _load_rank_detail(
            ROOT / "analysis_cache" / "visual_positive_strong18_next_clean_core10_target_rank" / "condition_target_rank_detail.csv"
        )
    )
    rank_map = {(r["sample_id"], r["prompt"]): r for r in rank_rows}

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in behavior_rows:
        merged = dict(row)
        merged.update(rank_map.get((row["sample_id"], row["prompt"]), {}))
        grouped[row["prompt"]].append(merged)

    out_rows = []
    for prompt in ("B_direct", "D_visual_only"):
        rows = grouped[prompt]
        n = len(rows)
        changed = sum(1 for r in rows if r.get("answer_changed") == "True")
        prefix_fail = sum(1 for r in rows if r.get("format_prefix_ok") != "True")
        empty = sum(1 for r in rows if r.get("empty_or_error") == "True")
        rank_worsened = sum(1 for r in rows if r.get("rank_worsened") == "True")
        mean_margin_drop = _mean([_safe_float(r.get("margin_drop_vs_clean")) for r in rows])
        mean_wrong_rank = _mean([_safe_float(r.get("wrong_image_target_rank")) for r in rows])
        mean_vqa = _mean([_safe_float(r.get("wrong_image_vqa_score")) for r in rows])
        out_rows.append(
            {
                "prompt": prompt,
                "n_samples": str(n),
                "answer_changed_count": str(changed),
                "answer_changed_rate": _fmt(changed / n if n else None),
                "format_failure_count": str(prefix_fail),
                "format_failure_rate": _fmt(prefix_fail / n if n else None),
                "empty_or_error_count": str(empty),
                "empty_or_error_rate": _fmt(empty / n if n else None),
                "rank_worsened_count": str(rank_worsened),
                "rank_worsened_rate": _fmt(rank_worsened / n if n else None),
                "mean_margin_drop_vs_clean": _fmt(mean_margin_drop),
                "mean_wrong_image_target_rank": _fmt(mean_wrong_rank),
                "mean_wrong_image_vqa_score": _fmt(mean_vqa),
            }
        )

    csv_path = out_dir / "table2_wrong_image_behavior_prompt_comparison.csv"
    _write_csv(csv_path, out_rows, list(out_rows[0].keys()))
    md_path = out_dir / "table2_wrong_image_behavior_prompt_comparison.md"
    with md_path.open("w", encoding="utf-8", newline="") as f:
        f.write("| prompt | n | answer changed | format failure | empty/error | rank worsened | mean margin drop | mean wrong-image rank | mean wrong-image VQA |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in out_rows:
            f.write(
                f"| {row['prompt']} | {row['n_samples']} | {row['answer_changed_count']} ({row['answer_changed_rate']}) | {row['format_failure_count']} ({row['format_failure_rate']}) | {row['empty_or_error_count']} ({row['empty_or_error_rate']}) | {row['rank_worsened_count']} ({row['rank_worsened_rate']}) | {row['mean_margin_drop_vs_clean']} | {row['mean_wrong_image_target_rank']} | {row['mean_wrong_image_vqa_score']} |\n"
            )


def _build_table3(out_dir: Path) -> None:
    rows = [
        {
            "criterion": "1. support source nodes show reproducible signed clean effect",
            "evidence_artifact": "analysis_cache/visual_positive_strong12_full7_cap0_summary/typed_modality_summary.csv; analysis_cache/visual_positive_strong18_next_full10_summary/typed_modality_summary.csv",
            "status": "success",
            "conservative_reading": "support rows are consistently negative in clean condition across both held-out branches",
        },
        {
            "criterion": "2. support effects weaken toward zero under wrong_image",
            "evidence_artifact": "analysis_cache/visual_positive_strong12_full7_cap0_summary/typed_modality_summary.csv; analysis_cache/visual_positive_strong18_next_full10_summary/typed_modality_summary.csv",
            "status": "success",
            "conservative_reading": "support clean effect is larger in magnitude than wrong_image effect in both branches, though not every single row is monotonic",
        },
        {
            "criterion": "3. support weakening is larger for source than matched controls",
            "evidence_artifact": "analysis_cache/visual_positive_random4_comparison/random_vs_nearest_overall_by_run.csv",
            "status": "success",
            "conservative_reading": "wrong_image support source-control gaps stay negative under both nearest and random4 controls",
        },
        {
            "criterion": "4. effect is not primarily explained by format collapse or empty answers",
            "evidence_artifact": "analysis_cache/visual_positive_strong12_clean_core7_condition_eval/condition_generation_summary.csv; analysis_cache/visual_positive_strong18_next_clean_core10_condition_eval/condition_generation_summary.csv",
            "status": "partial",
            "conservative_reading": "wrong_image remains usable as strongest behavior-linked corruption, but D_visual_only is mildly contaminated and D_visual_only+no_image is downgraded",
        },
        {
            "criterion": "Prompt increment: D_visual_only changes route use beyond B_direct",
            "evidence_artifact": "analysis_cache/visual_positive_case_figure_table/visual_positive_case_figure_table.csv; analysis_cache/visual_positive_pooled_wrong_image_evidence/*",
            "status": "partial",
            "conservative_reading": "prompt-increment effects appear in cases, but aggregate D-vs-B mechanism superiority is weak or unstable",
        },
    ]
    csv_path = out_dir / "table3_run_plan_success_criteria.csv"
    _write_csv(csv_path, rows, list(rows[0].keys()))
    md_path = out_dir / "table3_run_plan_success_criteria.md"
    with md_path.open("w", encoding="utf-8", newline="") as f:
        f.write("| criterion | status | conservative reading | evidence artifact |\n")
        f.write("|---|---|---|---|\n")
        for row in rows:
            f.write(
                f"| {row['criterion']} | {row['status']} | {row['conservative_reading']} | {row['evidence_artifact']} |\n"
            )


def _build_table4(out_dir: Path) -> None:
    src = ROOT / "analysis_cache" / "visual_positive_case_figure_table" / "visual_positive_case_figure_table.csv"
    rows = _read_csv(src)
    csv_path = out_dir / "table4_case_shortlist.csv"
    _write_csv(csv_path, rows, list(rows[0].keys()))
    md_src = ROOT / "analysis_cache" / "visual_positive_case_figure_table" / "visual_positive_case_figure_table.md"
    md_dst = out_dir / "table4_case_shortlist.md"
    md_dst.write_text(md_src.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> int:
    out_dir = ROOT / "analysis_cache" / "visual_positive_main_experiment_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    _build_table1(out_dir)
    _build_table2(out_dir)
    _build_table3(out_dir)
    _build_table4(out_dir)
    print(f"[done] out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
