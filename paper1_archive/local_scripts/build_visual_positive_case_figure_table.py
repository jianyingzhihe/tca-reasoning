#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except Exception:
        return math.nan


def _fmt(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.10g}"


def _run_slot(run_name: str) -> str:
    return "A" if run_name == "D_visual_only" else "B"


def _route_profile(row: dict[str, str]) -> str:
    support = _safe_float(row.get("support_route_weakening"))
    suppressor = _safe_float(row.get("suppressor_route_weakening"))
    if math.isnan(support) and math.isnan(suppressor):
        return "untyped"
    if math.isnan(support):
        return "suppressor_only"
    if math.isnan(suppressor):
        return "support_only"
    if suppressor > support:
        return "mixed_suppressor_dominant"
    if support > suppressor:
        return "mixed_support_dominant"
    return "mixed_tied"


def _agg_random_wrong_image(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(path):
        if row.get("condition") != "wrong_image":
            continue
        run_name = "D_visual_only" if row.get("run") == "A" else "B_direct"
        grouped[(row.get("sample_id", ""), run_name, row.get("node_role", ""))].append(row)

    for (sample_id, run_name, node_role), rows in grouped.items():
        key = (sample_id, run_name)
        record = out.setdefault(key, {})
        gap_vals = [_safe_float(r.get("mean_source_minus_control_dlogit")) for r in rows]
        source_vals = [_safe_float(r.get("mean_source_delta_target_logit")) for r in rows]
        control_vals = [_safe_float(r.get("mean_control_delta_target_logit")) for r in rows]
        gap_vals = [v for v in gap_vals if not math.isnan(v)]
        source_vals = [v for v in source_vals if not math.isnan(v)]
        control_vals = [v for v in control_vals if not math.isnan(v)]
        prefix = f"random4_{node_role}"
        record[f"{prefix}_gap_wrong_image"] = _fmt(sum(gap_vals) / len(gap_vals)) if gap_vals else ""
        record[f"{prefix}_source_wrong_image"] = _fmt(sum(source_vals) / len(source_vals)) if source_vals else ""
        record[f"{prefix}_control_wrong_image"] = _fmt(sum(control_vals) / len(control_vals)) if control_vals else ""
        record[f"{prefix}_n_sources"] = str(len(rows))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a compact figure-ready case table with wrong-image evidence plus random-control columns."
    )
    parser.add_argument("--top-candidates-csv", required=True)
    parser.add_argument("--pooled-detail-csv", required=True)
    parser.add_argument("--random4-strong12-csv", required=True)
    parser.add_argument("--random4-strong18-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    top_rows = _read_csv(Path(args.top_candidates_csv).expanduser().resolve())
    pooled_detail = _read_csv(Path(args.pooled_detail_csv).expanduser().resolve())
    detail_map = {(r.get("sample_id", ""), r.get("run_name", "")): r for r in pooled_detail}

    random_map: dict[tuple[str, str], dict[str, str]] = {}
    for path_str in (args.random4_strong12_csv, args.random4_strong18_csv):
        random_map.update(_agg_random_wrong_image(Path(path_str).expanduser().resolve()))

    out_rows: list[dict[str, str]] = []
    for row in top_rows:
        key = (row.get("sample_id", ""), row.get("run_name", ""))
        detail = detail_map[key]
        random_row = random_map.get(key, {})
        out_rows.append(
            {
                "subset": detail.get("subset", ""),
                "sample_id": detail.get("sample_id", ""),
                "run_name": detail.get("run_name", ""),
                "run_slot": _run_slot(detail.get("run_name", "")),
                "route_profile": _route_profile(detail),
                "figure_score": detail.get("figure_score", ""),
                "clean_predicted_answer": detail.get("clean_predicted_answer", ""),
                "wrong_image_predicted_answer": detail.get("wrong_image_predicted_answer", ""),
                "answer_changed_from_clean": detail.get("answer_changed_from_clean", ""),
                "wrong_image_vqa_score": detail.get("wrong_image_vqa_score", ""),
                "wrong_image_target_rank": detail.get("wrong_image_target_rank", ""),
                "clean_target_rank": detail.get("clean_target_rank", ""),
                "margin_drop_vs_clean": detail.get("margin_drop_vs_clean", ""),
                "support_route_weakening": detail.get("support_route_weakening", ""),
                "suppressor_route_weakening": detail.get("suppressor_route_weakening", ""),
                "nearest_support_gap_wrong_image": detail.get("support_wrong_image_strength_gap", ""),
                "nearest_suppressor_gap_wrong_image": detail.get("suppressor_wrong_image_strength_gap", ""),
                "nearest_support_gap_drop": detail.get("support_gap_drop", ""),
                "nearest_suppressor_gap_drop": detail.get("suppressor_gap_drop", ""),
                "random4_support_gap_wrong_image": random_row.get("random4_support_gap_wrong_image", ""),
                "random4_suppressor_gap_wrong_image": random_row.get("random4_suppressor_gap_wrong_image", ""),
                "random4_support_n_sources": random_row.get("random4_support_n_sources", "0"),
                "random4_suppressor_n_sources": random_row.get("random4_suppressor_n_sources", "0"),
            }
        )

    out_dir = Path(args.out_dir).expanduser().resolve()
    csv_path = out_dir / "visual_positive_case_figure_table.csv"
    md_path = out_dir / "visual_positive_case_figure_table.md"
    _write_csv(csv_path, out_rows, list(out_rows[0].keys()))

    with md_path.open("w", encoding="utf-8", newline="") as f:
        f.write("| subset | sample_id | prompt | profile | clean answer | wrong-image answer | wrong rank | margin drop | support weaken | suppressor weaken | nearest support gap | nearest suppressor gap | random4 support gap | random4 suppressor gap |\n")
        f.write("|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in out_rows:
            f.write(
                "| {subset} | {sample_id} | {run_name} | {route_profile} | {clean_predicted_answer} | {wrong_image_predicted_answer} | {wrong_image_target_rank} | {margin_drop_vs_clean} | {support_route_weakening} | {suppressor_route_weakening} | {nearest_support_gap_wrong_image} | {nearest_suppressor_gap_wrong_image} | {random4_support_gap_wrong_image} | {random4_suppressor_gap_wrong_image} |\n".format(
                    **row
                )
            )

    print(f"[done] csv={csv_path}")
    print(f"[done] md={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
