#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
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
    if value is None or math.isnan(value):
        return ""
    return f"{value:.10g}"


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    pos = (len(sorted_values) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_values[lower]
    frac = pos - lower
    return sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac


def _bootstrap_mean(values: list[float], *, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = random.Random(seed)
    observed = sum(values) / len(values)
    samples: list[float] = []
    for _ in range(n_boot):
        draw = [values[rng.randrange(len(values))] for _ in range(len(values))]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    return observed, _quantile(samples, 0.025), _quantile(samples, 0.975)


def _build_nearest_per_source(control_csv: Path) -> list[dict[str, str]]:
    rows = _read_csv(control_csv)
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.get("sample_id", ""),
                row.get("run", ""),
                row.get("condition", ""),
                row.get("node_role", ""),
            )
        ].append(row)
    out = []
    for (sample_id, run, condition, node_role), members in grouped.items():
        gap_vals = [_safe_float(r.get("delta_target_logit")) for r in members]
        gap_vals = [v for v in gap_vals if v is not None]
        if not gap_vals:
            continue
        out.append(
            {
                "sample_id": sample_id,
                "prompt": "D_visual_only" if run == "A" else "B_direct",
                "condition": condition,
                "node_role": node_role,
                "mean_control_delta_target_logit": str(sum(gap_vals) / len(gap_vals)),
            }
        )
    return out


def _load_nearest_units() -> list[dict[str, str]]:
    units = []
    for subset, pilot_path, control_path in [
        (
            "strong12",
            ROOT / "remote_sync" / "2026-05-14_bd_visual_positive_strong12_main64" / "modality_pilot_visual_positive_strong12_clean_core7_generic6_nobuf_full7_cap0_relpos.csv",
            ROOT / "remote_sync" / "2026-05-14_bd_visual_positive_strong12_main64" / "modality_matched_control_visual_positive_strong12_clean_core7_generic6_nobuf_full7_cap0_relpos.csv",
        ),
        (
            "strong18_next",
            ROOT / "remote_sync" / "2026-05-14_bd_visual_positive_strong18_next_main64" / "modality_pilot_visual_positive_strong18_next_clean_core10_generic8_nobuf_full10_cap0_relpos.csv",
            ROOT / "remote_sync" / "2026-05-14_bd_visual_positive_strong18_next_main64" / "modality_matched_control_visual_positive_strong18_next_clean_core10_generic8_nobuf_full10_cap0_relpos.csv",
        ),
    ]:
        pilot_rows = _read_csv(pilot_path)
        grouped_pilot: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
        for row in pilot_rows:
            grouped_pilot[
                (
                    row.get("sample_id", ""),
                    row.get("run", ""),
                    row.get("condition", ""),
                    row.get("node_role", ""),
                )
            ].append(row)
        control_grouped = defaultdict(list)
        for row in _read_csv(control_path):
            control_grouped[
                (
                    row.get("sample_id", ""),
                    row.get("run", ""),
                    row.get("condition", ""),
                    row.get("node_role", ""),
                )
            ].append(row)
        for key, members in grouped_pilot.items():
            sample_id, run, condition, node_role = key
            if condition != "wrong_image":
                continue
            source_vals = [_safe_float(r.get("delta_target_logit")) for r in members]
            source_vals = [v for v in source_vals if v is not None]
            control_vals = [_safe_float(r.get("delta_target_logit")) for r in control_grouped.get(key, [])]
            control_vals = [v for v in control_vals if v is not None]
            if not source_vals or not control_vals:
                continue
            units.append(
                {
                    "subset": subset,
                    "sample_id": sample_id,
                    "prompt": "D_visual_only" if run == "A" else "B_direct",
                    "condition": condition,
                    "node_role": node_role,
                    "gap": str((sum(source_vals) / len(source_vals)) - (sum(control_vals) / len(control_vals))),
                }
            )
    return units


def _load_random_units() -> list[dict[str, str]]:
    units = []
    for subset, path in [
        ("strong12", ROOT / "analysis_cache" / "visual_positive_strong12_random4_summary" / "modality_matched_control_per_source.csv"),
        ("strong18_next", ROOT / "analysis_cache" / "visual_positive_strong18_next_random4_summary" / "modality_matched_control_per_source.csv"),
    ]:
        for row in _read_csv(path):
            if row.get("condition") != "wrong_image":
                continue
            gap = _safe_float(row.get("mean_source_minus_control_dlogit"))
            if gap is None:
                continue
            units.append(
                {
                    "subset": subset,
                    "sample_id": row.get("sample_id", ""),
                    "prompt": "D_visual_only" if row.get("run") == "A" else "B_direct",
                    "condition": "wrong_image",
                    "node_role": row.get("node_role", ""),
                    "gap": str(gap),
                }
            )
    return units


def _load_prompt_behavior_units() -> list[dict[str, str]]:
    out = []
    for subset, behavior_path, rank_path in [
        (
            "strong12",
            ROOT / "analysis_cache" / "visual_positive_strong12_clean_core7_condition_eval" / "condition_generation_detail.csv",
            ROOT / "analysis_cache" / "visual_positive_strong12_clean_core7_target_rank" / "condition_target_rank_detail.csv",
        ),
        (
            "strong18_next",
            ROOT / "analysis_cache" / "visual_positive_strong18_next_clean_core10_condition_eval" / "condition_generation_detail.csv",
            ROOT / "analysis_cache" / "visual_positive_strong18_next_clean_core10_target_rank" / "condition_target_rank_detail.csv",
        ),
    ]:
        rank_map = {}
        for row in _read_csv(rank_path):
            if row.get("condition") != "wrong_image":
                continue
            rank_map[(row.get("sample_id", ""), row.get("run", ""))] = row
        for row in _read_csv(behavior_path):
            if row.get("condition") != "wrong_image":
                continue
            rank_row = rank_map.get((row.get("sample_id", ""), row.get("run", "")), {})
            out.append(
                {
                    "subset": subset,
                    "sample_id": row.get("sample_id", ""),
                    "prompt": row.get("run", ""),
                    "answer_changed": "1.0" if row.get("answer_changed_from_clean") == "True" else "0.0",
                    "margin_drop_vs_clean": rank_row.get("margin_drop_vs_clean", ""),
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap main visual-positive summary effects.")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", default=str(ROOT / "analysis_cache" / "visual_positive_main_experiment_summary"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    nearest_units = _load_nearest_units()
    random_units = _load_random_units()
    for control_kind, units in (("nearest", nearest_units), ("random4", random_units)):
        for role in ("support", "suppressor"):
            vals = [_safe_float(r.get("gap")) for r in units if r.get("node_role") == role]
            vals = [v for v in vals if v is not None]
            observed, lo, hi = _bootstrap_mean(vals, n_boot=args.n_boot, seed=args.seed + len(rows))
            rows.append(
                {
                    "analysis": f"wrong_image_{role}_gap_{control_kind}",
                    "unit": "sample_id x prompt x role",
                    "n_units": str(len(vals)),
                    "observed_mean": _fmt(observed),
                    "ci_lower": _fmt(lo),
                    "ci_upper": _fmt(hi),
                    "interpretation_label": "stable" if role == "support" else "heterogeneous",
                }
            )

    behavior_units = _load_prompt_behavior_units()
    for metric in ("answer_changed", "margin_drop_vs_clean"):
        b_vals = [_safe_float(r.get(metric)) for r in behavior_units if r.get("prompt") == "B_direct"]
        d_vals = [_safe_float(r.get(metric)) for r in behavior_units if r.get("prompt") == "D_visual_only"]
        b_vals = [v for v in b_vals if v is not None]
        d_vals = [v for v in d_vals if v is not None]
        rng = random.Random(args.seed + 101 + len(rows))
        observed = (sum(d_vals) / len(d_vals)) - (sum(b_vals) / len(b_vals))
        samples = []
        for _ in range(args.n_boot):
            d_draw = [d_vals[rng.randrange(len(d_vals))] for _ in range(len(d_vals))]
            b_draw = [b_vals[rng.randrange(len(b_vals))] for _ in range(len(b_vals))]
            samples.append((sum(d_draw) / len(d_draw)) - (sum(b_draw) / len(b_draw)))
        samples.sort()
        rows.append(
            {
                "analysis": f"exploratory_D_minus_B_{metric}",
                "unit": "sample_id x prompt",
                "n_units": str(min(len(d_vals), len(b_vals))),
                "observed_mean": _fmt(observed),
                "ci_lower": _fmt(_quantile(samples, 0.025)),
                "ci_upper": _fmt(_quantile(samples, 0.975)),
                "interpretation_label": "weak",
            }
        )

    csv_path = out_dir / "bootstrap_main_effects.csv"
    _write_csv(csv_path, rows, list(rows[0].keys()))
    md_path = out_dir / "bootstrap_main_effects.md"
    with md_path.open("w", encoding="utf-8", newline="") as f:
        f.write("| analysis | n units | observed mean | 95% CI | label |\n")
        f.write("|---|---:|---:|---|---|\n")
        for row in rows:
            f.write(
                f"| {row['analysis']} | {row['n_units']} | {row['observed_mean']} | [{row['ci_lower']}, {row['ci_upper']}] | {row['interpretation_label']} |\n"
            )

    print(f"[done] csv={csv_path}")
    print(f"[done] md={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
