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


def _mean(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return math.nan
    return sum(clean) / len(clean)


def _source_key_from_pilot(row: dict[str, str]) -> tuple[str, str, str, str, str, str, str, str]:
    return (
        row.get("bucket", ""),
        row.get("sample_id", ""),
        row.get("run", ""),
        row.get("condition", ""),
        row.get("node_role", ""),
        row.get("feature_layer", ""),
        row.get("feature_pos", ""),
        row.get("feature_id", ""),
    )


def _source_key_from_control(row: dict[str, str]) -> tuple[str, str, str, str, str, str, str, str]:
    return (
        row.get("bucket", ""),
        row.get("sample_id", ""),
        row.get("run", ""),
        row.get("condition", ""),
        row.get("node_role", ""),
        row.get("source_feature_layer", ""),
        row.get("source_feature_pos", ""),
        row.get("source_feature_id", ""),
    )


def _paired_source_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (
        row.get("bucket", ""),
        row.get("sample_id", ""),
        row.get("run", ""),
        row.get("condition", ""),
        row.get("node_role", ""),
        row.get("source_feature", ""),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare source pilot nodes against matched-control intervention results.")
    parser.add_argument("--pilot-csv", required=True)
    parser.add_argument("--control-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    pilot_rows = _read_csv(Path(args.pilot_csv).expanduser().resolve())
    control_rows = _read_csv(Path(args.control_csv).expanduser().resolve())
    out_dir = Path(args.out_dir).expanduser().resolve()

    pilot_map = {_source_key_from_pilot(row): row for row in pilot_rows}
    paired_rows: list[dict[str, str]] = []
    for row in control_rows:
        key = _source_key_from_control(row)
        pilot_row = pilot_map.get(key)
        if pilot_row is None:
            continue
        source_dlogit = _safe_float(pilot_row.get("delta_target_logit"))
        control_dlogit = _safe_float(row.get("delta_target_logit"))
        source_dprob = _safe_float(pilot_row.get("delta_target_prob"))
        control_dprob = _safe_float(row.get("delta_target_prob"))
        paired_rows.append(
            {
                "bucket": row.get("bucket", ""),
                "sample_id": row.get("sample_id", ""),
                "run": row.get("run", ""),
                "condition": row.get("condition", ""),
                "node_role": row.get("node_role", ""),
                "source_feature": "L{layer}:P{pos}:F{fid}".format(
                    layer=row.get("source_feature_layer", ""),
                    pos=row.get("source_feature_pos", ""),
                    fid=row.get("source_feature_id", ""),
                ),
                "control_feature": "L{layer}:P{pos}:F{fid}".format(
                    layer=row.get("control_feature_layer", ""),
                    pos=row.get("control_feature_pos", ""),
                    fid=row.get("control_feature_id", ""),
                ),
                "control_match_label": row.get("control_match_label", ""),
                "source_delta_target_logit": _fmt(source_dlogit),
                "control_delta_target_logit": _fmt(control_dlogit),
                "source_minus_control_dlogit": _fmt(source_dlogit - control_dlogit),
                "source_delta_target_prob": _fmt(source_dprob),
                "control_delta_target_prob": _fmt(control_dprob),
                "source_abs_ge_control_abs": str(abs(source_dlogit) >= abs(control_dlogit)).lower(),
            }
        )

    group_rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in paired_rows:
        group_rows[(row["condition"], row["node_role"])].append(row)

    rows_by_source: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in paired_rows:
        rows_by_source[_paired_source_key(row)].append(row)

    per_source_rows: list[dict[str, str]] = []
    per_source_group_rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for key, rows in sorted(rows_by_source.items()):
        bucket, sample_id, run, condition, node_role, source_feature = key
        source_vals = [_safe_float(r["source_delta_target_logit"]) for r in rows]
        control_vals = [_safe_float(r["control_delta_target_logit"]) for r in rows]
        mean_source = _mean(source_vals)
        mean_control = _mean(control_vals)
        per_source_row = {
            "bucket": bucket,
            "sample_id": sample_id,
            "run": run,
            "condition": condition,
            "node_role": node_role,
            "source_feature": source_feature,
            "n_control_draws": str(len(rows)),
            "match_modes": ",".join(sorted({r.get("match_mode", "") for r in rows if r.get("match_mode", "")})),
            "match_labels": ",".join(sorted({r.get("control_match_label", "") for r in rows if r.get("control_match_label", "")})),
            "mean_source_delta_target_logit": _fmt(mean_source),
            "mean_control_delta_target_logit": _fmt(mean_control),
            "mean_source_minus_control_dlogit": _fmt(mean_source - mean_control),
            "frac_source_abs_ge_control_abs": _fmt(
                _mean([1.0 if r["source_abs_ge_control_abs"] == "true" else 0.0 for r in rows])
            ),
        }
        per_source_rows.append(per_source_row)
        per_source_group_rows[(condition, node_role)].append(per_source_row)

    summary_rows: list[dict[str, str]] = []
    for (condition, node_role), rows in sorted(group_rows.items()):
        source_vals = [_safe_float(r["source_delta_target_logit"]) for r in rows]
        control_vals = [_safe_float(r["control_delta_target_logit"]) for r in rows]
        gaps = [_safe_float(r["source_minus_control_dlogit"]) for r in rows]
        summary_rows.append(
            {
                "condition": condition,
                "node_role": node_role,
                "n_pairs": str(len(rows)),
                "mean_source_delta_target_logit": _fmt(_mean(source_vals)),
                "mean_control_delta_target_logit": _fmt(_mean(control_vals)),
                "mean_source_minus_control_dlogit": _fmt(_mean(gaps)),
                "frac_source_abs_ge_control_abs": _fmt(
                    _mean([1.0 if r["source_abs_ge_control_abs"] == "true" else 0.0 for r in rows])
                ),
            }
        )

    per_source_summary_rows: list[dict[str, str]] = []
    for (condition, node_role), rows in sorted(per_source_group_rows.items()):
        per_source_summary_rows.append(
            {
                "condition": condition,
                "node_role": node_role,
                "n_sources": str(len(rows)),
                "mean_source_delta_target_logit": _fmt(_mean([_safe_float(r["mean_source_delta_target_logit"]) for r in rows])),
                "mean_control_delta_target_logit": _fmt(_mean([_safe_float(r["mean_control_delta_target_logit"]) for r in rows])),
                "mean_source_minus_control_dlogit": _fmt(_mean([_safe_float(r["mean_source_minus_control_dlogit"]) for r in rows])),
                "frac_source_abs_ge_control_abs": _fmt(_mean([_safe_float(r["frac_source_abs_ge_control_abs"]) for r in rows])),
            }
        )

    _write_csv(
        out_dir / "modality_matched_control_pairs.csv",
        paired_rows,
        [
            "bucket",
            "sample_id",
            "run",
            "condition",
            "node_role",
            "source_feature",
            "control_feature",
            "match_mode",
            "control_draw_idx",
            "control_pool_size",
            "sampled_match_label",
            "control_match_label",
            "source_delta_target_logit",
            "control_delta_target_logit",
            "source_minus_control_dlogit",
            "source_delta_target_prob",
            "control_delta_target_prob",
            "source_abs_ge_control_abs",
        ],
    )
    _write_csv(
        out_dir / "modality_matched_control_per_source.csv",
        per_source_rows,
        [
            "bucket",
            "sample_id",
            "run",
            "condition",
            "node_role",
            "source_feature",
            "n_control_draws",
            "match_modes",
            "match_labels",
            "mean_source_delta_target_logit",
            "mean_control_delta_target_logit",
            "mean_source_minus_control_dlogit",
            "frac_source_abs_ge_control_abs",
        ],
    )
    _write_csv(
        out_dir / "modality_matched_control_summary.csv",
        summary_rows,
        [
            "condition",
            "node_role",
            "n_pairs",
            "mean_source_delta_target_logit",
            "mean_control_delta_target_logit",
            "mean_source_minus_control_dlogit",
            "frac_source_abs_ge_control_abs",
        ],
    )
    _write_csv(
        out_dir / "modality_matched_control_per_source_summary.csv",
        per_source_summary_rows,
        [
            "condition",
            "node_role",
            "n_sources",
            "mean_source_delta_target_logit",
            "mean_control_delta_target_logit",
            "mean_source_minus_control_dlogit",
            "frac_source_abs_ge_control_abs",
        ],
    )

    lines = [
        "# Modality Matched Control Summary",
        "",
        "This compares the selected source pilot nodes against matched-control nodes using the same sample, condition, and target.",
        "",
        "## Row-Level Summary",
        "",
        "| condition | node_role | n_pairs | mean source dlogit | mean control dlogit | mean source-control | frac |source| >= |control| |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {condition} | {node_role} | {n} | {src} | {ctl} | {gap} | {frac} |".format(
                condition=row["condition"],
                node_role=row["node_role"],
                n=row["n_pairs"],
                src=row["mean_source_delta_target_logit"],
                ctl=row["mean_control_delta_target_logit"],
                gap=row["mean_source_minus_control_dlogit"],
                frac=row["frac_source_abs_ge_control_abs"],
            )
        )

    lines.extend(
        [
            "",
            "## Per-Source Summary",
            "",
            "| condition | node_role | n_sources | mean source dlogit | mean control dlogit | mean source-control | frac |source| >= |control| |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_source_summary_rows:
        lines.append(
            "| {condition} | {node_role} | {n} | {src} | {ctl} | {gap} | {frac} |".format(
                condition=row["condition"],
                node_role=row["node_role"],
                n=row["n_sources"],
                src=row["mean_source_delta_target_logit"],
                ctl=row["mean_control_delta_target_logit"],
                gap=row["mean_source_minus_control_dlogit"],
                frac=row["frac_source_abs_ge_control_abs"],
            )
        )

    (out_dir / "modality_matched_control_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
