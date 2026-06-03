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


def _per_node_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row.get("bucket", ""),
        row.get("sample_id", ""),
        row.get("run", ""),
        row.get("node_role", ""),
        row.get("feature_layer", ""),
        row.get("feature_pos", ""),
        row.get("feature_id", ""),
    )


def _condition_sort_key(condition: str) -> tuple[int, str]:
    priority = {
        "clean": 0,
        "no_image": 1,
        "wrong_image": 2,
        "masked_image": 3,
    }
    return (priority.get(condition, 100), condition)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize modality-counterfactual intervention pilot outputs.")
    parser.add_argument("--input", required=True, help="Pilot CSV produced by run_modality_counterfactual_intervention_pilot.py")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    rows = _read_csv(input_path)
    if not rows:
        raise ValueError(f"no rows loaded from {input_path}")

    group_rows: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        bucket = row.get("bucket", "") or "__all__"
        run = row.get("run", "") or "__all__"
        condition = row.get("condition", "") or "__all__"
        node_role = row.get("node_role", "") or "__all__"
        group_rows[(bucket, run, condition, node_role)].append(row)
        group_rows[("__all__", "__all__", condition, node_role)].append(row)

    summary_rows: list[dict[str, str]] = []
    for (bucket, run, condition, node_role), group in sorted(group_rows.items()):
        dlogit = [_safe_float(r.get("delta_target_logit")) for r in group]
        dprob = [_safe_float(r.get("delta_target_prob")) for r in group]
        original_logit = [_safe_float(r.get("original_target_logit")) for r in group]
        summary_rows.append(
            {
                "bucket": bucket,
                "run": run,
                "condition": condition,
                "node_role": node_role,
                "n_rows": str(len(group)),
                "mean_delta_target_logit": _fmt(_mean(dlogit)),
                "mean_delta_target_prob": _fmt(_mean(dprob)),
                "mean_original_target_logit": _fmt(_mean(original_logit)),
                "frac_negative_delta_target_logit": _fmt(
                    _mean([1.0 if v < 0 else 0.0 for v in dlogit if not math.isnan(v)])
                ),
            }
        )

    per_node_rows: list[dict[str, str]] = []
    per_node_conditions: dict[tuple[str, ...], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        per_node_conditions[_per_node_key(row)][row.get("condition", "")] = row

    all_conditions = sorted({row.get("condition", "") for row in rows if row.get("condition", "")}, key=_condition_sort_key)
    non_clean_conditions = [condition for condition in all_conditions if condition != "clean"]

    sign_flip_rows: list[dict[str, str]] = []
    for key, by_condition in sorted(per_node_conditions.items()):
        bucket, sample_id, run, node_role, layer, pos, feature_id = key
        clean_row = by_condition.get("clean", {})
        clean_dlogit = _safe_float(clean_row.get("delta_target_logit"))
        out_row = {
            "bucket": bucket,
            "sample_id": sample_id,
            "run": run,
            "node_role": node_role,
            "feature_layer": layer,
            "feature_pos": pos,
            "feature_id": feature_id,
            "clean_delta_target_logit": _fmt(clean_dlogit),
            "clean_original_target_logit": clean_row.get("original_target_logit", ""),
        }

        for condition in non_clean_conditions:
            condition_row = by_condition.get(condition, {})
            condition_dlogit = _safe_float(condition_row.get("delta_target_logit"))
            delta_minus_clean = (
                condition_dlogit - clean_dlogit if not math.isnan(clean_dlogit) and not math.isnan(condition_dlogit) else math.nan
            )
            out_row[f"{condition}_delta_target_logit"] = _fmt(condition_dlogit)
            out_row[f"{condition}_minus_clean"] = _fmt(delta_minus_clean)
            out_row[f"{condition}_original_target_logit"] = condition_row.get("original_target_logit", "")
        per_node_rows.append(out_row)

        for condition in non_clean_conditions:
            condition_row = by_condition.get(condition, {})
            delta = _safe_float(condition_row.get("delta_target_logit"))
            if math.isnan(clean_dlogit) or math.isnan(delta) or clean_dlogit == 0.0:
                continue
            if (clean_dlogit < 0 < delta) or (clean_dlogit > 0 > delta):
                sign_flip_rows.append(
                    {
                        "bucket": bucket,
                        "sample_id": sample_id,
                        "run": run,
                        "node_role": node_role,
                        "feature_layer": layer,
                        "feature_pos": pos,
                        "feature_id": feature_id,
                        "condition": condition,
                        "clean_delta_target_logit": _fmt(clean_dlogit),
                        "condition_delta_target_logit": _fmt(delta),
                    }
                )

    _write_csv(
        out_dir / "modality_pilot_summary.csv",
        summary_rows,
        [
            "bucket",
            "run",
            "condition",
            "node_role",
            "n_rows",
            "mean_delta_target_logit",
            "mean_delta_target_prob",
            "mean_original_target_logit",
            "frac_negative_delta_target_logit",
        ],
    )
    _write_csv(
        out_dir / "modality_pilot_per_node.csv",
        per_node_rows,
        (
            [
            "bucket",
            "sample_id",
            "run",
            "node_role",
            "feature_layer",
            "feature_pos",
            "feature_id",
            "clean_delta_target_logit",
            "clean_original_target_logit",
            ]
            + [f"{condition}_delta_target_logit" for condition in non_clean_conditions]
            + [f"{condition}_minus_clean" for condition in non_clean_conditions]
            + [f"{condition}_original_target_logit" for condition in non_clean_conditions]
        ),
    )
    _write_csv(
        out_dir / "modality_pilot_sign_flips.csv",
        sign_flip_rows,
        [
            "bucket",
            "sample_id",
            "run",
            "node_role",
            "feature_layer",
            "feature_pos",
            "feature_id",
            "condition",
            "clean_delta_target_logit",
            "condition_delta_target_logit",
        ],
    )

    overall_rows = [r for r in summary_rows if r["bucket"] == "__all__" and r["run"] == "__all__"]
    support_rows = [r for r in overall_rows if r["node_role"] == "support"]
    suppressor_rows = [r for r in overall_rows if r["node_role"] == "suppressor"]

    lines = [
        "# Modality Counterfactual Pilot Summary",
        "",
        f"Source CSV: `{input_path}`",
        "",
        "A more negative `delta_target_logit` means zeroing the node hurt the target more.",
        "For support nodes, weakening toward zero under `no_image` or `wrong_image` suggests image-sensitive support.",
        "For suppressor nodes, a stable positive sign suggests the node remains a target suppressor under corruption.",
        "",
        "## Overall Aggregates",
        "",
        "| condition | node_role | n | mean dlogit | mean dprob | mean original logit | frac dlogit<0 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(overall_rows, key=lambda r: (r["condition"], r["node_role"])):
        lines.append(
            "| {condition} | {node_role} | {n} | {dlogit} | {dprob} | {ologit} | {fneg} |".format(
                condition=row["condition"],
                node_role=row["node_role"],
                n=row["n_rows"],
                dlogit=row["mean_delta_target_logit"],
                dprob=row["mean_delta_target_prob"],
                ologit=row["mean_original_target_logit"],
                fneg=row["frac_negative_delta_target_logit"],
            )
        )

    lines.extend(
        [
            "",
            "## Per-Node Condition Comparison",
            "",
            "| bucket | sample_id | run | node_role | feature | clean dlogit |"
            + "".join(f" {condition} dlogit |" for condition in non_clean_conditions)
            + "".join(f" {condition}-clean |" for condition in non_clean_conditions),
            "|---|---|---|---|---|---:"
            + "".join("|---:" for _ in non_clean_conditions)
            + "".join("|---:" for _ in non_clean_conditions)
            + "|",
        ]
    )
    for row in per_node_rows:
        cells = [
            row["bucket"],
            row["sample_id"],
            row["run"],
            row["node_role"],
            f"L{row['feature_layer']}:P{row['feature_pos']}:F{row['feature_id']}",
            row["clean_delta_target_logit"],
        ]
        cells.extend(row.get(f"{condition}_delta_target_logit", "") for condition in non_clean_conditions)
        cells.extend(row.get(f"{condition}_minus_clean", "") for condition in non_clean_conditions)
        lines.append(
            "| " + " | ".join(cells) + " |"
        )

    if sign_flip_rows:
        lines.extend(
            [
                "",
                "## Sign Flips",
                "",
                "| bucket | sample_id | run | node_role | feature | condition | clean dlogit | condition dlogit |",
                "|---|---|---|---|---|---|---:|---:|",
            ]
        )
        for row in sign_flip_rows:
            lines.append(
                "| {bucket} | {sample_id} | {run} | {node_role} | L{layer}:P{pos}:F{feature} | {condition} | {clean} | {cond} |".format(
                    bucket=row["bucket"],
                    sample_id=row["sample_id"],
                    run=row["run"],
                    node_role=row["node_role"],
                    layer=row["feature_layer"],
                    pos=row["feature_pos"],
                    feature=row["feature_id"],
                    condition=row["condition"],
                    clean=row["clean_delta_target_logit"],
                    cond=row["condition_delta_target_logit"],
                )
            )

    support_clean = next((r for r in support_rows if r["condition"] == "clean"), None)
    suppressor_clean = next((r for r in suppressor_rows if r["condition"] == "clean"), None)

    lines.extend(
        [
            "",
            "## Quick Read",
            "",
            "- Support nodes under `clean` have the strongest average negative effect.",
            "- If `no_image` / `wrong_image` move support effects toward zero, that is evidence for image-sensitive support.",
            "- If suppressor effects stay positive under corruption, that is evidence for relatively image-insensitive suppressive routing.",
            "",
            "Concrete aggregate values:",
            "",
            f"- support clean mean dlogit: `{support_clean['mean_delta_target_logit'] if support_clean else ''}`",
            f"- suppressor clean mean dlogit: `{suppressor_clean['mean_delta_target_logit'] if suppressor_clean else ''}`",
        ]
    )
    for condition in non_clean_conditions:
        support_row = next((r for r in support_rows if r["condition"] == condition), None)
        suppressor_row = next((r for r in suppressor_rows if r["condition"] == condition), None)
        lines.append(f"- support {condition} mean dlogit: `{support_row['mean_delta_target_logit'] if support_row else ''}`")
        lines.append(f"- suppressor {condition} mean dlogit: `{suppressor_row['mean_delta_target_logit'] if suppressor_row else ''}`")

    (out_dir / "modality_pilot_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
