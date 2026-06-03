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


def _frac(values: list[bool]) -> float:
    if not values:
        return math.nan
    return sum(1.0 for v in values if v) / len(values)


def _signed_strength(node_role: str, delta_target_logit: float) -> float:
    if math.isnan(delta_target_logit):
        return math.nan
    if node_role == "support":
        return -delta_target_logit
    if node_role == "suppressor":
        return delta_target_logit
    return math.nan


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize label-aware modality follow-up outputs by visual type, knowledge level, bucket, and node role."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rows = _read_csv(Path(args.input).expanduser().resolve())
    if not rows:
        raise ValueError("no rows loaded")
    out_dir = Path(args.out_dir).expanduser().resolve()

    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        visual = (row.get("followup_visual_type_label") or "").strip() or "__all__"
        knowledge = (row.get("followup_knowledge_level_label") or "").strip() or "__all__"
        bucket = (row.get("bucket") or "").strip() or "__all__"
        node_role = (row.get("node_role") or "").strip() or "__all__"
        condition = (row.get("condition") or "").strip() or "__all__"

        grouped[("visual", visual, condition, node_role)].append(row)
        grouped[("knowledge", knowledge, condition, node_role)].append(row)
        grouped[("bucket", bucket, condition, node_role)].append(row)
        grouped[("visual_x_role_bucket", f"{visual}__{bucket}", condition, node_role)].append(row)
        grouped[("all", "__all__", condition, node_role)].append(row)

    summary_rows: list[dict[str, str]] = []
    for (group_family, group_value, condition, node_role), group in sorted(grouped.items()):
        dlogit = [_safe_float(r.get("delta_target_logit")) for r in group]
        dprob = [_safe_float(r.get("delta_target_prob")) for r in group]
        signed = [_signed_strength(node_role, v) for v in dlogit]
        summary_rows.append(
            {
                "group_family": group_family,
                "group_value": group_value,
                "condition": condition,
                "node_role": node_role,
                "n_rows": str(len(group)),
                "n_unique_samples": str(len({(r.get('bucket', ''), r.get('sample_id', '')) for r in group})),
                "n_unique_features": str(
                    len(
                        {
                            (
                                r.get("bucket", ""),
                                r.get("sample_id", ""),
                                r.get("run", ""),
                                r.get("feature_layer", ""),
                                r.get("feature_pos", ""),
                                r.get("feature_id", ""),
                            )
                            for r in group
                        }
                    )
                ),
                "mean_delta_target_logit": _fmt(_mean(dlogit)),
                "mean_delta_target_prob": _fmt(_mean(dprob)),
                "mean_signed_strength": _fmt(_mean(signed)),
                "frac_signed_positive": _fmt(_frac([v > 0 for v in signed if not math.isnan(v)])),
            }
        )

    _write_csv(
        out_dir / "modality_followup_label_summary.csv",
        summary_rows,
        [
            "group_family",
            "group_value",
            "condition",
            "node_role",
            "n_rows",
            "n_unique_samples",
            "n_unique_features",
            "mean_delta_target_logit",
            "mean_delta_target_prob",
            "mean_signed_strength",
            "frac_signed_positive",
        ],
    )

    md_lines = [
        "# Modality Follow-up by Label",
        "",
        "This table is intended for the 24-sample follow-up pack with human visual-type labels.",
        "For support nodes, stronger evidence usually means more negative raw `delta_target_logit` and larger positive `signed_strength`.",
        "For suppressor nodes, stronger evidence usually means more positive raw `delta_target_logit` and larger positive `signed_strength`.",
        "",
        "## Overall",
        "",
        "| condition | node_role | n_rows | n_samples | mean dlogit | mean signed strength |",
        "|---|---|---:|---:|---:|---:|",
    ]

    overall_rows = [
        row
        for row in summary_rows
        if row["group_family"] == "all" and row["group_value"] == "__all__"
    ]
    for row in sorted(overall_rows, key=lambda r: (r["condition"], r["node_role"])):
        md_lines.append(
            "| {condition} | {node_role} | {n_rows} | {n_unique_samples} | {mean_delta_target_logit} | {mean_signed_strength} |".format(
                **row
            )
        )

    md_lines.extend(
        [
            "",
            "## By Visual Type",
            "",
            "| visual_type | condition | node_role | n_rows | n_samples | mean dlogit | mean signed strength |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    visual_rows = [row for row in summary_rows if row["group_family"] == "visual"]
    for row in sorted(visual_rows, key=lambda r: (r["group_value"], r["condition"], r["node_role"])):
        md_lines.append(
            "| {group_value} | {condition} | {node_role} | {n_rows} | {n_unique_samples} | {mean_delta_target_logit} | {mean_signed_strength} |".format(
                **row
            )
        )

    (out_dir / "modality_followup_label_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[done] out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
