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


def _node_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str, str]:
    return (
        row.get("bucket", ""),
        row.get("sample_id", ""),
        row.get("run", ""),
        row.get("node_role", ""),
        row.get("feature_layer", ""),
        row.get("feature_pos", ""),
        row.get("feature_id", ""),
    )


def _sample_key(bucket: str, sample_id: str) -> tuple[str, str]:
    return bucket.strip(), sample_id.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Regroup region-mask results by effective evidence geometry.")
    parser.add_argument("--region-mask-csv", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--geometry-overrides-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    region_rows = _read_csv(Path(args.region_mask_csv).expanduser().resolve())
    label_rows = _read_csv(Path(args.labels_csv).expanduser().resolve())
    override_rows = _read_csv(Path(args.geometry_overrides_csv).expanduser().resolve())
    out_dir = Path(args.out_dir).expanduser().resolve()

    label_map = {(row.get("item_id") or "").strip(): row for row in label_rows}
    override_map = {
        (
            (row.get("bucket") or "").strip(),
            (row.get("sample_id") or "").strip(),
            (row.get("run") or "").strip(),
        ): row
        for row in override_rows
    }

    per_node_rows: list[dict[str, str]] = []
    by_node: dict[tuple[str, ...], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in region_rows:
        by_node[_node_key(row)][row.get("condition", "")] = row

    for key, by_condition in sorted(by_node.items()):
        bucket, sample_id, run, node_role, layer, pos, feature_id = key
        item_id = f"{bucket}__{sample_id}"
        label_row = label_map.get(item_id, {})
        override_row = override_map.get((bucket, sample_id, run), {})
        base_geometry = (label_row.get("visual_type_label") or "").strip()
        effective_geometry = (override_row.get("evidence_geometry_override") or base_geometry).strip()
        geometry_source = "override" if override_row else "base_type_label"

        clean = _safe_float(by_condition.get("clean", {}).get("delta_target_logit"))
        answer = _safe_float(by_condition.get("answer_mask", {}).get("delta_target_logit"))
        relate = _safe_float(by_condition.get("relate_mask", {}).get("delta_target_logit"))
        union = _safe_float(by_condition.get("union_mask", {}).get("delta_target_logit"))
        control = _safe_float(by_condition.get("auto_control_mask", {}).get("delta_target_logit"))

        per_node_rows.append(
            {
                "bucket": bucket,
                "sample_id": sample_id,
                "run": run,
                "node_role": node_role,
                "feature_layer": layer,
                "feature_pos": pos,
                "feature_id": feature_id,
                "base_geometry": base_geometry,
                "effective_geometry": effective_geometry,
                "geometry_source": geometry_source,
                "geometry_note": (override_row.get("evidence_geometry_notes") or "").strip(),
                "knowledge_level_label": (label_row.get("knowledge_level_label") or "").strip(),
                "clean_delta_target_logit": _fmt(clean),
                "answer_mask_delta_target_logit": _fmt(answer),
                "relate_mask_delta_target_logit": _fmt(relate),
                "union_mask_delta_target_logit": _fmt(union),
                "auto_control_mask_delta_target_logit": _fmt(control),
                "answer_minus_clean": _fmt(answer - clean if not math.isnan(answer) and not math.isnan(clean) else math.nan),
                "relate_minus_clean": _fmt(relate - clean if not math.isnan(relate) and not math.isnan(clean) else math.nan),
                "union_minus_clean": _fmt(union - clean if not math.isnan(union) and not math.isnan(clean) else math.nan),
                "auto_control_minus_clean": _fmt(control - clean if not math.isnan(control) and not math.isnan(clean) else math.nan),
            }
        )

    _write_csv(
        out_dir / "region_mask_per_node_with_effective_geometry.csv",
        per_node_rows,
        [
            "bucket",
            "sample_id",
            "run",
            "node_role",
            "feature_layer",
            "feature_pos",
            "feature_id",
            "base_geometry",
            "effective_geometry",
            "geometry_source",
            "geometry_note",
            "knowledge_level_label",
            "clean_delta_target_logit",
            "answer_mask_delta_target_logit",
            "relate_mask_delta_target_logit",
            "union_mask_delta_target_logit",
            "auto_control_mask_delta_target_logit",
            "answer_minus_clean",
            "relate_minus_clean",
            "union_minus_clean",
            "auto_control_minus_clean",
        ],
    )

    summary_rows: list[dict[str, str]] = []
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in per_node_rows:
        grouped[(row["node_role"], row["effective_geometry"])].append(row)

    for (node_role, effective_geometry), rows in sorted(grouped.items()):
        answer_gap = [_safe_float(row.get("answer_minus_clean")) for row in rows]
        relate_gap = [_safe_float(row.get("relate_minus_clean")) for row in rows]
        union_gap = [_safe_float(row.get("union_minus_clean")) for row in rows]
        control_gap = [_safe_float(row.get("auto_control_minus_clean")) for row in rows]
        summary_rows.append(
            {
                "node_role": node_role,
                "effective_geometry": effective_geometry,
                "n_rows": str(len(rows)),
                "n_unique_samples": str(len({(row["bucket"], row["sample_id"]) for row in rows})),
                "mean_answer_minus_clean": _fmt(_mean(answer_gap)),
                "mean_relate_minus_clean": _fmt(_mean(relate_gap)),
                "mean_union_minus_clean": _fmt(_mean(union_gap)),
                "mean_control_minus_clean": _fmt(_mean(control_gap)),
            }
        )

    _write_csv(
        out_dir / "region_mask_geometry_summary.csv",
        summary_rows,
        [
            "node_role",
            "effective_geometry",
            "n_rows",
            "n_unique_samples",
            "mean_answer_minus_clean",
            "mean_relate_minus_clean",
            "mean_union_minus_clean",
            "mean_control_minus_clean",
        ],
    )

    md_lines = [
        "# Region Mask by Effective Evidence Geometry",
        "",
        "This summary uses the base `visual_type_label` by default, but applies",
        "sample-level evidence-geometry overrides where human review indicated that the",
        "original coarse label was not the best description for the mask interpretation task.",
        "",
        "For support nodes, a larger positive `*_minus_clean` means the mask weakened the",
        "support effect more strongly relative to clean.",
        "For suppressor nodes, a more negative `*_minus_clean` means the mask weakened the",
        "suppressor effect more strongly relative to clean.",
        "",
        "| node_role | effective_geometry | n_rows | answer-clean | relate-clean | union-clean | control-clean |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        md_lines.append(
            "| {node_role} | {effective_geometry} | {n_rows} | {mean_answer_minus_clean} | {mean_relate_minus_clean} | {mean_union_minus_clean} | {mean_control_minus_clean} |".format(
                **row
            )
        )

    (out_dir / "region_mask_geometry_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[done] out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
