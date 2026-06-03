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


def _frac_positive(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return math.nan
    return sum(1.0 for v in clean if v > 0) / len(clean)


def _sample_key(bucket: str, sample_id: str) -> tuple[str, str]:
    return (bucket.strip(), sample_id.strip())


def _load_label_map(manifest_path: Path, labels_path: Path) -> dict[tuple[str, str], dict[str, str]]:
    manifest_rows = _read_csv(manifest_path)
    label_rows = _read_csv(labels_path)
    labels_by_item = {(row.get("item_id") or "").strip(): row for row in label_rows}

    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in manifest_rows:
        item_id = (row.get("item_id") or "").strip()
        label_row = labels_by_item.get(item_id, {})
        bucket = (row.get("bucket") or "").strip()
        sample_id = (row.get("sample_id") or "").strip()
        out[_sample_key(bucket, sample_id)] = {
            "item_id": item_id,
            "bucket": bucket,
            "sample_id": sample_id,
            "display_question": (row.get("display_question") or "").strip(),
            "answer_text": (row.get("answer_text") or "").strip(),
            "priority": (row.get("priority") or "").strip(),
            "rank": (row.get("rank") or "").strip(),
            "visual_type_label": (label_row.get("visual_type_label") or "").strip(),
            "knowledge_level_label": (label_row.get("knowledge_level_label") or "").strip(),
            "label_notes": (label_row.get("label_notes") or "").strip(),
        }
    return out


def _signed_strength(node_role: str, delta_target_logit: float) -> float:
    if math.isnan(delta_target_logit):
        return math.nan
    if node_role == "support":
        return -delta_target_logit
    if node_role == "suppressor":
        return delta_target_logit
    return math.nan


def _group_specs(row: dict[str, str]) -> list[tuple[str, str]]:
    visual = row.get("visual_type_label", "")
    knowledge = row.get("knowledge_level_label", "")
    bucket = row.get("bucket", "")
    specs = [("all", "__all__")]
    if visual:
        specs.append(("visual", visual))
    if knowledge:
        specs.append(("knowledge", knowledge))
    if visual and knowledge:
        specs.append(("visual_x_knowledge", f"{visual}__{knowledge}"))
    if bucket:
        specs.append(("bucket", bucket))
    if bucket and visual:
        specs.append(("bucket_x_visual", f"{bucket}__{visual}"))
    if bucket and knowledge:
        specs.append(("bucket_x_knowledge", f"{bucket}__{knowledge}"))
    return specs


def _summarize_modality(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        node_role = row.get("node_role", "")
        for family, group_value in _group_specs(row):
            grouped[(family, group_value, node_role)].append(row)

    out_rows: list[dict[str, str]] = []
    for (family, group_value, node_role), group in sorted(grouped.items()):
        clean_strength = [_safe_float(r.get("clean_signed_strength")) for r in group]
        no_image_strength = [_safe_float(r.get("no_image_signed_strength")) for r in group]
        wrong_image_strength = [_safe_float(r.get("wrong_image_signed_strength")) for r in group]
        masked_image_strength = [_safe_float(r.get("masked_image_signed_strength")) for r in group]
        no_image_weakening = [_safe_float(r.get("no_image_weakening")) for r in group]
        wrong_image_weakening = [_safe_float(r.get("wrong_image_weakening")) for r in group]
        masked_image_weakening = [_safe_float(r.get("masked_image_weakening")) for r in group]
        out_rows.append(
            {
                "group_family": family,
                "group_value": group_value,
                "node_role": node_role,
                "n_rows": str(len(group)),
                "n_unique_samples": str(len({(r.get('bucket', ''), r.get('sample_id', '')) for r in group})),
                "mean_clean_signed_strength": _fmt(_mean(clean_strength)),
                "mean_no_image_signed_strength": _fmt(_mean(no_image_strength)),
                "mean_wrong_image_signed_strength": _fmt(_mean(wrong_image_strength)),
                "mean_masked_image_signed_strength": _fmt(_mean(masked_image_strength)),
                "mean_no_image_weakening": _fmt(_mean(no_image_weakening)),
                "mean_wrong_image_weakening": _fmt(_mean(wrong_image_weakening)),
                "mean_masked_image_weakening": _fmt(_mean(masked_image_weakening)),
            }
        )
    return out_rows


def _summarize_restoration(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        node_role = row.get("node_role", "")
        for family, group_value in _group_specs(row):
            grouped[(family, group_value, node_role)].append(row)

    out_rows: list[dict[str, str]] = []
    for (family, group_value, node_role), group in sorted(grouped.items()):
        signed_restore = [_safe_float(r.get("signed_restore_effect")) for r in group]
        signed_zero = [_safe_float(r.get("signed_zero_effect")) for r in group]
        feature_gap = [_safe_float(r.get("feature_value_gap")) for r in group]
        clean_gap = [_safe_float(r.get("clean_corrupt_logit_gap")) for r in group]
        out_rows.append(
            {
                "group_family": family,
                "group_value": group_value,
                "node_role": node_role,
                "n_rows": str(len(group)),
                "n_unique_samples": str(len({(r.get('bucket', ''), r.get('sample_id', '')) for r in group})),
                "mean_signed_restore_effect": _fmt(_mean(signed_restore)),
                "frac_positive_signed_restore": _fmt(_frac_positive(signed_restore)),
                "mean_signed_zero_effect": _fmt(_mean(signed_zero)),
                "frac_positive_signed_zero": _fmt(_frac_positive(signed_zero)),
                "mean_feature_value_gap": _fmt(_mean(feature_gap)),
                "mean_clean_corrupt_logit_gap": _fmt(_mean(clean_gap)),
            }
        )
    return out_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage 1 type labels against modality/restoration pilot outputs.")
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--modality-per-node-csv", required=True)
    parser.add_argument("--restoration-support-csv", required=True)
    parser.add_argument("--restoration-suppressor-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest_csv).expanduser().resolve()
    labels_path = Path(args.labels_csv).expanduser().resolve()
    modality_path = Path(args.modality_per_node_csv).expanduser().resolve()
    restoration_support_path = Path(args.restoration_support_csv).expanduser().resolve()
    restoration_suppressor_path = Path(args.restoration_suppressor_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    label_map = _load_label_map(manifest_path, labels_path)

    merged_manifest_rows = list(label_map.values())
    _write_csv(
        out_dir / "type_labeled_manifest.csv",
        merged_manifest_rows,
        [
            "item_id",
            "bucket",
            "sample_id",
            "display_question",
            "answer_text",
            "priority",
            "rank",
            "visual_type_label",
            "knowledge_level_label",
            "label_notes",
        ],
    )

    modality_rows_raw = _read_csv(modality_path)
    modality_joined_rows: list[dict[str, str]] = []
    for row in modality_rows_raw:
        key = _sample_key(row.get("bucket", ""), row.get("sample_id", ""))
        label_info = label_map.get(key, {})
        clean_strength = _signed_strength(row.get("node_role", ""), _safe_float(row.get("clean_delta_target_logit")))
        no_image_strength = _signed_strength(row.get("node_role", ""), _safe_float(row.get("no_image_delta_target_logit")))
        wrong_image_strength = _signed_strength(row.get("node_role", ""), _safe_float(row.get("wrong_image_delta_target_logit")))
        masked_image_strength = _signed_strength(row.get("node_role", ""), _safe_float(row.get("masked_image_delta_target_logit")))
        joined = dict(row)
        joined.update(label_info)
        joined["clean_signed_strength"] = _fmt(clean_strength)
        joined["no_image_signed_strength"] = _fmt(no_image_strength)
        joined["wrong_image_signed_strength"] = _fmt(wrong_image_strength)
        joined["masked_image_signed_strength"] = _fmt(masked_image_strength)
        joined["no_image_weakening"] = _fmt(clean_strength - no_image_strength if not math.isnan(clean_strength) and not math.isnan(no_image_strength) else math.nan)
        joined["wrong_image_weakening"] = _fmt(clean_strength - wrong_image_strength if not math.isnan(clean_strength) and not math.isnan(wrong_image_strength) else math.nan)
        joined["masked_image_weakening"] = _fmt(clean_strength - masked_image_strength if not math.isnan(clean_strength) and not math.isnan(masked_image_strength) else math.nan)
        modality_joined_rows.append(joined)

    modality_fieldnames = list(modality_joined_rows[0].keys()) if modality_joined_rows else []
    _write_csv(out_dir / "modality_per_node_with_type_labels.csv", modality_joined_rows, modality_fieldnames)
    modality_summary_rows = _summarize_modality(modality_joined_rows)
    _write_csv(
        out_dir / "modality_type_label_summary.csv",
        modality_summary_rows,
        [
            "group_family",
            "group_value",
            "node_role",
            "n_rows",
            "n_unique_samples",
            "mean_clean_signed_strength",
            "mean_no_image_signed_strength",
            "mean_wrong_image_signed_strength",
            "mean_masked_image_signed_strength",
            "mean_no_image_weakening",
            "mean_wrong_image_weakening",
            "mean_masked_image_weakening",
        ],
    )

    restoration_rows_raw = _read_csv(restoration_support_path) + _read_csv(restoration_suppressor_path)
    restoration_joined_rows: list[dict[str, str]] = []
    for row in restoration_rows_raw:
        key = _sample_key(row.get("bucket", ""), row.get("sample_id", ""))
        label_info = label_map.get(key, {})
        restore_effect = _safe_float(row.get("restore_minus_corrupt_target_logit"))
        zero_effect = _safe_float(row.get("zero_minus_corrupt_target_logit"))
        node_role = row.get("node_role", "")
        signed_restore = restore_effect if node_role == "support" else -restore_effect
        signed_zero = -zero_effect if node_role == "support" else zero_effect
        joined = dict(row)
        joined.update(label_info)
        joined["signed_restore_effect"] = _fmt(signed_restore)
        joined["signed_zero_effect"] = _fmt(signed_zero)
        restoration_joined_rows.append(joined)

    restoration_fieldnames = list(restoration_joined_rows[0].keys()) if restoration_joined_rows else []
    _write_csv(out_dir / "restoration_with_type_labels.csv", restoration_joined_rows, restoration_fieldnames)
    restoration_summary_rows = _summarize_restoration(restoration_joined_rows)
    _write_csv(
        out_dir / "restoration_type_label_summary.csv",
        restoration_summary_rows,
        [
            "group_family",
            "group_value",
            "node_role",
            "n_rows",
            "n_unique_samples",
            "mean_signed_restore_effect",
            "frac_positive_signed_restore",
            "mean_signed_zero_effect",
            "frac_positive_signed_zero",
            "mean_feature_value_gap",
            "mean_clean_corrupt_logit_gap",
        ],
    )

    report_lines = [
        "# Stage 1 Type-Label Analysis",
        "",
        f"Manifest: `{manifest_path}`",
        f"Labels: `{labels_path}`",
        f"Modality rows: `{len(modality_joined_rows)}`",
        f"Restoration rows: `{len(restoration_joined_rows)}`",
        "",
        "## Quick counts",
        "",
        f"- labeled samples: `{len(merged_manifest_rows)}`",
        f"- modality traced nodes: `{len(modality_joined_rows)}`",
        f"- restoration candidates: `{len(restoration_joined_rows)}`",
        "",
        "## Files",
        "",
        "- `type_labeled_manifest.csv`",
        "- `modality_per_node_with_type_labels.csv`",
        "- `modality_type_label_summary.csv`",
        "- `restoration_with_type_labels.csv`",
        "- `restoration_type_label_summary.csv`",
    ]
    (out_dir / "README.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[done] out_dir={out_dir}")
    print(f"[done] labeled_samples={len(merged_manifest_rows)}")
    print(f"[done] modality_rows={len(modality_joined_rows)}")
    print(f"[done] restoration_rows={len(restoration_joined_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
