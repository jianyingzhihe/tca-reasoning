#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SOURCE_MODALITY_CSVS = [
    ROOT / "remote_sync" / "2026-05-14_bd_visual_positive_strong12_main64" / "modality_pilot_visual_positive_strong12_clean_core7_generic6_nobuf_full7_cap0_relpos.csv",
    ROOT / "remote_sync" / "2026-05-14_bd_visual_positive_strong18_next_main64" / "modality_pilot_visual_positive_strong18_next_clean_core10_generic8_nobuf_full10_cap0_relpos.csv",
]

DEFAULT_NEAREST_CONTROL_CSVS = [
    ROOT / "remote_sync" / "2026-05-14_bd_visual_positive_strong12_main64" / "modality_matched_control_visual_positive_strong12_clean_core7_generic6_nobuf_full7_cap0_relpos.csv",
    ROOT / "remote_sync" / "2026-05-14_bd_visual_positive_strong18_next_main64" / "modality_matched_control_visual_positive_strong18_next_clean_core10_generic8_nobuf_full10_cap0_relpos.csv",
]

RUN_TO_PROMPT = {
    "A": "D_visual_only",
    "B": "B_direct",
}

DEFAULT_ASSISTANT_PREFIX = "The answer is "


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_clean_source_rows(paths: list[Path]) -> dict[tuple[str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in paths:
        for row in _read_csv(path):
            if row.get("condition") != "clean":
                continue
            key = (
                (row.get("sample_id") or "").strip(),
                (row.get("run") or "").strip(),
                (row.get("node_role") or "").strip(),
            )
            if all(key) and key not in out:
                out[key] = row
    return out


def _load_clean_nearest_rows(paths: list[Path]) -> dict[tuple[str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in paths:
        for row in _read_csv(path):
            if row.get("condition") != "clean":
                continue
            if row.get("match_mode") != "nearest":
                continue
            key = (
                (row.get("sample_id") or "").strip(),
                (row.get("run") or "").strip(),
                (row.get("node_role") or "").strip(),
            )
            if all(key) and key not in out:
                out[key] = row
    return out


def _label_counts(json_path: Path) -> tuple[int, int]:
    if not json_path.exists():
        return 0, 0
    data = json.loads(json_path.read_text(encoding="utf-8"))
    shapes = data.get("shapes", [])
    answer_count = sum(1 for shape in shapes if shape.get("label") == "answer")
    relate_count = sum(1 for shape in shapes if shape.get("label") == "relate")
    return answer_count, relate_count


def _iter_complete_pack_rows(pack_dir: Path) -> list[dict[str, str]]:
    manifest_path = pack_dir / "manifest.csv"
    rows = _read_csv(manifest_path)
    out: list[dict[str, str]] = []
    for row in rows:
        image_filename = row.get("image_filename", "")
        stem = Path(image_filename).stem
        json_path = pack_dir / "images" / f"{stem}.json"
        answer_count, relate_count = _label_counts(json_path)
        if answer_count <= 0 or relate_count <= 0:
            continue
        copied = dict(row)
        copied["annotation_json_path"] = str(json_path)
        copied["answer_shape_count"] = str(answer_count)
        copied["relate_shape_count"] = str(relate_count)
        out.append(copied)
    return out


def _pack_row_by_sample(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = (row.get("sample_id") or "").strip()
        if sample_id and sample_id not in out:
            out[sample_id] = row
    return out


def _make_source_row(pack: dict[str, str], source_row: dict[str, str], run: str, node_role: str) -> dict[str, str]:
    return {
        "sample_id": pack.get("sample_id", ""),
        "run": run,
        "prompt_name": RUN_TO_PROMPT.get(run, run),
        "node_role": node_role,
        "node_source": "source",
        "assistant_prefix": source_row.get("assistant_prefix", DEFAULT_ASSISTANT_PREFIX),
        "question": source_row.get("question", pack.get("question_text", "")),
        "image_path": pack.get("local_image_path", ""),
        "remote_image_path": pack.get("remote_image_path", ""),
        "feature_layer": source_row.get("feature_layer", ""),
        "feature_pos": source_row.get("feature_pos", ""),
        "feature_id": source_row.get("feature_id", ""),
        "target_token_id": source_row.get("target_token_id", ""),
        "reasoning_operation": pack.get("reasoning_operation", pack.get("legacy_reasoning_operation", "")),
        "visual_structure": pack.get("visual_structure", pack.get("legacy_visual_structure", "")),
        "image_dependence": pack.get("image_dependence", pack.get("legacy_image_dependence", "")),
        "selection_tier": pack.get("selection_tier", ""),
        "annotation_pack": pack.get("annotation_pack", ""),
        "annotation_json_path": pack.get("annotation_json_path", ""),
        "answer_shape_count": pack.get("answer_shape_count", ""),
        "relate_shape_count": pack.get("relate_shape_count", ""),
        "reference_clean_delta_target_logit": source_row.get("reference_clean_delta_target_logit", source_row.get("delta_target_logit", "")),
        "position_alignment": source_row.get("position_alignment", ""),
        "clean_seq_len": source_row.get("clean_seq_len", ""),
        "source_bucket": source_row.get("bucket", ""),
        "match_mode": "",
        "sampled_match_label": "",
        "control_draw_idx": "",
    }


def _make_nearest_row(pack: dict[str, str], nearest_row: dict[str, str], run: str, node_role: str) -> dict[str, str]:
    return {
        "sample_id": pack.get("sample_id", ""),
        "run": run,
        "prompt_name": RUN_TO_PROMPT.get(run, run),
        "node_role": node_role,
        "node_source": "nearest_control",
        "assistant_prefix": nearest_row.get("assistant_prefix", DEFAULT_ASSISTANT_PREFIX),
        "question": nearest_row.get("question", pack.get("question_text", "")),
        "image_path": pack.get("local_image_path", ""),
        "remote_image_path": pack.get("remote_image_path", ""),
        "feature_layer": nearest_row.get("control_feature_layer", ""),
        "feature_pos": nearest_row.get("control_feature_pos", ""),
        "feature_id": nearest_row.get("control_feature_id", ""),
        "target_token_id": nearest_row.get("target_token_id", ""),
        "reasoning_operation": pack.get("reasoning_operation", pack.get("legacy_reasoning_operation", "")),
        "visual_structure": pack.get("visual_structure", pack.get("legacy_visual_structure", "")),
        "image_dependence": pack.get("image_dependence", pack.get("legacy_image_dependence", "")),
        "selection_tier": pack.get("selection_tier", ""),
        "annotation_pack": pack.get("annotation_pack", ""),
        "annotation_json_path": pack.get("annotation_json_path", ""),
        "answer_shape_count": pack.get("answer_shape_count", ""),
        "relate_shape_count": pack.get("relate_shape_count", ""),
        "reference_clean_delta_target_logit": nearest_row.get("delta_target_logit", ""),
        "position_alignment": nearest_row.get("control_position_alignment", ""),
        "clean_seq_len": nearest_row.get("clean_seq_len", ""),
        "source_bucket": nearest_row.get("bucket", ""),
        "match_mode": nearest_row.get("match_mode", ""),
        "sampled_match_label": nearest_row.get("sampled_match_label", ""),
        "control_draw_idx": nearest_row.get("control_draw_idx", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build region experiment manifest for a completed annotation pack by intersecting with clean-core source and nearest-control rows.")
    parser.add_argument("--annotation-pack-dir", required=True)
    parser.add_argument(
        "--source-modality-csv",
        action="append",
        default=[],
        help="Clean/source modality CSV. Can be passed multiple times. Defaults to the original strong12/strong18 CSVs.",
    )
    parser.add_argument(
        "--nearest-control-csv",
        action="append",
        default=[],
        help="Nearest matched-control CSV. Can be passed multiple times. Defaults to the original strong12/strong18 CSVs.",
    )
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--summary-md", default="")
    args = parser.parse_args()

    pack_dir = Path(args.annotation_pack_dir).expanduser().resolve()
    if not pack_dir.exists():
        raise FileNotFoundError(f"missing pack dir: {pack_dir}")

    complete_rows = _iter_complete_pack_rows(pack_dir)
    if not complete_rows:
        raise ValueError(f"no complete answer+relate annotations found in {pack_dir}")

    for row in complete_rows:
        row["annotation_pack"] = pack_dir.name

    pack_by_sample = _pack_row_by_sample(complete_rows)
    source_csvs = [Path(path).expanduser().resolve() for path in args.source_modality_csv] or DEFAULT_SOURCE_MODALITY_CSVS
    nearest_csvs = [Path(path).expanduser().resolve() for path in args.nearest_control_csv] or DEFAULT_NEAREST_CONTROL_CSVS
    source_rows = _load_clean_source_rows(source_csvs)
    nearest_rows = _load_clean_nearest_rows(nearest_csvs)

    region_rows: list[dict[str, str]] = []
    for sample_id in sorted(pack_by_sample):
        pack = pack_by_sample[sample_id]
        for run in ("A", "B"):
            for node_role in ("support", "suppressor"):
                key = (sample_id, run, node_role)
                if key in source_rows:
                    region_rows.append(_make_source_row(pack, source_rows[key], run, node_role))
                if key in nearest_rows:
                    region_rows.append(_make_nearest_row(pack, nearest_rows[key], run, node_role))

    if not region_rows:
        raise ValueError(f"no overlapping source/control rows found for {pack_dir}")

    out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else pack_dir / "region_experiment_manifest.csv"
    summary_json = Path(args.summary_json).expanduser().resolve() if args.summary_json else pack_dir / "region_experiment_summary.json"
    summary_md = Path(args.summary_md).expanduser().resolve() if args.summary_md else pack_dir / "region_experiment_summary.md"

    fieldnames = list(region_rows[0].keys())
    _write_csv(out_csv, region_rows, fieldnames)

    overlap_source_ids = sorted({row["sample_id"] for row in region_rows if row["node_source"] == "source"})
    overlap_nearest_ids = sorted({row["sample_id"] for row in region_rows if row["node_source"] == "nearest_control"})
    support_source_rows = sum(1 for row in region_rows if row["node_source"] == "source" and row["node_role"] == "support")
    support_nearest_rows = sum(1 for row in region_rows if row["node_source"] == "nearest_control" and row["node_role"] == "support")

    summary = {
        "annotation_pack": pack_dir.name,
        "complete_annotation_sample_count": len(complete_rows),
        "overlap_source_sample_count": len(overlap_source_ids),
        "overlap_nearest_sample_count": len(overlap_nearest_ids),
        "region_experiment_row_count": len(region_rows),
        "support_source_row_count": support_source_rows,
        "support_nearest_control_row_count": support_nearest_rows,
        "overlap_source_sample_ids": overlap_source_ids,
        "overlap_nearest_sample_ids": overlap_nearest_ids,
        "source_modality_csvs": [str(path) for path in source_csvs],
        "nearest_control_csvs": [str(path) for path in nearest_csvs],
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        f"# Region Experiment Summary: {pack_dir.name}",
        "",
        f"- complete annotation sample ids: `{len(complete_rows)}`",
        f"- overlap source sample ids: `{len(overlap_source_ids)}`",
        f"- overlap nearest-control sample ids: `{len(overlap_nearest_ids)}`",
        f"- region experiment rows: `{len(region_rows)}`",
        f"- support source rows: `{support_source_rows}`",
        f"- support nearest-control rows: `{support_nearest_rows}`",
        "",
        "## Overlap source sample ids",
        "",
    ]
    for sample_id in overlap_source_ids:
        md_lines.append(f"- `{sample_id}`")
    md_lines.extend(["", "## Overlap nearest-control sample ids", ""])
    for sample_id in overlap_nearest_ids:
        md_lines.append(f"- `{sample_id}`")
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[done] annotation_pack={pack_dir.name}")
    print(f"[done] complete_annotation_sample_count={len(complete_rows)}")
    print(f"[done] overlap_source_sample_count={len(overlap_source_ids)}")
    print(f"[done] overlap_nearest_sample_count={len(overlap_nearest_ids)}")
    print(f"[done] region_experiment_rows={len(region_rows)}")
    print(f"[done] out_csv={out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
