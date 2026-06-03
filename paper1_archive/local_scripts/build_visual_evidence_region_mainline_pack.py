#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "annotation" / "okvqa_evidence_labelme_round4_mainline16"

STRONG_SELECTED_CSVS = [
    ROOT / "doc" / "5.13" / "bd_visual_only_mech_pack" / "visual_positive_strong12_selected.csv",
    ROOT / "doc" / "5.13" / "bd_visual_only_mech_pack" / "visual_positive_strong18_next_selected.csv",
]

IMAGE_MANIFEST_CSVS = [
    ROOT / "annotation" / "okvqa_type_label_round4_400_mobile_package" / "manifest.csv",
    ROOT / "annotation" / "okvqa_type_label_round3_320" / "manifest.csv",
]

OLD_MASK_MANIFESTS = [
    ROOT / "annotation" / "okvqa_evidence_labelme_round2" / "manifest.csv",
    ROOT / "annotation" / "okvqa_evidence_labelme_round3_core" / "manifest.csv",
]

SOURCE_MODALITY_CSVS = [
    ROOT / "remote_sync" / "2026-05-14_bd_visual_positive_strong12_main64" / "modality_pilot_visual_positive_strong12_clean_core7_generic6_nobuf_full7_cap0_relpos.csv",
    ROOT / "remote_sync" / "2026-05-14_bd_visual_positive_strong18_next_main64" / "modality_pilot_visual_positive_strong18_next_clean_core10_generic8_nobuf_full10_cap0_relpos.csv",
]

NEAREST_CONTROL_CSVS = [
    ROOT / "remote_sync" / "2026-05-14_bd_visual_positive_strong12_main64" / "modality_matched_control_visual_positive_strong12_clean_core7_generic6_nobuf_full7_cap0_relpos.csv",
    ROOT / "remote_sync" / "2026-05-14_bd_visual_positive_strong18_next_main64" / "modality_matched_control_visual_positive_strong18_next_clean_core10_generic8_nobuf_full10_cap0_relpos.csv",
]

FIXED_PRIORITY_SAMPLE_IDS = [
    "okvqa_val_2847255",
    "okvqa_val_4157235",
    "okvqa_val_3605295",
    "okvqa_val_5176195",
    "okvqa_val_340155",
    "okvqa_val_3658865",
    "okvqa_val_299845",
    "okvqa_val_4739195",
    "okvqa_val_5334645",
    "okvqa_val_02444",
    "okvqa_val_00310",
]

RESERVE_BACKFILL_SAMPLE_IDS = [
    "okvqa_val_4043385",
    "okvqa_val_4502065",
    "okvqa_val_1593205",
    "okvqa_val_1058855",
]

TARGET_SAMPLE_COUNT = 16

RUN_TO_PROMPT = {
    "A": "D_visual_only",
    "B": "B_direct",
}

REASONING_ORDER = {
    "symbol_text_reading": 0,
    "visual_readout": 1,
    "scene_inference": 2,
}

PRIORITY_ORDER = {
    "high": 0,
    "medium": 1,
    "new": 2,
    "low": 3,
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bool_str(value: bool) -> str:
    return "True" if value else "False"


def _load_selected_metadata() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for path in STRONG_SELECTED_CSVS:
        for row in _read_csv(path):
            sample_id = (row.get("sample_id") or "").strip()
            if sample_id:
                out[sample_id] = row
    return out


def _load_image_metadata() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for path in IMAGE_MANIFEST_CSVS:
        for row in _read_csv(path):
            sample_id = (row.get("sample_id") or "").strip()
            if sample_id and sample_id not in out:
                row = dict(row)
                if not row.get("local_image_path"):
                    image_url = (row.get("image_url") or "").strip()
                    if image_url:
                        row["local_image_path"] = str(path.parent / image_url)
                out[sample_id] = row
    return out


def _load_existing_mask_status() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in OLD_MASK_MANIFESTS:
        round_name = path.parent.name
        for row in _read_csv(path):
            sample_id = (row.get("sample_id") or "").strip()
            if sample_id and sample_id not in out:
                out[sample_id] = round_name
    return out


def _load_clean_core_source_rows() -> dict[tuple[str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in SOURCE_MODALITY_CSVS:
        for row in _read_csv(path):
            if row.get("condition") != "clean":
                continue
            sample_id = (row.get("sample_id") or "").strip()
            run = (row.get("run") or "").strip()
            node_role = (row.get("node_role") or "").strip()
            key = (sample_id, run, node_role)
            if sample_id and run and node_role and key not in out:
                out[key] = row
    return out


def _load_clean_core_nearest_rows() -> dict[tuple[str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in NEAREST_CONTROL_CSVS:
        for row in _read_csv(path):
            if row.get("condition") != "clean":
                continue
            sample_id = (row.get("sample_id") or "").strip()
            run = (row.get("run") or "").strip()
            node_role = (row.get("node_role") or "").strip()
            key = (sample_id, run, node_role)
            if sample_id and run and node_role and key not in out:
                out[key] = row
    return out


def _sort_key(sample_id: str, selected_meta: dict[str, dict[str, str]], source_ids: set[str], nearest_ids: set[str]) -> tuple[int, int, int, str]:
    row = selected_meta[sample_id]
    return (
        REASONING_ORDER.get((row.get("reasoning_operation") or "").strip(), 99),
        0 if sample_id in nearest_ids else 1,
        PRIORITY_ORDER.get((row.get("priority") or "").strip(), 99),
        sample_id,
    )


def _choose_sample_ids(
    selected_meta: dict[str, dict[str, str]],
    source_rows: dict[tuple[str, str, str], dict[str, str]],
    nearest_rows: dict[tuple[str, str, str], dict[str, str]],
) -> list[str]:
    source_ids = {sample_id for sample_id, _, _ in source_rows.keys()}
    nearest_ids = {sample_id for sample_id, _, _ in nearest_rows.keys()}

    chosen = list(FIXED_PRIORITY_SAMPLE_IDS)

    extra_clean_core_ids = [
        sample_id
        for sample_id in sorted(source_ids)
        if sample_id not in chosen and sample_id in selected_meta
    ]
    extra_clean_core_ids.sort(key=lambda sid: _sort_key(sid, selected_meta, source_ids, nearest_ids))
    for sample_id in extra_clean_core_ids:
        if len(chosen) >= TARGET_SAMPLE_COUNT:
            break
        chosen.append(sample_id)

    if len(chosen) < TARGET_SAMPLE_COUNT:
        for sample_id in RESERVE_BACKFILL_SAMPLE_IDS:
            if sample_id in selected_meta and sample_id not in chosen:
                chosen.append(sample_id)
                if len(chosen) >= TARGET_SAMPLE_COUNT:
                    break

    if len(chosen) < TARGET_SAMPLE_COUNT:
        reserve_pool = [
            sample_id
            for sample_id, row in selected_meta.items()
            if sample_id not in chosen
            and (row.get("image_dependence") or "").strip() == "strong"
            and (row.get("reasoning_operation") or "").strip() != "entity_linking"
        ]
        reserve_pool.sort(key=lambda sid: _sort_key(sid, selected_meta, source_ids, nearest_ids))
        for sample_id in reserve_pool:
            if len(chosen) >= TARGET_SAMPLE_COUNT:
                break
            chosen.append(sample_id)

    if len(chosen) != TARGET_SAMPLE_COUNT:
        raise ValueError(f"expected {TARGET_SAMPLE_COUNT} sample ids, got {len(chosen)}")
    return chosen


def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def _build_pack_manifest_rows(
    chosen_ids: list[str],
    selected_meta: dict[str, dict[str, str]],
    image_meta: dict[str, dict[str, str]],
    existing_mask_status: dict[str, str],
    source_rows: dict[tuple[str, str, str], dict[str, str]],
    nearest_rows: dict[tuple[str, str, str], dict[str, str]],
) -> list[dict[str, str]]:
    source_ids = {sample_id for sample_id, _, _ in source_rows.keys()}
    nearest_ids = {sample_id for sample_id, _, _ in nearest_rows.keys()}
    out: list[dict[str, str]] = []

    for idx, sample_id in enumerate(chosen_ids, start=1):
        selected = selected_meta[sample_id]
        image = image_meta[sample_id]
        image_filename = image.get("image_filename") or Path(image.get("local_image_path", "")).name
        local_image_path = OUT_DIR / "images" / image_filename
        _copy_image(Path(image["local_image_path"]), local_image_path)

        selection_tier = "tier1_main_figures" if idx <= 6 else "tier2_mainline_anchors" if idx <= 11 else "tier3_fill_or_reserve"
        existing_round = existing_mask_status.get(sample_id, "")
        out.append(
            {
                "selection_rank": str(idx),
                "selection_tier": selection_tier,
                "sample_id": sample_id,
                "question_text": selected.get("question_text", ""),
                "answer_text": selected.get("answer_text", ""),
                "reasoning_operation": selected.get("reasoning_operation", ""),
                "visual_structure": selected.get("visual_structure", ""),
                "image_dependence": selected.get("image_dependence", ""),
                "priority": selected.get("priority", ""),
                "image_filename": image_filename,
                "local_image_path": str(local_image_path),
                "source_local_image_path": image.get("local_image_path", ""),
                "remote_image_path": image.get("remote_image_path", ""),
                "has_existing_mask": _bool_str(bool(existing_round)),
                "existing_mask_round": existing_round,
                "has_clean_core_source": _bool_str(sample_id in source_ids),
                "has_nearest_control": _bool_str(sample_id in nearest_ids),
                "annotation_goal": "Mark the smallest direct answer-bearing visual region as 'answer', and the broader supporting evidence or key context as 'relate'.",
                "analysis_status": "primary_pool" if sample_id in source_ids else "reserve_or_regression",
                "notes": "Reused old mask if available; otherwise new mainline annotation target.",
            }
        )
    return out


def _build_region_experiment_rows(
    chosen_ids: list[str],
    pack_rows: list[dict[str, str]],
    source_rows: dict[tuple[str, str, str], dict[str, str]],
    nearest_rows: dict[tuple[str, str, str], dict[str, str]],
) -> list[dict[str, str]]:
    pack_by_id = {row["sample_id"]: row for row in pack_rows}
    out: list[dict[str, str]] = []

    def add_source_row(sample_id: str, run: str, node_role: str, row: dict[str, str]) -> None:
        pack = pack_by_id[sample_id]
        out.append(
            {
                "sample_id": sample_id,
                "run": run,
                "prompt_name": RUN_TO_PROMPT.get(run, run),
                "node_role": node_role,
                "node_source": "source",
                "question": row.get("question", pack["question_text"]),
                "image_path": pack["local_image_path"],
                "remote_image_path": pack["remote_image_path"],
                "feature_layer": row.get("feature_layer", ""),
                "feature_pos": row.get("feature_pos", ""),
                "feature_id": row.get("feature_id", ""),
                "target_token_id": row.get("target_token_id", ""),
                "reasoning_operation": pack["reasoning_operation"],
                "visual_structure": pack["visual_structure"],
                "image_dependence": pack["image_dependence"],
                "selection_tier": pack["selection_tier"],
                "has_existing_mask": pack["has_existing_mask"],
                "existing_mask_round": pack["existing_mask_round"],
                "reference_clean_delta_target_logit": row.get("reference_clean_delta_target_logit", row.get("delta_target_logit", "")),
                "position_alignment": row.get("position_alignment", ""),
                "clean_seq_len": row.get("clean_seq_len", ""),
                "source_bucket": row.get("bucket", ""),
                "match_mode": "",
                "sampled_match_label": "",
                "control_draw_idx": "",
            }
        )

    def add_nearest_row(sample_id: str, run: str, node_role: str, row: dict[str, str]) -> None:
        pack = pack_by_id[sample_id]
        out.append(
            {
                "sample_id": sample_id,
                "run": run,
                "prompt_name": RUN_TO_PROMPT.get(run, run),
                "node_role": node_role,
                "node_source": "nearest_control",
                "question": row.get("question", pack["question_text"]),
                "image_path": pack["local_image_path"],
                "remote_image_path": pack["remote_image_path"],
                "feature_layer": row.get("control_feature_layer", ""),
                "feature_pos": row.get("control_feature_pos", ""),
                "feature_id": row.get("control_feature_id", ""),
                "target_token_id": row.get("target_token_id", ""),
                "reasoning_operation": pack["reasoning_operation"],
                "visual_structure": pack["visual_structure"],
                "image_dependence": pack["image_dependence"],
                "selection_tier": pack["selection_tier"],
                "has_existing_mask": pack["has_existing_mask"],
                "existing_mask_round": pack["existing_mask_round"],
                "reference_clean_delta_target_logit": row.get("delta_target_logit", ""),
                "position_alignment": row.get("control_position_alignment", ""),
                "clean_seq_len": row.get("clean_seq_len", ""),
                "source_bucket": row.get("bucket", ""),
                "match_mode": row.get("match_mode", ""),
                "sampled_match_label": row.get("sampled_match_label", ""),
                "control_draw_idx": row.get("control_draw_idx", ""),
            }
        )

    for sample_id in chosen_ids:
        for run in ("A", "B"):
            for node_role in ("support", "suppressor"):
                key = (sample_id, run, node_role)
                if key in source_rows:
                    add_source_row(sample_id, run, node_role, source_rows[key])
                if key in nearest_rows:
                    add_nearest_row(sample_id, run, node_role, nearest_rows[key])
    return out


def _write_pack_summary(
    chosen_ids: list[str],
    pack_rows: list[dict[str, str]],
    region_rows: list[dict[str, str]],
) -> None:
    reasoning_counts = Counter(row["reasoning_operation"] for row in pack_rows)
    primary_count = sum(1 for row in pack_rows if row["analysis_status"] == "primary_pool")
    nearest_count = sum(1 for row in pack_rows if row["has_nearest_control"] == "True")
    existing_mask_count = sum(1 for row in pack_rows if row["has_existing_mask"] == "True")
    support_source_rows = sum(1 for row in region_rows if row["node_source"] == "source" and row["node_role"] == "support")
    support_nearest_rows = sum(1 for row in region_rows if row["node_source"] == "nearest_control" and row["node_role"] == "support")

    summary = {
        "target_sample_count": TARGET_SAMPLE_COUNT,
        "selected_sample_count": len(chosen_ids),
        "primary_pool_sample_count": primary_count,
        "nearest_control_sample_count": nearest_count,
        "existing_mask_sample_count": existing_mask_count,
        "region_experiment_row_count": len(region_rows),
        "support_source_row_count": support_source_rows,
        "support_nearest_control_row_count": support_nearest_rows,
        "reasoning_operation_counts": dict(reasoning_counts),
        "selected_sample_ids": chosen_ids,
    }
    (OUT_DIR / "selection_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# Visual-Evidence Mainline16 Pack Summary",
        "",
        f"- selected sample ids: `{len(chosen_ids)}`",
        f"- primary-pool sample ids with clean-core source rows: `{primary_count}`",
        f"- sample ids with nearest-control coverage: `{nearest_count}`",
        f"- sample ids with existing old masks: `{existing_mask_count}`",
        f"- region experiment manifest rows: `{len(region_rows)}`",
        f"- support source rows: `{support_source_rows}`",
        f"- support nearest-control rows: `{support_nearest_rows}`",
        "",
        "## Reasoning counts",
        "",
    ]
    for name, count in sorted(reasoning_counts.items(), key=lambda kv: (REASONING_ORDER.get(kv[0], 99), kv[0])):
        md_lines.append(f"- `{name}`: `{count}`")
    md_lines.extend(
        [
            "",
            "## Selected sample ids",
            "",
        ]
    )
    for row in pack_rows:
        md_lines.append(
            f"- `{row['selection_rank']}` | `{row['sample_id']}` | `{row['reasoning_operation']}` | "
            f"`source={row['has_clean_core_source']}` | `nearest={row['has_nearest_control']}` | "
            f"`mask={row['existing_mask_round'] or 'new'}`"
        )
    (OUT_DIR / "selection_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> int:
    selected_meta = _load_selected_metadata()
    image_meta = _load_image_metadata()
    existing_mask_status = _load_existing_mask_status()
    source_rows = _load_clean_core_source_rows()
    nearest_rows = _load_clean_core_nearest_rows()

    chosen_ids = _choose_sample_ids(selected_meta, source_rows, nearest_rows)
    missing_selected = [sample_id for sample_id in chosen_ids if sample_id not in selected_meta]
    missing_image = [sample_id for sample_id in chosen_ids if sample_id not in image_meta]
    if missing_selected:
        raise ValueError(f"missing selected metadata for sample ids: {missing_selected}")
    if missing_image:
        raise ValueError(f"missing image metadata for sample ids: {missing_image}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pack_rows = _build_pack_manifest_rows(chosen_ids, selected_meta, image_meta, existing_mask_status, source_rows, nearest_rows)
    pack_fieldnames = list(pack_rows[0].keys())
    _write_csv(OUT_DIR / "manifest.csv", pack_rows, pack_fieldnames)
    (OUT_DIR / "manifest.json").write_text(json.dumps(pack_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    region_rows = _build_region_experiment_rows(chosen_ids, pack_rows, source_rows, nearest_rows)
    region_fieldnames = list(region_rows[0].keys())
    _write_csv(OUT_DIR / "region_experiment_manifest.csv", region_rows, region_fieldnames)

    _write_pack_summary(chosen_ids, pack_rows, region_rows)
    print(f"[done] out_dir={OUT_DIR}")
    print(f"[done] selected_sample_count={len(chosen_ids)}")
    print(f"[done] region_experiment_rows={len(region_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
