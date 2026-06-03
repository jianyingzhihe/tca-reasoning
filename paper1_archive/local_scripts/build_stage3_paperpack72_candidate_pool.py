#!/usr/bin/env python3
from __future__ import annotations

import csv
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
STAGE3 = ROOT / "doc" / "experiments" / "stage3"
CROSS = STAGE3 / "cross_model"
OUT_DIR = STAGE3 / "paperpack72"
ANNOTATION = ROOT / "annotation"

QUOTAS = {
    "symbol_text_reading": 24,
    "visual_readout": 24,
    "compact_scene_inference": 12,
    "mixed_localized": 12,
}
PRIMARY_SEED_SIZE = 72
SELECTION_POLICY = "visual_tier_first_primary_localized"

VISUAL_TIERS = {
    "localized": {"include": True, "priority": 0, "role": "primary"},
    "multi_region": {"include": True, "priority": 1, "role": "extension"},
    "unlabeled": {"include": True, "priority": 2, "role": "prescreen"},
    "diffuse_global": {"include": False, "priority": 9, "role": "exclude_or_boundary"},
}

SOURCE_FILES = [
    ANNOTATION / "okvqa_type_label_round4_400" / "manifest.csv",
    ANNOTATION / "okvqa_type_label_round3_320" / "manifest.csv",
    ANNOTATION / "okvqa_type_label_round2_new32" / "manifest.csv",
    ANNOTATION / "okvqa_type_label_round_80_combined" / "analysis_stage1_type_labels_80" / "type_labeled_manifest.csv",
]

LABEL_FILES = [
    ANNOTATION
    / "okvqa_type_label_round3_320_static_splits"
    / "okvqa_type_label_round3_320_part1_labels.csv",
    ANNOTATION
    / "okvqa_type_label_round3_320_static_splits"
    / "okvqa_type_label_round3_320_part3_labels.csv",
    ANNOTATION / "okvqa_type_label_round_80_combined" / "labels.csv",
    ANNOTATION / "okvqa_type_label_round1" / "labels.csv",
    ANNOTATION / "okvqa_type_label_round2_new32" / "labels.csv",
]

USED_GLOBS = [
    ROOT / "doc" / "experiments" / "stage2" / "cross_model" / "*.csv",
    ROOT / "doc" / "experiments" / "stage3" / "cross_model" / "*.csv",
    ANNOTATION / "okvqa_evidence_labelme_*" / "manifest*.csv",
    ANNOTATION / "stage2a_region_replication_top24_nearest8" / "region_experiment_manifest*.csv",
]

OUT_CANDIDATES = OUT_DIR / "paperpack72_candidate_pool.csv"
OUT_RESERVE = OUT_DIR / "paperpack72_reserve_candidates.csv"
OUT_SHORTFALL = OUT_DIR / "paperpack72_quota_shortfall.csv"
OUT_MANIFEST_TEMPLATE = OUT_DIR / "paperpack72_manifest_template.csv"
OUT_EXCLUSIONS = OUT_DIR / "paperpack72_exclusion_manifest.csv"
OUT_REPORT = OUT_DIR / "paperpack72_candidate_pool_report.json"
OUT_ANNOTATOR_A = OUT_DIR / "annotator_a_assignment.csv"
OUT_ANNOTATOR_B = OUT_DIR / "annotator_b_assignment.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def sample_sort_key(sample_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", sample_id or "")
    return (int(match.group(1)) if match else 10**12, sample_id)


def normalize_sample_id(row: dict[str, str]) -> str:
    sample_id = (row.get("sample_id") or "").strip()
    if sample_id:
        return sample_id
    item_id = row.get("item_id", "")
    match = re.search(r"okvqa_val_\d+", item_id)
    return match.group(0) if match else ""


def load_used_ids() -> tuple[set[str], dict[str, set[str]]]:
    used: set[str] = set()
    by_source: dict[str, set[str]] = defaultdict(set)
    for pattern in USED_GLOBS:
        # `Path.glob()` does not expand wildcard components in `pattern.parent`;
        # use glob over the full pattern so old labelme packs are excluded too.
        for path_str in glob.glob(str(pattern)):
            path = Path(path_str)
            for row in read_csv(path):
                sample_id = normalize_sample_id(row)
                if not sample_id:
                    continue
                used.add(sample_id)
                by_source[path.name].add(sample_id)
    return used, by_source


def load_visual_labels() -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    for path in LABEL_FILES:
        for row in read_csv(path):
            sample_id = normalize_sample_id(row)
            if not sample_id:
                item_id = row.get("item_id", "")
                match = re.search(r"okvqa_val_\d+", item_id)
                sample_id = match.group(0) if match else ""
            if not sample_id:
                continue
            labels.setdefault(sample_id, {}).update(
                {
                    "human_visual_type_label": row.get("visual_type_label", ""),
                    "human_knowledge_level_label": row.get("knowledge_level_label", ""),
                    "human_label_notes": row.get("label_notes", ""),
                    "human_label_source": path.name,
                }
            )
    return labels


def infer_reasoning_operation(row: dict[str, str]) -> tuple[str, str]:
    question = (row.get("question_text") or row.get("display_question") or row.get("raw_question") or "").lower()
    answer = (row.get("answer_text") or "").lower()
    text_markers = [
        "sign",
        "written",
        "word",
        "letter",
        "language",
        "logo",
        "brand",
        "symbol",
        "number",
        "label",
        "flag",
        "say",
        "mean",
    ]
    scene_markers = [
        "why",
        "where",
        "what city",
        "what country",
        "what service",
        "what type of",
        "what kind of",
        "purpose",
        "used for",
        "made of",
    ]
    visual_markers = [
        "color",
        "animal",
        "sport",
        "object",
        "shown",
        "picture",
        "image",
        "wearing",
        "holding",
        "shape",
    ]
    if any(marker in question for marker in text_markers):
        return "symbol_text_reading", "heuristic_question_text_marker"
    if any(marker in question for marker in scene_markers):
        return "compact_scene_inference", "heuristic_scene_question_marker"
    if any(marker in question for marker in visual_markers) or len(answer.split()) <= 2:
        return "visual_readout", "heuristic_visual_or_short_answer"
    return "mixed_localized", "heuristic_fallback_mixed"


def infer_visual_tier(row: dict[str, str]) -> tuple[str, str]:
    human_visual = (row.get("human_visual_type_label") or "").lower().strip()
    legacy_visual = (row.get("legacy_visual_type_label") or "").lower().strip()
    visual = (
        human_visual
        or legacy_visual
        or row.get("visual_structure")
        or row.get("type_label")
        or ""
    ).lower().strip()
    if human_visual == "localized" or legacy_visual == "localized":
        return "localized", "historical_localized_label"
    if human_visual == "multi_region" or legacy_visual == "multi_region":
        return "multi_region", "historical_multi_region_label"
    if "diffuse" in visual or "global" in visual:
        return "diffuse_global", "historical_diffuse_global_label"
    if not visual:
        return "unlabeled", "missing_visual_tier_label"
    if "diffuse" in visual or "global" in visual:
        return "diffuse_global", "visual_label_diffuse_or_global"
    if "localized" in visual or "single" in visual or "compact" in visual:
        return "localized", "visual_label_localized"
    if "multi_region" in visual:
        return "multi_region", "visual_label_multi_region_needs_screening"
    return "unlabeled", "unrecognized_visual_tier_needs_prescreening"


def row_priority(row: dict[str, str]) -> tuple[int, int, str]:
    priority_score = {"high": 0, "medium": 1, "low": 2}.get((row.get("priority") or "").lower(), 3)
    raw_tier_priority = row.get("visual_tier_priority")
    tier_priority = 9 if raw_tier_priority in {"", None} else int(raw_tier_priority)
    return (tier_priority, priority_score, row.get("sample_id", ""))


def merge_source_rows() -> list[dict[str, str]]:
    labels = load_visual_labels()
    rows_by_sample: dict[str, dict[str, str]] = {}
    for source_path in SOURCE_FILES:
        for row in read_csv(source_path):
            sample_id = normalize_sample_id(row)
            if not sample_id:
                continue
            merged = dict(row)
            merged["sample_id"] = sample_id
            merged["candidate_source"] = source_path.name
            merged.update(labels.get(sample_id, {}))
            old = rows_by_sample.get(sample_id)
            if old is None or row_priority(merged) < row_priority(old):
                rows_by_sample[sample_id] = merged
    return list(rows_by_sample.values())


def build() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    used_ids, used_by_source = load_used_ids()
    source_rows = merge_source_rows()
    candidates_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exclusions: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []

    for row in source_rows:
        sample_id = row.get("sample_id", "")
        visual_tier, localization_reason = infer_visual_tier(row)
        tier_info = VISUAL_TIERS.get(visual_tier, VISUAL_TIERS["unlabeled"])
        proposed_type, type_reason = infer_reasoning_operation(row)
        base = {
            "sample_id": sample_id,
            "image_filename": row.get("image_filename", ""),
            "local_image_path": row.get("local_image_path", ""),
            "image_url": row.get("image_url", ""),
            "question_text": row.get("question_text") or row.get("display_question") or row.get("raw_question") or "",
            "answer_text": row.get("answer_text", ""),
            "proposed_reasoning_operation": proposed_type,
            "proposed_type_reason": type_reason,
            "priority": row.get("priority", ""),
            "source_round": row.get("source_round", ""),
            "candidate_source": row.get("candidate_source", ""),
            "legacy_visual_type_label": row.get("legacy_visual_type_label", ""),
            "human_visual_type_label": row.get("human_visual_type_label", ""),
            "human_knowledge_level_label": row.get("human_knowledge_level_label", ""),
            "visual_tier": visual_tier,
            "visual_tier_role": tier_info["role"],
            "visual_tier_priority": tier_info["priority"],
            "localization_reason": localization_reason,
            "paperpack_status": "candidate_for_single_annotator_reliability",
            "annotation_required": "single_pass1_answer_mask;single_pass1_relate_mask;single_pass2_reliability_answer_mask;single_pass2_reliability_relate_mask;adjudicated_answer_mask;adjudicated_relate_mask;union_mask",
        }
        if sample_id in used_ids:
            exclusions.append({**base, "exclusion_reason": "sample_id_already_used_in_stage2_or_stage3"})
            continue
        if not tier_info["include"]:
            exclusions.append({**base, "exclusion_reason": localization_reason})
            continue
        candidates_by_type[proposed_type].append(base)
        eligible.append(base)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    shortfall_rows: list[dict[str, Any]] = []
    ranked_eligible = sorted(eligible, key=lambda item: (row_priority(item), sample_sort_key(item["sample_id"])))
    for item in ranked_eligible[:PRIMARY_SEED_SIZE]:
        selected_ids.add(item["sample_id"])
        type_name = item["proposed_reasoning_operation"]
        selected.append(
            {
                **item,
                "paperpack72_rank": len(selected) + 1,
                "quota_slot": type_name,
                "quota_target": QUOTAS.get(type_name, ""),
                "selection_note": f"{SELECTION_POLICY}_{item['visual_tier']}_selected_pending_single_annotation",
            }
        )

    selected_type_counts = Counter(row["proposed_reasoning_operation"] for row in selected)
    for type_name, quota in QUOTAS.items():
        observed = selected_type_counts.get(type_name, 0)
        if observed < quota:
            shortfall_rows.append(
                {
                    "quota_slot": type_name,
                    "quota_target": quota,
                    "strict_available": len(candidates_by_type[type_name]),
                    "shortfall": quota - observed,
                    "resolution": "type_quota_not_enforced_visual_tier_priority_primary_report_slice_as_observed",
                }
            )

    manifest_template = []
    for row in selected:
        manifest_template.append(
            {
                **row,
                "single_pass1_answer_mask_path": "",
                "single_pass1_relate_mask_path": "",
                "single_pass2_reliability_answer_mask_path": "",
                "single_pass2_reliability_relate_mask_path": "",
                "adjudicated_answer_mask_path": "",
                "adjudicated_relate_mask_path": "",
                "adjudicated_union_mask_path": "",
                "shifted_mask_path": "",
                "shuffled_mask_path": "",
                "random16_mask_dir": "",
                "answer_iou_pass1_pass2": "",
                "relate_iou_pass1_pass2": "",
                "answer_dice_pass1_pass2": "",
                "relate_dice_pass1_pass2": "",
                "compactness_label": "",
                "image_dependence_final": "",
                "final_reasoning_operation": "",
                "final_include": "",
                "adjudication_notes": "",
            }
        )

    candidate_rows = []
    for item in sorted(eligible, key=lambda row: (row_priority(row), sample_sort_key(row["sample_id"]))):
        is_selected = item["sample_id"] in selected_ids
        selected_row = next((row for row in selected if row["sample_id"] == item["sample_id"]), {})
        candidate_rows.append(
            {
                **item,
                "paperpack72_rank": selected_row.get("paperpack72_rank", ""),
                "quota_slot": selected_row.get("quota_slot", ""),
                "quota_target": selected_row.get("quota_target", ""),
                "selection_note": selected_row.get("selection_note", "eligible_not_selected_reserve_candidate"),
                "selected_for_annotation_seed": is_selected,
            }
        )

    selected_counts = Counter(row["proposed_reasoning_operation"] for row in selected)
    selected_quota_slots = Counter(row["quota_slot"] for row in selected)
    available_counts = {key: len(value) for key, value in candidates_by_type.items()}
    reserve_rows = [
        {
            **item,
            "paperpack72_rank": "",
            "quota_slot": item["proposed_reasoning_operation"],
            "quota_target": QUOTAS.get(item["proposed_reasoning_operation"], ""),
            "selection_note": "reserve_candidate_not_in_primary72",
        }
        for item in ranked_eligible
        if item["sample_id"] not in selected_ids
    ]
    report = {
        "status": "candidate_pool_built",
        "note": "This is a candidate pool and 72-row single-annotator annotation seed, not a completed paperpack72 manifest.",
        "source_files": [str(path) for path in SOURCE_FILES if path.exists()],
        "used_id_count": len(used_ids),
        "source_candidate_count": len(source_rows),
        "eligible_candidate_count": len(eligible),
        "primary_seed_size": PRIMARY_SEED_SIZE,
        "selection_policy": SELECTION_POLICY,
        "selected_annotation_seed_count": len(selected),
        "selected_annotation_seed_by_inferred_type": dict(selected_counts),
        "selected_annotation_seed_by_quota_slot": dict(selected_quota_slots),
        "quota": QUOTAS,
        "visual_tier_policy": VISUAL_TIERS,
        "available_after_exclusion_by_type": available_counts,
        "selected_annotation_seed_by_visual_tier": dict(Counter(row["visual_tier"] for row in selected)),
        "reserve_candidate_count": len(reserve_rows),
        "reserve_candidate_by_visual_tier": dict(Counter(row["visual_tier"] for row in reserve_rows)),
        "quota_shortfalls_before_backfill": shortfall_rows,
        "exclusion_count": len(exclusions),
        "used_id_sources": {key: len(value) for key, value in sorted(used_by_source.items())},
        "blocking_next_step": "Single pass-1 annotation, delayed pass-2 reliability annotation, and adjudication are required before paperpack72 experiments can run.",
    }
    return candidate_rows, manifest_template, exclusions, reserve_rows, shortfall_rows, report


def main() -> None:
    candidate_rows, manifest_template, exclusions, reserve_rows, shortfall_rows, report = build()
    fields = [
        "paperpack72_rank",
        "sample_id",
        "image_filename",
        "local_image_path",
        "image_url",
        "question_text",
        "answer_text",
        "proposed_reasoning_operation",
        "proposed_type_reason",
        "priority",
        "source_round",
        "candidate_source",
        "legacy_visual_type_label",
        "human_visual_type_label",
        "human_knowledge_level_label",
        "visual_tier",
        "visual_tier_role",
        "localization_reason",
        "paperpack_status",
        "annotation_required",
        "quota_slot",
        "quota_target",
        "selection_note",
    ]
    write_csv(
        OUT_CANDIDATES,
        candidate_rows,
        fields + ["selected_for_annotation_seed"],
    )
    write_csv(
        OUT_RESERVE,
        reserve_rows,
        fields,
    )
    write_csv(
        OUT_MANIFEST_TEMPLATE,
        manifest_template,
        fields
        + [
            "single_pass1_answer_mask_path",
            "single_pass1_relate_mask_path",
            "single_pass2_reliability_answer_mask_path",
            "single_pass2_reliability_relate_mask_path",
            "adjudicated_answer_mask_path",
            "adjudicated_relate_mask_path",
            "adjudicated_union_mask_path",
            "shifted_mask_path",
            "shuffled_mask_path",
            "random16_mask_dir",
            "answer_iou_pass1_pass2",
            "relate_iou_pass1_pass2",
            "answer_dice_pass1_pass2",
            "relate_dice_pass1_pass2",
            "compactness_label",
            "image_dependence_final",
            "final_reasoning_operation",
            "final_include",
            "adjudication_notes",
        ],
    )
    assignment_rows = [row for row in manifest_template]
    write_csv(OUT_ANNOTATOR_A, assignment_rows, fields)
    write_csv(OUT_ANNOTATOR_B, assignment_rows, fields)
    write_csv(
        OUT_SHORTFALL,
        shortfall_rows,
        ["quota_slot", "quota_target", "strict_available", "shortfall", "resolution"],
    )
    write_csv(
        OUT_EXCLUSIONS,
        exclusions,
        [
            "sample_id",
            "image_filename",
            "question_text",
            "answer_text",
            "proposed_reasoning_operation",
            "priority",
            "candidate_source",
            "legacy_visual_type_label",
            "human_visual_type_label",
            "localization_reason",
            "exclusion_reason",
        ],
    )
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
