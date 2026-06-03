#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[2]

MASK_PACKS = [
    ("core24_easy", ROOT / "annotation" / "okvqa_evidence_labelme_round4_core24_easy", 100),
    ("core16_extra", ROOT / "annotation" / "okvqa_evidence_labelme_round4_core16_extra", 90),
    ("ultraeasy16_fresh", ROOT / "annotation" / "okvqa_evidence_labelme_round4_ultraeasy16_fresh", 85),
    ("round5_expanded20_route7", ROOT / "annotation" / "okvqa_evidence_labelme_round5_expanded20_route7", 80),
    ("stage2a_nearest8", ROOT / "annotation" / "stage2a_region_replication_top24_nearest8", 75),
]

EXTRA_METADATA_TABLES = [
    ROOT / "doc" / "experiments" / "stage2" / "stage2a_candidate_selection" / "stage2a_candidate_samples.csv",
    ROOT / "doc" / "5.16" / "expanded_localized_discovery_screen20_readout_2026-05-18" / "annotation_candidate_sheet.csv",
    ROOT / "doc" / "5.16" / "core24_prefixfix_region_analysis_2026-05-19" / "current_backbone_replication_candidate_rows.csv",
]

PREVIOUS_STAGE2H = {
    "okvqa_val_2847255",
    "okvqa_val_4157235",
    "okvqa_val_3658865",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _first_nonempty(row: dict[str, str], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _image_name_from_row(row: dict[str, str]) -> str:
    value = _first_nonempty(row, ["image_filename", "image_name"])
    if value:
        return Path(value).name
    value = _first_nonempty(row, ["local_image_path", "source_local_image_path", "image_path"])
    return Path(value).name if value else ""


def _normalize_metadata_row(row: dict[str, str], source_table: str, pack_name: str = "") -> dict[str, str] | None:
    image_name = _image_name_from_row(row)
    sample_id = _first_nonempty(row, ["sample_id", "id"])
    if not image_name and not sample_id:
        return None
    return {
        "sample_id": sample_id,
        "image_filename": image_name,
        "question_text": _first_nonempty(row, ["question_text", "question"]),
        "answer_text": _first_nonempty(row, ["answer_text", "answer", "target_answer"]),
        "reasoning_operation": _first_nonempty(row, ["reasoning_operation"]),
        "visual_structure": _first_nonempty(row, ["visual_structure"]),
        "image_dependence": _first_nonempty(row, ["image_dependence"]),
        "legacy_visual_type_label": _first_nonempty(row, ["legacy_visual_type_label"]),
        "legacy_knowledge_level_label": _first_nonempty(row, ["legacy_knowledge_level_label"]),
        "compact_core_tier": _first_nonempty(row, ["compact_core_tier"]),
        "source_table": source_table,
        "metadata_pack": pack_name,
        "local_image_path": _first_nonempty(row, ["local_image_path", "source_local_image_path", "image_path"]),
        "remote_image_path": _first_nonempty(row, ["remote_image_path"]),
        "annotation_priority": _first_nonempty(row, ["annotation_priority", "priority"]),
        "support_source_available": _first_nonempty(row, ["support_source_available"]),
        "nearest_control_available": _first_nonempty(row, ["nearest_control_available"]),
        "source_effect": _first_nonempty(row, ["source_effect", "best_source_effect"]),
        "nearest_gap": _first_nonempty(row, ["nearest_gap", "best_nearest_gap"]),
        "random4_gap": _first_nonempty(row, ["random4_gap", "best_random4_gap"]),
        "notes": _first_nonempty(row, ["notes", "annotation_goal"]),
    }


def _metadata_index() -> tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]]]:
    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    tables: list[tuple[Path, str, str]] = []
    for pack_name, pack_root, _score in MASK_PACKS:
        tables.append((pack_root / "manifest.csv", f"{pack_name}/manifest.csv", pack_name))
    for table in EXTRA_METADATA_TABLES:
        tables.append((table, str(table.relative_to(ROOT)), ""))

    for path, source_table, pack_name in tables:
        for raw in _read_csv(path):
            row = _normalize_metadata_row(raw, source_table, pack_name)
            if row is None:
                continue
            if row["sample_id"]:
                by_sample[row["sample_id"]].append(row)
            if row["image_filename"]:
                by_image[row["image_filename"]].append(row)
    return by_sample, by_image


def _pick_metadata(
    *,
    image_name: str,
    pack_name: str,
    by_image: dict[str, list[dict[str, str]]],
) -> dict[str, str]:
    candidates = by_image.get(image_name, [])
    if not candidates:
        return {}
    same_pack = [row for row in candidates if row.get("metadata_pack") == pack_name]
    if same_pack:
        return _best_metadata(same_pack)
    return _best_metadata(candidates)


def _best_metadata(candidates: list[dict[str, str]]) -> dict[str, str]:
    def score(row: dict[str, str]) -> tuple[int, int, int]:
        return (
            int(bool(row.get("question_text"))),
            int(bool(row.get("answer_text"))),
            int(bool(row.get("reasoning_operation")) or bool(row.get("image_dependence"))),
        )

    return sorted(candidates, key=score, reverse=True)[0]


def _mask_area(mask: Image.Image) -> int:
    binary = mask.convert("L").point(lambda value: 255 if value > 0 else 0)
    return int(binary.histogram()[255])


def _find_image(pack_root: Path, image_name: str, metadata: dict[str, str]) -> Path | None:
    candidates = [
        pack_root / "images" / image_name,
        pack_root / image_name,
    ]
    local_image_path = metadata.get("local_image_path")
    if local_image_path:
        candidates.append(Path(local_image_path))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _float_or_none(value: str) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except Exception:
        return None


def _type_score(reasoning_operation: str, visual_structure: str, image_dependence: str, legacy_visual: str) -> int:
    score = 0
    if reasoning_operation == "symbol_text_reading":
        score += 35
    elif reasoning_operation == "visual_readout":
        score += 30
    elif reasoning_operation == "scene_inference":
        score += 15
    if visual_structure in {"single_core", "multi_core"}:
        score += 15
    if image_dependence == "strong":
        score += 20
    if legacy_visual == "localized":
        score += 10
    return score


def _compactness(answer_frac: float, union_frac: float, max_answer_frac: float, max_secondary_answer_frac: float) -> str:
    if answer_frac <= max_answer_frac:
        return "primary_localized"
    if answer_frac <= max_secondary_answer_frac and union_frac <= 0.85:
        return "large_localized_secondary"
    return "diffuse_or_fullscreen"


def _eligibility(row: dict[str, Any]) -> str:
    if not row.get("question_text") or not row.get("answer_text"):
        return "blocked_missing_question_or_answer"
    if row.get("compactness_label") == "primary_localized":
        return "primary_eligible"
    if row.get("compactness_label") == "large_localized_secondary":
        return "secondary_large_localized"
    return "excluded_diffuse_or_fullscreen"


def _build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    _by_sample, by_image = _metadata_index()
    rows: list[dict[str, Any]] = []
    seen_pack_image: set[tuple[str, str]] = set()
    for pack_name, pack_root, pack_score in MASK_PACKS:
        exported_root = pack_root / "exported_masks"
        if not exported_root.exists():
            continue
        for mask_dir in sorted(path for path in exported_root.iterdir() if path.is_dir()):
            image_name = f"{mask_dir.name}.jpg"
            if (pack_name, image_name) in seen_pack_image:
                continue
            seen_pack_image.add((pack_name, image_name))
            answer_path = mask_dir / "answer.png"
            relate_path = mask_dir / "relate.png"
            if not answer_path.exists():
                continue
            metadata = _pick_metadata(image_name=image_name, pack_name=pack_name, by_image=by_image)
            image_path = _find_image(pack_root, image_name, metadata)
            if image_path is None:
                continue
            with Image.open(image_path) as raw_image:
                image_size = raw_image.size
            answer = Image.open(answer_path).convert("L").resize(image_size)
            relate = Image.open(relate_path).convert("L").resize(image_size) if relate_path.exists() else None
            union = ImageChops.lighter(answer, relate) if relate is not None else answer.copy()
            total_area = image_size[0] * image_size[1]
            answer_area = _mask_area(answer)
            relate_area = _mask_area(relate) if relate is not None else 0
            union_area = _mask_area(union)
            answer_frac = answer_area / total_area if total_area else 0.0
            relate_frac = relate_area / total_area if total_area else 0.0
            union_frac = union_area / total_area if total_area else 0.0

            reasoning_operation = metadata.get("reasoning_operation", "")
            visual_structure = metadata.get("visual_structure", "")
            image_dependence = metadata.get("image_dependence", "")
            legacy_visual = metadata.get("legacy_visual_type_label", "")
            sample_id = metadata.get("sample_id", "")
            compactness = _compactness(
                answer_frac,
                union_frac,
                args.max_answer_frac,
                args.max_secondary_answer_frac,
            )
            row: dict[str, Any] = {
                "sample_id": sample_id,
                "image_filename": image_name,
                "question_text": metadata.get("question_text", ""),
                "answer_text": metadata.get("answer_text", ""),
                "reasoning_operation": reasoning_operation,
                "visual_structure": visual_structure,
                "image_dependence": image_dependence,
                "legacy_visual_type_label": legacy_visual,
                "legacy_knowledge_level_label": metadata.get("legacy_knowledge_level_label", ""),
                "compact_core_tier": metadata.get("compact_core_tier", ""),
                "mask_pack": pack_name,
                "mask_dir": str(mask_dir),
                "local_image_path": str(image_path),
                "width": image_size[0],
                "height": image_size[1],
                "answer_area_px": answer_area,
                "relate_area_px": relate_area,
                "union_area_px": union_area,
                "answer_area_frac": round(answer_frac, 6),
                "relate_area_frac": round(relate_frac, 6),
                "union_area_frac": round(union_frac, 6),
                "has_relate_mask": bool(relate_path.exists()),
                "compactness_label": compactness,
                "previous_stage2h": sample_id in PREVIOUS_STAGE2H,
                "metadata_source": metadata.get("source_table", ""),
                "annotation_priority": metadata.get("annotation_priority", ""),
                "support_source_available": metadata.get("support_source_available", ""),
                "nearest_control_available": metadata.get("nearest_control_available", ""),
                "source_effect": metadata.get("source_effect", ""),
                "nearest_gap": metadata.get("nearest_gap", ""),
                "random4_gap": metadata.get("random4_gap", ""),
                "notes": metadata.get("notes", ""),
            }
            row["eligibility"] = _eligibility(row)
            numeric_source_effect = _float_or_none(str(row["source_effect"]))
            score = pack_score + _type_score(reasoning_operation, visual_structure, image_dependence, legacy_visual)
            if row["eligibility"] == "primary_eligible":
                score += 100
            elif row["eligibility"] == "secondary_large_localized":
                score += 40
            else:
                score -= 100
            if row["previous_stage2h"]:
                score -= 25
            if str(row.get("support_source_available")).lower() == "true":
                score += 20
            if str(row.get("nearest_control_available")).lower() == "true":
                score += 15
            if numeric_source_effect is not None:
                score += min(20, max(0, numeric_source_effect * 5))
            score -= answer_frac * 25
            row["selection_score"] = round(score, 3)
            rows.append(row)
    return rows


def _dedupe_for_selection(rows: list[dict[str, Any]], include_previous: bool) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if row["eligibility"] in {"primary_eligible", "secondary_large_localized"}
        and (include_previous or not row["previous_stage2h"])
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        key = row["sample_id"] or row["image_filename"]
        grouped[key].append(row)
    best = [sorted(items, key=lambda row: row["selection_score"], reverse=True)[0] for items in grouped.values()]
    return sorted(best, key=lambda row: row["selection_score"], reverse=True)


def _summarize(rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
    by_eligibility = defaultdict(int)
    by_pack = defaultdict(int)
    by_type = defaultdict(int)
    for row in rows:
        by_eligibility[row["eligibility"]] += 1
        by_pack[row["mask_pack"]] += 1
        by_type[row.get("reasoning_operation") or "unknown"] += 1
    primary_fracs = [float(row["answer_area_frac"]) for row in rows if row["eligibility"] == "primary_eligible"]
    return {
        "total_mask_rows": len(rows),
        "unique_samples": len({row["sample_id"] or row["image_filename"] for row in rows}),
        "eligibility_counts": dict(sorted(by_eligibility.items())),
        "mask_pack_counts": dict(sorted(by_pack.items())),
        "reasoning_operation_counts": dict(sorted(by_type.items())),
        "primary_answer_area_frac_mean": round(mean(primary_fracs), 6) if primary_fracs else "",
        "selected_count": len(selected),
        "selected_samples": [row["sample_id"] for row in selected],
        "claim_boundary": (
            "Stage 2I manifest only selects candidates for cross-model hidden-state/decoded bridge expansion. "
            "It does not by itself establish source-control route replication."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage 2I cross-model expansion candidate manifest.")
    parser.add_argument("--out-all", default=str(ROOT / "doc" / "experiments" / "stage2" / "cross_model" / "stage2i_cross_model_candidate_manifest.csv"))
    parser.add_argument("--out-selected", default=str(ROOT / "doc" / "experiments" / "stage2" / "cross_model" / "stage2i_selected_12_manifest.csv"))
    parser.add_argument("--out-summary", default=str(ROOT / "doc" / "experiments" / "stage2" / "cross_model" / "stage2i_manifest_summary.json"))
    parser.add_argument("--target-count", type=int, default=12)
    parser.add_argument("--max-answer-frac", type=float, default=0.45)
    parser.add_argument("--max-secondary-answer-frac", type=float, default=0.65)
    parser.add_argument("--include-previous", action="store_true")
    args = parser.parse_args()

    rows = sorted(_build_rows(args), key=lambda row: row["selection_score"], reverse=True)
    selected = _dedupe_for_selection(rows, include_previous=args.include_previous)
    if len(selected) < args.target_count and not args.include_previous:
        selected = _dedupe_for_selection(rows, include_previous=True)
    selected = selected[: args.target_count]
    for idx, row in enumerate(selected, start=1):
        row["stage2i_selection_rank"] = idx

    fields = [
        "stage2i_selection_rank",
        "sample_id",
        "image_filename",
        "question_text",
        "answer_text",
        "reasoning_operation",
        "visual_structure",
        "image_dependence",
        "legacy_visual_type_label",
        "legacy_knowledge_level_label",
        "compact_core_tier",
        "mask_pack",
        "mask_dir",
        "local_image_path",
        "width",
        "height",
        "answer_area_px",
        "relate_area_px",
        "union_area_px",
        "answer_area_frac",
        "relate_area_frac",
        "union_area_frac",
        "has_relate_mask",
        "compactness_label",
        "eligibility",
        "previous_stage2h",
        "selection_score",
        "metadata_source",
        "annotation_priority",
        "support_source_available",
        "nearest_control_available",
        "source_effect",
        "nearest_gap",
        "random4_gap",
        "notes",
    ]
    _write_csv(Path(args.out_all), rows, fields)
    _write_csv(Path(args.out_selected), selected, fields)
    _write_json(Path(args.out_summary), _summarize(rows, selected))
    print(f"wrote {args.out_all}")
    print(f"wrote {args.out_selected}")
    print(f"wrote {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

