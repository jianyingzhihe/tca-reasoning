#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
CROSS_DIR = ROOT / "doc" / "experiments" / "stage2" / "cross_model"
SOURCE_MANIFEST = CROSS_DIR / "stage2i_cross_model_candidate_manifest.csv"


DEFAULT_QUOTAS = {
    "symbol_text_reading": 12,
    "visual_readout": 8,
    "scene_inference": 4,
}


def _read_csv(path: Path) -> list[dict[str, str]]:
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


def _float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("-inf")


def _eligible(row: dict[str, str]) -> bool:
    if row.get("eligibility") not in {"primary_eligible", "secondary_large_localized"}:
        return False
    if not row.get("sample_id") or not row.get("image_filename"):
        return False
    if not row.get("question_text") or not row.get("answer_text"):
        return False
    if not Path(row.get("local_image_path", "")).exists():
        return False
    mask_dir = Path(row.get("mask_dir", ""))
    if not (mask_dir / "answer.png").exists():
        return False
    return True


def _dedupe_best(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = row["sample_id"] or row["image_filename"]
        grouped[key].append(row)
    best = [sorted(items, key=lambda item: _float(item.get("selection_score", "")), reverse=True)[0] for items in grouped.values()]
    return sorted(best, key=lambda item: _float(item.get("selection_score", "")), reverse=True)


def _parse_quotas(raw: str) -> dict[str, int]:
    if not raw:
        return dict(DEFAULT_QUOTAS)
    out = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        key, value = part.split("=", 1)
        out[key.strip()] = int(value.strip())
    return out


def _select(rows: list[dict[str, str]], quotas: dict[str, int], target_count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_type[row.get("reasoning_operation", "")].append(row)

    deficits: dict[str, int] = {}
    for reasoning_type, quota in quotas.items():
        candidates = by_type.get(reasoning_type, [])
        take = min(quota, len(candidates), target_count - len(selected))
        for row in candidates[:take]:
            key = row["sample_id"] or row["image_filename"]
            selected.append(dict(row))
            selected_keys.add(key)
        if take < quota:
            deficits[reasoning_type] = quota - take

    if len(selected) < target_count:
        for row in rows:
            key = row["sample_id"] or row["image_filename"]
            if key in selected_keys:
                continue
            selected.append(dict(row))
            selected_keys.add(key)
            if len(selected) >= target_count:
                break

    for rank, row in enumerate(selected, start=1):
        row["stage2m_selection_rank"] = rank
        row["stage2m_selection_note"] = "quota_selected" if rank <= sum(min(quotas.get(t, 0), len(by_type.get(t, []))) for t in quotas) else "fill_selected"

    counts = Counter(row.get("reasoning_operation", "") for row in selected)
    summary = {
        "target_count": target_count,
        "selected_count": len(selected),
        "available_eligible_count": len(rows),
        "type_quotas": quotas,
        "selected_type_counts": dict(counts),
        "quota_deficits": deficits,
        "usable_status": "pass_min20" if len(selected) >= 20 else "blocked_need_more_annotations",
        "selected_samples": [row["sample_id"] for row in selected],
        "claim_boundary": (
            "Stage 2M manifest selects existing localized samples for cross-model replication. "
            "It does not establish replication by itself."
        ),
    }
    return selected, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage 2M expanded cross-model manifest.")
    parser.add_argument("--source", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--out-selected", type=Path, default=CROSS_DIR / "stage2m_selected_24_manifest.csv")
    parser.add_argument("--out-supplement", type=Path, default=CROSS_DIR / "stage2m_annotation_supplement_needed.csv")
    parser.add_argument("--out-summary", type=Path, default=CROSS_DIR / "stage2m_manifest_summary.json")
    parser.add_argument("--target-count", type=int, default=24)
    parser.add_argument("--quotas", default="")
    args = parser.parse_args()

    quotas = _parse_quotas(args.quotas)
    raw_rows = _read_csv(args.source)
    eligible = _dedupe_best([row for row in raw_rows if _eligible(row)])
    selected, summary = _select(eligible, quotas, args.target_count)

    selected_fields = [
        "stage2m_selection_rank",
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
        "stage2m_selection_note",
    ]
    supplement = []
    for reasoning_type, deficit in summary["quota_deficits"].items():
        supplement.append(
            {
                "reasoning_operation": reasoning_type,
                "needed_count": deficit,
                "preferred_properties": "localized compact answer mask, strong image_dependence, non-diffuse evidence region",
            }
        )

    _write_csv(args.out_selected, selected, selected_fields)
    _write_csv(args.out_supplement, supplement, ["reasoning_operation", "needed_count", "preferred_properties"])
    _write_json(args.out_summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
