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
STAGE2M_MANIFEST = CROSS_DIR / "stage2m_selected_24_manifest.csv"


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
        grouped[row["sample_id"] or row["image_filename"]].append(row)
    best = [
        sorted(items, key=lambda item: _float(item.get("selection_score", "")), reverse=True)[0]
        for items in grouped.values()
    ]
    return sorted(best, key=lambda item: _float(item.get("selection_score", "")), reverse=True)


def _stage2m_ids(path: Path) -> set[str]:
    return {row["sample_id"] for row in _read_csv(path) if row.get("sample_id")}


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "stage2n_selection_rank",
        "sample_id",
        "image_filename",
        "question_text",
        "answer_text",
        "reasoning_operation",
        "visual_structure",
        "image_dependence",
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
        "selection_score",
        "metadata_source",
        "annotation_priority",
        "stage2n_selection_note",
    ]
    seen = set(preferred)
    for row in rows:
        for key in row:
            if key not in seen:
                preferred.append(key)
                seen.add(key)
    return preferred


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage 2N heldout cross-model manifest.")
    parser.add_argument("--source", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--stage2m-manifest", type=Path, default=STAGE2M_MANIFEST)
    parser.add_argument("--out-heldout", type=Path, default=CROSS_DIR / "stage2n_heldout_manifest.csv")
    parser.add_argument("--out-all52", type=Path, default=CROSS_DIR / "stage2n_all52_manifest.csv")
    parser.add_argument("--out-supplement", type=Path, default=CROSS_DIR / "stage2n_annotation_supplement_needed.csv")
    parser.add_argument("--out-summary", type=Path, default=CROSS_DIR / "stage2n_manifest_summary.json")
    parser.add_argument("--min-heldout", type=int, default=20)
    args = parser.parse_args()

    raw_rows = _read_csv(args.source)
    eligible = _dedupe_best([row for row in raw_rows if _eligible(row)])
    used_ids = _stage2m_ids(args.stage2m_manifest)
    heldout = [dict(row) for row in eligible if row.get("sample_id") not in used_ids]
    all52 = [dict(row) for row in eligible]
    for rank, row in enumerate(heldout, start=1):
        if not row.get("reasoning_operation"):
            row["reasoning_operation"] = "untyped_localized"
        row["stage2n_selection_rank"] = rank
        row["stage2n_selection_note"] = "heldout_not_used_in_stage2m"
    for rank, row in enumerate(all52, start=1):
        if not row.get("reasoning_operation"):
            row["reasoning_operation"] = "untyped_localized"
        row["stage2n_selection_rank"] = rank
        row["stage2n_selection_note"] = "all52_pool_stage2m_or_stage2n"

    heldout_counts = Counter(row.get("reasoning_operation", "") for row in heldout)
    all_counts = Counter(row.get("reasoning_operation", "") for row in all52)
    status = "pass_min20" if len(heldout) >= args.min_heldout else "blocked_need_more_annotations"
    summary = {
        "created_from": str(args.source),
        "stage2m_excluded_count": len(used_ids),
        "eligible_count": len(eligible),
        "heldout_count": len(heldout),
        "heldout_type_counts": dict(heldout_counts),
        "all52_type_counts": dict(all_counts),
        "usable_status": status,
        "heldout_samples": [row["sample_id"] for row in heldout],
        "claim_boundary": (
            "Stage 2N manifest excludes Stage 2M selected samples and only prepares heldout hidden-state "
            "replication. It does not establish replication by itself."
        ),
    }
    supplement = []
    if status != "pass_min20":
        supplement.append(
            {
                "needed_count": args.min_heldout - len(heldout),
                "preferred_properties": "localized compact answer mask, strong image_dependence, not used in Stage 2M",
            }
        )

    fields = _fieldnames(all52)
    _write_csv(args.out_heldout, heldout, fields)
    _write_csv(args.out_all52, all52, fields)
    _write_csv(args.out_supplement, supplement, ["needed_count", "preferred_properties"])
    _write_json(args.out_summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
