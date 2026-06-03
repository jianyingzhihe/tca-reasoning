#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
STAGE3_CROSS = ROOT / "doc" / "experiments" / "stage3" / "cross_model"

INPUTS = [
    ("qwen2p5vl_plt", "KokosDev/qwen2p5vl-7b-plt", STAGE3_CROSS / "stage3_qwen2p5vl_plt_source_control.csv"),
    ("qwen2p5vl_clt", "KokosDev/qwen2p5vl-7b-clt", STAGE3_CROSS / "stage3_qwen2p5vl_clt_source_control.csv"),
]

OUT_MANIFEST = STAGE3_CROSS / "stage3_qwen_decoded_bridge_manifest.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def to_float(value: Any) -> float | None:
    try:
        if value == "" or value is None:
            return None
        return float(value)
    except Exception:
        return None


def build_asset_rows(asset_id: str, transcoder_ref: str, rows: list[dict[str, str]], *, limit: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, dict[str, dict[str, str]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row.get("mask_variant") != "real_mask":
            continue
        if row.get("position_group") != "top_hidden_delta_plus_answer_adjacent":
            continue
        key = (
            row.get("sample_id", ""),
            row.get("prompt_name", ""),
            row.get("mask_condition", ""),
            row.get("position_group", ""),
        )
        role = row.get("feature_role", "")
        intervention = row.get("intervention", "")
        grouped[key][role][intervention] = row

    candidates: list[dict[str, Any]] = []
    for (sample_id, prompt_name, mask_condition, position_group), roles in sorted(grouped.items()):
        source_restore = roles.get("source", {}).get("restore")
        source_zeroing = roles.get("source", {}).get("zeroing")
        control_restore = roles.get("matched_control", {}).get("restore")
        control_zeroing = roles.get("matched_control", {}).get("zeroing")
        if not all([source_restore, source_zeroing, control_restore, control_zeroing]):
            continue
        source_restore_logit = to_float(source_restore.get("effect_logit"))
        source_zeroing_logit = to_float(source_zeroing.get("effect_logit"))
        control_restore_logit = to_float(control_restore.get("effect_logit"))
        control_zeroing_logit = to_float(control_zeroing.get("effect_logit"))
        if None in [source_restore_logit, source_zeroing_logit, control_restore_logit, control_zeroing_logit]:
            continue
        restore_gap = float(source_restore_logit) - float(control_restore_logit)
        zeroing_gap = float(source_zeroing_logit) - float(control_zeroing_logit)
        score = restore_gap + zeroing_gap
        candidates.append(
            {
                "asset_id": asset_id,
                "transcoder_ref": transcoder_ref,
                "model_family": source_restore.get("model_family", "qwen"),
                "sample_id": sample_id,
                "prompt_name": prompt_name,
                "mask_condition": mask_condition,
                "position_group": position_group,
                "layer": source_restore.get("layer", "26"),
                "bucket": source_restore.get("bucket", ""),
                "target_answer": source_restore.get("target_answer", ""),
                "target_token_id": source_restore.get("target_token_id", ""),
                "target_token": source_restore.get("target_token", ""),
                "source_feature_id": source_restore.get("feature_id", ""),
                "control_feature_id": control_restore.get("feature_id", ""),
                "control_feature_group": control_restore.get("feature_group", ""),
                "source_restore_effect_logit": source_restore_logit,
                "control_restore_effect_logit": control_restore_logit,
                "source_zeroing_effect_logit": source_zeroing_logit,
                "control_zeroing_effect_logit": control_zeroing_logit,
                "restore_source_minus_control": restore_gap,
                "zeroing_source_minus_control": zeroing_gap,
                "selection_score": score,
            }
        )

    candidates.sort(key=lambda row: (float(row["selection_score"]), float(row["source_zeroing_effect_logit"])), reverse=True)
    return candidates[:limit]


def main() -> None:
    selected: list[dict[str, Any]] = []
    for asset_id, transcoder_ref, path in INPUTS:
        selected.extend(build_asset_rows(asset_id, transcoder_ref, read_csv(path), limit=4))

    fields = [
        "asset_id",
        "transcoder_ref",
        "model_family",
        "sample_id",
        "prompt_name",
        "mask_condition",
        "position_group",
        "layer",
        "bucket",
        "target_answer",
        "target_token_id",
        "target_token",
        "source_feature_id",
        "control_feature_id",
        "control_feature_group",
        "source_restore_effect_logit",
        "control_restore_effect_logit",
        "source_zeroing_effect_logit",
        "control_zeroing_effect_logit",
        "restore_source_minus_control",
        "zeroing_source_minus_control",
        "selection_score",
    ]
    write_csv(OUT_MANIFEST, selected, fields)
    print(f"wrote {len(selected)} rows to {OUT_MANIFEST}")
    for row in selected:
        print(
            row["asset_id"],
            row["sample_id"],
            row["prompt_name"],
            row["mask_condition"],
            f"score={float(row['selection_score']):.3f}",
        )


if __name__ == "__main__":
    main()
