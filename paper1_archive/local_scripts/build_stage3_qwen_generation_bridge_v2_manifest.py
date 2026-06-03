#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
STAGE3_CROSS = ROOT / "doc" / "experiments" / "stage3" / "cross_model"

CASE_TABLE = STAGE3_CROSS / "stage3_decoded_bridge_case_table.csv"
SOURCE_CONTROL = {
    "qwen2p5vl_plt": STAGE3_CROSS / "stage3_qwen2p5vl_plt_source_control.csv",
    "qwen2p5vl_clt": STAGE3_CROSS / "stage3_qwen2p5vl_clt_source_control.csv",
}
TRANSCODER = {
    "qwen2p5vl_plt": "KokosDev/qwen2p5vl-7b-plt",
    "qwen2p5vl_clt": "KokosDev/qwen2p5vl-7b-clt",
}

OUT_MANIFEST = STAGE3_CROSS / "stage3_qwen_generation_bridge_v2_manifest.csv"
OUT_REPORT = STAGE3_CROSS / "stage3_qwen_generation_bridge_v2_manifest_report.json"


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


def to_float(value: Any) -> float | None:
    try:
        if value == "" or value is None:
            return None
        return float(value)
    except Exception:
        return None


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def norm(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower()
    if text in {"nan", "none"}:
        return ""
    return text


def source_control_index() -> dict[tuple[str, str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for asset_id, path in SOURCE_CONTROL.items():
        grouped: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
        for row in read_csv(path):
            if row.get("feature_role") != "source" or row.get("mask_variant") != "real_mask":
                continue
            key = (row.get("sample_id", ""), row.get("prompt_name", ""), row.get("mask_condition", ""))
            grouped[key][row.get("intervention", "")] = row
        for (sample_id, prompt_name, mask_condition), rows in grouped.items():
            restore = rows.get("restore")
            zeroing = rows.get("zeroing")
            if not restore or not zeroing:
                continue
            clean_rank = to_float(zeroing.get("reference_target_rank"))
            mask_rank = to_float(restore.get("reference_target_rank"))
            restore_effect = to_float(restore.get("effect_logit"))
            zeroing_effect = to_float(zeroing.get("effect_logit"))
            if clean_rank is None or mask_rank is None:
                continue
            out[(asset_id, sample_id, prompt_name, mask_condition)] = {
                "clean_target_rank": clean_rank,
                "mask_target_rank": mask_rank,
                "rank_gap": mask_rank - clean_rank,
                "restore_effect_logit": restore_effect if restore_effect is not None else "",
                "zeroing_effect_logit": zeroing_effect if zeroing_effect is not None else "",
                "target_token_id": restore.get("target_token_id", ""),
                "target_token": restore.get("target_token", ""),
                "layer": restore.get("layer", "26"),
                "bucket": restore.get("bucket", ""),
            }
    return out


def main() -> None:
    sc = source_control_index()
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in read_csv(CASE_TABLE):
        asset_id = row.get("asset_id", "")
        if asset_id not in TRANSCODER:
            continue
        key = (asset_id, row.get("sample_id", ""), row.get("prompt_name", ""), row.get("mask_condition", ""))
        if key in seen:
            continue
        seen.add(key)
        clean_hit = is_true(row.get("clean_target_hit"))
        mask_hit = is_true(row.get("mask_target_hit"))
        clean_mask_diff = is_true(row.get("clean_mask_answer_diff"))
        info = sc.get(key)
        reason = []
        if not clean_hit:
            reason.append("clean_decoded_not_target_hit")
        if mask_hit:
            reason.append("mask_still_target_hit")
        if not clean_mask_diff:
            reason.append("clean_mask_answer_not_different")
        if info is None:
            reason.append("missing_source_control_rank")
        elif not (float(info["clean_target_rank"]) <= 5 and float(info["mask_target_rank"]) > float(info["clean_target_rank"])):
            reason.append("rank_condition_failed")
        if reason:
            rejected.append({"asset_id": asset_id, "sample_id": key[1], "prompt_name": key[2], "mask_condition": key[3], "reason": "|".join(reason)})
            continue
        restore = to_float(info.get("restore_effect_logit"))
        zeroing = to_float(info.get("zeroing_effect_logit"))
        source_positive = (restore is not None and restore > 0) or (zeroing is not None and zeroing > 0)
        if not source_positive:
            rejected.append({"asset_id": asset_id, "sample_id": key[1], "prompt_name": key[2], "mask_condition": key[3], "reason": "source_restore_zeroing_not_positive"})
            continue
        score = float(info["rank_gap"]) + 10.0 * max(float(restore or 0.0), 0.0) + 5.0 * max(float(zeroing or 0.0), 0.0)
        selected.append(
            {
                "asset_id": asset_id,
                "transcoder_ref": TRANSCODER[asset_id],
                "model_family": "qwen",
                "sample_id": key[1],
                "prompt_name": key[2],
                "mask_condition": key[3],
                "target_answer": row.get("target_answer", ""),
                "clean_answer": norm(row.get("clean_answer")),
                "mask_answer": norm(row.get("mask_answer")),
                "selection_mode_source": row.get("selection_mode", ""),
                "clean_target_rank": info["clean_target_rank"],
                "mask_target_rank": info["mask_target_rank"],
                "rank_gap": info["rank_gap"],
                "restore_effect_logit": info.get("restore_effect_logit", ""),
                "zeroing_effect_logit": info.get("zeroing_effect_logit", ""),
                "target_token_id": info.get("target_token_id", ""),
                "target_token": info.get("target_token", ""),
                "layer": info.get("layer", "26"),
                "bucket": info.get("bucket", ""),
                "selection_score": score,
                "selection_basis": "strict_behavior_rank_source_positive",
            }
        )

    final_rows: list[dict[str, Any]] = []
    by_asset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_asset[row["asset_id"]].append(row)
    for asset_id in sorted(TRANSCODER):
        rows = sorted(by_asset.get(asset_id, []), key=lambda item: float(item["selection_score"]), reverse=True)
        final_rows.extend(rows[:12])

    fields = [
        "asset_id",
        "transcoder_ref",
        "model_family",
        "sample_id",
        "prompt_name",
        "mask_condition",
        "target_answer",
        "clean_answer",
        "mask_answer",
        "selection_mode_source",
        "clean_target_rank",
        "mask_target_rank",
        "rank_gap",
        "restore_effect_logit",
        "zeroing_effect_logit",
        "target_token_id",
        "target_token",
        "layer",
        "bucket",
        "selection_score",
        "selection_basis",
    ]
    write_csv(OUT_MANIFEST, final_rows, fields)
    report = {
        "status": "completed",
        "selected_count": len(final_rows),
        "selected_by_asset": {asset_id: sum(1 for row in final_rows if row["asset_id"] == asset_id) for asset_id in sorted(TRANSCODER)},
        "rejected_count": len(rejected),
        "selection_rule": (
            "clean_target_hit && !mask_target_hit && clean_mask_answer_diff && "
            "clean_target_rank<=5 && mask_target_rank>clean_target_rank && positive source restore/zeroing"
        ),
        "note": "If fewer than 12 rows per asset are selected, the strict behavior-aware filter is intentionally not backfilled.",
        "rejected_sample": rejected[:20],
    }
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
