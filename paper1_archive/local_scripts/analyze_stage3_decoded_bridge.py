#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(r"E:\Bridging")
STAGE3_CROSS = ROOT / "doc" / "experiments" / "stage3" / "cross_model"

INPUTS = [
    ("qwen2p5vl_plt", "effect_gap", STAGE3_CROSS / "stage3_qwen2p5vl_plt_decoded_bridge.csv"),
    ("qwen2p5vl_clt", "effect_gap", STAGE3_CROSS / "stage3_qwen2p5vl_clt_decoded_bridge.csv"),
    ("qwen2p5vl_plt", "rankaware", STAGE3_CROSS / "stage3_qwen2p5vl_plt_decoded_bridge_rankaware.csv"),
    ("qwen2p5vl_clt", "rankaware", STAGE3_CROSS / "stage3_qwen2p5vl_clt_decoded_bridge_rankaware.csv"),
]

OUT_SUMMARY = STAGE3_CROSS / "stage3_decoded_bridge_summary.csv"
OUT_CASES = STAGE3_CROSS / "stage3_decoded_bridge_case_table.csv"
OUT_DECISION = STAGE3_CROSS / "stage3_decoded_bridge_decision.json"


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


def load_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for asset_id, selection_mode, path in INPUTS:
        for row in read_csv(path):
            row = dict(row)
            row["asset_id"] = asset_id
            row["selection_mode"] = selection_mode
            rows.append(row)
    return rows


def summarize(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (row["asset_id"], row["selection_mode"], row["sample_id"], row["prompt_name"] + "::" + row["mask_condition"])
        grouped[key][row["condition"]] = row

    case_rows: list[dict[str, Any]] = []
    for (asset_id, selection_mode, sample_id, prompt_mask), conds in sorted(grouped.items()):
        prompt_name, mask_condition = prompt_mask.split("::", 1)
        clean = conds.get("baseline_clean", {})
        mask = conds.get("baseline_mask", {})
        source_restore = conds.get("source_restore", {})
        control_restore = conds.get("control_restore", {})
        source_zeroing = conds.get("source_zeroing", {})
        control_zeroing = conds.get("control_zeroing", {})
        clean_answer = norm(clean.get("predicted_answer"))
        mask_answer = norm(mask.get("predicted_answer"))
        source_restore_answer = norm(source_restore.get("predicted_answer"))
        control_restore_answer = norm(control_restore.get("predicted_answer"))
        source_zero_answer = norm(source_zeroing.get("predicted_answer"))
        control_zero_answer = norm(control_zeroing.get("predicted_answer"))
        clean_logit = to_float(clean.get("first_step_target_logit"))
        mask_logit = to_float(mask.get("first_step_target_logit"))
        source_restore_logit = to_float(source_restore.get("first_step_target_logit"))
        control_restore_logit = to_float(control_restore.get("first_step_target_logit"))
        source_zero_logit = to_float(source_zeroing.get("first_step_target_logit"))
        control_zero_logit = to_float(control_zeroing.get("first_step_target_logit"))
        clean_rank = to_float(clean.get("first_step_target_rank"))
        mask_rank = to_float(mask.get("first_step_target_rank"))
        source_restore_rank = to_float(source_restore.get("first_step_target_rank"))
        control_restore_rank = to_float(control_restore.get("first_step_target_rank"))
        source_zero_rank = to_float(source_zeroing.get("first_step_target_rank"))
        control_zero_rank = to_float(control_zeroing.get("first_step_target_rank"))

        case_rows.append(
            {
                "asset_id": asset_id,
                "selection_mode": selection_mode,
                "sample_id": sample_id,
                "prompt_name": prompt_name,
                "mask_condition": mask_condition,
                "target_answer": clean.get("target_answer", ""),
                "clean_answer": clean_answer,
                "mask_answer": mask_answer,
                "source_restore_answer": source_restore_answer,
                "control_restore_answer": control_restore_answer,
                "source_zeroing_answer": source_zero_answer,
                "control_zeroing_answer": control_zero_answer,
                "clean_target_hit": is_true(clean.get("target_hit")),
                "mask_target_hit": is_true(mask.get("target_hit")),
                "source_restore_target_hit": is_true(source_restore.get("target_hit")),
                "control_restore_target_hit": is_true(control_restore.get("target_hit")),
                "source_zeroing_target_hit": is_true(source_zeroing.get("target_hit")),
                "control_zeroing_target_hit": is_true(control_zeroing.get("target_hit")),
                "clean_mask_answer_diff": clean_answer != mask_answer,
                "source_restore_to_clean": source_restore_answer == clean_answer and clean_answer != mask_answer,
                "source_restore_changed_vs_mask": source_restore_answer != mask_answer,
                "control_restore_to_clean": control_restore_answer == clean_answer and clean_answer != mask_answer,
                "source_zeroing_changed_vs_clean": source_zero_answer != clean_answer,
                "control_zeroing_changed_vs_clean": control_zero_answer != clean_answer,
                "source_restore_logit_vs_mask": "" if source_restore_logit is None or mask_logit is None else source_restore_logit - mask_logit,
                "control_restore_logit_vs_mask": "" if control_restore_logit is None or mask_logit is None else control_restore_logit - mask_logit,
                "source_minus_control_restore_logit": ""
                if source_restore_logit is None or control_restore_logit is None
                else source_restore_logit - control_restore_logit,
                "source_restore_rank_vs_mask": "" if source_restore_rank is None or mask_rank is None else mask_rank - source_restore_rank,
                "control_restore_rank_vs_mask": "" if control_restore_rank is None or mask_rank is None else mask_rank - control_restore_rank,
                "source_zeroing_logit_damage": "" if clean_logit is None or source_zero_logit is None else clean_logit - source_zero_logit,
                "control_zeroing_logit_damage": "" if clean_logit is None or control_zero_logit is None else clean_logit - control_zero_logit,
                "source_minus_control_zeroing_logit": ""
                if source_zero_logit is None or control_zero_logit is None
                else (clean_logit - source_zero_logit) - (clean_logit - control_zero_logit)
                if clean_logit is not None
                else "",
                "source_zeroing_rank_damage": "" if clean_rank is None or source_zero_rank is None else source_zero_rank - clean_rank,
                "control_zeroing_rank_damage": "" if clean_rank is None or control_zero_rank is None else control_zero_rank - clean_rank,
            }
        )

    summary_rows: list[dict[str, Any]] = []
    by_asset_mode: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        by_asset_mode[(row["asset_id"], row["selection_mode"])].append(row)
    for (asset_id, selection_mode), items in sorted(by_asset_mode.items()):
        def values(field: str) -> list[float]:
            return [float(row[field]) for row in items if row.get(field) != ""]

        summary_rows.append(
            {
                "asset_id": asset_id,
                "selection_mode": selection_mode,
                "pair_count": len(items),
                "clean_mask_answer_diff_count": sum(bool(row["clean_mask_answer_diff"]) for row in items),
                "clean_target_hit_count": sum(bool(row["clean_target_hit"]) for row in items),
                "mask_target_hit_count": sum(bool(row["mask_target_hit"]) for row in items),
                "source_restore_target_hit_count": sum(bool(row["source_restore_target_hit"]) for row in items),
                "source_restore_to_clean_count": sum(bool(row["source_restore_to_clean"]) for row in items),
                "source_restore_changed_vs_mask_count": sum(bool(row["source_restore_changed_vs_mask"]) for row in items),
                "control_restore_to_clean_count": sum(bool(row["control_restore_to_clean"]) for row in items),
                "source_zeroing_changed_vs_clean_count": sum(bool(row["source_zeroing_changed_vs_clean"]) for row in items),
                "control_zeroing_changed_vs_clean_count": sum(bool(row["control_zeroing_changed_vs_clean"]) for row in items),
                "mean_source_restore_logit_vs_mask": mean(values("source_restore_logit_vs_mask")) if values("source_restore_logit_vs_mask") else "",
                "mean_control_restore_logit_vs_mask": mean(values("control_restore_logit_vs_mask")) if values("control_restore_logit_vs_mask") else "",
                "mean_source_minus_control_restore_logit": mean(values("source_minus_control_restore_logit")) if values("source_minus_control_restore_logit") else "",
                "mean_source_restore_rank_vs_mask": mean(values("source_restore_rank_vs_mask")) if values("source_restore_rank_vs_mask") else "",
                "mean_control_restore_rank_vs_mask": mean(values("control_restore_rank_vs_mask")) if values("control_restore_rank_vs_mask") else "",
                "mean_source_zeroing_logit_damage": mean(values("source_zeroing_logit_damage")) if values("source_zeroing_logit_damage") else "",
                "mean_control_zeroing_logit_damage": mean(values("control_zeroing_logit_damage")) if values("control_zeroing_logit_damage") else "",
                "mean_source_minus_control_zeroing_logit": mean(values("source_minus_control_zeroing_logit")) if values("source_minus_control_zeroing_logit") else "",
                "mean_source_zeroing_rank_damage": mean(values("source_zeroing_rank_damage")) if values("source_zeroing_rank_damage") else "",
                "mean_control_zeroing_rank_damage": mean(values("control_zeroing_rank_damage")) if values("control_zeroing_rank_damage") else "",
            }
        )

    decisions = {}
    for row in summary_rows:
        decoded_restore = int(row["source_restore_to_clean_count"])
        zeroing_changes = int(row["source_zeroing_changed_vs_clean_count"])
        source_restore_logit = to_float(row["mean_source_restore_logit_vs_mask"])
        source_restore_rank = to_float(row["mean_source_restore_rank_vs_mask"])
        if decoded_restore >= 2:
            status = "partial_decoded_generation_bridge"
        elif zeroing_changes >= 2:
            status = "decoded_zeroing_changes_only"
        elif source_restore_logit is not None and source_restore_logit > 0 and source_restore_rank is not None and source_restore_rank > 0:
            status = "first_token_rank_bridge_only"
        else:
            status = "not_supported_generation_bridge"
        decisions[f"{row['asset_id']}::{row['selection_mode']}"] = {
            "status": status,
            "pair_count": row["pair_count"],
            "clean_mask_answer_diff_count": row["clean_mask_answer_diff_count"],
            "source_restore_to_clean_count": row["source_restore_to_clean_count"],
            "source_zeroing_changed_vs_clean_count": row["source_zeroing_changed_vs_clean_count"],
        }
    decision = {
        "status": "completed",
        "asset_mode_decisions": decisions,
        "claim_boundary": (
            "Decoded generation bridge remains a smoke test. Rank-aware rows show first-token logit/rank movement, "
            "but source_restore rarely changes the decoded answer back to clean. Do not claim generation-level "
            "cross-model causal bridge unless source_restore_to_clean becomes stable."
        ),
    }
    return summary_rows, case_rows, decision


def main() -> None:
    rows = load_rows()
    summary_rows, case_rows, decision = summarize(rows)
    write_csv(
        OUT_SUMMARY,
        summary_rows,
        [
            "asset_id",
            "selection_mode",
            "pair_count",
            "clean_mask_answer_diff_count",
            "clean_target_hit_count",
            "mask_target_hit_count",
            "source_restore_target_hit_count",
            "source_restore_to_clean_count",
            "source_restore_changed_vs_mask_count",
            "control_restore_to_clean_count",
            "source_zeroing_changed_vs_clean_count",
            "control_zeroing_changed_vs_clean_count",
            "mean_source_restore_logit_vs_mask",
            "mean_control_restore_logit_vs_mask",
            "mean_source_minus_control_restore_logit",
            "mean_source_restore_rank_vs_mask",
            "mean_control_restore_rank_vs_mask",
            "mean_source_zeroing_logit_damage",
            "mean_control_zeroing_logit_damage",
            "mean_source_minus_control_zeroing_logit",
            "mean_source_zeroing_rank_damage",
            "mean_control_zeroing_rank_damage",
        ],
    )
    write_csv(
        OUT_CASES,
        case_rows,
        [
            "asset_id",
            "selection_mode",
            "sample_id",
            "prompt_name",
            "mask_condition",
            "target_answer",
            "clean_answer",
            "mask_answer",
            "source_restore_answer",
            "control_restore_answer",
            "source_zeroing_answer",
            "control_zeroing_answer",
            "clean_target_hit",
            "mask_target_hit",
            "source_restore_target_hit",
            "control_restore_target_hit",
            "source_zeroing_target_hit",
            "control_zeroing_target_hit",
            "clean_mask_answer_diff",
            "source_restore_to_clean",
            "source_restore_changed_vs_mask",
            "control_restore_to_clean",
            "source_zeroing_changed_vs_clean",
            "control_zeroing_changed_vs_clean",
            "source_restore_logit_vs_mask",
            "control_restore_logit_vs_mask",
            "source_minus_control_restore_logit",
            "source_restore_rank_vs_mask",
            "control_restore_rank_vs_mask",
            "source_zeroing_logit_damage",
            "control_zeroing_logit_damage",
            "source_minus_control_zeroing_logit",
            "source_zeroing_rank_damage",
            "control_zeroing_rank_damage",
        ],
    )
    OUT_DECISION.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
