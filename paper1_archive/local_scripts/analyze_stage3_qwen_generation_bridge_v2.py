#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(r"E:\Bridging")
STAGE3_CROSS = ROOT / "doc" / "experiments" / "stage3" / "cross_model"

INPUTS = [
    ("qwen2p5vl_plt", STAGE3_CROSS / "stage3_qwen2p5vl_plt_generation_bridge_v2.csv"),
    ("qwen2p5vl_clt", STAGE3_CROSS / "stage3_qwen2p5vl_clt_generation_bridge_v2.csv"),
]
CONTROL_GROUPS = {
    "activation_matched_topk",
    "drop_matched_topk",
    "attribution_matched_mask_insensitive_topk",
    "random_active_topk",
}
SOURCE_GROUP = "evidence_attribution_topk"

OUT_SUMMARY = STAGE3_CROSS / "stage3_qwen_generation_bridge_v2_summary.csv"
OUT_CASES = STAGE3_CROSS / "stage3_qwen_generation_bridge_v2_case_table.csv"
OUT_DECISION = STAGE3_CROSS / "stage3_qwen_generation_bridge_v2_decision.json"


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


def bootstrap_ci(values: list[float], *, seed: int = 1729, rounds: int = 2000) -> tuple[float | str, float | str]:
    if not values:
        return "", ""
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed + len(values))
    boot = []
    for _ in range(rounds):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boot.append(mean(sample))
    boot.sort()
    return boot[int(0.025 * (rounds - 1))], boot[int(0.975 * (rounds - 1))]


def stable_seed(*parts: str) -> int:
    value = 1729
    for part in parts:
        for char in part:
            value = (value * 131 + ord(char)) % (2**31)
    return value


def load_rows() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for asset_id, path in INPUTS:
        for row in read_csv(path):
            row = dict(row)
            row["asset_id"] = asset_id
            out.append(row)
    return out


def build_case_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("direction") not in {"restore", "corrupt"}:
            continue
        key = (
            row["asset_id"],
            row["sample_id"],
            row["prompt_name"],
            row["mask_condition"],
            str(int(float(row["top_k"]))),
            row["direction"],
        )
        grouped[key][row.get("feature_group", "")].append(row)

    case_rows: list[dict[str, Any]] = []
    for (asset_id, sample_id, prompt_name, mask_condition, top_k, direction), by_group in sorted(grouped.items()):
        source_rows = by_group.get(SOURCE_GROUP, [])
        if not source_rows:
            continue
        source = source_rows[0]
        controls = [row for group in CONTROL_GROUPS for row in by_group.get(group, [])]
        if not controls:
            continue
        source_seq = to_float(source.get("sequence_effect_vs_reference"))
        source_first = to_float(source.get("first_token_effect_vs_reference"))
        source_rank = to_float(source.get("first_token_rank_effect_vs_reference"))
        control_seq_values = [to_float(row.get("sequence_effect_vs_reference")) for row in controls]
        control_first_values = [to_float(row.get("first_token_effect_vs_reference")) for row in controls]
        control_rank_values = [to_float(row.get("first_token_rank_effect_vs_reference")) for row in controls]
        control_seq = [value for value in control_seq_values if value is not None]
        control_first = [value for value in control_first_values if value is not None]
        control_rank = [value for value in control_rank_values if value is not None]
        if source_seq is None or not control_seq:
            continue
        case_rows.append(
            {
                "asset_id": asset_id,
                "sample_id": sample_id,
                "prompt_name": prompt_name,
                "mask_condition": mask_condition,
                "top_k": top_k,
                "direction": direction,
                "target_answer": source.get("target_answer", ""),
                "baseline_clean_answer": source.get("baseline_clean_answer", ""),
                "baseline_mask_answer": source.get("baseline_mask_answer", ""),
                "source_predicted_answer": source.get("predicted_answer", ""),
                "source_target_hit": is_true(source.get("target_hit")),
                "source_decoded_to_clean": is_true(source.get("decoded_to_clean")),
                "control_decoded_to_clean_count": sum(is_true(row.get("decoded_to_clean")) for row in controls),
                "source_sequence_effect": source_seq,
                "control_sequence_mean": mean(control_seq),
                "source_minus_control_sequence": source_seq - mean(control_seq),
                "source_first_token_effect": source_first if source_first is not None else "",
                "control_first_token_mean": mean(control_first) if control_first else "",
                "source_minus_control_first_token": ""
                if source_first is None or not control_first
                else source_first - mean(control_first),
                "source_first_rank_effect": source_rank if source_rank is not None else "",
                "control_first_rank_mean": mean(control_rank) if control_rank else "",
                "source_minus_control_first_rank": ""
                if source_rank is None or not control_rank
                else source_rank - mean(control_rank),
                "clean_mask_sequence_gap": source.get("clean_mask_sequence_gap", ""),
                "clean_mask_first_logit_gap": source.get("clean_mask_first_logit_gap", ""),
            }
        )
    return case_rows


def summarize(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[(row["asset_id"], row["top_k"], row["direction"])].append(row)

    out: list[dict[str, Any]] = []
    for (asset_id, top_k, direction), items in sorted(grouped.items(), key=lambda x: (x[0][0], int(x[0][1]), x[0][2])):
        seq_diffs = [float(row["source_minus_control_sequence"]) for row in items]
        first_diffs = [
            float(row["source_minus_control_first_token"])
            for row in items
            if row.get("source_minus_control_first_token") != ""
        ]
        rank_diffs = [
            float(row["source_minus_control_first_rank"])
            for row in items
            if row.get("source_minus_control_first_rank") != ""
        ]
        seq_ci_low, seq_ci_high = bootstrap_ci(seq_diffs, seed=stable_seed(asset_id, top_k, direction, "seq"))
        first_ci_low, first_ci_high = bootstrap_ci(first_diffs, seed=stable_seed(asset_id, top_k, direction, "first"))
        out.append(
            {
                "asset_id": asset_id,
                "top_k": top_k,
                "direction": direction,
                "pair_count": len(items),
                "mean_source_sequence_effect": mean(float(row["source_sequence_effect"]) for row in items),
                "mean_control_sequence_effect": mean(float(row["control_sequence_mean"]) for row in items),
                "mean_source_minus_control_sequence": mean(seq_diffs),
                "ci95_low_sequence": seq_ci_low,
                "ci95_high_sequence": seq_ci_high,
                "positive_sequence_count": sum(value > 0 for value in seq_diffs),
                "mean_source_minus_control_first_token": mean(first_diffs) if first_diffs else "",
                "ci95_low_first_token": first_ci_low,
                "ci95_high_first_token": first_ci_high,
                "positive_first_token_count": sum(value > 0 for value in first_diffs),
                "mean_source_minus_control_first_rank": mean(rank_diffs) if rank_diffs else "",
                "positive_first_rank_count": sum(value > 0 for value in rank_diffs),
                "source_decoded_to_clean_count": sum(bool(row["source_decoded_to_clean"]) for row in items),
                "control_decoded_to_clean_count": sum(int(row["control_decoded_to_clean_count"]) for row in items),
            }
        )
    return out


def decide(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    for asset_id in sorted({row["asset_id"] for row in summary_rows}):
        rows = [row for row in summary_rows if row["asset_id"] == asset_id]
        by_topk: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in rows:
            by_topk[row["top_k"]][row["direction"]] = row
        topk_decisions = {}
        supported = []
        partial = []
        first_only = []
        for top_k, dirs in sorted(by_topk.items(), key=lambda item: int(item[0])):
            restore = dirs.get("restore")
            corrupt = dirs.get("corrupt")
            if not restore or not corrupt:
                continue
            restore_lo = to_float(restore.get("ci95_low_sequence"))
            corrupt_lo = to_float(corrupt.get("ci95_low_sequence"))
            restore_mean = to_float(restore.get("mean_source_minus_control_sequence"))
            corrupt_mean = to_float(corrupt.get("mean_source_minus_control_sequence"))
            restore_first = to_float(restore.get("mean_source_minus_control_first_token"))
            corrupt_first = to_float(corrupt.get("mean_source_minus_control_first_token"))
            if restore_lo is not None and corrupt_lo is not None and restore_lo > 0 and corrupt_lo > 0:
                status = "supported_sequence_bridge"
                supported.append(top_k)
            elif restore_mean is not None and corrupt_mean is not None and restore_mean > 0 and corrupt_mean > 0:
                status = "partial_sequence_bridge"
                partial.append(top_k)
            elif (restore_first is not None and restore_first > 0) or (corrupt_first is not None and corrupt_first > 0):
                status = "first_token_only"
                first_only.append(top_k)
            else:
                status = "not_supported"
            topk_decisions[top_k] = {
                "status": status,
                "restore_sequence_gap": restore.get("mean_source_minus_control_sequence"),
                "corrupt_sequence_gap": corrupt.get("mean_source_minus_control_sequence"),
                "restore_decoded_to_clean": restore.get("source_decoded_to_clean_count"),
                "corrupt_decoded_to_clean": corrupt.get("source_decoded_to_clean_count"),
                "corrupt_decoded_changed_vs_clean": int(corrupt.get("pair_count", 0))
                - int(corrupt.get("source_decoded_to_clean_count", 0)),
            }
        if supported:
            status = "supported_sequence_bridge"
        elif partial:
            status = "partial_sequence_bridge"
        elif first_only:
            status = "first_token_only"
        else:
            status = "not_supported"
        decisions[asset_id] = {
            "status": status,
            "supported_topks": supported,
            "partial_topks": partial,
            "first_token_only_topks": first_only,
            "topk_decisions": topk_decisions,
        }
    return {
        "status": "completed",
        "asset_decisions": decisions,
        "claim_boundary": (
            "Sequence bridge is based on target answer sequence logprob. "
            "Decoded generation bridge requires stable source_decoded_to_clean and is not implied by sequence support."
        ),
    }


def main() -> None:
    rows = load_rows()
    case_rows = build_case_rows(rows)
    summary_rows = summarize(case_rows)
    decision = decide(summary_rows)
    write_csv(
        OUT_CASES,
        case_rows,
        [
            "asset_id",
            "sample_id",
            "prompt_name",
            "mask_condition",
            "top_k",
            "direction",
            "target_answer",
            "baseline_clean_answer",
            "baseline_mask_answer",
            "source_predicted_answer",
            "source_target_hit",
            "source_decoded_to_clean",
            "control_decoded_to_clean_count",
            "source_sequence_effect",
            "control_sequence_mean",
            "source_minus_control_sequence",
            "source_first_token_effect",
            "control_first_token_mean",
            "source_minus_control_first_token",
            "source_first_rank_effect",
            "control_first_rank_mean",
            "source_minus_control_first_rank",
            "clean_mask_sequence_gap",
            "clean_mask_first_logit_gap",
        ],
    )
    write_csv(
        OUT_SUMMARY,
        summary_rows,
        [
            "asset_id",
            "top_k",
            "direction",
            "pair_count",
            "mean_source_sequence_effect",
            "mean_control_sequence_effect",
            "mean_source_minus_control_sequence",
            "ci95_low_sequence",
            "ci95_high_sequence",
            "positive_sequence_count",
            "mean_source_minus_control_first_token",
            "ci95_low_first_token",
            "ci95_high_first_token",
            "positive_first_token_count",
            "mean_source_minus_control_first_rank",
            "positive_first_rank_count",
            "source_decoded_to_clean_count",
            "control_decoded_to_clean_count",
        ],
    )
    OUT_DECISION.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
