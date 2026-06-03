#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
STAGE6_CROSS = ROOT / "doc" / "experiments" / "stage6" / "cross_model"
PREFIX = "stage6_node_generation_bridge"
SOURCE_GROUPS = {"source_hidden", "source_plt_topk", "source_plt_error", "evidence_attribution_topk"}
CONTROL_GROUPS = {
    "control_hidden",
    "control_plt_topk",
    "control_plt_error",
    "activation_matched_topk",
    "drop_matched_topk",
    "attribution_matched_mask_insensitive_topk",
    "random_active_topk",
}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _b(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _stem(mode: str, tag: str) -> str:
    return f"{PREFIX}_{mode}_{tag}"


def _load_rows(mode: str, tag: str) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    stem = _stem(mode, tag)
    paths = [
        ("gemma3_hidden", STAGE6_CROSS / f"{stem}_gemma.csv"),
        ("qwen2p5vl_hidden", STAGE6_CROSS / f"{stem}_qwen2p5vl_hidden.csv"),
        ("qwen2p5vl_plt", STAGE6_CROSS / f"{stem}_qwen2p5vl_plt.csv"),
        ("qwen2p5vl_clt", STAGE6_CROSS / f"{stem}_qwen2p5vl_clt.csv"),
    ]
    rows: list[dict[str, str]] = []
    artifacts: list[dict[str, Any]] = []
    for asset_id, path in paths:
        loaded = _read_csv(path)
        for row in loaded:
            row.setdefault("asset_id", asset_id)
        rows.extend(loaded)
        artifacts.append({"asset_id": asset_id, "path": str(path), "exists": path.exists(), "rows": len(loaded)})
    return rows, artifacts


def _infer_bridge_operator(row: dict[str, str]) -> str:
    explicit = row.get("bridge_operator", "")
    if explicit:
        return explicit
    feature_group = row.get("feature_group", "")
    asset_id = row.get("asset_id", "")
    if feature_group in {"source_hidden", "control_hidden"} or asset_id.endswith("_hidden"):
        return "hidden_residual"
    if feature_group in {"source_plt_topk", "control_plt_topk"}:
        return "plt_topk_reconstruction"
    if feature_group in {"source_plt_error", "control_plt_error"}:
        return "plt_reconstruction_error"
    if asset_id.endswith("_clt"):
        return "qwen_clt_multifeature"
    if asset_id.endswith("_plt"):
        return "qwen_plt_multifeature"
    return "unspecified"


def _oriented_gap(direction: str, source: float | None, controls: list[float]) -> float | str:
    if source is None or not controls:
        return ""
    control_mean = _mean(controls)
    if direction == "corrupt":
        return control_mean - source
    return source - control_mean


def _build_case_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str, str], dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        direction = row.get("direction", "")
        if direction not in {"restore", "corrupt"}:
            continue
        bridge_operator = _infer_bridge_operator(row)
        key = (
            row.get("asset_id", ""),
            bridge_operator,
            row.get("sample_id", ""),
            row.get("prompt_name", ""),
            row.get("mask_condition", ""),
            row.get("top_k", ""),
            direction,
        )
        grouped[key][row.get("feature_group", "")].append(row)

    case_rows: list[dict[str, Any]] = []
    for (asset_id, bridge_operator, sample_id, prompt_name, mask_condition, top_k, direction), by_group in sorted(grouped.items()):
        source_candidates = [row for group in SOURCE_GROUPS for row in by_group.get(group, [])]
        controls = [row for group in CONTROL_GROUPS for row in by_group.get(group, [])]
        if not source_candidates or not controls:
            continue
        source = source_candidates[0]
        control_seq = [value for value in (_f(row.get("sequence_effect_vs_reference")) for row in controls) if value is not None]
        control_first = [value for value in (_f(row.get("first_token_effect_vs_reference")) for row in controls) if value is not None]
        control_rank = [value for value in (_f(row.get("first_token_rank_effect_vs_reference")) for row in controls) if value is not None]
        control_margin = [value for value in (_f(row.get("first_token_margin_effect_vs_reference")) for row in controls) if value is not None]
        source_seq = _f(source.get("sequence_effect_vs_reference"))
        source_first = _f(source.get("first_token_effect_vs_reference"))
        source_rank = _f(source.get("first_token_rank_effect_vs_reference"))
        source_margin = _f(source.get("first_token_margin_effect_vs_reference"))
        case_rows.append(
            {
                "asset_id": asset_id,
                "bridge_operator": bridge_operator,
                "sample_id": sample_id,
                "prompt_name": prompt_name,
                "mask_condition": mask_condition,
                "top_k": top_k,
                "direction": direction,
                "target_answer": source.get("target_answer", ""),
                "baseline_clean_answer": source.get("baseline_clean_answer", ""),
                "baseline_mask_answer": source.get("baseline_mask_answer", ""),
                "source_predicted_answer": source.get("predicted_answer", ""),
                "source_target_hit": _b(source.get("target_hit")),
                "source_decoded_to_clean": _b(source.get("decoded_to_clean")),
                "source_decoded_changed_vs_reference": _b(source.get("decoded_changed_vs_reference")),
                "control_decoded_to_clean_count": sum(_b(row.get("decoded_to_clean")) for row in controls),
                "source_sequence_effect": source_seq if source_seq is not None else "",
                "control_sequence_mean": _mean(control_seq) if control_seq else "",
                "oriented_source_minus_control_sequence": _oriented_gap(direction, source_seq, control_seq),
                "source_first_token_effect": source_first if source_first is not None else "",
                "control_first_token_mean": _mean(control_first) if control_first else "",
                "oriented_source_minus_control_first_token": _oriented_gap(direction, source_first, control_first),
                "source_first_rank_effect": source_rank if source_rank is not None else "",
                "control_first_rank_mean": _mean(control_rank) if control_rank else "",
                "oriented_source_minus_control_first_rank": _oriented_gap(direction, source_rank, control_rank),
                "source_margin_effect": source_margin if source_margin is not None else "",
                "control_margin_mean": _mean(control_margin) if control_margin else "",
                "oriented_source_minus_control_margin": _oriented_gap(direction, source_margin, control_margin),
                "clean_mask_sequence_gap": source.get("clean_mask_sequence_gap", ""),
                "clean_mask_first_logit_gap": source.get("clean_mask_first_logit_gap", ""),
                "clean_mask_first_margin_gap": source.get("clean_mask_first_margin_gap", ""),
            }
        )
    return case_rows


def _summarize(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[(row["asset_id"], row.get("bridge_operator", ""), row["direction"])].append(row)

    out: list[dict[str, Any]] = []
    for (asset_id, bridge_operator, direction), items in sorted(grouped.items()):
        seq = [_f(row.get("oriented_source_minus_control_sequence")) for row in items]
        first = [_f(row.get("oriented_source_minus_control_first_token")) for row in items]
        rank = [_f(row.get("oriented_source_minus_control_first_rank")) for row in items]
        margin = [_f(row.get("oriented_source_minus_control_margin")) for row in items]
        seq_values = [value for value in seq if value is not None]
        first_values = [value for value in first if value is not None]
        rank_values = [value for value in rank if value is not None]
        margin_values = [value for value in margin if value is not None]
        out.append(
            {
                "asset_id": asset_id,
                "bridge_operator": bridge_operator,
                "direction": direction,
                "case_count": len(items),
                "mean_oriented_sequence_gap": _mean(seq_values) if seq_values else "",
                "positive_sequence_frac": sum(value > 0 for value in seq_values) / len(seq_values) if seq_values else "",
                "mean_oriented_first_token_gap": _mean(first_values) if first_values else "",
                "positive_first_token_frac": sum(value > 0 for value in first_values) / len(first_values) if first_values else "",
                "mean_oriented_rank_gap": _mean(rank_values) if rank_values else "",
                "positive_rank_frac": sum(value > 0 for value in rank_values) / len(rank_values) if rank_values else "",
                "mean_oriented_margin_gap": _mean(margin_values) if margin_values else "",
                "positive_margin_frac": sum(value > 0 for value in margin_values) / len(margin_values) if margin_values else "",
                "source_decoded_to_clean_count": sum(bool(row.get("source_decoded_to_clean")) for row in items),
                "source_decoded_changed_vs_reference_count": sum(bool(row.get("source_decoded_changed_vs_reference")) for row in items),
            }
        )
    return out


def _decide(summary_rows: list[dict[str, Any]], artifacts: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    asset_decisions: dict[str, Any] = {}
    for asset_id, bridge_operator in sorted({(row["asset_id"], row.get("bridge_operator", "")) for row in summary_rows}):
        rows = [row for row in summary_rows if row["asset_id"] == asset_id and row.get("bridge_operator", "") == bridge_operator]
        decoded = any(
            (
                row.get("direction") == "restore"
                and int(row.get("source_decoded_to_clean_count", 0) or 0) > 0
            )
            or (
                row.get("direction") == "corrupt"
                and int(row.get("source_decoded_changed_vs_reference_count", 0) or 0) > 0
            )
            for row in rows
        )
        directional = any(
            (_f(row.get("mean_oriented_sequence_gap")) or 0) > 0
            or (_f(row.get("mean_oriented_first_token_gap")) or 0) > 0
            or (_f(row.get("mean_oriented_rank_gap")) or 0) > 0
            or (_f(row.get("mean_oriented_margin_gap")) or 0) > 0
            for row in rows
        )
        if decoded:
            status = "decoded_bridge_case_supported"
        elif directional:
            status = "decoded_bridge_rank_margin_only"
        else:
            status = "decoded_bridge_not_supported_in_cases"
        asset_decisions[f"{asset_id}:{bridge_operator}"] = {"status": status, "asset_id": asset_id, "bridge_operator": bridge_operator, "directions": rows}
    overall_status = f"decoded_bridge_{mode}_ready" if asset_decisions else "blocked_missing_or_empty_outputs"
    return {
        "updated": _now(),
        "mode": mode,
        "status": overall_status,
        "asset_decisions": asset_decisions,
        "artifacts": artifacts,
        "claim_boundary": (
            "Stage6-022 is illustrative case-level bridge evidence. "
            "Rank/margin/sequence movement is accepted; stable greedy decoded restoration is not required."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Stage6-022 node-to-generation bridge outputs.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--tag", default="defensive_v1")
    args = parser.parse_args()

    rows, artifacts = _load_rows(args.mode, args.tag)
    case_rows = _build_case_rows(rows)
    summary_rows = _summarize(case_rows)
    decision = _decide(summary_rows, artifacts, args.mode)
    stem = _stem(args.mode, args.tag)
    _write_csv(STAGE6_CROSS / f"{stem}_case_table.csv", case_rows)
    _write_csv(STAGE6_CROSS / f"{stem}_summary.csv", summary_rows)
    _write_json(STAGE6_CROSS / f"{stem}_decision.json", decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
