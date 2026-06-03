#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


SOURCE_GROUPS = {
    "evidence_region",
    "top_hidden_delta",
    "evidence_region_plus_answer_adjacent",
    "top_hidden_delta_plus_answer_adjacent",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
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
        return float("nan")


def _mean(values: list[float]) -> float:
    vals = [value for value in values if not math.isnan(value)]
    return sum(vals) / len(vals) if vals else float("nan")


def _round(value: float) -> float | str:
    return "" if math.isnan(value) else round(value, 6)


def _prepare(model: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for raw in rows:
        row: dict[str, Any] = {
            **raw,
            "model_family": raw.get("model_family") or model,
            "effect_logit_f": _float(raw.get("effect_logit", "nan")),
            "effect_rank_f": _float(raw.get("effect_rank", "nan")),
            "effect_gap_closure_f": _float(raw.get("effect_gap_closure", "nan")),
            "clean_union_logit_gap_f": _float(raw.get("clean_union_logit_gap", "nan")),
            "clean_union_rank_gap_f": _float(raw.get("clean_union_rank_gap", "nan")),
            "top1_changed_bool": str(raw.get("top1_changed_vs_reference", "")).lower() == "true",
        }
        out.append(row)
    return out


def _baseline_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["condition"] == "union_mask":
            grouped[(row["model_family"], row["bucket"])].append(row)
    out = []
    for (model, bucket), group in sorted(grouped.items()):
        logit_gaps = [row["clean_union_logit_gap_f"] for row in group]
        rank_gaps = [row["clean_union_rank_gap_f"] for row in group]
        out.append(
            {
                "model_family": model,
                "bucket": bucket,
                "n_runs": len(group),
                "mean_clean_union_logit_gap": _round(_mean(logit_gaps)),
                "positive_logit_gap_n": sum(value > 0 for value in logit_gaps if not math.isnan(value)),
                "mean_clean_union_rank_gap": _round(_mean(rank_gaps)),
                "positive_rank_gap_n": sum(value > 0 for value in rank_gaps if not math.isnan(value)),
            }
        )
    return out


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["direction"] not in {"restore", "corrupt"}:
            continue
        grouped[
            (
                row["model_family"],
                row["layer"],
                row["bucket"],
                row["direction"],
                row["group_name"],
                row["group_kind"],
                row["scale"],
            )
        ].append(row)
    out = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        model, layer, bucket, direction, group_name, group_kind, scale = key
        effect_logit = [row["effect_logit_f"] for row in group]
        effect_rank = [row["effect_rank_f"] for row in group]
        effect_gap = [row["effect_gap_closure_f"] for row in group]
        out.append(
            {
                "model_family": model,
                "layer": layer,
                "bucket": bucket,
                "direction": direction,
                "group_name": group_name,
                "group_kind": group_kind,
                "scale": scale,
                "n_rows": len(group),
                "n_samples": len({row["sample_id"] for row in group}),
                "n_prompts": len({row["prompt_name"] for row in group}),
                "mean_effect_logit": _round(_mean(effect_logit)),
                "positive_effect_logit_n": sum(value > 0 for value in effect_logit if not math.isnan(value)),
                "mean_effect_rank": _round(_mean(effect_rank)),
                "positive_effect_rank_n": sum(value > 0 for value in effect_rank if not math.isnan(value)),
                "mean_effect_gap_closure": _round(_mean(effect_gap)),
                "top1_changed_n": sum(row["top1_changed_bool"] for row in group),
                "mean_position_count": _round(_mean([_float(row["position_count"]) for row in group])),
            }
        )
    return out


def _random_means(summary: list[dict[str, Any]]) -> dict[tuple[str, str, str, str, str], dict[str, float]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary:
        if str(row["group_name"]).startswith("random_control_"):
            grouped[(row["model_family"], row["layer"], row["bucket"], row["direction"], row["scale"])].append(row)
    out = {}
    for key, group in grouped.items():
        out[key] = {
            "mean_effect_logit": _mean([_float(str(row["mean_effect_logit"])) for row in group]),
            "mean_effect_rank": _mean([_float(str(row["mean_effect_rank"])) for row in group]),
            "mean_effect_gap_closure": _mean([_float(str(row["mean_effect_gap_closure"])) for row in group]),
            "positive_effect_logit_n": _mean([_float(str(row["positive_effect_logit_n"])) for row in group]),
            "positive_effect_rank_n": _mean([_float(str(row["positive_effect_rank_n"])) for row in group]),
        }
    return out


def _specificity(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    random = _random_means(summary)
    by_key = {
        (row["model_family"], row["layer"], row["bucket"], row["direction"], row["scale"], row["group_name"]): row
        for row in summary
    }
    out = []
    for row in summary:
        if row["group_name"] not in SOURCE_GROUPS:
            continue
        key = (row["model_family"], row["layer"], row["bucket"], row["direction"], row["scale"])
        random_row = random.get(key)
        low_delta = by_key.get((*key, "low_delta_control"))
        if random_row is None:
            continue
        source_logit = _float(str(row["mean_effect_logit"]))
        source_rank = _float(str(row["mean_effect_rank"]))
        source_gap = _float(str(row["mean_effect_gap_closure"]))
        low_logit = _float(str(low_delta["mean_effect_logit"])) if low_delta else float("nan")
        low_rank = _float(str(low_delta["mean_effect_rank"])) if low_delta else float("nan")
        low_gap = _float(str(low_delta["mean_effect_gap_closure"])) if low_delta else float("nan")
        out.append(
            {
                "model_family": row["model_family"],
                "layer": row["layer"],
                "bucket": row["bucket"],
                "direction": row["direction"],
                "scale": row["scale"],
                "source_group": row["group_name"],
                "source_kind": row["group_kind"],
                "source_mean_effect_logit": row["mean_effect_logit"],
                "random_mean_effect_logit": _round(random_row["mean_effect_logit"]),
                "source_minus_random_logit": _round(source_logit - random_row["mean_effect_logit"]),
                "low_delta_mean_effect_logit": _round(low_logit),
                "source_minus_low_delta_logit": _round(source_logit - low_logit),
                "source_mean_effect_rank": row["mean_effect_rank"],
                "random_mean_effect_rank": _round(random_row["mean_effect_rank"]),
                "source_minus_random_rank": _round(source_rank - random_row["mean_effect_rank"]),
                "low_delta_mean_effect_rank": _round(low_rank),
                "source_minus_low_delta_rank": _round(source_rank - low_rank),
                "source_mean_gap_closure": row["mean_effect_gap_closure"],
                "random_mean_gap_closure": _round(random_row["mean_effect_gap_closure"]),
                "source_minus_random_gap_closure": _round(source_gap - random_row["mean_effect_gap_closure"]),
                "low_delta_mean_gap_closure": _round(low_gap),
                "source_minus_low_delta_gap_closure": _round(source_gap - low_gap),
                "source_positive_logit_n": row["positive_effect_logit_n"],
                "source_positive_rank_n": row["positive_effect_rank_n"],
            }
        )
    return sorted(out, key=lambda item: (item["model_family"], item["source_group"], item["direction"]))


def _case_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep_groups = {
        "clean",
        "union_mask",
        "whole_bucket",
        "top_hidden_delta",
        "evidence_region",
        "low_delta_control",
        "delta_matched_control",
        "activation_matched_control",
        "answer_adjacent_text",
        "top_hidden_delta_plus_answer_adjacent",
        "delta_matched_plus_answer_adjacent",
        "activation_matched_plus_answer_adjacent",
        "evidence_region_plus_answer_adjacent",
    }
    out = []
    for row in rows:
        if row["group_name"] not in keep_groups:
            continue
        if row["direction"] in {"restore", "corrupt"} and row["scale"] not in {"1.0", "1"}:
            continue
        out.append(
            {
                "model_family": row["model_family"],
                "sample_id": row["sample_id"],
                "prompt_name": row["prompt_name"],
                "direction": row["direction"],
                "group_name": row["group_name"],
                "group_kind": row["group_kind"],
                "position_count": row["position_count"],
                "target_answer": row["target_answer"],
                "target_rank": row["target_rank"],
                "target_logit": row["target_logit"],
                "clean_target_rank": row["clean_target_rank"],
                "union_target_rank": row["union_target_rank"],
                "effect_logit": row["effect_logit"],
                "effect_rank": row["effect_rank"],
                "effect_gap_closure": row["effect_gap_closure"],
                "top1_token": row["top1_token"],
            }
        )
    return sorted(out, key=lambda item: (item["model_family"], item["sample_id"], item["prompt_name"], item["direction"], item["group_name"]))


def _decision_for_model(model: str, summary: list[dict[str, Any]], specificity: list[dict[str, Any]]) -> dict[str, Any]:
    model_specs = [row for row in specificity if row["model_family"] == model and row["scale"] in {"1.0", "1"}]
    best_restore = max(
        [row for row in model_specs if row["direction"] == "restore"],
        key=lambda row: (_float(str(row["source_minus_random_logit"])), _float(str(row["source_mean_effect_logit"]))),
        default=None,
    )
    best_corrupt = None
    if best_restore is not None:
        best_corrupt = next(
            (
                row
                for row in model_specs
                if row["direction"] == "corrupt" and row["source_group"] == best_restore["source_group"]
            ),
            None,
        )
    if (
        best_restore is not None
        and best_corrupt is not None
        and _float(str(best_restore["source_minus_random_logit"])) > 0
        and _float(str(best_corrupt["source_minus_random_logit"])) > 0
        and int(float(best_restore["source_positive_logit_n"])) >= 4
        and int(float(best_corrupt["source_positive_logit_n"])) >= 4
    ):
        status = "supported_bidirectional_hidden_position_localization"
    elif (
        best_restore is not None
        and _float(str(best_restore["source_minus_random_logit"])) > 0
        and int(float(best_restore["source_positive_logit_n"])) >= 4
    ):
        status = "partial_restore_hidden_position_localization"
    else:
        status = "not_supported_hidden_position_localization"
    return {
        "model_family": model,
        "best_restore_source_group": best_restore["source_group"] if best_restore else "",
        "best_restore_source_minus_random_logit": best_restore["source_minus_random_logit"] if best_restore else "",
        "best_restore_source_minus_random_rank": best_restore["source_minus_random_rank"] if best_restore else "",
        "best_restore_positive_logit_n": best_restore["source_positive_logit_n"] if best_restore else "",
        "matched_corrupt_source_minus_random_logit": best_corrupt["source_minus_random_logit"] if best_corrupt else "",
        "matched_corrupt_source_minus_random_rank": best_corrupt["source_minus_random_rank"] if best_corrupt else "",
        "matched_corrupt_positive_logit_n": best_corrupt["source_positive_logit_n"] if best_corrupt else "",
        "decision_status": status,
        "claim_boundary": "Hidden-position localization only; not CLT feature/source-control route replication.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage 2H hidden-position patch smoke.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--qwen-csv", default="stage2h_qwen_hidden_position_patch.csv")
    parser.add_argument("--llava-csv", default="stage2h_llava_hidden_position_patch.csv")
    parser.add_argument("--out-baseline-csv", required=True)
    parser.add_argument("--out-summary-csv", required=True)
    parser.add_argument("--out-specificity-csv", required=True)
    parser.add_argument("--out-case-csv", required=True)
    parser.add_argument("--out-decision-json", required=True)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    all_rows = []
    for model, filename in {
        "qwen": args.qwen_csv,
        "llava": args.llava_csv,
    }.items():
        path = artifact_dir / filename
        if path.exists():
            all_rows.extend(_prepare(model, _read_csv(path)))

    baseline = _baseline_summary(all_rows)
    summary = _summary(all_rows)
    specificity = _specificity(summary)
    cases = _case_table(all_rows)
    decisions = {
        "evidence_ladder": {
            "readout_replication": "supported by Stage 2F for Qwen/LLaVA",
            "whole_bucket_hidden_patch": "supported by Stage 2G-6",
            "hidden_position_localization": "evaluated here",
            "source_control_causal_replication": "not established cross-model; Gemma3 remains full-chain model",
        },
        "model_decisions": {
            model: _decision_for_model(model, summary, specificity)
            for model in sorted({row["model_family"] for row in all_rows})
        },
    }

    baseline_fields = [
        "model_family",
        "bucket",
        "n_runs",
        "mean_clean_union_logit_gap",
        "positive_logit_gap_n",
        "mean_clean_union_rank_gap",
        "positive_rank_gap_n",
    ]
    summary_fields = [
        "model_family",
        "layer",
        "bucket",
        "direction",
        "group_name",
        "group_kind",
        "scale",
        "n_rows",
        "n_samples",
        "n_prompts",
        "mean_effect_logit",
        "positive_effect_logit_n",
        "mean_effect_rank",
        "positive_effect_rank_n",
        "mean_effect_gap_closure",
        "top1_changed_n",
        "mean_position_count",
    ]
    specificity_fields = [
        "model_family",
        "layer",
        "bucket",
        "direction",
        "scale",
        "source_group",
        "source_kind",
        "source_mean_effect_logit",
        "random_mean_effect_logit",
        "source_minus_random_logit",
        "low_delta_mean_effect_logit",
        "source_minus_low_delta_logit",
        "source_mean_effect_rank",
        "random_mean_effect_rank",
        "source_minus_random_rank",
        "low_delta_mean_effect_rank",
        "source_minus_low_delta_rank",
        "source_mean_gap_closure",
        "random_mean_gap_closure",
        "source_minus_random_gap_closure",
        "low_delta_mean_gap_closure",
        "source_minus_low_delta_gap_closure",
        "source_positive_logit_n",
        "source_positive_rank_n",
    ]
    case_fields = [
        "model_family",
        "sample_id",
        "prompt_name",
        "direction",
        "group_name",
        "group_kind",
        "position_count",
        "target_answer",
        "target_rank",
        "target_logit",
        "clean_target_rank",
        "union_target_rank",
        "effect_logit",
        "effect_rank",
        "effect_gap_closure",
        "top1_token",
    ]
    _write_csv(Path(args.out_baseline_csv), baseline, baseline_fields)
    _write_csv(Path(args.out_summary_csv), summary, summary_fields)
    _write_csv(Path(args.out_specificity_csv), specificity, specificity_fields)
    _write_csv(Path(args.out_case_csv), cases, case_fields)
    _write_json(Path(args.out_decision_json), decisions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
