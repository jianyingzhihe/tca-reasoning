#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
STAGE4 = ROOT / "doc" / "experiments" / "stage4"
CROSS = STAGE4 / "cross_model"
PAPERPACK = ROOT / "doc" / "experiments" / "stage3" / "paperpack72"

PRIMARY_INTERVENTION = CROSS / "stage4_qwen_source_tracing_primary_full_hookfix_top8_intervention.csv"
STRICT_INTERVENTION = CROSS / "stage4_qwen_source_tracing_strict_full_hookfix_top8_intervention.csv"
PRIMARY_MANIFEST = PAPERPACK / "paperpack72_primary_manifest.csv"
PRIMARY_RUNS = PAPERPACK / "paperpack72_primary_prompt_runs.csv"

OUT_MANIFEST = CROSS / "stage4_qwen_causal_cutter_candidate_manifest.csv"
OUT_SUMMARY = CROSS / "stage4_qwen_causal_cutter_candidate_manifest_summary.json"


SPECIAL_TOKENS = {
    "",
    "<|im_end|>",
    "<|endoftext|>",
    "<pad>",
    "<unk>",
    "</s>",
    "<s>",
}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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


def _as_float(raw: str | None, default: float = math.nan) -> float:
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _as_int(raw: str | None, default: int = 0) -> int:
    try:
        return int(float(raw)) if raw not in (None, "") else default
    except ValueError:
        return default


def _clean_token(raw: str | None) -> str:
    return str(raw or "").strip()


def _is_special_token(raw: str | None) -> bool:
    token = _clean_token(raw)
    return token in SPECIAL_TOKENS or token.startswith("<|") or token.startswith("<extra_id_")


def _is_numeric_answer(target_token: str, answer_text: str) -> bool:
    token = target_token.strip().replace(",", "")
    answer = answer_text.strip().lower().replace(",", "")
    numeric_re = re.compile(r"^[+-]?(?:\d+|\d+\.\d+|\d+/\d+)$")
    word_numbers = {
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
    }
    return bool(numeric_re.match(token) or numeric_re.match(answer) or answer in word_numbers)


def _damage_score(row: dict[str, str]) -> float:
    delta_logit = _as_float(row.get("delta_target_logit"), 0.0)
    delta_rank = _as_int(row.get("delta_target_rank"), 0)
    return max(0.0, -delta_logit) + max(0, delta_rank) * 0.25


def _passes_damage(row: dict[str, str], logit_threshold: float) -> bool:
    return _as_float(row.get("delta_target_logit"), 0.0) <= -abs(logit_threshold) or _as_int(
        row.get("delta_target_rank"), 0
    ) >= 1


def _passes_basic_filters(row: dict[str, str], max_rank: int, logit_threshold: float) -> tuple[bool, str]:
    if row.get("status", "ok") not in {"", "ok"}:
        return False, "status_not_ok"
    if _as_int(row.get("original_target_rank"), 999999) > max_rank:
        return False, "target_rank_too_weak"
    if not _passes_damage(row, logit_threshold):
        return False, "damage_below_threshold"
    if _is_special_token(row.get("target_token")):
        return False, "target_token_special_or_empty"
    if _is_special_token(row.get("original_top1_token")):
        return False, "top1_special"
    if not row.get("sample_id") or not row.get("prompt_name") or not row.get("feature_id") or not row.get("pos"):
        return False, "missing_identity_fields"
    return True, ""


def _best_by_prompt_run(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["sample_id"]), str(row["prompt_name"]))
        if key not in best or float(row["damage_score"]) > float(best[key]["damage_score"]):
            best[key] = row
    return sorted(best.values(), key=lambda r: (float(r["damage_score"]), float(r.get("path_mass_best") or 0.0)), reverse=True)


def _select_main(rows: list[dict[str, Any]], numeric_quota: int, non_numeric_quota: int) -> set[str]:
    selected: set[str] = set()
    counts = {"numeric": 0, "non_numeric": 0}
    for row in rows:
        group = str(row["analysis_group"])
        quota = numeric_quota if group == "numeric" else non_numeric_quota
        if counts[group] >= quota:
            continue
        selected.add(str(row["candidate_id"]))
        counts[group] += 1
    return selected


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    samples = {row["sample_id"]: row for row in _read_csv(PRIMARY_MANIFEST)}
    runs = {(row["sample_id"], row["prompt_name"]): row for row in _read_csv(PRIMARY_RUNS)}
    primary_rows = _read_csv(args.primary_intervention)
    strict_rows = _read_csv(args.strict_intervention)

    rejection_counts: dict[str, int] = defaultdict(int)
    damaging_features_by_run: dict[tuple[str, str], set[int]] = defaultdict(set)
    candidates_rank10: list[dict[str, Any]] = []
    candidates_rank5: list[dict[str, Any]] = []

    for source_row in primary_rows:
        ok10, reason10 = _passes_basic_filters(source_row, args.sensitivity_max_rank, args.logit_threshold)
        if not ok10:
            rejection_counts[reason10] += 1
            continue
        key = (source_row["sample_id"], source_row["prompt_name"])
        damaging_features_by_run[key].add(_as_int(source_row.get("feature_id"), -1))
        damage = _damage_score(source_row)
        sample = samples.get(source_row["sample_id"], {})
        run = runs.get(key, {})
        answer_text = run.get("answer_text") or sample.get("answer_text", "")
        analysis_group = "numeric" if _is_numeric_answer(source_row.get("target_token", ""), answer_text) else "non_numeric"
        base = {
            "sample_id": source_row["sample_id"],
            "run": source_row.get("run", ""),
            "prompt_name": source_row["prompt_name"],
            "source_zeroing_mode": source_row.get("zeroing_mode", ""),
            "source_node_id": source_row.get("node_id", ""),
            "layer": source_row.get("layer", "26"),
            "source_pos": source_row.get("pos", ""),
            "source_feature_id": source_row.get("feature_id", ""),
            "path_mass_best": source_row.get("path_mass_best", ""),
            "target_token_id": source_row.get("target_token_id", ""),
            "target_token": source_row.get("target_token", ""),
            "original_target_logit": source_row.get("original_target_logit", ""),
            "intervened_target_logit": source_row.get("intervened_target_logit", ""),
            "delta_target_logit_discovery": source_row.get("delta_target_logit", ""),
            "original_target_rank": source_row.get("original_target_rank", ""),
            "intervened_target_rank": source_row.get("intervened_target_rank", ""),
            "delta_target_rank_discovery": source_row.get("delta_target_rank", ""),
            "original_top1_token": source_row.get("original_top1_token", ""),
            "intervened_top1_token": source_row.get("intervened_top1_token", ""),
            "damage_score": damage,
            "analysis_group": analysis_group,
            "image_filename": run.get("image_filename") or sample.get("image_filename", ""),
            "local_image_path": run.get("local_image_path") or sample.get("local_image_path", ""),
            "question_text": run.get("question_text") or sample.get("question_text", ""),
            "answer_text": answer_text,
            "reasoning_operation": run.get("reasoning_operation") or sample.get("reasoning_operation", ""),
            "image_dependence_tier": run.get("image_dependence_tier") or sample.get("image_dependence_tier", ""),
            "paperpack_source": run.get("paperpack_source") or sample.get("paperpack_source", ""),
            "mask_dir": run.get("mask_dir") or sample.get("mask_dir", ""),
            "answer_mask_path": sample.get("answer_mask_path", ""),
            "union_mask_path": sample.get("union_mask_path", ""),
            "shifted_mask_path": sample.get("shifted_mask_path", ""),
            "shuffled_mask_path": sample.get("shuffled_mask_path", ""),
        }
        candidates_rank10.append(base)
        if _as_int(source_row.get("original_target_rank"), 999999) <= args.main_max_rank:
            candidates_rank5.append(dict(base))

    strict_confirm_by_exact: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    strict_confirm_by_feature: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in strict_rows:
        ok, _reason = _passes_basic_filters(row, args.sensitivity_max_rank, args.logit_threshold)
        if not ok:
            continue
        exact_key = (row.get("sample_id", ""), row.get("prompt_name", ""), row.get("pos", ""), row.get("feature_id", ""))
        feat_key = (row.get("sample_id", ""), row.get("prompt_name", ""), row.get("feature_id", ""))
        payload = {
            "strict_delta_target_logit": row.get("delta_target_logit", ""),
            "strict_delta_target_rank": row.get("delta_target_rank", ""),
            "strict_damage_score": _damage_score(row),
            "strict_node_id": row.get("node_id", ""),
            "strict_zeroing_mode": row.get("zeroing_mode", ""),
        }
        if exact_key not in strict_confirm_by_exact or payload["strict_damage_score"] > strict_confirm_by_exact[exact_key]["strict_damage_score"]:
            strict_confirm_by_exact[exact_key] = payload
        if feat_key not in strict_confirm_by_feature or payload["strict_damage_score"] > strict_confirm_by_feature[feat_key]["strict_damage_score"]:
            strict_confirm_by_feature[feat_key] = payload

    rank10_best = _best_by_prompt_run(candidates_rank10)
    rank5_best = _best_by_prompt_run(candidates_rank5)
    for idx, row in enumerate(rank10_best, start=1):
        row["candidate_id"] = f"qwen_cutter_{idx:03d}"
        exact_key = (row["sample_id"], row["prompt_name"], str(row["source_pos"]), str(row["source_feature_id"]))
        feat_key = (row["sample_id"], row["prompt_name"], str(row["source_feature_id"]))
        strict = strict_confirm_by_exact.get(exact_key) or strict_confirm_by_feature.get(feat_key) or {}
        row.update(
            {
                "strict_confirmed": "1" if strict else "0",
                "strict_match_level": "exact_pos_feature" if exact_key in strict_confirm_by_exact else ("same_feature" if strict else ""),
                "strict_delta_target_logit": strict.get("strict_delta_target_logit", ""),
                "strict_delta_target_rank": strict.get("strict_delta_target_rank", ""),
                "strict_damage_score": strict.get("strict_damage_score", ""),
                "known_damaging_feature_ids_for_prompt_run": "|".join(
                    str(x)
                    for x in sorted(
                        fid
                        for fid in damaging_features_by_run[(row["sample_id"], row["prompt_name"])]
                        if fid >= 0
                    )
                ),
                "include_sensitivity_rank10": "1",
                "include_main": "0",
            }
        )

    rank5_keys = {(row["sample_id"], row["prompt_name"]) for row in rank5_best}
    main_pool = [row for row in rank10_best if (row["sample_id"], row["prompt_name"]) in rank5_keys]
    main_selected = _select_main(main_pool, args.numeric_quota, args.non_numeric_quota)
    for row in rank10_best:
        if row["candidate_id"] in main_selected:
            row["include_main"] = "1"

    fieldnames = [
        "candidate_id",
        "include_main",
        "include_sensitivity_rank10",
        "analysis_group",
        "sample_id",
        "run",
        "prompt_name",
        "source_zeroing_mode",
        "source_node_id",
        "layer",
        "source_pos",
        "source_feature_id",
        "path_mass_best",
        "target_token_id",
        "target_token",
        "original_target_logit",
        "intervened_target_logit",
        "delta_target_logit_discovery",
        "original_target_rank",
        "intervened_target_rank",
        "delta_target_rank_discovery",
        "original_top1_token",
        "intervened_top1_token",
        "damage_score",
        "strict_confirmed",
        "strict_match_level",
        "strict_delta_target_logit",
        "strict_delta_target_rank",
        "strict_damage_score",
        "known_damaging_feature_ids_for_prompt_run",
        "image_filename",
        "local_image_path",
        "question_text",
        "answer_text",
        "reasoning_operation",
        "image_dependence_tier",
        "paperpack_source",
        "mask_dir",
        "answer_mask_path",
        "union_mask_path",
        "shifted_mask_path",
        "shuffled_mask_path",
    ]
    _write_csv(args.out_manifest, rank10_best, fieldnames)

    main_rows = [row for row in rank10_best if row["include_main"] == "1"]
    payload = {
        "created_at": _now(),
        "primary_intervention": str(args.primary_intervention),
        "strict_intervention": str(args.strict_intervention),
        "out_manifest": str(args.out_manifest),
        "discovery_policy": {
            "source": "primary_full_hookfix_top8_intervention_only",
            "strict_usage": "confirmation_and_sensitivity_only",
            "main_max_rank": args.main_max_rank,
            "sensitivity_max_rank": args.sensitivity_max_rank,
            "logit_threshold": args.logit_threshold,
            "numeric_quota": args.numeric_quota,
            "non_numeric_quota": args.non_numeric_quota,
        },
        "counts": {
            "primary_rows": len(primary_rows),
            "strict_rows": len(strict_rows),
            "rank10_prompt_run_candidates": len(rank10_best),
            "main_candidates": len(main_rows),
            "main_numeric": sum(1 for row in main_rows if row["analysis_group"] == "numeric"),
            "main_non_numeric": sum(1 for row in main_rows if row["analysis_group"] == "non_numeric"),
            "strict_confirmed_main": sum(1 for row in main_rows if row["strict_confirmed"] == "1"),
            "rejections": dict(sorted(rejection_counts.items())),
        },
        "claim_boundary": (
            "Candidate manifest uses Qwen2.5-VL-PLT primary top8 intervention rows only. "
            "It does not import or reuse Gemma node ids, maps, or features."
        ),
    }
    _write_json(args.out_summary, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage4-014 Qwen-native causal cutter candidate manifest.")
    parser.add_argument("--primary-intervention", type=Path, default=PRIMARY_INTERVENTION)
    parser.add_argument("--strict-intervention", type=Path, default=STRICT_INTERVENTION)
    parser.add_argument("--out-manifest", type=Path, default=OUT_MANIFEST)
    parser.add_argument("--out-summary", type=Path, default=OUT_SUMMARY)
    parser.add_argument("--main-max-rank", type=int, default=5)
    parser.add_argument("--sensitivity-max-rank", type=int, default=10)
    parser.add_argument("--logit-threshold", type=float, default=0.25)
    parser.add_argument("--numeric-quota", type=int, default=12)
    parser.add_argument("--non-numeric-quota", type=int, default=12)
    args = parser.parse_args()
    payload = build_manifest(args)
    print(json.dumps(payload["counts"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
