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
CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"
PAPERPACK = ROOT / "doc" / "experiments" / "stage3" / "paperpack72"

DEFAULT_PRIMARY = CROSS / "stage4_qwen_source_tracing_primary_full_expanded_v2_L26_top32_intervention.csv"
DEFAULT_STRICT = CROSS / "stage4_qwen_source_tracing_strict_full_expanded_v2_L26_top32_intervention.csv"
PRIMARY_MANIFEST = PAPERPACK / "paperpack72_primary_manifest.csv"
PRIMARY_RUNS = PAPERPACK / "paperpack72_primary_prompt_runs.csv"
OUT_MANIFEST = CROSS / "stage4_qwen_expanded_cutter_candidate_manifest.csv"
OUT_SUMMARY = CROSS / "stage4_qwen_expanded_cutter_candidate_manifest_summary.json"


SPECIAL_TOKENS = {"", "<|im_end|>", "<|endoftext|>", "<pad>", "<unk>", "</s>", "<s>"}


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


def _f(raw: str | None, default: float = math.nan) -> float:
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _i(raw: str | None, default: int = 0) -> int:
    try:
        return int(float(raw)) if raw not in (None, "") else default
    except ValueError:
        return default


def _is_special(raw: str | None) -> bool:
    token = str(raw or "").strip()
    return token in SPECIAL_TOKENS or token.startswith("<|") or token.startswith("<extra_id_")


def _is_numeric(target_token: str, answer: str) -> bool:
    token = target_token.strip().replace(",", "")
    text = answer.strip().lower().replace(",", "")
    numeric_re = re.compile(r"^[+-]?(?:\d+|\d+\.\d+|\d+/\d+)$")
    word_numbers = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"}
    return bool(numeric_re.match(token) or numeric_re.match(text) or text in word_numbers)


def _damage(row: dict[str, str]) -> float:
    return max(0.0, -_f(row.get("delta_target_logit"), 0.0)) + max(0, _i(row.get("delta_target_rank"), 0)) * 0.25


def _passes(row: dict[str, str], max_rank: int, logit_threshold: float) -> tuple[bool, str]:
    if row.get("status", "ok") not in {"", "ok"}:
        return False, "status_not_ok"
    if _i(row.get("original_target_rank"), 999999) > max_rank:
        return False, "target_rank_too_weak"
    if _f(row.get("delta_target_logit"), 0.0) > -abs(logit_threshold) and _i(row.get("delta_target_rank"), 0) < 1:
        return False, "damage_below_threshold"
    if _is_special(row.get("target_token")):
        return False, "target_token_special_or_empty"
    if _is_special(row.get("original_top1_token")):
        return False, "top1_special"
    if not row.get("sample_id") or not row.get("prompt_name") or not row.get("feature_id") or not row.get("pos"):
        return False, "missing_identity_fields"
    return True, ""


def _strict_indexes(rows: list[dict[str, str]], max_rank: int, logit_threshold: float) -> tuple[dict[tuple[str, str, str, str], dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    exact: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    feature: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        ok, _reason = _passes(row, max_rank, logit_threshold)
        if not ok:
            continue
        payload = {
            "strict_delta_target_logit": row.get("delta_target_logit", ""),
            "strict_delta_target_rank": row.get("delta_target_rank", ""),
            "strict_damage_score": _damage(row),
            "strict_node_id": row.get("node_id", ""),
            "strict_zeroing_mode": row.get("zeroing_mode", ""),
        }
        exact_key = (row.get("sample_id", ""), row.get("prompt_name", ""), row.get("pos", ""), row.get("feature_id", ""))
        feat_key = (row.get("sample_id", ""), row.get("prompt_name", ""), row.get("feature_id", ""))
        if exact_key not in exact or payload["strict_damage_score"] > exact[exact_key]["strict_damage_score"]:
            exact[exact_key] = payload
        if feat_key not in feature or payload["strict_damage_score"] > feature[feat_key]["strict_damage_score"]:
            feature[feat_key] = payload
    return exact, feature


def build(args: argparse.Namespace) -> dict[str, Any]:
    samples = {row["sample_id"]: row for row in _read_csv(PRIMARY_MANIFEST)}
    runs = {(row["sample_id"], row["prompt_name"]): row for row in _read_csv(PRIMARY_RUNS)}
    primary = _read_csv(args.primary_intervention)
    strict = _read_csv(args.strict_intervention) if args.strict_intervention.exists() else []
    strict_exact, strict_feature = _strict_indexes(strict, args.max_rank, args.logit_threshold)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rejection_counts: dict[str, int] = defaultdict(int)
    damaging_features_by_run: dict[tuple[str, str], set[int]] = defaultdict(set)

    for source in primary:
        ok, reason = _passes(source, args.max_rank, args.logit_threshold)
        if not ok:
            rejection_counts[reason] += 1
            continue
        key = (source["sample_id"], source["prompt_name"])
        sample = samples.get(source["sample_id"], {})
        run = runs.get(key, {})
        answer = run.get("answer_text") or sample.get("answer_text", "")
        damaging_features_by_run[key].add(_i(source.get("feature_id"), -1))
        grouped[key].append(
            {
                "sample_id": source["sample_id"],
                "run": source.get("run", ""),
                "prompt_name": source["prompt_name"],
                "source_zeroing_mode": source.get("zeroing_mode", ""),
                "source_node_id": source.get("node_id", ""),
                "layer": source.get("layer", "26"),
                "source_pos": source.get("pos", ""),
                "source_feature_id": source.get("feature_id", ""),
                "path_mass_best": source.get("path_mass_best", ""),
                "target_token_id": source.get("target_token_id", ""),
                "target_token": source.get("target_token", ""),
                "original_target_logit": source.get("original_target_logit", ""),
                "intervened_target_logit": source.get("intervened_target_logit", ""),
                "delta_target_logit_discovery": source.get("delta_target_logit", ""),
                "original_target_rank": source.get("original_target_rank", ""),
                "intervened_target_rank": source.get("intervened_target_rank", ""),
                "delta_target_rank_discovery": source.get("delta_target_rank", ""),
                "original_top1_token": source.get("original_top1_token", ""),
                "intervened_top1_token": source.get("intervened_top1_token", ""),
                "damage_score": _damage(source),
                "analysis_group": "numeric" if _is_numeric(source.get("target_token", ""), answer) else "non_numeric",
                "image_filename": run.get("image_filename") or sample.get("image_filename", ""),
                "local_image_path": run.get("local_image_path") or sample.get("local_image_path", ""),
                "question_text": run.get("question_text") or sample.get("question_text", ""),
                "answer_text": answer,
                "reasoning_operation": run.get("reasoning_operation") or sample.get("reasoning_operation", ""),
                "image_dependence_tier": run.get("image_dependence_tier") or sample.get("image_dependence_tier", ""),
                "paperpack_source": run.get("paperpack_source") or sample.get("paperpack_source", ""),
                "mask_dir": run.get("mask_dir") or sample.get("mask_dir", ""),
                "answer_mask_path": sample.get("answer_mask_path", ""),
                "union_mask_path": sample.get("union_mask_path", ""),
                "shifted_mask_path": sample.get("shifted_mask_path", ""),
                "shuffled_mask_path": sample.get("shuffled_mask_path", ""),
            }
        )

    rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        prompt_rows = sorted(grouped[key], key=lambda row: (float(row["damage_score"]), float(row.get("path_mass_best") or 0.0)), reverse=True)
        for rank, row in enumerate(prompt_rows[: args.pool_per_prompt_run], start=1):
            row["candidate_rank_within_prompt_run"] = rank
            row["include_main"] = "1" if rank <= args.main_per_prompt_run else "0"
            row["include_pool"] = "1"
            exact_key = (row["sample_id"], row["prompt_name"], str(row["source_pos"]), str(row["source_feature_id"]))
            feat_key = (row["sample_id"], row["prompt_name"], str(row["source_feature_id"]))
            strict_payload = strict_exact.get(exact_key) or strict_feature.get(feat_key) or {}
            row["strict_confirmed"] = "1" if strict_payload else "0"
            row["strict_match_level"] = "exact_pos_feature" if exact_key in strict_exact else ("same_feature" if strict_payload else "")
            row["strict_delta_target_logit"] = strict_payload.get("strict_delta_target_logit", "")
            row["strict_delta_target_rank"] = strict_payload.get("strict_delta_target_rank", "")
            row["strict_damage_score"] = strict_payload.get("strict_damage_score", "")
            row["known_damaging_feature_ids_for_prompt_run"] = "|".join(
                str(x) for x in sorted(fid for fid in damaging_features_by_run[key] if fid >= 0)
            )
            rows.append(row)

    rows.sort(key=lambda row: (row["sample_id"], row["prompt_name"], int(row["candidate_rank_within_prompt_run"])))
    for idx, row in enumerate(rows, start=1):
        row["candidate_id"] = f"qwen_expanded_cutter_{idx:04d}"

    fields = [
        "candidate_id",
        "include_main",
        "include_pool",
        "candidate_rank_within_prompt_run",
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
    _write_csv(args.out_manifest, rows, fields)
    main_rows = [row for row in rows if row["include_main"] == "1"]
    payload = {
        "created_at": _now(),
        "primary_intervention": str(args.primary_intervention),
        "strict_intervention": str(args.strict_intervention),
        "out_manifest": str(args.out_manifest),
        "policy": {
            "primary_discovery_only": True,
            "strict_confirmation_only": True,
            "max_rank": args.max_rank,
            "logit_threshold": args.logit_threshold,
            "main_per_prompt_run": args.main_per_prompt_run,
            "pool_per_prompt_run": args.pool_per_prompt_run,
        },
        "counts": {
            "primary_rows": len(primary),
            "strict_rows": len(strict),
            "prompt_runs_with_candidates": len(grouped),
            "pool_candidates": len(rows),
            "main_candidates": len(main_rows),
            "main_numeric": sum(1 for row in main_rows if row["analysis_group"] == "numeric"),
            "main_non_numeric": sum(1 for row in main_rows if row["analysis_group"] == "non_numeric"),
            "strict_confirmed_main": sum(1 for row in main_rows if row["strict_confirmed"] == "1"),
            "rejections": dict(sorted(rejection_counts.items())),
        },
        "claim_boundary": "Expanded Qwen cutter manifest is selected from Qwen primary source-tracing rows only; strict is confirmation only.",
    }
    _write_json(args.out_summary, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage4-016 expanded Qwen cutter manifest from top32 source-tracing rows.")
    parser.add_argument("--primary-intervention", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--strict-intervention", type=Path, default=DEFAULT_STRICT)
    parser.add_argument("--out-manifest", type=Path, default=OUT_MANIFEST)
    parser.add_argument("--out-summary", type=Path, default=OUT_SUMMARY)
    parser.add_argument("--max-rank", type=int, default=10)
    parser.add_argument("--logit-threshold", type=float, default=0.15)
    parser.add_argument("--main-per-prompt-run", type=int, default=2)
    parser.add_argument("--pool-per-prompt-run", type=int, default=16)
    args = parser.parse_args()
    payload = build(args)
    print(json.dumps(payload["counts"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
