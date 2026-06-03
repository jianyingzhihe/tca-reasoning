#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _f(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _i(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw)) if raw not in (None, "") else default
    except ValueError:
        return default


def _evidence_indexes(rows: list[dict[str, str]]) -> tuple[dict[tuple[str, str, str, str], dict[str, str]], dict[tuple[str, str, str], dict[str, str]]]:
    exact: dict[tuple[str, str, str, str], dict[str, str]] = {}
    feature: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        exact_key = (row.get("sample_id", ""), row.get("prompt_name", ""), str(row.get("source_pos", "")), str(row.get("source_feature_id", "")))
        feature_key = (row.get("sample_id", ""), row.get("prompt_name", ""), str(row.get("source_feature_id", "")))
        if exact_key not in exact or _f(row.get("evidence_first_score")) > _f(exact[exact_key].get("evidence_first_score")):
            exact[exact_key] = row
        if feature_key not in feature or _f(row.get("evidence_first_score")) > _f(feature[feature_key].get("evidence_first_score")):
            feature[feature_key] = row
    return exact, feature


def _sample_meta(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    out = {}
    for row in rows:
        out[(row.get("sample_id", ""), row.get("prompt_name", ""))] = row
    return out


def _adapter_score(source: dict[str, str], evidence: dict[str, str]) -> float:
    path_mass = max(0.0, _f(source.get("path_mass_best")))
    evidence_specificity = max(0.0, _f(evidence.get("evidence_specificity")))
    target_contribution = max(0.0, _f(evidence.get("target_contribution")))
    return path_mass * evidence_specificity * target_contribution


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_rows = _read_csv(args.source_tracing_intervention)
    evidence_rows = _read_csv(args.evidence_discovery)
    evidence_exact, evidence_feature = _evidence_indexes(evidence_rows)
    meta_by_run = _sample_meta(evidence_rows)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rejected: dict[str, int] = defaultdict(int)

    for source in source_rows:
        if source.get("status", "ok") not in {"", "ok"}:
            rejected["source_status_not_ok"] += 1
            continue
        if _i(source.get("original_target_rank"), 999999) > args.max_clean_rank:
            rejected["target_rank_too_weak"] += 1
            continue
        sample_id = source.get("sample_id", "")
        prompt_name = source.get("prompt_name", "")
        pos = str(source.get("pos", ""))
        feature_id = str(source.get("feature_id", ""))
        exact_key = (sample_id, prompt_name, pos, feature_id)
        feature_key = (sample_id, prompt_name, feature_id)
        evidence = evidence_exact.get(exact_key)
        match_level = "exact_pos_feature"
        if evidence is None and args.allow_feature_fallback:
            evidence = evidence_feature.get(feature_key)
            match_level = "same_feature"
        if evidence is None:
            rejected["no_evidence_match"] += 1
            continue
        score = _adapter_score(source, evidence)
        if score <= args.min_adapter_score:
            rejected["adapter_score_too_low"] += 1
            continue
        meta = meta_by_run.get((sample_id, prompt_name), evidence)
        row = {
            "candidate_id": "",
            "candidate_source": "adapter_v3_evidence_biased_route",
            "include_main": "0",
            "include_pool": "1",
            "include_sensitivity_rank10": "1",
            "candidate_rank_within_prompt_run": "",
            "analysis_group": evidence.get("analysis_group", ""),
            "sample_id": sample_id,
            "run": source.get("run", ""),
            "prompt_name": prompt_name,
            "source_zeroing_mode": source.get("zeroing_mode", "subtract"),
            "source_node_id": source.get("node_id") or f"F:L{source.get('layer', '26')}:P{pos}:ID{feature_id}",
            "layer": source.get("layer", "26"),
            "source_pos": pos,
            "source_feature_id": feature_id,
            "position_group": evidence.get("position_group", ""),
            "adapter_v3_match_level": match_level,
            "best_real_condition": evidence.get("best_real_condition", ""),
            "path_mass_best": source.get("path_mass_best", ""),
            "target_token_id": source.get("target_token_id", evidence.get("target_token_id", "")),
            "target_token": source.get("target_token", evidence.get("target_token", "")),
            "wrong_token_id": evidence.get("wrong_token_id", ""),
            "wrong_token": evidence.get("wrong_token", ""),
            "original_target_logit": source.get("original_target_logit", evidence.get("original_target_logit", "")),
            "original_target_rank": source.get("original_target_rank", evidence.get("original_target_rank", "")),
            "original_top1_token": source.get("original_top1_token", evidence.get("original_top1_token", "")),
            "clean_activation": evidence.get("clean_activation", ""),
            "answer_mask_drop": evidence.get("answer_mask_drop", ""),
            "union_mask_drop": evidence.get("union_mask_drop", ""),
            "shifted_mask_drop": evidence.get("shifted_mask_drop", ""),
            "shuffled_mask_drop": evidence.get("shuffled_mask_drop", ""),
            "real_drop_best": evidence.get("real_drop_best", ""),
            "control_drop_max": evidence.get("control_drop_max", ""),
            "evidence_specificity": evidence.get("evidence_specificity", ""),
            "target_contribution": evidence.get("target_contribution", ""),
            "wrong_target_contribution": evidence.get("wrong_target_contribution", ""),
            "correct_minus_wrong_contribution": evidence.get("correct_minus_wrong_contribution", ""),
            "evidence_first_score": evidence.get("evidence_first_score", ""),
            "adapter_v3_score": score,
            "damage_score": score,
            "known_damaging_feature_ids_for_prompt_run": "",
            "image_filename": meta.get("image_filename", ""),
            "local_image_path": meta.get("local_image_path", ""),
            "question_text": meta.get("question_text", ""),
            "answer_text": meta.get("answer_text", ""),
            "reasoning_operation": meta.get("reasoning_operation", ""),
            "image_dependence_tier": meta.get("image_dependence_tier", ""),
            "paperpack_source": meta.get("paperpack_source", ""),
            "mask_dir": meta.get("mask_dir", ""),
            "answer_mask_path": meta.get("answer_mask_path", ""),
            "union_mask_path": meta.get("union_mask_path", ""),
            "shifted_mask_path": meta.get("shifted_mask_path", ""),
            "shuffled_mask_path": meta.get("shuffled_mask_path", ""),
        }
        grouped[(sample_id, prompt_name)].append(row)

    rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        candidates = sorted(grouped[key], key=lambda item: float(item["adapter_v3_score"]), reverse=True)
        for rank, row in enumerate(candidates[: args.top_per_prompt_run], start=1):
            row["candidate_rank_within_prompt_run"] = rank
            row["include_main"] = "1" if rank <= args.main_per_prompt_run else "0"
            rows.append(row)
    for idx, row in enumerate(rows, start=1):
        row["candidate_id"] = f"qwen_adapter_v3_{idx:04d}"

    fields = [
        "candidate_id",
        "candidate_source",
        "include_main",
        "include_pool",
        "include_sensitivity_rank10",
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
        "position_group",
        "adapter_v3_match_level",
        "best_real_condition",
        "path_mass_best",
        "target_token_id",
        "target_token",
        "wrong_token_id",
        "wrong_token",
        "original_target_logit",
        "original_target_rank",
        "original_top1_token",
        "clean_activation",
        "answer_mask_drop",
        "union_mask_drop",
        "shifted_mask_drop",
        "shuffled_mask_drop",
        "real_drop_best",
        "control_drop_max",
        "evidence_specificity",
        "target_contribution",
        "wrong_target_contribution",
        "correct_minus_wrong_contribution",
        "evidence_first_score",
        "adapter_v3_score",
        "damage_score",
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
    payload = {
        "created_at": _now(),
        "status": "ok" if rows else "empty",
        "source_tracing_intervention": str(args.source_tracing_intervention),
        "evidence_discovery": str(args.evidence_discovery),
        "out_manifest": str(args.out_manifest),
        "policy": {
            "score": "path_mass_best * positive(evidence_specificity) * positive(target_contribution)",
            "allow_feature_fallback": args.allow_feature_fallback,
            "max_clean_rank": args.max_clean_rank,
            "top_per_prompt_run": args.top_per_prompt_run,
            "main_per_prompt_run": args.main_per_prompt_run,
        },
        "rows": len(rows),
        "main_rows": sum(1 for row in rows if row.get("include_main") == "1"),
        "prompt_runs": len({(row.get("sample_id"), row.get("prompt_name")) for row in rows}),
        "rejected": dict(rejected),
    }
    _write_json(args.summary_json, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an evidence-biased Adapter V3 Qwen route manifest from automatic source-tracing rows.")
    parser.add_argument("--source-tracing-intervention", type=Path, required=True)
    parser.add_argument("--evidence-discovery", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--max-clean-rank", type=int, default=10)
    parser.add_argument("--top-per-prompt-run", type=int, default=4)
    parser.add_argument("--main-per-prompt-run", type=int, default=2)
    parser.add_argument("--min-adapter-score", type=float, default=0.0)
    parser.add_argument("--allow-feature-fallback", action="store_true")
    args = parser.parse_args()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
