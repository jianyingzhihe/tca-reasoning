#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
STAGE4 = ROOT / "doc" / "experiments" / "stage4"
CROSS = STAGE4 / "cross_model"
PAPERPACK = ROOT / "doc" / "experiments" / "stage3" / "paperpack72"
PRIMARY_RUNS = PAPERPACK / "paperpack72_primary_prompt_runs.csv"
STRICT_RUNS = PAPERPACK / "paperpack72_strict_sensitivity_prompt_runs.csv"

PRIMARY_OUT = CROSS / "stage4_qwen_evidence_specific_nodes_primary_manifest.csv"
STRICT_OUT = CROSS / "stage4_qwen_evidence_specific_nodes_strict_manifest.csv"
SUMMARY_OUT = CROSS / "stage4_qwen_evidence_specific_nodes_manifest_summary.json"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
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


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "")) if row.get(key, "") != "" else default
    except ValueError:
        return default


def _rank(row: dict[str, str]) -> int:
    raw = row.get("clean_target_rank") or row.get("original_target_rank") or "999999"
    try:
        return int(float(raw))
    except ValueError:
        return 999999


def _rank_bucket(rank: int) -> str:
    if rank <= 1:
        return "rank1"
    if rank <= 5:
        return "rank_le5"
    if rank <= 10:
        return "rank_le10"
    return "rank_gt10"


def _mask_paths(run: dict[str, str]) -> dict[str, str]:
    mask_dir = Path(run.get("mask_dir", ""))
    return {
        "answer_mask_path": str(mask_dir / "answer.png") if run.get("mask_dir") else "",
        "union_mask_path": str(mask_dir / "union.png") if run.get("mask_dir") else "",
        "shifted_mask_path": str(mask_dir / "shifted.png") if run.get("mask_dir") else "",
        "shuffled_mask_path": str(mask_dir / "shuffled.png") if run.get("mask_dir") else "",
    }


def _source_files() -> list[Path]:
    files: list[Path] = []
    for layer in range(10, 18):
        files.append(CROSS / f"stage4_qwen_all_layer_bounded_exhaustive_primary_full_L{layer}_candidates.csv")
    files.extend(sorted(CROSS.glob("stage4_qwen_middle_dense_primary_full_L10_*_candidates.csv")))
    return [path for path in files if path.exists() and path.stat().st_size > 0]


def _passes(row: dict[str, str], args: argparse.Namespace) -> tuple[bool, str]:
    if _f(row, "real_drop_best") < args.min_real_drop:
        return False, "real_drop_below_threshold"
    if _f(row, "evidence_specificity") < args.min_specificity:
        return False, "specificity_below_threshold"
    if _f(row, "target_contribution") <= 0:
        return False, "target_contribution_nonpositive"
    if _f(row, "correct_minus_wrong_contribution") <= 0:
        return False, "correct_minus_wrong_nonpositive"
    if _rank(row) > args.max_rank:
        return False, "target_rank_too_weak"
    required = ["sample_id", "prompt_name", "layer", "source_pos", "source_feature_id", "image_filename"]
    missing = [key for key in required if not row.get(key)]
    if missing:
        return False, "missing_identity_fields"
    return True, ""


def _score(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row.get("evidence_specificity") or 0),
        float(row.get("real_drop_best") or 0),
        float(row.get("evidence_first_score") or 0),
        float(row.get("target_contribution") or 0),
    )


def _load_candidates(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Counter[str]]:
    rejections: Counter[str] = Counter()
    best: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for path in _source_files():
        source_name = path.name
        for row in _read_csv(path):
            ok, reason = _passes(row, args)
            if not ok:
                rejections[reason] += 1
                continue
            layer = str(int(float(row["layer"])))
            rank = _rank(row)
            out: dict[str, Any] = dict(row)
            out.update(
                {
                    "candidate_source": "qwen_evidence_specific_discovery",
                    "source_artifact": source_name,
                    "layer": layer,
                    "source_zeroing_mode": row.get("source_zeroing_mode") or "subtract",
                    "source_node_id": row.get("source_node_id") or f"F:L{layer}:P{row.get('source_pos')}:ID{row.get('source_feature_id')}",
                    "clean_target_rank": rank,
                    "rank_bucket": _rank_bucket(rank),
                    "include_pool": "1",
                    "include_main": "0",
                    "include_sensitivity_rank10": "1",
                }
            )
            key = (
                out["sample_id"],
                out["prompt_name"],
                out["layer"],
                out["source_pos"],
                out["source_feature_id"],
            )
            if key not in best or _score(out) > _score(best[key]):
                best[key] = out
    return list(best.values()), rejections


def _identity(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("sample_id", "")),
        str(row.get("prompt_name", "")),
        str(row.get("layer", "")),
        str(row.get("source_pos", "")),
        str(row.get("source_feature_id", "")),
    )


def _select_keys(candidates: list[dict[str, Any]], args: argparse.Namespace) -> set[tuple[str, str, str, str, str]]:
    by_prompt_layer: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_prompt_layer[(row["sample_id"], row["prompt_name"], str(row["layer"]))].append(row)

    capped_prompt_layer: list[dict[str, Any]] = []
    for rows in by_prompt_layer.values():
        capped_prompt_layer.extend(sorted(rows, key=_score, reverse=True)[: args.per_prompt_layer_cap])

    selected: list[dict[str, Any]] = []
    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in capped_prompt_layer:
        by_layer[str(row["layer"])].append(row)
    for layer in [str(x) for x in range(10, 18)]:
        selected.extend(sorted(by_layer[layer], key=_score, reverse=True)[: args.per_layer_cap])

    return {_identity(row) for row in selected}


def _materialize_pack(rows: list[dict[str, Any]], runs_path: Path, pack: str) -> tuple[list[dict[str, Any]], Counter[str]]:
    run_index = {(row["sample_id"], row["prompt_name"]): row for row in _read_csv(runs_path)}
    out: list[dict[str, Any]] = []
    missing: Counter[str] = Counter()
    for row in rows:
        run = run_index.get((row["sample_id"], row["prompt_name"]))
        if not run:
            missing["missing_prompt_run"] += 1
            continue
        merged = dict(row)
        merged.update(
            {
                "pack": pack,
                "image_filename": run.get("image_filename", row.get("image_filename", "")),
                "local_image_path": run.get("local_image_path", row.get("local_image_path", "")),
                "question_text": run.get("question_text", row.get("question_text", "")),
                "answer_text": run.get("answer_text", row.get("answer_text", "")),
                "reasoning_operation": run.get("reasoning_operation", row.get("reasoning_operation", "")),
                "image_dependence_tier": run.get("image_dependence_tier", row.get("image_dependence_tier", "")),
                "paperpack_source": run.get("paperpack_source", row.get("paperpack_source", "")),
                "mask_dir": run.get("mask_dir", row.get("mask_dir", "")),
            }
        )
        merged.update(_mask_paths(merged))
        out.append(merged)
    return out, missing


def _path_checks(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for key in ["local_image_path", "answer_mask_path", "union_mask_path", "shifted_mask_path", "shuffled_mask_path"]:
            if not row.get(key) or not Path(str(row[key])).exists():
                counts[f"missing_{key}"] += 1
    return counts


def build(args: argparse.Namespace) -> dict[str, Any]:
    candidates, rejections = _load_candidates(args)
    selected_keys = _select_keys(candidates, args)
    candidates.sort(key=lambda row: (int(row["layer"]), -_score(row)[0], row["sample_id"], row["prompt_name"], row["source_pos"], row["source_feature_id"]))
    for idx, row in enumerate(candidates, start=1):
        row["candidate_id"] = f"qwen_evidence_specific_{idx:04d}"
        row["candidate_rank_global"] = idx
        row["candidate_rank_within_prompt_run"] = row.get("candidate_rank_within_prompt_run") or ""
        row["include_main"] = "1" if _identity(row) in selected_keys else "0"

    primary, primary_missing = _materialize_pack(candidates, PRIMARY_RUNS, "primary")
    strict, strict_missing = _materialize_pack(candidates, STRICT_RUNS, "strict")

    fields = [
        "candidate_id",
        "candidate_source",
        "source_artifact",
        "pack",
        "include_main",
        "include_pool",
        "include_sensitivity_rank10",
        "candidate_rank_global",
        "candidate_rank_within_prompt_run",
        "analysis_group",
        "rank_bucket",
        "sample_id",
        "run",
        "prompt_name",
        "source_zeroing_mode",
        "source_node_id",
        "layer",
        "source_pos",
        "source_feature_id",
        "position_group",
        "best_real_condition",
        "path_mass_best",
        "target_token_id",
        "target_token",
        "wrong_token_id",
        "wrong_token",
        "original_target_logit",
        "original_target_rank",
        "original_top1_token",
        "clean_target_logit",
        "clean_target_rank",
        "clean_top1_token",
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
        "damage_score",
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
    _write_csv(PRIMARY_OUT, primary, fields)
    _write_csv(STRICT_OUT, strict, fields)

    layer_counts = Counter(str(row["layer"]) for row in primary)
    main_layer_counts = Counter(str(row["layer"]) for row in primary if row.get("include_main") == "1")
    sample_counts = Counter(str(row["sample_id"]) for row in primary)
    main_sample_counts = Counter(str(row["sample_id"]) for row in primary if row.get("include_main") == "1")
    summary = {
        "status": "ok" if primary and strict else "incomplete",
        "updated_at": _now(),
        "source_files": [str(path) for path in _source_files()],
        "pool_candidates_after_filter": len(candidates),
        "primary_pool_rows": len(primary),
        "strict_pool_rows": len(strict),
        "primary_main_rows": sum(1 for row in primary if row.get("include_main") == "1"),
        "strict_main_rows": sum(1 for row in strict if row.get("include_main") == "1"),
        "layer_counts": dict(sorted(layer_counts.items(), key=lambda item: int(item[0]))),
        "main_layer_counts": dict(sorted(main_layer_counts.items(), key=lambda item: int(item[0]))),
        "unique_samples": len(sample_counts),
        "main_unique_samples": len(main_sample_counts),
        "top_sample_counts": sample_counts.most_common(10),
        "main_top_sample_counts": main_sample_counts.most_common(10),
        "rejections": dict(rejections),
        "primary_missing": dict(primary_missing),
        "strict_missing": dict(strict_missing),
        "primary_path_issues": dict(_path_checks(primary)),
        "strict_path_issues": dict(_path_checks(strict)),
        "thresholds": {
            "min_real_drop": args.min_real_drop,
            "min_specificity": args.min_specificity,
            "max_rank": args.max_rank,
            "per_prompt_layer_cap": args.per_prompt_layer_cap,
            "per_layer_cap": args.per_layer_cap,
        },
        "outputs": {
            "primary_manifest": str(PRIMARY_OUT),
            "strict_manifest": str(STRICT_OUT),
        },
    }
    _write_json(SUMMARY_OUT, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage4-052 Qwen evidence-specific node manifests.")
    parser.add_argument("--min-real-drop", type=float, default=20.0)
    parser.add_argument("--min-specificity", type=float, default=20.0)
    parser.add_argument("--max-rank", type=int, default=10)
    parser.add_argument("--per-prompt-layer-cap", type=int, default=2)
    parser.add_argument("--per-layer-cap", type=int, default=20)
    args = parser.parse_args()
    args.per_layer_cap = min(args.per_layer_cap, 24)
    summary = build(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
