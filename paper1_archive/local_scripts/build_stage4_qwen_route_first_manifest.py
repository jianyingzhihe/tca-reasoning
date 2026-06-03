#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"
PAPERPACK = ROOT / "doc" / "experiments" / "stage3" / "paperpack72"
PRIMARY_RUNS = PAPERPACK / "paperpack72_primary_prompt_runs.csv"
STRICT_RUNS = PAPERPACK / "paperpack72_strict_sensitivity_prompt_runs.csv"

PRIMARY_OUT = CROSS / "stage4_qwen_route_first_primary_manifest.csv"
STRICT_OUT = CROSS / "stage4_qwen_route_first_strict_manifest.csv"
SUMMARY_OUT = CROSS / "stage4_qwen_route_first_manifest_summary.json"


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


def _f(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _rank(row: dict[str, str]) -> int:
    return int(_f(row.get("clean_target_rank") or row.get("original_target_rank"), 999999))


def _source_files() -> list[Path]:
    files = [CROSS / f"stage4_qwen_all_layer_bounded_exhaustive_primary_full_L{layer}_candidates.csv" for layer in range(10, 18)]
    files.extend(sorted(CROSS.glob("stage4_qwen_middle_dense_primary_full_L10_*_candidates.csv")))
    return [path for path in files if path.exists() and path.stat().st_size > 0]


def _score(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        _f(row.get("damage_score")),
        _f(row.get("target_contribution")),
        _f(row.get("clean_activation")),
        _f(row.get("evidence_first_score")),
    )


def _identity(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("sample_id", "")),
        str(row.get("prompt_name", "")),
        str(int(_f(row.get("layer"), -999))),
        str(row.get("source_pos", "")),
        str(row.get("source_feature_id", "")),
    )


def _mask_paths(run: dict[str, str]) -> dict[str, str]:
    mask_dir = Path(run.get("mask_dir", ""))
    return {
        "answer_mask_path": str(mask_dir / "answer.png") if run.get("mask_dir") else "",
        "union_mask_path": str(mask_dir / "union.png") if run.get("mask_dir") else "",
        "shifted_mask_path": str(mask_dir / "shifted.png") if run.get("mask_dir") else "",
        "shuffled_mask_path": str(mask_dir / "shuffled.png") if run.get("mask_dir") else "",
    }


def _passes_minimal(row: dict[str, str], max_rank: int) -> tuple[bool, str]:
    required = ["sample_id", "prompt_name", "layer", "source_pos", "source_feature_id", "target_token_id", "image_filename"]
    missing = [key for key in required if not row.get(key)]
    if missing:
        return False, "missing_identity_fields"
    if _rank(row) > max_rank:
        return False, "target_rank_too_weak"
    return True, ""


def _load_candidates(max_rank: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    best: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    source_counts: dict[str, dict[str, int]] = {}
    for path in _source_files():
        total = 0
        kept = 0
        for row in _read_csv(path):
            total += 1
            ok, reason = _passes_minimal(row, max_rank)
            if not ok:
                counts[reason] += 1
                continue
            layer = str(int(_f(row["layer"], -999)))
            out: dict[str, Any] = dict(row)
            out.update(
                {
                    "candidate_source": "qwen_route_first_broad_discovery",
                    "source_artifact": path.name,
                    "layer": layer,
                    "source_zeroing_mode": row.get("source_zeroing_mode") or "subtract",
                    "source_node_id": row.get("source_node_id") or f"F:L{layer}:P{row.get('source_pos')}:ID{row.get('source_feature_id')}",
                    "clean_target_rank": _rank(row),
                    "include_pool": "1",
                    "include_main": "1",
                    "include_sensitivity_rank10": "1",
                    "include_frozen": "0",
                }
            )
            key = _identity(out)
            if key not in best or _score(out) > _score(best[key]):
                best[key] = out
            kept += 1
        source_counts[path.name] = {"rows": total, "kept_minimal": kept}
    rows = list(best.values())
    rows.sort(key=lambda row: (int(row["layer"]), row["sample_id"], row["prompt_name"], int(_f(row["source_pos"])), int(_f(row["source_feature_id"]))))
    for idx, row in enumerate(rows, start=1):
        row["candidate_id"] = f"qwen_route_first_{idx:06d}"
        row["candidate_rank_global"] = idx
    return rows, {"source_counts": source_counts, "rejections": dict(counts)}


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


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "candidate_id",
        "candidate_rank_global",
        "candidate_source",
        "source_artifact",
        "include_pool",
        "include_main",
        "include_sensitivity_rank10",
        "include_frozen",
        "pack",
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
        "best_real_condition",
        "target_token_id",
        "target_token",
        "wrong_token_id",
        "wrong_token",
        "clean_target_logit",
        "clean_target_rank",
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
    extras = sorted({key for row in rows for key in row if key not in preferred})
    return [key for key in preferred if any(key in row for row in rows)] + extras


def _build_base(args: argparse.Namespace) -> dict[str, Any]:
    candidates, load_summary = _load_candidates(args.max_rank)
    primary, primary_missing = _materialize_pack(candidates, PRIMARY_RUNS, "primary")
    strict, strict_missing = _materialize_pack(candidates, STRICT_RUNS, "strict")
    primary_path_issues = _path_checks(primary)
    strict_path_issues = _path_checks(strict)
    if primary_path_issues:
        primary = [row for row in primary if all(row.get(key) and Path(str(row[key])).exists() for key in ["local_image_path", "answer_mask_path", "union_mask_path", "shifted_mask_path", "shuffled_mask_path"])]
    if strict_path_issues:
        strict = [row for row in strict if all(row.get(key) and Path(str(row[key])).exists() for key in ["local_image_path", "answer_mask_path", "union_mask_path", "shifted_mask_path", "shuffled_mask_path"])]
    _write_csv(PRIMARY_OUT, primary, _fieldnames(primary))
    _write_csv(STRICT_OUT, strict, _fieldnames(strict))
    summary = {
        "updated_at": _now(),
        "max_rank": args.max_rank,
        "unique_base_candidates": len(candidates),
        "primary_rows": len(primary),
        "strict_rows": len(strict),
        "primary_unique_samples": len({row["sample_id"] for row in primary}),
        "strict_unique_samples": len({row["sample_id"] for row in strict}),
        "primary_layer_counts": dict(Counter(str(row["layer"]) for row in primary)),
        "strict_layer_counts": dict(Counter(str(row["layer"]) for row in strict)),
        "primary_missing": dict(primary_missing),
        "strict_missing": dict(strict_missing),
        "primary_path_issues": dict(primary_path_issues),
        "strict_path_issues": dict(strict_path_issues),
        **load_summary,
    }
    _write_json(SUMMARY_OUT, summary)
    return summary


def _build_frozen(tag: str) -> dict[str, Any]:
    route_path = CROSS / f"stage4_qwen_route_first_{tag}_route_candidates.csv"
    if not route_path.exists():
        raise FileNotFoundError(f"route candidate table not found: {route_path}")
    selected = [row for row in _read_csv(route_path) if row.get("pack") == "primary" and row.get("mode") == "full" and row.get("route_first_234") == "1"]
    selected_ids = {row["candidate_id"] for row in selected}
    strict_rows = _read_csv(STRICT_OUT)
    frozen = []
    for row in strict_rows:
        if row.get("candidate_id") in selected_ids:
            out = dict(row)
            out["include_frozen"] = "1"
            frozen.append(out)
    out_path = CROSS / f"stage4_qwen_route_first_strict_frozen_{tag}_manifest.csv"
    if frozen:
        _write_csv(out_path, frozen, _fieldnames(frozen))
    else:
        _write_csv(out_path, [], ["candidate_id"])
    summary = {
        "updated_at": _now(),
        "tag": tag,
        "primary_route_first_rows": len(selected),
        "primary_route_first_unique_candidates": len(selected_ids),
        "strict_frozen_rows": len(frozen),
        "strict_frozen_unique_samples": len({row.get("sample_id", "") for row in frozen}),
        "strict_frozen_layers": dict(Counter(str(row.get("layer", "")) for row in frozen)),
        "strict_frozen_manifest": str(out_path),
        "missing_in_strict": len(selected_ids - {row.get("candidate_id", "") for row in frozen}),
    }
    _write_json(CROSS / f"stage4_qwen_route_first_strict_frozen_{tag}_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage4-060 Qwen route-first manifests.")
    parser.add_argument("--max-rank", type=int, default=10)
    parser.add_argument("--strict-from-primary", default="")
    args = parser.parse_args()
    if args.strict_from_primary:
        if not STRICT_OUT.exists():
            _build_base(args)
        summary = _build_frozen(args.strict_from_primary)
    else:
        summary = _build_base(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
