#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"
ROUTE_FIRST_PREFIX = "stage4_qwen_route_first"
OUT_PREFIX = "stage4_qwen_feature_route"
DEFAULT_TOPKS = [4, 8, 16, 32, 64]


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


def _parse_topks(raw: str) -> list[int]:
    if not raw:
        return DEFAULT_TOPKS
    return sorted({int(part.strip()) for part in raw.split(",") if part.strip()})


def _safe_id(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")[:80]


def _node_identity(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("sample_id", ""),
        row.get("prompt_name", ""),
        str(int(_f(row.get("layer"), -999))),
        str(int(_f(row.get("source_pos"), -999))),
        str(int(_f(row.get("source_feature_id"), -999))),
    )


def _route_score(row: dict[str, str]) -> float:
    return (
        _f(row.get("clean_source_minus_controls"))
        + _f(row.get("restore_source_minus_controls"))
        + _f(row.get("real_minus_shifted"))
        + _f(row.get("real_minus_shuffled"))
        + _f(row.get("evidence_specificity"))
        + max(_f(row.get("clean_correct_minus_wrong")), _f(row.get("restore_correct_minus_wrong")))
    )


def _manifest_index(pack: str) -> dict[str, dict[str, str]]:
    path = CROSS / f"{ROUTE_FIRST_PREFIX}_{pack}_manifest.csv"
    return {row.get("candidate_id", ""): row for row in _read_csv(path) if row.get("candidate_id")}


def _route_candidates(source_tag: str) -> list[dict[str, str]]:
    path = CROSS / f"{ROUTE_FIRST_PREFIX}_{source_tag}_route_candidates.csv"
    rows = _read_csv(path)
    if not rows:
        raise FileNotFoundError(f"missing or empty route-first candidate table: {path}")
    return rows


def _merge_node(row: dict[str, str], manifest_index: dict[str, dict[str, str]]) -> dict[str, Any]:
    base = manifest_index.get(row.get("candidate_id", ""), {})
    out: dict[str, Any] = dict(base)
    out.update(row)
    out["layer"] = str(int(_f(out.get("layer"), -999)))
    out["source_pos"] = str(int(_f(out.get("source_pos"), -999)))
    out["source_feature_id"] = str(int(_f(out.get("source_feature_id"), -999)))
    out["source_zeroing_mode"] = out.get("source_zeroing_mode") or "subtract"
    out["route_node_score"] = _route_score(row)
    return out


def _join(values: list[Any]) -> str:
    return "|".join(str(value) for value in values)


def _route_row(
    *,
    pack: str,
    route_idx: int,
    topk: int,
    base_key: tuple[str, str],
    nodes: list[dict[str, Any]],
    primary_node_ids: list[str] | None = None,
) -> dict[str, Any]:
    sample_id, prompt_name = base_key
    route_id = f"qwen_feature_route_{route_idx:05d}_{_safe_id(sample_id)}_{_safe_id(prompt_name)}_top{topk}"
    first = nodes[0] if nodes else {}
    primary_ids = primary_node_ids or [node.get("candidate_id", "") for node in nodes]
    node_ids = [node.get("candidate_id", "") for node in nodes]
    missing = [node_id for node_id in primary_ids if node_id not in set(node_ids)]
    route_score = sum(_f(node.get("route_node_score")) for node in nodes)
    layer_counts = Counter(str(node.get("layer", "")) for node in nodes)
    mask_counts = Counter(str(node.get("best_real_condition", "")) for node in nodes)
    best_real_condition = mask_counts.most_common(1)[0][0] if mask_counts else ""
    return {
        "route_id": route_id,
        "route_base_id": f"{_safe_id(sample_id)}::{_safe_id(prompt_name)}",
        "pack": pack,
        "sample_id": sample_id,
        "prompt_name": prompt_name,
        "topk": topk,
        "requested_topk": topk,
        "route_node_count": len(nodes),
        "primary_node_count": len(primary_ids),
        "strict_missing_count": len(missing),
        "strict_missing_fraction": len(missing) / len(primary_ids) if primary_ids else 0.0,
        "node_candidate_ids": _join(node_ids),
        "primary_node_candidate_ids": _join(primary_ids),
        "missing_candidate_ids": _join(missing),
        "node_layers": _join([node.get("layer", "") for node in nodes]),
        "node_source_positions": _join([node.get("source_pos", "") for node in nodes]),
        "node_feature_ids": _join([node.get("source_feature_id", "") for node in nodes]),
        "node_zeroing_modes": _join([node.get("source_zeroing_mode", "subtract") for node in nodes]),
        "node_scores": _join([f"{_f(node.get('route_node_score')):.8g}" for node in nodes]),
        "route_score": route_score,
        "best_real_condition": best_real_condition,
        "layer_distribution_json": json.dumps(dict(layer_counts), sort_keys=True),
        "target_token_id": first.get("target_token_id", ""),
        "target_token": first.get("target_token", ""),
        "wrong_token_id": first.get("wrong_token_id", ""),
        "wrong_token": first.get("wrong_token", ""),
        "answer_text": first.get("answer_text") or first.get("target_answer", ""),
        "question_text": first.get("question_text", ""),
        "image_filename": first.get("image_filename", ""),
        "local_image_path": first.get("local_image_path", ""),
        "mask_dir": first.get("mask_dir", ""),
        "answer_mask_path": first.get("answer_mask_path", ""),
        "union_mask_path": first.get("union_mask_path", ""),
        "shifted_mask_path": first.get("shifted_mask_path", ""),
        "shuffled_mask_path": first.get("shuffled_mask_path", ""),
        "reasoning_operation": first.get("reasoning_operation", ""),
        "image_dependence_tier": first.get("image_dependence_tier", ""),
        "claim_boundary": "Qwen-native route-first feature route bundle; no Gemma node ids or route maps used.",
    }


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "route_id",
        "route_base_id",
        "pack",
        "sample_id",
        "prompt_name",
        "topk",
        "requested_topk",
        "route_node_count",
        "primary_node_count",
        "strict_missing_count",
        "strict_missing_fraction",
        "node_candidate_ids",
        "primary_node_candidate_ids",
        "missing_candidate_ids",
        "node_layers",
        "node_source_positions",
        "node_feature_ids",
        "node_zeroing_modes",
        "node_scores",
        "route_score",
        "best_real_condition",
        "layer_distribution_json",
        "target_token_id",
        "target_token",
        "wrong_token_id",
        "wrong_token",
        "answer_text",
        "question_text",
        "image_filename",
        "local_image_path",
        "mask_dir",
        "answer_mask_path",
        "union_mask_path",
        "shifted_mask_path",
        "shuffled_mask_path",
        "reasoning_operation",
        "image_dependence_tier",
        "claim_boundary",
    ]
    extras = sorted({key for row in rows for key in row if key not in preferred})
    return [key for key in preferred if any(key in row for row in rows)] + extras


def build(source_tag: str, tag: str, topks: list[int], dry_run: bool = False) -> dict[str, Any]:
    primary_manifest = _manifest_index("primary")
    strict_manifest = _manifest_index("strict")
    route_rows = _route_candidates(source_tag)
    primary_nodes = [
        _merge_node(row, primary_manifest)
        for row in route_rows
        if row.get("pack") == "primary"
        and row.get("mode") == "full"
        and row.get("route_first_234") == "1"
    ]
    strict_node_by_id = {
        row.get("candidate_id", ""): _merge_node(row, strict_manifest)
        for row in route_rows
        if row.get("pack") == "strict" and row.get("mode") == "full" and row.get("candidate_id")
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for node in primary_nodes:
        grouped[(node.get("sample_id", ""), node.get("prompt_name", ""))].append(node)
    primary_routes: list[dict[str, Any]] = []
    strict_routes: list[dict[str, Any]] = []
    route_idx = 0
    for base_key, nodes in sorted(grouped.items()):
        nodes = sorted(nodes, key=lambda node: _f(node.get("route_node_score")), reverse=True)
        for topk in topks:
            selected = nodes[: min(topk, len(nodes))]
            if not selected:
                continue
            route_idx += 1
            primary_ids = [node.get("candidate_id", "") for node in selected]
            primary_routes.append(
                _route_row(pack="primary", route_idx=route_idx, topk=topk, base_key=base_key, nodes=selected)
            )
            strict_nodes = [strict_node_by_id[node_id] for node_id in primary_ids if node_id in strict_node_by_id]
            if strict_nodes:
                strict_routes.append(
                    _route_row(
                        pack="strict",
                        route_idx=route_idx,
                        topk=topk,
                        base_key=base_key,
                        nodes=strict_nodes,
                        primary_node_ids=primary_ids,
                    )
                )

    summary = {
        "updated_at": _now(),
        "source_tag": source_tag,
        "tag": tag,
        "topks": topks,
        "primary_route_count": len(primary_routes),
        "strict_route_count": len(strict_routes),
        "primary_base_route_count": len(grouped),
        "primary_node_count": len(primary_nodes),
        "strict_exact_node_count": len(strict_node_by_id),
        "primary_unique_samples": len({row["sample_id"] for row in primary_routes}),
        "strict_unique_samples": len({row["sample_id"] for row in strict_routes}),
        "primary_topk_counts": dict(Counter(str(row["topk"]) for row in primary_routes)),
        "strict_topk_counts": dict(Counter(str(row["topk"]) for row in strict_routes)),
        "strict_missing_fraction_mean": (
            sum(_f(row["strict_missing_fraction"]) for row in strict_routes) / len(strict_routes)
            if strict_routes
            else 0.0
        ),
    }
    if dry_run:
        summary["dry_run"] = True
        return summary

    primary_out = CROSS / f"{OUT_PREFIX}_primary_{tag}_manifest.csv"
    strict_out = CROSS / f"{OUT_PREFIX}_strict_{tag}_manifest.csv"
    summary_out = CROSS / f"{OUT_PREFIX}_{tag}_manifest_summary.json"
    _write_csv(primary_out, primary_routes, _fieldnames(primary_routes))
    _write_csv(strict_out, strict_routes, _fieldnames(strict_routes))
    summary.update({"primary_manifest": str(primary_out), "strict_manifest": str(strict_out)})
    _write_json(summary_out, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage4-066 Qwen feature-route grouped manifests.")
    parser.add_argument("--source-tag", default="routefirst_v1")
    parser.add_argument("--tag", default="featureroute_v1")
    parser.add_argument("--topks", default="4,8,16,32,64")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = build(args.source_tag, args.tag, _parse_topks(args.topks), args.dry_run)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
