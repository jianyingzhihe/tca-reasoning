#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
STAGE3_CROSS = ROOT / "doc" / "experiments" / "stage3" / "cross_model"
STAGE4_CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"
STAGE6_CROSS = ROOT / "doc" / "experiments" / "stage6" / "cross_model"
OUT_PREFIX = "stage6_defensive"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}


def _read_csv_head(path: Path, limit: int = 5) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for idx, row in enumerate(reader):
            if idx >= limit:
                break
            rows.append(row)
        return rows


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


def _artifact(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }


def build_status(tag: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifacts = [
        _artifact(
            STAGE6_CROSS / "stage6_hidden_to_plt_crossmodel_full_decomp_v1_decision.json",
            "gemma_hidden_to_plt_decision",
        ),
        _artifact(
            STAGE6_CROSS / "stage6_hidden_to_plt_crossmodel_full_decomp_v1_gemma_metrics.csv",
            "gemma_hidden_to_plt_gemma_metrics",
        ),
        _artifact(
            STAGE6_CROSS / "stage6_hidden_to_plt_crossmodel_full_decomp_v1_qwen_reference_metrics.csv",
            "qwen_hidden_to_plt_reference_metrics",
        ),
        _artifact(
            STAGE4_CROSS / "stage4_qwen_feature_route_featureroute_v1_decision.json",
            "qwen_grouped_route_decision",
        ),
        _artifact(
            STAGE4_CROSS / "stage4_qwen_feature_route_featureroute_v1_route_metrics.csv",
            "qwen_grouped_route_metrics",
        ),
        _artifact(
            STAGE4_CROSS / "stage4_qwen_route_first_routefirst_v1_route_candidates.csv",
            "qwen_route_first_candidates",
        ),
        _artifact(
            STAGE3_CROSS / "stage3_gemma_source_tracing_primary_full_sample_compare_controlled.csv",
            "gemma_source_tracing_primary_compare",
        ),
        _artifact(
            STAGE3_CROSS / "stage3_gemma_source_tracing_strict_full_sample_compare_controlled.csv",
            "gemma_source_tracing_strict_compare",
        ),
        _artifact(
            STAGE3_CROSS / "stage3_decoded_bridge_decision.json",
            "existing_decoded_bridge_decision",
        ),
    ]

    missing_required = [
        row["role"]
        for row in artifacts
        if row["role"]
        in {
            "gemma_hidden_to_plt_decision",
            "qwen_grouped_route_decision",
            "qwen_route_first_candidates",
        }
        and not row["exists"]
    ]
    status = "ready_for_defensive_smoke" if not missing_required else "blocked_missing_artifact"

    gemma_decision = _read_json(STAGE6_CROSS / "stage6_hidden_to_plt_crossmodel_full_decomp_v1_decision.json")
    qwen_grouped_decision = _read_json(STAGE4_CROSS / "stage4_qwen_feature_route_featureroute_v1_decision.json")

    decision = {
        "tag": tag,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "missing_required": missing_required,
        "gemma_hidden_to_plt_status": gemma_decision.get("status", ""),
        "qwen_grouped_route_status": qwen_grouped_decision.get("status", ""),
        "next_steps": [
            "build Gemma error-heavy vs source-route case panel",
            "run gpu1 Qwen grouped composition smoke",
            "run gpu1 mask robustness smoke",
            "attempt decoded bridge only after cleaner smoke results",
        ],
    }
    return artifacts, decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="defensive_v1")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    artifacts, decision = build_status(args.tag)
    fields = ["role", "exists", "bytes", "path"]

    if args.write:
        stem = f"{OUT_PREFIX}_{args.tag}"
        _write_csv(STAGE6_CROSS / f"{stem}_artifact_status.csv", artifacts, fields)
        _write_json(STAGE6_CROSS / f"{stem}_decision.json", decision)

    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
