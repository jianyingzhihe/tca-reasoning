#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(r"E:\Bridging")
STAGE3_CROSS = ROOT / "doc" / "experiments" / "stage3" / "cross_model"

INPUTS = [
    ("qwen2p5vl_plt", STAGE3_CROSS / "stage3_qwen2p5vl_plt_feature_union.csv"),
    ("qwen2p5vl_clt", STAGE3_CROSS / "stage3_qwen2p5vl_clt_feature_union.csv"),
    ("llava15_clt", STAGE3_CROSS / "stage3_llava15_clt_feature_union.csv"),
]

CONTROL_GROUPS = {
    "activation_matched_topk",
    "drop_matched_topk",
    "attribution_matched_mask_insensitive_topk",
    "random_active_topk",
}


def _float(value: str) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except Exception:
        return None


def _effect(row: dict[str, str]) -> float | None:
    if row.get("direction") == "restore":
        return _float(row.get("logit_restore_vs_mask", ""))
    if row.get("direction") == "corrupt":
        return _float(row.get("logit_damage_vs_clean", ""))
    return None


def _summarize_rows(rows: list[dict[str, str]], asset_id: str) -> dict[str, Any]:
    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(dict)
    for row in rows:
        effect = _effect(row)
        if effect is None:
            continue
        key = (
            row.get("sample_id", ""),
            row.get("prompt_name", ""),
            row.get("position_group", ""),
            row.get("direction", ""),
        )
        group = row.get("feature_group", "")
        if group == "evidence_attribution_topk":
            by_key[key]["evidence"] = effect
        elif group in CONTROL_GROUPS:
            by_key[key].setdefault("controls", []).append(effect)

    comparisons: list[dict[str, Any]] = []
    for (sample_id, prompt_name, position_group, direction), values in sorted(by_key.items()):
        controls = values.get("controls", [])
        if "evidence" not in values or not controls:
            continue
        control_mean = mean(controls)
        evidence = float(values["evidence"])
        comparisons.append(
            {
                "asset_id": asset_id,
                "sample_id": sample_id,
                "prompt_name": prompt_name,
                "position_group": position_group,
                "direction": direction,
                "evidence_effect": evidence,
                "control_mean": control_mean,
                "evidence_minus_control": evidence - control_mean,
                "evidence_above_control": evidence > control_mean,
            }
        )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        grouped[(row["position_group"], row["direction"])].append(row)

    slices = []
    for (position_group, direction), items in sorted(grouped.items()):
        diffs = [float(item["evidence_minus_control"]) for item in items]
        effects = [float(item["evidence_effect"]) for item in items]
        controls = [float(item["control_mean"]) for item in items]
        slices.append(
            {
                "asset_id": asset_id,
                "position_group": position_group,
                "direction": direction,
                "comparison_count": len(items),
                "mean_evidence_effect": mean(effects) if effects else "",
                "mean_control_effect": mean(controls) if controls else "",
                "mean_evidence_minus_control": mean(diffs) if diffs else "",
                "positive_count": sum(diff > 0 for diff in diffs),
            }
        )

    return {
        "asset_id": asset_id,
        "raw_row_count": len(rows),
        "prompt_run_count": len({(row.get("sample_id", ""), row.get("prompt_name", "")) for row in rows}),
        "comparison_count": len(comparisons),
        "slices": slices,
        "comparisons": comparisons,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> int:
    summaries = []
    all_slices: list[dict[str, Any]] = []
    all_comparisons: list[dict[str, Any]] = []
    for asset_id, path in INPUTS:
        if not path.exists():
            summaries.append({"asset_id": asset_id, "status": "missing", "path": str(path)})
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        summary = _summarize_rows(rows, asset_id)
        summary["status"] = "ok"
        summaries.append({key: value for key, value in summary.items() if key not in {"slices", "comparisons"}})
        all_slices.extend(summary["slices"])
        all_comparisons.extend(summary["comparisons"])

    payload = {
        "claim_boundary": (
            "Stage3 smoke summary only. It compares attribution-weighted feature patch effects across assets; "
            "it is not full PLT-aligned replication and not Gemma-style source tracing."
        ),
        "summaries": summaries,
    }
    (STAGE3_CROSS / "stage3_smoke_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(
        STAGE3_CROSS / "stage3_smoke_summary_slices.csv",
        all_slices,
        [
            "asset_id",
            "position_group",
            "direction",
            "comparison_count",
            "mean_evidence_effect",
            "mean_control_effect",
            "mean_evidence_minus_control",
            "positive_count",
        ],
    )
    _write_csv(
        STAGE3_CROSS / "stage3_smoke_comparisons.csv",
        all_comparisons,
        [
            "asset_id",
            "sample_id",
            "prompt_name",
            "position_group",
            "direction",
            "evidence_effect",
            "control_mean",
            "evidence_minus_control",
            "evidence_above_control",
        ],
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
