#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(r"E:\Bridging")
PAPER2 = ROOT / "doc" / "experiments" / "paper2"
STAGE1 = PAPER2 / "stage1"
STAGE2 = PAPER2 / "stage2"
TAG = "paper2_stage2_hidden_expanded_v1"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _f(value: object) -> float:
    try:
        if value in ("", None):
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def _mean(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else math.nan


def main() -> None:
    label_path = STAGE1 / "composition_effect_by_sample_mask_full_v1.csv"
    if not label_path.exists():
        label_path = STAGE1 / "composition_effect_by_sample_mask_v1.csv"
    labels = {(row["model_family"], row["sample_id"]): row for row in _read_csv(label_path)}
    hidden = _read_csv(STAGE2 / f"{TAG}_target_restore_rows.csv")
    joined: list[dict[str, object]] = []
    for row in hidden:
        model = row["model"]
        sample_id = row["sample_id"]
        label_row = labels.get((model, sample_id))
        if label_row is None:
            continue
        joined.append(
            {
                "model": model,
                "sample_id": sample_id,
                "sample_type": label_row.get("sample_type", ""),
                "stage1_label": label_row.get("stage1_label", ""),
                "layer": row.get("layer", ""),
                "mask_condition": row.get("mask_condition", ""),
                "position_group": row.get("position_group", ""),
                "target_effect": _f(row.get("target_effect")),
                "target_minus_wrong_effect": _f(row.get("target_minus_wrong_effect")),
                "rank_effect": _f(row.get("rank_effect")),
                "composition_effect_margin": _f(label_row.get("composition_effect_margin")),
                "composition_effect_rank": _f(label_row.get("composition_effect_rank")),
            }
        )

    summary: list[dict[str, object]] = []
    buckets: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in joined:
        buckets[(str(row["model"]), str(row["stage1_label"]))].append(row)
    for (model, label), rows in sorted(buckets.items()):
        summary.append(
            {
                "model": model,
                "stage1_label": label,
                "target_rows": len(rows),
                "sample_count": len({row["sample_id"] for row in rows}),
                "target_effect_mean": _mean([float(row["target_effect"]) for row in rows]),
                "target_positive_frac": _mean([1.0 if float(row["target_effect"]) > 0 else 0.0 for row in rows]),
                "target_minus_wrong_mean": _mean([float(row["target_minus_wrong_effect"]) for row in rows]),
                "rank_effect_mean": _mean([float(row["rank_effect"]) for row in rows]),
            }
        )

    decision = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "paper2_stage2_composition_label_join_ready",
        "label_path": str(label_path),
        "joined_rows": len(joined),
        "summary_rows": len(summary),
        "interpretation": "Use mask-composition labels as Paper2 sample strata; do not treat all wrong-image-sensitive cases as compositional.",
    }
    _write_csv(STAGE2 / "paper2_stage2_hidden_by_composition_label.csv", summary)
    _write_csv(STAGE2 / "paper2_stage2_hidden_by_composition_label_rows.csv", joined)
    (STAGE2 / "paper2_stage2_hidden_by_composition_label_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
