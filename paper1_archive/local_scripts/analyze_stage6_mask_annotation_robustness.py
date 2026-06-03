#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
STAGE4_CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"
STAGE6_CROSS = ROOT / "doc" / "experiments" / "stage6" / "cross_model"
PREFIX = "stage4_qwen_route_first"


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


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _tag(variant: str, tag: str) -> str:
    return f"stage6mask_{variant}_{tag}"


def _rows_for(variant: str, tag: str) -> list[dict[str, str]]:
    path = STAGE4_CROSS / f"{PREFIX}_{_tag(variant, tag)}_route_candidates.csv"
    rows = _read_csv(path)
    return [row for row in rows if row.get("pack") == "primary" and row.get("mode") == "smoke"]


def _summary_row(variant: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "variant": variant,
        "n": len(rows),
        "route_first_234_frac": _mean([_f(row.get("route_first_234")) for row in rows]),
        "route_first_evidence_gold_frac": _mean([_f(row.get("route_first_evidence_gold")) for row in rows]),
        "restore_source_minus_controls_mean": _mean([_f(row.get("restore_source_minus_controls")) for row in rows]),
        "real_minus_shifted_mean": _mean([_f(row.get("real_minus_shifted")) for row in rows]),
        "real_minus_shuffled_mean": _mean([_f(row.get("real_minus_shuffled")) for row in rows]),
        "evidence_specificity_mean": _mean([_f(row.get("evidence_specificity")) for row in rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="defensive_v1")
    args = parser.parse_args()

    rows_by_variant = {variant: _rows_for(variant, args.tag) for variant in ["original", "dilate", "erode"]}
    summary = [_summary_row(variant, rows) for variant, rows in rows_by_variant.items()]

    original = next((row for row in summary if row["variant"] == "original"), None)
    dilate = next((row for row in summary if row["variant"] == "dilate"), None)
    erode = next((row for row in summary if row["variant"] == "erode"), None)

    decision = {
        "tag": args.tag,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "mask_robustness_smoke_ready" if all(row["n"] > 0 for row in summary) else "blocked_missing_variant_outputs",
        "original_vs_dilate_route234_delta": (_f(dilate["route_first_234_frac"]) - _f(original["route_first_234_frac"])) if original and dilate else "",
        "original_vs_erode_route234_delta": (_f(erode["route_first_234_frac"]) - _f(original["route_first_234_frac"])) if original and erode else "",
        "original_vs_dilate_specificity_delta": (_f(dilate["evidence_specificity_mean"]) - _f(original["evidence_specificity_mean"])) if original and dilate else "",
        "original_vs_erode_specificity_delta": (_f(erode["evidence_specificity_mean"]) - _f(original["evidence_specificity_mean"])) if original and erode else "",
    }

    stem = f"stage6_mask_robustness_smoke_{args.tag}"
    _write_csv(
        STAGE6_CROSS / f"{stem}_summary.csv",
        summary,
        [
            "variant",
            "n",
            "route_first_234_frac",
            "route_first_evidence_gold_frac",
            "restore_source_minus_controls_mean",
            "real_minus_shifted_mean",
            "real_minus_shuffled_mean",
            "evidence_specificity_mean",
        ],
    )
    _write_json(STAGE6_CROSS / f"{stem}_decision.json", decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
