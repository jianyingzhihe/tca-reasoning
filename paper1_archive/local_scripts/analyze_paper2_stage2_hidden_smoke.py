#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(r"E:\Bridging")
PAPER2 = ROOT / "doc" / "experiments" / "paper2"
STAGE2 = PAPER2 / "stage2"

def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except ValueError:
        return math.nan


def _mean(vals: list[float]) -> float:
    clean = [v for v in vals if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else math.nan


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_rows(tag: str) -> list[dict[str, str]]:
    gemma_raw = ROOT / "doc" / "experiments" / "stage6" / "cross_model" / f"stage6_gemma_hidden_lattice_primary_smoke_{tag}_raw.csv"
    qwen_raw = ROOT / "doc" / "experiments" / "stage4" / "cross_model" / f"stage4_qwen_decisive_route_hidden_primary_smoke_{tag}_raw.csv"
    rows: list[dict[str, str]] = []
    for model, path in [("gemma", gemma_raw), ("qwen", qwen_raw)]:
        for row in _read_csv(path):
            row["paper2_model"] = model
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Paper2 Stage2 hidden-lens smoke outputs.")
    parser.add_argument("--tag", default="paper2_stage2_hidden_v1")
    args = parser.parse_args()
    tag = args.tag
    rows = _load_rows(tag)
    target_rows = [r for r in rows if r.get("token_scored") == "target" and r.get("direction") == "restore"]
    wrong_rows = [r for r in rows if r.get("token_scored") == "wrong" and r.get("direction") == "restore"]

    wrong_lookup = {
        (
            r["paper2_model"],
            r["sample_id"],
            r.get("layer", ""),
            r.get("mask_condition", ""),
            r.get("position_group", ""),
        ): _f(r, "logit_effect")
        for r in wrong_rows
    }
    enriched: list[dict[str, object]] = []
    for row in target_rows:
        key = (
            row["paper2_model"],
            row["sample_id"],
            row.get("layer", ""),
            row.get("mask_condition", ""),
            row.get("position_group", ""),
        )
        target_effect = _f(row, "logit_effect")
        wrong_effect = wrong_lookup.get(key, math.nan)
        enriched.append(
            {
                "model": row["paper2_model"],
                "sample_id": row["sample_id"],
                "layer": int(float(row.get("layer", "0"))),
                "mask_condition": row.get("mask_condition", ""),
                "position_group": row.get("position_group", ""),
                "target_effect": target_effect,
                "wrong_effect": wrong_effect,
                "target_minus_wrong_effect": target_effect - wrong_effect if not math.isnan(wrong_effect) else math.nan,
                "rank_effect": _f(row, "rank_effect"),
                "clean_target_rank": _f(row, "clean_target_rank"),
                "mask_target_rank": _f(row, "mask_target_rank"),
                "position_count": _f(row, "position_count"),
            }
        )

    summary: list[dict[str, object]] = []
    for group_key in ["model", "sample_id", "mask_condition", "position_group"]:
        buckets: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
        for row in enriched:
            if group_key == "sample_id":
                key = (row["model"], row["sample_id"])
            else:
                key = (row["model"], row[group_key])
            buckets[key].append(row)
        for key, bucket in sorted(buckets.items()):
            model = str(key[0])
            label = str(key[1])
            summary.append(
                {
                    "group": group_key,
                    "model": model,
                    "label": label,
                    "n": len(bucket),
                    "target_effect_mean": _mean([float(r["target_effect"]) for r in bucket]),
                    "target_positive_frac": _mean([1.0 if float(r["target_effect"]) > 0 else 0.0 for r in bucket]),
                    "target_minus_wrong_mean": _mean([float(r["target_minus_wrong_effect"]) for r in bucket]),
                    "rank_effect_mean": _mean([float(r["rank_effect"]) for r in bucket]),
                }
            )

    specificity: list[dict[str, object]] = []
    by_key: dict[tuple[object, ...], dict[str, float]] = defaultdict(dict)
    for row in enriched:
        key = (row["model"], row["sample_id"], row["layer"], row["position_group"])
        by_key[key][str(row["mask_condition"])] = float(row["target_effect"])
    for (model, sample_id, layer, position_group), effects in sorted(by_key.items()):
        real = _mean([effects.get("answer_mask", math.nan), effects.get("union_mask", math.nan)])
        control = _mean([effects.get("shifted_mask", math.nan), effects.get("shuffled_mask", math.nan)])
        specificity.append(
            {
                "model": model,
                "sample_id": sample_id,
                "layer": layer,
                "position_group": position_group,
                "real_mask_effect_mean": real,
                "control_mask_effect_mean": control,
                "real_minus_control": real - control if not math.isnan(real) and not math.isnan(control) else math.nan,
            }
        )

    decision = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tag": tag,
        "status": "paper2_stage2_hidden_smoke_ready",
        "raw_rows": len(rows),
        "target_restore_rows": len(enriched),
        "models": {},
    }
    for model in ["gemma", "qwen"]:
        mrows = [r for r in enriched if r["model"] == model]
        spec_rows = [r for r in specificity if r["model"] == model]
        decision["models"][model] = {
            "target_rows": len(mrows),
            "samples": sorted({str(r["sample_id"]) for r in mrows}),
            "target_effect_mean": _mean([float(r["target_effect"]) for r in mrows]),
            "target_positive_frac": _mean([1.0 if float(r["target_effect"]) > 0 else 0.0 for r in mrows]),
            "target_minus_wrong_mean": _mean([float(r["target_minus_wrong_effect"]) for r in mrows]),
            "real_minus_control_mean": _mean([float(r["real_minus_control"]) for r in spec_rows]),
        }

    _write_csv(STAGE2 / f"{tag}_target_restore_rows.csv", enriched)
    _write_csv(STAGE2 / f"{tag}_summary.csv", summary)
    _write_csv(STAGE2 / f"{tag}_specificity.csv", specificity)
    (STAGE2 / f"{tag}_decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Paper2 Stage2 Hidden-Lens Smoke",
        "",
        f"Generated: {decision['created_at']}",
        "",
        "## Result",
        "",
        "This smoke uses two common Paper2 behavior-strong cases and applies the same hidden restore lens to Gemma and Qwen. It is a mechanism viability check, not a final Paper2 claim.",
        "",
        "| model | target rows | samples | target effect mean | positive frac | target-minus-wrong mean | real-minus-control mean |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for model, stats in decision["models"].items():
        lines.append(
            "| {model} | {target_rows} | {samples} | {target_effect_mean:.3f} | {target_positive_frac:.3f} | {target_minus_wrong_mean:.3f} | {real_minus_control_mean:.3f} |".format(
                model=model,
                samples=", ".join(stats["samples"]),
                target_rows=stats["target_rows"],
                target_effect_mean=stats["target_effect_mean"],
                target_positive_frac=stats["target_positive_frac"],
                target_minus_wrong_mean=stats["target_minus_wrong_mean"],
                real_minus_control_mean=stats["real_minus_control_mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The input samples are selected by Paper2 Stage1 clean-vs-wrong-image behavior differences.",
            "- The masks for this smoke use Stage2 hidden-lens assets: `answer/union` are rendered from evidence-region union, and `shifted/shuffled` are spatial controls.",
            "- Because Gemma clean target ranks were weaker than the Stage6 mainline gate, this smoke uses `max_clean_rank=1000`; it should be reported as exploratory.",
        ]
    )
    (STAGE2 / f"002_paper2_stage2_hidden_smoke_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
