#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


EVIDENCE_GROUP = "evidence_attribution_topk"
CONTROL_GROUPS = [
    "activation_matched_topk",
    "drop_matched_topk",
    "attribution_matched_mask_insensitive_topk",
    "random_active_topk",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    match = re.search(r"layer(\d+)_top(\d+)", path.name)
    layer = match.group(1) if match else ""
    topk = match.group(2) if match else ""
    for row in rows:
        row["sweep_layer"] = row.get("layer", "") or layer
        row["top_k"] = topk or str(len([x for x in row.get("feature_ids", "").split("|") if x]))
        row["source_file"] = path.name
    return rows


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


def _float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _mean(values: list[float]) -> float | str:
    if not values:
        return ""
    return round(mean(values), 6)


def _bootstrap(values: list[float], seed: int = 1729, n_boot: int = 2000) -> tuple[float | str, float | str, str]:
    if not values:
        return "", "", "missing"
    if len(values) == 1:
        value = round(values[0], 6)
        return value, value, "single_row_positive" if values[0] > 0 else "not_positive"
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _idx in range(len(values))]
        means.append(mean(sample))
    means.sort()
    low = means[int(0.025 * (len(means) - 1))]
    high = means[int(0.975 * (len(means) - 1))]
    status = "stable_positive" if low > 0 else ("weak_or_heterogeneous_positive" if mean(values) > 0 else "not_positive")
    return round(low, 6), round(high, 6), status


def _group(rows: list[dict[str, str]], keys: list[str]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    out: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[tuple(row.get(key, "") for key in keys)].append(row)
    return out


def _effect(row: dict[str, str]) -> float | None:
    if row.get("direction") == "restore":
        return _float(row.get("logit_restore_vs_mask"))
    if row.get("direction") == "corrupt":
        return _float(row.get("logit_damage_vs_clean"))
    return None


def _rank_effect(row: dict[str, str]) -> float | None:
    if row.get("direction") == "restore":
        return _float(row.get("rank_restore_vs_mask"))
    if row.get("direction") == "corrupt":
        return _float(row.get("rank_damage_vs_clean"))
    return None


def _summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = ["sweep_layer", "top_k", "position_group", "direction", "feature_group"]
    for key, items in sorted(_group(rows, keys).items()):
        layer, top_k, position_group, direction, feature_group = key
        effects = [value for value in (_effect(row) for row in items) if value is not None]
        ranks = [value for value in (_rank_effect(row) for row in items) if value is not None]
        ci_low, ci_high, status = _bootstrap(effects)
        out.append(
            {
                "layer": layer,
                "top_k": top_k,
                "position_group": position_group,
                "direction": direction,
                "feature_group": feature_group,
                "n_rows": len(items),
                "positive_logit_n": sum(1 for value in effects if value > 0),
                "positive_rank_n": sum(1 for value in ranks if value > 0),
                "mean_logit_effect": _mean(effects),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "effect_status": status,
                "mean_rank_effect": _mean(ranks),
                "mean_gap_closure": _mean([value for value in (_float(row.get("gap_closure")) for row in items) if value is not None]),
            }
        )
    return out


def _specificity(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = ["sample_id", "prompt_name", "sweep_layer", "top_k", "position_group", "direction"]
    for key, items in sorted(_group(rows, keys).items()):
        sample_id, prompt_name, layer, top_k, position_group, direction = key
        by_group = {row.get("feature_group", ""): row for row in items}
        evidence = by_group.get(EVIDENCE_GROUP)
        if evidence is None:
            continue
        evidence_effect = _effect(evidence)
        evidence_rank = _rank_effect(evidence)
        row_out: dict[str, Any] = {
            "sample_id": sample_id,
            "prompt_name": prompt_name,
            "layer": layer,
            "top_k": top_k,
            "position_group": position_group,
            "direction": direction,
            "target_answer": evidence.get("target_answer", ""),
            "evidence_logit_effect": evidence_effect,
            "evidence_rank_effect": evidence_rank,
        }
        positive_controls = 0
        for control in CONTROL_GROUPS:
            control_row = by_group.get(control)
            control_effect = _effect(control_row) if control_row else None
            control_rank = _rank_effect(control_row) if control_row else None
            logit_gap = "" if evidence_effect is None or control_effect is None else evidence_effect - control_effect
            rank_gap = "" if evidence_rank is None or control_rank is None else evidence_rank - control_rank
            row_out[f"evidence_minus_{control}_logit"] = logit_gap
            row_out[f"evidence_minus_{control}_rank"] = rank_gap
            if isinstance(logit_gap, float) and logit_gap > 0:
                positive_controls += 1
        row_out["positive_control_count"] = positive_controls
        row_out["above_all_controls"] = positive_controls == len(CONTROL_GROUPS)
        out.append(row_out)
    return out


def _decision(summary_rows: list[dict[str, Any]], specificity_rows: list[dict[str, Any]], expected_layers: list[str]) -> dict[str, Any]:
    configs: list[dict[str, Any]] = []
    for row in summary_rows:
        if row["feature_group"] != EVIDENCE_GROUP or row["position_group"] != "top_hidden_delta_plus_answer_adjacent":
            continue
        spec_rows = [
            item
            for item in specificity_rows
            if item["layer"] == row["layer"]
            and item["top_k"] == row["top_k"]
            and item["position_group"] == row["position_group"]
            and item["direction"] == row["direction"]
        ]
        above = [item for item in spec_rows if item.get("above_all_controls") is True or item.get("above_all_controls") == "True"]
        configs.append(
            {
                "layer": row["layer"],
                "top_k": row["top_k"],
                "direction": row["direction"],
                "n_rows": row["n_rows"],
                "mean_logit_effect": row["mean_logit_effect"],
                "effect_status": row["effect_status"],
                "above_all_controls_n": len(above),
                "specificity_n": len(spec_rows),
                "mean_positive_control_count": _mean([int(item.get("positive_control_count", 0)) for item in spec_rows]),
            }
        )
    strong = [
        item
        for item in configs
        if item["effect_status"] == "stable_positive"
        and item["specificity_n"]
        and item["above_all_controls_n"] >= max(1, int(0.6 * item["specificity_n"]))
    ]
    weak = [
        item
        for item in configs
        if item["effect_status"] in {"stable_positive", "weak_or_heterogeneous_positive"}
        and item["specificity_n"]
        and item["above_all_controls_n"] > 0
    ]
    observed_layers = sorted({row["layer"] for row in summary_rows})
    missing_layers = [layer for layer in expected_layers if layer not in observed_layers]
    if strong:
        status = "llava_layer_sensitive_feature_bridge_supported"
    elif weak:
        status = "llava_layer_sweep_weak_or_partial"
    else:
        status = "llava_feature_bridge_not_established"
    return {
        "claim_boundary": (
            "Stage 2P-2 diagnoses LLaVA feature-level localization across layers/top-k. "
            "Failure does not prove absence of cross-modal features."
        ),
        "status": status,
        "strong_configs": strong,
        "weak_configs": weak[:10],
        "observed_layers": observed_layers,
        "missing_layers": missing_layers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage 2P LLaVA layer/top-k feature sweep.")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--expected-layers", default="12,15,18,21")
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--out-specificity", required=True)
    parser.add_argument("--out-decision", required=True)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for raw in args.inputs.split(","):
        path = Path(raw.strip())
        rows.extend(_read_csv(path))
    summary_rows = _summary(rows)
    specificity_rows = _specificity(rows)
    decision = _decision(summary_rows, specificity_rows, [x.strip() for x in args.expected_layers.split(",") if x.strip()])
    _write_csv(
        Path(args.out_summary),
        summary_rows,
        [
            "layer",
            "top_k",
            "position_group",
            "direction",
            "feature_group",
            "n_rows",
            "positive_logit_n",
            "positive_rank_n",
            "mean_logit_effect",
            "ci95_low",
            "ci95_high",
            "effect_status",
            "mean_rank_effect",
            "mean_gap_closure",
        ],
    )
    fields = [
        "sample_id",
        "prompt_name",
        "layer",
        "top_k",
        "position_group",
        "direction",
        "target_answer",
        "evidence_logit_effect",
        "evidence_rank_effect",
        "positive_control_count",
        "above_all_controls",
    ]
    for control in CONTROL_GROUPS:
        fields.extend([f"evidence_minus_{control}_logit", f"evidence_minus_{control}_rank"])
    _write_csv(Path(args.out_specificity), specificity_rows, fields)
    _write_json(Path(args.out_decision), decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
