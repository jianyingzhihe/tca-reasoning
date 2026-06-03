#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"
PREFIX = "stage4_qwen_middle_dense"
CONTROL = {"same_position_matched_feature_control", "same_feature_random_position_control", "random_active_feature_control"}


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def _f(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _ci(values: list[float], seed: int = 48048, n: int = 1000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        means.append(_mean([values[rng.randrange(len(values))] for _ in values]))
    means.sort()
    return means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]


def _metric(metric: str, values: list[float], extra: dict[str, Any]) -> dict[str, Any]:
    lo, hi = _ci(values)
    row = {
        "metric": metric,
        "n": len(values),
        "mean": _mean(values),
        "ci95_low": lo,
        "ci95_high": hi,
        "positive_frac": sum(1 for value in values if value > 0) / len(values) if values else 0.0,
    }
    row.update(extra)
    return row


def _files(pack: str, mode: str) -> list[tuple[int, str, Path, Path, Path]]:
    out = []
    for cand in CROSS.glob(f"{PREFIX}_{pack}_{mode}_L*_candidates.csv"):
        rest = cand.name[len(f"{PREFIX}_{pack}_{mode}_L") : -len("_candidates.csv")]
        layer_raw, group = rest.split("_", 1)
        layer = int(layer_raw)
        zero = CROSS / f"{PREFIX}_{pack}_{mode}_L{layer}_{group}_zeroing_raw.csv"
        grouped = CROSS / f"{PREFIX}_{pack}_{mode}_L{layer}_{group}_group_raw.csv"
        out.append((layer, group, cand, zero, grouped))
    return sorted(out, key=lambda item: (item[0], item[1]))


def _zero_metrics(rows: list[dict[str, str]], extra: dict[str, Any]) -> list[dict[str, Any]]:
    target = [row for row in rows if row.get("status", "ok") in {"", "ok"} and row.get("intervention_kind") == "clean_zeroing" and row.get("token_scored") == "target"]
    wrong = [row for row in rows if row.get("status", "ok") in {"", "ok"} and row.get("intervention_kind") == "clean_zeroing" and row.get("token_scored") == "wrong"]
    by_candidate: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in target:
        by_candidate[row.get("candidate_id", "")][row.get("control_group", "")] = row
    wrong_source = {row.get("candidate_id", ""): row for row in wrong if row.get("control_group") == "source"}
    source_control = []
    correct_wrong = []
    rank = []
    source_effect = []
    for cid, group in by_candidate.items():
        source = group.get("source")
        controls = [group[name] for name in CONTROL if name in group]
        if source:
            source_effect.append(_f(source, "logit_effect"))
            rank.append(_f(source, "rank_effect"))
        if source and controls:
            source_control.append(_f(source, "logit_effect") - _mean([_f(row, "logit_effect") for row in controls]))
        if source and cid in wrong_source:
            correct_wrong.append(_f(source, "logit_effect") - _f(wrong_source[cid], "logit_effect"))
    return [
        _metric("zeroing_source_effect", source_effect, extra),
        _metric("zeroing_source_minus_controls", source_control, extra),
        _metric("zeroing_correct_minus_wrong", correct_wrong, extra),
        _metric("zeroing_rank_effect", rank, extra),
    ]


def _group_metrics(rows: list[dict[str, str]], extra_base: dict[str, Any]) -> list[dict[str, Any]]:
    idx: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("record_type") == "activation_drop":
            continue
        idx[(row.get("top_k", ""), row.get("mask_condition", ""), row.get("control_group", ""))].append(row)
    out = []
    for topk in sorted({key[0] for key in idx}, key=lambda raw: int(float(raw or 0))):
        for real in ["answer_mask", "union_mask"]:
            source = idx.get((topk, real, "source"), [])
            shifted = {(row.get("sample_id", ""), row.get("prompt_name", "")): row for row in idx.get((topk, "shifted_mask", "source"), [])}
            shuffled = {(row.get("sample_id", ""), row.get("prompt_name", "")): row for row in idx.get((topk, "shuffled_mask", "source"), [])}
            controls: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
            for name in CONTROL:
                for row in idx.get((topk, real, name), []):
                    controls[(row.get("sample_id", ""), row.get("prompt_name", ""))].append(row)
            effect = [_f(row, "target_logit_effect") for row in source]
            rank = [_f(row, "target_rank_effect") for row in source]
            real_shifted = [_f(row, "target_logit_effect") - _f(shifted[(row.get("sample_id", ""), row.get("prompt_name", ""))], "target_logit_effect") for row in source if (row.get("sample_id", ""), row.get("prompt_name", "")) in shifted]
            real_shuffled = [_f(row, "target_logit_effect") - _f(shuffled[(row.get("sample_id", ""), row.get("prompt_name", ""))], "target_logit_effect") for row in source if (row.get("sample_id", ""), row.get("prompt_name", "")) in shuffled]
            source_controls = [
                _f(row, "target_logit_effect") - _mean([_f(control, "target_logit_effect") for control in controls[(row.get("sample_id", ""), row.get("prompt_name", ""))]])
                for row in source
                if controls.get((row.get("sample_id", ""), row.get("prompt_name", "")))
            ]
            extra = dict(extra_base)
            extra.update({"top_k": topk, "mask": real})
            out.extend(
                [
                    _metric("restore_source_effect", effect, extra),
                    _metric("restore_source_minus_controls", source_controls, extra),
                    _metric("restore_real_minus_shifted", real_shifted, extra),
                    _metric("restore_real_minus_shuffled", real_shuffled, extra),
                    _metric("restore_rank_effect", rank, extra),
                ]
            )
    return out


def _positive(row: dict[str, Any], min_n: int = 24) -> bool:
    return int(row.get("n", 0)) >= min_n and float(row.get("ci95_low", 0.0)) > 0 and float(row.get("positive_frac", 0.0)) >= 0.5


def analyze(pack: str, mode: str) -> dict[str, Any]:
    summary = []
    specificity = []
    for layer, group, cand_path, zero_path, grouped_path in _files(pack, mode):
        candidates = _read(cand_path)
        zero = _read(zero_path)
        grouped = _read(grouped_path)
        extra = {"pack": pack, "mode": mode, "layer": layer, "position_group": group}
        summary.append(
            {
                **extra,
                "candidate_rows": len(candidates),
                "main_rows": sum(1 for row in candidates if row.get("include_main") == "1"),
                "prompt_runs": len({(row.get("sample_id", ""), row.get("prompt_name", "")) for row in candidates}),
                "zeroing_rows": len(zero),
                "group_rows": len(grouped),
            }
        )
        specificity.extend(_zero_metrics(zero, extra))
        specificity.extend(_group_metrics(grouped, extra))
    _write_csv(CROSS / f"{PREFIX}_{pack}_{mode}_summary.csv", summary)
    _write_csv(CROSS / f"{PREFIX}_{pack}_{mode}_specificity.csv", specificity)
    source_pos = [row for row in specificity if row.get("metric") in {"zeroing_source_minus_controls", "restore_source_minus_controls"} and _positive(row)]
    mask_pos = [row for row in specificity if row.get("metric") in {"restore_real_minus_shifted", "restore_real_minus_shuffled"} and _positive(row)]
    correct_pos = [row for row in specificity if row.get("metric") == "zeroing_correct_minus_wrong" and float(row.get("mean", 0.0)) > 0 and float(row.get("positive_frac", 0.0)) >= 0.4]
    if source_pos and mask_pos and correct_pos:
        status = "qwen_middle_sparse_or_distributed_route_near_pass"
    elif source_pos:
        status = "qwen_answer_support_not_evidence_linked"
    elif summary:
        status = "qwen_plt_localization_failed_hidden_supported"
    else:
        status = "blocked"
    decision = {
        "status": status,
        "pack": pack,
        "mode": mode,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configs": len(summary),
        "source_positive_metrics": len(source_pos),
        "mask_positive_metrics": len(mask_pos),
        "correct_positive_metrics": len(correct_pos),
        "source_positive_configs": sorted({f"L{row['layer']}:{row['position_group']}:topK{row.get('top_k', '')}:{row.get('mask', '')}" for row in source_pos}),
        "mask_positive_configs": sorted({f"L{row['layer']}:{row['position_group']}:topK{row.get('top_k', '')}:{row.get('mask', '')}" for row in mask_pos}),
        "claim_boundary": "Middle dense scan is Qwen-native PLT localization; failure does not negate hidden-level mechanism.",
    }
    (CROSS / f"{PREFIX}_{pack}_{mode}_decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage4-048 Qwen middle dense scan.")
    parser.add_argument("--pack", choices=["primary", "strict"], default="primary")
    parser.add_argument("--mode", choices=["smoke", "full", "strict-confirm"], default="smoke")
    args = parser.parse_args()
    print(json.dumps(analyze(args.pack, args.mode), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
