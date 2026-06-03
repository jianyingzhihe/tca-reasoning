#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("E:/Bridging")
DEFAULT_CROSS = DEFAULT_ROOT / "doc/experiments/stage2/cross_model"


def _read_csv(path: Path) -> list[dict[str, str]]:
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


def _float(value: Any) -> float:
    try:
        if value in ("", None):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _mean(values: list[float]) -> float:
    vals = [value for value in values if not math.isnan(value)]
    return sum(vals) / len(vals) if vals else float("nan")


def _round(value: float) -> float | str:
    return "" if math.isnan(value) else round(value, 6)


def _bootstrap_ci(values: list[float], seed: int = 20260521, n_boot: int = 2000) -> tuple[float, float]:
    vals = [value for value in values if not math.isnan(value)]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], vals[0]
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * len(means))], means[int(0.975 * len(means))]


def _prepare(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("direction") not in {"restore", "corrupt"}:
            continue
        out.append({**row, "effect_logit_f": _float(row.get("effect_logit")), "effect_rank_f": _float(row.get("effect_rank"))})
    return out


def build_case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["model_family"], row["sample_id"], row["prompt_name"], row["direction"], row["group_name"])
        grouped[key][row["mask_label"]] = row
    out = []
    for key, by_mask in sorted(grouped.items()):
        real = by_mask.get("evidence_mask")
        shuffled = by_mask.get("mask_shuffled")
        if not real or not shuffled:
            continue
        model, sample_id, prompt_name, direction, group_name = key
        real_effect = real["effect_logit_f"]
        shuffled_effect = shuffled["effect_logit_f"]
        out.append(
            {
                "model_family": model,
                "sample_id": sample_id,
                "prompt_name": prompt_name,
                "direction": direction,
                "group_name": group_name,
                "group_kind": real["group_kind"],
                "position_count": real["position_count"],
                "real_mask_effect_logit": _round(real_effect),
                "shuffled_mask_effect_logit": _round(shuffled_effect),
                "real_minus_shuffled_logit": _round(real_effect - shuffled_effect),
                "real_mask_effect_rank": _round(real["effect_rank_f"]),
                "shuffled_mask_effect_rank": _round(shuffled["effect_rank_f"]),
                "real_minus_shuffled_rank": _round(real["effect_rank_f"] - shuffled["effect_rank_f"]),
                "real_clean_mask_gap": real["clean_mask_logit_gap"],
                "shuffled_clean_mask_gap": shuffled["clean_mask_logit_gap"],
                "real_stronger": real_effect > shuffled_effect,
            }
        )
    return out


def _status(mean_value: float, low: float, high: float) -> str:
    if math.isnan(mean_value):
        return "no_data"
    if mean_value > 0 and low > 0:
        return "stable_real_gt_shuffled"
    if mean_value > 0:
        return "weak_real_gt_shuffled"
    if high < 0:
        return "shuffled_gt_real"
    return "heterogeneous"


def build_summary(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[(row["model_family"], row["direction"], row["group_name"])].append(row)
    out = []
    for (model, direction, group_name), group in sorted(grouped.items()):
        diff = [_float(row["real_minus_shuffled_logit"]) for row in group]
        low, high = _bootstrap_ci(diff)
        out.append(
            {
                "model_family": model,
                "direction": direction,
                "group_name": group_name,
                "n_rows": len(group),
                "n_samples": len({row["sample_id"] for row in group}),
                "mean_real_mask_effect_logit": _round(_mean([_float(row["real_mask_effect_logit"]) for row in group])),
                "mean_shuffled_mask_effect_logit": _round(_mean([_float(row["shuffled_mask_effect_logit"]) for row in group])),
                "mean_real_minus_shuffled_logit": _round(_mean(diff)),
                "ci95_low": _round(low),
                "ci95_high": _round(high),
                "real_stronger_n": sum(bool(row["real_stronger"]) for row in group),
                "status": _status(_mean(diff), low, high),
            }
        )
    return out


def build_decision(summary: list[dict[str, Any]]) -> dict[str, Any]:
    target_group = "top_hidden_delta_plus_answer_adjacent"
    models: dict[str, Any] = {}
    for model in sorted({row["model_family"] for row in summary}):
        rows = [row for row in summary if row["model_family"] == model and row["group_name"] == target_group]
        statuses = {row["direction"]: row["status"] for row in rows}
        stable = sum(status == "stable_real_gt_shuffled" for status in statuses.values())
        weak = sum(status in {"stable_real_gt_shuffled", "weak_real_gt_shuffled"} for status in statuses.values())
        if stable >= 1 and weak >= 2:
            verdict = "mask_shuffled_control_supported"
        elif weak >= 1:
            verdict = "mask_shuffled_control_partial"
        else:
            verdict = "mask_shuffled_control_not_supported"
        models[model] = {"verdict": verdict, "direction_statuses": statuses, "target_group_rows": rows}
    return {
        "target_group": target_group,
        "models": models,
        "claim_boundary": "Mask-shuffled control supports evidence-location specificity only at hidden-state bridge level.",
    }


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        "model_family",
        "direction",
        "group_name",
        "n_rows",
        "mean_real_mask_effect_logit",
        "mean_shuffled_mask_effect_logit",
        "mean_real_minus_shuffled_logit",
        "ci95_low",
        "ci95_high",
        "status",
    ]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def write_doc(path: Path, summary: list[dict[str, Any]], decision: dict[str, Any]) -> None:
    target_rows = [row for row in summary if row["group_name"] == decision["target_group"]]
    text = f"""# Stage 2L-3b：Mask-Shuffled Negative Control

## 1. 实验目的

这一步检验另一个替代解释：

```text
hidden bridge 是否只是由任意遮挡造成，而不是由真实证据区域遮挡造成？
```

方法是在同一张图内把 union mask 平移到非原始位置，形成 `mask_shuffled` 条件。然后固定使用真实 evidence mask 导出的 source-like position group，比较：

```text
evidence_mask patch effect
mask_shuffled patch effect
real_minus_shuffled
```

## 2. 主组结果

{_markdown_table(target_rows)}

## 3. 判定

```json
{json.dumps(decision["models"], ensure_ascii=False, indent=2)}
```

## 4. 读法

如果真实 evidence mask 的 patch effect 强于 shifted mask，说明 bridge 对证据位置有一定特异性。

如果 shifted mask 同样强，说明现象可能更多来自遮挡强度、全局分布变化或 answer-adjacent 聚合，而不是严格 evidence-region specificity。

本实验仍然是 hidden-state-level negative control，不是 feature-level source route replication。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-dir", type=Path, default=DEFAULT_CROSS)
    parser.add_argument("--qwen-csv", default="stage2l_qwen_mask_shuffled_negative_control.csv")
    parser.add_argument("--llava-csv", default="stage2l_llava_mask_shuffled_negative_control.csv")
    parser.add_argument("--out-case-csv", type=Path, default=DEFAULT_CROSS / "stage2l_mask_shuffled_case_table.csv")
    parser.add_argument("--out-summary-csv", type=Path, default=DEFAULT_CROSS / "stage2l_mask_shuffled_summary.csv")
    parser.add_argument("--out-decision-json", type=Path, default=DEFAULT_CROSS / "stage2l_mask_shuffled_decision.json")
    parser.add_argument("--out-doc", type=Path, default=DEFAULT_ROOT / "doc/experiments/stage2/047_stage2l_mask_shuffled_negative_control.md")
    args = parser.parse_args()

    rows = _prepare(_read_csv(args.cross_dir / args.qwen_csv) + _read_csv(args.cross_dir / args.llava_csv))
    case_rows = build_case_rows(rows)
    summary = build_summary(case_rows)
    decision = build_decision(summary)
    case_fields = [
        "model_family",
        "sample_id",
        "prompt_name",
        "direction",
        "group_name",
        "group_kind",
        "position_count",
        "real_mask_effect_logit",
        "shuffled_mask_effect_logit",
        "real_minus_shuffled_logit",
        "real_mask_effect_rank",
        "shuffled_mask_effect_rank",
        "real_minus_shuffled_rank",
        "real_clean_mask_gap",
        "shuffled_clean_mask_gap",
        "real_stronger",
    ]
    summary_fields = [
        "model_family",
        "direction",
        "group_name",
        "n_rows",
        "n_samples",
        "mean_real_mask_effect_logit",
        "mean_shuffled_mask_effect_logit",
        "mean_real_minus_shuffled_logit",
        "ci95_low",
        "ci95_high",
        "real_stronger_n",
        "status",
    ]
    _write_csv(args.out_case_csv, case_rows, case_fields)
    _write_csv(args.out_summary_csv, summary, summary_fields)
    _write_json(args.out_decision_json, decision)
    write_doc(args.out_doc, summary, decision)
    print(json.dumps({"case_rows": len(case_rows), "summary_rows": len(summary), "decision": decision["models"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
