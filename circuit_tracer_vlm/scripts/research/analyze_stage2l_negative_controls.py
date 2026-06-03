#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
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


def _bootstrap_ci(values: list[float], *, seed: int = 20260521, n_boot: int = 2000) -> tuple[float, float]:
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
        if str(row.get("scale")) not in {"1.0", "1"}:
            continue
        out.append(
            {
                **row,
                "effect_logit_f": _float(row.get("effect_logit")),
                "effect_rank_f": _float(row.get("effect_rank")),
                "effect_gap_closure_f": _float(row.get("effect_gap_closure")),
            }
        )
    return out


def build_case_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (
            row["model_family"],
            row["sample_id"],
            row["prompt_name"],
            row["direction"],
            row["group_name"],
        )
        grouped[key][row["target_label"]] = row

    out = []
    for key, by_target in sorted(grouped.items()):
        correct = by_target.get("correct")
        wrong = by_target.get("wrong")
        if not correct or not wrong:
            continue
        model, sample_id, prompt_name, direction, group_name = key
        correct_effect = correct["effect_logit_f"]
        wrong_effect = wrong["effect_logit_f"]
        correct_rank = correct["effect_rank_f"]
        wrong_rank = wrong["effect_rank_f"]
        out.append(
            {
                "model_family": model,
                "sample_id": sample_id,
                "prompt_name": prompt_name,
                "direction": direction,
                "group_name": group_name,
                "group_kind": correct["group_kind"],
                "position_count": correct["position_count"],
                "correct_answer": correct["target_answer"],
                "wrong_answer": wrong["target_answer"],
                "wrong_source_sample_id": wrong["target_source_sample_id"],
                "correct_clean_union_logit_gap": correct["clean_union_logit_gap"],
                "wrong_clean_union_logit_gap": wrong["clean_union_logit_gap"],
                "correct_effect_logit": _round(correct_effect),
                "wrong_effect_logit": _round(wrong_effect),
                "correct_minus_wrong_logit": _round(correct_effect - wrong_effect),
                "correct_effect_rank": _round(correct_rank),
                "wrong_effect_rank": _round(wrong_rank),
                "correct_minus_wrong_rank": _round(correct_rank - wrong_rank),
                "correct_positive": correct_effect > 0,
                "wrong_positive": wrong_effect > 0,
                "correct_stronger": correct_effect > wrong_effect,
            }
        )
    return out


def build_summary(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[(row["model_family"], row["direction"], row["group_name"])].append(row)

    out = []
    for (model, direction, group_name), group in sorted(grouped.items()):
        diff = [_float(row["correct_minus_wrong_logit"]) for row in group]
        correct = [_float(row["correct_effect_logit"]) for row in group]
        wrong = [_float(row["wrong_effect_logit"]) for row in group]
        low, high = _bootstrap_ci(diff)
        out.append(
            {
                "model_family": model,
                "direction": direction,
                "group_name": group_name,
                "n_rows": len(group),
                "n_samples": len({row["sample_id"] for row in group}),
                "mean_correct_effect_logit": _round(_mean(correct)),
                "mean_wrong_effect_logit": _round(_mean(wrong)),
                "mean_correct_minus_wrong_logit": _round(_mean(diff)),
                "ci95_low": _round(low),
                "ci95_high": _round(high),
                "correct_stronger_n": sum(bool(row["correct_stronger"]) for row in group),
                "correct_positive_n": sum(bool(row["correct_positive"]) for row in group),
                "wrong_positive_n": sum(bool(row["wrong_positive"]) for row in group),
                "status": _status(_mean(diff), low, high),
            }
        )
    return out


def _status(mean_value: float, low: float, high: float) -> str:
    if math.isnan(mean_value):
        return "no_data"
    if mean_value > 0 and low > 0:
        return "stable_correct_gt_wrong"
    if mean_value > 0:
        return "weak_correct_gt_wrong"
    if high < 0:
        return "wrong_gt_correct"
    return "heterogeneous"


def build_decision(summary: list[dict[str, Any]]) -> dict[str, Any]:
    target_group = "top_hidden_delta_plus_answer_adjacent"
    decision: dict[str, Any] = {"target_group": target_group, "models": {}}
    for model in sorted({row["model_family"] for row in summary}):
        rows = [row for row in summary if row["model_family"] == model and row["group_name"] == target_group]
        statuses = {row["direction"]: row["status"] for row in rows}
        stable = sum(status == "stable_correct_gt_wrong" for status in statuses.values())
        weak = sum(status in {"stable_correct_gt_wrong", "weak_correct_gt_wrong"} for status in statuses.values())
        if stable >= 1 and weak >= 2:
            verdict = "wrong_target_control_supported"
        elif weak >= 1:
            verdict = "wrong_target_control_partial"
        else:
            verdict = "wrong_target_control_not_supported"
        decision["models"][model] = {
            "verdict": verdict,
            "direction_statuses": statuses,
            "target_group_rows": rows,
        }
    decision["claim_boundary"] = (
        "This supports target-specific hidden bridge only when correct-target effects exceed wrong-target effects; "
        "it is still not Gemma-style source-control causal route replication."
    )
    return decision


def _markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 16) -> str:
    shown = rows[:limit]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in shown:
        body.append("| " + " | ".join(str(row.get(col, "")).replace("\n", " ") for col in columns) + " |")
    return "\n".join([header, sep, *body])


def write_doc(path: Path, summary: list[dict[str, Any]], decision: dict[str, Any]) -> None:
    target_rows = [row for row in summary if row["group_name"] == decision["target_group"]]
    columns = [
        "model_family",
        "direction",
        "group_name",
        "n_rows",
        "mean_correct_effect_logit",
        "mean_wrong_effect_logit",
        "mean_correct_minus_wrong_logit",
        "ci95_low",
        "ci95_high",
        "status",
    ]
    text = f"""# Stage 2L-3：Wrong-Target Negative Control

## 1. 实验目的

这一步检验一个关键替代解释：

```text
hidden-position bridge 是否只是普遍推高/拉低任意答案 token，而不是更偏向当前样本的正确 target answer？
```

如果同一组 patch 对正确答案 target 的 restore/corrupt 效果强于错误 target，就能支持：

```text
cross-model hidden bridge 有 answer-specific 成分。
```

但这仍然不能升级为：

```text
Qwen/LLaVA 已经复现 Gemma source-control causal route。
```

## 2. 方法

每个样本保留原来的 correct target，同时从 manifest 的下一个不同样本中取 wrong target answer。

对同一组 hidden-position patch 分别计算：

```text
correct_effect_logit
wrong_effect_logit
correct_minus_wrong_logit
```

主组固定为：

```text
top_hidden_delta_plus_answer_adjacent
```

控制组包括：

```text
delta_matched_plus_answer_adjacent
activation_matched_plus_answer_adjacent
answer_adjacent_text
low_delta_control
random_control_1
```

## 3. 主组结果

{_markdown_table(target_rows, columns)}

## 4. 判定

```json
{json.dumps(decision["models"], ensure_ascii=False, indent=2)}
```

## 5. 读法

如果 `top_hidden_delta_plus_answer_adjacent` 在 correct target 上稳定强于 wrong target，说明 bridge 更像 answer-specific bridge。

如果 wrong target 也被同等恢复，则说明现象可能更多是 general logit movement 或 answer-adjacent distribution shift。

本实验只验证 target-specificity，不验证 object-level semantic node，也不验证完整 source-control route replication。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-dir", type=Path, default=DEFAULT_CROSS)
    parser.add_argument("--qwen-csv", default="stage2l_qwen_wrong_target_negative_control.csv")
    parser.add_argument("--llava-csv", default="stage2l_llava_wrong_target_negative_control.csv")
    parser.add_argument("--out-case-csv", type=Path, default=DEFAULT_CROSS / "stage2l_wrong_target_case_table.csv")
    parser.add_argument("--out-summary-csv", type=Path, default=DEFAULT_CROSS / "stage2l_wrong_target_summary.csv")
    parser.add_argument("--out-decision-json", type=Path, default=DEFAULT_CROSS / "stage2l_wrong_target_decision.json")
    parser.add_argument("--out-doc", type=Path, default=DEFAULT_ROOT / "doc/experiments/stage2/045_stage2l_wrong_target_negative_control.md")
    args = parser.parse_args()

    rows = _prepare(_read_csv(args.cross_dir / args.qwen_csv) + _read_csv(args.cross_dir / args.llava_csv))
    case_rows = build_case_table(rows)
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
        "correct_answer",
        "wrong_answer",
        "wrong_source_sample_id",
        "correct_clean_union_logit_gap",
        "wrong_clean_union_logit_gap",
        "correct_effect_logit",
        "wrong_effect_logit",
        "correct_minus_wrong_logit",
        "correct_effect_rank",
        "wrong_effect_rank",
        "correct_minus_wrong_rank",
        "correct_positive",
        "wrong_positive",
        "correct_stronger",
    ]
    summary_fields = [
        "model_family",
        "direction",
        "group_name",
        "n_rows",
        "n_samples",
        "mean_correct_effect_logit",
        "mean_wrong_effect_logit",
        "mean_correct_minus_wrong_logit",
        "ci95_low",
        "ci95_high",
        "correct_stronger_n",
        "correct_positive_n",
        "wrong_positive_n",
        "status",
    ]
    _write_csv(args.out_case_csv, case_rows, case_fields)
    _write_csv(args.out_summary_csv, summary, summary_fields)
    _write_json(args.out_decision_json, decision)
    write_doc(args.out_doc, summary, decision)
    print(
        json.dumps(
            {
                "case_rows": len(case_rows),
                "summary_rows": len(summary),
                "decision": decision["models"],
                "category_counts": dict(Counter(row["status"] for row in summary)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
