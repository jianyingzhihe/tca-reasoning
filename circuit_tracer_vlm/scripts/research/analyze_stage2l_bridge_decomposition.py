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


def build_case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    needed = {
        "top_hidden_delta",
        "answer_adjacent_text",
        "top_hidden_delta_plus_answer_adjacent",
        "delta_matched_plus_answer_adjacent",
        "activation_matched_plus_answer_adjacent",
    }
    grouped: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["group_name"] in needed:
            grouped[(row["model_family"], row["sample_id"], row["prompt_name"], row["direction"])][row["group_name"]] = row

    out = []
    for (model, sample_id, prompt_name, direction), by_group in sorted(grouped.items()):
        image = by_group.get("top_hidden_delta")
        answer = by_group.get("answer_adjacent_text")
        combo = by_group.get("top_hidden_delta_plus_answer_adjacent")
        delta = by_group.get("delta_matched_plus_answer_adjacent")
        activation = by_group.get("activation_matched_plus_answer_adjacent")
        if not image or not answer or not combo:
            continue
        image_logit = image["effect_logit_f"]
        answer_logit = answer["effect_logit_f"]
        combo_logit = combo["effect_logit_f"]
        delta_logit = delta["effect_logit_f"] if delta else float("nan")
        activation_logit = activation["effect_logit_f"] if activation else float("nan")
        max_single = max(image_logit, answer_logit)
        combo_minus_max = combo_logit - max_single
        combo_minus_image = combo_logit - image_logit
        combo_minus_answer = combo_logit - answer_logit
        visual_contribution_over_answer = combo_minus_answer
        answer_contribution_over_image = combo_minus_image
        if combo_minus_image > 0 and combo_minus_answer > 0:
            pattern = "combo_stronger_than_both"
        elif abs(combo_minus_answer) <= 0.05 and answer_logit >= image_logit:
            pattern = "answer_adjacent_dominant"
        elif abs(combo_minus_image) <= 0.05 and image_logit >= answer_logit:
            pattern = "image_only_dominant"
        elif combo_minus_max < 0:
            pattern = "combo_not_best"
        else:
            pattern = "mixed_or_small_increment"
        out.append(
            {
                "model_family": model,
                "sample_id": sample_id,
                "prompt_name": prompt_name,
                "direction": direction,
                "image_only_logit": _round(image_logit),
                "answer_adjacent_logit": _round(answer_logit),
                "combo_logit": _round(combo_logit),
                "delta_matched_combo_logit": _round(delta_logit),
                "activation_matched_combo_logit": _round(activation_logit),
                "combo_minus_image": _round(combo_minus_image),
                "combo_minus_answer": _round(combo_minus_answer),
                "combo_minus_max_single": _round(combo_minus_max),
                "combo_minus_delta_matched": _round(combo_logit - delta_logit),
                "combo_minus_activation_matched": _round(combo_logit - activation_logit),
                "visual_contribution_over_answer": _round(visual_contribution_over_answer),
                "answer_contribution_over_image": _round(answer_contribution_over_image),
                "pattern": pattern,
            }
        )
    return out


def _status(mean_value: float, low: float, high: float) -> str:
    if math.isnan(mean_value):
        return "no_data"
    if mean_value > 0 and low > 0:
        return "stable_positive"
    if mean_value > 0:
        return "weak_positive"
    if high < 0:
        return "stable_negative"
    return "heterogeneous"


def build_summary(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[(row["model_family"], row["direction"])].append(row)
    metrics = [
        "combo_minus_image",
        "combo_minus_answer",
        "combo_minus_max_single",
        "combo_minus_delta_matched",
        "combo_minus_activation_matched",
    ]
    out = []
    for (model, direction), group in sorted(grouped.items()):
        base = {
            "model_family": model,
            "direction": direction,
            "n_rows": len(group),
            "n_samples": len({row["sample_id"] for row in group}),
            "pattern_counts": json.dumps(dict(Counter(row["pattern"] for row in group)), ensure_ascii=False),
        }
        for metric in metrics:
            values = [_float(row[metric]) for row in group]
            low, high = _bootstrap_ci(values)
            base[f"mean_{metric}"] = _round(_mean(values))
            base[f"{metric}_ci95_low"] = _round(low)
            base[f"{metric}_ci95_high"] = _round(high)
            base[f"{metric}_status"] = _status(_mean(values), low, high)
        out.append(base)
    return out


def build_decision(summary: list[dict[str, Any]]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model in sorted({row["model_family"] for row in summary}):
        rows = [row for row in summary if row["model_family"] == model]
        combo_gt_answer = {row["direction"]: row["combo_minus_answer_status"] for row in rows}
        combo_gt_image = {row["direction"]: row["combo_minus_image_status"] for row in rows}
        combo_gt_max = {row["direction"]: row["combo_minus_max_single_status"] for row in rows}
        if all(status == "stable_positive" for status in combo_gt_answer.values()) and any(
            status in {"stable_positive", "weak_positive"} for status in combo_gt_image.values()
        ):
            verdict = "visual_plus_answer_bridge_supported"
        elif any(status == "stable_positive" for status in combo_gt_answer.values()):
            verdict = "visual_increment_over_answer_partial"
        elif all(status in {"heterogeneous", "weak_positive"} for status in combo_gt_max.values()):
            verdict = "answer_adjacent_or_mixed_bridge"
        else:
            verdict = "bridge_decomposition_heterogeneous"
        models[model] = {
            "verdict": verdict,
            "combo_minus_answer_status_by_direction": combo_gt_answer,
            "combo_minus_image_status_by_direction": combo_gt_image,
            "combo_minus_max_single_status_by_direction": combo_gt_max,
        }
    return {
        "models": models,
        "claim_boundary": (
            "Bridge decomposition distinguishes image-only, answer-adjacent-only, and combo hidden-state effects. "
            "It does not establish feature-level or source-control route replication."
        ),
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")).replace("\n", " ") for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def write_doc(path: Path, summary: list[dict[str, Any]], decision: dict[str, Any]) -> None:
    columns = [
        "model_family",
        "direction",
        "n_rows",
        "mean_combo_minus_image",
        "combo_minus_image_status",
        "mean_combo_minus_answer",
        "combo_minus_answer_status",
        "mean_combo_minus_max_single",
        "combo_minus_max_single_status",
    ]
    text = f"""# Stage 2L-4：Evidence-to-Answer Bridge 位置分解

## 1. 实验目的

Stage 2I/2J 显示最强 bridge 往往是：

```text
top_hidden_delta visual positions + answer-adjacent text positions
```

这一步不重新跑模型，而是复用 Stage 2J patch CSV，把 bridge 拆成：

```text
image-only: top_hidden_delta
answer-adjacent-only: answer_adjacent_text
combo: top_hidden_delta_plus_answer_adjacent
```

核心问题是：

```text
视觉位置本身是否提供额外贡献？
答案附近位置是否是主要汇聚处？
combo 是否强于 image-only 和 answer-adjacent-only？
```

## 2. 输入与输出

输入：

```text
doc/experiments/stage2/cross_model/stage2j_qwen_matched_control_patch.csv
doc/experiments/stage2/cross_model/stage2j_llava_matched_control_patch.csv
```

输出：

```text
doc/experiments/stage2/cross_model/stage2l_bridge_decomposition_case_table.csv
doc/experiments/stage2/cross_model/stage2l_bridge_decomposition_summary.csv
doc/experiments/stage2/cross_model/stage2l_bridge_decomposition_decision.json
doc/experiments/stage2/046_stage2l_evidence_to_answer_bridge_decomposition.md
```

## 3. 总体结果

{_markdown_table(summary, columns)}

## 4. 判定

```json
{json.dumps(decision["models"], ensure_ascii=False, indent=2)}
```

## 5. 读法

如果 `combo_minus_answer` 为正，说明在 answer-adjacent 位置之外，visual source-like positions 仍有额外贡献。

如果 `combo_minus_image` 为正，说明 answer-adjacent 位置在 image-only 的基础上也有额外贡献。

如果 `combo_minus_max_single` 为正，说明 combo 强于任一单独位置组，更符合 evidence-to-answer bridge / convergence 的解释。

这一步仍然只是 hidden-state-level decomposition，不是 feature-level source route replication。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-dir", type=Path, default=DEFAULT_CROSS)
    parser.add_argument("--qwen-csv", default="stage2j_qwen_matched_control_patch.csv")
    parser.add_argument("--llava-csv", default="stage2j_llava_matched_control_patch.csv")
    parser.add_argument("--out-case-csv", type=Path, default=DEFAULT_CROSS / "stage2l_bridge_decomposition_case_table.csv")
    parser.add_argument("--out-summary-csv", type=Path, default=DEFAULT_CROSS / "stage2l_bridge_decomposition_summary.csv")
    parser.add_argument("--out-decision-json", type=Path, default=DEFAULT_CROSS / "stage2l_bridge_decomposition_decision.json")
    parser.add_argument("--out-doc", type=Path, default=DEFAULT_ROOT / "doc/experiments/stage2/046_stage2l_evidence_to_answer_bridge_decomposition.md")
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
        "image_only_logit",
        "answer_adjacent_logit",
        "combo_logit",
        "delta_matched_combo_logit",
        "activation_matched_combo_logit",
        "combo_minus_image",
        "combo_minus_answer",
        "combo_minus_max_single",
        "combo_minus_delta_matched",
        "combo_minus_activation_matched",
        "visual_contribution_over_answer",
        "answer_contribution_over_image",
        "pattern",
    ]
    summary_fields = [
        "model_family",
        "direction",
        "n_rows",
        "n_samples",
        "pattern_counts",
        "mean_combo_minus_image",
        "combo_minus_image_ci95_low",
        "combo_minus_image_ci95_high",
        "combo_minus_image_status",
        "mean_combo_minus_answer",
        "combo_minus_answer_ci95_low",
        "combo_minus_answer_ci95_high",
        "combo_minus_answer_status",
        "mean_combo_minus_max_single",
        "combo_minus_max_single_ci95_low",
        "combo_minus_max_single_ci95_high",
        "combo_minus_max_single_status",
        "mean_combo_minus_delta_matched",
        "combo_minus_delta_matched_ci95_low",
        "combo_minus_delta_matched_ci95_high",
        "combo_minus_delta_matched_status",
        "mean_combo_minus_activation_matched",
        "combo_minus_activation_matched_ci95_low",
        "combo_minus_activation_matched_ci95_high",
        "combo_minus_activation_matched_status",
    ]
    _write_csv(args.out_case_csv, case_rows, case_fields)
    _write_csv(args.out_summary_csv, summary, summary_fields)
    _write_json(args.out_decision_json, decision)
    write_doc(args.out_doc, summary, decision)
    print(json.dumps({"case_rows": len(case_rows), "summary_rows": len(summary), "decision": decision["models"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
