"""Build a Stage 2L cross-model case panel from Stage 2I/2J/2K artifacts.

This is a local, read-only analysis over existing cross-model outputs. It does
not run Qwen/LLaVA forward passes. The goal is to separate strong auxiliary
cases from diagnostic/failure cases before expanding the cross-model run.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


DEFAULT_ROOT = Path("E:/Bridging")
DEFAULT_CROSS = DEFAULT_ROOT / "doc/experiments/stage2/cross_model"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _bool_text(value: str | None) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _index_manifest(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        indexed[row["sample_id"]] = row
    return indexed


def _index_decoded(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    indexed: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("group_name") != "top_hidden_delta_plus_answer_adjacent":
            continue
        key = (row["model_family"], row["sample_id"], row["prompt_name"])
        # Prefer restore rows if present; otherwise keep the first top bridge row.
        if key not in indexed or row.get("direction") == "restore":
            indexed[key] = row
    return indexed


def _case_category(
    model: str,
    positive_count: int,
    above_both_count: int,
    not_above_count: int,
    delta_absorbed_count: int,
    activation_absorbed_count: int,
) -> str:
    if model == "qwen":
        if positive_count >= 3 and above_both_count >= 1:
            return "qwen_strong_positive"
        if positive_count >= 2:
            return "qwen_moderate_positive"
        return "qwen_weak_or_failure"
    if positive_count >= 4 and above_both_count >= 1:
        return "llava_strong_positive"
    if activation_absorbed_count > delta_absorbed_count or (
        positive_count >= 2 and delta_absorbed_count == 0 and not_above_count <= 1
    ):
        return "llava_activation_supported"
    if delta_absorbed_count > 0 or (positive_count >= 2 and not_above_count >= 1):
        return "llava_delta_diagnostic"
    return "llava_weak_or_failure"


def build_case_panel(
    case_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
    decoded_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    manifest = _index_manifest(manifest_rows)
    decoded = _index_decoded(decoded_rows)
    grouped: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in case_rows:
        key = (row["model_family"], row["sample_id"], row["prompt_name"])
        grouped[key][row["direction"]] = row

    panel: list[dict[str, object]] = []
    for (model, sample_id, prompt), by_direction in sorted(grouped.items()):
        restore = by_direction.get("restore", {})
        corrupt = by_direction.get("corrupt", {})
        manifest_row = manifest.get(sample_id, {})
        decoded_row = decoded.get((model, sample_id, prompt), {})

        metrics = {
            "restore_combo_minus_delta": _float(restore, "combo_minus_delta_combo"),
            "restore_combo_minus_activation": _float(restore, "combo_minus_activation_combo"),
            "corrupt_combo_minus_delta": _float(corrupt, "combo_minus_delta_combo"),
            "corrupt_combo_minus_activation": _float(corrupt, "combo_minus_activation_combo"),
            "restore_visual_increment": _float(restore, "source_visual_increment_over_answer_adjacent"),
            "corrupt_visual_increment": _float(corrupt, "source_visual_increment_over_answer_adjacent"),
        }
        positive_count = sum(1 for key in (
            "restore_combo_minus_delta",
            "restore_combo_minus_activation",
            "corrupt_combo_minus_delta",
            "corrupt_combo_minus_activation",
        ) if metrics[key] > 0)

        labels = [
            restore.get("visual_specificity_label", ""),
            corrupt.get("visual_specificity_label", ""),
        ]
        label_counts = Counter(label for label in labels if label)
        above_both_count = label_counts.get("above_both_matched_controls", 0)
        not_above_count = label_counts.get("not_above_matched_controls", 0)
        delta_absorbed_count = label_counts.get("delta_absorbed_activation_positive", 0)
        activation_absorbed_count = label_counts.get("activation_absorbed_delta_positive", 0)

        decoded_changed_vs_union = _bool_text(decoded_row.get("answer_changed_vs_union"))
        decoded_changed_vs_clean = _bool_text(decoded_row.get("answer_changed_vs_clean"))
        decoded_rank = _float(decoded_row, "first_step_target_rank", default=0.0)
        decoded_logit = _float(decoded_row, "first_step_target_logit", default=0.0)

        category = _case_category(
            model,
            positive_count,
            above_both_count,
            not_above_count,
            delta_absorbed_count,
            activation_absorbed_count,
        )
        evidence_score = (
            positive_count * 2.0
            + above_both_count * 1.5
            + (1.0 if decoded_changed_vs_union else 0.0)
            - not_above_count * 1.0
            - delta_absorbed_count * (0.5 if model == "llava" else 0.25)
        )

        if category in {"qwen_strong_positive", "llava_strong_positive"}:
            recommended_use = "main_or_appendix_positive_case"
        elif category in {"llava_delta_diagnostic", "qwen_weak_or_failure", "llava_weak_or_failure"}:
            recommended_use = "diagnostic_or_failure_case"
        else:
            recommended_use = "appendix_positive_case"

        panel.append({
            "model_family": model,
            "sample_id": sample_id,
            "prompt_name": prompt,
            "question_text": manifest_row.get("question_text", ""),
            "answer_text": manifest_row.get("answer_text", ""),
            "reasoning_operation": manifest_row.get("reasoning_operation", restore.get("reasoning_operation", "")),
            "visual_structure": manifest_row.get("visual_structure", restore.get("visual_structure", "")),
            "answer_area_frac": manifest_row.get("answer_area_frac", restore.get("answer_area_frac", "")),
            "restore_combo_minus_delta": round(metrics["restore_combo_minus_delta"], 6),
            "restore_combo_minus_activation": round(metrics["restore_combo_minus_activation"], 6),
            "corrupt_combo_minus_delta": round(metrics["corrupt_combo_minus_delta"], 6),
            "corrupt_combo_minus_activation": round(metrics["corrupt_combo_minus_activation"], 6),
            "restore_visual_increment": round(metrics["restore_visual_increment"], 6),
            "corrupt_visual_increment": round(metrics["corrupt_visual_increment"], 6),
            "positive_matched_control_count": positive_count,
            "above_both_count": above_both_count,
            "not_above_count": not_above_count,
            "delta_absorbed_count": delta_absorbed_count,
            "activation_absorbed_count": activation_absorbed_count,
            "decoded_changed_vs_union": decoded_changed_vs_union,
            "decoded_changed_vs_clean": decoded_changed_vs_clean,
            "decoded_first_step_target_rank": decoded_rank,
            "decoded_first_step_target_logit": decoded_logit,
            "clean_generated_answer": decoded_row.get("clean_generated_answer", ""),
            "union_generated_answer": decoded_row.get("union_generated_answer", ""),
            "bridge_generated_answer": decoded_row.get("predicted_answer", ""),
            "case_category": category,
            "recommended_use": recommended_use,
            "evidence_score": round(evidence_score, 6),
        })

    panel.sort(key=lambda row: (str(row["model_family"]), -float(row["evidence_score"]), str(row["sample_id"]), str(row["prompt_name"])))

    by_model = defaultdict(list)
    for row in panel:
        by_model[str(row["model_family"])].append(row)

    decision: dict[str, object] = {
        "n_rows": len(panel),
        "category_counts": dict(Counter(str(row["case_category"]) for row in panel)),
        "recommended_counts": dict(Counter(str(row["recommended_use"]) for row in panel)),
        "models": {},
    }
    for model, rows in sorted(by_model.items()):
        decision["models"][model] = {
            "n_rows": len(rows),
            "top_positive_cases": [
                {
                    "sample_id": row["sample_id"],
                    "prompt_name": row["prompt_name"],
                    "question_text": row["question_text"],
                    "answer_text": row["answer_text"],
                    "case_category": row["case_category"],
                    "evidence_score": row["evidence_score"],
                    "positive_matched_control_count": row["positive_matched_control_count"],
                }
                for row in sorted(rows, key=lambda item: -float(item["evidence_score"]))[:5]
            ],
            "diagnostic_cases": [
                {
                    "sample_id": row["sample_id"],
                    "prompt_name": row["prompt_name"],
                    "question_text": row["question_text"],
                    "answer_text": row["answer_text"],
                    "case_category": row["case_category"],
                    "evidence_score": row["evidence_score"],
                }
                for row in rows
                if row["recommended_use"] == "diagnostic_or_failure_case"
            ][:5],
            "mean_evidence_score": round(_safe_mean([float(row["evidence_score"]) for row in rows]), 6),
            "mean_positive_count": round(_safe_mean([float(row["positive_matched_control_count"]) for row in rows]), 6),
        }
    return panel, decision


def _markdown_table(rows: list[dict[str, object]], columns: list[str], limit: int = 8) -> str:
    shown = rows[:limit]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in shown:
        body.append("| " + " | ".join(str(row.get(col, "")).replace("\n", " ") for col in columns) + " |")
    return "\n".join([header, sep, *body])


def write_markdown(path: Path, panel: list[dict[str, object]], decision: dict[str, object]) -> None:
    qwen = [row for row in panel if row["model_family"] == "qwen"]
    llava = [row for row in panel if row["model_family"] == "llava"]
    qwen_top = sorted(qwen, key=lambda row: -float(row["evidence_score"]))[:6]
    llava_top = sorted(llava, key=lambda row: -float(row["evidence_score"]))[:6]
    diagnostics = [row for row in panel if row["recommended_use"] == "diagnostic_or_failure_case"]
    diagnostic_top = sorted(diagnostics, key=lambda row: (str(row["model_family"]), -float(row["evidence_score"])))[:8]

    columns = [
        "model_family",
        "sample_id",
        "prompt_name",
        "answer_text",
        "reasoning_operation",
        "positive_matched_control_count",
        "case_category",
        "evidence_score",
    ]
    text = f"""# Stage 2L-1：Cross-Model Case Panel 与失败模式面板

## 1. 实验目的

这一步不重新跑 Qwen/LLaVA，而是复用 Stage 2I/2J/2K 的结果，挑出最适合进入正文或附录的跨模型 case，并把失败模式单独列出来。

核心问题是：

```text
哪些 case 最支持 Qwen 的 matched-control hidden bridge？
哪些 case 说明 LLaVA 的 bridge 仍然 partial / heterogeneous？
下一步扩大样本时应该优先扩大哪些类型？
```

## 2. 输入与输出

输入：

```text
doc/experiments/stage2/cross_model/stage2k_matched_control_explanation_case.csv
doc/experiments/stage2/cross_model/stage2i_selected_12_manifest.csv
doc/experiments/stage2/cross_model/stage2i_qwen_decoded_bridge_expansion.csv
doc/experiments/stage2/cross_model/stage2i_llava_decoded_bridge_expansion.csv
```

输出：

```text
doc/experiments/stage2/cross_model/stage2l_case_panel.csv
doc/experiments/stage2/cross_model/stage2l_case_panel_decision.json
doc/experiments/stage2/044_stage2l_cross_model_case_panel.md
```

## 3. 方法

对每个 `model x sample x prompt` 汇总 restore 与 corrupt 两个方向：

```text
restore_combo_minus_delta
restore_combo_minus_activation
corrupt_combo_minus_delta
corrupt_combo_minus_activation
```

同时记录：

```text
是否 above_both_matched_controls
是否被 delta-matched / activation-matched controls 吸收
decoded bridge 是否改变 union_mask 下的答案
题型、答案、问题、answer mask 面积比例
```

分类规则是保守的：

```text
Qwen strong-positive: matched-control 四项里多数为正，且至少一个方向 above_both。
LLaVA strong-positive: 同时强于 delta/activation matched controls。
LLaVA delta-diagnostic: activation control 下有信号，但 delta control 吸收明显。
weak/failure: 不足以作为正证据，只适合失败分析。
```

## 4. 总体统计

```json
{json.dumps(decision["category_counts"], ensure_ascii=False, indent=2)}
```

## 5. Qwen 正证据候选

{_markdown_table(qwen_top, columns)}

读法：

```text
Qwen 的正文/附录正证据应优先从这些 case 里选。
它们主要用于支持：Qwen 不是只有 readout sensitivity，而是存在更强的 hidden-state-level evidence-to-answer bridge。
仍然不能写成 Qwen 已经完成 Gemma-style source-control causal route replication。
```

## 6. LLaVA 候选与诊断

{_markdown_table(llava_top, columns)}

读法：

```text
LLaVA 可以作为“不是 Gemma-only / Qwen-only”的辅助证据。
但它的 specificity 弱于 Qwen，尤其 delta-matched controls 会解释一部分效果。
因此 LLaVA 更适合放在 partial cross-model support 和 heterogeneity analysis。
```

## 7. 失败或诊断 case

{_markdown_table(diagnostic_top, columns)}

这些 case 的作用不是削弱主线，而是帮助我们防止过度叙事：

```text
如果一个 case 被 delta-matched control 吸收，就不能写成强 source-like specificity。
如果一个 case 只有 answer-adjacent text 位置有效，就更像 answer-local aggregation，而不是纯视觉位置路径。
如果 decoded answer 不动，只能写 first-token / rank bridge。
```

## 8. 当前结论

Stage 2L-1 支持继续推进，但也把边界画得更清楚：

```text
Qwen：适合继续做扩大样本、negative controls、decoded bridge only-on-passing-cases。
LLaVA：适合继续做 diagnostic replication，重点解释 delta-matched control 为什么吸收效果。
Gemma：仍然是唯一完整主链模型，cross-model 目前只作为 auxiliary support。
```

## 9. 下一步建议

推荐马上做两件事：

```text
1. Stage 2L-3 negative controls：mask-shuffled 与 wrong-target token。
2. Stage 2L-4 evidence-to-answer 汇聚位置分析：image-only / answer-adjacent-only / image+answer-adjacent。
```

如果这两步继续支持 Qwen，再扩到 24 / 36 个样本会更有意义；否则先不要盲目扩大样本。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-dir", type=Path, default=DEFAULT_CROSS)
    parser.add_argument("--out-doc", type=Path, default=DEFAULT_ROOT / "doc/experiments/stage2/044_stage2l_cross_model_case_panel.md")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CROSS / "stage2l_case_panel.csv")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_CROSS / "stage2l_case_panel_decision.json")
    args = parser.parse_args()

    case_rows = _read_csv(args.cross_dir / "stage2k_matched_control_explanation_case.csv")
    manifest_rows = _read_csv(args.cross_dir / "stage2i_selected_12_manifest.csv")
    decoded_rows = (
        _read_csv(args.cross_dir / "stage2i_qwen_decoded_bridge_expansion.csv")
        + _read_csv(args.cross_dir / "stage2i_llava_decoded_bridge_expansion.csv")
    )

    panel, decision = build_case_panel(case_rows, manifest_rows, decoded_rows)
    fieldnames = [
        "model_family",
        "sample_id",
        "prompt_name",
        "question_text",
        "answer_text",
        "reasoning_operation",
        "visual_structure",
        "answer_area_frac",
        "restore_combo_minus_delta",
        "restore_combo_minus_activation",
        "corrupt_combo_minus_delta",
        "corrupt_combo_minus_activation",
        "restore_visual_increment",
        "corrupt_visual_increment",
        "positive_matched_control_count",
        "above_both_count",
        "not_above_count",
        "delta_absorbed_count",
        "activation_absorbed_count",
        "decoded_changed_vs_union",
        "decoded_changed_vs_clean",
        "decoded_first_step_target_rank",
        "decoded_first_step_target_logit",
        "clean_generated_answer",
        "union_generated_answer",
        "bridge_generated_answer",
        "case_category",
        "recommended_use",
        "evidence_score",
    ]
    _write_csv(args.out_csv, panel, fieldnames)
    args.out_json.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(args.out_doc, panel, decision)
    print(json.dumps({
        "out_doc": str(args.out_doc),
        "out_csv": str(args.out_csv),
        "out_json": str(args.out_json),
        "n_rows": len(panel),
        "category_counts": decision["category_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
