#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(r"E:\Bridging")
STAGE = ROOT / "doc" / "experiments" / "paper2" / "stage1"
TAG = "viability_v1"


def _read_rows(model: str) -> list[dict[str, str]]:
    path = STAGE / f"stage1_composition_screen_{TAG}_{model}.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _mean(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else math.nan


def _median(values: list[float]) -> float:
    vals = sorted(v for v in values if not math.isnan(v))
    if not vals:
        return math.nan
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _norm_text(text: str) -> str:
    return " ".join((text or "").lower().strip().replace("-", " ").split())


def _summarize_condition(rows: list[dict[str, str]]) -> dict[str, float | int]:
    ranks = [_f(r, "target_rank") for r in rows]
    margins = [_f(r, "target_minus_wrong_margin") for r in rows]
    logits = [_f(r, "target_logit") for r in rows]
    return {
        "n": len(rows),
        "ok_rows": sum(1 for r in rows if r.get("status") == "ok"),
        "format_ok_frac": _mean([_f(r, "format_ok") for r in rows]),
        "empty_answer_frac": _mean([_f(r, "empty_answer") for r in rows]),
        "target_rank_mean": _mean(ranks),
        "target_rank_median": _median(ranks),
        "target_top1_frac": _mean([1.0 if v == 1 else 0.0 for v in ranks]),
        "target_top10_frac": _mean([1.0 if v <= 10 else 0.0 for v in ranks]),
        "target_logit_mean": _mean(logits),
        "target_margin_mean": _mean(margins),
    }


def _paired(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_sample: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_sample[row["sample_id"]][row["condition"]] = row

    paired_rows: list[dict[str, object]] = []
    for sample_id, conds in sorted(by_sample.items()):
        if "clean" not in conds or "wrong_image" not in conds:
            continue
        clean = conds["clean"]
        wrong = conds["wrong_image"]
        clean_rank = _f(clean, "target_rank")
        wrong_rank = _f(wrong, "target_rank")
        clean_logit = _f(clean, "target_logit")
        wrong_logit = _f(wrong, "target_logit")
        clean_margin = _f(clean, "target_minus_wrong_margin")
        wrong_margin = _f(wrong, "target_minus_wrong_margin")
        clean_pred = _norm_text(clean.get("predicted_answer", ""))
        wrong_pred = _norm_text(wrong.get("predicted_answer", ""))
        answer = _norm_text(clean.get("answer_text", ""))
        paired_rows.append(
            {
                "model_family": clean["model_family"],
                "sample_id": sample_id,
                "sample_type": clean.get("sample_type", ""),
                "question_text": clean.get("question_text", ""),
                "answer_text": clean.get("answer_text", ""),
                "clean_predicted_answer": clean.get("predicted_answer", ""),
                "wrong_predicted_answer": wrong.get("predicted_answer", ""),
                "clean_target_rank": clean_rank,
                "wrong_target_rank": wrong_rank,
                "rank_delta_wrong_minus_clean": wrong_rank - clean_rank,
                "clean_target_logit": clean_logit,
                "wrong_target_logit": wrong_logit,
                "target_logit_drop_clean_minus_wrong": clean_logit - wrong_logit,
                "clean_margin": clean_margin,
                "wrong_margin": wrong_margin,
                "margin_drop_clean_minus_wrong": clean_margin - wrong_margin,
                "decoded_changed": int(clean_pred != wrong_pred),
                "clean_exact_answer": int(clean_pred == answer),
                "wrong_exact_answer": int(wrong_pred == answer),
                "format_ok_both": int(_f(clean, "format_ok", 0) == 1 and _f(wrong, "format_ok", 0) == 1),
                "strong_case_score": (
                    max(0.0, clean_logit - wrong_logit)
                    + 0.05 * max(0.0, wrong_rank - clean_rank)
                    + 4.0 * int(clean_pred != wrong_pred)
                    + 3.0 * int(_f(clean, "format_ok", 0) == 1 and _f(wrong, "format_ok", 0) == 1)
                ),
            }
        )
    return paired_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    all_summary: list[dict[str, object]] = []
    all_paired: list[dict[str, object]] = []
    all_strong: list[dict[str, object]] = []
    decision: dict[str, object] = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tag": TAG,
        "status": "paper2_stage1_behavior_viability_ready",
        "models": {},
    }

    for model in ["gemma", "qwen"]:
        rows = _read_rows(model)
        by_condition: dict[str, list[dict[str, str]]] = defaultdict(list)
        by_type: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_condition[row["condition"]].append(row)
            by_type[(row.get("sample_type", ""), row["condition"])].append(row)

        for condition, cond_rows in sorted(by_condition.items()):
            summary = {"model_family": model, "group": "condition", "sample_type": "ALL", "condition": condition}
            summary.update(_summarize_condition(cond_rows))
            all_summary.append(summary)

        for (sample_type, condition), type_rows in sorted(by_type.items()):
            summary = {"model_family": model, "group": "sample_type", "sample_type": sample_type, "condition": condition}
            summary.update(_summarize_condition(type_rows))
            all_summary.append(summary)

        paired = _paired(rows)
        all_paired.extend(paired)
        strong = sorted(
            [
                r
                for r in paired
                if r["format_ok_both"] == 1
                and r["decoded_changed"] == 1
                and r["target_logit_drop_clean_minus_wrong"] > 0
                and r["rank_delta_wrong_minus_clean"] > 0
            ],
            key=lambda r: r["strong_case_score"],
            reverse=True,
        )[:8]
        all_strong.extend(strong)

        decision["models"][model] = {
            "raw_rows": len(rows),
            "paired_samples": len(paired),
            "strong_cases": len(strong),
            "rank_worsened_frac": _mean([1.0 if r["rank_delta_wrong_minus_clean"] > 0 else 0.0 for r in paired]),
            "logit_drop_mean": _mean([float(r["target_logit_drop_clean_minus_wrong"]) for r in paired]),
            "margin_drop_mean": _mean([float(r["margin_drop_clean_minus_wrong"]) for r in paired]),
            "decoded_changed_frac": _mean([float(r["decoded_changed"]) for r in paired]),
            "clean_exact_answer_frac": _mean([float(r["clean_exact_answer"]) for r in paired]),
            "wrong_exact_answer_frac": _mean([float(r["wrong_exact_answer"]) for r in paired]),
        }

    _write_csv(STAGE / f"paper2_stage1_behavior_{TAG}_summary.csv", all_summary)
    _write_csv(STAGE / f"paper2_stage1_behavior_{TAG}_paired.csv", all_paired)
    _write_csv(STAGE / f"paper2_stage1_behavior_{TAG}_strong_cases.csv", all_strong)

    with (STAGE / f"paper2_stage1_behavior_{TAG}_decision.json").open("w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, ensure_ascii=False)

    lines = [
        "# Paper2 Stage1 Behavior Viability",
        "",
        f"Generated: {decision['created_at']}",
        "",
        "## Result",
        "",
        "This is a behavior-level viability screen, not a mechanism claim. It tests whether replacing the image with a wrong image changes target rank/logit/margin/decoded answer under the same question and answer target.",
        "",
        "## Model Summary",
        "",
        "| model | paired samples | rank worsened | mean target logit drop | mean margin drop | decoded changed | clean exact | wrong-image exact | strong cases |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, stats in decision["models"].items():
        lines.append(
            "| {model} | {paired_samples} | {rank_worsened_frac:.3f} | {logit_drop_mean:.3f} | {margin_drop_mean:.3f} | {decoded_changed_frac:.3f} | {clean_exact_answer_frac:.3f} | {wrong_exact_answer_frac:.3f} | {strong_cases} |".format(
                model=model, **stats
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Both models show image-dependent behavior: wrong-image substitution usually worsens target rank/logit and changes decoded answers.",
            "- Qwen has higher clean exact-answer rate in this screen, but both models lose answer support under wrong images.",
            "- The selected strong cases are the recommended input for Paper2 Stage2 mechanism smoke; they should be treated as case-selection diagnostics, not final evidence.",
            "",
            "## Outputs",
            "",
            f"- `paper2_stage1_behavior_{TAG}_summary.csv`",
            f"- `paper2_stage1_behavior_{TAG}_paired.csv`",
            f"- `paper2_stage1_behavior_{TAG}_strong_cases.csv`",
            f"- `paper2_stage1_behavior_{TAG}_decision.json`",
        ]
    )
    (STAGE / f"001_paper2_stage1_behavior_viability_{TAG}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
