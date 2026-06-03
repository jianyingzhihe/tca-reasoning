#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("E:/Bridging")
DEFAULT_STAGE = DEFAULT_ROOT / "doc/experiments/stage2"
DEFAULT_CROSS = DEFAULT_STAGE / "cross_model"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _model_matched_status(decision: dict[str, Any], model: str) -> tuple[str, str]:
    model_decision = decision.get("model_decisions", {}).get(model, {})
    statuses = [
        model_decision.get("restore_combo_minus_delta_status", ""),
        model_decision.get("restore_combo_minus_activation_status", ""),
        model_decision.get("corrupt_combo_minus_delta_status", ""),
        model_decision.get("corrupt_combo_minus_activation_status", ""),
    ]
    stable = sum(status == "stable_positive" for status in statuses)
    weak = sum(status in {"stable_positive", "weak_or_heterogeneous_positive"} for status in statuses)
    if stable >= 3:
        return "matched_specificity_supported", f"{stable}/4 stable positive"
    if weak >= 3:
        return "matched_specificity_partial", f"{weak}/4 positive or weak-positive"
    return "matched_specificity_not_supported", f"{weak}/4 positive or weak-positive"


def _control_status(decision: dict[str, Any], model: str) -> tuple[str, str]:
    item = decision.get("models", {}).get(model, {})
    verdict = item.get("verdict", "missing")
    statuses = item.get("direction_statuses", {})
    return verdict, json.dumps(statuses, ensure_ascii=False)


def _tier1_status(matched: str, wrong: str, shuffled: str, model: str) -> str:
    wrong_ok = wrong.endswith("_supported")
    shuffled_ok = shuffled.endswith("_supported")
    if matched == "matched_specificity_supported" and wrong_ok and shuffled_ok:
        return "tier1_hidden_full_supported"
    if wrong_ok and shuffled_ok and matched in {"matched_specificity_supported", "matched_specificity_partial"}:
        return "tier1_hidden_smaller_or_partial_supported" if model == "llava" else "tier1_hidden_mostly_supported"
    if wrong_ok or shuffled_ok:
        return "tier1_hidden_partial"
    return "tier1_hidden_not_supported"


def build_table(cross_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = _read_json(cross_dir / "stage2m_manifest_summary.json")
    matched = _read_json(cross_dir / "stage2m_matched_control_decision.json")
    wrong = _read_json(cross_dir / "stage2m_wrong_target_decision.json")
    shuffled = _read_json(cross_dir / "stage2m_mask_shuffled_decision.json")
    hidden = _read_json(cross_dir / "stage2m_hidden_position_patch_decision.json")
    table = []
    for model in ["qwen", "llava"]:
        matched_status, matched_note = _model_matched_status(matched, model)
        wrong_status, wrong_note = _control_status(wrong, model)
        shuffled_status, shuffled_note = _control_status(shuffled, model)
        tier1 = _tier1_status(matched_status, wrong_status, shuffled_status, model)
        table.append(
            {
                "model_family": model,
                "tier1_status": tier1,
                "matched_control_status": matched_status,
                "matched_control_note": matched_note,
                "wrong_target_status": wrong_status,
                "wrong_target_note": wrong_note,
                "mask_shuffled_status": shuffled_status,
                "mask_shuffled_note": shuffled_note,
                "hidden_patch_decision": json.dumps(hidden.get(model, {}), ensure_ascii=False),
            }
        )
    verdict = {
        "manifest": manifest,
        "tier1_table": table,
        "overall_status": _overall_status(table),
        "claim_boundary": (
            "Stage 2M Tier 1 can establish expanded hidden-state bridge replication. "
            "It cannot establish feature-level or Gemma-style source-control route replication."
        ),
    }
    return table, verdict


def _overall_status(table: list[dict[str, Any]]) -> str:
    statuses = {row["model_family"]: row["tier1_status"] for row in table}
    qwen_ok = statuses.get("qwen") in {"tier1_hidden_full_supported", "tier1_hidden_mostly_supported"}
    llava_ok = statuses.get("llava") in {"tier1_hidden_full_supported", "tier1_hidden_smaller_or_partial_supported", "tier1_hidden_mostly_supported"}
    if qwen_ok and llava_ok:
        return "cross_model_hidden_replication_supported_qwen_stronger_llava_smaller"
    if qwen_ok:
        return "qwen_hidden_replication_supported_llava_not_yet"
    return "tier1_hidden_replication_not_established"


def _read_summary_rows(path: Path, target_group: str) -> list[dict[str, str]]:
    return [row for row in _read_csv(path) if row.get("group_name") == target_group]


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")).replace("\n", " ") for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def write_doc(stage_dir: Path, cross_dir: Path, table: list[dict[str, Any]], verdict: dict[str, Any]) -> None:
    manifest = verdict.get("manifest", {})
    wrong_rows = _read_summary_rows(cross_dir / "stage2m_wrong_target_summary.csv", "top_hidden_delta_plus_answer_adjacent")
    shuffled_rows = _read_summary_rows(cross_dir / "stage2m_mask_shuffled_summary.csv", "top_hidden_delta_plus_answer_adjacent")
    matched_rows = _read_csv(cross_dir / "stage2m_matched_control_model_summary.csv")
    tier_cols = ["model_family", "tier1_status", "matched_control_status", "wrong_target_status", "mask_shuffled_status"]
    control_cols = [
        "model_family",
        "direction",
        "group_name",
        "n_rows",
        "mean_correct_minus_wrong_logit",
        "mean_real_minus_shuffled_logit",
        "status",
    ]
    wrong_compact = [
        {
            "model_family": row.get("model_family"),
            "direction": row.get("direction"),
            "group_name": row.get("group_name"),
            "n_rows": row.get("n_rows"),
            "mean_correct_minus_wrong_logit": row.get("mean_correct_minus_wrong_logit"),
            "mean_real_minus_shuffled_logit": "",
            "status": row.get("status"),
        }
        for row in wrong_rows
    ]
    shuffled_compact = [
        {
            "model_family": row.get("model_family"),
            "direction": row.get("direction"),
            "group_name": row.get("group_name"),
            "n_rows": row.get("n_rows"),
            "mean_correct_minus_wrong_logit": "",
            "mean_real_minus_shuffled_logit": row.get("mean_real_minus_shuffled_logit"),
            "status": row.get("status"),
        }
        for row in shuffled_rows
    ]
    doc = f"""# Stage 2M-1：Expanded Hidden-State Cross-Model Replication

## 1. 目的

本实验把 Stage 2L 的 12-sample hidden bridge 证据扩到 24 个 localized samples，目标是验证：

```text
Qwen 和 LLaVA 是否都存在 evidence-sensitive、target-specific、evidence-location-specific hidden-state bridge。
```

这一步只判断 Tier 1 hidden-state replication，不判断 feature-level causal bridge，也不判断 Gemma-style source-control route replication。

## 2. Manifest

```json
{json.dumps(manifest, ensure_ascii=False, indent=2)}
```

类型配比没有强行伪造：现有样本只有 11 个 `symbol_text_reading` 和 2 个 `scene_inference` 可用，因此缺口由高分 `visual_readout` 补齐。

## 3. Tier 1 Verdict

{_markdown_table(table, tier_cols)}

Overall:

```text
{verdict.get("overall_status")}
```

## 4. Wrong-Target 与 Mask-Shuffled 负控制

{_markdown_table(wrong_compact + shuffled_compact, control_cols)}

## 5. Matched-Control 读法

Matched-control model summary 文件：

```text
doc/experiments/stage2/cross_model/stage2m_matched_control_model_summary.csv
```

Qwen 若在 delta/activation matched controls 下仍稳定为正，可写成 stronger cross-model auxiliary replication。

LLaVA 若 wrong-target 和 mask-shuffled 成立，但 delta-matched controls 仍吸收部分效果，应写成 smaller-effect / partial specificity，而不是强 full source-control replication。

## 6. 结论边界

可写：

```text
Qwen and LLaVA both show expanded hidden-state bridge replication with target-specific and evidence-location-specific controls; Qwen is stronger, while LLaVA shows a smaller but still positive hidden-state bridge.
```

不可写：

```text
Qwen/LLaVA fully replicate Gemma source-control causal routes.
```
"""
    (stage_dir / "049_stage2m_expanded_hidden_bridge_replication.md").write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--cross-dir", type=Path, default=DEFAULT_CROSS)
    parser.add_argument("--out-table", type=Path, default=DEFAULT_CROSS / "stage2m_full_replication_tier1_table.csv")
    parser.add_argument("--out-decision", type=Path, default=DEFAULT_CROSS / "stage2m_full_replication_tier1_decision.json")
    args = parser.parse_args()
    table, verdict = build_table(args.cross_dir)
    _write_csv(
        args.out_table,
        table,
        [
            "model_family",
            "tier1_status",
            "matched_control_status",
            "matched_control_note",
            "wrong_target_status",
            "wrong_target_note",
            "mask_shuffled_status",
            "mask_shuffled_note",
            "hidden_patch_decision",
        ],
    )
    _write_json(args.out_decision, verdict)
    write_doc(args.stage_dir, args.cross_dir, table, verdict)
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
