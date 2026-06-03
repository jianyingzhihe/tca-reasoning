#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
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


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _float(value: Any) -> float | None:
    try:
        if value == "" or value is None:
            return None
        return float(value)
    except Exception:
        return None


def _mean(values: list[float | None]) -> float | str:
    valid = [value for value in values if value is not None]
    if not valid:
        return ""
    return round(mean(valid), 6)


def _sum_bool(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if _bool(row.get(key)))


def _semantic_target_hit(row: dict[str, Any]) -> bool:
    target = str(row.get("target_answer", "")).strip().lower()
    pred = str(row.get("predicted_answer", "")).strip().lower()
    synonyms = {
        "china": ["china", "中国", "chinese"],
        "dog": ["dog"],
        "samsung": ["samsung"],
    }.get(target, [target])
    return any(item and item in pred for item in synonyms)


def _group_rows(rows: list[dict[str, str]], keys: list[str]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    return grouped


def _summarize(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, items in sorted(_group_rows(rows, ["model_family", "direction", "group_name", "group_kind"]).items()):
        model_family, direction, group_name, group_kind = key
        logit_restore = [
            (_float(row.get("prompt_target_logit")) or 0.0) - (_float(row.get("union_target_logit")) or 0.0)
            for row in items
            if _float(row.get("prompt_target_logit")) is not None and _float(row.get("union_target_logit")) is not None
        ]
        rank_restore = [
            (_float(row.get("union_target_rank")) or 0.0) - (_float(row.get("prompt_target_rank")) or 0.0)
            for row in items
            if _float(row.get("prompt_target_rank")) is not None and _float(row.get("union_target_rank")) is not None
        ]
        out.append(
            {
                "model_family": model_family,
                "direction": direction,
                "group_name": group_name,
                "group_kind": group_kind,
                "n_rows": len(items),
                "target_hit_n": _sum_bool(items, "target_hit"),
                "semantic_target_hit_n": sum(1 for row in items if _semantic_target_hit(row)),
                "answer_changed_vs_clean_n": _sum_bool(items, "answer_changed_vs_clean"),
                "answer_changed_vs_union_n": _sum_bool(items, "answer_changed_vs_union"),
                "same_as_clean_n": sum(1 for row in items if row.get("predicted_answer") == row.get("clean_generated_answer")),
                "same_as_union_n": sum(1 for row in items if row.get("predicted_answer") == row.get("union_generated_answer")),
                "mean_prompt_target_logit": _mean([_float(row.get("prompt_target_logit")) for row in items]),
                "mean_prompt_target_rank": _mean([_float(row.get("prompt_target_rank")) for row in items]),
                "mean_first_step_target_rank": _mean([_float(row.get("first_step_target_rank")) for row in items]),
                "mean_logit_restore_vs_union": round(mean(logit_restore), 6) if logit_restore else "",
                "mean_rank_restore_vs_union": round(mean(rank_restore), 6) if rank_restore else "",
                "unique_predicted_answers": " | ".join(sorted({row.get("predicted_answer", "") for row in items})),
            }
        )
    return out


def _case_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        prompt_rank = _float(row.get("prompt_target_rank"))
        union_rank = _float(row.get("union_target_rank"))
        prompt_logit = _float(row.get("prompt_target_logit"))
        union_logit = _float(row.get("union_target_logit"))
        out.append(
            {
                "model_family": row.get("model_family"),
                "sample_id": row.get("sample_id"),
                "prompt_name": row.get("prompt_name"),
                "condition": row.get("condition"),
                "target_answer": row.get("target_answer"),
                "clean_generated_answer": row.get("clean_generated_answer"),
                "union_generated_answer": row.get("union_generated_answer"),
                "predicted_answer": row.get("predicted_answer"),
                "target_hit": row.get("target_hit"),
                "semantic_target_hit": _semantic_target_hit(row),
                "answer_changed_vs_union": row.get("answer_changed_vs_union"),
                "prompt_target_rank": row.get("prompt_target_rank"),
                "rank_restore_vs_union": "" if prompt_rank is None or union_rank is None else union_rank - prompt_rank,
                "logit_restore_vs_union": "" if prompt_logit is None or union_logit is None else prompt_logit - union_logit,
                "first_generated_token": row.get("first_generated_token"),
                "generated_continuation": row.get("generated_continuation"),
            }
        )
    return out


def _decision(rows: list[dict[str, str]], summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "claim_boundary": (
            "Decoded answer smoke only. A positive result can support generation-level bridge smoke, "
            "but cannot establish source-control causal route replication."
        ),
        "model_decisions": {},
    }
    by_model = _group_rows(rows, ["model_family"])
    summary_lookup = {
        (row["model_family"], row["direction"], row["group_name"]): row
        for row in summary_rows
    }
    for (model_family,), model_rows in sorted(by_model.items()):
        baseline_clean = [row for row in model_rows if row.get("condition") == "baseline::clean"]
        informative = [row for row in baseline_clean if row.get("clean_generated_answer") != row.get("union_generated_answer")]
        best = summary_lookup.get((model_family, "restore", "top_hidden_delta_plus_answer_adjacent"), {})
        low = summary_lookup.get((model_family, "restore", "low_delta_control"), {})
        random = summary_lookup.get((model_family, "restore", "random_control_1"), {})
        best_rows = [
            row
            for row in model_rows
            if row.get("condition") == "restore::top_hidden_delta_plus_answer_adjacent"
        ]
        best_informative = [
            row
            for row in best_rows
            if row.get("clean_generated_answer") != row.get("union_generated_answer")
        ]
        best_to_clean = sum(
            1
            for row in best_informative
            if row.get("predicted_answer") == row.get("clean_generated_answer")
            and row.get("predicted_answer") != row.get("union_generated_answer")
        )
        best_target_hits_informative = sum(1 for row in best_informative if _bool(row.get("target_hit")))
        status = "rank_bridge_only_no_generation_change"
        if best_to_clean >= 2:
            status = "partial_generation_bridge_smoke"
        elif len(informative) == 0:
            status = "decoded_generation_underpowered_no_clean_union_answer_gap"
        out["model_decisions"][model_family] = {
            "status": status,
            "informative_clean_vs_union_rows": len(informative),
            "best_group": "top_hidden_delta_plus_answer_adjacent",
            "best_target_hit_n": best.get("target_hit_n", ""),
            "best_semantic_target_hit_n": best.get("semantic_target_hit_n", ""),
            "best_answer_changed_vs_union_n": best.get("answer_changed_vs_union_n", ""),
            "best_same_as_clean_on_informative_n": best_to_clean,
            "best_target_hits_on_informative_n": best_target_hits_informative,
            "best_mean_logit_restore_vs_union": best.get("mean_logit_restore_vs_union", ""),
            "best_mean_rank_restore_vs_union": best.get("mean_rank_restore_vs_union", ""),
            "low_delta_target_hit_n": low.get("target_hit_n", ""),
            "random_control_target_hit_n": random.get("target_hit_n", ""),
            "low_delta_semantic_target_hit_n": low.get("semantic_target_hit_n", ""),
            "random_control_semantic_target_hit_n": random.get("semantic_target_hit_n", ""),
            "reading": (
                "Decoded answer bridge is at most partial; use first-token/rank bridge as the stronger Stage 2H evidence."
                if status != "partial_generation_bridge_smoke"
                else "Decoded answer bridge smoke is partially supported on informative clean-vs-union rows."
            ),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage 2H decoded answer smoke outputs.")
    parser.add_argument("--inputs", required=True, help="Comma-separated decoded answer CSV files.")
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--out-case-table", required=True)
    parser.add_argument("--out-decision", required=True)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for raw_path in args.inputs.split(","):
        path = Path(raw_path.strip())
        if path.exists():
            rows.extend(_read_csv(path))
    summary = _summarize(rows)
    cases = _case_table(rows)
    decision = _decision(rows, summary)
    _write_csv(
        Path(args.out_summary),
        summary,
        [
            "model_family",
            "direction",
            "group_name",
            "group_kind",
            "n_rows",
            "target_hit_n",
            "semantic_target_hit_n",
            "answer_changed_vs_clean_n",
            "answer_changed_vs_union_n",
            "same_as_clean_n",
            "same_as_union_n",
            "mean_prompt_target_logit",
            "mean_prompt_target_rank",
            "mean_first_step_target_rank",
            "mean_logit_restore_vs_union",
            "mean_rank_restore_vs_union",
            "unique_predicted_answers",
        ],
    )
    _write_csv(
        Path(args.out_case_table),
        cases,
        [
            "model_family",
            "sample_id",
            "prompt_name",
            "condition",
            "target_answer",
            "clean_generated_answer",
            "union_generated_answer",
            "predicted_answer",
            "target_hit",
            "semantic_target_hit",
            "answer_changed_vs_union",
            "prompt_target_rank",
            "rank_restore_vs_union",
            "logit_restore_vs_union",
            "first_generated_token",
            "generated_continuation",
        ],
    )
    _write_json(Path(args.out_decision), decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
