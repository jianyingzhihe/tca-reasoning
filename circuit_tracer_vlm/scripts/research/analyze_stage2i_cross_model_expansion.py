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
    if not path.exists():
        return []
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


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _float(value: Any) -> float | None:
    try:
        if value == "" or value is None:
            return None
        return float(value)
    except Exception:
        return None


def _mean(values: list[float]) -> float | str:
    return round(mean(values), 6) if values else ""


def _group(rows: list[dict[str, str]], keys: list[str]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    out: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[tuple(row.get(key, "") for key in keys)].append(row)
    return out


def _baseline_keys(rows: list[dict[str, str]]) -> set[tuple[str, str, str]]:
    return {
        (row.get("model_family", ""), row.get("sample_id", ""), row.get("prompt_name", ""))
        for row in rows
        if row.get("condition") == "baseline::clean"
    }


def _is_informative(row: dict[str, str]) -> bool:
    return bool(row.get("clean_generated_answer")) and row.get("clean_generated_answer") != row.get("union_generated_answer")


def _restore_values(rows: list[dict[str, str]]) -> tuple[list[float], list[float]]:
    logit: list[float] = []
    rank: list[float] = []
    for row in rows:
        prompt_logit = _float(row.get("prompt_target_logit"))
        union_logit = _float(row.get("union_target_logit"))
        prompt_rank = _float(row.get("prompt_target_rank"))
        union_rank = _float(row.get("union_target_rank"))
        if prompt_logit is not None and union_logit is not None:
            logit.append(prompt_logit - union_logit)
        if prompt_rank is not None and union_rank is not None:
            rank.append(union_rank - prompt_rank)
    return logit, rank


def _summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for (model, direction, group_name, group_kind), items in sorted(_group(rows, ["model_family", "direction", "group_name", "group_kind"]).items()):
        logit, rank = _restore_values(items)
        informative_items = [row for row in items if _is_informative(row)]
        out.append(
            {
                "model_family": model,
                "direction": direction,
                "group_name": group_name,
                "group_kind": group_kind,
                "n_rows": len(items),
                "informative_rows": len(informative_items),
                "target_hit_n": sum(1 for row in items if _bool(row.get("target_hit"))),
                "changed_away_from_union_n": sum(1 for row in items if _bool(row.get("answer_changed_vs_union"))),
                "changed_away_from_union_informative_n": sum(1 for row in informative_items if _bool(row.get("answer_changed_vs_union"))),
                "same_as_clean_n": sum(1 for row in items if row.get("predicted_answer") == row.get("clean_generated_answer")),
                "same_as_clean_informative_n": sum(
                    1 for row in informative_items if row.get("predicted_answer") == row.get("clean_generated_answer")
                ),
                "mean_logit_restore_vs_union": _mean(logit),
                "mean_rank_restore_vs_union": _mean(rank),
                "unique_predicted_answers": " | ".join(sorted({row.get("predicted_answer", "") for row in items if row.get("predicted_answer", "")})),
            }
        )
    return out


def _case_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        prompt_logit = _float(row.get("prompt_target_logit"))
        union_logit = _float(row.get("union_target_logit"))
        prompt_rank = _float(row.get("prompt_target_rank"))
        union_rank = _float(row.get("union_target_rank"))
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
                "informative_clean_vs_union": _is_informative(row),
                "target_hit": row.get("target_hit"),
                "answer_changed_vs_union": row.get("answer_changed_vs_union"),
                "logit_restore_vs_union": "" if prompt_logit is None or union_logit is None else round(prompt_logit - union_logit, 6),
                "rank_restore_vs_union": "" if prompt_rank is None or union_rank is None else round(union_rank - prompt_rank, 6),
                "first_generated_token": row.get("first_generated_token"),
                "generated_continuation": row.get("generated_continuation"),
            }
        )
    return out


def _decision(rows: list[dict[str, str]], summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {(row["model_family"], row["direction"], row["group_name"]): row for row in summary_rows}
    out: dict[str, Any] = {
        "claim_boundary": (
            "Stage 2I expansion can strengthen cross-model decoded/hidden-state bridge evidence, "
            "but it is still below Gemma-style source-control causal route replication."
        ),
        "model_decisions": {},
    }
    for (model,), model_rows in sorted(_group(rows, ["model_family"]).items()):
        baseline_clean = [row for row in model_rows if row.get("condition") == "baseline::clean"]
        informative_keys = {
            (row.get("sample_id", ""), row.get("prompt_name", ""))
            for row in baseline_clean
            if _is_informative(row)
        }
        total_runs = len(_baseline_keys(model_rows))
        best = lookup.get((model, "restore", "top_hidden_delta_plus_answer_adjacent"), {})
        low = lookup.get((model, "restore", "low_delta_control"), {})
        random = lookup.get((model, "restore", "random_control_1"), {})
        best_away = int(best.get("changed_away_from_union_informative_n") or 0)
        control_away = max(
            int(low.get("changed_away_from_union_informative_n") or 0),
            int(random.get("changed_away_from_union_informative_n") or 0),
        )
        if best:
            if len(informative_keys) >= 4 and best_away > control_away:
                status = "partial_decoded_bridge_expanded"
            elif len(informative_keys) < 4:
                status = "underpowered_generation_gap_for_decoded_bridge"
            else:
                status = "decoded_bridge_not_specific_over_controls"
        else:
            status = "generation_gap_screen_only_pass" if len(informative_keys) >= 4 else "generation_gap_screen_underpowered"
        out["model_decisions"][model] = {
            "status": status,
            "total_sample_prompt_runs": total_runs,
            "informative_clean_vs_union_runs": len(informative_keys),
            "best_group": "top_hidden_delta_plus_answer_adjacent" if best else "",
            "best_changed_away_from_union_informative_n": best.get("changed_away_from_union_informative_n", ""),
            "best_same_as_clean_informative_n": best.get("same_as_clean_informative_n", ""),
            "best_target_hit_n": best.get("target_hit_n", ""),
            "best_mean_logit_restore_vs_union": best.get("mean_logit_restore_vs_union", ""),
            "best_mean_rank_restore_vs_union": best.get("mean_rank_restore_vs_union", ""),
            "low_delta_changed_away_from_union_informative_n": low.get("changed_away_from_union_informative_n", ""),
            "random_changed_away_from_union_informative_n": random.get("changed_away_from_union_informative_n", ""),
            "reading": (
                "Use as decoded bridge expansion only; do not write source-control route replication."
                if best
                else "Use as generation-gap screening result before running bridge conditions."
            ),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage 2I cross-model expansion decoded/gap outputs.")
    parser.add_argument("--inputs", required=True, help="Comma-separated CSV outputs from decode/gap runs.")
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--out-case-table", required=True)
    parser.add_argument("--out-decision", required=True)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for raw in args.inputs.split(","):
        path = Path(raw.strip())
        rows.extend(_read_csv(path))
    summary = _summary(rows)
    case_table = _case_table(rows)
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
            "informative_rows",
            "target_hit_n",
            "changed_away_from_union_n",
            "changed_away_from_union_informative_n",
            "same_as_clean_n",
            "same_as_clean_informative_n",
            "mean_logit_restore_vs_union",
            "mean_rank_restore_vs_union",
            "unique_predicted_answers",
        ],
    )
    _write_csv(
        Path(args.out_case_table),
        case_table,
        [
            "model_family",
            "sample_id",
            "prompt_name",
            "condition",
            "target_answer",
            "clean_generated_answer",
            "union_generated_answer",
            "predicted_answer",
            "informative_clean_vs_union",
            "target_hit",
            "answer_changed_vs_union",
            "logit_restore_vs_union",
            "rank_restore_vs_union",
            "first_generated_token",
            "generated_continuation",
        ],
    )
    _write_json(Path(args.out_decision), decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

