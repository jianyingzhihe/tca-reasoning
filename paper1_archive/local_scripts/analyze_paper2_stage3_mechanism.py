#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
PAPER2 = ROOT / "doc" / "experiments" / "paper2"
STAGE1 = PAPER2 / "stage1"
STAGE2 = PAPER2 / "stage2"
STAGE3 = PAPER2 / "stage3_mechanism"
STAGE4_CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"
STAGE6_CROSS = ROOT / "doc" / "experiments" / "stage6" / "cross_model"
PREFIX = "paper2_stage3_qwen_plt_mechanism"
HIDDEN_TAG = "paper2_stage2_hidden_expanded_v1"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "model",
        "lens",
        "stage1_label",
        "rows",
        "samples",
        "positive_frac",
        "sample_id",
        "layer",
        "candidate_id",
        "source_pos",
        "source_feature_id",
        "target_effect",
        "rank_effect",
        "score",
        "clean_source_minus_controls",
        "real_source_effect",
        "real_minus_shifted",
        "real_minus_shuffled",
        "real_target_minus_wrong",
        "status",
    ]
    extras = sorted({key for row in rows for key in row if key not in preferred})
    fields = [key for key in preferred if any(key in row for row in rows)] + extras
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _f(raw: Any, default: float = math.nan) -> float:
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _mean(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return statistics.fmean(clean) if clean else math.nan


def _labels() -> dict[tuple[str, str], str]:
    return {(row["model_family"], row["sample_id"]): row["stage1_label"] for row in _read_csv(STAGE1 / "composition_effect_by_sample_mask_full_v1.csv")}


def _hidden_rows(model: str) -> list[dict[str, Any]]:
    if model == "qwen":
        path = STAGE4_CROSS / f"stage4_qwen_decisive_route_hidden_primary_smoke_{HIDDEN_TAG}_raw.csv"
    else:
        path = STAGE6_CROSS / f"stage6_gemma_hidden_lattice_primary_smoke_{HIDDEN_TAG}_raw.csv"
    rows = []
    for row in _read_csv(path):
        if row.get("token_scored") != "target" or row.get("direction") != "restore":
            continue
        rows.append(
            {
                "model": model,
                "lens": "hidden_restore",
                "sample_id": row.get("sample_id", ""),
                "layer": row.get("layer", ""),
                "mask_condition": row.get("mask_condition", ""),
                "position_group": row.get("position_group", ""),
                "target_effect": _f(row.get("logit_effect")),
                "rank_effect": _f(row.get("rank_effect")),
                "target_minus_wrong": math.nan,
                "stage1_label": _labels().get((model, row.get("sample_id", "")), ""),
                "status": "ok",
            }
        )
    return rows


def _qwen_plt_candidate_rows(tag: str, mode: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(STAGE3.glob(f"{PREFIX}_primary_{mode}_L*_{tag}_candidates.csv")):
        layer = path.name.split("_L", 1)[1].split("_", 1)[0]
        for row in _read_csv(path):
            out.append(
                {
                    "model": "qwen",
                    "lens": "plt_candidate_discovery",
                    "sample_id": row.get("sample_id", ""),
                    "layer": layer,
                    "candidate_id": row.get("candidate_id", ""),
                    "source_pos": row.get("source_pos", ""),
                    "source_feature_id": row.get("source_feature_id", ""),
                    "score": _f(row.get("evidence_first_score") or row.get("damage_score") or row.get("target_contribution")),
                    "clean_target_rank": _f(row.get("clean_target_rank") or row.get("original_target_rank")),
                    "stage1_label": _labels().get(("qwen", row.get("sample_id", "")), ""),
                    "status": "ok",
                }
            )
    return out


def _qwen_candidate_index(tag: str, mode: str) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for path in sorted(STAGE3.glob(f"{PREFIX}_primary_{mode}_L*_{tag}_candidates.csv")):
        layer = path.name.split("_L", 1)[1].split("_", 1)[0]
        for row in _read_csv(path):
            item = dict(row)
            item["layer"] = item.get("layer") or layer
            out[(layer, item.get("candidate_id", ""))] = item
    return out


def _qwen_plt_zeroing_rows(tag: str, mode: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_candidate: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    cindex = _qwen_candidate_index(tag, mode)
    for path in sorted(STAGE3.glob(f"{PREFIX}_primary_{mode}_L*_{tag}_zeroing_raw.csv")):
        layer = path.name.split("_L", 1)[1].split("_", 1)[0]
        for row in _read_csv(path):
            if row.get("status", "ok") == "ok":
                by_candidate[(layer, row.get("candidate_id", ""))].append(row)
    for (layer, cid), rows in by_candidate.items():
        first = rows[0]
        cand = cindex.get((layer, cid), {})
        def effect(kind: str, mask: str, control: str, token: str = "target") -> float:
            for row in rows:
                if (
                    row.get("intervention_kind") == kind
                    and row.get("mask_condition") == mask
                    and row.get("control_group") == control
                    and row.get("token_scored") == token
                ):
                    return _f(row.get("logit_effect"))
            return math.nan

        clean_source = effect("clean_zeroing", "clean", "source")
        clean_controls = [
            effect("clean_zeroing", "clean", "same_position_matched_feature_control"),
            effect("clean_zeroing", "clean", "same_feature_random_position_control"),
            effect("clean_zeroing", "clean", "random_active_feature_control"),
        ]
        real_source = _mean([effect("mask_restore", "answer_mask", "source"), effect("mask_restore", "union_mask", "source")])
        shifted = effect("mask_restore", "shifted_mask", "source")
        shuffled = effect("mask_restore", "shuffled_mask", "source")
        wrong_real = _mean([effect("mask_restore", "answer_mask", "source", "wrong"), effect("mask_restore", "union_mask", "source", "wrong")])
        out.append(
            {
                "model": "qwen",
                "lens": "plt_feature_zeroing_restore",
                "sample_id": first.get("sample_id", "") or cand.get("sample_id", ""),
                "layer": layer,
                "candidate_id": cid,
                "source_pos": first.get("source_pos", "") or cand.get("source_pos", ""),
                "source_feature_id": first.get("source_feature_id", "") or cand.get("source_feature_id", ""),
                "clean_source_minus_controls": clean_source - _mean(clean_controls),
                "real_source_effect": real_source,
                "real_minus_shifted": real_source - shifted,
                "real_minus_shuffled": real_source - shuffled,
                "real_target_minus_wrong": real_source - wrong_real,
                "stage1_label": _labels().get(("qwen", first.get("sample_id", "") or cand.get("sample_id", "")), ""),
                "status": "ok",
            }
        )
    return out


def _qwen_plt_group_rows(tag: str, mode: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(STAGE3.glob(f"{PREFIX}_primary_{mode}_L*_{tag}_group_raw.csv")):
        layer = path.name.split("_L", 1)[1].split("_", 1)[0]
        for row in _read_csv(path):
            if row.get("record_type") != "group_restore" or row.get("control_group") != "source":
                continue
            out.append(
                {
                    "model": "qwen",
                    "lens": "plt_group_restore",
                    "sample_id": row.get("sample_id", ""),
                    "layer": layer,
                    "top_k": row.get("top_k", ""),
                    "mask_condition": row.get("mask_condition", ""),
                    "target_effect": _f(row.get("target_logit_effect")),
                    "rank_effect": _f(row.get("target_rank_effect")),
                    "wrong_effect": _f(row.get("wrong_logit_effect")),
                    "real_target_minus_wrong": _f(row.get("target_logit_effect")) - _f(row.get("wrong_logit_effect")),
                    "stage1_label": _labels().get(("qwen", row.get("sample_id", "")), ""),
                    "status": "ok",
                }
            )
    return out


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row.get("model", "")), str(row.get("lens", "")), str(row.get("stage1_label", "")))].append(row)
    summary: list[dict[str, Any]] = []
    for (model, lens, label), bucket in sorted(buckets.items()):
        numeric_keys = [
            "target_effect",
            "rank_effect",
            "wrong_effect",
            "score",
            "clean_source_minus_controls",
            "real_source_effect",
            "real_minus_shifted",
            "real_minus_shuffled",
            "real_target_minus_wrong",
        ]
        row: dict[str, Any] = {
            "model": model,
            "lens": lens,
            "stage1_label": label,
            "rows": len(bucket),
            "samples": len({r.get("sample_id", "") for r in bucket}),
            "positive_frac": _mean([1.0 if any(_f(r.get(k)) > 0 for k in numeric_keys) else 0.0 for r in bucket]),
        }
        for key in numeric_keys:
            vals = [_f(r.get(key)) for r in bucket if key in r]
            if vals:
                row[f"{key}_mean"] = _mean(vals)
        summary.append(row)
    return summary


def _write_docs(tag: str, decision: dict[str, Any], summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Paper2 Stage3 Mechanism Viability",
        "",
        f"Generated: {decision['created_at']}",
        "",
        "## Current Verdict",
        "",
        f"Status: `{decision['status']}`.",
        "",
        "This stage filters Paper2 to behavior-supported multi-evidence cases before asking whether hidden/PLT mechanisms show matching evidence-to-answer movement.",
        "",
        "## Summary Table",
        "",
        "| model | lens | label | rows | samples | key mean |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary:
        key_mean = row.get("real_source_effect_mean", row.get("target_effect_mean", row.get("score_mean", "")))
        lines.append(
            f"| {row['model']} | {row['lens']} | {row['stage1_label']} | {row['rows']} | {row['samples']} | {key_mean} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Hidden-lens rows are inherited from the symmetric Gemma/Qwen Paper2 Stage2 run.",
            "- Qwen PLT rows come from Paper2-specific discovery/validation, not from the old Stage4 paperpack route pool.",
            "- Gemma PLT/source-route remains case-level pending unless a custom compact Gemma Paper2 route runner is launched; with only one Gemma supported case, this should be reported as sample-limited rather than as a model-level negative result.",
        ]
    )
    (STAGE3 / f"003_paper2_stage3_mechanism_viability_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Paper2 Stage3 mechanism outputs.")
    parser.add_argument("--tag", default="paper2_mechanism_v1")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = parser.parse_args()
    tag = args.tag
    mode = args.mode

    rows: list[dict[str, Any]] = []
    rows.extend(_hidden_rows("gemma"))
    rows.extend(_hidden_rows("qwen"))
    rows.extend(_qwen_plt_candidate_rows(tag, mode))
    rows.extend(_qwen_plt_zeroing_rows(tag, mode))
    rows.extend(_qwen_plt_group_rows(tag, mode))
    summary = _summarize(rows)
    qwen_plt_rows = [r for r in rows if r.get("model") == "qwen" and str(r.get("lens", "")).startswith("plt")]
    gemma_supported_hidden = [r for r in rows if r.get("model") == "gemma" and r.get("stage1_label") == "multi_evidence_supported"]
    decision = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": tag,
        "mode": mode,
        "status": f"paper2_stage3_mechanism_{mode}_ready" if qwen_plt_rows and gemma_supported_hidden else "partial_hidden_only_or_pending_plt",
        "total_rows": len(rows),
        "summary_rows": len(summary),
        "qwen_plt_rows": len(qwen_plt_rows),
        "gemma_supported_hidden_rows": len(gemma_supported_hidden),
        "outputs": {
            "rows": str(STAGE3 / f"paper2_stage3_mechanism_rows_{tag}.csv"),
            "summary": str(STAGE3 / f"paper2_stage3_mechanism_summary_{tag}.csv"),
            "decision": str(STAGE3 / f"paper2_stage3_mechanism_{tag}_decision.json"),
        },
    }
    _write_csv(STAGE3 / f"paper2_stage3_mechanism_rows_{tag}.csv", rows)
    _write_csv(STAGE3 / f"paper2_stage3_mechanism_summary_{tag}.csv", summary)
    (STAGE3 / f"paper2_stage3_mechanism_{tag}_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_docs(tag, decision, summary)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
