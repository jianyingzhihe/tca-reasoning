#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


SYMBOL_PATTERNS = [
    r"\bsign\b",
    r"\bsigns\b",
    r"\bwritten\b",
    r"\bwriting\b",
    r"\bword\b",
    r"\bwords\b",
    r"\btext\b",
    r"\blabel\b",
    r"\bbrand\b",
    r"\blogo\b",
    r"\binsignia\b",
    r"\bsymbol\b",
    r"\bnumber\b",
    r"\bletter\b",
    r"\bflag\b",
    r"\bjersey number\b",
    r"\blicense plate\b",
    r"\bwhat does .* say\b",
]

VISUAL_PATTERNS = [
    r"\bhow many\b",
    r"\bhow old\b",
    r"\bwhat color\b",
    r"\bwhich color\b",
    r"\bwhat colour\b",
    r"\bpattern\b",
    r"\bwearing\b",
    r"\bholding\b",
    r"\bin his hand\b",
    r"\bin her hand\b",
    r"\bequipment\b",
    r"\bposition\b",
    r"\bbehind the plate\b",
    r"\bwhat animal\b",
    r"\bwhich animal\b",
    r"\bwhat kind of animal\b",
    r"\bwhat type of animal\b",
    r"\bwhat sport\b",
    r"\bplaying\b",
    r"\bwhat is .* called\b",
    r"\bwhat is the name of .* in this picture\b",
    r"\bwho is .* in this picture\b",
    r"\bwho is playing\b",
    r"\bwhat is .* in his hand\b",
    r"\bwhat is .* in her hand\b",
]

CONCRETE_ANSWER_PATTERNS = [
    r"^[a-z0-9 -]{1,24}$",
]

DIFFUSE_PATTERNS = [
    r"\bweather\b",
    r"\bclimate\b",
    r"\bterrain\b",
    r"\bhabitat\b",
    r"\bnatural habitat\b",
    r"\bpurpose\b",
    r"\bused for\b",
    r"\broom used\b",
    r"\bkind of resort\b",
    r"\bwhat place\b",
    r"\bwhere\b",
    r"\btime of day\b",
    r"\bmain color tint\b",
    r"\bphoto\b.*\btint\b",
    r"\bprobably in a high rise\b",
    r"\bfamous\w*\b",
    r"\bthreatened\b",
    r"\bpresident\b",
    r"\binvented\b",
    r"\bevolve\b",
    r"\bevolved\b",
    r"\bclosely related\b",
    r"\bopposite\b",
    r"\bcola wars\b",
    r"\bwhat american animal\b",
    r"\bwhich .* sisters\b",
    r"\bwhat kind of food\b",
    r"\bwhat does .* eat\b",
    r"\bcalories\b",
    r"\bstomach\b",
    r"\bstomachs\b",
    r"\busually\b",
    r"\bnormally\b",
    r"\bfemale of this animal\b",
    r"\bcompound word\b",
    r"\bderived from\b",
    r"\btypically carry\b",
    r"\bcould they do\b",
    r"\bwho is playing\?\s*$",
    r"\bwhat city\b",
    r"\bwhat country might\b",
    r"\bwhat country is\b",
    r"\bwhat language\b",
]

BAD_ANSWER_PATTERNS = [
    r"\bunknown\b",
    r"\bnone\b",
    r"\bcan't tell\b",
    r"\bnot sure\b",
]


def _match_any(text: str, patterns: list[str]) -> bool:
    text = str(text).lower()
    return any(re.search(pattern, text) for pattern in patterns)


def _op(question: str) -> str:
    if _match_any(question, SYMBOL_PATTERNS):
        return "symbol_text_reading"
    if _match_any(question, VISUAL_PATTERNS):
        return "visual_readout"
    return "uncertain_localized"


def _answer_concrete(answer: str) -> bool:
    answer = str(answer).lower().strip()
    if not answer or _match_any(answer, BAD_ANSWER_PATTERNS):
        return False
    return any(re.search(pattern, answer) for pattern in CONCRETE_ANSWER_PATTERNS)


def _load_exclude(paths: list[str], manual: list[str]) -> set[str]:
    out = {sid.strip() for sid in manual if sid.strip()}
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "sample_id" in df.columns:
            out.update(df["sample_id"].dropna().astype(str).tolist())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an expanded localized discovery pool including heuristic new320 candidates.")
    parser.add_argument("--round400-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=32)
    parser.add_argument("--exclude-sample-csv", action="append", default=[])
    parser.add_argument("--exclude-sample-id", action="append", default=[])
    parser.add_argument("--include-multi-region", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(Path(args.round400_manifest).expanduser().resolve())
    exclude = _load_exclude(args.exclude_sample_csv, args.exclude_sample_id)

    out = df.copy()
    out["question_lower"] = out["question_text"].astype(str).str.lower()
    out["answer_lower"] = out["answer_text"].astype(str).str.lower()
    out["heuristic_reasoning_operation"] = out["question_text"].map(_op)
    out["heuristic_diffuse"] = out["question_text"].map(lambda text: _match_any(text, DIFFUSE_PATTERNS))
    out["answer_concrete"] = out["answer_text"].map(_answer_concrete)
    legacy_visual = out["legacy_visual_type_label"].fillna("").astype(str).str.lower()
    legacy_knowledge = out["legacy_knowledge_level_label"].fillna("").astype(str).str.lower()
    out["legacy_localized_ok"] = legacy_visual.eq("localized") | (legacy_visual.eq("multi_region") & bool(args.include_multi_region))
    out["heuristic_localized_ok"] = (
        legacy_visual.eq("")
        & out["heuristic_reasoning_operation"].isin(["symbol_text_reading", "visual_readout"])
        & out["answer_concrete"]
    )
    out["not_high_knowledge"] = ~legacy_knowledge.eq("high")
    out["excluded_existing_or_manual"] = out["sample_id"].astype(str).isin(exclude)
    out["screen_tier"] = "reject"
    old_ok = out["legacy_localized_ok"] & out["not_high_knowledge"]
    new_ok = out["heuristic_localized_ok"]
    out.loc[old_ok, "screen_tier"] = "legacy_localized"
    out.loc[new_ok, "screen_tier"] = "heuristic_new"
    out["screen_candidate"] = (
        out["screen_tier"].ne("reject")
        & ~out["heuristic_diffuse"]
        & ~out["excluded_existing_or_manual"]
        & out["heuristic_reasoning_operation"].isin(["symbol_text_reading", "visual_readout"])
    )

    priority_score = out["priority"].map({"high": 3, "medium": 2, "low": 1, "new": 1}).fillna(0)
    knowledge_score = legacy_knowledge.map({"low": 3, "medium": 2, "": 2, "high": -5}).fillna(0)
    op_score = out["heuristic_reasoning_operation"].map({"symbol_text_reading": 4, "visual_readout": 3}).fillna(0)
    tier_score = out["screen_tier"].map({"legacy_localized": 20, "heuristic_new": 12, "reject": 0}).fillna(0)
    old_rank_bonus = out["source_round"].astype(str).eq("old80").astype(int) * 2
    out["discovery_score"] = (
        out["screen_candidate"].astype(float) * 100
        + tier_score
        + priority_score * 4
        + knowledge_score * 3
        + op_score * 4
        + old_rank_bonus
        - out["heuristic_diffuse"].astype(float) * 80
    )
    out = out.sort_values(
        ["screen_candidate", "discovery_score", "source_round", "rank", "sample_id"],
        ascending=[False, False, True, True, True],
    )
    selected = out[out["screen_candidate"]].head(args.max_rows).copy()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "expanded_localized_discovery_pool_all.csv", index=False)
    selected.to_csv(out_dir / "expanded_localized_discovery_pool_top.csv", index=False)

    cols = [
        "sample_id",
        "source_round",
        "screen_tier",
        "priority",
        "bucket",
        "question_text",
        "answer_text",
        "image_filename",
        "legacy_visual_type_label",
        "legacy_knowledge_level_label",
        "heuristic_reasoning_operation",
        "discovery_score",
    ]
    lines = [
        "# Expanded Localized Discovery Pool",
        "",
        "This pool expands beyond old legacy-localized labels by admitting heuristic-localized new320 candidates.",
        "",
        f"- total rows: `{len(out)}`",
        f"- screen candidates: `{int(out['screen_candidate'].sum())}`",
        f"- exported top rows: `{len(selected)}`",
        f"- include multi_region: `{args.include_multi_region}`",
        "",
        "Candidate tiers:",
        "",
        *out[out["screen_candidate"]]["screen_tier"].value_counts().to_markdown().splitlines(),
        "",
        "Top candidates:",
        "",
        *selected[cols].to_markdown(index=False).splitlines(),
        "",
        "Use: run B/D eval + answer-aligned trace first; only ask for masks after clean source/control screening.",
    ]
    (out_dir / "EXPANDED_LOCALIZED_DISCOVERY_POOL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {out_dir}")
    print(f"[summary] screen_candidates={int(out['screen_candidate'].sum())} selected={len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
