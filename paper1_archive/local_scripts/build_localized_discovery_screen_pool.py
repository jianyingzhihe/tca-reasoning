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
    r"\btext\b",
    r"\blabel\b",
    r"\bbrand\b",
    r"\blogo\b",
    r"\binsignia\b",
    r"\bsymbol\b",
    r"\blanguage\b",
    r"\bnumber\b",
    r"\bletter\b",
    r"\bsay\b",
    r"\bsays\b",
    r"\bmentioned\b",
]

VISUAL_PATTERNS = [
    r"\bcolor\b",
    r"\bcolour\b",
    r"\bhow many\b",
    r"\bhow old\b",
    r"\bpattern\b",
    r"\bequipment\b",
    r"\bin his hand\b",
    r"\bin her hand\b",
    r"\bwearing\b",
    r"\bplaying\b",
    r"\banimal\b",
    r"\bperson\b",
    r"\bwho is\b",
    r"\bwhat is .* called\b",
    r"\bwhich .* is\b",
]

DIFFUSE_PATTERNS = [
    r"\bweather\b",
    r"\bclimate\b",
    r"\bterrain\b",
    r"\bhabitat\b",
    r"\bpurpose\b",
    r"\bused for\b",
    r"\broom used\b",
    r"\bkind of resort\b",
    r"\bwhat place\b",
    r"\btime of day\b",
    r"\bmain color tint\b",
    r"\bprobably in a high rise\b",
    r"\bfamous\w*\b",
    r"\bthreatened\b",
    r"\bpresident\b",
    r"\bevolve\b",
    r"\bevolved\b",
    r"\bclosely related\b",
    r"\bopposite\b",
    r"\bcola wars\b",
    r"\binvented\b",
    r"\bnatural habitat\b",
    r"\bwhat american animal\b",
    r"\bwhich .* sisters\b",
]


def _match_any(text: str, patterns: list[str]) -> bool:
    text = str(text).lower()
    return any(re.search(p, text) for p in patterns)


def _heuristic_operation(question: str) -> str:
    if _match_any(question, SYMBOL_PATTERNS):
        return "symbol_text_reading"
    if _match_any(question, VISUAL_PATTERNS):
        return "visual_readout"
    return "uncertain_localized"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a conservative localized discovery-screen pool from round4-400 legacy labels.")
    parser.add_argument("--round400-manifest", required=True)
    parser.add_argument("--exclude-sample-csv", action="append", default=[])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=40)
    parser.add_argument("--diffuse-sample-id", action="append", default=[])
    args = parser.parse_args()

    df = pd.read_csv(Path(args.round400_manifest).expanduser().resolve())
    exclude = set(args.diffuse_sample_id)
    for path in args.exclude_sample_csv:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            continue
        ex = pd.read_csv(p)
        if "sample_id" in ex.columns:
            exclude.update(ex["sample_id"].dropna().astype(str).tolist())

    out = df.copy()
    out["question_lower"] = out["question_text"].astype(str).str.lower()
    out["heuristic_reasoning_operation"] = out["question_text"].map(_heuristic_operation)
    out["heuristic_diffuse"] = out["question_text"].map(lambda x: _match_any(x, DIFFUSE_PATTERNS))
    out["legacy_localized"] = out["legacy_visual_type_label"].astype(str).str.lower().eq("localized")
    out["not_high_knowledge"] = ~out["legacy_knowledge_level_label"].astype(str).str.lower().eq("high")
    out["excluded_existing_or_diffuse"] = out["sample_id"].astype(str).isin(exclude)
    out["screen_candidate"] = (
        out["legacy_localized"]
        & out["not_high_knowledge"]
        & ~out["heuristic_diffuse"]
        & ~out["excluded_existing_or_diffuse"]
        & out["heuristic_reasoning_operation"].isin(["symbol_text_reading", "visual_readout"])
    )
    priority_score = out["priority"].map({"high": 3, "medium": 2, "low": 1}).fillna(0)
    knowledge_score = out["legacy_knowledge_level_label"].map({"low": 3, "medium": 2, "high": 1}).fillna(0)
    op_score = out["heuristic_reasoning_operation"].map({"symbol_text_reading": 3, "visual_readout": 2}).fillna(0)
    out["discovery_score"] = (
        out["screen_candidate"].astype(float) * 100
        + priority_score * 5
        + knowledge_score * 3
        + op_score * 4
        - out["heuristic_diffuse"].astype(float) * 50
    )
    out = out.sort_values(["screen_candidate", "discovery_score", "rank"], ascending=[False, False, True])

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "round400_localized_discovery_pool_all.csv", index=False)
    selected = out[out["screen_candidate"]].head(args.max_rows).copy()
    selected.to_csv(out_dir / "round400_localized_discovery_pool_top.csv", index=False)

    cols = [
        "sample_id",
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
        "# Round400 Localized Discovery-Screen Pool",
        "",
        "This is a heuristic pool for clean-intervention discovery only. It is not a final typed analysis table.",
        "",
        f"- total rows: `{len(out)}`",
        f"- screen candidates: `{int(out['screen_candidate'].sum())}`",
        f"- exported top rows: `{len(selected)}`",
        "",
        "Top candidates:",
        "",
        *selected[cols].to_markdown(index=False).splitlines(),
        "",
        "Use: run a cheap clean source-intervention / control screen first. Only annotate candidates that pass node-quality filters.",
    ]
    (out_dir / "ROUND400_LOCALIZED_DISCOVERY_POOL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {out_dir}")
    print(f"[summary] screen_candidates={int(out['screen_candidate'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
