#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
STAGE6 = ROOT / "doc" / "experiments" / "stage6" / "cross_model"
STAGE4_CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"
ROUTE_FIRST_PREFIX = "stage4_qwen_route_first"
OUT_PREFIX = "stage6_prompt_text_cot"

PROMPT_FAMILIES = ["B_direct", "D_visual_only", "C_step_only", "A_step_visual"]
QUESTION_VARIANTS = ["original", "paraphrase_1", "paraphrase_2"]

SAMPLES: dict[str, dict[str, str]] = {
    "okvqa_val_216885": {
        "type": "visual_readout",
        "original": "Is redwood or cedar more prevalent as a siding material?",
        "paraphrase_1": "Which siding material appears more common here, redwood or cedar?",
        "paraphrase_2": "Looking at the siding, does it look more like redwood or cedar?",
    },
    "okvqa_val_1499095": {
        "type": "visual_readout",
        "original": "What did the animal just finishing doing?",
        "paraphrase_1": "What did the animal just finish doing?",
        "paraphrase_2": "What action has the animal apparently just completed?",
    },
    "okvqa_val_4943285": {
        "type": "visual_readout",
        "original": "What does this animal eat?",
        "paraphrase_1": "What food does this animal eat?",
        "paraphrase_2": "What is this animal likely eating?",
    },
    "okvqa_val_4075245": {
        "type": "visual_readout",
        "original": "What is the plate made out of?",
        "paraphrase_1": "What material is the plate made of?",
        "paraphrase_2": "What does the plate appear to be made from?",
    },
    "okvqa_val_4837235": {
        "type": "visual_readout",
        "original": "What sport could they do with these?",
        "paraphrase_1": "Which sport could someone do with these items?",
        "paraphrase_2": "What sport are these used for?",
    },
    "okvqa_val_03979": {
        "type": "symbol_text_reading",
        "original": "A beverage is represented here what brand is its opposite in the cola wars?",
        "paraphrase_1": "Which cola brand is the rival of the beverage shown here?",
        "paraphrase_2": "In the cola wars, what brand is opposite to the beverage represented here?",
    },
    "okvqa_val_4987585": {
        "type": "symbol_text_reading",
        "original": "Which brand of bike is rided by the person in the photo?",
        "paraphrase_1": "What brand is the bike being ridden in the photo?",
        "paraphrase_2": "Which bike brand does the rider appear to be using?",
    },
    "okvqa_val_00327": {
        "type": "compact_scene_inference",
        "original": "What is this vehicle used for?",
        "paraphrase_1": "What purpose is this vehicle used for?",
        "paraphrase_2": "What is the main use of this vehicle?",
    },
    "okvqa_val_1440035": {
        "type": "compact_scene_inference",
        "original": "The red items on the pastry are a specific ingredient for what type of cake?",
        "paraphrase_1": "The red pieces on the pastry point to what kind of cake?",
        "paraphrase_2": "What cake type is suggested by the red ingredient on the pastry?",
    },
    "okvqa_val_147735": {
        "type": "compact_scene_inference",
        "original": "Why type of restaurant would serve this food?",
        "paraphrase_1": "What type of restaurant would serve this food?",
        "paraphrase_2": "This food would most likely be served at what kind of restaurant?",
    },
    "okvqa_val_03609": {
        "type": "compact_scene_inference",
        "original": "What is the shower made of?",
        "paraphrase_1": "What material is the shower made from?",
        "paraphrase_2": "What does the shower appear to be made of?",
    },
    "okvqa_val_03085": {
        "type": "compact_scene_inference",
        "original": "What is this machine used for?",
        "paraphrase_1": "What is the purpose of this machine?",
        "paraphrase_2": "What is this machine mainly used to do?",
    },
}

SMOKE_SAMPLES = {
    "okvqa_val_216885",
    "okvqa_val_1499095",
    "okvqa_val_03979",
    "okvqa_val_00327",
}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
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


def _f(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _route_score(row: dict[str, str]) -> float:
    return (
        _f(row.get("clean_source_minus_controls"))
        + _f(row.get("restore_source_minus_controls"))
        + _f(row.get("real_minus_shifted"))
        + _f(row.get("real_minus_shuffled"))
        + _f(row.get("evidence_specificity"))
        + max(_f(row.get("clean_correct_minus_wrong")), _f(row.get("restore_correct_minus_wrong")))
    )


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "candidate_id",
        "stage6_base_candidate_id",
        "stage6_route_score",
        "stage6_sample_type",
        "stage6_question_variant",
        "stage6_prompt_family",
        "stage6_source_prompt_name",
        "stage6_include_smoke",
        "stage6_include_full",
        "stage6_visual_source_only",
        "pack",
        "analysis_group",
        "sample_id",
        "run",
        "prompt_name",
        "source_zeroing_mode",
        "source_node_id",
        "layer",
        "source_pos",
        "source_feature_id",
        "position_group",
        "best_real_condition",
        "target_token_id",
        "target_token",
        "wrong_token_id",
        "wrong_token",
        "clean_target_logit",
        "clean_target_rank",
        "question_text",
        "answer_text",
        "reasoning_operation",
        "image_dependence_tier",
        "image_filename",
        "local_image_path",
        "mask_dir",
        "answer_mask_path",
        "union_mask_path",
        "shifted_mask_path",
        "shuffled_mask_path",
        "known_damaging_feature_ids_for_prompt_run",
        "visual_position_count",
    ]
    extras = sorted({key for row in rows for key in row if key not in preferred})
    return [key for key in preferred if any(key in row for row in rows)] + extras


def _manifest_index(pack: str) -> dict[str, dict[str, str]]:
    path = STAGE4_CROSS / f"{ROUTE_FIRST_PREFIX}_{pack}_manifest.csv"
    rows = _read_csv(path)
    if not rows:
        raise FileNotFoundError(f"missing route-first manifest: {path}")
    return {row.get("candidate_id", ""): row for row in rows if row.get("candidate_id")}


def _route_candidates(source_tag: str) -> list[dict[str, str]]:
    path = STAGE4_CROSS / f"{ROUTE_FIRST_PREFIX}_{source_tag}_route_candidates.csv"
    rows = _read_csv(path)
    if not rows:
        raise FileNotFoundError(f"missing route-first candidate table: {path}")
    return rows


def build(tag: str, source_tag: str, nodes_per_sample: int, dry_run: bool = False) -> dict[str, Any]:
    primary_manifest = _manifest_index("primary")
    route_rows = _route_candidates(source_tag)
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for route in route_rows:
        if route.get("pack") != "primary" or route.get("mode") != "full":
            continue
        if route.get("route_first_evidence_gold") != "1":
            continue
        sample_id = route.get("sample_id", "")
        if sample_id not in SAMPLES:
            continue
        base = primary_manifest.get(route.get("candidate_id", ""))
        if not base:
            continue
        source_pos = int(_f(base.get("source_pos"), -1))
        visual_count = int(_f(base.get("visual_position_count"), -1))
        if source_pos < 0 or visual_count <= 0 or source_pos >= visual_count:
            continue
        merged: dict[str, Any] = dict(base)
        merged.update(route)
        merged["stage6_route_score"] = _route_score(route)
        merged["stage6_visual_source_only"] = "1"
        by_sample[sample_id].append(merged)

    rows: list[dict[str, Any]] = []
    base_counts: Counter[str] = Counter()
    for sample_id in sorted(SAMPLES):
        nodes = sorted(by_sample.get(sample_id, []), key=lambda row: _f(row.get("stage6_route_score")), reverse=True)
        nodes = nodes[:nodes_per_sample]
        for node_rank, node in enumerate(nodes, start=1):
            for question_variant in QUESTION_VARIANTS:
                question = SAMPLES[sample_id][question_variant]
                for prompt_family in PROMPT_FAMILIES:
                    out = dict(node)
                    out["stage6_base_candidate_id"] = node.get("candidate_id", "")
                    out["candidate_id"] = (
                        f"stage6_{tag}_{node.get('candidate_id', '')}_n{node_rank:02d}_"
                        f"{question_variant}_{prompt_family}"
                    )
                    out["stage6_sample_type"] = SAMPLES[sample_id]["type"]
                    out["stage6_question_variant"] = question_variant
                    out["stage6_prompt_family"] = prompt_family
                    out["stage6_source_prompt_name"] = node.get("prompt_name", "")
                    out["stage6_include_smoke"] = (
                        "1"
                        if sample_id in SMOKE_SAMPLES
                        and node_rank <= 4
                        and question_variant in {"original", "paraphrase_1"}
                        and prompt_family in {"B_direct", "A_step_visual"}
                        else "0"
                    )
                    out["stage6_include_full"] = "1"
                    out["prompt_name"] = prompt_family
                    out["question_text"] = question
                    out["run"] = "stage6_prompt_text_cot"
                    out["analysis_group"] = "stage6_visual_source_route_first"
                    rows.append(out)
                    base_counts[sample_id] += 1

    summary = {
        "updated_at": _now(),
        "tag": tag,
        "source_tag": source_tag,
        "nodes_per_sample": nodes_per_sample,
        "samples_requested": len(SAMPLES),
        "samples_with_nodes": len({row["sample_id"] for row in rows}),
        "manifest_rows": len(rows),
        "smoke_rows": sum(1 for row in rows if row.get("stage6_include_smoke") == "1"),
        "type_counts": dict(Counter(row.get("stage6_sample_type", "") for row in rows)),
        "layer_counts": dict(Counter(str(row.get("layer", "")) for row in rows)),
        "prompt_counts": dict(Counter(row.get("prompt_name", "") for row in rows)),
        "variant_counts": dict(Counter(row.get("stage6_question_variant", "") for row in rows)),
        "sample_row_counts": dict(base_counts),
        "claim_boundary": (
            "Exploratory Qwen Stage6 prompt/text/CoT probe. Candidate features come from "
            "Qwen route-first evidence-gold nodes and are restricted to visual source positions."
        ),
    }
    if dry_run:
        summary["dry_run"] = True
        return summary
    if not rows:
        raise RuntimeError("no Stage6 rows built")
    out_manifest = STAGE6 / f"{OUT_PREFIX}_{tag}_manifest.csv"
    out_summary = STAGE6 / f"{OUT_PREFIX}_{tag}_manifest_summary.json"
    _write_csv(out_manifest, rows, _fieldnames(rows))
    summary["manifest"] = str(out_manifest)
    _write_json(out_summary, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage6 prompt/text/CoT exploratory Qwen manifest.")
    parser.add_argument("--tag", default="prompttext_v1")
    parser.add_argument("--source-tag", default="routefirst_v1")
    parser.add_argument("--nodes-per-sample", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = build(args.tag, args.source_tag, args.nodes_per_sample, args.dry_run)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
