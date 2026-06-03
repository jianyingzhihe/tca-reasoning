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
STAGE4_CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"
STAGE6_CROSS = ROOT / "doc" / "experiments" / "stage6" / "cross_model"
STAGE6_PREFIX = "stage6_unified"
QWEN_SOURCE_MANIFEST = STAGE4_CROSS / "stage4_qwen_route_first_primary_manifest.csv"
GEMMA_PREFIX = "stage6_gemma_prompt_text_cot"

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

SMOKE_SAMPLES = {"okvqa_val_00327", "okvqa_val_03979"}
SMOKE_PROMPTS = {"B_direct", "A_step_visual"}
SMOKE_VARIANTS = {"original", "paraphrase_1"}


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


def _fieldnames(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    extras = sorted({key for row in rows for key in row if key not in preferred})
    return [key for key in preferred if any(key in row for row in rows)] + extras


def _condition_id(sample_id: str, variant: str, prompt: str) -> str:
    return f"{sample_id}__{variant}__{prompt}"


def _sample_from_condition(condition_id: str) -> str:
    parts = condition_id.split("__", 1)
    return parts[0]


def _is_visual_source(row: dict[str, str]) -> bool:
    try:
        pos = int(float(row.get("source_pos", "-1")))
        visual_count = int(float(row.get("visual_position_count", "-1")))
    except ValueError:
        return False
    return visual_count > 0 and 0 <= pos < visual_count


def build_qwen_routeidentity_manifest(tag: str, dry_run: bool = False) -> tuple[Path, dict[str, Any]]:
    source_rows = _read_csv(QWEN_SOURCE_MANIFEST)
    if not source_rows:
        raise FileNotFoundError(QWEN_SOURCE_MANIFEST)

    rows: list[dict[str, Any]] = []
    for base in source_rows:
        sample_id = base.get("sample_id", "")
        if sample_id not in SAMPLES:
            continue
        if base.get("include_pool", "1") != "1":
            continue
        if not _is_visual_source(base):
            continue
        layer = int(_f(base.get("layer"), -1))
        if layer < 10 or layer > 17:
            continue
        for variant in QUESTION_VARIANTS:
            for prompt in PROMPT_FAMILIES:
                out = dict(base)
                out["candidate_id"] = f"stage6_routeidentity_{tag}_{base['candidate_id']}_{variant}_{prompt}"
                out["stage6_base_candidate_id"] = base["candidate_id"]
                out["stage6_question_variant"] = variant
                out["stage6_prompt_family"] = prompt
                out["stage6_sample_type"] = SAMPLES[sample_id]["type"]
                out["stage6_condition_id"] = _condition_id(sample_id, variant, prompt)
                out["prompt_name"] = prompt
                out["question_text"] = SAMPLES[sample_id][variant]
                out["run"] = "stage6_unified_routeidentity"
                out["analysis_group"] = "stage6_routeidentity_broad_visual_pool"
                out["stage6_include_smoke"] = (
                    "1"
                    if sample_id in SMOKE_SAMPLES
                    and variant in SMOKE_VARIANTS
                    and prompt in SMOKE_PROMPTS
                    and layer in {10, 13, 15}
                    else "0"
                )
                out["stage6_include_full"] = "1"
                rows.append(out)

    preferred = [
        "candidate_id",
        "stage6_base_candidate_id",
        "stage6_condition_id",
        "stage6_sample_type",
        "stage6_question_variant",
        "stage6_prompt_family",
        "stage6_include_smoke",
        "stage6_include_full",
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
        "target_token_id",
        "target_token",
        "wrong_token_id",
        "wrong_token",
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
        "visual_position_count",
    ]
    out_path = STAGE6_CROSS / f"{STAGE6_PREFIX}_routeidentity_{tag}_manifest.csv"
    summary = {
        "updated_at": _now(),
        "tag": tag,
        "kind": "qwen_routeidentity",
        "source_manifest": str(QWEN_SOURCE_MANIFEST),
        "rows": len(rows),
        "smoke_rows": sum(1 for row in rows if row["stage6_include_smoke"] == "1"),
        "unique_samples": len({row["sample_id"] for row in rows}),
        "unique_base_candidates": len({row["stage6_base_candidate_id"] for row in rows}),
        "layer_counts": dict(Counter(str(row["layer"]) for row in rows)),
        "prompt_counts": dict(Counter(row["stage6_prompt_family"] for row in rows)),
        "variant_counts": dict(Counter(row["stage6_question_variant"] for row in rows)),
        "design_note": "Qwen route-identity proxy: broad visual candidate pool is revalidated per Stage6 prompt/text condition, then topK route identity is compared with B_direct/original.",
    }
    if not dry_run:
        _write_csv(out_path, rows, _fieldnames(rows, preferred))
        _write_json(STAGE6_CROSS / f"{STAGE6_PREFIX}_routeidentity_{tag}_manifest_summary.json", summary)
    return out_path, summary


def _gemma_paths(source_tag: str) -> dict[str, Path]:
    prefix = f"{GEMMA_PREFIX}_full_{source_tag}"
    return {
        "manifest": STAGE6_CROSS / f"{GEMMA_PREFIX}_{source_tag}_manifest.csv",
        "meta_a": STAGE6_CROSS / f"{prefix}_meta_a.csv",
        "nodes": STAGE6_CROSS / f"{prefix}_nodes_detailed_controlled.csv",
        "compare": STAGE6_CROSS / f"{prefix}_sample_compare_controlled.csv",
    }


def _prefix_ok(meta: dict[str, str]) -> bool:
    return bool((meta.get("assistant_prefix") or "").strip())


def _format_ok(meta: dict[str, str]) -> bool:
    text = (meta.get("generated_text") or "").lower()
    return "the answer is" in text


def _select_gemma_baseline_nodes(source_tag: str, nodes_per_sample: int, controls_per_node: int) -> dict[str, list[dict[str, Any]]]:
    paths = _gemma_paths(source_tag)
    nodes = _read_csv(paths["nodes"])
    if not nodes:
        raise FileNotFoundError(paths["nodes"])

    by_original: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in nodes:
        sample_id = row.get("sample_id", "")
        if not sample_id.endswith("__original__B_direct"):
            continue
        if row.get("run") != "A":
            continue
        if row.get("node_type") != "feature":
            continue
        original = _sample_from_condition(sample_id)
        by_original[original].append(row)

    selected: dict[str, list[dict[str, Any]]] = {}
    for original, rows in by_original.items():
        rows.sort(key=lambda row: _f(row.get("path_mass_best")), reverse=True)
        picked = []
        for row in rows[:nodes_per_sample]:
            controls = []
            for control in rows:
                if control is row:
                    continue
                if control.get("feature_id") == row.get("feature_id") and control.get("pos") == row.get("pos"):
                    continue
                controls.append(
                    f"{int(_f(control.get('layer'), -1))}:{int(_f(control.get('pos'), -1))}:{int(_f(control.get('feature_id'), -1))}"
                )
                if len(controls) >= controls_per_node:
                    break
            picked.append(
                {
                    "baseline_node_id": row.get("node_id", ""),
                    "source_layer": int(_f(row.get("layer"), -1)),
                    "source_pos": int(_f(row.get("pos"), -1)),
                    "source_feature_id": int(_f(row.get("feature_id"), -1)),
                    "source_path_mass_best": _f(row.get("path_mass_best")),
                    "control_nodes": "|".join(controls),
                }
            )
        selected[original] = picked
    return selected


def build_gemma_fixednode_manifest(
    tag: str,
    source_tag: str,
    nodes_per_sample: int,
    controls_per_node: int,
    dry_run: bool = False,
) -> tuple[Path, dict[str, Any]]:
    paths = _gemma_paths(source_tag)
    manifest_rows = {row["sample_id"]: row for row in _read_csv(paths["manifest"]) if row.get("sample_id")}
    meta_rows = {row["sample_id"]: row for row in _read_csv(paths["meta_a"]) if row.get("sample_id")}
    compare_rows = {row["sample_id"]: row for row in _read_csv(paths["compare"]) if row.get("sample_id")}
    if not manifest_rows or not meta_rows:
        raise FileNotFoundError(f"missing Gemma Stage6 source rows for tag={source_tag}")

    baseline_nodes = _select_gemma_baseline_nodes(source_tag, nodes_per_sample, controls_per_node)
    baseline_target_by_original: dict[str, str] = {}
    for condition_id, meta in meta_rows.items():
        if condition_id.endswith("__original__B_direct"):
            baseline_target_by_original[_sample_from_condition(condition_id)] = meta.get("target_token_id", "")

    rows: list[dict[str, Any]] = []
    for condition_id, meta in sorted(meta_rows.items()):
        original = _sample_from_condition(condition_id)
        if original not in SAMPLES:
            continue
        manifest = manifest_rows.get(condition_id, {})
        compare = compare_rows.get(condition_id, {})
        parts = condition_id.split("__")
        variant = parts[1] if len(parts) > 1 else manifest.get("stage6_question_variant", "")
        prompt = parts[2] if len(parts) > 2 else manifest.get("stage6_prompt_family", "")
        baseline_target = baseline_target_by_original.get(original, "")
        for idx, node in enumerate(baseline_nodes.get(original, []), start=1):
            row = {
                "candidate_id": f"stage6_fixednode_{tag}_{condition_id}_n{idx:02d}",
                "stage6_source_tag": source_tag,
                "stage6_original_sample_id": original,
                "stage6_condition_id": condition_id,
                "stage6_question_variant": variant,
                "stage6_prompt_family": prompt,
                "stage6_sample_type": manifest.get("stage6_sample_type", SAMPLES[original]["type"]),
                "stage6_include_smoke": (
                    "1"
                    if original in SMOKE_SAMPLES and variant in SMOKE_VARIANTS and prompt in SMOKE_PROMPTS and idx <= 2
                    else "0"
                ),
                "stage6_include_full": "1",
                "sample_id": condition_id,
                "question": meta.get("question", ""),
                "assistant_prefix": meta.get("assistant_prefix", ""),
                "generated_text": meta.get("generated_text", ""),
                "answer_text": meta.get("answer_text", manifest.get("answer_text", "")),
                "target_token_id": meta.get("target_token_id", ""),
                "target_token_text": meta.get("target_token_text", ""),
                "baseline_target_token_id": baseline_target,
                "target_token_same": "1" if baseline_target and meta.get("target_token_id") == baseline_target else "0",
                "prefix_ok": "1" if _prefix_ok(meta) else "0",
                "format_ok": "1" if _format_ok(meta) else "0",
                "local_image_path": manifest.get("image_path", ""),
                "image_filename": manifest.get("image_filename", ""),
                "a_target_token_id": compare.get("a_target_token_id", ""),
                "b_target_token_id": compare.get("b_target_token_id", ""),
                "old_node_overlap_jaccard": compare.get("node_overlap_jaccard", ""),
                "old_edge_overlap_jaccard": compare.get("edge_overlap_jaccard", ""),
                **node,
            }
            rows.append(row)

    preferred = [
        "candidate_id",
        "stage6_source_tag",
        "stage6_original_sample_id",
        "stage6_condition_id",
        "stage6_question_variant",
        "stage6_prompt_family",
        "stage6_sample_type",
        "stage6_include_smoke",
        "stage6_include_full",
        "sample_id",
        "source_layer",
        "source_pos",
        "source_feature_id",
        "source_path_mass_best",
        "control_nodes",
        "target_token_id",
        "baseline_target_token_id",
        "target_token_same",
        "prefix_ok",
        "format_ok",
        "question",
        "assistant_prefix",
        "answer_text",
        "local_image_path",
        "image_filename",
    ]
    out_path = STAGE6_CROSS / f"{STAGE6_PREFIX}_fixednode_{tag}_manifest.csv"
    summary = {
        "updated_at": _now(),
        "tag": tag,
        "kind": "gemma_fixednode",
        "source_tag": source_tag,
        "nodes_per_sample": nodes_per_sample,
        "controls_per_node": controls_per_node,
        "rows": len(rows),
        "smoke_rows": sum(1 for row in rows if row["stage6_include_smoke"] == "1"),
        "usable_aligned_rows": sum(1 for row in rows if row["prefix_ok"] == "1" and row["target_token_same"] == "1"),
        "unique_samples": len({row["stage6_original_sample_id"] for row in rows}),
        "prompt_counts": dict(Counter(row["stage6_prompt_family"] for row in rows)),
        "variant_counts": dict(Counter(row["stage6_question_variant"] for row in rows)),
        "format_flag_counts": dict(Counter(f"prefix{row['prefix_ok']}_target{row['target_token_same']}" for row in rows)),
        "design_note": "Gemma fixed-node probe: B_direct/original traced feature nodes are fixed and ablated across Stage6 prompt/text conditions. Main analysis must filter prefix_ok=1 and target_token_same=1.",
    }
    if not dry_run:
        _write_csv(out_path, rows, _fieldnames(rows, preferred))
        _write_json(STAGE6_CROSS / f"{STAGE6_PREFIX}_fixednode_{tag}_manifest_summary.json", summary)
    return out_path, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage6 unified metric manifests.")
    parser.add_argument("--tag", default="unified_v1")
    parser.add_argument("--gemma-source-tag", default="gemmaprompt_v1_mf8_sharded")
    parser.add_argument("--gemma-nodes-per-sample", type=int, default=4)
    parser.add_argument("--gemma-controls-per-node", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    qwen_path, qwen_summary = build_qwen_routeidentity_manifest(args.tag, dry_run=args.dry_run)
    gemma_path, gemma_summary = build_gemma_fixednode_manifest(
        args.tag,
        source_tag=args.gemma_source_tag,
        nodes_per_sample=args.gemma_nodes_per_sample,
        controls_per_node=args.gemma_controls_per_node,
        dry_run=args.dry_run,
    )
    payload = {
        "updated_at": _now(),
        "tag": args.tag,
        "dry_run": args.dry_run,
        "qwen_manifest": str(qwen_path),
        "gemma_manifest": str(gemma_path),
        "qwen": qwen_summary,
        "gemma": gemma_summary,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
