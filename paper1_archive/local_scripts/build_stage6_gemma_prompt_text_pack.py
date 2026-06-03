#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
STAGE3_CROSS = ROOT / "doc" / "experiments" / "stage3" / "cross_model"
STAGE6_CROSS = ROOT / "doc" / "experiments" / "stage6" / "cross_model"
PREFIX = "stage6_gemma_prompt_text_cot"

PROMPT_TEMPLATES = {
    "B_direct": "{question} Reply with only one short sentence in exactly this format: The answer is <short answer>.",
    "D_visual_only": "{question} Use visual evidence, then reply with only one short sentence in exactly this format: The answer is <short answer>.",
    "C_step_only": "{question} Think step by step, then end with exactly this format: The answer is <short answer>.",
    "A_step_visual": "{question} Use visual evidence and think step by step, then end with exactly this format: The answer is <short answer>.",
}

FORMAT_CLEAN_PROMPT_TEMPLATES = {
    "B_direct": (
        "{question} Reply with exactly one final line and no extra text: "
        "The answer is <short answer>."
    ),
    "D_visual_only": (
        "{question} Use the image evidence briefly if needed. Your final line must be exactly "
        "The answer is <short answer>. Stop after that final line."
    ),
    "C_step_only": (
        "{question} Think briefly in at most two sentences. Your final line must be exactly "
        "The answer is <short answer>. Stop after that final line."
    ),
    "A_step_visual": (
        "{question} Use visual evidence and think briefly in at most two sentences. "
        "Your final line must be exactly The answer is <short answer>. Stop after that final line."
    ),
}

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

SMOKE_SAMPLES = {"okvqa_val_216885", "okvqa_val_03979", "okvqa_val_00327"}
SMOKE_VARIANTS = {"original", "paraphrase_1"}


def _read_csv(path: Path) -> list[dict[str, str]]:
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


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "sample_id",
        "stage6_original_sample_id",
        "stage6_condition_id",
        "stage6_question_variant",
        "stage6_prompt_family",
        "stage6_compare_role",
        "stage6_baseline_prompt_family",
        "stage6_sample_type",
        "stage6_include_smoke",
        "stage6_include_full",
        "question_id",
        "image_id",
        "image_filename",
        "image_path",
        "question",
        "question_text",
        "gold_answer",
        "answer_text",
        "prompt_name",
        "pack",
        "reasoning_operation",
        "image_dependence_tier",
        "paperpack_source",
        "mask_dir",
        "notes",
    ]
    extras = sorted({key for row in rows for key in row if key not in preferred})
    return [key for key in preferred if any(key in row for row in rows)] + extras


def _condition_id(sample_id: str, variant: str, prompt_family: str) -> str:
    return f"{sample_id}__{variant}__{prompt_family}"


def _stage6_meta(base: dict[str, str], sample_id: str, variant: str, prompt_family: str, role: str) -> dict[str, str]:
    cond = _condition_id(sample_id, variant, prompt_family)
    include_smoke = sample_id in SMOKE_SAMPLES and variant in SMOKE_VARIANTS
    return {
        "sample_id": cond,
        "stage6_original_sample_id": sample_id,
        "stage6_condition_id": cond,
        "stage6_question_variant": variant,
        "stage6_prompt_family": prompt_family,
        "stage6_compare_role": role,
        "stage6_baseline_prompt_family": "B_direct",
        "stage6_sample_type": SAMPLES[sample_id]["type"],
        "stage6_include_smoke": "1" if include_smoke else "0",
        "stage6_include_full": "1",
        "pack": "stage6_gemma",
        "paperpack_source": base.get("paperpack_source", "stage6_prompt_text_cot"),
    }


def build(tag: str, dry_run: bool = False) -> dict[str, Any]:
    prompt_templates = FORMAT_CLEAN_PROMPT_TEMPLATES if "formatclean" in tag.lower() else PROMPT_TEMPLATES
    base_path = STAGE3_CROSS / "stage3_gemma_eval_primary_B_direct.csv"
    base_rows = {row["sample_id"]: row for row in _read_csv(base_path)}
    missing = [sample_id for sample_id in SAMPLES if sample_id not in base_rows]
    if missing:
        raise FileNotFoundError(f"Stage6 samples missing from Gemma primary manifest: {missing}")

    a_rows: list[dict[str, Any]] = []
    b_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for sample_id, sample in SAMPLES.items():
        base = base_rows[sample_id]
        baseline_question_text = sample["original"]
        baseline_prompt = prompt_templates["B_direct"].format(question=baseline_question_text)
        for variant in QUESTION_VARIANTS:
            question_text = sample[variant]
            for prompt_family in PROMPT_FAMILIES:
                condition = _condition_id(sample_id, variant, prompt_family)
                meta_a = _stage6_meta(base, sample_id, variant, prompt_family, "A_condition")
                meta_b = _stage6_meta(base, sample_id, variant, prompt_family, "B_baseline")

                a = dict(base)
                a.update(meta_a)
                a["question_text"] = question_text
                a["question"] = prompt_templates[prompt_family].format(question=question_text)
                a["prompt_name"] = f"{prompt_family}__{variant}"
                a["notes"] = (
                    f"{base.get('notes', '')};stage6_condition={condition};"
                    f"stage6_role=A_condition;original_sample_id={sample_id}"
                )
                a_rows.append(a)

                b = dict(base)
                b.update(meta_b)
                b["question_text"] = baseline_question_text
                b["question"] = baseline_prompt
                b["prompt_name"] = "B_direct__baseline_original"
                b["notes"] = (
                    f"{base.get('notes', '')};stage6_condition={condition};"
                    f"stage6_role=B_baseline;original_sample_id={sample_id}"
                )
                b_rows.append(b)

                manifest = {
                    **meta_a,
                    "bucket": f"stage6_gemma_prompt_text_cot_{tag}",
                    "question_text": question_text,
                    "baseline_question_text": baseline_question_text,
                    "answer_text": base.get("answer_text", base.get("gold_answer", "")),
                    "reasoning_operation": base.get("reasoning_operation", ""),
                    "image_dependence_tier": base.get("image_dependence_tier", ""),
                    "image_filename": base.get("image_filename", ""),
                    "image_path": base.get("image_path", ""),
                    "mask_dir": base.get("mask_dir", ""),
                }
                manifest_rows.append(manifest)

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": tag,
        "prefix": PREFIX,
        "source_manifest": str(base_path),
        "rows_a_condition": len(a_rows),
        "rows_b_baseline": len(b_rows),
        "manifest_rows": len(manifest_rows),
        "unique_original_samples": len({row["stage6_original_sample_id"] for row in manifest_rows}),
        "smoke_rows": sum(1 for row in manifest_rows if row["stage6_include_smoke"] == "1"),
        "prompt_families": dict(Counter(row["stage6_prompt_family"] for row in manifest_rows)),
        "question_variants": dict(Counter(row["stage6_question_variant"] for row in manifest_rows)),
        "sample_types": dict(Counter(row["stage6_sample_type"] for row in manifest_rows)),
        "design_note": (
            "A side is the condition prompt; B side is the repeated B_direct/original baseline "
            "with the same synthetic sample_id, enabling one Gemma source-tracing comparison per condition."
        ),
        "format_clean_prompts": "formatclean" in tag.lower(),
    }

    if not dry_run:
        STAGE6_CROSS.mkdir(parents=True, exist_ok=True)
        _write_csv(STAGE6_CROSS / f"{PREFIX}_{tag}_eval_A_condition.csv", a_rows, _fieldnames(a_rows))
        _write_csv(STAGE6_CROSS / f"{PREFIX}_{tag}_eval_B_baseline.csv", b_rows, _fieldnames(b_rows))
        _write_csv(STAGE6_CROSS / f"{PREFIX}_{tag}_manifest.csv", manifest_rows, _fieldnames(manifest_rows))
        _write_json(STAGE6_CROSS / f"{PREFIX}_{tag}_manifest_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage6 Gemma prompt/text/CoT source-tracing pack.")
    parser.add_argument("--tag", default="gemmaprompt_v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.tag, dry_run=args.dry_run), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
