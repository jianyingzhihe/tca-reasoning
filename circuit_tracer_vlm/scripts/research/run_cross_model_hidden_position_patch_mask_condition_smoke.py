#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from run_cross_model_feature_intervention_smoke import (
    SAMPLES,
    _apply_mask,
    _env_presence,
    _first_param_device,
    _gpu_info,
    _llava_bucket_positions,
    _llava_inputs,
    _load_masks,
    _parse_csv,
    _parse_float_csv,
    _processor_patch,
    _prompt,
    _qwen_bucket_positions,
    _qwen_inputs,
    _rank_and_top,
    _replace_hidden,
    _target_candidates,
)
from run_cross_model_hidden_position_patch_smoke import (
    _answer_adjacent_positions,
    _load_sample_manifest,
    _make_groups,
    _patch_hidden_tensor,
    _safe_ratio,
    _scale_label,
)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage2n-mask-position-patch] {message}", flush=True)


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


def _score_with_patch(
    model,
    inputs: dict[str, Any],
    tokenizer,
    target_ids: list[int],
    module,
    intervention: dict[str, Any] | None,
) -> dict[str, Any]:
    handle = None
    if intervention is not None:

        def _hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            patched = _patch_hidden_tensor(hidden, **intervention)
            return _replace_hidden(output, patched)

        handle = module.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=False, use_cache=False)
    finally:
        if handle is not None:
            handle.remove()
    return _rank_and_top(outputs.logits, tokenizer, target_ids)


def _mask_for_condition(mask_info: dict[str, Any], condition: str):
    if condition == "answer_mask":
        return mask_info["answer_mask"]
    if condition == "union_mask":
        return mask_info["union_mask"]
    raise ValueError(f"Unknown mask_condition: {condition}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2N hidden-position patch with answer/union mask conditions.")
    parser.add_argument("--model-family", choices=["qwen", "llava"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--annotation-roots", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--sample-manifest", default="")
    parser.add_argument("--samples", default="manifest")
    parser.add_argument("--prompts", default="B_direct,D_visual_only")
    parser.add_argument("--mask-conditions", default="answer_mask,union_mask")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--scales", default="1.0")
    parser.add_argument("--default-position-count", type=int, default=32)
    parser.add_argument("--max-evidence-positions", type=int, default=64)
    parser.add_argument("--answer-adjacent-count", type=int, default=4)
    parser.add_argument("--random-controls", type=int, default=4)
    parser.add_argument("--answer-prefix", default="The answer is ")
    parser.add_argument("--min-gpu-free-gb", type=float, default=18.0)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    sample_lookup = _load_sample_manifest(Path(args.sample_manifest)) if args.sample_manifest else dict(SAMPLES)
    sample_ids = list(sample_lookup) if args.samples.strip().lower() in {"manifest", "auto", "all"} else _parse_csv(args.samples)
    prompt_names = _parse_csv(args.prompts)
    mask_conditions = _parse_csv(args.mask_conditions)
    scales = _parse_float_csv(args.scales)
    roots = [Path(path) for path in _parse_csv(args.annotation_roots)]
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "created_at": _now(),
        "model_family": args.model_family,
        "model_name": args.model_name,
        "layer": args.layer,
        "scales": scales,
        "mask_conditions": mask_conditions,
        "sample_manifest": args.sample_manifest,
        "samples": sample_ids,
        "prompts": prompt_names,
        "env_presence": _env_presence(),
        "gpu_before": _gpu_info(),
        "groups": [],
        "decision": {},
        "claim_boundary": (
            "Stage 2N hidden-position mask-condition patch only; not CLT feature-level or "
            "Gemma-style source-control route replication."
        ),
    }
    gpu = payload["gpu_before"]
    if not gpu.get("available") or float(gpu.get("free_gb", 0.0)) < args.min_gpu_free_gb:
        payload["decision"] = {"status": "partial", "reason": "insufficient_gpu_free_memory"}
        _write_json(Path(args.out_json), payload)
        return 0

    if args.model_family == "qwen":
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(args.model_name, local_files_only=True)
        tokenizer = processor.tokenizer
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()
        device = _first_param_device(model)
        target_module = model.language_model.layers[args.layer]

        def build_inputs(image: Image.Image, image_path: Path, question: str):
            return _qwen_inputs(processor, image, str(image_path), question, args.answer_prefix, device)

        def bucket_positions(input_ids: list[int], token_texts: list[str]) -> tuple[str, list[int]]:
            buckets = _qwen_bucket_positions(input_ids, token_texts)
            return "image_marker_or_span", buckets["image_marker_or_span"]

        hidden_index = args.layer
    else:
        from transformers import AutoConfig, AutoProcessor, LlavaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(args.model_name)
        config = AutoConfig.from_pretrained(args.model_name)
        payload["processor_patch"] = _processor_patch(processor, config)
        tokenizer = processor.tokenizer
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()
        device = _first_param_device(model)
        target_module = model.language_model.layers[args.layer]
        image_token_id = int(getattr(config, "image_token_index", 32000))

        def build_inputs(image: Image.Image, image_path: Path, question: str):
            del image_path
            return _llava_inputs(processor, image, question, args.answer_prefix, device)

        def bucket_positions(input_ids: list[int], token_texts: list[str]) -> tuple[str, list[int]]:
            del token_texts
            buckets = _llava_bucket_positions(input_ids, image_token_id)
            return "image_token_span", buckets["image_token_span"]

        hidden_index = args.layer + 1

    rows: list[dict[str, Any]] = []
    usable_runs = 0
    skipped: list[dict[str, Any]] = []

    for sample_id in sample_ids:
        sample = sample_lookup.get(sample_id)
        if sample is None:
            skipped.append({"sample_id": sample_id, "reason": "sample_missing_from_manifest"})
            continue
        mask_info = _load_masks(sample, roots, work_dir)
        if mask_info["status"] != "ok":
            skipped.append({"sample_id": sample_id, "reason": "mask_missing", "mask_info": mask_info})
            continue
        image_path = Path(mask_info["image_path"])
        clean_image = Image.open(image_path).convert("RGB")

        for prompt_name in prompt_names:
            question = _prompt(sample["question"], prompt_name)
            clean_inputs = build_inputs(clean_image, image_path, question)
            input_ids = clean_inputs["input_ids"][0].detach().cpu().tolist()
            token_texts = tokenizer.convert_ids_to_tokens(input_ids)
            bucket_name, visual_positions = bucket_positions(input_ids, token_texts)
            answer_positions = _answer_adjacent_positions(len(input_ids), set(visual_positions), args.answer_adjacent_count)
            target_candidates = _target_candidates(tokenizer, sample["answer"])
            target_ids = [int(item["token_id"]) for item in target_candidates]
            if not visual_positions or not target_ids:
                skipped.append({"sample_id": sample_id, "prompt_name": prompt_name, "reason": "positions_or_target_missing"})
                continue

            with torch.inference_mode():
                clean_outputs = model(**clean_inputs, output_hidden_states=True, use_cache=False)
            clean_hidden = clean_outputs.hidden_states[hidden_index].detach()
            clean_score = _rank_and_top(clean_outputs.logits, tokenizer, target_ids)
            chosen_target_id = int(clean_score["target_token_id"])

            for mask_condition in mask_conditions:
                evidence_mask = _mask_for_condition(mask_info, mask_condition)
                masked_image = _apply_mask(clean_image, evidence_mask, (128, 128, 128))
                masked_inputs = build_inputs(masked_image, image_path, question)
                with torch.inference_mode():
                    masked_outputs = model(**masked_inputs, output_hidden_states=True, use_cache=False)
                masked_hidden = masked_outputs.hidden_states[hidden_index].detach()
                masked_score = _rank_and_top(masked_outputs.logits, tokenizer, [chosen_target_id])
                clean_mask_logit_gap = float(clean_score["target_logit"]) - float(masked_score["target_logit"])
                clean_mask_rank_gap = int(masked_score["target_rank"]) - int(clean_score["target_rank"])

                groups, diagnostics = _make_groups(
                    model_family=args.model_family,
                    clean_inputs=clean_inputs,
                    clean_hidden=clean_hidden,
                    union_hidden=masked_hidden,
                    visual_positions=visual_positions,
                    answer_positions=answer_positions,
                    evidence_mask=evidence_mask,
                    sample_id=sample_id,
                    prompt_name=f"{prompt_name}:{mask_condition}",
                    default_count=args.default_position_count,
                    max_evidence_positions=args.max_evidence_positions,
                    random_controls=args.random_controls,
                )
                payload["groups"].append(
                    {
                        "sample_id": sample_id,
                        "prompt_name": prompt_name,
                        "mask_condition": mask_condition,
                        "bucket": bucket_name,
                        "diagnostics": diagnostics,
                        "groups": [
                            {
                                "group_name": group["group_name"],
                                "group_kind": group["group_kind"],
                                "position_count": len(group["positions"]),
                            }
                            for group in groups
                        ],
                    }
                )
                usable_runs += 1

                for direction, condition, score in [
                    ("baseline", "clean", clean_score),
                    ("baseline", mask_condition, masked_score),
                ]:
                    rows.append(
                        {
                            "model_family": args.model_family,
                            "sample_id": sample_id,
                            "prompt_name": prompt_name,
                            "mask_condition": mask_condition,
                            "layer": args.layer,
                            "bucket": bucket_name,
                            "direction": direction,
                            "group_name": condition,
                            "group_kind": "baseline",
                            "condition": condition,
                            "scale": "",
                            "position_count": 0,
                            "target_answer": sample["answer"],
                            "target_token_id": chosen_target_id,
                            "target_token": clean_score["target_token"],
                            "target_logit": score["target_logit"],
                            "target_prob": score["target_prob"],
                            "target_rank": score["target_rank"],
                            "clean_target_logit": clean_score["target_logit"],
                            "clean_target_rank": clean_score["target_rank"],
                            "masked_target_logit": masked_score["target_logit"],
                            "masked_target_rank": masked_score["target_rank"],
                            "clean_mask_logit_gap": clean_mask_logit_gap,
                            "clean_mask_rank_gap": clean_mask_rank_gap,
                            "effect_logit": "",
                            "effect_rank": "",
                            "effect_gap_closure": "",
                            "top1_token": score["top1_token"],
                            "clean_top1_token": clean_score["top1_token"],
                            "masked_top1_token": masked_score["top1_token"],
                            "top1_changed_vs_reference": False,
                            "target_candidates": json.dumps(target_candidates, ensure_ascii=False),
                        }
                    )

                for group in groups:
                    if not group["positions"]:
                        continue
                    for scale in scales:
                        scale_label = _scale_label(scale)
                        restore_intervention = {
                            "positions": group["positions"],
                            "source_hidden": clean_hidden,
                            "target_hidden": masked_hidden,
                            "scale": scale,
                        }
                        corrupt_intervention = {
                            "positions": group["positions"],
                            "source_hidden": masked_hidden,
                            "target_hidden": clean_hidden,
                            "scale": scale,
                        }
                        for direction, inputs, reference, intervention in [
                            ("restore", masked_inputs, masked_score, restore_intervention),
                            ("corrupt", clean_inputs, clean_score, corrupt_intervention),
                        ]:
                            score = _score_with_patch(
                                model,
                                inputs,
                                tokenizer,
                                [chosen_target_id],
                                target_module,
                                intervention,
                            )
                            if direction == "restore":
                                effect_logit = float(score["target_logit"]) - float(masked_score["target_logit"])
                                effect_rank = int(masked_score["target_rank"]) - int(score["target_rank"])
                            else:
                                effect_logit = float(clean_score["target_logit"]) - float(score["target_logit"])
                                effect_rank = int(score["target_rank"]) - int(clean_score["target_rank"])
                            rows.append(
                                {
                                    "model_family": args.model_family,
                                    "sample_id": sample_id,
                                    "prompt_name": prompt_name,
                                    "mask_condition": mask_condition,
                                    "layer": args.layer,
                                    "bucket": bucket_name,
                                    "direction": direction,
                                    "group_name": group["group_name"],
                                    "group_kind": group["group_kind"],
                                    "condition": f"{direction}_{group['group_name']}_{mask_condition}_s{scale_label}",
                                    "scale": scale,
                                    "position_count": len(group["positions"]),
                                    "target_answer": sample["answer"],
                                    "target_token_id": chosen_target_id,
                                    "target_token": clean_score["target_token"],
                                    "target_logit": score["target_logit"],
                                    "target_prob": score["target_prob"],
                                    "target_rank": score["target_rank"],
                                    "clean_target_logit": clean_score["target_logit"],
                                    "clean_target_rank": clean_score["target_rank"],
                                    "masked_target_logit": masked_score["target_logit"],
                                    "masked_target_rank": masked_score["target_rank"],
                                    "clean_mask_logit_gap": clean_mask_logit_gap,
                                    "clean_mask_rank_gap": clean_mask_rank_gap,
                                    "effect_logit": effect_logit,
                                    "effect_rank": effect_rank,
                                    "effect_gap_closure": _safe_ratio(effect_logit, abs(clean_mask_logit_gap)),
                                    "top1_token": score["top1_token"],
                                    "clean_top1_token": clean_score["top1_token"],
                                    "masked_top1_token": masked_score["top1_token"],
                                    "top1_changed_vs_reference": score["top1_token_id"] != reference["top1_token_id"],
                                    "target_candidates": json.dumps(target_candidates, ensure_ascii=False),
                                }
                            )
                del masked_outputs, masked_hidden
                torch.cuda.empty_cache()
            del clean_outputs, clean_hidden
            torch.cuda.empty_cache()

    payload["gpu_after"] = _gpu_info()
    payload["decision"] = {
        "status": "pass_hidden_mask_condition_patch_smoke" if usable_runs else "blocked_no_usable_runs",
        "usable_runs": usable_runs,
        "requested_runs": len(sample_ids) * len(prompt_names) * len(mask_conditions),
        "skipped": skipped,
    }
    fields = [
        "model_family",
        "sample_id",
        "prompt_name",
        "mask_condition",
        "layer",
        "bucket",
        "direction",
        "group_name",
        "group_kind",
        "condition",
        "scale",
        "position_count",
        "target_answer",
        "target_token_id",
        "target_token",
        "target_logit",
        "target_prob",
        "target_rank",
        "clean_target_logit",
        "clean_target_rank",
        "masked_target_logit",
        "masked_target_rank",
        "clean_mask_logit_gap",
        "clean_mask_rank_gap",
        "effect_logit",
        "effect_rank",
        "effect_gap_closure",
        "top1_token",
        "clean_top1_token",
        "masked_top1_token",
        "top1_changed_vs_reference",
        "target_candidates",
    ]
    _write_csv(Path(args.out_csv), rows, fields)
    _write_json(Path(args.out_json), payload)
    _log(f"done status={payload['decision']['status']} usable_runs={usable_runs} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
