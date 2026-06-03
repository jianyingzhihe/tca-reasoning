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
    _make_groups,
    _patch_hidden_tensor,
    _safe_ratio,
    _scale_label,
)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage2l-wrong-target] {message}", flush=True)


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


def _first_nonempty(row: dict[str, str], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _load_sample_manifest(path: Path) -> dict[str, dict[str, str]]:
    samples: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = _first_nonempty(row, ["sample_id", "id"])
            image_name = _first_nonempty(row, ["image_filename", "image", "image_name"])
            question = _first_nonempty(row, ["question_text", "question"])
            answer = _first_nonempty(row, ["answer_text", "answer", "target_answer"])
            if not sample_id or not image_name or not question or not answer:
                continue
            samples[sample_id] = {
                "image": Path(image_name).name,
                "question": question,
                "answer": answer,
                "reasoning_operation": _first_nonempty(row, ["reasoning_operation"]),
                "visual_structure": _first_nonempty(row, ["visual_structure"]),
                "image_dependence": _first_nonempty(row, ["image_dependence"]),
                "mask_pack": _first_nonempty(row, ["mask_pack"]),
                "compactness_label": _first_nonempty(row, ["compactness_label"]),
            }
    return samples


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


def _wrong_answer_for(
    *,
    sample_ids: list[str],
    sample_lookup: dict[str, dict[str, str]],
    sample_id: str,
    tokenizer,
    correct_ids: set[int],
) -> tuple[str, str, list[dict[str, Any]]]:
    if sample_id not in sample_ids:
        sample_ids = list(sample_lookup)
    start = sample_ids.index(sample_id) if sample_id in sample_ids else 0
    for offset in range(1, len(sample_ids) + 1):
        wrong_id = sample_ids[(start + offset) % len(sample_ids)]
        wrong_answer = sample_lookup[wrong_id]["answer"]
        candidates = _target_candidates(tokenizer, wrong_answer)
        filtered = [item for item in candidates if int(item["token_id"]) not in correct_ids]
        # LLaVA/SentencePiece often shares a leading-space token across answers.
        # Keep the non-overlapping candidate tokens instead of rejecting the whole
        # wrong answer whenever any candidate overlaps the correct target.
        if filtered:
            return wrong_id, wrong_answer, filtered
    return "", "", []


def _keep_groups(groups: list[dict[str, Any]], names: set[str]) -> list[dict[str, Any]]:
    return [group for group in groups if group["group_name"] in names and group["positions"]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2L wrong-target negative control for hidden-position patch.")
    parser.add_argument("--model-family", choices=["qwen", "llava"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--annotation-roots", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--sample-manifest", default="")
    parser.add_argument("--samples", default="manifest")
    parser.add_argument("--prompts", default="B_direct,D_visual_only")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--scales", default="1.0")
    parser.add_argument("--default-position-count", type=int, default=32)
    parser.add_argument("--max-evidence-positions", type=int, default=64)
    parser.add_argument("--answer-adjacent-count", type=int, default=4)
    parser.add_argument("--random-controls", type=int, default=1)
    parser.add_argument(
        "--groups",
        default="top_hidden_delta_plus_answer_adjacent,delta_matched_plus_answer_adjacent,activation_matched_plus_answer_adjacent,answer_adjacent_text,low_delta_control,random_control_1",
    )
    parser.add_argument("--answer-prefix", default="The answer is ")
    parser.add_argument("--min-gpu-free-gb", type=float, default=18.0)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    sample_lookup = _load_sample_manifest(Path(args.sample_manifest)) if args.sample_manifest else dict(SAMPLES)
    sample_ids = list(sample_lookup) if args.samples.strip().lower() in {"manifest", "auto", "all"} else _parse_csv(args.samples)
    prompt_names = _parse_csv(args.prompts)
    scales = [float(value) for value in _parse_csv(args.scales)]
    keep_group_names = set(_parse_csv(args.groups))
    roots = [Path(path) for path in _parse_csv(args.annotation_roots)]
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "created_at": _now(),
        "model_family": args.model_family,
        "model_name": args.model_name,
        "layer": args.layer,
        "samples": sample_ids,
        "prompts": prompt_names,
        "groups": sorted(keep_group_names),
        "env_presence": _env_presence(),
        "gpu_before": _gpu_info(),
        "diagnostics": [],
        "decision": {},
        "claim_boundary": "Wrong-target negative control only; not source-control causal route replication.",
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
        union_image = _apply_mask(clean_image, mask_info["union_mask"], (128, 128, 128))

        for prompt_name in prompt_names:
            question = _prompt(sample["question"], prompt_name)
            clean_inputs = build_inputs(clean_image, image_path, question)
            union_inputs = build_inputs(union_image, image_path, question)
            input_ids = clean_inputs["input_ids"][0].detach().cpu().tolist()
            token_texts = tokenizer.convert_ids_to_tokens(input_ids)
            bucket_name, visual_positions = bucket_positions(input_ids, token_texts)
            answer_positions = _answer_adjacent_positions(len(input_ids), set(visual_positions), args.answer_adjacent_count)

            correct_candidates = _target_candidates(tokenizer, sample["answer"])
            correct_candidate_ids = {int(item["token_id"]) for item in correct_candidates}
            wrong_sample_id, wrong_answer, wrong_candidates = _wrong_answer_for(
                sample_ids=sample_ids,
                sample_lookup=sample_lookup,
                sample_id=sample_id,
                tokenizer=tokenizer,
                correct_ids=correct_candidate_ids,
            )
            if not visual_positions or not correct_candidates or not wrong_candidates:
                skipped.append(
                    {
                        "sample_id": sample_id,
                        "prompt_name": prompt_name,
                        "reason": "positions_or_targets_missing",
                        "visual_position_count": len(visual_positions),
                        "correct_candidate_count": len(correct_candidates),
                        "wrong_candidate_count": len(wrong_candidates),
                        "wrong_sample_id": wrong_sample_id,
                        "wrong_answer": wrong_answer,
                        "input_seq_len": len(input_ids),
                    }
                )
                continue

            with torch.inference_mode():
                clean_outputs = model(**clean_inputs, output_hidden_states=True, use_cache=False)
                union_outputs = model(**union_inputs, output_hidden_states=True, use_cache=False)
            clean_hidden = clean_outputs.hidden_states[hidden_index].detach()
            union_hidden = union_outputs.hidden_states[hidden_index].detach()

            target_specs: list[dict[str, Any]] = []
            for target_label, target_answer, source_sample, candidates in [
                ("correct", sample["answer"], sample_id, correct_candidates),
                ("wrong", wrong_answer, wrong_sample_id, wrong_candidates),
            ]:
                candidate_ids = [int(item["token_id"]) for item in candidates]
                clean_score = _rank_and_top(clean_outputs.logits, tokenizer, candidate_ids)
                chosen_target_id = int(clean_score["target_token_id"])
                union_score = _rank_and_top(union_outputs.logits, tokenizer, [chosen_target_id])
                target_specs.append(
                    {
                        "target_label": target_label,
                        "target_answer": target_answer,
                        "target_source_sample_id": source_sample,
                        "target_candidates": candidates,
                        "chosen_target_id": chosen_target_id,
                        "clean_score": clean_score,
                        "union_score": union_score,
                    }
                )

            groups, diagnostics = _make_groups(
                model_family=args.model_family,
                clean_inputs=clean_inputs,
                clean_hidden=clean_hidden,
                union_hidden=union_hidden,
                visual_positions=visual_positions,
                answer_positions=answer_positions,
                evidence_mask=mask_info["union_mask"],
                sample_id=sample_id,
                prompt_name=prompt_name,
                default_count=args.default_position_count,
                max_evidence_positions=args.max_evidence_positions,
                random_controls=args.random_controls,
            )
            groups = _keep_groups(groups, keep_group_names)
            payload["diagnostics"].append(
                {
                    "sample_id": sample_id,
                    "prompt_name": prompt_name,
                    "bucket": bucket_name,
                    "wrong_sample_id": wrong_sample_id,
                    "wrong_answer": wrong_answer,
                    "group_position_counts": {group["group_name"]: len(group["positions"]) for group in groups},
                    "group_diagnostics": diagnostics,
                }
            )
            usable_runs += 1

            for spec in target_specs:
                clean_score = spec["clean_score"]
                union_score = spec["union_score"]
                gap = float(clean_score["target_logit"]) - float(union_score["target_logit"])
                rank_gap = int(union_score["target_rank"]) - int(clean_score["target_rank"])
                for condition, score in [("clean", clean_score), ("union_mask", union_score)]:
                    rows.append(
                        {
                            "model_family": args.model_family,
                            "sample_id": sample_id,
                            "prompt_name": prompt_name,
                            "layer": args.layer,
                            "bucket": bucket_name,
                            "direction": "baseline",
                            "group_name": condition,
                            "group_kind": "baseline",
                            "condition": condition,
                            "scale": "",
                            "position_count": 0,
                            "target_label": spec["target_label"],
                            "target_answer": spec["target_answer"],
                            "target_source_sample_id": spec["target_source_sample_id"],
                            "target_token_id": score["target_token_id"],
                            "target_token": score["target_token"],
                            "target_logit": score["target_logit"],
                            "target_prob": score["target_prob"],
                            "target_rank": score["target_rank"],
                            "clean_target_logit": clean_score["target_logit"],
                            "clean_target_rank": clean_score["target_rank"],
                            "union_target_logit": union_score["target_logit"],
                            "union_target_rank": union_score["target_rank"],
                            "clean_union_logit_gap": gap,
                            "clean_union_rank_gap": rank_gap,
                            "effect_logit": "",
                            "effect_rank": "",
                            "effect_gap_closure": "",
                            "top1_token": score["top1_token"],
                            "clean_top1_token": clean_score["top1_token"],
                            "union_top1_token": union_score["top1_token"],
                            "target_candidates": json.dumps(spec["target_candidates"], ensure_ascii=False),
                        }
                    )

            for group in groups:
                for scale in scales:
                    scale_label = _scale_label(scale)
                    restore_intervention = {
                        "positions": group["positions"],
                        "source_hidden": clean_hidden,
                        "target_hidden": union_hidden,
                        "scale": scale,
                    }
                    corrupt_intervention = {
                        "positions": group["positions"],
                        "source_hidden": union_hidden,
                        "target_hidden": clean_hidden,
                        "scale": scale,
                    }
                    for spec in target_specs:
                        clean_score = spec["clean_score"]
                        union_score = spec["union_score"]
                        chosen_target_id = int(spec["chosen_target_id"])
                        gap = float(clean_score["target_logit"]) - float(union_score["target_logit"])
                        for direction, inputs, reference, intervention in [
                            ("restore", union_inputs, union_score, restore_intervention),
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
                                effect_logit = float(score["target_logit"]) - float(union_score["target_logit"])
                                effect_rank = int(union_score["target_rank"]) - int(score["target_rank"])
                            else:
                                effect_logit = float(clean_score["target_logit"]) - float(score["target_logit"])
                                effect_rank = int(score["target_rank"]) - int(clean_score["target_rank"])
                            rows.append(
                                {
                                    "model_family": args.model_family,
                                    "sample_id": sample_id,
                                    "prompt_name": prompt_name,
                                    "layer": args.layer,
                                    "bucket": bucket_name,
                                    "direction": direction,
                                    "group_name": group["group_name"],
                                    "group_kind": group["group_kind"],
                                    "condition": f"{direction}_{group['group_name']}_{spec['target_label']}_s{scale_label}",
                                    "scale": scale,
                                    "position_count": len(group["positions"]),
                                    "target_label": spec["target_label"],
                                    "target_answer": spec["target_answer"],
                                    "target_source_sample_id": spec["target_source_sample_id"],
                                    "target_token_id": score["target_token_id"],
                                    "target_token": score["target_token"],
                                    "target_logit": score["target_logit"],
                                    "target_prob": score["target_prob"],
                                    "target_rank": score["target_rank"],
                                    "clean_target_logit": clean_score["target_logit"],
                                    "clean_target_rank": clean_score["target_rank"],
                                    "union_target_logit": union_score["target_logit"],
                                    "union_target_rank": union_score["target_rank"],
                                    "clean_union_logit_gap": gap,
                                    "clean_union_rank_gap": int(union_score["target_rank"]) - int(clean_score["target_rank"]),
                                    "effect_logit": effect_logit,
                                    "effect_rank": effect_rank,
                                    "effect_gap_closure": _safe_ratio(effect_logit, abs(gap)),
                                    "top1_token": score["top1_token"],
                                    "clean_top1_token": clean_score["top1_token"],
                                    "union_top1_token": union_score["top1_token"],
                                    "top1_changed_vs_reference": score["top1_token_id"] != reference["top1_token_id"],
                                    "target_candidates": json.dumps(spec["target_candidates"], ensure_ascii=False),
                                }
                            )
            del clean_outputs, union_outputs, clean_hidden, union_hidden
            torch.cuda.empty_cache()

    payload["gpu_after"] = _gpu_info()
    payload["decision"] = {
        "status": "pass_wrong_target_negative_control_smoke" if usable_runs else "blocked_no_usable_runs",
        "usable_runs": usable_runs,
        "requested_runs": len(sample_ids) * len(prompt_names),
        "skipped": skipped,
    }
    fields = [
        "model_family",
        "sample_id",
        "prompt_name",
        "layer",
        "bucket",
        "direction",
        "group_name",
        "group_kind",
        "condition",
        "scale",
        "position_count",
        "target_label",
        "target_answer",
        "target_source_sample_id",
        "target_token_id",
        "target_token",
        "target_logit",
        "target_prob",
        "target_rank",
        "clean_target_logit",
        "clean_target_rank",
        "union_target_logit",
        "union_target_rank",
        "clean_union_logit_gap",
        "clean_union_rank_gap",
        "effect_logit",
        "effect_rank",
        "effect_gap_closure",
        "top1_token",
        "clean_top1_token",
        "union_top1_token",
        "top1_changed_vs_reference",
        "target_candidates",
    ]
    _write_csv(Path(args.out_csv), rows, fields)
    _write_json(Path(args.out_json), payload)
    _log(f"done status={payload['decision']['status']} usable_runs={usable_runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
