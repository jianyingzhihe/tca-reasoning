#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
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
)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage2h-decode-smoke] {message}", flush=True)


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


def _decode_ids(tokenizer, ids: list[int]) -> str:
    if not ids:
        return ""
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def _normalize_answer(text: str) -> str:
    value = (text or "").strip().lower()
    value = re.sub(r"^the answer is\s*", "", value)
    value = value.splitlines()[0] if value else ""
    value = re.split(r"[.;,\n]", value)[0].strip()
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _target_hit(generated_text: str, target_answer: str) -> bool:
    pred = _normalize_answer(generated_text)
    target = _normalize_answer(target_answer)
    if not pred or not target:
        return False
    return target in pred or pred in target


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


def _clone_inputs_with_ids(base_inputs: dict[str, Any], input_ids: torch.Tensor) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in base_inputs.items():
        if key == "input_ids":
            out[key] = input_ids
        elif key == "attention_mask":
            out[key] = torch.ones_like(input_ids, device=input_ids.device)
        else:
            out[key] = value
    if "attention_mask" not in out:
        out["attention_mask"] = torch.ones_like(input_ids, device=input_ids.device)
    return out


def _filter_intervention_for_seq(intervention: dict[str, Any], seq_len: int) -> dict[str, Any] | None:
    positions = [
        int(position)
        for position in intervention["positions"]
        if int(position) < seq_len
        and int(position) < int(intervention["source_hidden"].shape[1])
        and int(position) < int(intervention["target_hidden"].shape[1])
    ]
    if not positions:
        return None
    return {
        "positions": positions,
        "source_hidden": intervention["source_hidden"],
        "target_hidden": intervention["target_hidden"],
        "scale": intervention["scale"],
    }


def _greedy_decode(
    *,
    model,
    tokenizer,
    base_inputs: dict[str, Any],
    module,
    intervention: dict[str, Any] | None,
    target_ids: list[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    input_ids = base_inputs["input_ids"].clone()
    prompt_len = int(input_ids.shape[1])
    generated: list[int] = []
    first_score: dict[str, Any] | None = None
    hook_handle = None

    if intervention is not None:

        def _hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            filtered = _filter_intervention_for_seq(intervention, int(hidden.shape[1]))
            if filtered is None:
                return output
            patched = _patch_hidden_tensor(hidden, **filtered)
            return _replace_hidden(output, patched)

        hook_handle = module.register_forward_hook(_hook)

    try:
        for _step in range(max_new_tokens):
            step_inputs = _clone_inputs_with_ids(base_inputs, input_ids)
            with torch.inference_mode():
                outputs = model(**step_inputs, output_hidden_states=False, use_cache=False)
            if first_score is None:
                first_score = _rank_and_top(outputs.logits, tokenizer, target_ids)
            next_id = int(torch.argmax(outputs.logits[0, -1].float()).item())
            generated.append(next_id)
            next_tensor = torch.tensor([[next_id]], device=input_ids.device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, next_tensor], dim=1)
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is not None and next_id == int(eos_id):
                break
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    continuation = _decode_ids(tokenizer, generated)
    full_generated = f"The answer is {continuation}".strip()
    return {
        "prompt_len": prompt_len,
        "generated_token_ids": generated,
        "generated_continuation": continuation,
        "generated_text": full_generated,
        "predicted_answer": _normalize_answer(full_generated),
        "first_generated_token_id": generated[0] if generated else "",
        "first_generated_token": _decode_ids(tokenizer, generated[:1]),
        "first_step_score": first_score or {},
    }


def _score_with_patch(
    *,
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
            filtered = _filter_intervention_for_seq(intervention, int(hidden.shape[1]))
            if filtered is None:
                return output
            patched = _patch_hidden_tensor(hidden, **filtered)
            return _replace_hidden(output, patched)

        handle = module.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=False, use_cache=False)
    finally:
        if handle is not None:
            handle.remove()
    return _rank_and_top(outputs.logits, tokenizer, target_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2H-4 decoded answer smoke for cross-model hidden-position patch.")
    parser.add_argument("--model-family", choices=["qwen", "llava"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--annotation-roots", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--sample-manifest", default="", help="Optional CSV with sample_id,image_filename,question_text,answer_text.")
    parser.add_argument("--samples", default="okvqa_val_2847255,okvqa_val_4157235,okvqa_val_3658865")
    parser.add_argument("--prompts", default="B_direct,D_visual_only")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--groups",
        default=(
            "top_hidden_delta_plus_answer_adjacent,answer_adjacent_text,"
            "low_delta_control,random_control_1,evidence_region_plus_answer_adjacent"
        ),
    )
    parser.add_argument("--directions", default="restore")
    parser.add_argument("--max-new-tokens", type=int, default=3)
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
    if args.samples.strip().lower() in {"manifest", "auto", "all"}:
        sample_ids = list(sample_lookup)
    else:
        sample_ids = _parse_csv(args.samples)
    prompt_names = _parse_csv(args.prompts)
    selected_groups = set(_parse_csv(args.groups))
    directions = set(_parse_csv(args.directions))
    roots = [Path(path) for path in _parse_csv(args.annotation_roots)]
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "created_at": _now(),
        "model_family": args.model_family,
        "model_name": args.model_name,
        "layer": args.layer,
        "sample_manifest": args.sample_manifest,
        "samples": sample_ids,
        "prompts": prompt_names,
        "groups": sorted(selected_groups),
        "directions": sorted(directions),
        "max_new_tokens": args.max_new_tokens,
        "env_presence": _env_presence(),
        "gpu_before": _gpu_info(),
        "group_diagnostics": [],
        "decision": {},
        "claim_boundary": (
            "Decoded answer smoke only. Success can support generation-level bridge smoke, "
            "but not source-control causal route replication."
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
    skipped: list[dict[str, Any]] = []
    usable_runs = 0

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
            target_candidates = _target_candidates(tokenizer, sample["answer"])
            target_ids = [item["token_id"] for item in target_candidates]
            if not visual_positions or not target_ids:
                skipped.append({"sample_id": sample_id, "prompt_name": prompt_name, "reason": "positions_or_target_missing"})
                continue

            with torch.inference_mode():
                clean_outputs = model(**clean_inputs, output_hidden_states=True, use_cache=False)
                union_outputs = model(**union_inputs, output_hidden_states=True, use_cache=False)
            clean_hidden = clean_outputs.hidden_states[hidden_index].detach()
            union_hidden = union_outputs.hidden_states[hidden_index].detach()
            clean_score = _rank_and_top(clean_outputs.logits, tokenizer, target_ids)
            chosen_target_id = int(clean_score["target_token_id"])
            union_score = _rank_and_top(union_outputs.logits, tokenizer, [chosen_target_id])
            clean_union_logit_gap = clean_score["target_logit"] - union_score["target_logit"]
            clean_union_rank_gap = union_score["target_rank"] - clean_score["target_rank"]
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
            payload["group_diagnostics"].append(
                {
                    "sample_id": sample_id,
                    "prompt_name": prompt_name,
                    "bucket": bucket_name,
                    "diagnostics": diagnostics,
                    "available_groups": [group["group_name"] for group in groups],
                }
            )
            group_lookup = {group["group_name"]: group for group in groups}

            conditions: list[dict[str, Any]] = [
                {"direction": "baseline", "group_name": "clean", "group_kind": "baseline", "inputs": clean_inputs, "intervention": None, "score": clean_score},
                {
                    "direction": "baseline",
                    "group_name": "union_mask",
                    "group_kind": "baseline",
                    "inputs": union_inputs,
                    "intervention": None,
                    "score": union_score,
                },
            ]
            if "restore" in directions:
                for group_name in sorted(selected_groups):
                    group = group_lookup.get(group_name)
                    if group is None or not group["positions"]:
                        continue
                    conditions.append(
                        {
                            "direction": "restore",
                            "group_name": group_name,
                            "group_kind": group["group_kind"],
                            "inputs": union_inputs,
                            "intervention": {
                                "positions": group["positions"],
                                "source_hidden": clean_hidden,
                                "target_hidden": union_hidden,
                                "scale": 1.0,
                            },
                            "score": None,
                        }
                    )
            if "corrupt" in directions:
                for group_name in sorted(selected_groups):
                    group = group_lookup.get(group_name)
                    if group is None or not group["positions"]:
                        continue
                    conditions.append(
                        {
                            "direction": "corrupt",
                            "group_name": group_name,
                            "group_kind": group["group_kind"],
                            "inputs": clean_inputs,
                            "intervention": {
                                "positions": group["positions"],
                                "source_hidden": union_hidden,
                                "target_hidden": clean_hidden,
                                "scale": 1.0,
                            },
                            "score": None,
                        }
                    )

            clean_generated_pred = ""
            union_generated_pred = ""
            condition_outputs: dict[str, str] = {}

            for condition in conditions:
                condition_name = f"{condition['direction']}::{condition['group_name']}"
                score = condition["score"] or _score_with_patch(
                    model=model,
                    inputs=condition["inputs"],
                    tokenizer=tokenizer,
                    target_ids=[chosen_target_id],
                    module=target_module,
                    intervention=condition["intervention"],
                )
                decoded = _greedy_decode(
                    model=model,
                    tokenizer=tokenizer,
                    base_inputs=condition["inputs"],
                    module=target_module,
                    intervention=condition["intervention"],
                    target_ids=[chosen_target_id],
                    max_new_tokens=args.max_new_tokens,
                )
                predicted = decoded["predicted_answer"]
                condition_outputs[condition_name] = predicted
                if condition_name == "baseline::clean":
                    clean_generated_pred = predicted
                if condition_name == "baseline::union_mask":
                    union_generated_pred = predicted
                first_score = decoded["first_step_score"]
                rows.append(
                    {
                        "model_family": args.model_family,
                        "sample_id": sample_id,
                        "prompt_name": prompt_name,
                        "layer": args.layer,
                        "bucket": bucket_name,
                        "direction": condition["direction"],
                        "group_name": condition["group_name"],
                        "group_kind": condition["group_kind"],
                        "condition": condition_name,
                        "position_count": len(condition["intervention"]["positions"]) if condition["intervention"] else 0,
                        "target_answer": sample["answer"],
                        "target_token_id": chosen_target_id,
                        "target_token": clean_score["target_token"],
                        "clean_target_logit": clean_score["target_logit"],
                        "clean_target_rank": clean_score["target_rank"],
                        "union_target_logit": union_score["target_logit"],
                        "union_target_rank": union_score["target_rank"],
                        "clean_union_logit_gap": clean_union_logit_gap,
                        "clean_union_rank_gap": clean_union_rank_gap,
                        "prompt_target_logit": score["target_logit"],
                        "prompt_target_rank": score["target_rank"],
                        "prompt_top1_token": score["top1_token"],
                        "first_step_target_logit": first_score.get("target_logit", ""),
                        "first_step_target_rank": first_score.get("target_rank", ""),
                        "first_step_top1_token": first_score.get("top1_token", ""),
                        "first_generated_token": decoded["first_generated_token"],
                        "generated_continuation": decoded["generated_continuation"],
                        "generated_text": decoded["generated_text"],
                        "predicted_answer": predicted,
                        "target_hit": _target_hit(decoded["generated_text"], sample["answer"]),
                        "generated_token_ids": json.dumps(decoded["generated_token_ids"]),
                        "target_candidates": json.dumps(target_candidates, ensure_ascii=False),
                        "clean_generated_answer": "",
                        "union_generated_answer": "",
                        "answer_changed_vs_clean": "",
                        "answer_changed_vs_union": "",
                    }
                )

            for row in rows:
                if row["model_family"] == args.model_family and row["sample_id"] == sample_id and row["prompt_name"] == prompt_name:
                    row["clean_generated_answer"] = clean_generated_pred
                    row["union_generated_answer"] = union_generated_pred
                    row["answer_changed_vs_clean"] = row["predicted_answer"] != clean_generated_pred
                    row["answer_changed_vs_union"] = row["predicted_answer"] != union_generated_pred
            usable_runs += 1
            del clean_outputs, union_outputs, clean_hidden, union_hidden
            torch.cuda.empty_cache()

    payload["gpu_after"] = _gpu_info()
    payload["decision"] = {
        "status": "pass_decoded_answer_smoke" if usable_runs else "blocked_no_usable_runs",
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
        "position_count",
        "target_answer",
        "target_token_id",
        "target_token",
        "clean_target_logit",
        "clean_target_rank",
        "union_target_logit",
        "union_target_rank",
        "clean_union_logit_gap",
        "clean_union_rank_gap",
        "prompt_target_logit",
        "prompt_target_rank",
        "prompt_top1_token",
        "first_step_target_logit",
        "first_step_target_rank",
        "first_step_top1_token",
        "first_generated_token",
        "generated_continuation",
        "generated_text",
        "predicted_answer",
        "target_hit",
        "generated_token_ids",
        "target_candidates",
        "clean_generated_answer",
        "union_generated_answer",
        "answer_changed_vs_clean",
        "answer_changed_vs_union",
    ]
    _write_csv(Path(args.out_csv), rows, fields)
    _write_json(Path(args.out_json), payload)
    _log(f"done status={payload['decision']['status']} usable_runs={usable_runs} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
