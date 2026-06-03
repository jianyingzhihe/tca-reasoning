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
    _download_llava_transcoder,
    _env_presence,
    _first_param_device,
    _gpu_info,
    _llava_bucket_positions,
    _llava_decoder_vectors,
    _llava_encode,
    _llava_inputs,
    _load_llava_encoder_decoder,
    _load_masks,
    _parse_csv,
    _parse_float_csv,
    _processor_patch,
    _prompt,
    _qwen_bucket_positions,
    _qwen_inputs,
    _rank_and_top,
    _replace_hidden,
    _select_features,
    _target_candidates,
)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage2g-restoration] {message}", flush=True)


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


def _scale_label(scale: float) -> str:
    return str(scale).replace(".", "p").replace("-", "m")


def _safe_gap_closure(restore: float, gap: float) -> float | str:
    if gap <= 1e-6:
        return ""
    return restore / gap


def _patch_restore_tensor(
    hidden: torch.Tensor,
    *,
    positions: list[int],
    feature_ids: list[int],
    feature_drops: torch.Tensor,
    decoder_vectors: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if not positions or not feature_ids:
        return hidden
    pos_tensor = torch.tensor(positions, device=hidden.device, dtype=torch.long)
    feat_tensor = torch.tensor(feature_ids, device=feature_drops.device, dtype=torch.long)
    drops = feature_drops[:, pos_tensor, :][:, :, feat_tensor].to(hidden.device, dtype=hidden.dtype)
    vectors = decoder_vectors.to(hidden.device, dtype=hidden.dtype)
    patch = torch.einsum("bpf,fd->bpd", drops, vectors)
    new_hidden = hidden.clone()
    new_hidden[:, pos_tensor, :] = new_hidden[:, pos_tensor, :] + scale * patch
    return new_hidden


def _score_restore_with_hook(
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
            patched = _patch_restore_tensor(hidden, **intervention)
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
    parser = argparse.ArgumentParser(description="Stage 2G mask-to-clean feature restoration smoke.")
    parser.add_argument("--model-family", choices=["qwen", "llava"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--transcoder-ref", required=True)
    parser.add_argument("--annotation-roots", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--samples", default="okvqa_val_2847255,okvqa_val_4157235,okvqa_val_3658865")
    parser.add_argument("--prompts", default="B_direct,D_visual_only")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--top-k-features", type=int, default=4)
    parser.add_argument("--control-pool-size", type=int, default=512)
    parser.add_argument("--scales", default="0.5,1.0,1.5")
    parser.add_argument("--answer-prefix", default="The answer is ")
    parser.add_argument("--min-gpu-free-gb", type=float, default=18.0)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    sample_ids = _parse_csv(args.samples)
    prompt_names = _parse_csv(args.prompts)
    scales = _parse_float_csv(args.scales)
    roots = [Path(path) for path in _parse_csv(args.annotation_roots)]
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "created_at": _now(),
        "model_family": args.model_family,
        "model_name": args.model_name,
        "transcoder_ref": args.transcoder_ref,
        "layer": args.layer,
        "scales": scales,
        "samples": sample_ids,
        "prompts": prompt_names,
        "env_presence": _env_presence(),
        "gpu_before": _gpu_info(),
        "selection": [],
        "decision": {},
        "claim_boundary": (
            "Mask-to-clean feature restoration smoke only; not source tracing, not nearest matched "
            "source-control causal route replication."
        ),
    }
    gpu = payload["gpu_before"]
    if not gpu.get("available") or float(gpu.get("free_gb", 0.0)) < args.min_gpu_free_gb:
        payload["decision"] = {"status": "partial", "reason": "insufficient_gpu_free_memory"}
        _write_json(Path(args.out_json), payload)
        return 0

    if args.model_family == "qwen":
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

        processor = AutoProcessor.from_pretrained(args.model_name, local_files_only=True)
        tokenizer = processor.tokenizer
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        dtype = torch.bfloat16
        model.eval()
        device = _first_param_device(model)
        transcoders, config = load_transcoder_from_hub(
            args.transcoder_ref,
            device=device,
            dtype=dtype,
            lazy_encoder=True,
            lazy_decoder=True,
        )
        target_module = model.language_model.layers[args.layer]

        def build_inputs(image: Image.Image, image_path: Path, question: str):
            return _qwen_inputs(processor, image, str(image_path), question, args.answer_prefix, device)

        def bucket_positions(input_ids: list[int], token_texts: list[str]) -> tuple[str, list[int]]:
            buckets = _qwen_bucket_positions(input_ids, token_texts)
            return "image_marker_or_span", buckets["image_marker_or_span"]

        def encode_features(hidden):
            return transcoders.encode_layer(hidden.to(device), args.layer, apply_activation_function=True)

        def decoder_vectors(feature_ids: list[int]):
            ids = torch.tensor(feature_ids, device=device, dtype=torch.long)
            vectors = transcoders._get_decoder_vectors(args.layer, ids)
            if vectors.ndim == 3:
                vectors = vectors[:, 0, :]
            return vectors.to(device=device, dtype=dtype)

        hidden_index = args.layer
        payload["transcoder"] = {
            "type": type(transcoders).__name__,
            "config_model_kind": config.get("model_kind", ""),
            "hook_approximation": "add clean-minus-union feature drop times decoder offset 0 at native language layer output",
        }
    else:
        from transformers import AutoConfig, AutoProcessor, LlavaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(args.model_name)
        config = AutoConfig.from_pretrained(args.model_name)
        patch_info = _processor_patch(processor, config)
        tokenizer = processor.tokenizer
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        dtype = torch.float16
        model.eval()
        device = _first_param_device(model)
        transcoder_path = _download_llava_transcoder(args.transcoder_ref, args.layer)
        llava_encoder = _load_llava_encoder_decoder(transcoder_path, device=device, dtype=dtype)
        target_module = model.language_model.layers[args.layer]
        image_token_id = int(getattr(config, "image_token_index", 32000))

        def build_inputs(image: Image.Image, image_path: Path, question: str):
            return _llava_inputs(processor, image, question, args.answer_prefix, device)

        def bucket_positions(input_ids: list[int], token_texts: list[str]) -> tuple[str, list[int]]:
            del token_texts
            buckets = _llava_bucket_positions(input_ids, image_token_id)
            return "image_token_span", buckets["image_token_span"]

        def encode_features(hidden):
            return _llava_encode(hidden.to(device=device, dtype=dtype), llava_encoder)

        def decoder_vectors(feature_ids: list[int]):
            return _llava_decoder_vectors(llava_encoder, feature_ids).to(device=device, dtype=dtype)

        hidden_index = args.layer + 1
        payload["transcoder"] = {
            "type": "KokosDev/llava15-7b-clt custom pt",
            "path": str(transcoder_path),
            "processor_patch": patch_info,
            "hook_approximation": "add clean-minus-union feature drop times decoder vector at native language layer output",
        }

    rows: list[dict[str, Any]] = []
    usable_runs = 0
    skipped: list[dict[str, Any]] = []

    for sample_id in sample_ids:
        sample = SAMPLES[sample_id]
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
            bucket_name, positions = bucket_positions(input_ids, token_texts)
            target_candidates = _target_candidates(tokenizer, sample["answer"])
            target_ids = [item["token_id"] for item in target_candidates]
            if not positions or not target_ids:
                skipped.append({"sample_id": sample_id, "prompt_name": prompt_name, "reason": "positions_or_target_missing"})
                continue

            with torch.inference_mode():
                clean_outputs = model(**clean_inputs, output_hidden_states=True, use_cache=False)
                union_outputs = model(**union_inputs, output_hidden_states=True, use_cache=False)
            clean_hidden = clean_outputs.hidden_states[hidden_index].to(device=device, dtype=dtype)
            union_hidden = union_outputs.hidden_states[hidden_index].to(device=device, dtype=dtype)
            clean_features = encode_features(clean_hidden).detach()
            union_features = encode_features(union_hidden).detach()
            feature_drops = (clean_features - union_features).detach()
            selection = _select_features(
                clean_features,
                union_features,
                positions,
                top_k=args.top_k_features,
                control_pool_size=args.control_pool_size,
            )
            payload["selection"].append(
                {
                    "sample_id": sample_id,
                    "prompt_name": prompt_name,
                    "bucket": bucket_name,
                    "status": selection["status"],
                    "evidence_features": selection.get("evidence_features", []),
                    "control_features": selection.get("control_features", []),
                }
            )
            if selection["status"] != "ok":
                skipped.append({"sample_id": sample_id, "prompt_name": prompt_name, "reason": selection["status"]})
                continue

            clean_score = _rank_and_top(clean_outputs.logits, tokenizer, target_ids)
            chosen_target_id = int(clean_score["target_token_id"])
            union_score = _rank_and_top(union_outputs.logits, tokenizer, [chosen_target_id])
            evidence_ids = selection["evidence_feature_ids"]
            control_ids = selection["control_feature_ids"]
            groups = {
                "evidence_top1": evidence_ids[:1],
                "evidence_topk": evidence_ids,
                "control_topk": control_ids,
            }
            interventions: dict[str, dict[str, Any] | None] = {
                "clean": None,
                "union_mask": None,
            }
            for scale in scales:
                for group_name, group_ids in groups.items():
                    interventions[f"{group_name}_restore_s{_scale_label(scale)}"] = {
                        "positions": positions,
                        "feature_ids": group_ids,
                        "feature_drops": feature_drops,
                        "decoder_vectors": decoder_vectors(group_ids),
                        "scale": scale,
                    }

            clean_union_logit_gap = clean_score["target_logit"] - union_score["target_logit"]
            clean_union_rank_gap = union_score["target_rank"] - clean_score["target_rank"]

            usable_runs += 1
            for condition, intervention in interventions.items():
                if condition == "clean":
                    score = clean_score
                elif condition == "union_mask":
                    score = union_score
                else:
                    score = _score_restore_with_hook(
                        model,
                        union_inputs,
                        tokenizer,
                        [chosen_target_id],
                        target_module,
                        intervention,
                    )

                logit_restore_vs_union = score["target_logit"] - union_score["target_logit"]
                rank_restore_vs_union = union_score["target_rank"] - score["target_rank"]
                row = {
                    "model_family": args.model_family,
                    "sample_id": sample_id,
                    "prompt_name": prompt_name,
                    "layer": args.layer,
                    "bucket": bucket_name,
                    "condition": condition,
                    "target_answer": sample["answer"],
                    "target_token_id": score["target_token_id"],
                    "target_token": score["target_token"],
                    "target_logit": score["target_logit"],
                    "target_prob": score["target_prob"],
                    "target_rank": score["target_rank"],
                    "clean_target_logit": clean_score["target_logit"],
                    "clean_target_rank": clean_score["target_rank"],
                    "union_target_logit": union_score["target_logit"],
                    "union_target_rank": union_score["target_rank"],
                    "clean_union_logit_gap": clean_union_logit_gap,
                    "clean_union_rank_gap": clean_union_rank_gap,
                    "logit_restore_vs_union": logit_restore_vs_union,
                    "rank_restore_vs_union": rank_restore_vs_union,
                    "logit_gap_closure": _safe_gap_closure(logit_restore_vs_union, clean_union_logit_gap),
                    "top1_token": score["top1_token"],
                    "clean_top1_token": clean_score["top1_token"],
                    "union_top1_token": union_score["top1_token"],
                    "top1_changed_vs_union": score["top1_token_id"] != union_score["top1_token_id"],
                    "evidence_feature_ids": "|".join(str(x) for x in evidence_ids),
                    "control_feature_ids": "|".join(str(x) for x in control_ids),
                    "target_candidates": json.dumps(target_candidates, ensure_ascii=False),
                }
                rows.append(row)

            del clean_outputs, union_outputs, clean_features, union_features, feature_drops
            torch.cuda.empty_cache()

    payload["gpu_after"] = _gpu_info()
    payload["decision"] = {
        "status": "pass_feature_restoration_smoke" if usable_runs else "blocked_no_usable_runs",
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
        "condition",
        "target_answer",
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
        "logit_restore_vs_union",
        "rank_restore_vs_union",
        "logit_gap_closure",
        "top1_token",
        "clean_top1_token",
        "union_top1_token",
        "top1_changed_vs_union",
        "evidence_feature_ids",
        "control_feature_ids",
        "target_candidates",
    ]
    _write_csv(Path(args.out_csv), rows, fields)
    _write_json(Path(args.out_json), payload)
    _log(f"done status={payload['decision']['status']} usable_runs={usable_runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
