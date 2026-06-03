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
)
from run_cross_model_mask_shuffled_negative_control_smoke import _shift_mask
from run_cross_model_wrong_target_negative_control_smoke import _wrong_answer_for
from run_stage2o_attribution_weighted_feature_bridge import (
    _load_run_manifest,
    _run_specs,
    _safe_gap_closure,
    _select_attribution_feature_groups,
    _stable_seed,
)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage2o-source-control] {message}", flush=True)


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


def _load_samples(path: str) -> dict[str, dict[str, str]]:
    if path:
        return _load_sample_manifest(Path(path))
    return dict(SAMPLES)


def _mask_for_condition(mask_info: dict[str, Any], condition: str):
    if condition == "answer_mask":
        return mask_info["answer_mask"]
    if condition == "union_mask":
        return mask_info["union_mask"]
    raise ValueError(f"Unknown mask_condition: {condition}")


def _patch_feature_tensor(
    hidden: torch.Tensor,
    *,
    positions: list[int],
    feature_ids: list[int],
    feature_values: torch.Tensor,
    decoder_vectors: torch.Tensor,
    scale: float,
    mode: str,
) -> torch.Tensor:
    if not positions or not feature_ids:
        return hidden
    pos_tensor = torch.tensor(positions, device=hidden.device, dtype=torch.long)
    feat_tensor = torch.tensor(feature_ids, device=feature_values.device, dtype=torch.long)
    vals = feature_values[:, pos_tensor, :][:, :, feat_tensor].to(hidden.device, dtype=hidden.dtype)
    vectors = decoder_vectors.to(hidden.device, dtype=hidden.dtype)
    patch = torch.einsum("bpf,fd->bpd", vals, vectors)
    new_hidden = hidden.clone()
    if mode in {"restore", "add"}:
        new_hidden[:, pos_tensor, :] = new_hidden[:, pos_tensor, :] + scale * patch
    elif mode in {"zero", "corrupt", "subtract"}:
        new_hidden[:, pos_tensor, :] = new_hidden[:, pos_tensor, :] - scale * patch
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return new_hidden


def _score_with_feature_patch(
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
            patched = _patch_feature_tensor(hidden, **intervention)
            return _replace_hidden(output, patched)

        handle = module.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=False, use_cache=False)
    finally:
        if handle is not None:
            handle.remove()
    return _rank_and_top(outputs.logits, tokenizer, target_ids)


def _feature_payload_map(selection: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for payload in selection.get("feature_payload", {}).values():
        for item in payload:
            out[int(item["feature_id"])] = item
    return out


def _choose_control(selection: dict[str, Any], source_feature: int) -> tuple[str, int] | None:
    groups: dict[str, list[int]] = selection["feature_groups"]
    payload = _feature_payload_map(selection)
    source = payload.get(source_feature, {})
    source_activation = float(source.get("clean_activation", 0.0) or 0.0)
    source_drop = float(source.get("weighted_drop", 0.0) or 0.0)
    source_contrib = float(source.get("target_contribution", 0.0) or 0.0)
    candidates: list[tuple[float, str, int]] = []
    for group_name in [
        "attribution_matched_mask_insensitive_topk",
        "activation_matched_topk",
        "drop_matched_topk",
        "random_active_topk",
    ]:
        for feat_id in groups.get(group_name, []):
            item = payload.get(int(feat_id), {})
            activation = float(item.get("clean_activation", 0.0) or 0.0)
            drop = float(item.get("weighted_drop", 0.0) or 0.0)
            contrib = float(item.get("target_contribution", 0.0) or 0.0)
            distance = abs(activation - source_activation) + abs(drop - source_drop) + 0.25 * abs(contrib - source_contrib)
            candidates.append((distance, group_name, int(feat_id)))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1], candidates[0][2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2O approximate source-control route probe.")
    parser.add_argument("--model-family", choices=["qwen", "llava"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--transcoder-ref", required=True)
    parser.add_argument("--annotation-roots", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--sample-manifest", default="")
    parser.add_argument("--run-manifest", default="")
    parser.add_argument("--max-runs", type=int, default=0, help="Optional smoke limit; 0 means run all specs.")
    parser.add_argument("--samples", default="manifest")
    parser.add_argument("--prompts", default="B_direct,D_visual_only")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--mask-conditions", default="answer_mask,union_mask")
    parser.add_argument("--position-group", default="top_hidden_delta_plus_answer_adjacent")
    parser.add_argument("--top-k-features", type=int, default=8)
    parser.add_argument("--control-pool-size", type=int, default=2048)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--answer-prefix", default="The answer is ")
    parser.add_argument("--min-gpu-free-gb", type=float, default=18.0)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    sample_lookup = _load_samples(args.sample_manifest)
    sample_ids = list(sample_lookup) if args.samples.strip().lower() in {"manifest", "auto", "all"} else _parse_csv(args.samples)
    prompt_names = _parse_csv(args.prompts)
    run_manifest = _load_run_manifest(args.run_manifest)
    run_specs = _run_specs(sample_ids, prompt_names, run_manifest)
    original_run_count = len(run_specs)
    if args.max_runs and args.max_runs > 0:
        run_specs = run_specs[: args.max_runs]
    mask_conditions = _parse_csv(args.mask_conditions)
    roots = [Path(path) for path in _parse_csv(args.annotation_roots)]
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "created_at": _now(),
        "model_family": args.model_family,
        "model_name": args.model_name,
        "transcoder_ref": args.transcoder_ref,
        "layer": args.layer,
        "mask_conditions": mask_conditions,
        "position_group": args.position_group,
        "run_count": len(run_specs),
        "original_run_count": original_run_count,
        "max_runs": args.max_runs,
        "env_presence": _env_presence(),
        "gpu_before": _gpu_info(),
        "pairs": [],
        "decision": {},
        "claim_boundary": (
            "Stage 2O source-control probe creates approximate CLT feature source/control pairs. "
            "It is not full Gemma-style source tracing."
        ),
    }
    gpu = payload["gpu_before"]
    if not gpu.get("available") or float(gpu.get("free_gb", 0.0)) < args.min_gpu_free_gb:
        payload["decision"] = {"status": "partial", "reason": "insufficient_gpu_free_memory"}
        _write_json(Path(args.out_json), payload)
        return 0

    if args.model_family == "qwen":
        from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
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
        payload["transcoder"] = {"type": type(transcoders).__name__, "config_model_kind": config.get("model_kind", "")}
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
        dtype = torch.float16
        model.eval()
        device = _first_param_device(model)
        transcoder_path = _download_llava_transcoder(args.transcoder_ref, args.layer)
        llava_encoder = _load_llava_encoder_decoder(transcoder_path, device=device, dtype=dtype)
        target_module = model.language_model.layers[args.layer]
        image_token_id = int(getattr(config, "image_token_index", 32000))

        def build_inputs(image: Image.Image, image_path: Path, question: str):
            del image_path
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
        payload["transcoder"] = {"type": "KokosDev/llava15-7b-clt custom pt", "path": str(transcoder_path)}

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    usable_pairs = 0
    output_weight = model.get_output_embeddings().weight

    for sample_id, prompt_name in run_specs:
        sample = sample_lookup.get(sample_id)
        if sample is None:
            skipped.append({"sample_id": sample_id, "prompt_name": prompt_name, "reason": "sample_missing_from_manifest"})
            continue
        mask_info = _load_masks(sample, roots, work_dir)
        if mask_info["status"] != "ok":
            skipped.append({"sample_id": sample_id, "prompt_name": prompt_name, "reason": "mask_missing", "mask_info": mask_info})
            continue
        image_path = Path(mask_info["image_path"])
        clean_image = Image.open(image_path).convert("RGB")
        question = _prompt(sample["question"], prompt_name)
        clean_inputs = build_inputs(clean_image, image_path, question)
        input_ids = clean_inputs["input_ids"][0].detach().cpu().tolist()
        token_texts = tokenizer.convert_ids_to_tokens(input_ids)
        bucket_name, visual_positions = bucket_positions(input_ids, token_texts)
        answer_positions = _answer_adjacent_positions(len(input_ids), set(visual_positions), 4)
        target_candidates = _target_candidates(tokenizer, sample["answer"])
        target_ids = [item["token_id"] for item in target_candidates]
        if not visual_positions or not target_ids:
            skipped.append({"sample_id": sample_id, "prompt_name": prompt_name, "reason": "positions_or_target_missing"})
            continue
        with torch.inference_mode():
            clean_outputs = model(**clean_inputs, output_hidden_states=True, use_cache=False)
        clean_score = _rank_and_top(clean_outputs.logits, tokenizer, target_ids)
        chosen_target_id = int(clean_score["target_token_id"])
        wrong_sample_id, wrong_answer, wrong_candidates = _wrong_answer_for(
            sample_ids=list(sample_lookup),
            sample_lookup=sample_lookup,
            sample_id=sample_id,
            tokenizer=tokenizer,
            correct_ids={chosen_target_id},
        )
        wrong_ids = [int(item["token_id"]) for item in wrong_candidates]
        wrong_clean_score = _rank_and_top(clean_outputs.logits, tokenizer, wrong_ids) if wrong_ids else {}
        clean_hidden = clean_outputs.hidden_states[hidden_index].to(device=device, dtype=dtype).detach()
        clean_features = encode_features(clean_hidden).detach()
        target_direction = output_weight[chosen_target_id].detach().to(device=device, dtype=dtype)

        for mask_condition in mask_conditions:
            evidence_mask = _mask_for_condition(mask_info, mask_condition)
            mask_image = _apply_mask(clean_image, evidence_mask, (128, 128, 128))
            shifted_mask = mask_info.get("shifted_mask") or _shift_mask(evidence_mask, frac_x=0.37, frac_y=0.29)
            shuffled_mask = mask_info.get("shuffled_mask") or _shift_mask(evidence_mask, frac_x=0.61, frac_y=0.43)
            shifted_image = _apply_mask(clean_image, shifted_mask, (128, 128, 128))
            shuffled_image = _apply_mask(clean_image, shuffled_mask, (128, 128, 128))
            mask_inputs = build_inputs(mask_image, image_path, question)
            shifted_inputs = build_inputs(shifted_image, image_path, question)
            shuffled_inputs = build_inputs(shuffled_image, image_path, question)
            with torch.inference_mode():
                mask_outputs = model(**mask_inputs, output_hidden_states=True, use_cache=False)
                shifted_outputs = model(**shifted_inputs, output_hidden_states=True, use_cache=False)
                shuffled_outputs = model(**shuffled_inputs, output_hidden_states=True, use_cache=False)
            mask_score = _rank_and_top(mask_outputs.logits, tokenizer, [chosen_target_id])
            wrong_mask_score = _rank_and_top(mask_outputs.logits, tokenizer, wrong_ids) if wrong_ids else {}
            shifted_score = _rank_and_top(shifted_outputs.logits, tokenizer, [chosen_target_id])
            shuffled_score = _rank_and_top(shuffled_outputs.logits, tokenizer, [chosen_target_id])
            mask_hidden = mask_outputs.hidden_states[hidden_index].to(device=device, dtype=dtype).detach()
            shifted_hidden = shifted_outputs.hidden_states[hidden_index].to(device=device, dtype=dtype).detach()
            shuffled_hidden = shuffled_outputs.hidden_states[hidden_index].to(device=device, dtype=dtype).detach()
            mask_features = encode_features(mask_hidden).detach()
            shifted_features = encode_features(shifted_hidden).detach()
            shuffled_features = encode_features(shuffled_hidden).detach()
            feature_drops = (clean_features - mask_features).detach()
            shifted_feature_drops = (clean_features - shifted_features).detach()
            shuffled_feature_drops = (clean_features - shuffled_features).detach()
            groups, diagnostics = _make_groups(
                model_family=args.model_family,
                clean_inputs=clean_inputs,
                clean_hidden=clean_hidden,
                union_hidden=mask_hidden,
                visual_positions=visual_positions,
                answer_positions=answer_positions,
                evidence_mask=evidence_mask,
                sample_id=sample_id,
                prompt_name=prompt_name,
                default_count=32,
                max_evidence_positions=64,
                random_controls=4,
            )
            group_lookup = {group["group_name"]: group for group in groups}
            selected_group = group_lookup.get(args.position_group)
            if selected_group is None or not selected_group["positions"]:
                skipped.append({"sample_id": sample_id, "prompt_name": prompt_name, "mask_condition": mask_condition, "reason": "position_group_missing"})
                continue
            positions = [int(position) for position in selected_group["positions"] if int(position) < clean_hidden.shape[1]]
            selection = _select_attribution_feature_groups(
                clean_features=clean_features,
                mask_features=mask_features,
                clean_hidden=clean_hidden,
                mask_hidden=mask_hidden,
                positions=positions,
                target_direction=target_direction,
                decoder_fn=decoder_vectors,
                top_k=args.top_k_features,
                control_pool_size=args.control_pool_size,
                seed=_stable_seed(args.model_family, sample_id, prompt_name, mask_condition, args.position_group),
            )
            if selection["status"] != "ok":
                skipped.append({"sample_id": sample_id, "prompt_name": prompt_name, "mask_condition": mask_condition, "reason": selection["status"]})
                continue
            source_candidates = selection["feature_groups"]["evidence_attribution_topk"]
            source_scores: list[tuple[float, int]] = []
            for feat_id in source_candidates:
                vectors = decoder_vectors([feat_id])
                zero_score = _score_with_feature_patch(
                    model=model,
                    inputs=clean_inputs,
                    tokenizer=tokenizer,
                    target_ids=[chosen_target_id],
                    module=target_module,
                    intervention={
                        "positions": positions,
                        "feature_ids": [feat_id],
                        "feature_values": clean_features,
                        "decoder_vectors": vectors,
                        "scale": args.scale,
                        "mode": "zero",
                    },
                )
                damage = clean_score["target_logit"] - zero_score["target_logit"]
                source_scores.append((float(damage), int(feat_id)))
            source_scores.sort(reverse=True)
            source_scores = [item for item in source_scores if item[0] > 0]
            if not source_scores:
                skipped.append({"sample_id": sample_id, "prompt_name": prompt_name, "mask_condition": mask_condition, "reason": "no_positive_source_zeroing"})
                continue
            source_feature = source_scores[0][1]
            control = _choose_control(selection, source_feature)
            if control is None:
                skipped.append({"sample_id": sample_id, "prompt_name": prompt_name, "mask_condition": mask_condition, "reason": "no_matched_control"})
                continue
            control_group, control_feature = control
            pair_features = [("source", "evidence_attribution_topk", source_feature), ("matched_control", control_group, control_feature)]
            clean_mask_gap = clean_score["target_logit"] - mask_score["target_logit"]
            usable_pairs += 1
            payload["pairs"].append(
                {
                    "sample_id": sample_id,
                    "prompt_name": prompt_name,
                    "mask_condition": mask_condition,
                    "source_feature": source_feature,
                    "control_feature": control_feature,
                    "control_group": control_group,
                    "source_zeroing_screen_damage": source_scores[0][0],
                    "diagnostics": diagnostics,
                    "selection": selection.get("feature_payload", {}),
                }
            )
            for feature_role, feature_group, feat_id in pair_features:
                vectors = decoder_vectors([feat_id])
                interventions = [
                    (
                        "zeroing",
                        "real_mask",
                        clean_inputs,
                        clean_score,
                        wrong_clean_score,
                        clean_features,
                        "zero",
                    ),
                    (
                        "restore",
                        "real_mask",
                        mask_inputs,
                        mask_score,
                        wrong_mask_score,
                        feature_drops,
                        "restore",
                    ),
                    (
                        "restore",
                        "mask_shifted",
                        shifted_inputs,
                        shifted_score,
                        {},
                        shifted_feature_drops,
                        "restore",
                    ),
                    (
                        "restore",
                        "mask_shuffled",
                        shuffled_inputs,
                        shuffled_score,
                        {},
                        shuffled_feature_drops,
                        "restore",
                    ),
                ]
                for intervention_name, mask_variant, inputs, reference, wrong_reference, feature_values, mode in interventions:
                    score = _score_with_feature_patch(
                        model=model,
                        inputs=inputs,
                        tokenizer=tokenizer,
                        target_ids=[chosen_target_id],
                        module=target_module,
                        intervention={
                            "positions": positions,
                            "feature_ids": [feat_id],
                            "feature_values": feature_values,
                            "decoder_vectors": vectors,
                            "scale": args.scale,
                            "mode": mode,
                        },
                    )
                    wrong_score = (
                        _score_with_feature_patch(
                            model=model,
                            inputs=inputs,
                            tokenizer=tokenizer,
                            target_ids=wrong_ids,
                            module=target_module,
                            intervention={
                                "positions": positions,
                                "feature_ids": [feat_id],
                                "feature_values": feature_values,
                                "decoder_vectors": vectors,
                                "scale": args.scale,
                                "mode": mode,
                            },
                        )
                        if wrong_ids and mask_variant == "real_mask"
                        else {}
                    )
                    if intervention_name == "restore":
                        effect_logit = score["target_logit"] - reference["target_logit"]
                        effect_rank = reference["target_rank"] - score["target_rank"]
                        wrong_effect_logit = (
                            wrong_score.get("target_logit", 0.0) - wrong_reference.get("target_logit", 0.0)
                            if wrong_score and wrong_reference
                            else ""
                        )
                    else:
                        effect_logit = reference["target_logit"] - score["target_logit"]
                        effect_rank = score["target_rank"] - reference["target_rank"]
                        wrong_effect_logit = (
                            wrong_reference.get("target_logit", 0.0) - wrong_score.get("target_logit", 0.0)
                            if wrong_score and wrong_reference
                            else ""
                        )
                    rows.append(
                        {
                            "model_family": args.model_family,
                            "sample_id": sample_id,
                            "prompt_name": prompt_name,
                            "layer": args.layer,
                            "bucket": bucket_name,
                            "mask_condition": mask_condition,
                            "mask_variant": mask_variant,
                            "position_group": args.position_group,
                            "position_count": len(positions),
                            "feature_role": feature_role,
                            "feature_group": feature_group,
                            "feature_id": feat_id,
                            "intervention": intervention_name,
                            "target_answer": sample["answer"],
                            "wrong_sample_id": wrong_sample_id,
                            "wrong_answer": wrong_answer,
                            "target_token_id": chosen_target_id,
                            "target_token": clean_score["target_token"],
                            "target_logit": score["target_logit"],
                            "target_rank": score["target_rank"],
                            "reference_target_logit": reference["target_logit"],
                            "reference_target_rank": reference["target_rank"],
                            "effect_logit": effect_logit,
                            "effect_rank": effect_rank,
                            "wrong_effect_logit": wrong_effect_logit,
                            "correct_minus_wrong_logit": effect_logit - wrong_effect_logit if wrong_effect_logit != "" else "",
                            "clean_mask_logit_gap": clean_mask_gap,
                            "gap_closure": _safe_gap_closure(float(effect_logit), clean_mask_gap),
                            "top1_token": score["top1_token"],
                            "reference_top1_token": reference["top1_token"],
                        }
                    )
            del (
                mask_outputs,
                shifted_outputs,
                shuffled_outputs,
                mask_features,
                shifted_features,
                shuffled_features,
                feature_drops,
                shifted_feature_drops,
                shuffled_feature_drops,
            )
            torch.cuda.empty_cache()
        del clean_outputs, clean_features
        torch.cuda.empty_cache()

    payload["gpu_after"] = _gpu_info()
    payload["decision"] = {
        "status": "pass_stage2o_source_control_probe" if usable_pairs else "blocked_no_usable_pairs",
        "usable_pairs": usable_pairs,
        "requested_runs": len(run_specs) * len(mask_conditions),
        "skipped": skipped,
    }
    fields = [
        "model_family",
        "sample_id",
        "prompt_name",
        "layer",
        "bucket",
        "mask_condition",
        "mask_variant",
        "position_group",
        "position_count",
        "feature_role",
        "feature_group",
        "feature_id",
        "intervention",
        "target_answer",
        "wrong_sample_id",
        "wrong_answer",
        "target_token_id",
        "target_token",
        "target_logit",
        "target_rank",
        "reference_target_logit",
        "reference_target_rank",
        "effect_logit",
        "effect_rank",
        "wrong_effect_logit",
        "correct_minus_wrong_logit",
        "clean_mask_logit_gap",
        "gap_closure",
        "top1_token",
        "reference_top1_token",
    ]
    _write_csv(Path(args.out_csv), rows, fields)
    _write_json(Path(args.out_json), payload)
    _log(f"done status={payload['decision']['status']} usable_pairs={usable_pairs} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
