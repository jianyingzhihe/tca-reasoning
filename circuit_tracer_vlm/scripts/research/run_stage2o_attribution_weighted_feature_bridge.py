#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
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
from run_cross_model_wrong_target_negative_control_smoke import _wrong_answer_for


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage2o-feature-bridge] {message}", flush=True)


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


def _stable_seed(*parts: str) -> int:
    value = 1729
    for part in parts:
        for char in part:
            value = (value * 131 + ord(char)) % (2**32)
    return value


def _safe_gap_closure(effect: float, gap: float) -> float | str:
    if abs(gap) <= 1e-6:
        return ""
    return effect / gap


def _load_samples(path: str) -> dict[str, dict[str, str]]:
    if path:
        return _load_sample_manifest(Path(path))
    return dict(SAMPLES)


def _load_run_manifest(path: str) -> list[dict[str, str]]:
    if not path:
        return []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {
                "sample_id": row.get("sample_id", "").strip(),
                "prompt_name": row.get("prompt_name", "").strip(),
            }
            for row in csv.DictReader(handle)
            if row.get("sample_id", "").strip() and row.get("prompt_name", "").strip()
        ]


def _run_specs(sample_ids: list[str], prompt_names: list[str], run_manifest: list[dict[str, str]]) -> list[tuple[str, str]]:
    if run_manifest:
        return [(row["sample_id"], row["prompt_name"]) for row in run_manifest]
    return [(sample_id, prompt_name) for sample_id in sample_ids for prompt_name in prompt_names]


def _select_nearest(
    candidate_ids: list[int],
    scores: torch.Tensor,
    target_value: float,
    count: int,
    excluded: set[int],
) -> list[int]:
    ranked: list[tuple[float, int]] = []
    for feat_id in candidate_ids:
        if feat_id in excluded:
            continue
        ranked.append((abs(float(scores[feat_id].item()) - target_value), int(feat_id)))
    ranked.sort()
    return [feat_id for _dist, feat_id in ranked[:count]]


def _decoder_vectors_for(decoder_fn, feature_ids: list[int]) -> torch.Tensor:
    if not feature_ids:
        raise ValueError("feature_ids is empty")
    return decoder_fn(feature_ids)


def _select_attribution_feature_groups(
    *,
    clean_features: torch.Tensor,
    mask_features: torch.Tensor,
    clean_hidden: torch.Tensor,
    mask_hidden: torch.Tensor,
    positions: list[int],
    target_direction: torch.Tensor,
    decoder_fn,
    top_k: int,
    control_pool_size: int,
    seed: int,
) -> dict[str, Any]:
    if not positions:
        return {"status": "empty_positions"}
    pos = torch.tensor(positions, device=clean_features.device, dtype=torch.long)
    clean = clean_features[:, pos, :].detach()
    mask = mask_features[:, pos, :].detach()
    clean_max = clean.amax(dim=(0, 1)).float()
    mask_max = mask.amax(dim=(0, 1)).float()
    raw_drop = (clean - mask).float().clamp_min(0)
    hidden_delta = (clean_hidden[:, pos, :] - mask_hidden[:, pos, :]).float().norm(dim=-1).squeeze(0)
    if hidden_delta.numel() != len(positions) or float(hidden_delta.sum().item()) <= 1e-6:
        position_weight = torch.ones(len(positions), device=clean_features.device, dtype=torch.float32)
    else:
        position_weight = hidden_delta / hidden_delta.sum()
    weighted_drop = (raw_drop.squeeze(0) * position_weight[:, None]).sum(dim=0)
    active = clean_max > 0
    positive_drop = weighted_drop > 0
    if not (active & positive_drop).any():
        return {"status": "no_positive_drop_features"}

    pool_k = min(control_pool_size, clean_max.numel())
    _clean_vals, clean_pool_tensor = torch.topk(clean_max, k=pool_k)
    _drop_vals, drop_pool_tensor = torch.topk(weighted_drop, k=pool_k)
    candidate_ids = sorted(
        {
            int(x)
            for x in clean_pool_tensor.detach().cpu().tolist() + drop_pool_tensor.detach().cpu().tolist()
            if bool(active[int(x)].item())
        }
    )
    if len(candidate_ids) < top_k * 5:
        return {"status": "insufficient_candidate_features", "candidate_count": len(candidate_ids)}

    decoder_vectors = _decoder_vectors_for(decoder_fn, candidate_ids)
    target = target_direction.to(decoder_vectors.device, dtype=decoder_vectors.dtype)
    target_contrib_values = torch.mv(decoder_vectors.float(), target.float()).to(clean_features.device)
    target_contrib = torch.zeros(clean_max.numel(), device=clean_features.device, dtype=torch.float32)
    target_contrib[torch.tensor(candidate_ids, device=clean_features.device, dtype=torch.long)] = target_contrib_values.float()
    feature_score = weighted_drop.clamp_min(0) * target_contrib.clamp_min(0)
    feature_score[~active] = -float("inf")
    if not torch.isfinite(feature_score).any() or float(feature_score.max().item()) <= 0:
        return {"status": "no_positive_attribution_features"}
    k = min(top_k, int(torch.isfinite(feature_score).sum().item()))
    evidence_vals, evidence_ids_tensor = torch.topk(feature_score, k=k)
    evidence_ids = [int(x) for x in evidence_ids_tensor.detach().cpu().tolist()]
    evidence_set = set(evidence_ids)
    evidence_mean_activation = float(clean_max[evidence_ids_tensor].mean().item())
    evidence_mean_drop = float(weighted_drop[evidence_ids_tensor].mean().item())
    evidence_mean_contrib = float(target_contrib[evidence_ids_tensor].mean().item())

    active_pool = [feat_id for feat_id in candidate_ids if feat_id not in evidence_set]
    if len(active_pool) < k * 4:
        return {"status": "insufficient_control_features", "candidate_count": len(active_pool), "required": k * 4}

    activation_matched = _select_nearest(active_pool, clean_max, evidence_mean_activation, k, evidence_set)
    drop_matched = _select_nearest(active_pool, weighted_drop, evidence_mean_drop, k, evidence_set | set(activation_matched))
    excluded = evidence_set | set(activation_matched) | set(drop_matched)
    insensitive_candidates = [
        feat_id
        for feat_id in active_pool
        if feat_id not in excluded and float(weighted_drop[feat_id].item()) <= max(1e-6, 0.1 * max(evidence_mean_drop, 1e-6))
    ]
    if len(insensitive_candidates) < k:
        insensitive_candidates = [feat_id for feat_id in active_pool if feat_id not in excluded]
    insensitive_candidates.sort(
        key=lambda feat_id: (
            abs(float(target_contrib[feat_id].item()) - evidence_mean_contrib),
            abs(float(weighted_drop[feat_id].item())),
        )
    )
    attribution_matched_mask_insensitive = insensitive_candidates[:k]
    excluded |= set(attribution_matched_mask_insensitive)
    random_candidates = [feat_id for feat_id in active_pool if feat_id not in excluded]
    rng = random.Random(seed)
    rng.shuffle(random_candidates)
    random_active = random_candidates[:k]

    groups = {
        "evidence_attribution_topk": evidence_ids,
        "activation_matched_topk": activation_matched,
        "drop_matched_topk": drop_matched,
        "attribution_matched_mask_insensitive_topk": attribution_matched_mask_insensitive,
        "random_active_topk": random_active,
    }
    if any(len(ids) < k for ids in groups.values()):
        return {"status": "incomplete_feature_controls", "group_sizes": {name: len(ids) for name, ids in groups.items()}}

    def payload(feature_ids: list[int]) -> list[dict[str, Any]]:
        return [
            {
                "feature_id": int(feat_id),
                "clean_activation": float(clean_max[feat_id].item()),
                "mask_activation": float(mask_max[feat_id].item()),
                "weighted_drop": float(weighted_drop[feat_id].item()),
                "target_contribution": float(target_contrib[feat_id].item()),
                "feature_score": float(feature_score[feat_id].item()) if torch.isfinite(feature_score[feat_id]) else "",
            }
            for feat_id in feature_ids
        ]

    return {
        "status": "ok",
        "top_k": k,
        "feature_groups": groups,
        "feature_payload": {name: payload(ids) for name, ids in groups.items()},
        "evidence_mean_activation": evidence_mean_activation,
        "evidence_mean_drop": evidence_mean_drop,
        "evidence_mean_contribution": evidence_mean_contrib,
        "evidence_mean_score": float(evidence_vals.float().mean().item()),
    }


def _patch_delta_tensor(
    hidden: torch.Tensor,
    *,
    positions: list[int],
    feature_ids: list[int],
    feature_drops: torch.Tensor,
    decoder_vectors: torch.Tensor,
    scale: float,
    mode: str,
) -> torch.Tensor:
    if not positions or not feature_ids:
        return hidden
    pos_tensor = torch.tensor(positions, device=hidden.device, dtype=torch.long)
    feat_tensor = torch.tensor(feature_ids, device=feature_drops.device, dtype=torch.long)
    drops = feature_drops[:, pos_tensor, :][:, :, feat_tensor].to(hidden.device, dtype=hidden.dtype)
    vectors = decoder_vectors.to(hidden.device, dtype=hidden.dtype)
    patch = torch.einsum("bpf,fd->bpd", drops, vectors)
    new_hidden = hidden.clone()
    if mode == "restore":
        new_hidden[:, pos_tensor, :] = new_hidden[:, pos_tensor, :] + scale * patch
    elif mode == "corrupt":
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
            patched = _patch_delta_tensor(hidden, **intervention)
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
    parser = argparse.ArgumentParser(description="Stage 2O attribution-weighted feature bridge.")
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
    parser.add_argument("--mask-condition", default="union_mask", choices=["answer_mask", "union_mask"])
    parser.add_argument(
        "--position-groups",
        default="top_hidden_delta_plus_answer_adjacent,top_hidden_delta,answer_adjacent_text",
    )
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
    position_group_names = _parse_csv(args.position_groups)
    roots = [Path(path) for path in _parse_csv(args.annotation_roots)]
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "created_at": _now(),
        "model_family": args.model_family,
        "model_name": args.model_name,
        "transcoder_ref": args.transcoder_ref,
        "layer": args.layer,
        "mask_condition": args.mask_condition,
        "position_groups": position_group_names,
        "run_count": len(run_specs),
        "original_run_count": original_run_count,
        "max_runs": args.max_runs,
        "env_presence": _env_presence(),
        "gpu_before": _gpu_info(),
        "selection": [],
        "decision": {},
        "claim_boundary": (
            "Stage 2O attribution-weighted feature bridge. Positive results can support feature-level "
            "bridge only; they do not establish Gemma-style source-control route replication."
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
    usable_runs = 0
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
        evidence_mask = _mask_for_condition(mask_info, args.mask_condition)
        mask_image = _apply_mask(clean_image, evidence_mask, (128, 128, 128))
        question = _prompt(sample["question"], prompt_name)
        clean_inputs = build_inputs(clean_image, image_path, question)
        mask_inputs = build_inputs(mask_image, image_path, question)
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
            mask_outputs = model(**mask_inputs, output_hidden_states=True, use_cache=False)
        clean_score = _rank_and_top(clean_outputs.logits, tokenizer, target_ids)
        chosen_target_id = int(clean_score["target_token_id"])
        mask_score = _rank_and_top(mask_outputs.logits, tokenizer, [chosen_target_id])
        wrong_sample_id, wrong_answer, wrong_candidates = _wrong_answer_for(
            sample_ids=list(sample_lookup),
            sample_lookup=sample_lookup,
            sample_id=sample_id,
            tokenizer=tokenizer,
            correct_ids={chosen_target_id},
        )
        wrong_ids = [int(item["token_id"]) for item in wrong_candidates]

        clean_hidden = clean_outputs.hidden_states[hidden_index].to(device=device, dtype=dtype).detach()
        mask_hidden = mask_outputs.hidden_states[hidden_index].to(device=device, dtype=dtype).detach()
        clean_features = encode_features(clean_hidden).detach()
        mask_features = encode_features(mask_hidden).detach()
        feature_drops = (clean_features - mask_features).detach()
        clean_mask_logit_gap = clean_score["target_logit"] - mask_score["target_logit"]
        clean_mask_rank_gap = mask_score["target_rank"] - clean_score["target_rank"]
        target_direction = output_weight[chosen_target_id].detach().to(device=device, dtype=dtype)
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
        usable_this_run = False

        for position_group in position_group_names:
            selected_group = group_lookup.get(position_group)
            if selected_group is None or not selected_group["positions"]:
                skipped.append({"sample_id": sample_id, "prompt_name": prompt_name, "position_group": position_group, "reason": "position_group_missing"})
                continue
            positions = [int(position) for position in selected_group["positions"] if int(position) < clean_hidden.shape[1]]
            if not positions:
                skipped.append({"sample_id": sample_id, "prompt_name": prompt_name, "position_group": position_group, "reason": "position_group_out_of_range"})
                continue
            feature_selection = _select_attribution_feature_groups(
                clean_features=clean_features,
                mask_features=mask_features,
                clean_hidden=clean_hidden,
                mask_hidden=mask_hidden,
                positions=positions,
                target_direction=target_direction,
                decoder_fn=decoder_vectors,
                top_k=args.top_k_features,
                control_pool_size=args.control_pool_size,
                seed=_stable_seed(args.model_family, sample_id, prompt_name, position_group, args.mask_condition),
            )
            payload["selection"].append(
                {
                    "sample_id": sample_id,
                    "prompt_name": prompt_name,
                    "bucket": bucket_name,
                    "mask_condition": args.mask_condition,
                    "position_group": position_group,
                    "position_count": len(positions),
                    "status": feature_selection["status"],
                    "diagnostics": diagnostics,
                    "feature_payload": feature_selection.get("feature_payload", {}),
                }
            )
            if feature_selection["status"] != "ok":
                skipped.append({"sample_id": sample_id, "prompt_name": prompt_name, "position_group": position_group, "reason": feature_selection["status"]})
                continue
            usable_this_run = True
            feature_groups: dict[str, list[int]] = feature_selection["feature_groups"]
            for feature_group_name, feature_ids in feature_groups.items():
                vectors = decoder_vectors(feature_ids)
                for direction in ["restore", "corrupt"]:
                    if direction == "restore":
                        inputs = mask_inputs
                        reference = mask_score
                        intervention = {
                            "positions": positions,
                            "feature_ids": feature_ids,
                            "feature_drops": feature_drops,
                            "decoder_vectors": vectors,
                            "scale": args.scale,
                            "mode": "restore",
                        }
                    else:
                        inputs = clean_inputs
                        reference = clean_score
                        intervention = {
                            "positions": positions,
                            "feature_ids": feature_ids,
                            "feature_drops": feature_drops,
                            "decoder_vectors": vectors,
                            "scale": args.scale,
                            "mode": "corrupt",
                        }
                    score = _score_with_feature_patch(
                        model=model,
                        inputs=inputs,
                        tokenizer=tokenizer,
                        target_ids=[chosen_target_id],
                        module=target_module,
                        intervention=intervention,
                    )
                    wrong_score = (
                        _score_with_feature_patch(
                            model=model,
                            inputs=inputs,
                            tokenizer=tokenizer,
                            target_ids=wrong_ids,
                            module=target_module,
                            intervention=intervention,
                        )
                        if wrong_ids
                        else {}
                    )
                    wrong_reference = (
                        _rank_and_top(reference.get("logits", None), tokenizer, wrong_ids) if False else {}
                    )
                    logit_restore = score["target_logit"] - mask_score["target_logit"] if direction == "restore" else ""
                    rank_restore = mask_score["target_rank"] - score["target_rank"] if direction == "restore" else ""
                    logit_damage = clean_score["target_logit"] - score["target_logit"] if direction == "corrupt" else ""
                    rank_damage = score["target_rank"] - clean_score["target_rank"] if direction == "corrupt" else ""
                    effect = logit_restore if direction == "restore" else logit_damage
                    rows.append(
                        {
                            "model_family": args.model_family,
                            "sample_id": sample_id,
                            "prompt_name": prompt_name,
                            "layer": args.layer,
                            "bucket": bucket_name,
                            "mask_condition": args.mask_condition,
                            "position_group": position_group,
                            "position_count": len(positions),
                            "direction": direction,
                            "feature_group": feature_group_name,
                            "feature_ids": "|".join(str(x) for x in feature_ids),
                            "target_answer": sample["answer"],
                            "wrong_sample_id": wrong_sample_id,
                            "wrong_answer": wrong_answer,
                            "target_token_id": chosen_target_id,
                            "target_token": clean_score["target_token"],
                            "target_logit": score["target_logit"],
                            "target_rank": score["target_rank"],
                            "clean_target_logit": clean_score["target_logit"],
                            "clean_target_rank": clean_score["target_rank"],
                            "mask_target_logit": mask_score["target_logit"],
                            "mask_target_rank": mask_score["target_rank"],
                            "clean_mask_logit_gap": clean_mask_logit_gap,
                            "clean_mask_rank_gap": clean_mask_rank_gap,
                            "logit_restore_vs_mask": logit_restore,
                            "rank_restore_vs_mask": rank_restore,
                            "logit_damage_vs_clean": logit_damage,
                            "rank_damage_vs_clean": rank_damage,
                            "gap_closure": _safe_gap_closure(float(effect), clean_mask_logit_gap) if effect != "" else "",
                            "wrong_target_logit": wrong_score.get("target_logit", ""),
                            "wrong_target_rank": wrong_score.get("target_rank", ""),
                            "top1_token": score["top1_token"],
                            "clean_top1_token": clean_score["top1_token"],
                            "mask_top1_token": mask_score["top1_token"],
                        }
                    )
        if usable_this_run:
            usable_runs += 1
        del clean_outputs, mask_outputs, clean_features, mask_features, feature_drops
        torch.cuda.empty_cache()

    payload["gpu_after"] = _gpu_info()
    payload["decision"] = {
        "status": "pass_stage2o_feature_bridge" if usable_runs else "blocked_no_usable_runs",
        "usable_runs": usable_runs,
        "requested_runs": len(run_specs),
        "skipped": skipped,
    }
    fields = [
        "model_family",
        "sample_id",
        "prompt_name",
        "layer",
        "bucket",
        "mask_condition",
        "position_group",
        "position_count",
        "direction",
        "feature_group",
        "feature_ids",
        "target_answer",
        "wrong_sample_id",
        "wrong_answer",
        "target_token_id",
        "target_token",
        "target_logit",
        "target_rank",
        "clean_target_logit",
        "clean_target_rank",
        "mask_target_logit",
        "mask_target_rank",
        "clean_mask_logit_gap",
        "clean_mask_rank_gap",
        "logit_restore_vs_mask",
        "rank_restore_vs_mask",
        "logit_damage_vs_clean",
        "rank_damage_vs_clean",
        "gap_closure",
        "wrong_target_logit",
        "wrong_target_rank",
        "top1_token",
        "clean_top1_token",
        "mask_top1_token",
    ]
    _write_csv(Path(args.out_csv), rows, fields)
    _write_json(Path(args.out_json), payload)
    _log(f"done status={payload['decision']['status']} usable_runs={usable_runs} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
