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
    _apply_mask,
    _env_presence,
    _first_param_device,
    _gpu_info,
    _load_masks,
    _parse_csv,
    _prompt,
    _qwen_bucket_positions,
    _qwen_inputs,
    _replace_hidden,
)
from run_cross_model_hidden_position_patch_smoke import (
    _answer_adjacent_positions,
    _load_sample_manifest,
    _make_groups,
)
from run_stage2o_attribution_weighted_feature_bridge import (
    _patch_delta_tensor,
    _select_attribution_feature_groups,
    _stable_seed,
)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage3-qwen-sequence-bridge] {message}", flush=True)


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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _decode_ids(tokenizer, ids: list[int]) -> str:
    if not ids:
        return ""
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def _answer_token_ids(tokenizer, answer: str) -> list[int]:
    ids = tokenizer.encode((answer or "").strip(), add_special_tokens=False)
    return [int(x) for x in ids if int(x) >= 0]


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


def _filter_delta_intervention(intervention: dict[str, Any], seq_len: int) -> dict[str, Any] | None:
    positions = [
        int(position)
        for position in intervention["positions"]
        if int(position) < seq_len and int(position) < int(intervention["feature_drops"].shape[1])
    ]
    if not positions:
        return None
    return {**intervention, "positions": positions}


def _register_delta_hook(module, intervention: dict[str, Any] | None):
    if intervention is None:
        return None

    def _hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        filtered = _filter_delta_intervention(intervention, int(hidden.shape[1]))
        if filtered is None:
            return output
        patched = _patch_delta_tensor(hidden, **filtered)
        return _replace_hidden(output, patched)

    return module.register_forward_hook(_hook)


def _rank_from_logits(row: torch.Tensor, tokenizer, target_id: int) -> dict[str, Any]:
    row = row.float()
    target_logit = float(row[target_id].item())
    rank = int((row > row[target_id]).sum().item() + 1)
    top_vals, top_ids = torch.topk(row, k=5)
    return {
        "target_logit": target_logit,
        "target_rank": rank,
        "top1_token_id": int(top_ids[0].item()),
        "top1_token": tokenizer.decode([int(top_ids[0].item())]),
        "top1_logit": float(top_vals[0].item()),
    }


def _sequence_score_with_patch(
    *,
    model,
    tokenizer,
    base_inputs: dict[str, Any],
    module,
    intervention: dict[str, Any] | None,
    answer_ids: list[int],
    first_target_id: int,
) -> dict[str, Any]:
    input_ids = base_inputs["input_ids"].clone()
    prompt_len = int(input_ids.shape[1])
    answer_tensor = torch.tensor([answer_ids], device=input_ids.device, dtype=input_ids.dtype)
    extended_ids = torch.cat([input_ids, answer_tensor], dim=1)
    extended_inputs = _clone_inputs_with_ids(base_inputs, extended_ids)
    handle = _register_delta_hook(module, intervention)
    try:
        with torch.inference_mode():
            outputs = model(**extended_inputs, output_hidden_states=False, use_cache=False)
    finally:
        if handle is not None:
            handle.remove()
    logits = outputs.logits[0].float()
    pred_logits = logits[prompt_len - 1 : prompt_len - 1 + len(answer_ids), :]
    log_probs = torch.log_softmax(pred_logits, dim=-1)
    ids_tensor = torch.tensor(answer_ids, device=log_probs.device, dtype=torch.long)
    token_logprobs = log_probs[torch.arange(len(answer_ids), device=log_probs.device), ids_tensor]
    first = _rank_from_logits(logits[prompt_len - 1, :], tokenizer, first_target_id)
    return {
        "answer_token_ids": "|".join(str(x) for x in answer_ids),
        "answer_token_text": tokenizer.decode(answer_ids),
        "sequence_logprob_sum": float(token_logprobs.sum().item()),
        "sequence_logprob_mean": float(token_logprobs.mean().item()),
        "sequence_token_logprobs": "|".join(f"{float(x):.6f}" for x in token_logprobs.detach().cpu().tolist()),
        "first_token_logit": first["target_logit"],
        "first_token_rank": first["target_rank"],
        "first_token_top1": first["top1_token"],
    }


def _greedy_decode_with_patch(
    *,
    model,
    tokenizer,
    base_inputs: dict[str, Any],
    module,
    intervention: dict[str, Any] | None,
    first_target_id: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    input_ids = base_inputs["input_ids"].clone()
    generated: list[int] = []
    first_step: dict[str, Any] | None = None
    handle = _register_delta_hook(module, intervention)
    try:
        for _ in range(max_new_tokens):
            step_inputs = _clone_inputs_with_ids(base_inputs, input_ids)
            with torch.inference_mode():
                outputs = model(**step_inputs, output_hidden_states=False, use_cache=False)
            if first_step is None:
                first_step = _rank_from_logits(outputs.logits[0, -1, :], tokenizer, first_target_id)
            next_id = int(torch.argmax(outputs.logits[0, -1].float()).item())
            generated.append(next_id)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], device=input_ids.device, dtype=input_ids.dtype)],
                dim=1,
            )
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is not None and next_id == int(eos_id):
                break
    finally:
        if handle is not None:
            handle.remove()
    generated_text = f"The answer is {_decode_ids(tokenizer, generated)}".strip()
    return {
        "generated_token_ids": "|".join(str(x) for x in generated),
        "generated_text": generated_text,
        "predicted_answer": _normalize_answer(generated_text),
        "first_generated_token": _decode_ids(tokenizer, generated[:1]),
        "greedy_first_token_logit": (first_step or {}).get("target_logit", ""),
        "greedy_first_token_rank": (first_step or {}).get("target_rank", ""),
        "greedy_first_token_top1": (first_step or {}).get("top1_token", ""),
    }


def _make_intervention(
    *,
    positions: list[int],
    feature_ids: list[int],
    feature_drops: torch.Tensor,
    decoder_vectors: torch.Tensor,
    mode: str,
    scale: float,
) -> dict[str, Any]:
    return {
        "positions": positions,
        "feature_ids": feature_ids,
        "feature_drops": feature_drops,
        "decoder_vectors": decoder_vectors,
        "scale": scale,
        "mode": mode,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage3 Qwen multi-feature answer sequence bridge.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--transcoder-ref", required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--annotation-roots", required=True)
    parser.add_argument("--sample-manifest", required=True)
    parser.add_argument("--bridge-manifest", required=True)
    parser.add_argument("--topks", default="1,4,8,16,32")
    parser.add_argument(
        "--feature-groups",
        default="evidence_attribution_topk,activation_matched_topk,drop_matched_topk,attribution_matched_mask_insensitive_topk,random_active_topk",
    )
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--answer-prefix", default="The answer is ")
    parser.add_argument("--min-gpu-free-gb", type=float, default=18.0)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    bridge_rows = [row for row in _read_csv(Path(args.bridge_manifest)) if row.get("asset_id") == args.asset_id]
    if args.max_pairs and args.max_pairs > 0:
        bridge_rows = bridge_rows[: args.max_pairs]
    sample_lookup = _load_sample_manifest(Path(args.sample_manifest))
    roots = [Path(path) for path in _parse_csv(args.annotation_roots)]
    topks = [int(x) for x in _parse_csv(args.topks)]
    feature_group_names = _parse_csv(args.feature_groups)
    max_topk = max(topks)
    work_dir = Path(args.out_json).parent / f"{args.asset_id}_sequence_bridge_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "created_at": _now(),
        "asset_id": args.asset_id,
        "model_name": args.model_name,
        "transcoder_ref": args.transcoder_ref,
        "pair_count": len(bridge_rows),
        "topks": topks,
        "feature_groups": feature_group_names,
        "env_presence": _env_presence(),
        "gpu_before": _gpu_info(),
        "claim_boundary": (
            "Stage3 Qwen sequence bridge tests answer sequence likelihood and short greedy decoding. "
            "It is not Gemma-style source tracing and not a full generation-level claim by itself."
        ),
        "skipped": [],
        "selection": [],
    }
    gpu = payload["gpu_before"]
    if not gpu.get("available") or float(gpu.get("free_gb", 0.0)) < args.min_gpu_free_gb:
        payload["decision"] = {"status": "blocked", "reason": "insufficient_gpu_free_memory"}
        _write_json(Path(args.out_json), payload)
        return 0

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
    model.eval()
    device = _first_param_device(model)
    dtype = torch.bfloat16
    transcoders, config = load_transcoder_from_hub(
        args.transcoder_ref,
        device=device,
        dtype=dtype,
        lazy_encoder=True,
        lazy_decoder=True,
    )
    payload["transcoder"] = {"type": type(transcoders).__name__, "config_model_kind": config.get("model_kind", "")}

    def encode_features(hidden: torch.Tensor, layer: int) -> torch.Tensor:
        return transcoders.encode_layer(hidden.to(device), layer, apply_activation_function=True)

    def decoder_vectors(layer: int, feature_ids: list[int]) -> torch.Tensor:
        ids = torch.tensor(feature_ids, device=device, dtype=torch.long)
        vectors = transcoders._get_decoder_vectors(layer, ids)
        if vectors.ndim == 3:
            vectors = vectors[:, 0, :]
        return vectors.to(device=device, dtype=dtype)

    out_rows: list[dict[str, Any]] = []
    for manifest_row in bridge_rows:
        sample_id = manifest_row["sample_id"]
        prompt_name = manifest_row["prompt_name"]
        mask_condition = manifest_row["mask_condition"]
        layer = int(manifest_row.get("layer", 26) or 26)
        target_module = model.language_model.layers[layer]
        hidden_index = layer
        sample = sample_lookup.get(sample_id)
        if sample is None:
            payload["skipped"].append({"sample_id": sample_id, "prompt_name": prompt_name, "reason": "sample_missing"})
            continue
        answer_text = manifest_row.get("target_answer") or sample.get("answer", "")
        answer_ids = _answer_token_ids(tokenizer, answer_text)
        if not answer_ids:
            payload["skipped"].append({"sample_id": sample_id, "prompt_name": prompt_name, "reason": "answer_tokenization_empty"})
            continue
        first_target_id = int(manifest_row.get("target_token_id") or answer_ids[0])
        mask_info = _load_masks(sample, roots, work_dir)
        if mask_info["status"] != "ok":
            payload["skipped"].append({"sample_id": sample_id, "prompt_name": prompt_name, "reason": "mask_missing", "mask_info": mask_info})
            continue
        image_path = Path(mask_info["image_path"])
        clean_image = Image.open(image_path).convert("RGB")
        evidence_mask = mask_info["answer_mask"] if mask_condition == "answer_mask" else mask_info["union_mask"]
        mask_image = _apply_mask(clean_image, evidence_mask, (128, 128, 128))
        question = _prompt(sample["question"], prompt_name)
        clean_inputs = _qwen_inputs(processor, clean_image, str(image_path), question, args.answer_prefix, device)
        mask_inputs = _qwen_inputs(processor, mask_image, str(image_path), question, args.answer_prefix, device)
        input_ids = clean_inputs["input_ids"][0].detach().cpu().tolist()
        token_texts = tokenizer.convert_ids_to_tokens(input_ids)
        buckets = _qwen_bucket_positions(input_ids, token_texts)
        visual_positions = buckets.get("image_marker_or_span", [])
        answer_positions = _answer_adjacent_positions(len(input_ids), set(visual_positions), 4)
        if not visual_positions:
            payload["skipped"].append({"sample_id": sample_id, "prompt_name": prompt_name, "reason": "visual_positions_missing"})
            continue

        with torch.inference_mode():
            clean_outputs = model(**clean_inputs, output_hidden_states=True, use_cache=False)
            mask_outputs = model(**mask_inputs, output_hidden_states=True, use_cache=False)
        clean_hidden = clean_outputs.hidden_states[hidden_index].to(device=device, dtype=dtype).detach()
        mask_hidden = mask_outputs.hidden_states[hidden_index].to(device=device, dtype=dtype).detach()
        clean_features = encode_features(clean_hidden, layer).detach()
        mask_features = encode_features(mask_hidden, layer).detach()
        feature_drops = (clean_features - mask_features).detach()
        output_weight = model.get_output_embeddings().weight
        target_direction = output_weight[first_target_id].detach().to(device=device, dtype=dtype)
        groups, diagnostics = _make_groups(
            model_family="qwen",
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
        selected_group = group_lookup.get("top_hidden_delta_plus_answer_adjacent")
        if selected_group is None or not selected_group["positions"]:
            payload["skipped"].append({"sample_id": sample_id, "prompt_name": prompt_name, "reason": "position_group_missing"})
            continue
        positions = [int(position) for position in selected_group["positions"] if int(position) < clean_hidden.shape[1]]
        selection = _select_attribution_feature_groups(
            clean_features=clean_features,
            mask_features=mask_features,
            clean_hidden=clean_hidden,
            mask_hidden=mask_hidden,
            positions=positions,
            target_direction=target_direction,
            decoder_fn=lambda feature_ids: decoder_vectors(layer, feature_ids),
            top_k=max_topk,
            control_pool_size=4096,
            seed=_stable_seed(args.asset_id, sample_id, prompt_name, mask_condition, "stage3_13"),
        )
        payload["selection"].append(
            {
                "sample_id": sample_id,
                "prompt_name": prompt_name,
                "mask_condition": mask_condition,
                "position_count": len(positions),
                "status": selection["status"],
                "diagnostics": diagnostics,
                "feature_payload": selection.get("feature_payload", {}),
            }
        )
        if selection["status"] != "ok":
            payload["skipped"].append({"sample_id": sample_id, "prompt_name": prompt_name, "reason": selection["status"]})
            continue

        clean_seq = _sequence_score_with_patch(
            model=model,
            tokenizer=tokenizer,
            base_inputs=clean_inputs,
            module=target_module,
            intervention=None,
            answer_ids=answer_ids,
            first_target_id=first_target_id,
        )
        mask_seq = _sequence_score_with_patch(
            model=model,
            tokenizer=tokenizer,
            base_inputs=mask_inputs,
            module=target_module,
            intervention=None,
            answer_ids=answer_ids,
            first_target_id=first_target_id,
        )
        clean_decoded = _greedy_decode_with_patch(
            model=model,
            tokenizer=tokenizer,
            base_inputs=clean_inputs,
            module=target_module,
            intervention=None,
            first_target_id=first_target_id,
            max_new_tokens=args.max_new_tokens,
        )
        mask_decoded = _greedy_decode_with_patch(
            model=model,
            tokenizer=tokenizer,
            base_inputs=mask_inputs,
            module=target_module,
            intervention=None,
            first_target_id=first_target_id,
            max_new_tokens=args.max_new_tokens,
        )
        clean_answer = clean_decoded["predicted_answer"]
        mask_answer = mask_decoded["predicted_answer"]
        clean_mask_seq_gap = clean_seq["sequence_logprob_sum"] - mask_seq["sequence_logprob_sum"]
        clean_mask_first_logit_gap = clean_seq["first_token_logit"] - mask_seq["first_token_logit"]

        out_rows.append(
            {
                "asset_id": args.asset_id,
                "sample_id": sample_id,
                "prompt_name": prompt_name,
                "mask_condition": mask_condition,
                "condition": "baseline_clean",
                "direction": "baseline",
                "feature_group": "",
                "top_k": "",
                "feature_ids": "",
                "target_answer": answer_text,
                "answer_token_ids": clean_seq["answer_token_ids"],
                "answer_token_text": clean_seq["answer_token_text"],
                "sequence_logprob_sum": clean_seq["sequence_logprob_sum"],
                "sequence_logprob_mean": clean_seq["sequence_logprob_mean"],
                "sequence_effect_vs_reference": "",
                "first_token_logit": clean_seq["first_token_logit"],
                "first_token_rank": clean_seq["first_token_rank"],
                "first_token_effect_vs_reference": "",
                "first_token_rank_effect_vs_reference": "",
                "generated_text": clean_decoded["generated_text"],
                "predicted_answer": clean_answer,
                "target_hit": _target_hit(clean_decoded["generated_text"], answer_text),
                "baseline_clean_answer": clean_answer,
                "baseline_mask_answer": mask_answer,
                "decoded_to_clean": "",
                "decoded_changed_vs_reference": "",
                "clean_mask_sequence_gap": clean_mask_seq_gap,
                "clean_mask_first_logit_gap": clean_mask_first_logit_gap,
            }
        )
        out_rows.append(
            {
                "asset_id": args.asset_id,
                "sample_id": sample_id,
                "prompt_name": prompt_name,
                "mask_condition": mask_condition,
                "condition": "baseline_mask",
                "direction": "baseline",
                "feature_group": "",
                "top_k": "",
                "feature_ids": "",
                "target_answer": answer_text,
                "answer_token_ids": mask_seq["answer_token_ids"],
                "answer_token_text": mask_seq["answer_token_text"],
                "sequence_logprob_sum": mask_seq["sequence_logprob_sum"],
                "sequence_logprob_mean": mask_seq["sequence_logprob_mean"],
                "sequence_effect_vs_reference": "",
                "first_token_logit": mask_seq["first_token_logit"],
                "first_token_rank": mask_seq["first_token_rank"],
                "first_token_effect_vs_reference": "",
                "first_token_rank_effect_vs_reference": "",
                "generated_text": mask_decoded["generated_text"],
                "predicted_answer": mask_answer,
                "target_hit": _target_hit(mask_decoded["generated_text"], answer_text),
                "baseline_clean_answer": clean_answer,
                "baseline_mask_answer": mask_answer,
                "decoded_to_clean": "",
                "decoded_changed_vs_reference": "",
                "clean_mask_sequence_gap": clean_mask_seq_gap,
                "clean_mask_first_logit_gap": clean_mask_first_logit_gap,
            }
        )

        feature_groups: dict[str, list[int]] = selection["feature_groups"]
        for top_k in topks:
            for feature_group in feature_group_names:
                full_ids = feature_groups.get(feature_group, [])
                if len(full_ids) < top_k:
                    continue
                feature_ids = [int(x) for x in full_ids[:top_k]]
                vectors = decoder_vectors(layer, feature_ids)
                restore_intervention = _make_intervention(
                    positions=positions,
                    feature_ids=feature_ids,
                    feature_drops=feature_drops,
                    decoder_vectors=vectors,
                    mode="restore",
                    scale=args.scale,
                )
                corrupt_intervention = _make_intervention(
                    positions=positions,
                    feature_ids=feature_ids,
                    feature_drops=feature_drops,
                    decoder_vectors=vectors,
                    mode="corrupt",
                    scale=args.scale,
                )
                for direction, condition, base_inputs, reference_seq, reference_answer, intervention in [
                    ("restore", f"{feature_group}_restore_top{top_k}", mask_inputs, mask_seq, mask_answer, restore_intervention),
                    ("corrupt", f"{feature_group}_corrupt_top{top_k}", clean_inputs, clean_seq, clean_answer, corrupt_intervention),
                ]:
                    seq = _sequence_score_with_patch(
                        model=model,
                        tokenizer=tokenizer,
                        base_inputs=base_inputs,
                        module=target_module,
                        intervention=intervention,
                        answer_ids=answer_ids,
                        first_target_id=first_target_id,
                    )
                    decoded = _greedy_decode_with_patch(
                        model=model,
                        tokenizer=tokenizer,
                        base_inputs=base_inputs,
                        module=target_module,
                        intervention=intervention,
                        first_target_id=first_target_id,
                        max_new_tokens=args.max_new_tokens,
                    )
                    if direction == "restore":
                        sequence_effect = seq["sequence_logprob_sum"] - reference_seq["sequence_logprob_sum"]
                        first_effect = seq["first_token_logit"] - reference_seq["first_token_logit"]
                        rank_effect = reference_seq["first_token_rank"] - seq["first_token_rank"]
                        decoded_changed = decoded["predicted_answer"] != reference_answer
                    else:
                        sequence_effect = reference_seq["sequence_logprob_sum"] - seq["sequence_logprob_sum"]
                        first_effect = reference_seq["first_token_logit"] - seq["first_token_logit"]
                        rank_effect = seq["first_token_rank"] - reference_seq["first_token_rank"]
                        decoded_changed = decoded["predicted_answer"] != reference_answer
                    out_rows.append(
                        {
                            "asset_id": args.asset_id,
                            "sample_id": sample_id,
                            "prompt_name": prompt_name,
                            "mask_condition": mask_condition,
                            "condition": condition,
                            "direction": direction,
                            "feature_group": feature_group,
                            "top_k": top_k,
                            "feature_ids": "|".join(str(x) for x in feature_ids),
                            "target_answer": answer_text,
                            "answer_token_ids": seq["answer_token_ids"],
                            "answer_token_text": seq["answer_token_text"],
                            "sequence_logprob_sum": seq["sequence_logprob_sum"],
                            "sequence_logprob_mean": seq["sequence_logprob_mean"],
                            "sequence_effect_vs_reference": sequence_effect,
                            "first_token_logit": seq["first_token_logit"],
                            "first_token_rank": seq["first_token_rank"],
                            "first_token_effect_vs_reference": first_effect,
                            "first_token_rank_effect_vs_reference": rank_effect,
                            "generated_text": decoded["generated_text"],
                            "predicted_answer": decoded["predicted_answer"],
                            "target_hit": _target_hit(decoded["generated_text"], answer_text),
                            "baseline_clean_answer": clean_answer,
                            "baseline_mask_answer": mask_answer,
                            "decoded_to_clean": decoded["predicted_answer"] == clean_answer and clean_answer != mask_answer,
                            "decoded_changed_vs_reference": decoded_changed,
                            "clean_mask_sequence_gap": clean_mask_seq_gap,
                            "clean_mask_first_logit_gap": clean_mask_first_logit_gap,
                        }
                    )
        del clean_outputs, mask_outputs, clean_features, mask_features, feature_drops
        torch.cuda.empty_cache()

    payload["gpu_after"] = _gpu_info()
    payload["decision"] = {
        "status": "completed" if out_rows else "blocked_no_rows",
        "rows": len(out_rows),
        "pairs_completed": len({(row["sample_id"], row["prompt_name"], row["mask_condition"]) for row in out_rows}),
        "skipped": payload["skipped"],
    }
    fields = [
        "asset_id",
        "sample_id",
        "prompt_name",
        "mask_condition",
        "condition",
        "direction",
        "feature_group",
        "top_k",
        "feature_ids",
        "target_answer",
        "answer_token_ids",
        "answer_token_text",
        "sequence_logprob_sum",
        "sequence_logprob_mean",
        "sequence_effect_vs_reference",
        "first_token_logit",
        "first_token_rank",
        "first_token_effect_vs_reference",
        "first_token_rank_effect_vs_reference",
        "generated_text",
        "predicted_answer",
        "target_hit",
        "baseline_clean_answer",
        "baseline_mask_answer",
        "decoded_to_clean",
        "decoded_changed_vs_reference",
        "clean_mask_sequence_gap",
        "clean_mask_first_logit_gap",
    ]
    _write_csv(Path(args.out_csv), out_rows, fields)
    _write_json(Path(args.out_json), payload)
    _log(f"done status={payload['decision']['status']} rows={len(out_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
