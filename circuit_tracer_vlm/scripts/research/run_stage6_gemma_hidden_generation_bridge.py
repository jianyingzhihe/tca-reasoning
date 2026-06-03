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

from run_stage6_gemma_hidden_to_plt_decomposition import (
    _apply_mask,
    _first_param_device,
    _gemma_inputs,
    _image_path,
    _mask_path,
    _position_groups,
    _prompt,
    _rank_and_top,
    _replace_hidden,
    _target_candidates,
    _top_feature_patch,
    _top_wrong_token,
)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_int_csv(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


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
    return bool(pred and target and (target in pred or pred in target))


def _answer_token_ids(tokenizer, answer: str) -> list[int]:
    ids = tokenizer.encode((answer or "").strip(), add_special_tokens=False)
    return [int(token_id) for token_id in ids if int(token_id) >= 0]


def _decode_ids(tokenizer, ids: list[int]) -> str:
    if not ids:
        return ""
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def _clone_inputs_with_ids(base_inputs: dict[str, Any], input_ids: torch.Tensor) -> dict[str, Any]:
    out: dict[str, Any] = {}
    base_len = int(base_inputs["input_ids"].shape[1])
    new_len = int(input_ids.shape[1])
    append_len = max(0, new_len - base_len)
    for key, value in base_inputs.items():
        if key == "input_ids":
            out[key] = input_ids
        elif key == "attention_mask":
            out[key] = torch.ones_like(input_ids, device=input_ids.device)
        elif key == "token_type_ids" and append_len > 0:
            pad = torch.zeros((value.shape[0], append_len), device=value.device, dtype=value.dtype)
            out[key] = torch.cat([value, pad], dim=1)
        else:
            out[key] = value
    if "attention_mask" not in out:
        out["attention_mask"] = torch.ones_like(input_ids, device=input_ids.device)
    return out


def _rank_from_logits(row: torch.Tensor, tokenizer, target_id: int, wrong_id: int) -> dict[str, Any]:
    row = row.float()
    target_logit = float(row[int(target_id)].item())
    wrong_logit = float(row[int(wrong_id)].item())
    return {
        "target_logit": target_logit,
        "target_rank": int((row > row[int(target_id)]).sum().item() + 1),
        "wrong_logit": wrong_logit,
        "wrong_rank": int((row > row[int(wrong_id)]).sum().item() + 1),
        "target_minus_wrong_margin": target_logit - wrong_logit,
        "target_token": tokenizer.decode([int(target_id)]),
        "wrong_token": tokenizer.decode([int(wrong_id)]),
    }


def _register_hidden_patch(module, patch_by_position: dict[int, torch.Tensor], scale: float):
    if not patch_by_position:
        return None

    def _hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        patched = hidden.clone()
        for position, delta in patch_by_position.items():
            if 0 <= int(position) < patched.shape[1]:
                patched[:, int(position), :] = patched[:, int(position), :] + scale * delta.to(
                    device=patched.device,
                    dtype=patched.dtype,
                )
        return _replace_hidden(output, patched)

    return module.register_forward_hook(_hook)


def _sequence_score_with_patch(
    *,
    model,
    tokenizer,
    base_inputs: dict[str, Any],
    module,
    patch_by_position: dict[int, torch.Tensor],
    answer_ids: list[int],
    first_target_id: int,
    wrong_token_id: int,
    scale: float,
) -> dict[str, Any]:
    input_ids = base_inputs["input_ids"].clone()
    prompt_len = int(input_ids.shape[1])
    answer_tensor = torch.tensor([answer_ids], device=input_ids.device, dtype=input_ids.dtype)
    extended_ids = torch.cat([input_ids, answer_tensor], dim=1)
    extended_inputs = _clone_inputs_with_ids(base_inputs, extended_ids)
    handle = _register_hidden_patch(module, patch_by_position, scale)
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
    first = _rank_from_logits(logits[prompt_len - 1, :], tokenizer, first_target_id, wrong_token_id)
    return {
        "answer_token_ids": "|".join(str(token_id) for token_id in answer_ids),
        "answer_token_text": tokenizer.decode(answer_ids),
        "sequence_logprob_sum": float(token_logprobs.sum().item()),
        "sequence_logprob_mean": float(token_logprobs.mean().item()),
        "sequence_token_logprobs": "|".join(f"{float(value):.6f}" for value in token_logprobs.detach().cpu().tolist()),
        "first_token_logit": first["target_logit"],
        "first_token_rank": first["target_rank"],
        "first_token_wrong_logit": first["wrong_logit"],
        "first_token_wrong_rank": first["wrong_rank"],
        "first_token_margin": first["target_minus_wrong_margin"],
        "first_token_text": first["target_token"],
        "wrong_token_text": first["wrong_token"],
    }


def _greedy_decode_with_patch(
    *,
    model,
    tokenizer,
    base_inputs: dict[str, Any],
    module,
    patch_by_position: dict[int, torch.Tensor],
    first_target_id: int,
    wrong_token_id: int,
    max_new_tokens: int,
    scale: float,
) -> dict[str, Any]:
    input_ids = base_inputs["input_ids"].clone()
    generated: list[int] = []
    first_step: dict[str, Any] | None = None
    handle = _register_hidden_patch(module, patch_by_position, scale)
    try:
        for _ in range(max_new_tokens):
            step_inputs = _clone_inputs_with_ids(base_inputs, input_ids)
            with torch.inference_mode():
                outputs = model(**step_inputs, output_hidden_states=False, use_cache=False)
            if first_step is None:
                first_step = _rank_from_logits(outputs.logits[0, -1, :], tokenizer, first_target_id, wrong_token_id)
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
        "generated_token_ids": "|".join(str(token_id) for token_id in generated),
        "generated_text": generated_text,
        "predicted_answer": _normalize_answer(generated_text),
        "first_generated_token": _decode_ids(tokenizer, generated[:1]),
        "greedy_first_token_logit": (first_step or {}).get("target_logit", ""),
        "greedy_first_token_rank": (first_step or {}).get("target_rank", ""),
        "greedy_first_token_margin": (first_step or {}).get("target_minus_wrong_margin", ""),
    }


def _bottom_delta_positions(
    clean_hidden: torch.Tensor,
    mask_hidden: torch.Tensor,
    candidates: list[int],
    source_positions: list[int],
    count: int,
) -> list[int]:
    source = set(int(position) for position in source_positions)
    values: list[tuple[float, int]] = []
    for position in candidates:
        if int(position) in source:
            continue
        score = float((clean_hidden[:, int(position), :].float() - mask_hidden[:, int(position), :].float()).norm().item())
        values.append((score, int(position)))
    values.sort()
    return [position for _score, position in values[:count]]


def _negate_patch(patch_by_position: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    return {position: -delta for position, delta in patch_by_position.items()}


def _output_fields() -> list[str]:
    return [
        "asset_id",
        "bridge_operator",
        "sample_id",
        "prompt_name",
        "mask_condition",
        "condition",
        "direction",
        "feature_group",
        "top_k",
        "layer",
        "position_group",
        "position_count",
        "positions",
        "target_answer",
        "target_token_id",
        "target_token",
        "wrong_token_id",
        "wrong_token",
        "answer_token_ids",
        "answer_token_text",
        "sequence_logprob_sum",
        "sequence_logprob_mean",
        "sequence_effect_vs_reference",
        "first_token_logit",
        "first_token_rank",
        "first_token_margin",
        "first_token_effect_vs_reference",
        "first_token_rank_effect_vs_reference",
        "first_token_margin_effect_vs_reference",
        "generated_text",
        "predicted_answer",
        "target_hit",
        "baseline_clean_answer",
        "baseline_mask_answer",
        "decoded_to_clean",
        "decoded_changed_vs_reference",
        "clean_mask_sequence_gap",
        "clean_mask_first_logit_gap",
        "clean_mask_first_margin_gap",
        "selected_features",
    ]


def _row(
    *,
    asset_id: str,
    bridge_operator: str = "hidden_residual",
    source: dict[str, str],
    condition: str,
    direction: str,
    feature_group: str,
    top_k: str,
    layer: int,
    position_group: str,
    positions: list[int],
    target_answer: str,
    target_token_id: int,
    target_token: str,
    wrong_token_id: int,
    wrong_token: str,
    seq: dict[str, Any],
    decoded: dict[str, Any],
    reference_seq: dict[str, Any] | None,
    reference_answer: str,
    clean_answer: str,
    mask_answer: str,
    clean_seq: dict[str, Any],
    mask_seq: dict[str, Any],
    selected_features: str = "",
) -> dict[str, Any]:
    return {
        "asset_id": asset_id,
        "bridge_operator": bridge_operator,
        "sample_id": source.get("sample_id", ""),
        "prompt_name": source.get("prompt_name", ""),
        "mask_condition": source.get("mask_condition", ""),
        "condition": condition,
        "direction": direction,
        "feature_group": feature_group,
        "top_k": top_k,
        "layer": layer,
        "position_group": position_group,
        "position_count": len(positions),
        "positions": "|".join(str(position) for position in positions[:96]),
        "target_answer": target_answer,
        "target_token_id": target_token_id,
        "target_token": target_token,
        "wrong_token_id": wrong_token_id,
        "wrong_token": wrong_token,
        "answer_token_ids": seq["answer_token_ids"],
        "answer_token_text": seq["answer_token_text"],
        "sequence_logprob_sum": seq["sequence_logprob_sum"],
        "sequence_logprob_mean": seq["sequence_logprob_mean"],
        "sequence_effect_vs_reference": "" if reference_seq is None else float(seq["sequence_logprob_sum"]) - float(reference_seq["sequence_logprob_sum"]),
        "first_token_logit": seq["first_token_logit"],
        "first_token_rank": seq["first_token_rank"],
        "first_token_margin": seq["first_token_margin"],
        "first_token_effect_vs_reference": "" if reference_seq is None else float(seq["first_token_logit"]) - float(reference_seq["first_token_logit"]),
        "first_token_rank_effect_vs_reference": "" if reference_seq is None else int(reference_seq["first_token_rank"]) - int(seq["first_token_rank"]),
        "first_token_margin_effect_vs_reference": "" if reference_seq is None else float(seq["first_token_margin"]) - float(reference_seq["first_token_margin"]),
        "generated_text": decoded["generated_text"],
        "predicted_answer": decoded["predicted_answer"],
        "target_hit": _target_hit(decoded["generated_text"], target_answer),
        "baseline_clean_answer": clean_answer,
        "baseline_mask_answer": mask_answer,
        "decoded_to_clean": "" if reference_seq is None else decoded["predicted_answer"] == clean_answer and clean_answer != reference_answer,
        "decoded_changed_vs_reference": "" if reference_seq is None else decoded["predicted_answer"] != reference_answer,
        "clean_mask_sequence_gap": float(clean_seq["sequence_logprob_sum"]) - float(mask_seq["sequence_logprob_sum"]),
        "clean_mask_first_logit_gap": float(clean_seq["first_token_logit"]) - float(mask_seq["first_token_logit"]),
        "clean_mask_first_margin_gap": float(clean_seq["first_token_margin"]) - float(mask_seq["first_token_margin"]),
        "selected_features": selected_features,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage6 Gemma hidden node-to-generation bridge.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--layers", default="1")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--answer-prefix", default="The answer is ")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--bridge-operators", default="hidden_residual,plt_topk_reconstruction,plt_reconstruction_error")
    parser.add_argument("--transcoder-ref", default="tianhux2/gemma3-4b-it-plt")
    parser.add_argument("--topks", default="8,16,32")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    rows = _read_csv(Path(args.manifest))
    if args.max_cases > 0:
        rows = rows[: args.max_cases]
    layers = [int(value) for value in _parse_csv(args.layers)]
    bridge_operators = set(_parse_csv(args.bridge_operators))
    topks = _parse_int_csv(args.topks)
    payload: dict[str, Any] = {
        "created_at": _now(),
        "script": "run_stage6_gemma_hidden_generation_bridge.py",
        "args": vars(args),
        "case_count": len(rows),
        "layers": layers,
        "bridge_operators": sorted(bridge_operators),
        "topks": topks,
        "claim_boundary": (
            "Gemma Stage6-022 bridge tests hidden-residual, PLT topK reconstruction, and PLT reconstruction-error "
            "restore/corrupt at selected positions. It is not graph-level source-route zeroing."
        ),
        "skipped": [],
    }
    _write_json(Path(args.out_json), payload)

    transcoders = None
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model_name, local_files_only=True)
    tokenizer = processor.tokenizer
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()
    device = _first_param_device(model)
    layer_count = len(model.model.language_model.layers)
    layers = [layer for layer in layers if 0 <= layer < layer_count]
    modules = {layer: model.model.language_model.layers[layer] for layer in layers}
    if {"plt_topk_reconstruction", "plt_reconstruction_error"} & bridge_operators:
        from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

        transcoders, config = load_transcoder_from_hub(
            args.transcoder_ref,
            device=device,
            dtype=torch.bfloat16,
            lazy_encoder=True,
            lazy_decoder=True,
        )
        payload["transcoder"] = {"type": type(transcoders).__name__, "config_model_kind": config.get("model_kind", "")}

    out_rows: list[dict[str, Any]] = []
    for source in rows:
        try:
            image = Image.open(_image_path(source, args.image_root)).convert("RGB")
            mask_file = _mask_path(source, args.mask_root, source.get("mask_condition") or "answer_mask")
            if not mask_file.exists():
                payload["skipped"].append({"sample_id": source.get("sample_id"), "reason": "mask_missing", "mask": str(mask_file)})
                continue
            masked_image = _apply_mask(image, Image.open(mask_file).convert("L").resize(image.size), (128, 128, 128))
            question = _prompt(source.get("question_text", ""), source.get("prompt_name", ""))
            clean_inputs = _gemma_inputs(processor, image, question, args.answer_prefix, device)
            mask_inputs = _gemma_inputs(processor, masked_image, question, args.answer_prefix, device)
            target_answer = source.get("target_answer") or source.get("answer_text", "")
            answer_ids = _answer_token_ids(tokenizer, target_answer)
            target_candidates = _target_candidates(tokenizer, target_answer)
            if not answer_ids or not target_candidates:
                payload["skipped"].append({"sample_id": source.get("sample_id"), "reason": "target_tokenization_empty"})
                continue
            with torch.inference_mode():
                clean_outputs = model(**clean_inputs, output_hidden_states=True, use_cache=False)
                mask_outputs = model(**mask_inputs, output_hidden_states=True, use_cache=False)
            clean_score = _rank_and_top(clean_outputs.logits, tokenizer, target_candidates)
            target_token_id = int(clean_score["target_token_id"])
            wrong = _top_wrong_token(clean_outputs.logits, tokenizer, target_token_id)
            wrong_token_id = int(wrong["wrong_token_id"])
            input_ids = clean_inputs["input_ids"][0].detach().cpu().tolist()
            token_texts = tokenizer.convert_ids_to_tokens(input_ids)

            for layer in layers:
                hidden_index = min(layer + 1, len(clean_outputs.hidden_states) - 1)
                clean_hidden = clean_outputs.hidden_states[hidden_index].detach()
                mask_hidden = mask_outputs.hidden_states[hidden_index].detach()
                clean_features = None
                mask_features = None
                if transcoders is not None:
                    clean_features = transcoders.encode_layer(clean_hidden, layer, apply_activation_function=True).detach()
                    mask_features = transcoders.encode_layer(mask_hidden, layer, apply_activation_function=True).detach()
                groups = _position_groups(
                    clean_hidden=clean_hidden,
                    mask_hidden=mask_hidden,
                    clean_inputs=clean_inputs,
                    input_ids=input_ids,
                    token_texts=token_texts,
                    answer_adjacent_count=4,
                    requested_groups=[source.get("position_group") or "top_hidden_delta_32"],
                )
                source_positions = groups.get(source.get("position_group") or "top_hidden_delta_32", [])
                visual_answer = groups.get("visual+answer", [])
                control_positions = _bottom_delta_positions(
                    clean_hidden,
                    mask_hidden,
                    visual_answer,
                    source_positions,
                    len(source_positions),
                )
                if not source_positions or not control_positions:
                    payload["skipped"].append(
                        {
                            "sample_id": source.get("sample_id"),
                            "prompt_name": source.get("prompt_name"),
                            "reason": "position_selection_empty",
                            "source_positions": len(source_positions),
                            "control_positions": len(control_positions),
                        }
                    )
                    continue
                source_restore_patch = {position: clean_hidden[:, position, :] - mask_hidden[:, position, :] for position in source_positions}
                control_restore_patch = {position: clean_hidden[:, position, :] - mask_hidden[:, position, :] for position in control_positions}
                source_corrupt_patch = {position: mask_hidden[:, position, :] - clean_hidden[:, position, :] for position in source_positions}
                control_corrupt_patch = {position: mask_hidden[:, position, :] - clean_hidden[:, position, :] for position in control_positions}
                plt_interventions: list[tuple[str, str, str, str, dict[int, torch.Tensor], list[int], str, int]] = []
                if transcoders is not None and clean_features is not None and mask_features is not None:
                    for top_k in topks:
                        source_recon, source_residual, source_selected = _top_feature_patch(
                            transcoders=transcoders,
                            layer=layer,
                            clean_features=clean_features,
                            mask_features=mask_features,
                            clean_hidden=clean_hidden,
                            mask_hidden=mask_hidden,
                            positions=source_positions,
                            top_k=top_k,
                            device=device,
                            dtype=torch.bfloat16,
                        )
                        control_recon, control_residual, control_selected = _top_feature_patch(
                            transcoders=transcoders,
                            layer=layer,
                            clean_features=clean_features,
                            mask_features=mask_features,
                            clean_hidden=clean_hidden,
                            mask_hidden=mask_hidden,
                            positions=control_positions,
                            top_k=top_k,
                            device=device,
                            dtype=torch.bfloat16,
                        )
                        if "plt_topk_reconstruction" in bridge_operators and source_recon and control_recon:
                            plt_interventions.extend(
                                [
                                    (
                                        "plt_topk_reconstruction",
                                        "plt_topk_restore_source",
                                        "restore",
                                        "source_plt_topk",
                                        source_recon,
                                        source_positions,
                                        source_selected,
                                        top_k,
                                    ),
                                    (
                                        "plt_topk_reconstruction",
                                        "plt_topk_restore_control",
                                        "restore",
                                        "control_plt_topk",
                                        control_recon,
                                        control_positions,
                                        control_selected,
                                        top_k,
                                    ),
                                    (
                                        "plt_topk_reconstruction",
                                        "plt_topk_corrupt_source",
                                        "corrupt",
                                        "source_plt_topk",
                                        _negate_patch(source_recon),
                                        source_positions,
                                        source_selected,
                                        top_k,
                                    ),
                                    (
                                        "plt_topk_reconstruction",
                                        "plt_topk_corrupt_control",
                                        "corrupt",
                                        "control_plt_topk",
                                        _negate_patch(control_recon),
                                        control_positions,
                                        control_selected,
                                        top_k,
                                    ),
                                ]
                            )
                        if "plt_reconstruction_error" in bridge_operators and source_residual and control_residual:
                            plt_interventions.extend(
                                [
                                    (
                                        "plt_reconstruction_error",
                                        "plt_error_restore_source",
                                        "restore",
                                        "source_plt_error",
                                        source_residual,
                                        source_positions,
                                        source_selected,
                                        top_k,
                                    ),
                                    (
                                        "plt_reconstruction_error",
                                        "plt_error_restore_control",
                                        "restore",
                                        "control_plt_error",
                                        control_residual,
                                        control_positions,
                                        control_selected,
                                        top_k,
                                    ),
                                    (
                                        "plt_reconstruction_error",
                                        "plt_error_corrupt_source",
                                        "corrupt",
                                        "source_plt_error",
                                        _negate_patch(source_residual),
                                        source_positions,
                                        source_selected,
                                        top_k,
                                    ),
                                    (
                                        "plt_reconstruction_error",
                                        "plt_error_corrupt_control",
                                        "corrupt",
                                        "control_plt_error",
                                        _negate_patch(control_residual),
                                        control_positions,
                                        control_selected,
                                        top_k,
                                    ),
                                ]
                            )

                clean_seq = _sequence_score_with_patch(
                    model=model,
                    tokenizer=tokenizer,
                    base_inputs=clean_inputs,
                    module=modules[layer],
                    patch_by_position={},
                    answer_ids=answer_ids,
                    first_target_id=target_token_id,
                    wrong_token_id=wrong_token_id,
                    scale=args.scale,
                )
                mask_seq = _sequence_score_with_patch(
                    model=model,
                    tokenizer=tokenizer,
                    base_inputs=mask_inputs,
                    module=modules[layer],
                    patch_by_position={},
                    answer_ids=answer_ids,
                    first_target_id=target_token_id,
                    wrong_token_id=wrong_token_id,
                    scale=args.scale,
                )
                clean_decoded = _greedy_decode_with_patch(
                    model=model,
                    tokenizer=tokenizer,
                    base_inputs=clean_inputs,
                    module=modules[layer],
                    patch_by_position={},
                    first_target_id=target_token_id,
                    wrong_token_id=wrong_token_id,
                    max_new_tokens=args.max_new_tokens,
                    scale=args.scale,
                )
                mask_decoded = _greedy_decode_with_patch(
                    model=model,
                    tokenizer=tokenizer,
                    base_inputs=mask_inputs,
                    module=modules[layer],
                    patch_by_position={},
                    first_target_id=target_token_id,
                    wrong_token_id=wrong_token_id,
                    max_new_tokens=args.max_new_tokens,
                    scale=args.scale,
                )
                clean_answer = clean_decoded["predicted_answer"]
                mask_answer = mask_decoded["predicted_answer"]
                out_rows.append(
                    _row(
                        asset_id="gemma3_hidden",
                        source=source,
                        condition="baseline_clean",
                        direction="baseline",
                        feature_group="",
                        top_k="",
                        layer=layer,
                        position_group=source.get("position_group") or "top_hidden_delta_32",
                        positions=[],
                        target_answer=target_answer,
                        target_token_id=target_token_id,
                        target_token=clean_score["target_token"],
                        wrong_token_id=wrong_token_id,
                        wrong_token=wrong["wrong_token"],
                        seq=clean_seq,
                        decoded=clean_decoded,
                        reference_seq=None,
                        reference_answer="",
                        clean_answer=clean_answer,
                        mask_answer=mask_answer,
                        clean_seq=clean_seq,
                        mask_seq=mask_seq,
                    )
                )
                out_rows.append(
                    _row(
                        asset_id="gemma3_hidden",
                        source=source,
                        condition="baseline_mask",
                        direction="baseline",
                        feature_group="",
                        top_k="",
                        layer=layer,
                        position_group=source.get("position_group") or "top_hidden_delta_32",
                        positions=[],
                        target_answer=target_answer,
                        target_token_id=target_token_id,
                        target_token=clean_score["target_token"],
                        wrong_token_id=wrong_token_id,
                        wrong_token=wrong["wrong_token"],
                        seq=mask_seq,
                        decoded=mask_decoded,
                        reference_seq=None,
                        reference_answer="",
                        clean_answer=clean_answer,
                        mask_answer=mask_answer,
                        clean_seq=clean_seq,
                        mask_seq=mask_seq,
                    )
                )
                interventions = []
                if "hidden_residual" in bridge_operators:
                    interventions.extend(
                        [
                            (
                                "hidden_residual",
                                "hidden_restore_source",
                                "restore",
                                "source_hidden",
                                mask_inputs,
                                mask_seq,
                                mask_answer,
                                source_restore_patch,
                                source_positions,
                                "",
                                len(source_positions),
                                "gemma3_hidden",
                            ),
                            (
                                "hidden_residual",
                                "hidden_restore_control",
                                "restore",
                                "control_hidden",
                                mask_inputs,
                                mask_seq,
                                mask_answer,
                                control_restore_patch,
                                control_positions,
                                "",
                                len(control_positions),
                                "gemma3_hidden",
                            ),
                            (
                                "hidden_residual",
                                "hidden_corrupt_source",
                                "corrupt",
                                "source_hidden",
                                clean_inputs,
                                clean_seq,
                                clean_answer,
                                source_corrupt_patch,
                                source_positions,
                                "",
                                len(source_positions),
                                "gemma3_hidden",
                            ),
                            (
                                "hidden_residual",
                                "hidden_corrupt_control",
                                "corrupt",
                                "control_hidden",
                                clean_inputs,
                                clean_seq,
                                clean_answer,
                                control_corrupt_patch,
                                control_positions,
                                "",
                                len(control_positions),
                                "gemma3_hidden",
                            ),
                        ]
                    )
                for operator, condition, direction, group, patch, positions, selected, top_k in plt_interventions:
                    base_inputs = mask_inputs if direction == "restore" else clean_inputs
                    reference_seq = mask_seq if direction == "restore" else clean_seq
                    reference_answer = mask_answer if direction == "restore" else clean_answer
                    interventions.append(
                        (
                            operator,
                            condition,
                            direction,
                            group,
                            base_inputs,
                            reference_seq,
                            reference_answer,
                            patch,
                            positions,
                            selected,
                            top_k,
                            "gemma3_plt",
                        )
                    )
                for (
                    bridge_operator,
                    condition,
                    direction,
                    group,
                    base_inputs,
                    reference_seq,
                    reference_answer,
                    patch,
                    positions,
                    selected_features,
                    top_k,
                    asset_id,
                ) in interventions:
                    seq = _sequence_score_with_patch(
                        model=model,
                        tokenizer=tokenizer,
                        base_inputs=base_inputs,
                        module=modules[layer],
                        patch_by_position=patch,
                        answer_ids=answer_ids,
                        first_target_id=target_token_id,
                        wrong_token_id=wrong_token_id,
                        scale=args.scale,
                    )
                    decoded = _greedy_decode_with_patch(
                        model=model,
                        tokenizer=tokenizer,
                        base_inputs=base_inputs,
                        module=modules[layer],
                        patch_by_position=patch,
                        first_target_id=target_token_id,
                        wrong_token_id=wrong_token_id,
                        max_new_tokens=args.max_new_tokens,
                        scale=args.scale,
                    )
                    out_rows.append(
                        _row(
                            asset_id=asset_id,
                            bridge_operator=bridge_operator,
                            source=source,
                            condition=condition,
                            direction=direction,
                            feature_group=group,
                            top_k=str(top_k),
                            layer=layer,
                            position_group=source.get("position_group") or "top_hidden_delta_32",
                            positions=positions,
                            target_answer=target_answer,
                            target_token_id=target_token_id,
                            target_token=clean_score["target_token"],
                            wrong_token_id=wrong_token_id,
                            wrong_token=wrong["wrong_token"],
                            seq=seq,
                            decoded=decoded,
                            reference_seq=reference_seq,
                            reference_answer=reference_answer,
                            clean_answer=clean_answer,
                            mask_answer=mask_answer,
                            clean_seq=clean_seq,
                            mask_seq=mask_seq,
                            selected_features=selected_features,
                        )
                    )
            del clean_outputs, mask_outputs
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            payload["skipped"].append({"sample_id": source.get("sample_id"), "prompt_name": source.get("prompt_name"), "reason": repr(exc)})
            torch.cuda.empty_cache()

    _write_csv(Path(args.out_csv), out_rows, _output_fields())
    payload.update({"finished_at": _now(), "status": "ok" if out_rows else "empty", "rows": len(out_rows)})
    _write_json(Path(args.out_json), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
