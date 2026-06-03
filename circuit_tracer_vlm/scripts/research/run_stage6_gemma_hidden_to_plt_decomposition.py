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


MASK_FILE_NAMES = {
    "answer_mask": "answer.png",
    "union_mask": "union.png",
    "shifted_mask": "shifted.png",
    "shuffled_mask": "shuffled.png",
}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage6-gemma-hidden-to-plt] {message}", flush=True)


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


def _image_path(row: dict[str, str], image_root: str) -> Path:
    if image_root:
        return Path(image_root) / Path(row["image_filename"]).name
    return Path(row.get("local_image_path") or row.get("image_path") or row.get("image_filename"))


def _mask_path(row: dict[str, str], mask_root: str, condition: str) -> Path:
    stem = Path(row["image_filename"]).stem
    if mask_root:
        return Path(mask_root) / stem / MASK_FILE_NAMES[condition]
    key = {
        "answer_mask": "answer_mask_path",
        "union_mask": "union_mask_path",
        "shifted_mask": "shifted_mask_path",
        "shuffled_mask": "shuffled_mask_path",
    }[condition]
    if row.get(key):
        return Path(row[key])
    return Path(row.get("mask_dir", "")) / MASK_FILE_NAMES[condition]


def _apply_mask(image: Image.Image, mask: Image.Image, fill_rgb: tuple[int, int, int]) -> Image.Image:
    base = image.convert("RGB")
    fill = Image.new("RGB", base.size, fill_rgb)
    return Image.composite(fill, base, mask.convert("L"))


def _prompt(question: str, prompt_name: str) -> str:
    if prompt_name == "D_visual_only":
        return f"{question} Use visual evidence, then reply with only one short sentence in exactly this format: The answer is <short answer>."
    if prompt_name == "B_direct":
        return f"{question} Reply with only one short sentence in exactly this format: The answer is <short answer>."
    if prompt_name == "C_step_only":
        return f"{question} Think step by step internally, then reply with only one short sentence in exactly this format: The answer is <short answer>."
    if prompt_name == "A_step_visual":
        return f"{question} Think step by step from visual evidence internally, then reply with only one short sentence in exactly this format: The answer is <short answer>."
    return f"{question} Reply with only one short sentence in exactly this format: The answer is <short answer>."


def _gemma_inputs(processor, image: Image.Image, question: str, answer_prefix: str, device: torch.device) -> dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + answer_prefix
        inputs = processor(text=[text], images=[image], return_tensors="pt")
    except Exception:
        text = f"<start_of_image> {question}{answer_prefix}"
        inputs = processor(text=text, images=image, return_tensors="pt")
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _target_candidates(tokenizer, answer: str) -> list[int]:
    variants = [answer, " " + answer, answer.capitalize(), " " + answer.capitalize()]
    out: list[int] = []
    seen: set[int] = set()
    for variant in variants:
        ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        token_id = int(ids[0])
        if token_id not in seen:
            out.append(token_id)
            seen.add(token_id)
    return out


def _rank_and_top(logits: torch.Tensor, tokenizer, candidate_ids: list[int]) -> dict[str, Any]:
    row = logits[0, -1].float()
    best = None
    for token_id in candidate_ids:
        token_logit = float(row[int(token_id)].item())
        rank = int((row > row[int(token_id)]).sum().item() + 1)
        item = {
            "target_token_id": int(token_id),
            "target_token": tokenizer.decode([int(token_id)]),
            "target_logit": token_logit,
            "target_rank": rank,
        }
        if best is None or rank < best["target_rank"]:
            best = item
    assert best is not None
    return best


def _top_wrong_token(logits: torch.Tensor, tokenizer, target_token_id: int) -> dict[str, Any]:
    row = logits[0, -1].float()
    scores = row.detach().clone()
    scores[int(target_token_id)] = -float("inf")
    token_id = int(torch.argmax(scores).item())
    return {
        "wrong_token_id": token_id,
        "wrong_token": tokenizer.decode([token_id]),
        "wrong_logit": float(row[token_id].item()),
        "wrong_rank": int((row > row[token_id]).sum().item() + 1),
    }


def _score_token_map(logits: torch.Tensor, tokenizer, token_ids: list[int]) -> dict[int, dict[str, Any]]:
    return {int(token_id): _rank_and_top(logits, tokenizer, [int(token_id)]) for token_id in token_ids}


def _replace_hidden(output: Any, new_hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (new_hidden, *output[1:])
    return new_hidden


def _dedupe(values: list[int]) -> list[int]:
    return sorted({int(value) for value in values if int(value) >= 0})


def _answer_adjacent_positions(seq_len: int, visual_positions: set[int], count: int) -> list[int]:
    out = []
    for position in range(seq_len - 1, -1, -1):
        if position not in visual_positions:
            out.append(position)
        if len(out) >= count:
            break
    return sorted(out)


def _gemma_bucket_positions(inputs: dict[str, Any], input_ids: list[int], token_texts: list[str]) -> list[int]:
    token_type_ids = inputs.get("token_type_ids")
    if token_type_ids is not None:
        ids = token_type_ids[0].detach().cpu().tolist()
        visual = [idx for idx, value in enumerate(ids) if int(value) == 1]
        if visual:
            return _dedupe(visual)
    image_pos = [
        idx
        for idx, token in enumerate(token_texts)
        if "image" in token.lower() or "vision" in token.lower() or token in {"<image>", "<start_of_image>"}
    ]
    if image_pos:
        return list(range(min(image_pos), max(image_pos) + 1))
    return []


def _top_delta_positions(clean_hidden: torch.Tensor, mask_hidden: torch.Tensor, positions: list[int], count: int) -> list[int]:
    if not positions:
        return []
    pos = torch.tensor(positions, device=clean_hidden.device, dtype=torch.long)
    scores = (clean_hidden[:, pos, :].float() - mask_hidden[:, pos, :].float()).norm(dim=-1)[0]
    pairs = [(float(score), int(position)) for score, position in zip(scores.detach().cpu().tolist(), positions, strict=False)]
    pairs.sort(reverse=True)
    return _dedupe([position for _score, position in pairs[:count]])


def _position_groups(
    *,
    clean_hidden: torch.Tensor,
    mask_hidden: torch.Tensor,
    clean_inputs: dict[str, Any],
    input_ids: list[int],
    token_texts: list[str],
    answer_adjacent_count: int,
    requested_groups: list[str],
) -> dict[str, list[int]]:
    visual = _gemma_bucket_positions(clean_inputs, input_ids, token_texts)
    answer = _answer_adjacent_positions(len(input_ids), set(visual), answer_adjacent_count)
    visual_answer = _dedupe(visual + answer)
    groups = {
        "visual_span": _dedupe(visual),
        "answer_adjacent": _dedupe(answer),
        "visual+answer": visual_answer,
    }
    for group in requested_groups:
        match = re.fullmatch(r"top_hidden_delta_(\d+)", group)
        if match:
            groups[group] = _top_delta_positions(clean_hidden, mask_hidden, visual_answer, int(match.group(1)))
        elif group == "top_hidden_delta":
            groups[group] = _top_delta_positions(clean_hidden, mask_hidden, visual_answer, 32)
    return groups


def _score_with_patch(
    *,
    model,
    module,
    inputs: dict[str, Any],
    tokenizer,
    token_ids: list[int],
    patch_by_position: dict[int, torch.Tensor],
    scale: float,
) -> dict[int, dict[str, Any]]:
    def _hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        patched = hidden.clone()
        for position, delta in patch_by_position.items():
            if 0 <= position < patched.shape[1]:
                patched[:, position, :] = patched[:, position, :] + scale * delta.to(device=patched.device, dtype=patched.dtype)
        return _replace_hidden(output, patched)

    handle = module.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=False, use_cache=False)
    finally:
        handle.remove()
    return {int(token_id): _rank_and_top(outputs.logits, tokenizer, [int(token_id)]) for token_id in token_ids}


def _decoder_vectors(transcoders, layer: int, feature_ids: list[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ids = torch.tensor(feature_ids, device=device, dtype=torch.long)
    vectors = transcoders._get_decoder_vectors(layer, ids)
    if vectors.ndim == 3:
        vectors = vectors[:, 0, :]
    return vectors.to(device=device, dtype=dtype)


def _top_feature_patch(
    *,
    transcoders,
    layer: int,
    clean_features: torch.Tensor,
    mask_features: torch.Tensor,
    clean_hidden: torch.Tensor,
    mask_hidden: torch.Tensor,
    positions: list[int],
    top_k: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], str]:
    candidates: list[tuple[float, int, int, float]] = []
    for position in positions:
        if position < 0 or position >= clean_features.shape[1]:
            continue
        delta = (clean_features[0, position, :].float() - mask_features[0, position, :].float()).detach()
        active = clean_features[0, position, :].float() > 0
        delta[~active] = 0
        vals, ids = torch.topk(delta.abs(), k=min(top_k, delta.numel()))
        for value, feature_id in zip(vals.cpu().tolist(), ids.cpu().tolist(), strict=False):
            if value <= 0:
                continue
            signed = float(delta[int(feature_id)].item())
            candidates.append((float(value), int(position), int(feature_id), signed))
    candidates.sort(reverse=True)
    selected = candidates[:top_k]
    if not selected:
        return {}, {}, ""
    feature_ids = [feature_id for _value, _position, feature_id, _signed in selected]
    vectors = _decoder_vectors(transcoders, layer, feature_ids, device, dtype)
    recon: dict[int, torch.Tensor] = {}
    for idx, (_value, position, feature_id, signed_delta) in enumerate(selected):
        recon[position] = recon.get(position, torch.zeros_like(clean_hidden[:, position, :])) + signed_delta * vectors[idx].view(1, -1)
    residual: dict[int, torch.Tensor] = {}
    for position in positions:
        full_delta = clean_hidden[:, position, :].to(device=device, dtype=dtype) - mask_hidden[:, position, :].to(device=device, dtype=dtype)
        residual[position] = full_delta - recon.get(position, torch.zeros_like(full_delta)).to(device=device, dtype=dtype)
    selected_text = "|".join(f"P{position}:F{feature_id}" for _value, position, feature_id, _signed in selected)
    return recon, residual, selected_text


def _output_fields() -> list[str]:
    return [
        "model_family",
        "sample_id",
        "prompt_name",
        "reasoning_operation",
        "image_dependence_tier",
        "paperpack_source",
        "layer",
        "hidden_state_index",
        "mask_condition",
        "operator",
        "top_k",
        "position_group",
        "position_count",
        "positions",
        "target_answer",
        "token_scored",
        "scored_token_id",
        "scored_token",
        "target_token_id",
        "target_token",
        "wrong_token_id",
        "wrong_token",
        "before_logit",
        "after_logit",
        "logit_effect",
        "before_rank",
        "after_rank",
        "rank_effect",
        "clean_target_logit",
        "mask_target_logit",
        "clean_target_rank",
        "mask_target_rank",
        "selected_features",
    ]


def _first_param_device(model) -> torch.device:
    return next(model.parameters()).device


def _gpu_info() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage6-016 Gemma hidden-to-PLT decomposition.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--transcoder-ref", default="tianhux2/gemma3-4b-it-plt")
    parser.add_argument("--prompt-runs", required=True)
    parser.add_argument("--image-root", default="")
    parser.add_argument("--mask-root", default="")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--layers", default="1")
    parser.add_argument("--mask-conditions", default="answer_mask,union_mask,shifted_mask,shuffled_mask")
    parser.add_argument("--position-groups", default="visual+answer,top_hidden_delta_16,top_hidden_delta_32")
    parser.add_argument("--top-ks", default="8,16,32,64,128")
    parser.add_argument("--answer-prefix", default="The answer is ")
    parser.add_argument("--answer-adjacent-count", type=int, default=4)
    parser.add_argument("--max-prompt-runs", type=int, default=0)
    parser.add_argument("--max-clean-rank", type=int, default=10)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    args = parser.parse_args()

    prompt_rows = _read_csv(Path(args.prompt_runs))
    if args.max_prompt_runs > 0:
        prompt_rows = prompt_rows[: args.max_prompt_runs]
    layers = _parse_int_csv(args.layers)
    mask_conditions = _parse_csv(args.mask_conditions)
    position_group_names = _parse_csv(args.position_groups)
    top_ks = _parse_int_csv(args.top_ks)
    fields = _output_fields()
    out_path = Path(args.out_csv)
    summary_path = Path(args.summary_json)

    payload: dict[str, Any] = {
        "created_at": _now(),
        "script": "run_stage6_gemma_hidden_to_plt_decomposition.py",
        "args": vars(args),
        "requested_prompt_runs": len(prompt_rows),
        "layers": layers,
        "mask_conditions": mask_conditions,
        "position_groups": position_group_names,
        "top_ks": top_ks,
        "gpu_before": _gpu_info(),
        "skipped": [],
    }
    _write_json(summary_path, payload)

    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
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
    dtype = torch.bfloat16
    transcoders, config = load_transcoder_from_hub(args.transcoder_ref, device=device, dtype=dtype, lazy_encoder=True, lazy_decoder=True)
    layer_count = len(model.model.language_model.layers)
    layers = [layer for layer in layers if 0 <= layer < layer_count]
    modules = {layer: model.model.language_model.layers[layer] for layer in layers}
    payload["transcoder"] = {"type": type(transcoders).__name__, "config_model_kind": config.get("model_kind", "")}
    payload["available_layer_count"] = layer_count
    payload["layers"] = layers
    _write_json(summary_path, payload)
    if not layers:
        payload.update({"status": "blocked_no_valid_layers", "rows": 0, "finished_at": _now()})
        _write_json(summary_path, payload)
        return 2

    rows: list[dict[str, Any]] = []
    processed_keys: set[tuple[str, str]] = set()
    if args.resume and out_path.exists() and out_path.stat().st_size > 0:
        existing = _read_csv(out_path)
        rows.extend(existing)
        processed_keys = {(row.get("sample_id", ""), row.get("prompt_name", "")) for row in existing}
        _log(f"resume loaded rows={len(existing)} processed_prompt_runs={len(processed_keys)}")
    usable_runs = len(processed_keys)

    for idx, row in enumerate(prompt_rows, start=1):
        run_key = (row.get("sample_id", ""), row.get("prompt_name", ""))
        if run_key in processed_keys:
            continue
        try:
            image_path = _image_path(row, args.image_root)
            image = Image.open(image_path).convert("RGB")
            question = _prompt(row.get("question_text", ""), row.get("prompt_name", ""))
            clean_inputs = _gemma_inputs(processor, image, question, args.answer_prefix, device)
            target_ids = _target_candidates(tokenizer, row.get("answer_text", ""))
            if not target_ids:
                payload["skipped"].append({"sample_id": row.get("sample_id"), "prompt_name": row.get("prompt_name"), "reason": "no_target_ids"})
                continue
            with torch.inference_mode():
                clean_outputs = model(**clean_inputs, output_hidden_states=True, use_cache=False)
            clean_score = _rank_and_top(clean_outputs.logits, tokenizer, target_ids)
            target_token_id = int(clean_score["target_token_id"])
            if int(clean_score["target_rank"]) > args.max_clean_rank:
                payload["skipped"].append(
                    {
                        "sample_id": row.get("sample_id"),
                        "prompt_name": row.get("prompt_name"),
                        "reason": "clean_rank_too_weak",
                        "clean_rank": clean_score["target_rank"],
                    }
                )
                del clean_outputs
                torch.cuda.empty_cache()
                continue
            wrong = _top_wrong_token(clean_outputs.logits, tokenizer, target_token_id)
            wrong_token_id = int(wrong["wrong_token_id"])
            clean_scores = _score_token_map(clean_outputs.logits, tokenizer, [target_token_id, wrong_token_id])
            input_ids = clean_inputs["input_ids"][0].detach().cpu().tolist()
            token_texts = tokenizer.convert_ids_to_tokens(input_ids)

            for mask_condition in mask_conditions:
                mask_file = _mask_path(row, args.mask_root, mask_condition)
                if not mask_file.exists():
                    payload["skipped"].append({"sample_id": row.get("sample_id"), "mask_condition": mask_condition, "reason": "mask_missing"})
                    continue
                mask = Image.open(mask_file).convert("L").resize(image.size)
                masked_image = _apply_mask(image, mask, (128, 128, 128))
                mask_inputs = _gemma_inputs(processor, masked_image, question, args.answer_prefix, device)
                with torch.inference_mode():
                    mask_outputs = model(**mask_inputs, output_hidden_states=True, use_cache=False)
                mask_scores = _score_token_map(mask_outputs.logits, tokenizer, [target_token_id, wrong_token_id])

                for layer in layers:
                    hidden_index = min(layer + 1, len(clean_outputs.hidden_states) - 1)
                    clean_hidden = clean_outputs.hidden_states[hidden_index].detach().to(device=device, dtype=dtype)
                    mask_hidden = mask_outputs.hidden_states[hidden_index].detach().to(device=device, dtype=dtype)
                    groups = _position_groups(
                        clean_hidden=clean_hidden,
                        mask_hidden=mask_hidden,
                        clean_inputs=clean_inputs,
                        input_ids=input_ids,
                        token_texts=token_texts,
                        answer_adjacent_count=args.answer_adjacent_count,
                        requested_groups=position_group_names,
                    )
                    clean_features = transcoders.encode_layer(clean_hidden, layer, apply_activation_function=True).detach()
                    mask_features = transcoders.encode_layer(mask_hidden, layer, apply_activation_function=True).detach()

                    for position_group in position_group_names:
                        positions = groups.get(position_group, [])
                        if not positions:
                            payload["skipped"].append(
                                {"sample_id": row.get("sample_id"), "prompt_name": row.get("prompt_name"), "mask_condition": mask_condition, "position_group": position_group, "reason": "no_positions"}
                            )
                            continue
                        operators: list[tuple[str, str, dict[int, torch.Tensor], str]] = [
                            ("hidden_residual", "full", {pos: clean_hidden[:, pos, :] - mask_hidden[:, pos, :] for pos in positions}, ""),
                        ]
                        for top_k in top_ks:
                            recon, residual, selected = _top_feature_patch(
                                transcoders=transcoders,
                                layer=layer,
                                clean_features=clean_features,
                                mask_features=mask_features,
                                clean_hidden=clean_hidden,
                                mask_hidden=mask_hidden,
                                positions=positions,
                                top_k=top_k,
                                device=device,
                                dtype=dtype,
                            )
                            if recon:
                                operators.append(("plt_topk_reconstruction", str(top_k), recon, selected))
                                operators.append(("plt_reconstruction_error", str(top_k), residual, selected))
                        for operator, top_k, patch, selected in operators:
                            restored = _score_with_patch(
                                model=model,
                                module=modules[layer],
                                inputs=mask_inputs,
                                tokenizer=tokenizer,
                                token_ids=[target_token_id, wrong_token_id],
                                patch_by_position=patch,
                                scale=args.scale,
                            )
                            for token_label, token_id in [("target", target_token_id), ("wrong", wrong_token_id)]:
                                before = mask_scores[token_id]
                                after = restored[token_id]
                                logit_effect = float(after["target_logit"]) - float(before["target_logit"])
                                rank_effect = int(before["target_rank"]) - int(after["target_rank"])
                                rows.append(
                                    {
                                        "model_family": "gemma3",
                                        "sample_id": row.get("sample_id", ""),
                                        "prompt_name": row.get("prompt_name", ""),
                                        "reasoning_operation": row.get("reasoning_operation", ""),
                                        "image_dependence_tier": row.get("image_dependence_tier", ""),
                                        "paperpack_source": row.get("paperpack_source", ""),
                                        "layer": layer,
                                        "hidden_state_index": hidden_index,
                                        "mask_condition": mask_condition,
                                        "operator": operator,
                                        "top_k": top_k,
                                        "position_group": position_group,
                                        "position_count": len(positions),
                                        "positions": "|".join(str(position) for position in positions[:96]),
                                        "target_answer": row.get("answer_text", ""),
                                        "token_scored": token_label,
                                        "scored_token_id": token_id,
                                        "scored_token": before["target_token"],
                                        "target_token_id": target_token_id,
                                        "target_token": clean_score["target_token"],
                                        "wrong_token_id": wrong_token_id,
                                        "wrong_token": wrong["wrong_token"],
                                        "before_logit": before["target_logit"],
                                        "after_logit": after["target_logit"],
                                        "logit_effect": logit_effect,
                                        "before_rank": before["target_rank"],
                                        "after_rank": after["target_rank"],
                                        "rank_effect": rank_effect,
                                        "clean_target_logit": clean_scores[target_token_id]["target_logit"],
                                        "mask_target_logit": mask_scores[target_token_id]["target_logit"],
                                        "clean_target_rank": clean_scores[target_token_id]["target_rank"],
                                        "mask_target_rank": mask_scores[target_token_id]["target_rank"],
                                        "selected_features": selected,
                                    }
                                )
                    del clean_features, mask_features
                del mask_outputs
                torch.cuda.empty_cache()
            usable_runs += 1
            if idx % 5 == 0:
                _log(f"prompt-runs processed: {idx}/{len(prompt_rows)} usable={usable_runs} rows={len(rows)}")
            if args.checkpoint_every > 0 and usable_runs % args.checkpoint_every == 0:
                _write_csv(out_path, rows, fields)
                payload.update({"status": "running_checkpoint", "usable_prompt_runs": usable_runs, "rows": len(rows), "checkpoint_at": _now()})
                _write_json(summary_path, payload)
            del clean_outputs
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            payload["skipped"].append({"sample_id": row.get("sample_id"), "prompt_name": row.get("prompt_name"), "reason": repr(exc)})
            torch.cuda.empty_cache()

    _write_csv(out_path, rows, fields)
    payload.update(
        {
            "finished_at": _now(),
            "status": "ok" if rows else "empty",
            "usable_prompt_runs": usable_runs,
            "rows": len(rows),
            "gpu_after": _gpu_info(),
        }
    )
    _write_json(summary_path, payload)
    _log(f"wrote {len(rows)} rows to {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
