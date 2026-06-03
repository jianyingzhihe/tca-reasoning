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


MASK_FILE_NAMES = {
    "answer_mask": "answer.png",
    "union_mask": "union.png",
    "shifted_mask": "shifted.png",
    "shuffled_mask": "shuffled.png",
}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage6-gemma-hidden-lattice] {message}", flush=True)


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


def _parse_float_csv(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


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
    if row.get("mask_dir"):
        return Path(row["mask_dir"]) / MASK_FILE_NAMES[condition]
    return Path("")


def _apply_mask(image: Image.Image, mask: Image.Image, fill_rgb: tuple[int, int, int]) -> Image.Image:
    base = image.convert("RGB")
    fill = Image.new("RGB", base.size, fill_rgb)
    return Image.composite(fill, base, mask.convert("L"))


def _prompt(question: str, prompt_name: str) -> str:
    if prompt_name == "D_visual_only":
        return (
            f"{question} Use visual evidence, then reply with only one short sentence "
            "in exactly this format: The answer is <short answer>."
        )
    if prompt_name == "B_direct":
        return (
            f"{question} Reply with only one short sentence in exactly this format: "
            "The answer is <short answer>."
        )
    if prompt_name == "C_step_only":
        return (
            f"{question} Think step by step internally, then reply with only one short sentence "
            "in exactly this format: The answer is <short answer>."
        )
    if prompt_name == "A_step_visual":
        return (
            f"{question} Think step by step from visual evidence internally, then reply with only one short sentence "
            "in exactly this format: The answer is <short answer>."
        )
    raise ValueError(f"Unknown prompt_name: {prompt_name}")


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


def _target_candidates(tokenizer, answer: str) -> list[dict[str, Any]]:
    variants = [answer, " " + answer, answer.capitalize(), " " + answer.capitalize()]
    out = []
    seen = set()
    for variant in variants:
        ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        token_id = int(ids[0])
        if token_id in seen:
            continue
        seen.add(token_id)
        try:
            token_text = tokenizer.decode([token_id])
        except Exception:
            token_text = str(token_id)
        out.append({"variant": variant, "token_id": token_id, "token_text": token_text})
    return out


def _rank_and_top(logits: torch.Tensor, tokenizer, candidate_ids: list[int]) -> dict[str, Any]:
    row = logits[0, -1].float()
    probs = torch.softmax(row, dim=-1)
    best = None
    for token_id in candidate_ids:
        token_logit = float(row[token_id].item())
        rank = int((row > row[token_id]).sum().item() + 1)
        item = {
            "target_token_id": int(token_id),
            "target_token": tokenizer.decode([int(token_id)]),
            "target_logit": token_logit,
            "target_prob": float(probs[token_id].item()),
            "target_rank": rank,
        }
        if best is None or rank < best["target_rank"]:
            best = item
    top_vals, top_ids = torch.topk(row, k=5)
    assert best is not None
    best["top1_token_id"] = int(top_ids[0].item())
    best["top1_token"] = tokenizer.decode([int(top_ids[0].item())])
    best["top1_logit"] = float(top_vals[0].item())
    best["top5"] = [
        {"token_id": int(token_id), "token": tokenizer.decode([int(token_id)]), "logit": float(value)}
        for token_id, value in zip(top_ids.cpu().tolist(), top_vals.float().cpu().tolist(), strict=False)
    ]
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


def _gemma_bucket_positions(inputs: dict[str, Any], input_ids: list[int], token_texts: list[str]) -> dict[str, list[int]]:
    token_type_ids = inputs.get("token_type_ids")
    if token_type_ids is not None:
        ids = token_type_ids[0].detach().cpu().tolist()
        visual = [idx for idx, value in enumerate(ids) if int(value) == 1]
        if visual:
            return {"image_marker_or_span": _dedupe(visual)}
    image_pos = [
        idx
        for idx, token in enumerate(token_texts)
        if "image" in token.lower() or "vision" in token.lower() or token in {"<image>", "<start_of_image>"}
    ]
    if image_pos:
        return {"image_marker_or_span": list(range(min(image_pos), max(image_pos) + 1))}
    image_token_id = getattr(getattr(inputs, "config", None), "image_token_id", None)
    if image_token_id is not None:
        visual = [idx for idx, token_id in enumerate(input_ids) if int(token_id) == int(image_token_id)]
        if visual:
            return {"image_marker_or_span": _dedupe(visual)}
    return {"image_marker_or_span": []}


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
    top_delta_count: int,
) -> dict[str, list[int]]:
    visual = _gemma_bucket_positions(clean_inputs, input_ids, token_texts)["image_marker_or_span"]
    answer = _answer_adjacent_positions(len(input_ids), set(visual), answer_adjacent_count)
    visual_answer = _dedupe(visual + answer)
    return {
        "visual_span": _dedupe(visual),
        "answer_adjacent": _dedupe(answer),
        "visual+answer": visual_answer,
        "top_hidden_delta": _top_delta_positions(clean_hidden, mask_hidden, visual_answer, top_delta_count),
    }


def _patch_hidden(
    hidden: torch.Tensor,
    *,
    positions: list[int],
    source_hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if not positions:
        return hidden
    pos = torch.tensor(positions, device=hidden.device, dtype=torch.long)
    source = source_hidden[:, pos, :].to(device=hidden.device, dtype=hidden.dtype)
    target = target_hidden[:, pos, :].to(device=hidden.device, dtype=hidden.dtype)
    out = hidden.clone()
    out[:, pos, :] = out[:, pos, :] + scale * (source - target)
    return out


def _score_with_patch(
    *,
    model,
    module,
    inputs: dict[str, Any],
    tokenizer,
    token_ids: list[int],
    positions: list[int],
    source_hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    scale: float,
) -> dict[int, dict[str, Any]]:
    def _hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        patched = _patch_hidden(
            hidden,
            positions=positions,
            source_hidden=source_hidden,
            target_hidden=target_hidden,
            scale=scale,
        )
        return _replace_hidden(output, patched)

    handle = module.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=False, use_cache=False)
    finally:
        handle.remove()
    return {int(token_id): _rank_and_top(outputs.logits, tokenizer, [int(token_id)]) for token_id in token_ids}


def _row_base(
    row: dict[str, str],
    *,
    layer: int,
    hidden_state_index: int,
    mask_condition: str,
    direction: str,
    position_group: str,
    positions: list[int],
    scale: float,
) -> dict[str, Any]:
    return {
        "model_family": "gemma3",
        "sample_id": row.get("sample_id", ""),
        "prompt_name": row.get("prompt_name", ""),
        "reasoning_operation": row.get("reasoning_operation", ""),
        "image_dependence_tier": row.get("image_dependence_tier", ""),
        "paperpack_source": row.get("paperpack_source", ""),
        "layer": layer,
        "hidden_state_index": hidden_state_index,
        "mask_condition": mask_condition,
        "direction": direction,
        "position_group": position_group,
        "scale": scale,
        "position_count": len(positions),
        "positions": "|".join(str(position) for position in positions[:64]),
        "target_answer": row.get("answer_text", ""),
    }


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
        "direction",
        "position_group",
        "scale",
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
        "clean_wrong_logit",
        "mask_wrong_logit",
        "effect_sign",
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


def _env_presence() -> dict[str, bool]:
    import os

    return {
        "HF_HOME": bool(os.environ.get("HF_HOME")),
        "HUGGINGFACE_HUB_CACHE": bool(os.environ.get("HUGGINGFACE_HUB_CACHE")),
        "CUDA_VISIBLE_DEVICES": bool(os.environ.get("CUDA_VISIBLE_DEVICES")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage6-014 Gemma hidden causal lattice sweep.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--prompt-runs", required=True)
    parser.add_argument("--image-root", default="")
    parser.add_argument("--mask-root", default="")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--mask-conditions", default="answer_mask,union_mask,shifted_mask,shuffled_mask")
    parser.add_argument("--position-groups", default="visual_span,answer_adjacent,top_hidden_delta,visual+answer")
    parser.add_argument("--scales", default="1.0")
    parser.add_argument("--answer-prefix", default="The answer is ")
    parser.add_argument("--answer-adjacent-count", type=int, default=4)
    parser.add_argument("--top-delta-count", type=int, default=32)
    parser.add_argument("--max-prompt-runs", type=int, default=0)
    parser.add_argument("--max-clean-rank", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    args = parser.parse_args()

    requested_layers = args.layers
    mask_conditions = _parse_csv(args.mask_conditions)
    position_group_names = _parse_csv(args.position_groups)
    scales = _parse_float_csv(args.scales)
    prompt_rows = _read_csv(Path(args.prompt_runs))
    if args.max_prompt_runs > 0:
        prompt_rows = prompt_rows[: args.max_prompt_runs]

    payload: dict[str, Any] = {
        "created_at": _now(),
        "script": "run_stage6_gemma_hidden_causal_lattice.py",
        "args": vars(args),
        "requested_layers": requested_layers,
        "mask_conditions": mask_conditions,
        "position_groups": position_group_names,
        "env_presence": _env_presence(),
        "gpu_before": _gpu_info(),
        "skipped": [],
    }
    _write_json(Path(args.summary_json), payload)

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
    if requested_layers == "all":
        layers = list(range(layer_count))
    else:
        parsed_layers = _parse_int_csv(requested_layers)
        layers = [layer for layer in parsed_layers if 0 <= layer < layer_count]
        skipped_layers = [layer for layer in parsed_layers if layer not in layers]
        if skipped_layers:
            payload["skipped"].append(
                {"reason": "layer_out_of_range", "requested_layers": skipped_layers, "available_layer_count": layer_count}
            )
            _log(f"skipping unavailable layers {skipped_layers}; available language layers={layer_count}")
    payload["layers"] = layers
    payload["available_layer_count"] = layer_count
    if not layers:
        payload.update({"finished_at": _now(), "status": "blocked_no_valid_layers", "rows": 0})
        _write_json(Path(args.summary_json), payload)
        return 2
    modules = {layer: model.model.language_model.layers[layer] for layer in layers}
    fields = _output_fields()
    rows: list[dict[str, Any]] = []
    processed_keys: set[tuple[str, str]] = set()
    out_path = Path(args.out_csv)
    if args.resume and out_path.exists() and out_path.stat().st_size > 0:
        existing = _read_csv(out_path)
        rows.extend(existing)
        processed_keys = {(row.get("sample_id", ""), row.get("prompt_name", "")) for row in existing}
        _log(f"resume loaded rows={len(existing)} processed_prompt_runs={len(processed_keys)}")
        payload.update({"status": "resume_loaded", "usable_prompt_runs": len(processed_keys), "rows": len(rows), "checkpoint_at": _now()})
        _write_json(Path(args.summary_json), payload)
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
            target_ids = [int(item["token_id"]) for item in _target_candidates(tokenizer, row.get("answer_text", ""))]
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
            input_ids = clean_inputs["input_ids"][0].detach().cpu().tolist()
            token_texts = tokenizer.convert_ids_to_tokens(input_ids)
            clean_token_scores = _score_token_map(clean_outputs.logits, tokenizer, [target_token_id, wrong_token_id])

            for mask_condition in mask_conditions:
                mask_file = _mask_path(row, args.mask_root, mask_condition)
                if not mask_file.exists():
                    payload["skipped"].append(
                        {"sample_id": row.get("sample_id"), "prompt_name": row.get("prompt_name"), "mask_condition": mask_condition, "reason": "mask_missing"}
                    )
                    continue
                mask = Image.open(mask_file).convert("L").resize(image.size)
                masked_image = _apply_mask(image, mask, (128, 128, 128))
                mask_inputs = _gemma_inputs(processor, masked_image, question, args.answer_prefix, device)
                with torch.inference_mode():
                    mask_outputs = model(**mask_inputs, output_hidden_states=True, use_cache=False)
                mask_token_scores = _score_token_map(mask_outputs.logits, tokenizer, [target_token_id, wrong_token_id])

                for layer in layers:
                    hidden_index = min(layer + 1, len(clean_outputs.hidden_states) - 1)
                    clean_hidden = clean_outputs.hidden_states[hidden_index].detach()
                    mask_hidden = mask_outputs.hidden_states[hidden_index].detach()
                    groups = _position_groups(
                        clean_hidden=clean_hidden,
                        mask_hidden=mask_hidden,
                        clean_inputs=clean_inputs,
                        input_ids=input_ids,
                        token_texts=token_texts,
                        answer_adjacent_count=args.answer_adjacent_count,
                        top_delta_count=args.top_delta_count,
                    )
                    for position_group in position_group_names:
                        positions = groups.get(position_group, [])
                        if not positions:
                            continue
                        for scale in scales:
                            for direction in ["restore", "corrupt"]:
                                if direction == "restore":
                                    base_scores = mask_token_scores
                                    patched_scores = _score_with_patch(
                                        model=model,
                                        module=modules[layer],
                                        inputs=mask_inputs,
                                        tokenizer=tokenizer,
                                        token_ids=[target_token_id, wrong_token_id],
                                        positions=positions,
                                        source_hidden=clean_hidden,
                                        target_hidden=mask_hidden,
                                        scale=scale,
                                    )
                                    effect_sign = 1.0
                                else:
                                    base_scores = clean_token_scores
                                    patched_scores = _score_with_patch(
                                        model=model,
                                        module=modules[layer],
                                        inputs=clean_inputs,
                                        tokenizer=tokenizer,
                                        token_ids=[target_token_id, wrong_token_id],
                                        positions=positions,
                                        source_hidden=mask_hidden,
                                        target_hidden=clean_hidden,
                                        scale=scale,
                                    )
                                    effect_sign = -1.0
                                for token_label, token_id in [("target", target_token_id), ("wrong", wrong_token_id)]:
                                    before = base_scores[token_id]
                                    after = patched_scores[token_id]
                                    row_out = _row_base(
                                        row,
                                        layer=layer,
                                        hidden_state_index=hidden_index,
                                        mask_condition=mask_condition,
                                        direction=direction,
                                        position_group=position_group,
                                        positions=positions,
                                        scale=scale,
                                    )
                                    if direction == "restore":
                                        logit_effect = float(after["target_logit"]) - float(before["target_logit"])
                                        rank_effect = int(before["target_rank"]) - int(after["target_rank"])
                                    else:
                                        logit_effect = float(before["target_logit"]) - float(after["target_logit"])
                                        rank_effect = int(after["target_rank"]) - int(before["target_rank"])
                                    row_out.update(
                                        {
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
                                            "clean_target_logit": clean_token_scores[target_token_id]["target_logit"],
                                            "mask_target_logit": mask_token_scores[target_token_id]["target_logit"],
                                            "clean_target_rank": clean_token_scores[target_token_id]["target_rank"],
                                            "mask_target_rank": mask_token_scores[target_token_id]["target_rank"],
                                            "clean_wrong_logit": clean_token_scores[wrong_token_id]["target_logit"],
                                            "mask_wrong_logit": mask_token_scores[wrong_token_id]["target_logit"],
                                            "effect_sign": effect_sign,
                                        }
                                    )
                                    rows.append(row_out)
                del mask_outputs
                torch.cuda.empty_cache()
            usable_runs += 1
            if idx % 5 == 0:
                _log(f"prompt-runs processed: {idx}/{len(prompt_rows)} usable={usable_runs} rows={len(rows)}")
            if args.checkpoint_every > 0 and usable_runs % args.checkpoint_every == 0:
                _write_csv(out_path, rows, fields)
                payload.update({"status": "running_checkpoint", "usable_prompt_runs": usable_runs, "rows": len(rows), "checkpoint_at": _now()})
                _write_json(Path(args.summary_json), payload)
            del clean_outputs
            torch.cuda.empty_cache()
        except Exception as exc:
            payload["skipped"].append({"sample_id": row.get("sample_id"), "prompt_name": row.get("prompt_name"), "reason": repr(exc)})
            torch.cuda.empty_cache()

    _write_csv(Path(args.out_csv), rows, fields)
    payload.update(
        {
            "finished_at": _now(),
            "status": "ok" if rows else "empty",
            "usable_prompt_runs": usable_runs,
            "rows": len(rows),
            "gpu_after": _gpu_info(),
        }
    )
    _write_json(Path(args.summary_json), payload)
    _log(f"wrote {len(rows)} rows to {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
