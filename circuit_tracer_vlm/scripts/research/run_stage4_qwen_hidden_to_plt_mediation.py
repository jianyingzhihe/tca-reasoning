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
    _apply_mask,
    _env_presence,
    _first_param_device,
    _gpu_info,
    _prompt,
    _qwen_bucket_positions,
    _qwen_inputs,
    _rank_and_top,
    _replace_hidden,
)
from run_cross_model_hidden_position_patch_smoke import _answer_adjacent_positions
from run_stage4_qwen_causal_cutter_validation import _top_wrong_token
from run_stage4_qwen_evidence_linked_cutter_v2 import _decoder_vectors


MASK_FILE_NAMES = {
    "answer_mask": "answer.png",
    "union_mask": "union.png",
    "shifted_mask": "shifted.png",
    "shuffled_mask": "shuffled.png",
}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage4-qwen-hidden-to-plt] {message}", flush=True)


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


def _i(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw)) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


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


def _dedupe(values: list[int]) -> list[int]:
    return sorted({int(value) for value in values if int(value) >= 0})


def _top_hidden_delta_positions(clean_hidden: torch.Tensor, mask_hidden: torch.Tensor, positions: list[int], count: int) -> list[int]:
    if not positions:
        return []
    pos = torch.tensor(positions, device=clean_hidden.device, dtype=torch.long)
    scores = (clean_hidden[:, pos, :].float() - mask_hidden[:, pos, :].float()).norm(dim=-1)[0]
    pairs = [(float(score), int(position)) for score, position in zip(scores.detach().cpu().tolist(), positions, strict=False)]
    pairs.sort(reverse=True)
    return _dedupe([position for _score, position in pairs[:count]])


def _positions(clean_hidden: torch.Tensor, mask_hidden: torch.Tensor, input_ids: list[int], token_texts: list[str], count: int) -> list[int]:
    visual = _qwen_bucket_positions(input_ids, token_texts)["image_marker_or_span"]
    answer = _answer_adjacent_positions(len(input_ids), set(visual), 4)
    return _top_hidden_delta_positions(clean_hidden, mask_hidden, _dedupe(visual + answer), count)


def _score_with_patch(
    *,
    model,
    module,
    inputs: dict[str, Any],
    tokenizer,
    target_ids: list[int],
    patch_by_position: dict[int, torch.Tensor],
    scale: float,
) -> dict[int, dict[str, Any]]:
    def _hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        patched = hidden.clone()
        for pos, delta in patch_by_position.items():
            if 0 <= pos < patched.shape[1]:
                patched[:, pos, :] = patched[:, pos, :] + scale * delta.to(device=patched.device, dtype=patched.dtype)
        return _replace_hidden(output, patched)

    handle = module.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=False, use_cache=False)
    finally:
        handle.remove()
    return {int(token_id): _rank_and_top(outputs.logits, tokenizer, [int(token_id)]) for token_id in target_ids}


def _score_clean(logits: torch.Tensor, tokenizer, target_ids: list[int]) -> dict[int, dict[str, Any]]:
    return {int(token_id): _rank_and_top(logits, tokenizer, [int(token_id)]) for token_id in target_ids}


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
    pairs: list[tuple[float, int, int, float]] = []
    for pos in positions:
        if pos < 0 or pos >= clean_features.shape[1]:
            continue
        drop = (clean_features[0, pos, :].float() - mask_features[0, pos, :].float()).detach()
        active = clean_features[0, pos, :].float() > 0
        drop[~active] = 0
        vals, ids = torch.topk(drop.abs(), k=min(top_k, drop.numel()))
        for value, feature_id in zip(vals.cpu().tolist(), ids.cpu().tolist(), strict=False):
            if value <= 0:
                continue
            signed = float(clean_features[0, pos, int(feature_id)].float().item() - mask_features[0, pos, int(feature_id)].float().item())
            pairs.append((float(value), int(pos), int(feature_id), signed))
    pairs.sort(reverse=True)
    selected = pairs[:top_k]
    feature_ids = [item[2] for item in selected]
    if not feature_ids:
        return {}, {}, ""
    vectors = _decoder_vectors(transcoders, layer, feature_ids, device, dtype)
    recon: dict[int, torch.Tensor] = {}
    for idx, (_abs_value, pos, feature_id, signed_drop) in enumerate(selected):
        recon[pos] = recon.get(pos, torch.zeros_like(clean_hidden[:, pos, :])) + signed_drop * vectors[idx].view(1, -1)
    residual: dict[int, torch.Tensor] = {}
    for pos in positions:
        full_delta = clean_hidden[:, pos, :].to(device=device, dtype=dtype) - mask_hidden[:, pos, :].to(device=device, dtype=dtype)
        residual[pos] = full_delta - recon.get(pos, torch.zeros_like(full_delta)).to(device=device, dtype=dtype)
    selected_text = "|".join(f"P{pos}:F{feature_id}" for _value, pos, feature_id, _signed in selected)
    return recon, residual, selected_text


def _row_base(row: dict[str, str], *, mask_condition: str, operator: str, top_k: str, positions: list[int]) -> dict[str, Any]:
    return {
        "sample_id": row.get("sample_id", ""),
        "prompt_name": row.get("prompt_name", ""),
        "reasoning_operation": row.get("reasoning_operation", ""),
        "image_dependence_tier": row.get("image_dependence_tier", ""),
        "paperpack_source": row.get("paperpack_source", ""),
        "mask_condition": mask_condition,
        "operator": operator,
        "top_k": top_k,
        "position_group": "top_hidden_delta",
        "position_count": len(positions),
        "positions": "|".join(str(pos) for pos in positions),
        "target_answer": row.get("answer_text", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage4-038 Qwen hidden-to-PLT mediation test.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--transcoder-ref", required=True)
    parser.add_argument("--prompt-runs", required=True)
    parser.add_argument("--image-root", default="")
    parser.add_argument("--mask-root", default="")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--answer-prefix", default="The answer is ")
    parser.add_argument("--mask-conditions", default="answer_mask,union_mask,shifted_mask,shuffled_mask")
    parser.add_argument("--top-ks", default="8,16,32,64,128")
    parser.add_argument("--top-hidden-count", type=int, default=16)
    parser.add_argument("--max-prompt-runs", type=int, default=6)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--min-gpu-free-gb", type=float, default=12.0)
    args = parser.parse_args()

    prompt_rows = _read_csv(Path(args.prompt_runs))
    if args.max_prompt_runs > 0:
        prompt_rows = prompt_rows[: args.max_prompt_runs]
    payload: dict[str, Any] = {
        "created_at": _now(),
        "model_name": args.model_name,
        "transcoder_ref": args.transcoder_ref,
        "prompt_runs": args.prompt_runs,
        "layer": args.layer,
        "requested_prompt_runs": len(prompt_rows),
        "env_presence": _env_presence(),
        "gpu_before": _gpu_info(),
        "skipped": [],
    }
    gpu = payload["gpu_before"]
    if not gpu.get("available") or float(gpu.get("free_gb", 0.0)) < args.min_gpu_free_gb:
        payload["decision"] = {"status": "blocked", "reason": "insufficient_gpu_free_memory"}
        _write_json(Path(args.summary_json), payload)
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
    transcoders, config = load_transcoder_from_hub(args.transcoder_ref, device=device, dtype=dtype, lazy_encoder=True, lazy_decoder=True)
    module = model.language_model.layers[args.layer]
    payload["transcoder"] = {"type": type(transcoders).__name__, "config_model_kind": config.get("model_kind", "")}

    mask_conditions = _parse_csv(args.mask_conditions)
    top_ks = _parse_int_csv(args.top_ks)
    rows: list[dict[str, Any]] = []
    usable = 0

    for idx, row in enumerate(prompt_rows, start=1):
        try:
            image_path = _image_path(row, args.image_root)
            image = Image.open(image_path).convert("RGB")
            question = _prompt(row.get("question_text", ""), row.get("prompt_name", ""))
            clean_inputs = _qwen_inputs(processor, image, str(image_path), question, args.answer_prefix, device)
            with torch.inference_mode():
                clean_outputs = model(**clean_inputs, output_hidden_states=True, use_cache=False)
            clean_hidden = clean_outputs.hidden_states[args.layer + 1].to(device=device, dtype=dtype)
            target_candidates = tokenizer.encode(" " + row.get("answer_text", ""), add_special_tokens=False)
            if not target_candidates:
                payload["skipped"].append({"sample_id": row.get("sample_id"), "reason": "target_tokenization_empty"})
                continue
            target_token_id = int(target_candidates[0])
            wrong = _top_wrong_token(clean_outputs.logits, tokenizer, target_token_id)
            wrong_token_id = int(wrong["wrong_token_id"])
            clean_scores = _score_clean(clean_outputs.logits, tokenizer, [target_token_id, wrong_token_id])
            input_ids = clean_inputs["input_ids"][0].detach().cpu().tolist()
            token_texts = tokenizer.convert_ids_to_tokens(input_ids)

            for mask_condition in mask_conditions:
                mask_file = _mask_path(row, args.mask_root, mask_condition)
                if not mask_file.exists():
                    payload["skipped"].append({"sample_id": row.get("sample_id"), "mask_condition": mask_condition, "reason": "mask_missing"})
                    continue
                mask = Image.open(mask_file).convert("L").resize(image.size)
                masked_image = _apply_mask(image, mask, (128, 128, 128))
                mask_inputs = _qwen_inputs(processor, masked_image, str(image_path), question, args.answer_prefix, device)
                with torch.inference_mode():
                    mask_outputs = model(**mask_inputs, output_hidden_states=True, use_cache=False)
                mask_hidden = mask_outputs.hidden_states[args.layer + 1].to(device=device, dtype=dtype)
                mask_scores = _score_clean(mask_outputs.logits, tokenizer, [target_token_id, wrong_token_id])
                positions = _positions(clean_hidden, mask_hidden, input_ids, token_texts, args.top_hidden_count)
                if not positions:
                    payload["skipped"].append({"sample_id": row.get("sample_id"), "mask_condition": mask_condition, "reason": "no_positions"})
                    continue
                full_patch = {pos: clean_hidden[:, pos, :] - mask_hidden[:, pos, :] for pos in positions}
                clean_features = transcoders.encode_layer(clean_hidden, args.layer, apply_activation_function=True).detach()
                mask_features = transcoders.encode_layer(mask_hidden, args.layer, apply_activation_function=True).detach()

                operators: list[tuple[str, str, dict[int, torch.Tensor], str]] = [
                    ("hidden_residual", "full", full_patch, ""),
                ]
                for top_k in top_ks:
                    recon, residual, selected = _top_feature_patch(
                        transcoders=transcoders,
                        layer=args.layer,
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
                        module=module,
                        inputs=mask_inputs,
                        tokenizer=tokenizer,
                        target_ids=[target_token_id, wrong_token_id],
                        patch_by_position=patch,
                        scale=args.scale,
                    )
                    for token_label, token_id in [("target", target_token_id), ("wrong", wrong_token_id)]:
                        before = mask_scores[token_id]
                        after = restored[token_id]
                        base = _row_base(row, mask_condition=mask_condition, operator=operator, top_k=top_k, positions=positions)
                        base.update(
                            {
                                "token_scored": token_label,
                                "scored_token_id": token_id,
                                "scored_token": tokenizer.decode([token_id]),
                                "target_token_id": target_token_id,
                                "target_token": tokenizer.decode([target_token_id]),
                                "wrong_token_id": wrong_token_id,
                                "wrong_token": tokenizer.decode([wrong_token_id]),
                                "before_logit": before["target_logit"],
                                "after_logit": after["target_logit"],
                                "logit_effect": float(after["target_logit"]) - float(before["target_logit"]),
                                "before_rank": before["target_rank"],
                                "after_rank": after["target_rank"],
                                "rank_effect": int(before["target_rank"]) - int(after["target_rank"]),
                                "clean_target_logit": clean_scores[target_token_id]["target_logit"],
                                "mask_target_logit": mask_scores[target_token_id]["target_logit"],
                                "clean_target_rank": clean_scores[target_token_id]["target_rank"],
                                "mask_target_rank": mask_scores[target_token_id]["target_rank"],
                                "selected_features": selected,
                            }
                        )
                        rows.append(base)
                del mask_outputs, mask_hidden, mask_features
                torch.cuda.empty_cache()
            usable += 1
            if idx % 5 == 0:
                _log(f"prompt-runs complete: {idx}/{len(prompt_rows)} rows={len(rows)}")
            del clean_outputs, clean_hidden
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            payload["skipped"].append({"sample_id": row.get("sample_id"), "prompt_name": row.get("prompt_name"), "reason": repr(exc)})
            torch.cuda.empty_cache()

    fields = [
        "sample_id",
        "prompt_name",
        "reasoning_operation",
        "image_dependence_tier",
        "paperpack_source",
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
    _write_csv(Path(args.out_csv), rows, fields)
    payload["gpu_after"] = _gpu_info()
    payload["decision"] = {
        "status": "ok" if rows else "blocked_no_rows",
        "usable_prompt_runs": usable,
        "raw_rows": len(rows),
        "skipped_count": len(payload["skipped"]),
    }
    _write_json(Path(args.summary_json), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
