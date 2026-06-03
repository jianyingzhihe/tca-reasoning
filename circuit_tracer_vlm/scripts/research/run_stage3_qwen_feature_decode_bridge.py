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
    _rank_and_top,
    _replace_hidden,
)
from run_cross_model_hidden_position_patch_smoke import (
    _answer_adjacent_positions,
    _load_sample_manifest,
    _make_groups,
)
from run_stage2o_cross_model_source_control_probe import _patch_feature_tensor


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage3-qwen-decode] {message}", flush=True)


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


def _filter_feature_intervention(intervention: dict[str, Any], seq_len: int) -> dict[str, Any] | None:
    positions = [
        int(position)
        for position in intervention["positions"]
        if int(position) < seq_len and int(position) < int(intervention["feature_values"].shape[1])
    ]
    if not positions:
        return None
    return {**intervention, "positions": positions}


def _greedy_decode_with_feature_patch(
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
            filtered = _filter_feature_intervention(intervention, int(hidden.shape[1]))
            if filtered is None:
                return output
            patched = _patch_feature_tensor(hidden, **filtered)
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
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], device=input_ids.device, dtype=input_ids.dtype)],
                dim=1,
            )
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is not None and next_id == int(eos_id):
                break
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    continuation = _decode_ids(tokenizer, generated)
    generated_text = f"The answer is {continuation}".strip()
    score = first_score or {}
    return {
        "prompt_len": prompt_len,
        "generated_token_ids": generated,
        "generated_continuation": continuation,
        "generated_text": generated_text,
        "predicted_answer": _normalize_answer(generated_text),
        "first_generated_token_id": generated[0] if generated else "",
        "first_generated_token": _decode_ids(tokenizer, generated[:1]),
        "first_step_target_logit": score.get("target_logit", ""),
        "first_step_target_rank": score.get("target_rank", ""),
        "first_step_top1_token": score.get("top1_token", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage3 Qwen source/control feature decoded bridge smoke.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--transcoder-ref", required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--annotation-roots", required=True)
    parser.add_argument("--sample-manifest", required=True)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--answer-prefix", default="The answer is ")
    parser.add_argument("--min-gpu-free-gb", type=float, default=18.0)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    pair_rows = [row for row in _read_csv(Path(args.pair_manifest)) if row.get("asset_id") == args.asset_id]
    sample_lookup = _load_sample_manifest(Path(args.sample_manifest))
    roots = [Path(path) for path in _parse_csv(args.annotation_roots)]
    work_dir = Path(args.out_json).parent / f"{args.asset_id}_decode_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "created_at": _now(),
        "asset_id": args.asset_id,
        "model_name": args.model_name,
        "transcoder_ref": args.transcoder_ref,
        "pair_count": len(pair_rows),
        "env_presence": _env_presence(),
        "gpu_before": _gpu_info(),
        "claim_boundary": (
            "Short greedy decoded generation smoke for selected Qwen feature source/control pairs. "
            "This is not full source tracing and not a replacement for Gemma mainline."
        ),
        "skipped": [],
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
    for pair in pair_rows:
        sample_id = pair["sample_id"]
        prompt_name = pair["prompt_name"]
        mask_condition = pair["mask_condition"]
        layer = int(pair.get("layer", 26))
        target_module = model.language_model.layers[layer]
        hidden_index = layer
        sample = sample_lookup.get(sample_id)
        if sample is None:
            payload["skipped"].append({"sample_id": sample_id, "prompt_name": prompt_name, "reason": "sample_missing"})
            continue
        mask_info = _load_masks(sample, roots, work_dir)
        if mask_info["status"] != "ok":
            payload["skipped"].append({"sample_id": sample_id, "prompt_name": prompt_name, "reason": "mask_missing", "mask_info": mask_info})
            continue
        image_path = Path(mask_info["image_path"])
        clean_image = Image.open(image_path).convert("RGB")
        mask = mask_info["answer_mask"] if mask_condition == "answer_mask" else mask_info["union_mask"]
        mask_image = _apply_mask(clean_image, mask, (128, 128, 128))
        question = _prompt(sample["question"], prompt_name)
        clean_inputs = _qwen_inputs(processor, clean_image, str(image_path), question, args.answer_prefix, device)
        mask_inputs = _qwen_inputs(processor, mask_image, str(image_path), question, args.answer_prefix, device)
        input_ids = clean_inputs["input_ids"][0].detach().cpu().tolist()
        token_texts = tokenizer.convert_ids_to_tokens(input_ids)
        buckets = _qwen_bucket_positions(input_ids, token_texts)
        visual_positions = buckets.get("image_marker_or_span", [])
        answer_positions = _answer_adjacent_positions(len(input_ids), set(visual_positions), 4)
        target_id = int(pair["target_token_id"])
        target_ids = [target_id]

        with torch.inference_mode():
            clean_outputs = model(**clean_inputs, output_hidden_states=True, use_cache=False)
            mask_outputs = model(**mask_inputs, output_hidden_states=True, use_cache=False)
        clean_hidden = clean_outputs.hidden_states[hidden_index].to(device=device, dtype=dtype).detach()
        mask_hidden = mask_outputs.hidden_states[hidden_index].to(device=device, dtype=dtype).detach()
        clean_features = encode_features(clean_hidden, layer).detach()
        mask_features = encode_features(mask_hidden, layer).detach()
        feature_drop = (clean_features - mask_features).detach()

        groups, _diagnostics = _make_groups(
            model_family="qwen",
            clean_inputs=clean_inputs,
            clean_hidden=clean_hidden,
            union_hidden=mask_hidden,
            visual_positions=visual_positions,
            answer_positions=answer_positions,
            evidence_mask=mask,
            sample_id=sample_id,
            prompt_name=prompt_name,
            default_count=32,
            max_evidence_positions=64,
            random_controls=4,
        )
        group_lookup = {group["group_name"]: group for group in groups}
        selected_group = group_lookup.get(pair.get("position_group", "top_hidden_delta_plus_answer_adjacent"))
        if selected_group is None or not selected_group["positions"]:
            payload["skipped"].append({"sample_id": sample_id, "prompt_name": prompt_name, "reason": "position_group_missing"})
            continue
        positions = [int(position) for position in selected_group["positions"] if int(position) < clean_hidden.shape[1]]
        source_feature = int(pair["source_feature_id"])
        control_feature = int(pair["control_feature_id"])

        source_vectors = decoder_vectors(layer, [source_feature])
        control_vectors = decoder_vectors(layer, [control_feature])
        conditions = [
            ("baseline_clean", "baseline", clean_inputs, None),
            ("baseline_mask", "baseline", mask_inputs, None),
            (
                "source_restore",
                "restore",
                mask_inputs,
                {
                    "positions": positions,
                    "feature_ids": [source_feature],
                    "feature_values": feature_drop,
                    "decoder_vectors": source_vectors,
                    "scale": 1.0,
                    "mode": "restore",
                },
            ),
            (
                "control_restore",
                "restore",
                mask_inputs,
                {
                    "positions": positions,
                    "feature_ids": [control_feature],
                    "feature_values": feature_drop,
                    "decoder_vectors": control_vectors,
                    "scale": 1.0,
                    "mode": "restore",
                },
            ),
            (
                "source_zeroing",
                "zeroing",
                clean_inputs,
                {
                    "positions": positions,
                    "feature_ids": [source_feature],
                    "feature_values": clean_features,
                    "decoder_vectors": source_vectors,
                    "scale": 1.0,
                    "mode": "zero",
                },
            ),
            (
                "control_zeroing",
                "zeroing",
                clean_inputs,
                {
                    "positions": positions,
                    "feature_ids": [control_feature],
                    "feature_values": clean_features,
                    "decoder_vectors": control_vectors,
                    "scale": 1.0,
                    "mode": "zero",
                },
            ),
        ]
        decoded: dict[str, dict[str, Any]] = {}
        for condition_name, direction, inputs, intervention in conditions:
            result = _greedy_decode_with_feature_patch(
                model=model,
                tokenizer=tokenizer,
                base_inputs=inputs,
                module=target_module,
                intervention=intervention,
                target_ids=target_ids,
                max_new_tokens=args.max_new_tokens,
            )
            decoded[condition_name] = result
            out_rows.append(
                {
                    "asset_id": args.asset_id,
                    "sample_id": sample_id,
                    "prompt_name": prompt_name,
                    "mask_condition": mask_condition,
                    "condition": condition_name,
                    "direction": direction,
                    "target_answer": sample["answer"],
                    "target_token_id": target_id,
                    "target_token": pair.get("target_token", ""),
                    "source_feature_id": source_feature,
                    "control_feature_id": control_feature,
                    "position_count": len(positions),
                    "generated_text": result["generated_text"],
                    "predicted_answer": result["predicted_answer"],
                    "target_hit": _target_hit(result["generated_text"], sample["answer"]),
                    "first_generated_token": result["first_generated_token"],
                    "first_step_target_logit": result["first_step_target_logit"],
                    "first_step_target_rank": result["first_step_target_rank"],
                    "first_step_top1_token": result["first_step_top1_token"],
                    "baseline_clean_answer": "",
                    "baseline_mask_answer": "",
                    "changed_vs_clean": "",
                    "changed_vs_mask": "",
                }
            )
        clean_answer = decoded["baseline_clean"]["predicted_answer"]
        mask_answer = decoded["baseline_mask"]["predicted_answer"]
        for row in out_rows[-len(conditions) :]:
            row["baseline_clean_answer"] = clean_answer
            row["baseline_mask_answer"] = mask_answer
            row["changed_vs_clean"] = row["predicted_answer"] != clean_answer
            row["changed_vs_mask"] = row["predicted_answer"] != mask_answer
        del clean_outputs, mask_outputs, clean_features, mask_features, feature_drop
        torch.cuda.empty_cache()

    payload["gpu_after"] = _gpu_info()
    payload["decision"] = {
        "status": "completed" if out_rows else "blocked_no_rows",
        "rows": len(out_rows),
        "pairs_completed": len({(row["asset_id"], row["sample_id"], row["prompt_name"], row["mask_condition"]) for row in out_rows}),
        "skipped": payload["skipped"],
    }
    fields = [
        "asset_id",
        "sample_id",
        "prompt_name",
        "mask_condition",
        "condition",
        "direction",
        "target_answer",
        "target_token_id",
        "target_token",
        "source_feature_id",
        "control_feature_id",
        "position_count",
        "generated_text",
        "predicted_answer",
        "target_hit",
        "first_generated_token",
        "first_step_target_logit",
        "first_step_target_rank",
        "first_step_top1_token",
        "baseline_clean_answer",
        "baseline_mask_answer",
        "changed_vs_clean",
        "changed_vs_mask",
    ]
    _write_csv(Path(args.out_csv), out_rows, fields)
    _write_json(Path(args.out_json), payload)
    _log(f"done status={payload['decision']['status']} rows={len(out_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
