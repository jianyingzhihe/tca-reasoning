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
    _prompt,
    _qwen_bucket_positions,
    _qwen_inputs,
    _rank_and_top,
)
from run_cross_model_hidden_position_patch_smoke import _answer_adjacent_positions, _load_sample_manifest
from run_stage4_qwen_causal_cutter_validation import _top_wrong_token
from run_stage6_gemma_hidden_generation_bridge import (
    _answer_token_ids,
    _bottom_delta_positions,
    _greedy_decode_with_patch,
    _output_fields,
    _row,
    _sequence_score_with_patch,
    _write_csv,
    _write_json,
)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_int_csv(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _dedupe(values: list[int]) -> list[int]:
    return sorted({int(value) for value in values if int(value) >= 0})


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
    input_ids: list[int],
    token_texts: list[str],
    answer_adjacent_count: int,
    top_delta_count: int,
) -> dict[str, list[int]]:
    visual = _qwen_bucket_positions(input_ids, token_texts)["image_marker_or_span"]
    answer = _answer_adjacent_positions(len(input_ids), set(visual), answer_adjacent_count)
    visual_answer = _dedupe(visual + answer)
    return {
        "visual_span": _dedupe(visual),
        "answer_adjacent": _dedupe(answer),
        "visual+answer": visual_answer,
        "top_hidden_delta": _top_delta_positions(clean_hidden, mask_hidden, visual_answer, top_delta_count),
    }


def _target_candidate_ids(tokenizer, answer: str) -> list[int]:
    variants = [answer, " " + answer, answer.capitalize(), " " + answer.capitalize()]
    out: list[int] = []
    seen: set[int] = set()
    for variant in variants:
        ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        token_id = int(ids[0])
        if token_id not in seen:
            seen.add(token_id)
            out.append(token_id)
    return out


def _qwen_manifest_rows(path: Path, max_pairs: int) -> list[dict[str, str]]:
    rows = _read_csv(path)
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row.get("sample_id", ""), row.get("prompt_name", ""), row.get("mask_condition", "answer_mask"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        if max_pairs > 0 and len(deduped) >= max_pairs:
            break
    return deduped


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage6-022 Qwen hidden-residual node-to-generation bridge.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--annotation-roots", required=True)
    parser.add_argument("--sample-manifest", required=True)
    parser.add_argument("--bridge-manifest", required=True)
    parser.add_argument("--layers", default="14")
    parser.add_argument("--position-group", default="top_hidden_delta")
    parser.add_argument("--top-delta-count", type=int, default=32)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--answer-prefix", default="The answer is ")
    parser.add_argument("--min-gpu-free-gb", type=float, default=18.0)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    rows = _qwen_manifest_rows(Path(args.bridge_manifest), args.max_pairs)
    layers = _parse_int_csv(args.layers)
    sample_lookup = _load_sample_manifest(Path(args.sample_manifest))
    roots = [Path(path) for path in _parse_csv(args.annotation_roots)]
    work_dir = Path(args.out_json).parent / "qwen_hidden_generation_bridge_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "created_at": _now(),
        "script": "run_stage6_qwen_hidden_generation_bridge.py",
        "args": vars(args),
        "pair_count": len(rows),
        "layers": layers,
        "env_presence": _env_presence(),
        "gpu_before": _gpu_info(),
        "claim_boundary": (
            "Qwen Stage6-022 hidden bridge uses the same hidden-residual restore/corrupt generation-scoring lens "
            "as Gemma, not CLT/PLT feature-store interventions."
        ),
        "skipped": [],
    }
    gpu = payload["gpu_before"]
    if not gpu.get("available") or float(gpu.get("free_gb", 0.0)) < args.min_gpu_free_gb:
        payload["decision"] = {"status": "blocked", "reason": "insufficient_gpu_free_memory"}
        _write_json(Path(args.out_json), payload)
        return 0
    _write_json(Path(args.out_json), payload)

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model_name, local_files_only=True)
    tokenizer = processor.tokenizer
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()
    device = _first_param_device(model)
    layer_count = len(model.language_model.layers)
    layers = [layer for layer in layers if 0 <= layer < layer_count]
    modules = {layer: model.language_model.layers[layer] for layer in layers}
    payload["available_layer_count"] = layer_count
    payload["layers"] = layers
    _write_json(Path(args.out_json), payload)

    out_rows: list[dict[str, Any]] = []
    for bridge_row in rows:
        sample_id = bridge_row.get("sample_id", "")
        prompt_name = bridge_row.get("prompt_name", "")
        sample = sample_lookup.get(sample_id)
        if sample is None:
            payload["skipped"].append({"sample_id": sample_id, "prompt_name": prompt_name, "reason": "sample_missing"})
            continue
        try:
            mask_condition = bridge_row.get("mask_condition") or "answer_mask"
            answer_text = bridge_row.get("target_answer") or sample.get("answer", "")
            answer_ids = _answer_token_ids(tokenizer, answer_text)
            target_ids = _target_candidate_ids(tokenizer, answer_text)
            if not answer_ids or not target_ids:
                payload["skipped"].append({"sample_id": sample_id, "prompt_name": prompt_name, "reason": "target_tokenization_empty"})
                continue
            mask_info = _load_masks(sample, roots, work_dir)
            if mask_info["status"] != "ok":
                payload["skipped"].append({"sample_id": sample_id, "prompt_name": prompt_name, "reason": "mask_missing", "mask_info": mask_info})
                continue
            image_path = Path(mask_info["image_path"])
            clean_image = Image.open(image_path).convert("RGB")
            evidence_mask = mask_info["answer_mask"] if mask_condition == "answer_mask" else mask_info["union_mask"]
            masked_image = _apply_mask(clean_image, evidence_mask, (128, 128, 128))
            question = _prompt(sample["question"], prompt_name)
            clean_inputs = _qwen_inputs(processor, clean_image, str(image_path), question, args.answer_prefix, device)
            mask_inputs = _qwen_inputs(processor, masked_image, str(image_path), question, args.answer_prefix, device)
            with torch.inference_mode():
                clean_outputs = model(**clean_inputs, output_hidden_states=True, use_cache=False)
                mask_outputs = model(**mask_inputs, output_hidden_states=True, use_cache=False)
            clean_score = _rank_and_top(clean_outputs.logits, tokenizer, target_ids)
            target_token_id = int(bridge_row.get("target_token_id") or clean_score["target_token_id"])
            wrong = _top_wrong_token(clean_outputs.logits, tokenizer, target_token_id)
            wrong_token_id = int(wrong["wrong_token_id"])
            input_ids = clean_inputs["input_ids"][0].detach().cpu().tolist()
            token_texts = tokenizer.convert_ids_to_tokens(input_ids)

            source = {
                **bridge_row,
                "sample_id": sample_id,
                "prompt_name": prompt_name,
                "mask_condition": mask_condition,
                "position_group": args.position_group,
            }
            for layer in layers:
                hidden_index = min(layer + 1, len(clean_outputs.hidden_states) - 1)
                clean_hidden = clean_outputs.hidden_states[hidden_index].detach()
                mask_hidden = mask_outputs.hidden_states[hidden_index].detach()
                groups = _position_groups(
                    clean_hidden=clean_hidden,
                    mask_hidden=mask_hidden,
                    input_ids=input_ids,
                    token_texts=token_texts,
                    answer_adjacent_count=4,
                    top_delta_count=args.top_delta_count,
                )
                source_positions = groups.get(args.position_group, [])
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
                            "sample_id": sample_id,
                            "prompt_name": prompt_name,
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
                interventions = [
                    ("hidden_restore_source", "restore", "source_hidden", mask_inputs, mask_seq, mask_answer, source_restore_patch, source_positions),
                    ("hidden_restore_control", "restore", "control_hidden", mask_inputs, mask_seq, mask_answer, control_restore_patch, control_positions),
                    ("hidden_corrupt_source", "corrupt", "source_hidden", clean_inputs, clean_seq, clean_answer, source_corrupt_patch, source_positions),
                    ("hidden_corrupt_control", "corrupt", "control_hidden", clean_inputs, clean_seq, clean_answer, control_corrupt_patch, control_positions),
                ]
                for condition, direction, group, base_inputs, reference_seq, reference_answer, patch, positions in interventions:
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
                            asset_id="qwen2p5vl_hidden",
                            bridge_operator="hidden_residual",
                            source=source,
                            condition=condition,
                            direction=direction,
                            feature_group=group,
                            top_k=str(len(positions)),
                            layer=layer,
                            position_group=args.position_group,
                            positions=positions,
                            target_answer=answer_text,
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
                        )
                    )
            del clean_outputs, mask_outputs
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            payload["skipped"].append({"sample_id": sample_id, "prompt_name": prompt_name, "reason": repr(exc)})
            torch.cuda.empty_cache()

    _write_csv(Path(args.out_csv), out_rows, _output_fields())
    payload.update(
        {
            "finished_at": _now(),
            "status": "ok" if out_rows else "empty",
            "rows": len(out_rows),
            "gpu_after": _gpu_info(),
        }
    )
    _write_json(Path(args.out_json), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
