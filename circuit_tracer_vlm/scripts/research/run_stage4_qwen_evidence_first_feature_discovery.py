#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import defaultdict
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
    _target_candidates,
)
from run_cross_model_hidden_position_patch_smoke import _answer_adjacent_positions
from run_stage4_qwen_causal_cutter_validation import (
    _decoder_vectors,
    _forward_with_capture,
    _top_wrong_token,
)


MASK_FILE_NAMES = {
    "answer_mask": "answer.png",
    "union_mask": "union.png",
    "shifted_mask": "shifted.png",
    "shuffled_mask": "shuffled.png",
}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage4-qwen-evidence-first-discovery] {message}", flush=True)


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


def _f(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _i(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw)) if raw not in (None, "") else default
    except ValueError:
        return default


def _image_path(row: dict[str, str], image_root: str) -> Path:
    if image_root:
        return Path(image_root) / Path(row["image_filename"]).name
    return Path(row.get("local_image_path") or row.get("image_path") or row.get("image_filename"))


def _mask_path(row: dict[str, str], mask_root: str, condition: str) -> Path:
    stem = Path(row["image_filename"]).stem
    if condition not in MASK_FILE_NAMES:
        raise ValueError(f"unknown mask condition: {condition}")
    if mask_root:
        return Path(mask_root) / stem / MASK_FILE_NAMES[condition]
    explicit_key = {
        "answer_mask": "answer_mask_path",
        "union_mask": "union_mask_path",
        "shifted_mask": "shifted_mask_path",
        "shuffled_mask": "shuffled_mask_path",
    }[condition]
    if row.get(explicit_key):
        return Path(row[explicit_key])
    if row.get("mask_dir"):
        return Path(row["mask_dir"]) / MASK_FILE_NAMES[condition]
    return Path("")


def _path_str(row: dict[str, str], condition: str) -> str:
    explicit_key = {
        "answer_mask": "answer_mask_path",
        "union_mask": "union_mask_path",
        "shifted_mask": "shifted_mask_path",
        "shuffled_mask": "shuffled_mask_path",
    }[condition]
    if row.get(explicit_key):
        return row[explicit_key]
    if row.get("mask_dir"):
        return str(Path(row["mask_dir"]) / MASK_FILE_NAMES[condition])
    return ""


def _is_numeric(target_token: str, answer: str) -> bool:
    token = target_token.strip().replace(",", "")
    text = answer.strip().lower().replace(",", "")
    numeric_re = re.compile(r"^[+-]?(?:\d+|\d+\.\d+|\d+/\d+)$")
    word_numbers = {
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
    }
    return bool(numeric_re.match(token) or numeric_re.match(text) or text in word_numbers)


def _position_group(input_ids: list[int], token_texts: list[str], group: str) -> tuple[list[int], list[int], list[int]]:
    visual_positions = _qwen_bucket_positions(input_ids, token_texts)["image_marker_or_span"]
    answer_positions = _answer_adjacent_positions(len(input_ids), set(visual_positions), 4)
    if group == "visual_only":
        allowed = visual_positions
    elif group == "answer_adjacent_only":
        allowed = answer_positions
    elif group == "visual_answer":
        allowed = sorted(set(visual_positions + answer_positions))
    else:
        raise ValueError(f"unknown position group: {group}")
    return allowed, visual_positions, answer_positions


def _top_candidates_from_chunk(
    *,
    clean_features: torch.Tensor,
    mask_features: dict[str, torch.Tensor],
    positions: list[int],
    real_conditions: list[str],
    control_conditions: list[str],
    top_k: int,
    min_clean_activation: float,
    min_specificity: float,
) -> list[dict[str, Any]]:
    if not positions:
        return []
    pos_tensor = torch.tensor(positions, device=clean_features.device, dtype=torch.long)
    clean = clean_features[0, pos_tensor, :].float()
    real_drops = torch.stack(
        [(clean - mask_features[condition][0, pos_tensor, :].float()) for condition in real_conditions],
        dim=0,
    )
    control_drops = torch.stack(
        [(clean - mask_features[condition][0, pos_tensor, :].float()) for condition in control_conditions],
        dim=0,
    )
    real_best, real_idx = torch.max(real_drops, dim=0)
    control_max, _control_idx = torch.max(control_drops, dim=0)
    specificity = real_best - control_max
    score = torch.relu(specificity) * torch.log1p(torch.relu(clean))
    score = score.masked_fill(clean <= min_clean_activation, 0.0)
    score = score.masked_fill(specificity <= min_specificity, 0.0)
    positive = score > 0
    if not bool(positive.any().item()):
        return []
    flat = score.flatten()
    k = min(top_k, int(positive.sum().item()))
    vals, flat_ids = torch.topk(flat, k=k)
    n_features = int(clean.shape[1])
    out: list[dict[str, Any]] = []
    for value, flat_id in zip(vals.detach().cpu().tolist(), flat_ids.detach().cpu().tolist(), strict=False):
        local_pos_idx = int(flat_id) // n_features
        feature_id = int(flat_id) % n_features
        condition_idx = int(real_idx[local_pos_idx, feature_id].detach().cpu().item())
        item = {
            "pre_score": float(value),
            "pos": int(positions[local_pos_idx]),
            "feature_id": feature_id,
            "best_real_condition": real_conditions[condition_idx],
            "clean_activation": float(clean[local_pos_idx, feature_id].detach().cpu().item()),
            "real_drop_best": float(real_best[local_pos_idx, feature_id].detach().cpu().item()),
            "control_drop_max": float(control_max[local_pos_idx, feature_id].detach().cpu().item()),
            "evidence_specificity": float(specificity[local_pos_idx, feature_id].detach().cpu().item()),
        }
        for condition in real_conditions + control_conditions:
            drop = clean[local_pos_idx, feature_id] - mask_features[condition][0, int(positions[local_pos_idx]), feature_id].float()
            item[f"{condition}_drop"] = float(drop.detach().cpu().item())
        out.append(item)
    return out


def _select_evidence_first_candidates(
    *,
    clean_features: torch.Tensor,
    mask_features: dict[str, torch.Tensor],
    positions: list[int],
    transcoders,
    layer: int,
    target_direction: torch.Tensor,
    wrong_direction: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    real_conditions: list[str],
    control_conditions: list[str],
    candidate_pool_size: int,
    top_per_prompt_run: int,
    position_chunk_size: int,
    min_clean_activation: float,
    min_specificity: float,
    min_target_contribution: float,
    min_correct_minus_wrong_contribution: float,
    wrong_contribution_ratio: float,
) -> list[dict[str, Any]]:
    pre_candidates: list[dict[str, Any]] = []
    chunk_top_k = max(candidate_pool_size, top_per_prompt_run * 8)
    for start in range(0, len(positions), position_chunk_size):
        chunk = positions[start : start + position_chunk_size]
        pre_candidates.extend(
            _top_candidates_from_chunk(
                clean_features=clean_features,
                mask_features=mask_features,
                positions=chunk,
                real_conditions=real_conditions,
                control_conditions=control_conditions,
                top_k=chunk_top_k,
                min_clean_activation=min_clean_activation,
                min_specificity=min_specificity,
            )
        )
    pre_candidates.sort(key=lambda item: float(item["pre_score"]), reverse=True)
    pre_candidates = pre_candidates[: max(candidate_pool_size, top_per_prompt_run)]
    if not pre_candidates:
        return []

    feature_ids = sorted({int(item["feature_id"]) for item in pre_candidates})
    vectors = _decoder_vectors(transcoders, layer, feature_ids, device, dtype)
    target = target_direction.to(device=vectors.device, dtype=vectors.dtype)
    wrong = wrong_direction.to(device=vectors.device, dtype=vectors.dtype)
    target_contrib = torch.mv(vectors.float(), target.float()).detach().cpu().tolist()
    wrong_contrib = torch.mv(vectors.float(), wrong.float()).detach().cpu().tolist()
    target_lookup = {feature_id: float(target_contrib[idx]) for idx, feature_id in enumerate(feature_ids)}
    wrong_lookup = {feature_id: float(wrong_contrib[idx]) for idx, feature_id in enumerate(feature_ids)}

    out = []
    for item in pre_candidates:
        feature_id = int(item["feature_id"])
        t_contrib = target_lookup.get(feature_id, 0.0)
        w_contrib = wrong_lookup.get(feature_id, 0.0)
        correct_minus_wrong = t_contrib - max(0.0, w_contrib) * wrong_contribution_ratio
        if t_contrib <= min_target_contribution:
            continue
        if correct_minus_wrong <= min_correct_minus_wrong_contribution:
            continue
        final_score = max(0.0, item["evidence_specificity"]) * max(0.0, correct_minus_wrong) * math.log1p(max(0.0, item["clean_activation"]))
        if final_score <= 0:
            continue
        enriched = dict(item)
        enriched.update(
            {
                "target_contribution": t_contrib,
                "wrong_target_contribution": w_contrib,
                "correct_minus_wrong_contribution": correct_minus_wrong,
                "evidence_first_score": final_score,
            }
        )
        out.append(enriched)
    out.sort(key=lambda item: float(item["evidence_first_score"]), reverse=True)
    return out[:top_per_prompt_run]


def _candidate_base(
    *,
    row: dict[str, str],
    selected: dict[str, Any],
    clean_score: dict[str, Any],
    wrong: dict[str, Any],
    rank_within_prompt: int,
    layer: int,
    position_group: str,
    visual_positions: list[int],
    answer_positions: list[int],
    include_main_rank: int,
) -> dict[str, Any]:
    target_token = str(clean_score.get("target_token", ""))
    answer = row.get("answer_text", "")
    pos = int(selected["pos"])
    feature_id = int(selected["feature_id"])
    include_main = "1" if rank_within_prompt <= include_main_rank else "0"
    return {
        "candidate_id": "",
        "candidate_source": "evidence_first_discovery",
        "include_main": include_main,
        "include_pool": "1",
        "include_sensitivity_rank10": "1" if int(clean_score.get("target_rank", 999999)) <= 10 else "0",
        "candidate_rank_within_prompt_run": rank_within_prompt,
        "analysis_group": "numeric" if _is_numeric(target_token, answer) else "non_numeric",
        "sample_id": row.get("sample_id", ""),
        "run": "evidence_first",
        "prompt_name": row.get("prompt_name", ""),
        "source_zeroing_mode": "subtract",
        "source_node_id": f"F:L{layer}:P{pos}:ID{feature_id}",
        "layer": layer,
        "source_pos": pos,
        "source_feature_id": feature_id,
        "position_group": position_group,
        "best_real_condition": selected.get("best_real_condition", ""),
        "path_mass_best": selected.get("evidence_first_score", ""),
        "target_token_id": clean_score.get("target_token_id", ""),
        "target_token": target_token,
        "wrong_token_id": wrong.get("wrong_token_id", ""),
        "wrong_token": wrong.get("wrong_token", ""),
        "original_target_logit": clean_score.get("target_logit", ""),
        "original_target_rank": clean_score.get("target_rank", ""),
        "original_top1_token": clean_score.get("top1_token", ""),
        "clean_target_logit": clean_score.get("target_logit", ""),
        "clean_target_rank": clean_score.get("target_rank", ""),
        "clean_top1_token": clean_score.get("top1_token", ""),
        "clean_activation": selected.get("clean_activation", ""),
        "answer_mask_drop": selected.get("answer_mask_drop", ""),
        "union_mask_drop": selected.get("union_mask_drop", ""),
        "shifted_mask_drop": selected.get("shifted_mask_drop", ""),
        "shuffled_mask_drop": selected.get("shuffled_mask_drop", ""),
        "real_drop_best": selected.get("real_drop_best", ""),
        "control_drop_max": selected.get("control_drop_max", ""),
        "evidence_specificity": selected.get("evidence_specificity", ""),
        "target_contribution": selected.get("target_contribution", ""),
        "wrong_target_contribution": selected.get("wrong_target_contribution", ""),
        "correct_minus_wrong_contribution": selected.get("correct_minus_wrong_contribution", ""),
        "evidence_first_score": selected.get("evidence_first_score", ""),
        "damage_score": selected.get("evidence_first_score", ""),
        "visual_position_count": len(visual_positions),
        "answer_adjacent_position_count": len(answer_positions),
        "known_damaging_feature_ids_for_prompt_run": "",
        "image_filename": row.get("image_filename", ""),
        "local_image_path": row.get("local_image_path", ""),
        "question_text": row.get("question_text", ""),
        "answer_text": answer,
        "reasoning_operation": row.get("reasoning_operation", ""),
        "image_dependence_tier": row.get("image_dependence_tier", ""),
        "paperpack_source": row.get("paperpack_source", ""),
        "mask_dir": row.get("mask_dir", ""),
        "answer_mask_path": _path_str(row, "answer_mask"),
        "union_mask_path": _path_str(row, "union_mask"),
        "shifted_mask_path": _path_str(row, "shifted_mask"),
        "shuffled_mask_path": _path_str(row, "shuffled_mask"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover Qwen PLT feature/position candidates from evidence-mask sensitivity first.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--transcoder-ref", required=True)
    parser.add_argument("--prompt-runs", required=True)
    parser.add_argument("--image-root", default="")
    parser.add_argument("--mask-root", default="")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--layer", type=int, default=26)
    parser.add_argument("--answer-prefix", default="The answer is ")
    parser.add_argument("--real-conditions", default="answer_mask,union_mask")
    parser.add_argument("--control-conditions", default="shifted_mask,shuffled_mask")
    parser.add_argument("--position-group", choices=["visual_answer", "visual_only", "answer_adjacent_only"], default="visual_answer")
    parser.add_argument("--candidate-pool-size", type=int, default=8192)
    parser.add_argument("--top-per-prompt-run", type=int, default=4)
    parser.add_argument("--main-per-prompt-run", type=int, default=2)
    parser.add_argument("--position-chunk-size", type=int, default=64)
    parser.add_argument("--max-prompt-runs", type=int, default=0)
    parser.add_argument("--max-clean-rank", type=int, default=10)
    parser.add_argument("--min-clean-activation", type=float, default=0.0)
    parser.add_argument("--min-specificity", type=float, default=0.0)
    parser.add_argument("--min-target-contribution", type=float, default=0.0)
    parser.add_argument("--min-correct-minus-wrong-contribution", type=float, default=0.0)
    parser.add_argument("--wrong-contribution-ratio", type=float, default=1.0)
    parser.add_argument("--min-gpu-free-gb", type=float, default=12.0)
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "created_at": _now(),
        "script": "run_stage4_qwen_evidence_first_feature_discovery.py",
        "args": vars(args),
        "env_presence": _env_presence(),
        "gpu_before": _gpu_info(),
        "skipped": [],
    }
    _write_json(Path(args.summary_json), payload)

    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
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
    dtype = torch.bfloat16
    transcoders, config = load_transcoder_from_hub(args.transcoder_ref, device=device, dtype=dtype, lazy_encoder=True, lazy_decoder=True)
    module = model.language_model.layers[args.layer]
    output_weight = model.get_output_embeddings().weight
    payload["transcoder"] = {"type": type(transcoders).__name__, "config_model_kind": config.get("model_kind", "")}

    prompt_rows = _read_csv(Path(args.prompt_runs))
    if args.max_prompt_runs > 0:
        prompt_rows = prompt_rows[: args.max_prompt_runs]
    real_conditions = _parse_csv(args.real_conditions)
    control_conditions = _parse_csv(args.control_conditions)
    rows: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = defaultdict(int)

    for idx, row in enumerate(prompt_rows, start=1):
        try:
            image_path = _image_path(row, args.image_root)
            image = Image.open(image_path).convert("RGB")
            question = _prompt(row.get("question_text", ""), row.get("prompt_name", ""))
            clean_inputs = _qwen_inputs(processor, image, str(image_path), question, args.answer_prefix, device)
            target_candidates = _target_candidates(tokenizer, row.get("answer_text", ""))
            target_ids = [int(item["token_id"]) for item in target_candidates]
            if not target_ids:
                rejection_counts["no_target_token_candidates"] += 1
                payload["skipped"].append({"sample_id": row.get("sample_id"), "prompt_name": row.get("prompt_name"), "reason": "no_target_token_candidates"})
                continue
            clean_outputs, clean_hidden = _forward_with_capture(model, module, clean_inputs)
            clean_score = _rank_and_top(clean_outputs.logits, tokenizer, target_ids)
            if int(clean_score["target_rank"]) > args.max_clean_rank:
                rejection_counts["clean_rank_too_weak"] += 1
                payload["skipped"].append(
                    {
                        "sample_id": row.get("sample_id"),
                        "prompt_name": row.get("prompt_name"),
                        "reason": "clean_rank_too_weak",
                        "clean_rank": clean_score["target_rank"],
                    }
                )
                del clean_outputs, clean_hidden
                torch.cuda.empty_cache()
                continue
            wrong = _top_wrong_token(clean_outputs.logits, tokenizer, int(clean_score["target_token_id"]))
            input_ids = clean_inputs["input_ids"][0].detach().cpu().tolist()
            token_texts = tokenizer.convert_ids_to_tokens(input_ids)
            positions, visual_positions, answer_positions = _position_group(input_ids, token_texts, args.position_group)
            if not positions:
                rejection_counts["empty_position_group"] += 1
                payload["skipped"].append({"sample_id": row.get("sample_id"), "prompt_name": row.get("prompt_name"), "reason": "empty_position_group"})
                del clean_outputs, clean_hidden
                torch.cuda.empty_cache()
                continue

            clean_hidden = clean_hidden.to(device=device, dtype=dtype)
            clean_features = transcoders.encode_layer(clean_hidden, args.layer, apply_activation_function=True).detach()
            mask_features: dict[str, torch.Tensor] = {}
            missing_mask = False
            for condition in real_conditions + control_conditions:
                mask_file = _mask_path(row, args.mask_root, condition)
                if not mask_file.exists():
                    missing_mask = True
                    rejection_counts[f"missing_{condition}"] += 1
                    payload["skipped"].append(
                        {"sample_id": row.get("sample_id"), "prompt_name": row.get("prompt_name"), "reason": f"missing_{condition}", "path": str(mask_file)}
                    )
                    break
                mask = Image.open(mask_file).convert("L").resize(image.size)
                masked_image = _apply_mask(image, mask, (128, 128, 128))
                mask_inputs = _qwen_inputs(processor, masked_image, str(image_path), question, args.answer_prefix, device)
                _mask_outputs, mask_hidden = _forward_with_capture(model, module, mask_inputs)
                mask_features[condition] = transcoders.encode_layer(mask_hidden.to(device=device, dtype=dtype), args.layer, apply_activation_function=True).detach()
                del _mask_outputs, mask_hidden
                torch.cuda.empty_cache()
            if missing_mask:
                del clean_outputs, clean_hidden, clean_features, mask_features
                torch.cuda.empty_cache()
                continue

            selected = _select_evidence_first_candidates(
                clean_features=clean_features,
                mask_features=mask_features,
                positions=positions,
                transcoders=transcoders,
                layer=args.layer,
                target_direction=output_weight[int(clean_score["target_token_id"])].detach().to(device=device, dtype=dtype),
                wrong_direction=output_weight[int(wrong["wrong_token_id"])].detach().to(device=device, dtype=dtype),
                device=device,
                dtype=dtype,
                real_conditions=real_conditions,
                control_conditions=control_conditions,
                candidate_pool_size=args.candidate_pool_size,
                top_per_prompt_run=args.top_per_prompt_run,
                position_chunk_size=args.position_chunk_size,
                min_clean_activation=args.min_clean_activation,
                min_specificity=args.min_specificity,
                min_target_contribution=args.min_target_contribution,
                min_correct_minus_wrong_contribution=args.min_correct_minus_wrong_contribution,
                wrong_contribution_ratio=args.wrong_contribution_ratio,
            )
            if not selected:
                rejection_counts["no_evidence_first_candidates"] += 1
                payload["skipped"].append({"sample_id": row.get("sample_id"), "prompt_name": row.get("prompt_name"), "reason": "no_evidence_first_candidates"})
            for rank, item in enumerate(selected, start=1):
                rows.append(
                    _candidate_base(
                        row=row,
                        selected=item,
                        clean_score=clean_score,
                        wrong=wrong,
                        rank_within_prompt=rank,
                        layer=args.layer,
                        position_group=args.position_group,
                        visual_positions=visual_positions,
                        answer_positions=answer_positions,
                        include_main_rank=args.main_per_prompt_run,
                    )
                )
            if idx % 5 == 0:
                _log(f"prompt-runs processed: {idx}/{len(prompt_rows)} candidates={len(rows)}")
            del clean_outputs, clean_hidden, clean_features, mask_features
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            rejection_counts["exception"] += 1
            payload["skipped"].append({"sample_id": row.get("sample_id"), "prompt_name": row.get("prompt_name"), "reason": repr(exc)})
            torch.cuda.empty_cache()

    rows.sort(key=lambda item: (item.get("sample_id", ""), item.get("prompt_name", ""), int(item.get("candidate_rank_within_prompt_run", 0))))
    for idx, row in enumerate(rows, start=1):
        row["candidate_id"] = f"qwen_evidence_first_{idx:04d}"

    fields = [
        "candidate_id",
        "candidate_source",
        "include_main",
        "include_pool",
        "include_sensitivity_rank10",
        "candidate_rank_within_prompt_run",
        "analysis_group",
        "sample_id",
        "run",
        "prompt_name",
        "source_zeroing_mode",
        "source_node_id",
        "layer",
        "source_pos",
        "source_feature_id",
        "position_group",
        "best_real_condition",
        "path_mass_best",
        "target_token_id",
        "target_token",
        "wrong_token_id",
        "wrong_token",
        "original_target_logit",
        "original_target_rank",
        "original_top1_token",
        "clean_target_logit",
        "clean_target_rank",
        "clean_top1_token",
        "clean_activation",
        "answer_mask_drop",
        "union_mask_drop",
        "shifted_mask_drop",
        "shuffled_mask_drop",
        "real_drop_best",
        "control_drop_max",
        "evidence_specificity",
        "target_contribution",
        "wrong_target_contribution",
        "correct_minus_wrong_contribution",
        "evidence_first_score",
        "damage_score",
        "visual_position_count",
        "answer_adjacent_position_count",
        "known_damaging_feature_ids_for_prompt_run",
        "image_filename",
        "local_image_path",
        "question_text",
        "answer_text",
        "reasoning_operation",
        "image_dependence_tier",
        "paperpack_source",
        "mask_dir",
        "answer_mask_path",
        "union_mask_path",
        "shifted_mask_path",
        "shuffled_mask_path",
    ]
    _write_csv(Path(args.out_csv), rows, fields)
    payload.update(
        {
            "finished_at": _now(),
            "status": "ok" if rows else "empty",
            "candidate_rows": len(rows),
            "main_rows": sum(1 for row in rows if row.get("include_main") == "1"),
            "prompt_runs_with_candidates": len({(row.get("sample_id"), row.get("prompt_name")) for row in rows}),
            "rejection_counts": dict(rejection_counts),
            "gpu_after": _gpu_info(),
        }
    )
    _write_json(Path(args.summary_json), payload)
    _log(f"wrote {len(rows)} candidates to {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
