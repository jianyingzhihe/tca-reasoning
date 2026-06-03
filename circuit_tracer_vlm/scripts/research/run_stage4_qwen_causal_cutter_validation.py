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


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage4-qwen-cutter] {message}", flush=True)


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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _stable_seed(*parts: str) -> int:
    value = 2166136261
    for part in parts:
        for char in str(part):
            value ^= ord(char)
            value = (value * 16777619) % (2**32)
    return value


def _as_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw)) if raw not in (None, "") else default
    except ValueError:
        return default


def _as_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _candidate_rows(path: Path, selection: str, max_candidates: int) -> list[dict[str, str]]:
    rows = []
    for row in _read_csv(path):
        if selection == "main" and row.get("include_main") != "1":
            continue
        if selection == "sensitivity" and row.get("include_sensitivity_rank10") != "1":
            continue
        rows.append(row)
    rows.sort(key=lambda r: int(str(r.get("candidate_id", "0")).split("_")[-1]) if str(r.get("candidate_id", "")).split("_")[-1].isdigit() else 9999)
    if max_candidates > 0:
        rows = rows[:max_candidates]
    return rows


def _image_path(row: dict[str, str], image_root: str) -> Path:
    if image_root:
        return Path(image_root) / Path(row["image_filename"]).name
    raw = row.get("local_image_path") or row.get("image_path") or row.get("image_filename")
    return Path(raw)


def _mask_path(row: dict[str, str], mask_root: str, condition: str) -> Path:
    stem = Path(row["image_filename"]).stem
    name = {
        "answer_mask": "answer.png",
        "union_mask": "union.png",
        "shifted_mask": "shifted.png",
        "shuffled_mask": "shuffled.png",
    }[condition]
    if mask_root:
        return Path(mask_root) / stem / name
    manifest_key = {
        "answer_mask": "answer_mask_path",
        "union_mask": "union_mask_path",
        "shifted_mask": "shifted_mask_path",
        "shuffled_mask": "shuffled_mask_path",
    }[condition]
    return Path(row.get(manifest_key, ""))


def _decoder_vectors(transcoders, layer: int, feature_ids: list[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ids = torch.tensor(feature_ids, device=device, dtype=torch.long)
    vectors = transcoders._get_decoder_vectors(layer, ids)
    if vectors.ndim == 3:
        vectors = vectors[:, 0, :]
    return vectors.to(device=device, dtype=dtype)


def _score_token(logits: torch.Tensor, tokenizer, token_id: int) -> dict[str, Any]:
    return _rank_and_top(logits, tokenizer, [int(token_id)])


def _top_wrong_token(logits: torch.Tensor, tokenizer, target_token_id: int) -> dict[str, Any]:
    row = logits[0, -1].float()
    top_vals, top_ids = torch.topk(row, k=25)
    for token_id, value in zip(top_ids.detach().cpu().tolist(), top_vals.detach().cpu().tolist(), strict=False):
        if int(token_id) == int(target_token_id):
            continue
        token_text = tokenizer.decode([int(token_id)])
        if token_text.strip() == "" or token_text.startswith("<|"):
            continue
        rank = int((row > row[int(token_id)]).sum().item() + 1)
        return {
            "wrong_token_id": int(token_id),
            "wrong_token": token_text,
            "wrong_logit": float(value),
            "wrong_rank": rank,
        }
    fallback = int(top_ids[0].item())
    return {
        "wrong_token_id": fallback,
        "wrong_token": tokenizer.decode([fallback]),
        "wrong_logit": float(top_vals[0].item()),
        "wrong_rank": 1,
    }


def _forward_with_capture(model, module, inputs: dict[str, Any]) -> tuple[Any, torch.Tensor]:
    capture: dict[str, torch.Tensor] = {}

    def _hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        capture["hidden"] = hidden.detach()
        return output

    handle = module.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=False, use_cache=False)
    finally:
        handle.remove()
    if "hidden" not in capture:
        raise RuntimeError("layer hook did not capture hidden state")
    return outputs, capture["hidden"]


def _feature_activation(features: torch.Tensor, pos: int, feature_id: int) -> float:
    if pos < 0 or pos >= features.shape[1] or feature_id < 0 or feature_id >= features.shape[2]:
        return 0.0
    return float(features[0, pos, feature_id].float().item())


def _choose_controls(
    *,
    features: torch.Tensor,
    transcoders,
    layer: int,
    source_pos: int,
    source_feature_id: int,
    source_zeroing_mode: str,
    target_direction: torch.Tensor,
    allowed_positions: list[int],
    excluded_features: set[int],
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, dict[str, Any]]:
    n_features = int(features.shape[2])
    pos = max(0, min(source_pos, int(features.shape[1]) - 1))
    values = features[0, pos].float()
    active_count = min(512, n_features)
    active_vals, active_ids_tensor = torch.topk(values.abs(), k=active_count)
    active_ids = [
        int(fid)
        for fid, val in zip(active_ids_tensor.detach().cpu().tolist(), active_vals.detach().cpu().tolist(), strict=False)
        if float(val) > 0 and int(fid) != int(source_feature_id) and int(fid) not in excluded_features
    ]
    if not active_ids:
        active_ids = [
            int(fid)
            for fid, val in zip(active_ids_tensor.detach().cpu().tolist(), active_vals.detach().cpu().tolist(), strict=False)
            if float(val) > 0 and int(fid) != int(source_feature_id)
        ]
    source_vec = _decoder_vectors(transcoders, layer, [source_feature_id], device, dtype)[0]
    source_activation = _feature_activation(features, pos, source_feature_id)
    source_direct = float(source_activation * torch.dot(source_vec.float(), target_direction.float()).item())

    matched_feature = None
    if active_ids:
        vectors = _decoder_vectors(transcoders, layer, active_ids, device, dtype)
        target = target_direction.to(device=vectors.device, dtype=vectors.dtype)
        contrib = torch.mv(vectors.float(), target.float()).detach().cpu().tolist()
        scored: list[tuple[float, int, float, float]] = []
        for idx, feat_id in enumerate(active_ids):
            activation = _feature_activation(features, pos, feat_id)
            direct = float(activation * contrib[idx])
            sign_penalty = 0.0 if source_direct == 0 or direct == 0 or (source_direct > 0) == (direct > 0) else 10.0
            score = (
                abs(activation - source_activation) / (abs(source_activation) + 1e-4)
                + abs(direct - source_direct) / (abs(source_direct) + 1e-4)
                + sign_penalty
            )
            scored.append((score, int(feat_id), activation, direct))
        scored.sort(key=lambda item: item[0])
        if scored:
            _score, feat_id, activation, direct = scored[0]
            matched_feature = {
                "feature_id": feat_id,
                "pos": pos,
                "activation": activation,
                "direct_effect": direct,
            }

    rng = random.Random(seed)
    random_feature = None
    random_pool = [fid for fid in active_ids if matched_feature is None or fid != matched_feature["feature_id"]]
    if random_pool:
        feat_id = rng.choice(random_pool)
        random_feature = {
            "feature_id": int(feat_id),
            "pos": pos,
            "activation": _feature_activation(features, pos, int(feat_id)),
            "direct_effect": "",
        }

    allowed = [p for p in sorted(set(int(p) for p in allowed_positions)) if 0 <= p < features.shape[1] and p != pos]
    same_feature_positions = [
        (abs(_feature_activation(features, p, source_feature_id) - source_activation), p)
        for p in allowed
        if _feature_activation(features, p, source_feature_id) > 0
    ]
    if not same_feature_positions:
        same_feature_positions = [(0.0, p) for p in allowed]
    same_feature_position = None
    if same_feature_positions:
        same_feature_positions.sort(key=lambda item: item[0])
        _, control_pos = same_feature_positions[0]
        same_feature_position = {
            "feature_id": source_feature_id,
            "pos": int(control_pos),
            "activation": _feature_activation(features, int(control_pos), source_feature_id),
            "direct_effect": "",
        }

    controls: dict[str, dict[str, Any]] = {
        "source": {
            "feature_id": int(source_feature_id),
            "pos": int(pos),
            "activation": source_activation,
            "direct_effect": source_direct,
            "zeroing_mode": source_zeroing_mode,
        }
    }
    if matched_feature:
        matched_feature["zeroing_mode"] = source_zeroing_mode
        controls["same_position_matched_feature_control"] = matched_feature
    if same_feature_position:
        same_feature_position["zeroing_mode"] = source_zeroing_mode
        controls["same_feature_random_position_control"] = same_feature_position
    if random_feature:
        random_feature["zeroing_mode"] = source_zeroing_mode
        controls["random_active_feature_control"] = random_feature
    return controls


def _score_with_zeroing(
    *,
    model,
    inputs: dict[str, Any],
    tokenizer,
    transcoders,
    module,
    layer: int,
    pos: int,
    feature_id: int,
    token_ids: list[int],
    scale: float,
    zeroing_mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, dict[str, Any]]:
    vector = _decoder_vectors(transcoders, layer, [feature_id], device, dtype)[0]

    def _hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if pos < 0 or pos >= hidden.shape[1]:
            return output
        with torch.no_grad():
            acts = transcoders.encode_layer(hidden.to(device), layer, apply_activation_function=True)
            activation = acts[:, pos, int(feature_id)].to(hidden.device, dtype=hidden.dtype)
            patch = activation[:, None] * vector.to(hidden.device, dtype=hidden.dtype)[None, :]
            new_hidden = hidden.clone()
            if zeroing_mode == "subtract":
                new_hidden[:, pos, :] = new_hidden[:, pos, :] - scale * patch
            elif zeroing_mode == "add":
                new_hidden[:, pos, :] = new_hidden[:, pos, :] + scale * patch
            else:
                raise ValueError(f"unknown zeroing_mode: {zeroing_mode}")
        return _replace_hidden(output, new_hidden)

    handle = module.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=False, use_cache=False)
    finally:
        handle.remove()
    return {int(token_id): _score_token(outputs.logits, tokenizer, int(token_id)) for token_id in token_ids}


def _score_with_restore(
    *,
    model,
    inputs: dict[str, Any],
    tokenizer,
    transcoders,
    module,
    layer: int,
    pos: int,
    feature_id: int,
    clean_features: torch.Tensor,
    mask_features: torch.Tensor,
    token_ids: list[int],
    scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, dict[str, Any]]:
    vector = _decoder_vectors(transcoders, layer, [feature_id], device, dtype)[0]
    if pos < 0 or pos >= clean_features.shape[1] or pos >= mask_features.shape[1]:
        drop = torch.zeros((1,), device=device, dtype=dtype)
    else:
        drop = (clean_features[:, pos, int(feature_id)] - mask_features[:, pos, int(feature_id)]).to(device=device, dtype=dtype)

    def _hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if pos < 0 or pos >= hidden.shape[1]:
            return output
        with torch.no_grad():
            patch = drop.to(hidden.device, dtype=hidden.dtype)[:, None] * vector.to(hidden.device, dtype=hidden.dtype)[None, :]
            new_hidden = hidden.clone()
            new_hidden[:, pos, :] = new_hidden[:, pos, :] + scale * patch
        return _replace_hidden(output, new_hidden)

    handle = module.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=False, use_cache=False)
    finally:
        handle.remove()
    return {int(token_id): _score_token(outputs.logits, tokenizer, int(token_id)) for token_id in token_ids}


def _effect_row(
    *,
    candidate: dict[str, str],
    control_group: str,
    control: dict[str, Any],
    intervention_kind: str,
    mask_condition: str,
    token_scored: str,
    token_id: int,
    before: dict[str, Any],
    after: dict[str, Any],
    clean_target_logit: float,
    clean_target_rank: int,
    mask_target_logit: float | str = "",
    mask_target_rank: int | str = "",
    clean_feature_activation: float | str = "",
    mask_feature_activation: float | str = "",
    activation_drop: float | str = "",
) -> dict[str, Any]:
    if intervention_kind == "clean_zeroing":
        logit_effect = float(before["target_logit"]) - float(after["target_logit"])
        rank_effect = int(after["target_rank"]) - int(before["target_rank"])
        gap_closure = ""
    else:
        logit_effect = float(after["target_logit"]) - float(before["target_logit"])
        rank_effect = int(before["target_rank"]) - int(after["target_rank"])
        gap = clean_target_logit - float(before["target_logit"])
        gap_closure = logit_effect / gap if abs(gap) > 1e-6 else ""
    return {
        "candidate_id": candidate.get("candidate_id", ""),
        "include_main": candidate.get("include_main", ""),
        "analysis_group": candidate.get("analysis_group", ""),
        "sample_id": candidate.get("sample_id", ""),
        "run": candidate.get("run", ""),
        "prompt_name": candidate.get("prompt_name", ""),
        "reasoning_operation": candidate.get("reasoning_operation", ""),
        "image_dependence_tier": candidate.get("image_dependence_tier", ""),
        "source_node_id": candidate.get("source_node_id", ""),
        "source_feature_id": candidate.get("source_feature_id", ""),
        "source_pos": candidate.get("source_pos", ""),
        "source_zeroing_mode": candidate.get("source_zeroing_mode", ""),
        "target_answer": candidate.get("answer_text", ""),
        "target_token_id": candidate.get("target_token_id", ""),
        "target_token": candidate.get("target_token", ""),
        "wrong_token_id": candidate.get("_wrong_token_id", ""),
        "wrong_token": candidate.get("_wrong_token", ""),
        "intervention_kind": intervention_kind,
        "mask_condition": mask_condition,
        "control_group": control_group,
        "control_feature_id": control.get("feature_id", ""),
        "control_pos": control.get("pos", ""),
        "control_activation": control.get("activation", ""),
        "control_direct_effect": control.get("direct_effect", ""),
        "clean_feature_activation": clean_feature_activation,
        "mask_feature_activation": mask_feature_activation,
        "activation_drop": activation_drop,
        "token_scored": token_scored,
        "scored_token_id": token_id,
        "before_logit": before["target_logit"],
        "after_logit": after["target_logit"],
        "logit_effect": logit_effect,
        "before_rank": before["target_rank"],
        "after_rank": after["target_rank"],
        "rank_effect": rank_effect,
        "gap_closure": gap_closure,
        "clean_target_logit": clean_target_logit,
        "clean_target_rank": clean_target_rank,
        "mask_target_logit": mask_target_logit,
        "mask_target_rank": mask_target_rank,
        "status": "ok",
        "error_message": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Qwen-native causal cutter candidates with matched controls.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--transcoder-ref", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--selection", choices=["main", "sensitivity", "all"], default="main")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--mask-root", default="")
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--layer", type=int, default=26)
    parser.add_argument("--answer-prefix", default="The answer is ")
    parser.add_argument("--mask-conditions", default="answer_mask,union_mask,shifted_mask,shuffled_mask")
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--min-gpu-free-gb", type=float, default=12.0)
    parser.add_argument("--checkpoint-every", type=int, default=30)
    args = parser.parse_args()

    candidates = _candidate_rows(Path(args.candidate_manifest), args.selection, args.max_candidates)
    payload: dict[str, Any] = {
        "created_at": _now(),
        "model_name": args.model_name,
        "transcoder_ref": args.transcoder_ref,
        "candidate_manifest": args.candidate_manifest,
        "selection": args.selection,
        "layer": args.layer,
        "requested_candidates": len(candidates),
        "env_presence": _env_presence(),
        "gpu_before": _gpu_info(),
        "skipped": [],
        "claim_boundary": (
            "Qwen-native causal cutter validation. Candidates come from Qwen top8 intervention rows; "
            "no Gemma nodes or feature maps are used."
        ),
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
    transcoders, config = load_transcoder_from_hub(
        args.transcoder_ref,
        device=device,
        dtype=dtype,
        lazy_encoder=True,
        lazy_decoder=True,
    )
    module = model.language_model.layers[args.layer]
    output_weight = model.get_output_embeddings().weight
    payload["transcoder"] = {"type": type(transcoders).__name__, "config_model_kind": config.get("model_kind", "")}

    mask_conditions = _parse_csv(args.mask_conditions)
    rows: list[dict[str, Any]] = []
    usable_candidates = 0

    def _checkpoint() -> None:
        if not rows:
            return
        _write_csv(Path(args.out_csv), rows, list(rows[0].keys()))
        payload["gpu_after"] = _gpu_info()
        payload["decision"] = {
            "status": "running_checkpoint",
            "usable_candidates": usable_candidates,
            "requested_candidates": len(candidates),
            "raw_rows": len(rows),
            "skipped_count": len(payload["skipped"]),
        }
        _write_json(Path(args.summary_json), payload)

    for idx, candidate in enumerate(candidates, start=1):
        sample_id = candidate.get("sample_id", "")
        prompt_name = candidate.get("prompt_name", "")
        try:
            image_path = _image_path(candidate, args.image_root)
            image = Image.open(image_path).convert("RGB")
            question = _prompt(candidate.get("question_text", ""), prompt_name)
            clean_inputs = _qwen_inputs(processor, image, str(image_path), question, args.answer_prefix, device)
            clean_outputs, clean_hidden = _forward_with_capture(model, module, clean_inputs)
            clean_hidden = clean_hidden.to(device=device, dtype=dtype)
            clean_features = transcoders.encode_layer(clean_hidden, args.layer, apply_activation_function=True).detach()
            target_token_id = _as_int(candidate.get("target_token_id"))
            clean_target_score = _score_token(clean_outputs.logits, tokenizer, target_token_id)
            wrong = _top_wrong_token(clean_outputs.logits, tokenizer, target_token_id)
            wrong_token_id = int(wrong["wrong_token_id"])
            candidate["_wrong_token_id"] = wrong_token_id
            candidate["_wrong_token"] = wrong["wrong_token"]
            clean_wrong_score = _score_token(clean_outputs.logits, tokenizer, wrong_token_id)

            input_ids = clean_inputs["input_ids"][0].detach().cpu().tolist()
            token_texts = tokenizer.convert_ids_to_tokens(input_ids)
            visual_positions = _qwen_bucket_positions(input_ids, token_texts)["image_marker_or_span"]
            answer_positions = _answer_adjacent_positions(len(input_ids), set(visual_positions), 4)
            allowed_positions = sorted(set(visual_positions + answer_positions))
            source_pos = _as_int(candidate.get("source_pos"), -1)
            source_feature_id = _as_int(candidate.get("source_feature_id"), -1)
            if source_pos < 0 or source_pos >= clean_hidden.shape[1] or source_feature_id < 0:
                payload["skipped"].append({"candidate_id": candidate.get("candidate_id"), "reason": "source_out_of_range"})
                continue
            excluded = {
                _as_int(part, -1)
                for part in str(candidate.get("known_damaging_feature_ids_for_prompt_run", "")).split("|")
                if part.strip()
            }
            controls = _choose_controls(
                features=clean_features,
                transcoders=transcoders,
                layer=args.layer,
                source_pos=source_pos,
                source_feature_id=source_feature_id,
                source_zeroing_mode=candidate.get("source_zeroing_mode", "subtract"),
                target_direction=output_weight[target_token_id].detach().to(device=device, dtype=dtype),
                allowed_positions=allowed_positions,
                excluded_features=excluded,
                seed=_stable_seed(candidate.get("candidate_id", ""), sample_id, prompt_name),
                device=device,
                dtype=dtype,
            )
            if len(controls) < 2:
                payload["skipped"].append({"candidate_id": candidate.get("candidate_id"), "reason": "insufficient_controls"})
                continue

            for control_group, control in controls.items():
                clean_feature_activation = _feature_activation(
                    clean_features,
                    int(control["pos"]),
                    int(control["feature_id"]),
                )
                token_scores_after = _score_with_zeroing(
                    model=model,
                    inputs=clean_inputs,
                    tokenizer=tokenizer,
                    transcoders=transcoders,
                    module=module,
                    layer=args.layer,
                    pos=int(control["pos"]),
                    feature_id=int(control["feature_id"]),
                    token_ids=[target_token_id, wrong_token_id],
                    scale=args.scale,
                    zeroing_mode=str(control.get("zeroing_mode") or candidate.get("source_zeroing_mode") or "subtract"),
                    device=device,
                    dtype=dtype,
                )
                rows.append(
                    _effect_row(
                        candidate=candidate,
                        control_group=control_group,
                        control=control,
                        intervention_kind="clean_zeroing",
                        mask_condition="clean",
                        token_scored="target",
                        token_id=target_token_id,
                        before=clean_target_score,
                        after=token_scores_after[target_token_id],
                        clean_target_logit=float(clean_target_score["target_logit"]),
                        clean_target_rank=int(clean_target_score["target_rank"]),
                        clean_feature_activation=clean_feature_activation,
                    )
                )
                rows.append(
                    _effect_row(
                        candidate=candidate,
                        control_group=control_group,
                        control=control,
                        intervention_kind="clean_zeroing",
                        mask_condition="clean",
                        token_scored="wrong",
                        token_id=wrong_token_id,
                        before=clean_wrong_score,
                        after=token_scores_after[wrong_token_id],
                        clean_target_logit=float(clean_target_score["target_logit"]),
                        clean_target_rank=int(clean_target_score["target_rank"]),
                        clean_feature_activation=clean_feature_activation,
                    )
                )

            for mask_condition in mask_conditions:
                mask_file = _mask_path(candidate, args.mask_root, mask_condition)
                if not mask_file.exists():
                    payload["skipped"].append(
                        {"candidate_id": candidate.get("candidate_id"), "mask_condition": mask_condition, "reason": "mask_missing"}
                    )
                    continue
                mask = Image.open(mask_file).convert("L").resize(image.size)
                mask_image = _apply_mask(image, mask, (128, 128, 128))
                mask_inputs = _qwen_inputs(processor, mask_image, str(image_path), question, args.answer_prefix, device)
                mask_outputs, mask_hidden = _forward_with_capture(model, module, mask_inputs)
                mask_hidden = mask_hidden.to(device=device, dtype=dtype)
                mask_features = transcoders.encode_layer(mask_hidden, args.layer, apply_activation_function=True).detach()
                mask_target_score = _score_token(mask_outputs.logits, tokenizer, target_token_id)
                mask_wrong_score = _score_token(mask_outputs.logits, tokenizer, wrong_token_id)
                for control_group, control in controls.items():
                    clean_feature_activation = _feature_activation(
                        clean_features,
                        int(control["pos"]),
                        int(control["feature_id"]),
                    )
                    mask_feature_activation = _feature_activation(
                        mask_features,
                        int(control["pos"]),
                        int(control["feature_id"]),
                    )
                    activation_drop = clean_feature_activation - mask_feature_activation
                    token_scores_after = _score_with_restore(
                        model=model,
                        inputs=mask_inputs,
                        tokenizer=tokenizer,
                        transcoders=transcoders,
                        module=module,
                        layer=args.layer,
                        pos=int(control["pos"]),
                        feature_id=int(control["feature_id"]),
                        clean_features=clean_features,
                        mask_features=mask_features,
                        token_ids=[target_token_id, wrong_token_id],
                        scale=args.scale,
                        device=device,
                        dtype=dtype,
                    )
                    rows.append(
                        _effect_row(
                            candidate=candidate,
                            control_group=control_group,
                            control=control,
                            intervention_kind="mask_restore",
                            mask_condition=mask_condition,
                            token_scored="target",
                            token_id=target_token_id,
                            before=mask_target_score,
                            after=token_scores_after[target_token_id],
                            clean_target_logit=float(clean_target_score["target_logit"]),
                            clean_target_rank=int(clean_target_score["target_rank"]),
                            mask_target_logit=float(mask_target_score["target_logit"]),
                            mask_target_rank=int(mask_target_score["target_rank"]),
                            clean_feature_activation=clean_feature_activation,
                            mask_feature_activation=mask_feature_activation,
                            activation_drop=activation_drop,
                        )
                    )
                    rows.append(
                        _effect_row(
                            candidate=candidate,
                            control_group=control_group,
                            control=control,
                            intervention_kind="mask_restore",
                            mask_condition=mask_condition,
                            token_scored="wrong",
                            token_id=wrong_token_id,
                            before=mask_wrong_score,
                            after=token_scores_after[wrong_token_id],
                            clean_target_logit=float(clean_target_score["target_logit"]),
                            clean_target_rank=int(clean_target_score["target_rank"]),
                            mask_target_logit=float(mask_target_score["target_logit"]),
                            mask_target_rank=int(mask_target_score["target_rank"]),
                            clean_feature_activation=clean_feature_activation,
                            mask_feature_activation=mask_feature_activation,
                            activation_drop=activation_drop,
                        )
                    )
                del mask_outputs, mask_hidden, mask_features
                torch.cuda.empty_cache()

            usable_candidates += 1
            if idx % 3 == 0:
                _log(f"validated candidates: {idx}/{len(candidates)} rows={len(rows)}")
            if args.checkpoint_every > 0 and usable_candidates % args.checkpoint_every == 0:
                _checkpoint()
            del clean_outputs, clean_hidden, clean_features
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            payload["skipped"].append({"candidate_id": candidate.get("candidate_id"), "sample_id": sample_id, "reason": repr(exc)})
            torch.cuda.empty_cache()

    fieldnames = [
        "candidate_id",
        "include_main",
        "analysis_group",
        "sample_id",
        "run",
        "prompt_name",
        "reasoning_operation",
        "image_dependence_tier",
        "source_node_id",
        "source_feature_id",
        "source_pos",
        "source_zeroing_mode",
        "target_answer",
        "target_token_id",
        "target_token",
        "wrong_token_id",
        "wrong_token",
        "intervention_kind",
        "mask_condition",
        "control_group",
        "control_feature_id",
        "control_pos",
        "control_activation",
        "control_direct_effect",
        "clean_feature_activation",
        "mask_feature_activation",
        "activation_drop",
        "token_scored",
        "scored_token_id",
        "before_logit",
        "after_logit",
        "logit_effect",
        "before_rank",
        "after_rank",
        "rank_effect",
        "gap_closure",
        "clean_target_logit",
        "clean_target_rank",
        "mask_target_logit",
        "mask_target_rank",
        "status",
        "error_message",
    ]
    _write_csv(Path(args.out_csv), rows, fieldnames)
    payload["gpu_after"] = _gpu_info()
    payload["decision"] = {
        "status": "ok" if rows else "blocked_no_rows",
        "usable_candidates": usable_candidates,
        "requested_candidates": len(candidates),
        "raw_rows": len(rows),
        "skipped_count": len(payload["skipped"]),
    }
    _write_json(Path(args.summary_json), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
