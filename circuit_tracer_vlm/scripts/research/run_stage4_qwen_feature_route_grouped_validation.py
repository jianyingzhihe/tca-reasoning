#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
    _replace_hidden,
)
from run_cross_model_hidden_position_patch_smoke import _answer_adjacent_positions
from run_stage4_qwen_causal_cutter_validation import (
    _as_float,
    _as_int,
    _choose_controls,
    _decoder_vectors,
    _feature_activation,
    _image_path,
    _mask_path,
    _score_token,
    _stable_seed,
    _top_wrong_token,
    _write_csv,
    _write_json,
)


CONTROL_GROUPS = [
    "source",
    "same_size_matched_feature_route_control",
    "same_feature_random_position_route_control",
    "random_active_route_control",
]
SINGLE_NODE_TO_ROUTE_CONTROL = {
    "source": "source",
    "same_position_matched_feature_control": "same_size_matched_feature_route_control",
    "same_feature_random_position_control": "same_feature_random_position_route_control",
    "random_active_feature_control": "random_active_route_control",
}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage4-qwen-feature-route] {message}", flush=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_pipe_ints(raw: str) -> list[int]:
    out: list[int] = []
    for part in str(raw or "").split("|"):
        part = part.strip()
        if not part:
            continue
        out.append(_as_int(part, -1))
    return out


def _route_rows(path: Path, topks: set[int], max_routes: int) -> list[dict[str, str]]:
    rows = [row for row in _read_csv(path) if _as_int(row.get("topk"), -1) in topks]
    rows.sort(key=lambda row: (row.get("sample_id", ""), row.get("prompt_name", ""), _as_int(row.get("topk"), 999)))
    if max_routes > 0:
        rows = rows[:max_routes]
    return rows


def _route_nodes(route: dict[str, str]) -> list[dict[str, Any]]:
    layers = _parse_pipe_ints(route.get("node_layers", ""))
    positions = _parse_pipe_ints(route.get("node_source_positions", ""))
    feature_ids = _parse_pipe_ints(route.get("node_feature_ids", ""))
    modes = [part.strip() or "subtract" for part in str(route.get("node_zeroing_modes") or "").split("|")]
    candidate_ids = [part.strip() for part in str(route.get("node_candidate_ids") or "").split("|")]
    n = min(len(layers), len(positions), len(feature_ids))
    nodes: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for idx in range(n):
        layer = layers[idx]
        pos = positions[idx]
        feature_id = feature_ids[idx]
        if layer < 0 or pos < 0 or feature_id < 0:
            continue
        key = (layer, pos, feature_id)
        if key in seen:
            continue
        seen.add(key)
        nodes.append(
            {
                "layer": layer,
                "pos": pos,
                "feature_id": feature_id,
                "zeroing_mode": modes[idx] if idx < len(modes) else "subtract",
                "candidate_id": candidate_ids[idx] if idx < len(candidate_ids) else "",
            }
        )
    return nodes


def _forward_with_multi_capture(model, modules: dict[int, Any], inputs: dict[str, Any]) -> tuple[Any, dict[int, torch.Tensor]]:
    capture: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer: int):
        def _hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            capture[layer] = hidden.detach()
            return output

        return _hook

    for layer, module in modules.items():
        handles.append(module.register_forward_hook(make_hook(layer)))
    try:
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=False, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    missing = sorted(set(modules) - set(capture))
    if missing:
        raise RuntimeError(f"layer hooks did not capture hidden states: {missing}")
    return outputs, capture


def _features_by_layer(transcoders, hidden_by_layer: dict[int, torch.Tensor], device: torch.device, dtype: torch.dtype) -> dict[int, torch.Tensor]:
    out: dict[int, torch.Tensor] = {}
    for layer, hidden in hidden_by_layer.items():
        out[layer] = transcoders.encode_layer(
            hidden.to(device=device, dtype=dtype),
            layer,
            apply_activation_function=True,
        ).detach()
    return out


def _route_activation_sum(features_by_layer: dict[int, torch.Tensor], route_controls: list[dict[str, Any]]) -> float:
    total = 0.0
    for control in route_controls:
        features = features_by_layer.get(int(control["layer"]))
        if features is None:
            continue
        total += _feature_activation(features, int(control["pos"]), int(control["feature_id"]))
    return total


def _build_control_routes(
    *,
    nodes: list[dict[str, Any]],
    clean_features_by_layer: dict[int, torch.Tensor],
    transcoders,
    target_direction: torch.Tensor,
    allowed_positions: list[int],
    route: dict[str, str],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node_idx, node in enumerate(nodes):
        layer = int(node["layer"])
        features = clean_features_by_layer[layer]
        controls = _choose_controls(
            features=features,
            transcoders=transcoders,
            layer=layer,
            source_pos=int(node["pos"]),
            source_feature_id=int(node["feature_id"]),
            source_zeroing_mode=str(node.get("zeroing_mode") or "subtract"),
            target_direction=target_direction,
            allowed_positions=allowed_positions,
            excluded_features=set(),
            seed=_stable_seed(route.get("route_id", ""), route.get("sample_id", ""), str(node_idx)),
            device=device,
            dtype=dtype,
        )
        for single_name, control in controls.items():
            route_name = SINGLE_NODE_TO_ROUTE_CONTROL.get(single_name)
            if not route_name:
                continue
            enriched = dict(control)
            enriched["layer"] = layer
            enriched["source_candidate_id"] = node.get("candidate_id", "")
            enriched["source_pos"] = node.get("pos", "")
            enriched["source_feature_id"] = node.get("feature_id", "")
            grouped[route_name].append(enriched)
    return {name: controls for name, controls in grouped.items() if len(controls) == len(nodes)}


def _score_with_route_patch(
    *,
    model,
    inputs: dict[str, Any],
    tokenizer,
    transcoders,
    modules: dict[int, Any],
    route_controls: list[dict[str, Any]],
    token_ids: list[int],
    scale: float,
    patch_kind: str,
    clean_features_by_layer: dict[int, torch.Tensor] | None,
    mask_features_by_layer: dict[int, torch.Tensor] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, dict[str, Any]]:
    controls_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for control in route_controls:
        controls_by_layer[int(control["layer"])].append(control)

    vectors_by_layer: dict[int, torch.Tensor] = {}
    feature_ids_by_layer: dict[int, list[int]] = {}
    for layer, controls in controls_by_layer.items():
        feature_ids = [int(control["feature_id"]) for control in controls]
        feature_ids_by_layer[layer] = feature_ids
        vectors_by_layer[layer] = _decoder_vectors(transcoders, layer, feature_ids, device, dtype)

    drops_by_layer: dict[int, list[torch.Tensor]] = {}
    if patch_kind in {"restore", "corrupt"}:
        assert clean_features_by_layer is not None
        assert mask_features_by_layer is not None
        for layer, controls in controls_by_layer.items():
            clean_features = clean_features_by_layer[layer]
            mask_features = mask_features_by_layer[layer]
            drops: list[torch.Tensor] = []
            for control in controls:
                pos = int(control["pos"])
                feature_id = int(control["feature_id"])
                if pos < 0 or pos >= clean_features.shape[1] or pos >= mask_features.shape[1]:
                    drop = torch.zeros((1,), device=device, dtype=dtype)
                else:
                    drop = (clean_features[:, pos, feature_id] - mask_features[:, pos, feature_id]).to(device=device, dtype=dtype)
                drops.append(drop)
            drops_by_layer[layer] = drops

    handles = []

    def make_hook(layer: int):
        def _hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            controls = controls_by_layer[layer]
            if not controls:
                return output
            with torch.no_grad():
                new_hidden = hidden.clone()
                vectors = vectors_by_layer[layer].to(device=hidden.device, dtype=hidden.dtype)
                if patch_kind == "zeroing":
                    acts = transcoders.encode_layer(
                        hidden.to(device=device, dtype=dtype),
                        layer,
                        apply_activation_function=True,
                    )
                else:
                    acts = None
                for idx, control in enumerate(controls):
                    pos = int(control["pos"])
                    feature_id = int(control["feature_id"])
                    if pos < 0 or pos >= hidden.shape[1]:
                        continue
                    vector = vectors[idx]
                    if patch_kind == "zeroing":
                        assert acts is not None
                        activation = acts[:, pos, feature_id].to(hidden.device, dtype=hidden.dtype)
                        sign = -1.0 if str(control.get("zeroing_mode") or "subtract") == "subtract" else 1.0
                        patch = sign * scale * activation[:, None] * vector[None, :]
                    elif patch_kind == "restore":
                        drop = drops_by_layer[layer][idx].to(hidden.device, dtype=hidden.dtype)
                        patch = scale * drop[:, None] * vector[None, :]
                    elif patch_kind == "corrupt":
                        drop = drops_by_layer[layer][idx].to(hidden.device, dtype=hidden.dtype)
                        patch = -scale * drop[:, None] * vector[None, :]
                    else:
                        raise ValueError(f"unknown patch_kind: {patch_kind}")
                    new_hidden[:, pos, :] = new_hidden[:, pos, :] + patch
            return _replace_hidden(output, new_hidden)

        return _hook

    for layer, module in modules.items():
        if layer in controls_by_layer:
            handles.append(module.register_forward_hook(make_hook(layer)))
    try:
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=False, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return {int(token_id): _score_token(outputs.logits, tokenizer, int(token_id)) for token_id in token_ids}


def _effect_row(
    *,
    route: dict[str, str],
    control_group: str,
    route_controls: list[dict[str, Any]],
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
    route_clean_activation_sum: float | str = "",
    route_mask_activation_sum: float | str = "",
    route_activation_drop: float | str = "",
) -> dict[str, Any]:
    if intervention_kind in {"clean_route_zeroing", "route_corrupt"}:
        logit_effect = float(before["target_logit"]) - float(after["target_logit"])
        rank_effect = int(after["target_rank"]) - int(before["target_rank"])
        gap_closure = ""
    else:
        logit_effect = float(after["target_logit"]) - float(before["target_logit"])
        rank_effect = int(before["target_rank"]) - int(after["target_rank"])
        gap = clean_target_logit - float(before["target_logit"])
        gap_closure = logit_effect / gap if abs(gap) > 1e-6 else ""
    return {
        "route_id": route.get("route_id", ""),
        "route_base_id": route.get("route_base_id", ""),
        "pack": route.get("pack", ""),
        "sample_id": route.get("sample_id", ""),
        "prompt_name": route.get("prompt_name", ""),
        "topk": route.get("topk", ""),
        "best_real_condition": route.get("best_real_condition", ""),
        "route_node_count": route.get("route_node_count", ""),
        "primary_node_count": route.get("primary_node_count", ""),
        "strict_missing_count": route.get("strict_missing_count", ""),
        "strict_missing_fraction": route.get("strict_missing_fraction", ""),
        "node_candidate_ids": route.get("node_candidate_ids", ""),
        "node_layers": route.get("node_layers", ""),
        "node_source_positions": route.get("node_source_positions", ""),
        "node_feature_ids": route.get("node_feature_ids", ""),
        "target_answer": route.get("answer_text", ""),
        "target_token_id": route.get("target_token_id", ""),
        "target_token": route.get("target_token", ""),
        "wrong_token_id": route.get("_wrong_token_id", route.get("wrong_token_id", "")),
        "wrong_token": route.get("_wrong_token", route.get("wrong_token", "")),
        "intervention_kind": intervention_kind,
        "mask_condition": mask_condition,
        "control_group": control_group,
        "control_route_node_count": len(route_controls),
        "control_layers": "|".join(str(control.get("layer", "")) for control in route_controls),
        "control_positions": "|".join(str(control.get("pos", "")) for control in route_controls),
        "control_feature_ids": "|".join(str(control.get("feature_id", "")) for control in route_controls),
        "route_clean_activation_sum": route_clean_activation_sum,
        "route_mask_activation_sum": route_mask_activation_sum,
        "route_activation_drop": route_activation_drop,
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
    parser = argparse.ArgumentParser(description="Validate Qwen grouped feature-route interventions.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--transcoder-ref", required=True)
    parser.add_argument("--route-manifest", required=True)
    parser.add_argument("--image-root", default="")
    parser.add_argument("--mask-root", default="")
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--answer-prefix", default="The answer is ")
    parser.add_argument("--mask-conditions", default="answer_mask,union_mask,shifted_mask,shuffled_mask")
    parser.add_argument("--topks", default="4,8,16,32,64")
    parser.add_argument("--max-routes", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--include-route-corrupt", action="store_true")
    parser.add_argument("--min-gpu-free-gb", type=float, default=12.0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()

    topks = {_as_int(part, -1) for part in _parse_csv(args.topks)}
    routes = _route_rows(Path(args.route_manifest), topks, args.max_routes)
    payload: dict[str, Any] = {
        "created_at": _now(),
        "model_name": args.model_name,
        "transcoder_ref": args.transcoder_ref,
        "route_manifest": args.route_manifest,
        "requested_routes": len(routes),
        "topks": sorted(topks),
        "env_presence": _env_presence(),
        "gpu_before": _gpu_info(),
        "skipped": [],
        "claim_boundary": (
            "Qwen-native grouped feature-route validation. Routes are built from Stage4-060 "
            "Qwen route-first candidates; no Gemma node ids or route maps are used."
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
    output_weight = model.get_output_embeddings().weight
    payload["transcoder"] = {"type": type(transcoders).__name__, "config_model_kind": config.get("model_kind", "")}

    mask_conditions = _parse_csv(args.mask_conditions)
    rows: list[dict[str, Any]] = []
    usable_routes = 0

    def _checkpoint() -> None:
        if not rows:
            return
        _write_csv(Path(args.out_csv), rows, list(rows[0].keys()))
        payload["gpu_after"] = _gpu_info()
        payload["decision"] = {
            "status": "running_checkpoint",
            "usable_routes": usable_routes,
            "requested_routes": len(routes),
            "raw_rows": len(rows),
            "skipped_count": len(payload["skipped"]),
        }
        _write_json(Path(args.summary_json), payload)

    for idx, route in enumerate(routes, start=1):
        route_id = route.get("route_id", "")
        sample_id = route.get("sample_id", "")
        prompt_name = route.get("prompt_name", "")
        try:
            nodes = _route_nodes(route)
            if not nodes:
                payload["skipped"].append({"route_id": route_id, "reason": "empty_route_nodes"})
                continue
            route_layers = sorted({int(node["layer"]) for node in nodes})
            modules = {layer: model.language_model.layers[layer] for layer in route_layers}
            image_path = _image_path(route, args.image_root)
            image = Image.open(image_path).convert("RGB")
            question = _prompt(route.get("question_text", ""), prompt_name)
            clean_inputs = _qwen_inputs(processor, image, str(image_path), question, args.answer_prefix, device)
            clean_outputs, clean_hidden_by_layer = _forward_with_multi_capture(model, modules, clean_inputs)
            clean_features_by_layer = _features_by_layer(transcoders, clean_hidden_by_layer, device, dtype)
            target_token_id = _as_int(route.get("target_token_id"))
            clean_target_score = _score_token(clean_outputs.logits, tokenizer, target_token_id)
            wrong_token_id = _as_int(route.get("wrong_token_id"), -1)
            if wrong_token_id < 0:
                wrong = _top_wrong_token(clean_outputs.logits, tokenizer, target_token_id)
                wrong_token_id = int(wrong["wrong_token_id"])
                route["_wrong_token_id"] = wrong_token_id
                route["_wrong_token"] = wrong["wrong_token"]
            clean_wrong_score = _score_token(clean_outputs.logits, tokenizer, wrong_token_id)

            input_ids = clean_inputs["input_ids"][0].detach().cpu().tolist()
            token_texts = tokenizer.convert_ids_to_tokens(input_ids)
            visual_positions = _qwen_bucket_positions(input_ids, token_texts)["image_marker_or_span"]
            answer_positions = _answer_adjacent_positions(len(input_ids), set(visual_positions), 4)
            allowed_positions = sorted(set(visual_positions + answer_positions))
            control_routes = _build_control_routes(
                nodes=nodes,
                clean_features_by_layer=clean_features_by_layer,
                transcoders=transcoders,
                target_direction=output_weight[target_token_id].detach().to(device=device, dtype=dtype),
                allowed_positions=allowed_positions,
                route=route,
                device=device,
                dtype=dtype,
            )
            if "source" not in control_routes or len(control_routes) < 2:
                payload["skipped"].append({"route_id": route_id, "reason": "insufficient_route_controls"})
                continue

            clean_activation_sums = {
                name: _route_activation_sum(clean_features_by_layer, controls) for name, controls in control_routes.items()
            }
            for control_group, controls in control_routes.items():
                token_scores_after = _score_with_route_patch(
                    model=model,
                    inputs=clean_inputs,
                    tokenizer=tokenizer,
                    transcoders=transcoders,
                    modules=modules,
                    route_controls=controls,
                    token_ids=[target_token_id, wrong_token_id],
                    scale=args.scale,
                    patch_kind="zeroing",
                    clean_features_by_layer=None,
                    mask_features_by_layer=None,
                    device=device,
                    dtype=dtype,
                )
                rows.append(
                    _effect_row(
                        route=route,
                        control_group=control_group,
                        route_controls=controls,
                        intervention_kind="clean_route_zeroing",
                        mask_condition="clean",
                        token_scored="target",
                        token_id=target_token_id,
                        before=clean_target_score,
                        after=token_scores_after[target_token_id],
                        clean_target_logit=float(clean_target_score["target_logit"]),
                        clean_target_rank=int(clean_target_score["target_rank"]),
                        route_clean_activation_sum=clean_activation_sums.get(control_group, ""),
                    )
                )
                rows.append(
                    _effect_row(
                        route=route,
                        control_group=control_group,
                        route_controls=controls,
                        intervention_kind="clean_route_zeroing",
                        mask_condition="clean",
                        token_scored="wrong",
                        token_id=wrong_token_id,
                        before=clean_wrong_score,
                        after=token_scores_after[wrong_token_id],
                        clean_target_logit=float(clean_target_score["target_logit"]),
                        clean_target_rank=int(clean_target_score["target_rank"]),
                        route_clean_activation_sum=clean_activation_sums.get(control_group, ""),
                    )
                )

            for mask_condition in mask_conditions:
                mask_file = _mask_path(route, args.mask_root, mask_condition)
                if not mask_file.exists():
                    payload["skipped"].append({"route_id": route_id, "mask_condition": mask_condition, "reason": "mask_missing"})
                    continue
                mask = Image.open(mask_file).convert("L").resize(image.size)
                mask_image = _apply_mask(image, mask, (128, 128, 128))
                mask_inputs = _qwen_inputs(processor, mask_image, str(image_path), question, args.answer_prefix, device)
                mask_outputs, mask_hidden_by_layer = _forward_with_multi_capture(model, modules, mask_inputs)
                mask_features_by_layer = _features_by_layer(transcoders, mask_hidden_by_layer, device, dtype)
                mask_target_score = _score_token(mask_outputs.logits, tokenizer, target_token_id)
                mask_wrong_score = _score_token(mask_outputs.logits, tokenizer, wrong_token_id)
                for control_group, controls in control_routes.items():
                    mask_activation_sum = _route_activation_sum(mask_features_by_layer, controls)
                    clean_activation_sum = clean_activation_sums.get(control_group, 0.0)
                    activation_drop = float(clean_activation_sum) - float(mask_activation_sum)
                    token_scores_after = _score_with_route_patch(
                        model=model,
                        inputs=mask_inputs,
                        tokenizer=tokenizer,
                        transcoders=transcoders,
                        modules=modules,
                        route_controls=controls,
                        token_ids=[target_token_id, wrong_token_id],
                        scale=args.scale,
                        patch_kind="restore",
                        clean_features_by_layer=clean_features_by_layer,
                        mask_features_by_layer=mask_features_by_layer,
                        device=device,
                        dtype=dtype,
                    )
                    rows.append(
                        _effect_row(
                            route=route,
                            control_group=control_group,
                            route_controls=controls,
                            intervention_kind="mask_route_restore",
                            mask_condition=mask_condition,
                            token_scored="target",
                            token_id=target_token_id,
                            before=mask_target_score,
                            after=token_scores_after[target_token_id],
                            clean_target_logit=float(clean_target_score["target_logit"]),
                            clean_target_rank=int(clean_target_score["target_rank"]),
                            mask_target_logit=float(mask_target_score["target_logit"]),
                            mask_target_rank=int(mask_target_score["target_rank"]),
                            route_clean_activation_sum=clean_activation_sum,
                            route_mask_activation_sum=mask_activation_sum,
                            route_activation_drop=activation_drop,
                        )
                    )
                    rows.append(
                        _effect_row(
                            route=route,
                            control_group=control_group,
                            route_controls=controls,
                            intervention_kind="mask_route_restore",
                            mask_condition=mask_condition,
                            token_scored="wrong",
                            token_id=wrong_token_id,
                            before=mask_wrong_score,
                            after=token_scores_after[wrong_token_id],
                            clean_target_logit=float(clean_target_score["target_logit"]),
                            clean_target_rank=int(clean_target_score["target_rank"]),
                            mask_target_logit=float(mask_wrong_score["target_logit"]),
                            mask_target_rank=int(mask_wrong_score["target_rank"]),
                            route_clean_activation_sum=clean_activation_sum,
                            route_mask_activation_sum=mask_activation_sum,
                            route_activation_drop=activation_drop,
                        )
                    )
                    if args.include_route_corrupt:
                        corrupt_scores_after = _score_with_route_patch(
                            model=model,
                            inputs=clean_inputs,
                            tokenizer=tokenizer,
                            transcoders=transcoders,
                            modules=modules,
                            route_controls=controls,
                            token_ids=[target_token_id, wrong_token_id],
                            scale=args.scale,
                            patch_kind="corrupt",
                            clean_features_by_layer=clean_features_by_layer,
                            mask_features_by_layer=mask_features_by_layer,
                            device=device,
                            dtype=dtype,
                        )
                        rows.append(
                            _effect_row(
                                route=route,
                                control_group=control_group,
                                route_controls=controls,
                                intervention_kind="route_corrupt",
                                mask_condition=mask_condition,
                                token_scored="target",
                                token_id=target_token_id,
                                before=clean_target_score,
                                after=corrupt_scores_after[target_token_id],
                                clean_target_logit=float(clean_target_score["target_logit"]),
                                clean_target_rank=int(clean_target_score["target_rank"]),
                                route_clean_activation_sum=clean_activation_sum,
                                route_mask_activation_sum=mask_activation_sum,
                                route_activation_drop=activation_drop,
                            )
                        )
                del mask_outputs, mask_hidden_by_layer, mask_features_by_layer
                torch.cuda.empty_cache()

            usable_routes += 1
            if idx % 3 == 0:
                _log(f"validated routes: {idx}/{len(routes)} rows={len(rows)}")
            if args.checkpoint_every > 0 and usable_routes % args.checkpoint_every == 0:
                _checkpoint()
            del clean_outputs, clean_hidden_by_layer, clean_features_by_layer
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            payload["skipped"].append({"route_id": route_id, "sample_id": sample_id, "prompt_name": prompt_name, "reason": repr(exc)})
            torch.cuda.empty_cache()

    fieldnames = [
        "route_id",
        "route_base_id",
        "pack",
        "sample_id",
        "prompt_name",
        "topk",
        "best_real_condition",
        "route_node_count",
        "primary_node_count",
        "strict_missing_count",
        "strict_missing_fraction",
        "node_candidate_ids",
        "node_layers",
        "node_source_positions",
        "node_feature_ids",
        "target_answer",
        "target_token_id",
        "target_token",
        "wrong_token_id",
        "wrong_token",
        "intervention_kind",
        "mask_condition",
        "control_group",
        "control_route_node_count",
        "control_layers",
        "control_positions",
        "control_feature_ids",
        "route_clean_activation_sum",
        "route_mask_activation_sum",
        "route_activation_drop",
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
        "usable_routes": usable_routes,
        "requested_routes": len(routes),
        "raw_rows": len(rows),
        "skipped_count": len(payload["skipped"]),
    }
    _write_json(Path(args.summary_json), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
