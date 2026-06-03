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
    _rank_and_top,
    _replace_hidden,
)
from run_cross_model_hidden_position_patch_smoke import _answer_adjacent_positions
from run_stage4_qwen_causal_cutter_validation import (
    _choose_controls,
    _forward_with_capture,
    _score_token,
    _top_wrong_token,
)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage4-qwen-evidence-v2] {message}", flush=True)


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
    except ValueError:
        return default


def _f(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _image_path(row: dict[str, str], image_root: str) -> Path:
    if image_root:
        return Path(image_root) / Path(row["image_filename"]).name
    return Path(row.get("local_image_path") or row.get("image_path") or row.get("image_filename"))


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
    return Path(row.get(
        {
            "answer_mask": "answer_mask_path",
            "union_mask": "union_mask_path",
            "shifted_mask": "shifted_mask_path",
            "shuffled_mask": "shuffled_mask_path",
        }[condition],
        "",
    ))


def _decoder_vectors(transcoders, layer: int, feature_ids: list[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ids = torch.tensor(feature_ids, device=device, dtype=torch.long)
    vectors = transcoders._get_decoder_vectors(layer, ids)
    if vectors.ndim == 3:
        vectors = vectors[:, 0, :]
    return vectors.to(device=device, dtype=dtype)


def _candidate_groups(path: Path, selection: str, max_prompt_runs: int) -> list[tuple[tuple[str, str], list[dict[str, str]]]]:
    rows = []
    for row in _read_csv(path):
        if selection == "main" and row.get("include_main") != "1":
            continue
        if selection == "pool" and row.get("include_pool") != "1":
            continue
        rows.append(row)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["sample_id"], row["prompt_name"])].append(row)
    out = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda r: (_f(r.get("damage_score")), _f(r.get("path_mass_best"))), reverse=True)
        out.append((key, group))
    if max_prompt_runs > 0:
        out = out[:max_prompt_runs]
    return out


def _patch_group(
    hidden: torch.Tensor,
    *,
    candidates: list[dict[str, Any]],
    clean_features: torch.Tensor,
    mask_features: torch.Tensor,
    decoder_vectors: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    new_hidden = hidden.clone()
    for idx, candidate in enumerate(candidates):
        pos = int(candidate["pos"])
        feature_id = int(candidate["feature_id"])
        if pos < 0 or pos >= hidden.shape[1] or feature_id < 0 or feature_id >= clean_features.shape[2]:
            continue
        drop = (clean_features[:, pos, feature_id] - mask_features[:, pos, feature_id]).to(hidden.device, dtype=hidden.dtype)
        vector = decoder_vectors[idx].to(hidden.device, dtype=hidden.dtype)
        patch = drop[:, None] * vector[None, :]
        new_hidden[:, pos, :] = new_hidden[:, pos, :] + scale * patch
    return new_hidden


def _score_with_group_restore(
    *,
    model,
    inputs: dict[str, Any],
    tokenizer,
    module,
    candidates: list[dict[str, Any]],
    clean_features: torch.Tensor,
    mask_features: torch.Tensor,
    decoder_vectors: torch.Tensor,
    token_ids: list[int],
    scale: float,
) -> dict[int, dict[str, Any]]:
    def _hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        patched = _patch_group(
            hidden,
            candidates=candidates,
            clean_features=clean_features,
            mask_features=mask_features,
            decoder_vectors=decoder_vectors,
            scale=scale,
        )
        return _replace_hidden(output, patched)

    handle = module.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=False, use_cache=False)
    finally:
        handle.remove()
    return {int(token_id): _score_token(outputs.logits, tokenizer, int(token_id)) for token_id in token_ids}


def _activation_drop(clean_features: torch.Tensor, mask_features: torch.Tensor, candidates: list[dict[str, Any]]) -> tuple[float, float]:
    drops = []
    clean_vals = []
    for candidate in candidates:
        pos = int(candidate["pos"])
        feature_id = int(candidate["feature_id"])
        if pos < 0 or pos >= clean_features.shape[1] or feature_id < 0 or feature_id >= clean_features.shape[2]:
            continue
        clean = float(clean_features[0, pos, feature_id].float().item())
        masked = float(mask_features[0, pos, feature_id].float().item())
        clean_vals.append(clean)
        drops.append(clean - masked)
    if not drops:
        return 0.0, 0.0
    return sum(drops), sum(clean_vals)


def _row_base(first: dict[str, str], top_k: int, actual_k: int, control_group: str, mask_condition: str, record_type: str) -> dict[str, Any]:
    return {
        "record_type": record_type,
        "sample_id": first.get("sample_id", ""),
        "prompt_name": first.get("prompt_name", ""),
        "analysis_group": first.get("analysis_group", ""),
        "reasoning_operation": first.get("reasoning_operation", ""),
        "image_dependence_tier": first.get("image_dependence_tier", ""),
        "target_answer": first.get("answer_text", ""),
        "target_token_id": first.get("target_token_id", ""),
        "target_token": first.get("target_token", ""),
        "top_k": top_k,
        "actual_k": actual_k,
        "control_group": control_group,
        "mask_condition": mask_condition,
        "candidate_ids": "",
        "feature_ids": "",
        "positions": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage4-016 Qwen evidence-linked grouped cutter validation.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--transcoder-ref", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--selection", choices=["main", "pool", "all"], default="pool")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--mask-root", default="")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--layer", type=int, default=26)
    parser.add_argument("--answer-prefix", default="The answer is ")
    parser.add_argument("--top-ks", default="1,4,8,16")
    parser.add_argument("--mask-conditions", default="answer_mask,union_mask,shifted_mask,shuffled_mask")
    parser.add_argument("--max-prompt-runs", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--min-gpu-free-gb", type=float, default=12.0)
    args = parser.parse_args()

    groups = _candidate_groups(Path(args.candidate_manifest), args.selection, args.max_prompt_runs)
    payload: dict[str, Any] = {
        "created_at": _now(),
        "model_name": args.model_name,
        "transcoder_ref": args.transcoder_ref,
        "candidate_manifest": args.candidate_manifest,
        "selection": args.selection,
        "layer": args.layer,
        "requested_prompt_runs": len(groups),
        "env_presence": _env_presence(),
        "gpu_before": _gpu_info(),
        "skipped": [],
        "claim_boundary": "Grouped evidence-link validation; positive rows support Qwen-native evidence-linked cutter, not full Gemma-style tracing by themselves.",
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
    output_weight = model.get_output_embeddings().weight
    payload["transcoder"] = {"type": type(transcoders).__name__, "config_model_kind": config.get("model_kind", "")}

    rows: list[dict[str, Any]] = []
    top_ks = _parse_int_csv(args.top_ks)
    mask_conditions = _parse_csv(args.mask_conditions)
    usable = 0

    for group_idx, ((_sample_id, _prompt_name), candidate_rows) in enumerate(groups, start=1):
        first = candidate_rows[0]
        try:
            image_path = _image_path(first, args.image_root)
            image = Image.open(image_path).convert("RGB")
            question = _prompt(first.get("question_text", ""), first.get("prompt_name", ""))
            clean_inputs = _qwen_inputs(processor, image, str(image_path), question, args.answer_prefix, device)
            clean_outputs, clean_hidden = _forward_with_capture(model, module, clean_inputs)
            clean_hidden = clean_hidden.to(device=device, dtype=dtype)
            clean_features = transcoders.encode_layer(clean_hidden, args.layer, apply_activation_function=True).detach()
            target_token_id = _i(first.get("target_token_id"))
            clean_target_score = _score_token(clean_outputs.logits, tokenizer, target_token_id)
            wrong = _top_wrong_token(clean_outputs.logits, tokenizer, target_token_id)
            wrong_token_id = int(wrong["wrong_token_id"])

            input_ids = clean_inputs["input_ids"][0].detach().cpu().tolist()
            token_texts = tokenizer.convert_ids_to_tokens(input_ids)
            visual_positions = _qwen_bucket_positions(input_ids, token_texts)["image_marker_or_span"]
            answer_positions = _answer_adjacent_positions(len(input_ids), set(visual_positions), 4)
            allowed_positions = sorted(set(visual_positions + answer_positions))
            excluded = {
                _i(part, -1)
                for part in str(first.get("known_damaging_feature_ids_for_prompt_run", "")).split("|")
                if part.strip()
            }

            source_candidates = [
                {
                    "candidate_id": row.get("candidate_id", ""),
                    "pos": _i(row.get("source_pos"), -1),
                    "feature_id": _i(row.get("source_feature_id"), -1),
                    "damage_score": _f(row.get("damage_score"), 0.0),
                    "zeroing_mode": row.get("source_zeroing_mode") or "subtract",
                }
                for row in candidate_rows
            ]
            source_candidates = [c for c in source_candidates if c["pos"] >= 0 and c["feature_id"] >= 0]
            if not source_candidates:
                payload["skipped"].append({"sample_id": first.get("sample_id"), "prompt_name": first.get("prompt_name"), "reason": "no_source_candidates"})
                continue

            controls_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for candidate in source_candidates:
                controls = _choose_controls(
                    features=clean_features,
                    transcoders=transcoders,
                    layer=args.layer,
                    source_pos=int(candidate["pos"]),
                    source_feature_id=int(candidate["feature_id"]),
                    source_zeroing_mode=str(candidate["zeroing_mode"]),
                    target_direction=output_weight[target_token_id].detach().to(device=device, dtype=dtype),
                    allowed_positions=allowed_positions,
                    excluded_features=excluded,
                    seed=hash((first.get("sample_id"), first.get("prompt_name"), candidate["candidate_id"])) % (2**32),
                    device=device,
                    dtype=dtype,
                )
                for control_name, control in controls.items():
                    controls_by_source[control_name].append(
                        {
                            "candidate_id": candidate["candidate_id"],
                            "pos": int(control["pos"]),
                            "feature_id": int(control["feature_id"]),
                            "damage_score": candidate["damage_score"],
                        }
                    )

            for mask_condition in mask_conditions:
                mask_file = _mask_path(first, args.mask_root, mask_condition)
                if not mask_file.exists():
                    payload["skipped"].append({"sample_id": first.get("sample_id"), "mask_condition": mask_condition, "reason": "mask_missing"})
                    continue
                mask = Image.open(mask_file).convert("L").resize(image.size)
                masked_image = _apply_mask(image, mask, (128, 128, 128))
                mask_inputs = _qwen_inputs(processor, masked_image, str(image_path), question, args.answer_prefix, device)
                mask_outputs, mask_hidden = _forward_with_capture(model, module, mask_inputs)
                mask_hidden = mask_hidden.to(device=device, dtype=dtype)
                mask_features = transcoders.encode_layer(mask_hidden, args.layer, apply_activation_function=True).detach()
                mask_target_score = _score_token(mask_outputs.logits, tokenizer, target_token_id)
                mask_wrong_score = _score_token(mask_outputs.logits, tokenizer, wrong_token_id)

                for top_k in top_ks:
                    for control_group, control_candidates in controls_by_source.items():
                        selected = sorted(control_candidates, key=lambda item: float(item["damage_score"]), reverse=True)[: min(top_k, len(control_candidates))]
                        if not selected:
                            continue
                        feature_ids = [int(item["feature_id"]) for item in selected]
                        vectors = _decoder_vectors(transcoders, args.layer, feature_ids, device, dtype)
                        drop_sum, clean_sum = _activation_drop(clean_features, mask_features, selected)
                        base = _row_base(first, top_k, len(selected), control_group, mask_condition, "activation_drop")
                        base.update(
                            {
                                "candidate_ids": "|".join(str(item["candidate_id"]) for item in selected),
                                "feature_ids": "|".join(str(item["feature_id"]) for item in selected),
                                "positions": "|".join(str(item["pos"]) for item in selected),
                                "activation_drop_sum": drop_sum,
                                "clean_activation_sum": clean_sum,
                                "target_logit_effect": "",
                                "target_rank_effect": "",
                                "wrong_logit_effect": "",
                                "wrong_rank_effect": "",
                                "gap_closure": "",
                                "clean_target_logit": clean_target_score["target_logit"],
                                "mask_target_logit": mask_target_score["target_logit"],
                                "clean_target_rank": clean_target_score["target_rank"],
                                "mask_target_rank": mask_target_score["target_rank"],
                            }
                        )
                        rows.append(base)
                        restored = _score_with_group_restore(
                            model=model,
                            inputs=mask_inputs,
                            tokenizer=tokenizer,
                            module=module,
                            candidates=selected,
                            clean_features=clean_features,
                            mask_features=mask_features,
                            decoder_vectors=vectors,
                            token_ids=[target_token_id, wrong_token_id],
                            scale=args.scale,
                        )
                        target_effect = float(restored[target_token_id]["target_logit"]) - float(mask_target_score["target_logit"])
                        wrong_effect = float(restored[wrong_token_id]["target_logit"]) - float(mask_wrong_score["target_logit"])
                        gap = float(clean_target_score["target_logit"]) - float(mask_target_score["target_logit"])
                        base = _row_base(first, top_k, len(selected), control_group, mask_condition, "group_restore")
                        base.update(
                            {
                                "candidate_ids": "|".join(str(item["candidate_id"]) for item in selected),
                                "feature_ids": "|".join(str(item["feature_id"]) for item in selected),
                                "positions": "|".join(str(item["pos"]) for item in selected),
                                "activation_drop_sum": drop_sum,
                                "clean_activation_sum": clean_sum,
                                "target_logit_effect": target_effect,
                                "target_rank_effect": int(mask_target_score["target_rank"]) - int(restored[target_token_id]["target_rank"]),
                                "wrong_logit_effect": wrong_effect,
                                "wrong_rank_effect": int(mask_wrong_score["target_rank"]) - int(restored[wrong_token_id]["target_rank"]),
                                "gap_closure": target_effect / gap if abs(gap) > 1e-6 else "",
                                "clean_target_logit": clean_target_score["target_logit"],
                                "mask_target_logit": mask_target_score["target_logit"],
                                "clean_target_rank": clean_target_score["target_rank"],
                                "mask_target_rank": mask_target_score["target_rank"],
                            }
                        )
                        rows.append(base)
                del mask_outputs, mask_hidden, mask_features
                torch.cuda.empty_cache()
            usable += 1
            if group_idx % 5 == 0:
                _log(f"prompt-runs complete: {group_idx}/{len(groups)} rows={len(rows)}")
            del clean_outputs, clean_hidden, clean_features
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            payload["skipped"].append({"sample_id": first.get("sample_id"), "prompt_name": first.get("prompt_name"), "reason": repr(exc)})
            torch.cuda.empty_cache()

    fields = [
        "record_type",
        "sample_id",
        "prompt_name",
        "analysis_group",
        "reasoning_operation",
        "image_dependence_tier",
        "target_answer",
        "target_token_id",
        "target_token",
        "top_k",
        "actual_k",
        "control_group",
        "mask_condition",
        "candidate_ids",
        "feature_ids",
        "positions",
        "activation_drop_sum",
        "clean_activation_sum",
        "target_logit_effect",
        "target_rank_effect",
        "wrong_logit_effect",
        "wrong_rank_effect",
        "gap_closure",
        "clean_target_logit",
        "mask_target_logit",
        "clean_target_rank",
        "mask_target_rank",
    ]
    _write_csv(Path(args.out_csv), rows, fields)
    payload["gpu_after"] = _gpu_info()
    payload["decision"] = {
        "status": "ok" if rows else "blocked_no_rows",
        "usable_prompt_runs": usable,
        "requested_prompt_runs": len(groups),
        "raw_rows": len(rows),
        "skipped_count": len(payload["skipped"]),
    }
    _write_json(Path(args.summary_json), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
