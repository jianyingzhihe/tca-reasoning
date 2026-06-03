#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from run_cross_model_feature_intervention_smoke import (
    SAMPLES,
    _apply_mask,
    _env_presence,
    _first_param_device,
    _gpu_info,
    _llava_bucket_positions,
    _llava_inputs,
    _load_masks,
    _parse_csv,
    _parse_float_csv,
    _processor_patch,
    _prompt,
    _qwen_bucket_positions,
    _qwen_inputs,
    _rank_and_top,
    _replace_hidden,
    _target_candidates,
)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage2h-position-patch] {message}", flush=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _scale_label(scale: float) -> str:
    return str(scale).replace(".", "p").replace("-", "m")


def _safe_ratio(value: float, denom: float) -> float | str:
    if denom <= 1e-6:
        return ""
    return value / denom


def _stable_seed(*parts: str) -> int:
    raw = "||".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def _dedupe_sorted(values: list[int]) -> list[int]:
    return sorted({int(value) for value in values})


def _first_nonempty(row: dict[str, str], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _load_sample_manifest(path: Path) -> dict[str, dict[str, str]]:
    samples: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = _first_nonempty(row, ["sample_id", "id"])
            image_name = _first_nonempty(row, ["image_filename", "image", "image_name"])
            question = _first_nonempty(row, ["question_text", "question"])
            answer = _first_nonempty(row, ["answer_text", "answer", "target_answer"])
            if not sample_id or not image_name or not question or not answer:
                continue
            samples[sample_id] = {
                "image": Path(image_name).name,
                "question": question,
                "answer": answer,
                "reasoning_operation": _first_nonempty(row, ["reasoning_operation"]),
                "visual_structure": _first_nonempty(row, ["visual_structure"]),
                "image_dependence": _first_nonempty(row, ["image_dependence"]),
                "mask_pack": _first_nonempty(row, ["mask_pack"]),
                "compactness_label": _first_nonempty(row, ["compactness_label"]),
            }
    return samples


def _resize_mask_scores(mask: Image.Image, grid_h: int, grid_w: int) -> list[float]:
    resampling = getattr(Image, "Resampling", Image).BOX
    small = mask.convert("L").resize((grid_w, grid_h), resampling)
    return [float(value) / 255.0 for value in small.getdata()]


def _grid_evidence_positions(
    *,
    positions: list[int],
    mask: Image.Image,
    grid_h: int,
    grid_w: int,
    max_positions: int,
) -> tuple[list[int], dict[str, Any]]:
    expected = grid_h * grid_w
    if expected != len(positions):
        return [], {"status": "grid_position_count_mismatch", "expected": expected, "actual": len(positions)}
    scores = _resize_mask_scores(mask, grid_h, grid_w)
    candidates = [(score, idx) for idx, score in enumerate(scores) if score > 0.01]
    if not candidates:
        candidates = [(score, idx) for idx, score in enumerate(scores) if score > 0.0]
    candidates.sort(reverse=True)
    chosen = [positions[idx] for _score, idx in candidates[:max_positions]]
    return _dedupe_sorted(chosen), {
        "status": "ok" if chosen else "empty_overlap",
        "grid_h": grid_h,
        "grid_w": grid_w,
        "candidate_count": len(candidates),
        "position_count": len(chosen),
        "max_score": candidates[0][0] if candidates else 0.0,
    }


def _qwen_grid_from_inputs(inputs: dict[str, Any], position_count: int) -> tuple[int, int] | None:
    grid = inputs.get("image_grid_thw")
    if grid is None:
        return None
    try:
        vals = grid.detach().cpu().tolist()[0]
        t, h, w = int(vals[0]), int(vals[1]), int(vals[2])
    except Exception:
        return None
    if t * h * w == position_count:
        return h, w
    return None


def _position_delta_norms(clean_hidden: torch.Tensor, union_hidden: torch.Tensor, positions: list[int]) -> list[tuple[float, int]]:
    if not positions:
        return []
    pos = torch.tensor(positions, device=clean_hidden.device, dtype=torch.long)
    delta = (clean_hidden[:, pos, :].float() - union_hidden[:, pos, :].float()).norm(dim=-1)[0]
    return [(float(value), int(position)) for value, position in zip(delta.detach().cpu().tolist(), positions, strict=False)]


def _position_activation_norms(hidden: torch.Tensor, positions: list[int]) -> list[tuple[float, int]]:
    if not positions:
        return []
    pos = torch.tensor(positions, device=hidden.device, dtype=torch.long)
    norms = hidden[:, pos, :].float().norm(dim=-1)[0]
    return [(float(value), int(position)) for value, position in zip(norms.detach().cpu().tolist(), positions, strict=False)]


def _matched_positions_by_score(
    *,
    source_positions: list[int],
    candidate_positions: list[int],
    source_scores: dict[int, float],
    candidate_scores: dict[int, float],
    count: int,
) -> list[int]:
    source_values = [source_scores[position] for position in source_positions if position in source_scores]
    candidates = [position for position in candidate_positions if position in candidate_scores]
    if not source_values or not candidates:
        return []
    chosen: list[int] = []
    remaining = set(candidates)
    # Match the full source score profile, not just the mean, while preventing reuse.
    for value in sorted(source_values, reverse=True):
        if not remaining or len(chosen) >= count:
            break
        best = min(remaining, key=lambda position: (abs(candidate_scores[position] - value), position))
        chosen.append(best)
        remaining.remove(best)
    return _dedupe_sorted(chosen)


def _top_delta_positions(clean_hidden: torch.Tensor, union_hidden: torch.Tensor, positions: list[int], count: int) -> list[int]:
    scored = _position_delta_norms(clean_hidden, union_hidden, positions)
    scored.sort(reverse=True)
    return _dedupe_sorted([position for _score, position in scored[:count]])


def _low_delta_positions(
    clean_hidden: torch.Tensor,
    union_hidden: torch.Tensor,
    positions: list[int],
    count: int,
    exclude: set[int],
) -> list[int]:
    scored = [(score, position) for score, position in _position_delta_norms(clean_hidden, union_hidden, positions) if position not in exclude]
    scored.sort()
    return _dedupe_sorted([position for _score, position in scored[:count]])


def _random_positions(positions: list[int], count: int, exclude: set[int], seed: int) -> list[int]:
    pool = [position for position in positions if position not in exclude]
    if not pool:
        return []
    rng = random.Random(seed)
    return _dedupe_sorted(rng.sample(pool, k=min(count, len(pool))))


def _answer_adjacent_positions(seq_len: int, visual_positions: set[int], count: int) -> list[int]:
    out = []
    for position in range(seq_len - 1, -1, -1):
        if position not in visual_positions:
            out.append(position)
        if len(out) >= count:
            break
    return sorted(out)


def _patch_hidden_tensor(
    hidden: torch.Tensor,
    *,
    positions: list[int],
    source_hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if not positions:
        return hidden
    pos_tensor = torch.tensor(positions, device=hidden.device, dtype=torch.long)
    source = source_hidden[:, pos_tensor, :].to(hidden.device, dtype=hidden.dtype)
    target = target_hidden[:, pos_tensor, :].to(hidden.device, dtype=hidden.dtype)
    patch = source - target
    new_hidden = hidden.clone()
    new_hidden[:, pos_tensor, :] = new_hidden[:, pos_tensor, :] + scale * patch
    return new_hidden


def _score_with_patch(
    model,
    inputs: dict[str, Any],
    tokenizer,
    target_ids: list[int],
    module,
    intervention: dict[str, Any] | None,
) -> dict[str, Any]:
    handle = None
    if intervention is not None:

        def _hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            patched = _patch_hidden_tensor(hidden, **intervention)
            return _replace_hidden(output, patched)

        handle = module.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=False, use_cache=False)
    finally:
        if handle is not None:
            handle.remove()
    return _rank_and_top(outputs.logits, tokenizer, target_ids)


def _make_groups(
    *,
    model_family: str,
    clean_inputs: dict[str, Any],
    clean_hidden: torch.Tensor,
    union_hidden: torch.Tensor,
    visual_positions: list[int],
    answer_positions: list[int],
    evidence_mask: Image.Image,
    sample_id: str,
    prompt_name: str,
    default_count: int,
    max_evidence_positions: int,
    random_controls: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    visual_set = set(visual_positions)
    grid_diag: dict[str, Any] = {"status": "not_attempted"}
    evidence_positions: list[int] = []
    if model_family == "llava":
        grid_h = grid_w = int(round(len(visual_positions) ** 0.5))
        if grid_h * grid_w == len(visual_positions):
            evidence_positions, grid_diag = _grid_evidence_positions(
                positions=visual_positions,
                mask=evidence_mask,
                grid_h=grid_h,
                grid_w=grid_w,
                max_positions=max_evidence_positions,
            )
        else:
            grid_diag = {"status": "llava_not_square_grid", "position_count": len(visual_positions)}
    elif model_family == "qwen":
        grid = _qwen_grid_from_inputs(clean_inputs, len(visual_positions))
        if grid is not None:
            evidence_positions, grid_diag = _grid_evidence_positions(
                positions=visual_positions,
                mask=evidence_mask,
                grid_h=grid[0],
                grid_w=grid[1],
                max_positions=max_evidence_positions,
            )
        else:
            grid_diag = {
                "status": "qwen_grid_unavailable_or_mismatch",
                "position_count": len(visual_positions),
                "has_image_grid_thw": "image_grid_thw" in clean_inputs,
            }

    source_count = len(evidence_positions) if evidence_positions else min(default_count, len(visual_positions))
    source_count = max(1, min(source_count, len(visual_positions)))
    top_delta = _top_delta_positions(clean_hidden, union_hidden, visual_positions, source_count)
    exclude_for_controls = set(evidence_positions) | set(top_delta)
    low_delta = _low_delta_positions(clean_hidden, union_hidden, visual_positions, source_count, exclude_for_controls)
    candidate_controls = [position for position in visual_positions if position not in exclude_for_controls]
    delta_scores = {position: score for score, position in _position_delta_norms(clean_hidden, union_hidden, visual_positions)}
    activation_scores = {position: score for score, position in _position_activation_norms(clean_hidden, visual_positions)}
    delta_matched = _matched_positions_by_score(
        source_positions=top_delta,
        candidate_positions=candidate_controls,
        source_scores=delta_scores,
        candidate_scores=delta_scores,
        count=source_count,
    )
    activation_matched = _matched_positions_by_score(
        source_positions=top_delta,
        candidate_positions=candidate_controls,
        source_scores=activation_scores,
        candidate_scores=activation_scores,
        count=source_count,
    )

    groups: list[dict[str, Any]] = [
        {"group_name": "whole_bucket", "group_kind": "upper_bound", "positions": _dedupe_sorted(visual_positions)},
        {"group_name": "top_hidden_delta", "group_kind": "source_like", "positions": top_delta},
        {"group_name": "low_delta_control", "group_kind": "control", "positions": low_delta},
        {"group_name": "delta_matched_control", "group_kind": "matched_control", "positions": delta_matched},
        {"group_name": "activation_matched_control", "group_kind": "matched_control", "positions": activation_matched},
        {"group_name": "answer_adjacent_text", "group_kind": "bridge_text", "positions": answer_positions},
        {
            "group_name": "top_hidden_delta_plus_answer_adjacent",
            "group_kind": "bridge_combo",
            "positions": _dedupe_sorted(top_delta + answer_positions),
        },
        {
            "group_name": "delta_matched_plus_answer_adjacent",
            "group_kind": "matched_bridge_control",
            "positions": _dedupe_sorted(delta_matched + answer_positions),
        },
        {
            "group_name": "activation_matched_plus_answer_adjacent",
            "group_kind": "matched_bridge_control",
            "positions": _dedupe_sorted(activation_matched + answer_positions),
        },
    ]
    if evidence_positions:
        groups.extend(
            [
                {"group_name": "evidence_region", "group_kind": "source_like", "positions": evidence_positions},
                {
                    "group_name": "evidence_region_plus_answer_adjacent",
                    "group_kind": "bridge_combo",
                    "positions": _dedupe_sorted(evidence_positions + answer_positions),
                },
            ]
        )
    for idx in range(1, random_controls + 1):
        positions = _random_positions(
            visual_positions,
            source_count,
            set(evidence_positions),
            _stable_seed(sample_id, prompt_name, str(idx), model_family),
        )
        groups.append({"group_name": f"random_control_{idx}", "group_kind": "random_control", "positions": positions})

    diagnostics = {
        "grid": grid_diag,
        "visual_position_count": len(visual_positions),
        "answer_adjacent_positions": answer_positions,
        "evidence_position_count": len(evidence_positions),
        "top_delta_position_count": len(top_delta),
        "delta_matched_position_count": len(delta_matched),
        "activation_matched_position_count": len(activation_matched),
        "source_count": source_count,
        "visual_position_min": min(visual_set) if visual_set else "",
        "visual_position_max": max(visual_set) if visual_set else "",
    }
    return groups, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2H cross-model hidden position patch smoke.")
    parser.add_argument("--model-family", choices=["qwen", "llava"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--annotation-roots", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--sample-manifest", default="", help="Optional CSV with sample_id,image_filename,question_text,answer_text.")
    parser.add_argument("--samples", default="okvqa_val_2847255,okvqa_val_4157235,okvqa_val_3658865")
    parser.add_argument("--prompts", default="B_direct,D_visual_only")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--scales", default="1.0")
    parser.add_argument("--default-position-count", type=int, default=32)
    parser.add_argument("--max-evidence-positions", type=int, default=64)
    parser.add_argument("--answer-adjacent-count", type=int, default=4)
    parser.add_argument("--random-controls", type=int, default=4)
    parser.add_argument("--answer-prefix", default="The answer is ")
    parser.add_argument("--min-gpu-free-gb", type=float, default=18.0)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    sample_lookup = _load_sample_manifest(Path(args.sample_manifest)) if args.sample_manifest else dict(SAMPLES)
    if args.samples.strip().lower() in {"manifest", "auto", "all"}:
        sample_ids = list(sample_lookup)
    else:
        sample_ids = _parse_csv(args.samples)
    prompt_names = _parse_csv(args.prompts)
    scales = _parse_float_csv(args.scales)
    roots = [Path(path) for path in _parse_csv(args.annotation_roots)]
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "created_at": _now(),
        "model_family": args.model_family,
        "model_name": args.model_name,
        "layer": args.layer,
        "scales": scales,
        "sample_manifest": args.sample_manifest,
        "samples": sample_ids,
        "prompts": prompt_names,
        "env_presence": _env_presence(),
        "gpu_before": _gpu_info(),
        "groups": [],
        "decision": {},
        "claim_boundary": (
            "Hidden-position patch smoke only; not CLT feature-level specificity, not source-control "
            "causal route replication."
        ),
    }
    gpu = payload["gpu_before"]
    if not gpu.get("available") or float(gpu.get("free_gb", 0.0)) < args.min_gpu_free_gb:
        payload["decision"] = {"status": "partial", "reason": "insufficient_gpu_free_memory"}
        _write_json(Path(args.out_json), payload)
        return 0

    if args.model_family == "qwen":
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
        target_module = model.language_model.layers[args.layer]

        def build_inputs(image: Image.Image, image_path: Path, question: str):
            return _qwen_inputs(processor, image, str(image_path), question, args.answer_prefix, device)

        def bucket_positions(input_ids: list[int], token_texts: list[str]) -> tuple[str, list[int]]:
            buckets = _qwen_bucket_positions(input_ids, token_texts)
            return "image_marker_or_span", buckets["image_marker_or_span"]

        hidden_index = args.layer
    else:
        from transformers import AutoConfig, AutoProcessor, LlavaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(args.model_name)
        config = AutoConfig.from_pretrained(args.model_name)
        patch_info = _processor_patch(processor, config)
        tokenizer = processor.tokenizer
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()
        device = _first_param_device(model)
        target_module = model.language_model.layers[args.layer]
        image_token_id = int(getattr(config, "image_token_index", 32000))
        payload["processor_patch"] = patch_info

        def build_inputs(image: Image.Image, image_path: Path, question: str):
            return _llava_inputs(processor, image, question, args.answer_prefix, device)

        def bucket_positions(input_ids: list[int], token_texts: list[str]) -> tuple[str, list[int]]:
            del token_texts
            buckets = _llava_bucket_positions(input_ids, image_token_id)
            return "image_token_span", buckets["image_token_span"]

        hidden_index = args.layer + 1

    rows: list[dict[str, Any]] = []
    usable_runs = 0
    skipped: list[dict[str, Any]] = []

    for sample_id in sample_ids:
        sample = sample_lookup.get(sample_id)
        if sample is None:
            skipped.append({"sample_id": sample_id, "reason": "sample_missing_from_manifest"})
            continue
        mask_info = _load_masks(sample, roots, work_dir)
        if mask_info["status"] != "ok":
            skipped.append({"sample_id": sample_id, "reason": "mask_missing", "mask_info": mask_info})
            continue
        image_path = Path(mask_info["image_path"])
        clean_image = Image.open(image_path).convert("RGB")
        union_image = _apply_mask(clean_image, mask_info["union_mask"], (128, 128, 128))

        for prompt_name in prompt_names:
            question = _prompt(sample["question"], prompt_name)
            clean_inputs = build_inputs(clean_image, image_path, question)
            union_inputs = build_inputs(union_image, image_path, question)
            input_ids = clean_inputs["input_ids"][0].detach().cpu().tolist()
            token_texts = tokenizer.convert_ids_to_tokens(input_ids)
            bucket_name, visual_positions = bucket_positions(input_ids, token_texts)
            answer_positions = _answer_adjacent_positions(len(input_ids), set(visual_positions), args.answer_adjacent_count)
            target_candidates = _target_candidates(tokenizer, sample["answer"])
            target_ids = [item["token_id"] for item in target_candidates]
            if not visual_positions or not target_ids:
                skipped.append({"sample_id": sample_id, "prompt_name": prompt_name, "reason": "positions_or_target_missing"})
                continue

            with torch.inference_mode():
                clean_outputs = model(**clean_inputs, output_hidden_states=True, use_cache=False)
                union_outputs = model(**union_inputs, output_hidden_states=True, use_cache=False)
            clean_hidden = clean_outputs.hidden_states[hidden_index].detach()
            union_hidden = union_outputs.hidden_states[hidden_index].detach()
            clean_score = _rank_and_top(clean_outputs.logits, tokenizer, target_ids)
            chosen_target_id = int(clean_score["target_token_id"])
            union_score = _rank_and_top(union_outputs.logits, tokenizer, [chosen_target_id])
            clean_union_logit_gap = clean_score["target_logit"] - union_score["target_logit"]
            clean_union_rank_gap = union_score["target_rank"] - clean_score["target_rank"]

            groups, diagnostics = _make_groups(
                model_family=args.model_family,
                clean_inputs=clean_inputs,
                clean_hidden=clean_hidden,
                union_hidden=union_hidden,
                visual_positions=visual_positions,
                answer_positions=answer_positions,
                evidence_mask=mask_info["union_mask"],
                sample_id=sample_id,
                prompt_name=prompt_name,
                default_count=args.default_position_count,
                max_evidence_positions=args.max_evidence_positions,
                random_controls=args.random_controls,
            )
            payload["groups"].append(
                {
                    "sample_id": sample_id,
                    "prompt_name": prompt_name,
                    "bucket": bucket_name,
                    "diagnostics": diagnostics,
                    "groups": [
                        {
                            "group_name": group["group_name"],
                            "group_kind": group["group_kind"],
                            "position_count": len(group["positions"]),
                            "positions": group["positions"],
                        }
                        for group in groups
                    ],
                }
            )

            usable_runs += 1
            baseline_rows = [
                ("baseline", "clean", "baseline", [], clean_score),
                ("baseline", "union_mask", "baseline", [], union_score),
            ]
            for direction, condition, group_kind, positions, score in baseline_rows:
                rows.append(
                    {
                        "model_family": args.model_family,
                        "sample_id": sample_id,
                        "prompt_name": prompt_name,
                        "layer": args.layer,
                        "bucket": bucket_name,
                        "direction": direction,
                        "group_name": condition,
                        "group_kind": group_kind,
                        "condition": condition,
                        "scale": "",
                        "position_count": len(positions),
                        "target_answer": sample["answer"],
                        "target_token_id": score["target_token_id"],
                        "target_token": score["target_token"],
                        "target_logit": score["target_logit"],
                        "target_prob": score["target_prob"],
                        "target_rank": score["target_rank"],
                        "clean_target_logit": clean_score["target_logit"],
                        "clean_target_rank": clean_score["target_rank"],
                        "union_target_logit": union_score["target_logit"],
                        "union_target_rank": union_score["target_rank"],
                        "clean_union_logit_gap": clean_union_logit_gap,
                        "clean_union_rank_gap": clean_union_rank_gap,
                        "logit_restore_vs_union": score["target_logit"] - union_score["target_logit"],
                        "rank_restore_vs_union": union_score["target_rank"] - score["target_rank"],
                        "logit_damage_vs_clean": clean_score["target_logit"] - score["target_logit"],
                        "rank_damage_vs_clean": score["target_rank"] - clean_score["target_rank"],
                        "effect_logit": "",
                        "effect_rank": "",
                        "effect_gap_closure": "",
                        "top1_token": score["top1_token"],
                        "clean_top1_token": clean_score["top1_token"],
                        "union_top1_token": union_score["top1_token"],
                        "top1_changed_vs_reference": False,
                        "target_candidates": json.dumps(target_candidates, ensure_ascii=False),
                    }
                )

            for group in groups:
                group_positions = group["positions"]
                if not group_positions:
                    continue
                for scale in scales:
                    scale_label = _scale_label(scale)
                    restore_intervention = {
                        "positions": group_positions,
                        "source_hidden": clean_hidden,
                        "target_hidden": union_hidden,
                        "scale": scale,
                    }
                    corrupt_intervention = {
                        "positions": group_positions,
                        "source_hidden": union_hidden,
                        "target_hidden": clean_hidden,
                        "scale": scale,
                    }
                    for direction, inputs, reference, intervention in [
                        ("restore", union_inputs, union_score, restore_intervention),
                        ("corrupt", clean_inputs, clean_score, corrupt_intervention),
                    ]:
                        score = _score_with_patch(
                            model,
                            inputs,
                            tokenizer,
                            [chosen_target_id],
                            target_module,
                            intervention,
                        )
                        if direction == "restore":
                            effect_logit = score["target_logit"] - union_score["target_logit"]
                            effect_rank = union_score["target_rank"] - score["target_rank"]
                            effect_gap = _safe_ratio(effect_logit, clean_union_logit_gap)
                        else:
                            effect_logit = clean_score["target_logit"] - score["target_logit"]
                            effect_rank = score["target_rank"] - clean_score["target_rank"]
                            effect_gap = _safe_ratio(effect_logit, clean_union_logit_gap)
                        rows.append(
                            {
                                "model_family": args.model_family,
                                "sample_id": sample_id,
                                "prompt_name": prompt_name,
                                "layer": args.layer,
                                "bucket": bucket_name,
                                "direction": direction,
                                "group_name": group["group_name"],
                                "group_kind": group["group_kind"],
                                "condition": f"{direction}_{group['group_name']}_s{scale_label}",
                                "scale": scale,
                                "position_count": len(group_positions),
                                "target_answer": sample["answer"],
                                "target_token_id": score["target_token_id"],
                                "target_token": score["target_token"],
                                "target_logit": score["target_logit"],
                                "target_prob": score["target_prob"],
                                "target_rank": score["target_rank"],
                                "clean_target_logit": clean_score["target_logit"],
                                "clean_target_rank": clean_score["target_rank"],
                                "union_target_logit": union_score["target_logit"],
                                "union_target_rank": union_score["target_rank"],
                                "clean_union_logit_gap": clean_union_logit_gap,
                                "clean_union_rank_gap": clean_union_rank_gap,
                                "logit_restore_vs_union": score["target_logit"] - union_score["target_logit"],
                                "rank_restore_vs_union": union_score["target_rank"] - score["target_rank"],
                                "logit_damage_vs_clean": clean_score["target_logit"] - score["target_logit"],
                                "rank_damage_vs_clean": score["target_rank"] - clean_score["target_rank"],
                                "effect_logit": effect_logit,
                                "effect_rank": effect_rank,
                                "effect_gap_closure": effect_gap,
                                "top1_token": score["top1_token"],
                                "clean_top1_token": clean_score["top1_token"],
                                "union_top1_token": union_score["top1_token"],
                                "top1_changed_vs_reference": score["top1_token_id"] != reference["top1_token_id"],
                                "target_candidates": json.dumps(target_candidates, ensure_ascii=False),
                            }
                        )
            del clean_outputs, union_outputs, clean_hidden, union_hidden
            torch.cuda.empty_cache()

    payload["gpu_after"] = _gpu_info()
    payload["decision"] = {
        "status": "pass_hidden_position_patch_smoke" if usable_runs else "blocked_no_usable_runs",
        "usable_runs": usable_runs,
        "requested_runs": len(sample_ids) * len(prompt_names),
        "skipped": skipped,
    }
    fields = [
        "model_family",
        "sample_id",
        "prompt_name",
        "layer",
        "bucket",
        "direction",
        "group_name",
        "group_kind",
        "condition",
        "scale",
        "position_count",
        "target_answer",
        "target_token_id",
        "target_token",
        "target_logit",
        "target_prob",
        "target_rank",
        "clean_target_logit",
        "clean_target_rank",
        "union_target_logit",
        "union_target_rank",
        "clean_union_logit_gap",
        "clean_union_rank_gap",
        "logit_restore_vs_union",
        "rank_restore_vs_union",
        "logit_damage_vs_clean",
        "rank_damage_vs_clean",
        "effect_logit",
        "effect_rank",
        "effect_gap_closure",
        "top1_token",
        "clean_top1_token",
        "union_top1_token",
        "top1_changed_vs_reference",
        "target_candidates",
    ]
    _write_csv(Path(args.out_csv), rows, fields)
    _write_json(Path(args.out_json), payload)
    _log(f"done status={payload['decision']['status']} usable_runs={usable_runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
