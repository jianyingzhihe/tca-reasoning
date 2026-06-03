#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except Exception:
        return math.nan


def _safe_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _fmt(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.10g}"


def _device_batch(model, batch: dict) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(model.cfg.device) if torch.is_tensor(value) else value
    return out


def _infer_model_name_from_transcoder_set(repo_id: str) -> str:
    from huggingface_hub import hf_hub_download
    import yaml

    config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_name = (cfg or {}).get("model_name", "")
    if not model_name:
        raise ValueError(f"model_name missing in {repo_id}/config.yaml")
    return model_name


def _load_meta(outputs_root: Path, run_tag_base: str, bucket: str, run: str) -> dict[str, dict[str, str]]:
    suffix = "a" if run == "A" else "b"
    candidate_paths = [
        outputs_root / f"{run_tag_base}_{bucket}" / f"answer_aligned_meta_{suffix}.csv",
        outputs_root / run_tag_base / f"answer_aligned_meta_{suffix}.csv",
    ]
    for path in candidate_paths:
        if path.exists():
            return {row.get("sample_id", ""): row for row in _read_csv(path)}
    return {}


def _load_compare_nodes(outputs_root: Path, run_tag_base: str, bucket: str) -> list[dict[str, str]]:
    candidate_paths = [
        outputs_root / f"{run_tag_base}_{bucket}" / f"{run_tag_base}_{bucket}" / "nodes_detailed_controlled.csv",
        outputs_root / run_tag_base / "compare" / "nodes_detailed_controlled.csv",
    ]
    for path in candidate_paths:
        if path.exists():
            return _read_csv(path)
    return []


def _load_generic_features(path: Path | None) -> set[tuple[str, str, str]]:
    if path is None or not path.exists():
        return set()
    out = set()
    for row in _read_csv(path):
        if row.get("is_generic", "").lower() != "true":
            continue
        if row.get("node_type") != "feature":
            continue
        out.add((row.get("layer", ""), row.get("pos", ""), row.get("feature_or_token_id", "")))
    return out


def _pilot_node_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str, str]:
    return (
        row.get("bucket", ""),
        row.get("sample_id", ""),
        row.get("run", ""),
        row.get("node_role", ""),
        row.get("feature_layer", ""),
        row.get("feature_pos", ""),
        row.get("feature_id", ""),
    )


def _source_node_lookup_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("feature_layer", ""), row.get("feature_pos", ""), row.get("feature_id", ""))


def _match_label(source: dict[str, str], candidate: dict[str, str]) -> str:
    same_layer = source.get("layer") == candidate.get("layer")
    same_pos = source.get("pos") == candidate.get("pos")
    if same_layer and same_pos:
        return "same_layer_same_pos"
    if same_layer:
        return "same_layer"
    if same_pos:
        return "same_pos"
    return "fallback"


def _candidate_feature_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("layer", ""), row.get("pos", ""), row.get("feature_id", ""))


def _score_candidate(source: dict[str, str], candidate: dict[str, str]) -> tuple[float, float, float, str]:
    same_layer_penalty = 0.0 if source.get("layer") == candidate.get("layer") else 1.0
    same_pos_penalty = 0.0 if source.get("pos") == candidate.get("pos") else 1.0
    source_depth = _safe_float(source.get("depth_from_target"))
    cand_depth = _safe_float(candidate.get("depth_from_target"))
    depth_gap = abs(source_depth - cand_depth) if not math.isnan(source_depth) and not math.isnan(cand_depth) else 999.0

    source_mass = _safe_float(source.get("path_mass_best"))
    cand_mass = _safe_float(candidate.get("path_mass_best"))
    if source_mass > 0 and cand_mass > 0:
        mass_gap = abs(math.log10(source_mass) - math.log10(cand_mass))
    else:
        mass_gap = 999.0

    return (same_layer_penalty + same_pos_penalty * 0.25, mass_gap, depth_gap, candidate.get("node_id", ""))


def _build_candidate_pools(
    source: dict[str, str],
    candidates: list[dict[str, str]],
    excluded: set[tuple[str, str, str]],
) -> dict[str, list[dict[str, str]]]:
    pools: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        feature_key = _candidate_feature_key(row)
        if feature_key in excluded:
            continue
        pools[_match_label(source, row)].append(row)
    for rows in pools.values():
        rows.sort(key=lambda row: _score_candidate(source, row))
    return pools


def _pick_nearest_control_candidate(
    source: dict[str, str],
    candidates: list[dict[str, str]],
    excluded: set[tuple[str, str, str]],
) -> dict[str, str] | None:
    pools = _build_candidate_pools(source, candidates, excluded)
    for label in ("same_layer_same_pos", "same_pos", "same_layer", "fallback"):
        rows = pools.get(label, [])
        if rows:
            return rows[0]
    return None


def _pick_random_control_candidates(
    source: dict[str, str],
    candidates: list[dict[str, str]],
    excluded: set[tuple[str, str, str]],
    *,
    controls_per_source: int,
    rng: random.Random,
) -> tuple[str, list[dict[str, str]]]:
    pools = _build_candidate_pools(source, candidates, excluded)
    for label in ("same_layer_same_pos", "same_pos", "same_layer", "fallback"):
        rows = pools.get(label, [])
        if not rows:
            continue
        if len(rows) <= controls_per_source:
            return label, list(rows)
        picked = rng.sample(rows, controls_per_source)
        picked.sort(key=lambda row: _score_candidate(source, row))
        return label, picked
    return "fallback", []


def _condition_image(
    condition: str,
    original_image: Image.Image,
    wrong_image_path: str,
    *,
    mask_fraction: float,
    mask_fill_rgb: tuple[int, int, int],
) -> Image.Image:
    if condition == "clean":
        return original_image
    if condition == "no_image":
        return Image.new("RGB", original_image.size, color=(128, 128, 128))
    if condition == "wrong_image":
        return Image.open(wrong_image_path).convert("RGB")
    if condition == "masked_image":
        masked = original_image.copy()
        width, height = masked.size
        mask_w = max(1, int(round(width * mask_fraction)))
        mask_h = max(1, int(round(height * mask_fraction)))
        left = max(0, (width - mask_w) // 2)
        top = max(0, (height - mask_h) // 2)
        right = min(width, left + mask_w)
        bottom = min(height, top + mask_h)
        ImageDraw.Draw(masked).rectangle((left, top, right, bottom), fill=mask_fill_rgb)
        return masked
    raise ValueError(f"unknown condition: {condition}")


def _resolve_feature_pos(
    *,
    original_pos: int,
    clean_seq_len: int,
    current_seq_len: int,
    alignment_mode: str,
    max_position_shift: int,
) -> tuple[int | None, str]:
    if original_pos < current_seq_len:
        return original_pos, "exact"
    if alignment_mode != "relative_to_end":
        return None, "out_of_range"
    rel_from_end = (clean_seq_len - 1) - original_pos
    if rel_from_end < 0:
        return None, "invalid_clean_pos"
    mapped_pos = (current_seq_len - 1) - rel_from_end
    if mapped_pos < 0 or mapped_pos >= current_seq_len:
        return None, "mapped_out_of_range"
    if abs(mapped_pos - original_pos) > max_position_shift:
        return None, "mapped_shift_too_large"
    return mapped_pos, "relative_to_end"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run matched-control interventions for modality-counterfactual pilot nodes.")
    parser.add_argument("--run-tag-base", required=True)
    parser.add_argument("--pilot-csv", required=True)
    parser.add_argument("--outputs-root", default="outputs/phase_ab/ab_answer_aligned")
    parser.add_argument("--transcoder-set", default="tianhux2/gemma3-4b-it-plt")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16", "fp32", "bf16", "fp16"])
    parser.add_argument("--generic-nodes-csv", default="")
    parser.add_argument("--match-mode", default="nearest", choices=["nearest", "random"])
    parser.add_argument("--controls-per-source", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask-fraction", type=float, default=0.4)
    parser.add_argument("--mask-fill", default="128,128,128")
    parser.add_argument("--position-alignment", default="strict", choices=["strict", "relative_to_end"])
    parser.add_argument("--max-position-shift", type=int, default=8)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root).expanduser().resolve()
    pilot_rows = _read_csv(Path(args.pilot_csv).expanduser().resolve())
    if not pilot_rows:
        raise ValueError("no pilot rows loaded")
    if not 0.0 < args.mask_fraction <= 1.0:
        raise ValueError("--mask-fraction must be in (0, 1]")
    mask_fill_parts = [part.strip() for part in args.mask_fill.split(",")]
    if len(mask_fill_parts) != 3:
        raise ValueError("--mask-fill must have exactly 3 comma-separated integers")
    mask_fill_rgb = tuple(int(part) for part in mask_fill_parts)
    if any(v < 0 or v > 255 for v in mask_fill_rgb):
        raise ValueError("--mask-fill values must be between 0 and 255")

    generic_features = _load_generic_features(
        Path(args.generic_nodes_csv).expanduser().resolve() if args.generic_nodes_csv else None
    )

    rows_by_node: dict[tuple[str, ...], dict[str, dict[str, str]]] = defaultdict(dict)
    selected_feature_keys: dict[tuple[str, str, str], set[tuple[str, str, str]]] = defaultdict(set)
    for row in pilot_rows:
        key = _pilot_node_key(row)
        rows_by_node[key][row.get("condition", "")] = row
        sample_run_key = (row.get("bucket", ""), row.get("sample_id", ""), row.get("run", ""))
        selected_feature_keys[sample_run_key].add(_source_node_lookup_key(row))

    buckets = sorted({row.get("bucket", "") for row in pilot_rows})
    meta_a_by_bucket = {bucket: _load_meta(outputs_root, args.run_tag_base, bucket, "A") for bucket in buckets}
    meta_b_by_bucket = {bucket: _load_meta(outputs_root, args.run_tag_base, bucket, "B") for bucket in buckets}
    compare_nodes_by_bucket = {bucket: _load_compare_nodes(outputs_root, args.run_tag_base, bucket) for bucket in buckets}

    compare_lookup: dict[tuple[str, str, str, str, str, str], dict[str, str]] = {}
    sample_run_candidates: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for bucket, rows in compare_nodes_by_bucket.items():
        for row in rows:
            if row.get("node_type") != "feature":
                continue
            feature_key = (row.get("layer", ""), row.get("pos", ""), row.get("feature_id", ""))
            if feature_key in generic_features:
                continue
            lookup_key = (
                bucket,
                row.get("sample_id", ""),
                row.get("run", ""),
                row.get("layer", ""),
                row.get("pos", ""),
                row.get("feature_id", ""),
            )
            compare_lookup[lookup_key] = row
            sample_run_candidates[(bucket, row.get("sample_id", ""), row.get("run", ""))].append(row)

    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    dtype = dtype_map[args.dtype]

    from circuit_tracer import ReplacementModel
    from circuit_tracer.attribution.attribute import _build_multimodal_batch

    model_name = args.model_name or _infer_model_name_from_transcoder_set(args.transcoder_set)
    print(f"[info] loading model={model_name} transcoder_set={args.transcoder_set} dtype={dtype}")
    model = ReplacementModel.from_pretrained(
        model_name,
        args.transcoder_set,
        dtype=dtype,
        lazy_encoder=True,
        lazy_decoder=True,
    )

    results: list[dict[str, str]] = []
    rng = random.Random(args.seed)
    used_control_keys: dict[tuple[str, str, str], set[tuple[str, str, str]]] = defaultdict(set)
    for node_key, by_condition in sorted(rows_by_node.items()):
        bucket, sample_id, run, node_role, source_layer, source_pos, source_feature_id = node_key
        source_compare_row = compare_lookup.get((bucket, sample_id, run, source_layer, source_pos, source_feature_id))
        if source_compare_row is None:
            print(f"[skip] bucket={bucket} sample={sample_id} run={run} missing source node in compare csv")
            continue

        sample_run_key = (bucket, sample_id, run)
        base_excluded_controls = set(selected_feature_keys.get(sample_run_key, set()))
        if args.match_mode == "nearest":
            excluded_controls = set(base_excluded_controls)
            excluded_controls.update(used_control_keys.get(sample_run_key, set()))
            control_rows = []
            control_row = _pick_nearest_control_candidate(
                source_compare_row,
                sample_run_candidates.get(sample_run_key, []),
                excluded_controls,
            )
            if control_row is None:
                print(f"[skip] bucket={bucket} sample={sample_id} run={run} no control candidate found")
                continue
            used_control_keys[sample_run_key].add(_candidate_feature_key(control_row))
            control_rows = [control_row]
            sampled_match_label = _match_label(source_compare_row, control_row)
        else:
            sampled_match_label, control_rows = _pick_random_control_candidates(
                source_compare_row,
                sample_run_candidates.get(sample_run_key, []),
                base_excluded_controls,
                controls_per_source=max(1, args.controls_per_source),
                rng=rng,
            )
            if not control_rows:
                print(f"[skip] bucket={bucket} sample={sample_id} run={run} no random control candidates found")
                continue

        meta_map = meta_a_by_bucket[bucket] if run == "A" else meta_b_by_bucket[bucket]
        meta = meta_map.get(sample_id, {})
        target_token_id = _safe_int(meta.get("target_token_id") or by_condition.get("clean", {}).get("target_token_id"))
        image_path = (meta.get("image_path") or by_condition.get("clean", {}).get("image_path") or "").strip()
        question = meta.get("question", by_condition.get("clean", {}).get("question", ""))
        assistant_prefix = meta.get("assistant_prefix", "")
        if target_token_id is None or not image_path:
            print(f"[skip] bucket={bucket} sample={sample_id} run={run} missing target or image path")
            continue

        original_image = Image.open(image_path).convert("RGB")
        clean_batch = _build_multimodal_batch(
            model.processor,
            original_image,
            f"<start_of_image> {question}",
            assistant_prefix=assistant_prefix,
        )
        clean_seq_len = int(clean_batch["input_ids"].shape[1])
        control_pool_size = len(control_rows)
        for control_idx, control_row in enumerate(control_rows, start=1):
            control_layer = int(control_row["layer"])
            control_pos = int(control_row["pos"])
            control_feature_id = int(control_row["feature_id"])
            match_label = _match_label(source_compare_row, control_row)

            for condition, pilot_row in sorted(by_condition.items()):
                wrong_image_path = pilot_row.get("condition_image_path", "") if condition == "wrong_image" else ""
                condition_image = _condition_image(
                    condition,
                    original_image,
                    wrong_image_path,
                    mask_fraction=args.mask_fraction,
                    mask_fill_rgb=mask_fill_rgb,
                )
                batch = _build_multimodal_batch(
                    model.processor,
                    condition_image,
                    f"<start_of_image> {question}",
                    assistant_prefix=assistant_prefix,
                )
                batch["image"] = condition_image
                batch = _device_batch(model, batch)
                seq_len = int(batch["input_ids"].shape[1])
                applied_source_pos, source_alignment_status = _resolve_feature_pos(
                    original_pos=int(source_pos),
                    clean_seq_len=clean_seq_len,
                    current_seq_len=seq_len,
                    alignment_mode=args.position_alignment,
                    max_position_shift=args.max_position_shift,
                )
                applied_control_pos, control_alignment_status = _resolve_feature_pos(
                    original_pos=control_pos,
                    clean_seq_len=clean_seq_len,
                    current_seq_len=seq_len,
                    alignment_mode=args.position_alignment,
                    max_position_shift=args.max_position_shift,
                )
                if applied_source_pos is None or applied_control_pos is None:
                    print(
                        f"[skip] bucket={bucket} sample={sample_id} run={run} condition={condition} role={node_role} "
                        f"source_status={source_alignment_status} control_status={control_alignment_status} "
                        f"clean_seq_len={clean_seq_len} seq_len={seq_len}"
                    )
                    continue

                with torch.inference_mode():
                    original_logits = model.forward_from_batch(batch)
                    intervened_logits, _ = model.feature_intervention(
                        batch,
                        [(control_layer, applied_control_pos, control_feature_id, 0.0)],
                        freeze_attention=True,
                        apply_activation_function=True,
                        sparse=False,
                    )

                last_pos = original_logits.shape[1] - 1
                original_target_logit = float(original_logits[0, last_pos, target_token_id].item())
                intervened_target_logit = float(intervened_logits[0, last_pos, target_token_id].item())
                delta_target_logit = intervened_target_logit - original_target_logit

                original_probs = torch.softmax(original_logits[0, last_pos], dim=-1)
                intervened_probs = torch.softmax(intervened_logits[0, last_pos], dim=-1)
                original_target_prob = float(original_probs[target_token_id].item())
                intervened_target_prob = float(intervened_probs[target_token_id].item())
                delta_target_prob = intervened_target_prob - original_target_prob

                top1_before = int(torch.argmax(original_logits[0, last_pos]).item())
                top1_after = int(torch.argmax(intervened_logits[0, last_pos]).item())

                results.append(
                    {
                        "bucket": bucket,
                        "sample_id": sample_id,
                        "run": run,
                        "condition": condition,
                        "node_role": node_role,
                        "question": question,
                        "image_path": image_path,
                        "target_token_id": str(target_token_id),
                        "match_mode": args.match_mode,
                        "control_draw_idx": str(control_idx),
                        "control_pool_size": str(control_pool_size),
                        "sampled_match_label": sampled_match_label,
                        "source_feature_layer": source_layer,
                        "source_feature_pos": source_pos,
                        "applied_source_feature_pos": str(applied_source_pos),
                        "source_position_alignment": source_alignment_status,
                        "source_feature_id": source_feature_id,
                        "source_path_mass_best": _fmt(_safe_float(source_compare_row.get("path_mass_best"))),
                        "source_depth_from_target": source_compare_row.get("depth_from_target", ""),
                        "control_feature_layer": control_row.get("layer", ""),
                        "control_feature_pos": control_row.get("pos", ""),
                        "applied_control_feature_pos": str(applied_control_pos),
                        "control_position_alignment": control_alignment_status,
                        "clean_seq_len": str(clean_seq_len),
                        "condition_seq_len": str(seq_len),
                        "control_feature_id": control_row.get("feature_id", ""),
                        "control_path_mass_best": _fmt(_safe_float(control_row.get("path_mass_best"))),
                        "control_depth_from_target": control_row.get("depth_from_target", ""),
                        "control_match_label": match_label,
                        "original_target_logit": _fmt(original_target_logit),
                        "intervened_target_logit": _fmt(intervened_target_logit),
                        "delta_target_logit": _fmt(delta_target_logit),
                        "original_target_prob": _fmt(original_target_prob),
                        "intervened_target_prob": _fmt(intervened_target_prob),
                        "delta_target_prob": _fmt(delta_target_prob),
                        "top1_before_id": str(top1_before),
                        "top1_after_id": str(top1_after),
                        "top1_before_token": model.processor.tokenizer.convert_ids_to_tokens([top1_before])[0],
                        "top1_after_token": model.processor.tokenizer.convert_ids_to_tokens([top1_after])[0],
                    }
                )
                print(
                    f"[done] bucket={bucket} sample={sample_id} run={run} condition={condition} role={node_role} "
                    f"source=L{source_layer}:P{source_pos}->P{applied_source_pos}:F{source_feature_id} "
                    f"control#{control_idx}=L{control_layer}:P{control_pos}->P{applied_control_pos}:F{control_feature_id} "
                    f"source_align={source_alignment_status} control_align={control_alignment_status} "
                    f"match_mode={args.match_mode} delta_target_logit={delta_target_logit:.4f}"
                )

    _write_csv(
        Path(args.out_csv).expanduser().resolve(),
        results,
        [
            "bucket",
            "sample_id",
            "run",
            "condition",
            "node_role",
            "question",
            "image_path",
            "target_token_id",
            "match_mode",
            "control_draw_idx",
            "control_pool_size",
            "sampled_match_label",
            "source_feature_layer",
            "source_feature_pos",
            "applied_source_feature_pos",
            "source_position_alignment",
            "source_feature_id",
            "source_path_mass_best",
            "source_depth_from_target",
            "control_feature_layer",
            "control_feature_pos",
            "applied_control_feature_pos",
            "control_position_alignment",
            "clean_seq_len",
            "condition_seq_len",
            "control_feature_id",
            "control_path_mass_best",
            "control_depth_from_target",
            "control_match_label",
            "original_target_logit",
            "intervened_target_logit",
            "delta_target_logit",
            "original_target_prob",
            "intervened_target_prob",
            "delta_target_prob",
            "top1_before_id",
            "top1_after_id",
            "top1_before_token",
            "top1_after_token",
        ],
    )
    print(f"[done] out_csv={Path(args.out_csv).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
