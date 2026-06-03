#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
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


def _pick_candidate_rows(
    smoke_rows: list[dict[str, str]],
    *,
    max_samples_per_bucket: int,
    top_support_per_run: int,
    top_suppressor_per_run: int,
) -> list[dict[str, str]]:
    rows_by_bucket_sample: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in smoke_rows:
        key = (row.get("bucket", ""), row.get("sample_id", ""))
        rows_by_bucket_sample[key].append(row)

    selected_samples: dict[str, list[str]] = defaultdict(list)
    bucket_scores: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for (bucket, sample_id), rows in rows_by_bucket_sample.items():
        best_support = min((_safe_float(r.get("delta_target_logit")) for r in rows), default=math.nan)
        if math.isnan(best_support):
            continue
        bucket_scores[bucket].append((best_support, sample_id))

    for bucket, scored in bucket_scores.items():
        scored.sort(key=lambda item: item[0])
        selected_samples[bucket] = [sample_id for _, sample_id in scored[:max_samples_per_bucket]]

    selected_rows: list[dict[str, str]] = []
    rows_by_sample_run: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in smoke_rows:
        bucket = row.get("bucket", "")
        sample_id = row.get("sample_id", "")
        if sample_id not in selected_samples.get(bucket, []):
            continue
        key = (bucket, sample_id, row.get("run", ""))
        rows_by_sample_run[key].append(row)

    for key, rows in rows_by_sample_run.items():
        support_rows = [r for r in rows if _safe_float(r.get("delta_target_logit")) < 0]
        support_rows.sort(key=lambda r: _safe_float(r.get("delta_target_logit")))
        for row in support_rows[:top_support_per_run]:
            picked = dict(row)
            picked["node_role"] = "support"
            selected_rows.append(picked)

        suppressor_rows = [r for r in rows if _safe_float(r.get("delta_target_logit")) > 0]
        suppressor_rows.sort(key=lambda r: _safe_float(r.get("delta_target_logit")), reverse=True)
        for row in suppressor_rows[:top_suppressor_per_run]:
            picked = dict(row)
            picked["node_role"] = "suppressor"
            selected_rows.append(picked)

    return selected_rows


def _make_wrong_image_map(
    selected_rows: list[dict[str, str]],
    meta_a_by_bucket: dict[str, dict[str, dict[str, str]]],
) -> dict[str, tuple[str, str]]:
    sample_to_image: dict[str, tuple[str, str]] = {}
    ordered_sample_ids: list[str] = []
    seen = set()
    for row in selected_rows:
        bucket = row.get("bucket", "")
        sample_id = row.get("sample_id", "")
        if sample_id in seen:
            continue
        meta = meta_a_by_bucket.get(bucket, {}).get(sample_id, {})
        image_path = (meta.get("image_path") or row.get("image_path") or "").strip()
        if not image_path:
            continue
        ordered_sample_ids.append(sample_id)
        sample_to_image[sample_id] = (sample_id, image_path)
        seen.add(sample_id)

    if len(ordered_sample_ids) < 2:
        return {}

    wrong_map: dict[str, tuple[str, str]] = {}
    for idx, sample_id in enumerate(ordered_sample_ids):
        donor = ordered_sample_ids[(idx + 1) % len(ordered_sample_ids)]
        wrong_map[sample_id] = sample_to_image[donor]
    return wrong_map


def _condition_image(
    condition: str,
    original_image: Image.Image,
    wrong_image: Image.Image | None,
    *,
    mask_fraction: float,
    mask_fill_rgb: tuple[int, int, int],
) -> Image.Image | None:
    if condition == "clean":
        return original_image
    if condition == "no_image":
        return Image.new("RGB", original_image.size, color=(128, 128, 128))
    if condition == "wrong_image":
        return wrong_image if wrong_image is not None else None
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
    parser = argparse.ArgumentParser(
        description="Run a small prompt-modality counterfactual intervention pilot on selected support/suppressor nodes."
    )
    parser.add_argument("--run-tag-base", required=True)
    parser.add_argument("--smoke-csvs", nargs="+", required=True)
    parser.add_argument("--outputs-root", default="outputs/phase_ab/ab_answer_aligned")
    parser.add_argument("--transcoder-set", default="tianhux2/gemma3-4b-it-plt")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16", "fp32", "bf16", "fp16"])
    parser.add_argument("--conditions", default="clean,no_image,wrong_image")
    parser.add_argument("--mask-fraction", type=float, default=0.4)
    parser.add_argument("--mask-fill", default="128,128,128", help="RGB fill for masked_image, e.g. 128,128,128")
    parser.add_argument("--max-samples-per-bucket", type=int, default=4)
    parser.add_argument("--top-support-per-run", type=int, default=1)
    parser.add_argument("--top-suppressor-per-run", type=int, default=1)
    parser.add_argument("--position-alignment", default="strict", choices=["strict", "relative_to_end"])
    parser.add_argument("--max-position-shift", type=int, default=8)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root).expanduser().resolve()
    smoke_rows: list[dict[str, str]] = []
    for smoke_csv in args.smoke_csvs:
        smoke_rows.extend(_read_csv(Path(smoke_csv).expanduser().resolve()))
    if not smoke_rows:
        raise ValueError("no smoke rows loaded")

    conditions = [part.strip() for part in args.conditions.split(",") if part.strip()]
    if not 0.0 < args.mask_fraction <= 1.0:
        raise ValueError("--mask-fraction must be in (0, 1]")
    mask_fill_parts = [part.strip() for part in args.mask_fill.split(",")]
    if len(mask_fill_parts) != 3:
        raise ValueError("--mask-fill must have exactly 3 comma-separated integers")
    mask_fill_rgb = tuple(int(part) for part in mask_fill_parts)
    if any(v < 0 or v > 255 for v in mask_fill_rgb):
        raise ValueError("--mask-fill values must be between 0 and 255")

    selected_rows = _pick_candidate_rows(
        smoke_rows,
        max_samples_per_bucket=args.max_samples_per_bucket,
        top_support_per_run=args.top_support_per_run,
        top_suppressor_per_run=args.top_suppressor_per_run,
    )
    if not selected_rows:
        raise ValueError("no candidate support/suppressor rows selected")

    buckets = sorted({row.get("bucket", "") for row in selected_rows})
    meta_a_by_bucket = {bucket: _load_meta(outputs_root, args.run_tag_base, bucket, "A") for bucket in buckets}
    meta_b_by_bucket = {bucket: _load_meta(outputs_root, args.run_tag_base, bucket, "B") for bucket in buckets}
    wrong_image_map = _make_wrong_image_map(selected_rows, meta_a_by_bucket)

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
    for row in selected_rows:
        bucket = row["bucket"]
        sample_id = row["sample_id"]
        run = row["run"]
        meta_map = meta_a_by_bucket[bucket] if run == "A" else meta_b_by_bucket[bucket]
        meta = meta_map.get(sample_id, {})
        question = meta.get("question", row.get("question", ""))
        assistant_prefix = meta.get("assistant_prefix", "")
        target_token_id = _safe_int(meta.get("target_token_id") or row.get("target_token_id"))
        image_path = (meta.get("image_path") or row.get("image_path") or "").strip()
        if target_token_id is None or not image_path:
            print(f"[skip] bucket={bucket} sample={sample_id} run={run} missing target or image path")
            continue

        original_image = Image.open(image_path).convert("RGB")
        wrong_sample_id, wrong_image_path = wrong_image_map.get(sample_id, ("", ""))
        wrong_image = Image.open(wrong_image_path).convert("RGB") if wrong_image_path else None

        layer = int(row["feature_layer"])
        pos = int(row["feature_pos"])
        feature_id = int(row["feature_id"])
        reference_clean_delta = _safe_float(row.get("delta_target_logit"))
        clean_batch = _build_multimodal_batch(
            model.processor,
            original_image,
            f"<start_of_image> {question}",
            assistant_prefix=assistant_prefix,
        )
        clean_seq_len = int(clean_batch["input_ids"].shape[1])

        for condition in conditions:
            condition_image = _condition_image(
                condition,
                original_image,
                wrong_image,
                mask_fraction=args.mask_fraction,
                mask_fill_rgb=mask_fill_rgb,
            )
            if condition_image is None:
                print(f"[skip] bucket={bucket} sample={sample_id} condition={condition} unavailable")
                continue

            batch = _build_multimodal_batch(
                model.processor,
                condition_image,
                f"<start_of_image> {question}",
                assistant_prefix=assistant_prefix,
            )
            batch["image"] = condition_image
            batch = _device_batch(model, batch)
            seq_len = int(batch["input_ids"].shape[1])
            applied_pos, alignment_status = _resolve_feature_pos(
                original_pos=pos,
                clean_seq_len=clean_seq_len,
                current_seq_len=seq_len,
                alignment_mode=args.position_alignment,
                max_position_shift=args.max_position_shift,
            )
            if applied_pos is None:
                print(
                    f"[skip] bucket={bucket} sample={sample_id} run={run} condition={condition} "
                    f"role={row.get('node_role','')} feature=L{layer}:P{pos}:F{feature_id} "
                    f"position_unusable status={alignment_status} clean_seq_len={clean_seq_len} seq_len={seq_len}"
                )
                continue

            try:
                with torch.inference_mode():
                    original_logits = model.forward_from_batch(batch)
                    intervened_logits, _ = model.feature_intervention(
                        batch,
                        [(layer, applied_pos, feature_id, 0.0)],
                        freeze_attention=True,
                        apply_activation_function=True,
                        sparse=False,
                    )
            except IndexError as exc:
                print(
                    f"[skip] bucket={bucket} sample={sample_id} run={run} condition={condition} "
                    f"role={row.get('node_role','')} feature=L{layer}:P{pos}:F{feature_id} "
                    f"index_error={exc}"
                )
                continue

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
                    "node_role": row.get("node_role", ""),
                    "followup_visual_type_label": row.get("followup_visual_type_label", ""),
                    "followup_knowledge_level_label": row.get("followup_knowledge_level_label", ""),
                    "followup_priority": row.get("followup_priority", ""),
                    "followup_display_question": row.get("followup_display_question", ""),
                    "wrong_image_sample_id": wrong_sample_id if condition == "wrong_image" else "",
                    "mask_fraction": f"{args.mask_fraction:.10g}" if condition == "masked_image" else "",
                    "mask_fill_rgb": args.mask_fill if condition == "masked_image" else "",
                    "question": question,
                    "image_path": image_path,
                    "condition_image_path": wrong_image_path if condition == "wrong_image" else image_path,
                    "target_token_id": str(target_token_id),
                    "feature_layer": str(layer),
                    "feature_pos": str(pos),
                    "applied_feature_pos": str(applied_pos),
                    "position_alignment": alignment_status,
                    "clean_seq_len": str(clean_seq_len),
                    "condition_seq_len": str(seq_len),
                    "position_shift": str(applied_pos - pos),
                    "feature_id": str(feature_id),
                    "reference_clean_delta_target_logit": f"{reference_clean_delta:.10g}" if not math.isnan(reference_clean_delta) else "",
                    "original_target_logit": f"{original_target_logit:.10g}",
                    "intervened_target_logit": f"{intervened_target_logit:.10g}",
                    "delta_target_logit": f"{delta_target_logit:.10g}",
                    "original_target_prob": f"{original_target_prob:.10g}",
                    "intervened_target_prob": f"{intervened_target_prob:.10g}",
                    "delta_target_prob": f"{delta_target_prob:.10g}",
                    "top1_before_id": str(top1_before),
                    "top1_after_id": str(top1_after),
                    "top1_before_token": model.processor.tokenizer.convert_ids_to_tokens([top1_before])[0],
                    "top1_after_token": model.processor.tokenizer.convert_ids_to_tokens([top1_after])[0],
                }
            )
            print(
                f"[done] bucket={bucket} sample={sample_id} run={run} condition={condition} "
                f"role={row.get('node_role','')} feature=L{layer}:P{pos}->P{applied_pos}:F{feature_id} "
                f"align={alignment_status} "
                f"delta_target_logit={delta_target_logit:.4f} delta_target_prob={delta_target_prob:.6f}"
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
            "followup_visual_type_label",
            "followup_knowledge_level_label",
            "followup_priority",
            "followup_display_question",
            "wrong_image_sample_id",
            "mask_fraction",
            "mask_fill_rgb",
            "question",
            "image_path",
            "condition_image_path",
            "target_token_id",
            "feature_layer",
            "feature_pos",
            "applied_feature_pos",
            "position_alignment",
            "clean_seq_len",
            "condition_seq_len",
            "position_shift",
            "feature_id",
            "reference_clean_delta_target_logit",
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
