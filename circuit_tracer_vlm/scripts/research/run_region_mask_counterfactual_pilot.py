#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


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


def _safe_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


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


def _device_batch(model, batch: dict) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(model.cfg.device) if torch.is_tensor(value) else value
    return out


def _load_meta(outputs_root: Path, run_tag_base: str, bucket: str, run: str) -> dict[str, dict[str, str]]:
    suffix = "a" if run == "A" else "b"
    path = outputs_root / f"{run_tag_base}_{bucket}" / f"answer_aligned_meta_{suffix}.csv"
    if not path.exists():
        return {}
    return {row.get("sample_id", ""): row for row in _read_csv(path)}


def _parse_fill(fill: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in fill.split(",")]
    if len(parts) != 3:
        raise ValueError("--mask-fill must be r,g,b")
    if any(v < 0 or v > 255 for v in parts):
        raise ValueError("--mask-fill values must be in [0,255]")
    return tuple(parts)


def _load_mask(mask_root: Path, image_path: str, label: str) -> Image.Image:
    stem = Path(image_path).stem
    path = mask_root / stem / f"{label}.png"
    if not path.exists():
        raise FileNotFoundError(f"missing mask {path}")
    return Image.open(path).convert("L")


def _mask_array(mask_image: Image.Image) -> np.ndarray:
    arr = np.asarray(mask_image, dtype=np.uint8)
    return arr > 0


def _apply_mask(original: Image.Image, mask_bool: np.ndarray, fill_rgb: tuple[int, int, int]) -> Image.Image:
    out = np.asarray(original.convert("RGB")).copy()
    out[mask_bool] = np.array(fill_rgb, dtype=np.uint8)
    return Image.fromarray(out, mode="RGB")


def _bbox_from_mask(mask_bool: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _make_shifted_control_mask(reference_mask: np.ndarray, avoid_mask: np.ndarray) -> np.ndarray:
    bbox = _bbox_from_mask(reference_mask)
    h, w = reference_mask.shape
    control = np.zeros_like(reference_mask, dtype=bool)
    if bbox is None:
        return control

    x0, y0, x1, y1 = bbox
    ref_crop = reference_mask[y0:y1, x0:x1]
    crop_h, crop_w = ref_crop.shape
    candidates = [
        (0, 0),
        (w - crop_w, 0),
        (0, h - crop_h),
        (w - crop_w, h - crop_h),
        (max(0, (w - crop_w) // 2), 0),
        (max(0, (w - crop_w) // 2), h - crop_h),
        (0, max(0, (h - crop_h) // 2)),
        (w - crop_w, max(0, (h - crop_h) // 2)),
    ]

    best_mask = None
    best_score = None
    for tx, ty in candidates:
        tx = max(0, min(tx, w - crop_w))
        ty = max(0, min(ty, h - crop_h))
        candidate = np.zeros_like(reference_mask, dtype=bool)
        candidate[ty : ty + crop_h, tx : tx + crop_w] = ref_crop
        overlap = np.logical_and(candidate, avoid_mask).sum()
        score = (int(overlap), abs(tx - x0) + abs(ty - y0))
        if best_score is None or score < best_score:
            best_score = score
            best_mask = candidate

    if best_mask is None:
        return control
    return best_mask


def _target_stats(logits: torch.Tensor, target_token_id: int, tokenizer) -> dict[str, str]:
    last_pos = logits.shape[1] - 1
    target_logit = float(logits[0, last_pos, target_token_id].item())
    probs = torch.softmax(logits[0, last_pos], dim=-1)
    target_prob = float(probs[target_token_id].item())
    top1_id = int(torch.argmax(logits[0, last_pos]).item())
    return {
        "target_logit": f"{target_logit:.10g}",
        "target_prob": f"{target_prob:.10g}",
        "top1_id": str(top1_id),
        "top1_token": tokenizer.convert_ids_to_tokens([top1_id])[0],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a label-region-based counterfactual intervention pilot.")
    parser.add_argument("--run-tag-base", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--mask-root", required=True, help="Directory containing exported_masks/<image_stem>/<label>.png")
    parser.add_argument("--outputs-root", default="outputs/phase_ab/ab_answer_aligned")
    parser.add_argument("--transcoder-set", default="tianhux2/gemma3-4b-it-plt")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16", "fp32", "bf16", "fp16"])
    parser.add_argument("--conditions", default="clean,answer_mask,relate_mask,union_mask,auto_control_mask")
    parser.add_argument("--mask-fill", default="128,128,128")
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    manifest_rows = _read_csv(Path(args.manifest_csv).expanduser().resolve())
    if not manifest_rows:
        raise ValueError("no manifest rows loaded")

    mask_root = Path(args.mask_root).expanduser().resolve()
    outputs_root = Path(args.outputs_root).expanduser().resolve()
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    fill_rgb = _parse_fill(args.mask_fill)

    buckets = sorted({row.get("bucket", "") for row in manifest_rows})
    meta_a_by_bucket = {bucket: _load_meta(outputs_root, args.run_tag_base, bucket, "A") for bucket in buckets}
    meta_b_by_bucket = {bucket: _load_meta(outputs_root, args.run_tag_base, bucket, "B") for bucket in buckets}

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

    tokenizer = model.processor.tokenizer
    results: list[dict[str, str]] = []

    for row in manifest_rows:
        bucket = row.get("bucket", "")
        sample_id = row.get("sample_id", "")
        run = row.get("run", "")
        meta_map = meta_a_by_bucket[bucket] if run == "A" else meta_b_by_bucket[bucket]
        meta = meta_map.get(sample_id, {})
        target_token_id = _safe_int(meta.get("target_token_id") or "")
        if target_token_id is None:
            print(f"[skip] bucket={bucket} sample={sample_id} run={run} missing target_token_id")
            continue

        image_path = (row.get("remote_image_path") or meta.get("image_path") or "").strip()
        question = row.get("question", "") or meta.get("question", "")
        assistant_prefix = meta.get("assistant_prefix", "")
        if not image_path:
            print(f"[skip] bucket={bucket} sample={sample_id} run={run} missing image path")
            continue

        layer = int(row.get("feature_layer", "0"))
        pos = int(row.get("feature_pos", "0"))
        feature_id = int(row.get("feature_id", "0"))

        original_image = Image.open(image_path).convert("RGB")
        answer_mask_img = _load_mask(mask_root, image_path, "answer")
        relate_mask_img = _load_mask(mask_root, image_path, "relate")
        answer_mask = _mask_array(answer_mask_img)
        relate_mask = _mask_array(relate_mask_img)
        union_mask = np.logical_or(answer_mask, relate_mask)
        control_mask = _make_shifted_control_mask(answer_mask, union_mask)

        condition_to_mask = {
            "clean": None,
            "answer_mask": answer_mask,
            "relate_mask": relate_mask,
            "union_mask": union_mask,
            "auto_control_mask": control_mask,
        }

        for condition in conditions:
            if condition not in condition_to_mask:
                raise ValueError(f"unsupported condition: {condition}")
            mask_bool = condition_to_mask[condition]
            condition_image = (
                original_image if mask_bool is None else _apply_mask(original_image, mask_bool, fill_rgb)
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
            if pos >= seq_len:
                print(
                    f"[skip] bucket={bucket} sample={sample_id} run={run} condition={condition} "
                    f"feature=L{layer}:P{pos}:F{feature_id} position_out_of_range seq_len={seq_len}"
                )
                continue

            with torch.inference_mode():
                original_logits = model.forward_from_batch(batch)
                intervened_logits, _ = model.feature_intervention(
                    batch,
                    [(layer, pos, feature_id, 0.0)],
                    freeze_attention=True,
                    apply_activation_function=True,
                    sparse=False,
                )

            before = _target_stats(original_logits, target_token_id, tokenizer)
            after = _target_stats(intervened_logits, target_token_id, tokenizer)
            original_target_logit = float(before["target_logit"])
            intervened_target_logit = float(after["target_logit"])
            original_target_prob = float(before["target_prob"])
            intervened_target_prob = float(after["target_prob"])
            delta_target_logit = intervened_target_logit - original_target_logit
            delta_target_prob = intervened_target_prob - original_target_prob

            results.append(
                {
                    "bucket": bucket,
                    "sample_id": sample_id,
                    "priority": row.get("priority", ""),
                    "run": run,
                    "node_role": row.get("node_role", ""),
                    "condition": condition,
                    "question": question,
                    "image_path": image_path,
                    "feature_layer": str(layer),
                    "feature_pos": str(pos),
                    "feature_id": str(feature_id),
                    "target_token_id": str(target_token_id),
                    "mask_fill_rgb": args.mask_fill,
                    "answer_area_px": str(int(answer_mask.sum())),
                    "relate_area_px": str(int(relate_mask.sum())),
                    "union_area_px": str(int(union_mask.sum())),
                    "control_area_px": str(int(control_mask.sum())),
                    "original_target_logit": f"{original_target_logit:.10g}",
                    "intervened_target_logit": f"{intervened_target_logit:.10g}",
                    "delta_target_logit": f"{delta_target_logit:.10g}",
                    "original_target_prob": f"{original_target_prob:.10g}",
                    "intervened_target_prob": f"{intervened_target_prob:.10g}",
                    "delta_target_prob": f"{delta_target_prob:.10g}",
                    "top1_before_id": before["top1_id"],
                    "top1_before_token": before["top1_token"],
                    "top1_after_id": after["top1_id"],
                    "top1_after_token": after["top1_token"],
                }
            )
            print(
                f"[done] bucket={bucket} sample={sample_id} run={run} role={row.get('node_role','')} "
                f"condition={condition} feature=L{layer}:P{pos}:F{feature_id} "
                f"delta_target_logit={delta_target_logit:.4f} delta_target_prob={delta_target_prob:.6f}"
            )

    out_path = Path(args.out_csv).expanduser().resolve()
    _write_csv(
        out_path,
        results,
        [
            "bucket",
            "sample_id",
            "priority",
            "run",
            "node_role",
            "condition",
            "question",
            "image_path",
            "feature_layer",
            "feature_pos",
            "feature_id",
            "target_token_id",
            "mask_fill_rgb",
            "answer_area_px",
            "relate_area_px",
            "union_area_px",
            "control_area_px",
            "original_target_logit",
            "intervened_target_logit",
            "delta_target_logit",
            "original_target_prob",
            "intervened_target_prob",
            "delta_target_prob",
            "top1_before_id",
            "top1_before_token",
            "top1_after_id",
            "top1_after_token",
        ],
    )
    print(f"[done] out_csv={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
