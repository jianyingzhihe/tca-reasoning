#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_CONDITIONS = [
    "clean",
    "answer_mask",
    "relate_mask",
    "union_mask",
    "random_control_1",
    "random_control_2",
    "random_control_3",
    "random_control_4",
]


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


def _parse_fill(fill: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in fill.split(",")]
    if len(parts) != 3:
        raise ValueError("--mask-fill must be r,g,b")
    if any(v < 0 or v > 255 for v in parts):
        raise ValueError("--mask-fill values must be in [0,255]")
    return tuple(parts)


def _mask_array(mask_image: Image.Image) -> np.ndarray:
    arr = np.asarray(mask_image, dtype=np.uint8)
    return arr > 0


def _load_mask(mask_root: Path, image_path: str, label: str) -> Image.Image:
    stem = Path(image_path).stem
    path = mask_root / stem / f"{label}.png"
    if not path.exists():
        raise FileNotFoundError(f"missing mask {path}")
    return Image.open(path).convert("L")


def _apply_mask(original: Image.Image, mask_bool: np.ndarray, fill_rgb: tuple[int, int, int]) -> Image.Image:
    out = np.asarray(original.convert("RGB")).copy()
    out[mask_bool] = np.array(fill_rgb, dtype=np.uint8)
    return Image.fromarray(out, mode="RGB")


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    if union <= 0:
        return 0.0
    return inter / union


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _build_area_matched_rectangle(reference_mask: np.ndarray, x0: int, y0: int) -> np.ndarray | None:
    target_area = int(reference_mask.sum())
    if target_area <= 0:
        return np.zeros_like(reference_mask, dtype=bool)

    h, w = reference_mask.shape
    bbox = _bbox_from_mask(reference_mask)
    if bbox is None:
        return np.zeros_like(reference_mask, dtype=bool)
    bx0, by0, bx1, by1 = bbox
    bbox_w = max(1, bx1 - bx0)
    bbox_h = max(1, by1 - by0)
    aspect = bbox_w / bbox_h

    rect_w = max(1, int(round((target_area * aspect) ** 0.5)))
    rect_h = max(1, int(np.ceil(target_area / rect_w)))
    if rect_w > w or rect_h > h:
        scale = max(rect_w / max(1, w), rect_h / max(1, h))
        rect_w = max(1, int(rect_w / scale))
        rect_h = max(1, int(np.ceil(target_area / rect_w)))
    if x0 < 0 or y0 < 0 or x0 + rect_w > w or y0 + rect_h > h:
        return None

    out = np.zeros_like(reference_mask, dtype=bool)
    full_rows = target_area // rect_w
    rem = target_area % rect_w
    used_rows = min(full_rows, rect_h)
    if used_rows > 0:
        out[y0 : y0 + used_rows, x0 : x0 + rect_w] = True
    if rem > 0 and used_rows < rect_h:
        out[y0 + used_rows, x0 : x0 + rem] = True
    return out


def _make_random_control_mask(
    reference_mask: np.ndarray,
    avoid_mask: np.ndarray,
    seed: int,
    avoid_overlap_iou: float,
    max_tries: int,
) -> tuple[np.ndarray, float, bool]:
    target_area = int(reference_mask.sum())
    if target_area <= 0:
        return np.zeros_like(reference_mask, dtype=bool), 0.0, False

    h, w = reference_mask.shape
    bbox = _bbox_from_mask(reference_mask)
    if bbox is None:
        return np.zeros_like(reference_mask, dtype=bool), 0.0, False
    bx0, by0, bx1, by1 = bbox
    bbox_w = max(1, bx1 - bx0)
    bbox_h = max(1, by1 - by0)
    aspect = bbox_w / bbox_h
    rect_w = max(1, int(round((target_area * aspect) ** 0.5)))
    rect_h = max(1, int(np.ceil(target_area / rect_w)))
    rect_w = min(rect_w, w)
    rect_h = min(rect_h, h)

    rng = np.random.default_rng(seed)
    best_mask = None
    best_iou = None

    for _ in range(max_tries):
        x0 = int(rng.integers(0, max(1, w - rect_w + 1)))
        y0 = int(rng.integers(0, max(1, h - rect_h + 1)))
        candidate = _build_area_matched_rectangle(reference_mask, x0, y0)
        if candidate is None:
            continue
        iou = _mask_iou(candidate, avoid_mask)
        if best_iou is None or iou < best_iou:
            best_iou = iou
            best_mask = candidate
        if iou <= avoid_overlap_iou:
            return candidate, iou, True

    if best_mask is None:
        return np.zeros_like(reference_mask, dtype=bool), 0.0, False
    return best_mask, float(best_iou), False


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


def _device_batch(model, batch: dict) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(model.cfg.device) if torch.is_tensor(value) else value
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run mainline region-mask counterfactual interventions with answer/relate/union/random4 conditions.")
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--mask-root", required=True, help="Directory containing exported_masks/<image_stem>/<label>.png")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--transcoder-set", default="tianhux2/gemma3-4b-it-plt")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16", "fp32", "bf16", "fp16"])
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--mask-fill", default="128,128,128")
    parser.add_argument("--avoid-overlap-iou", type=float, default=0.05)
    parser.add_argument("--random-seeds", default="101,202,303,404")
    parser.add_argument("--random-max-tries", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    manifest_rows = _read_csv(Path(args.manifest_csv).expanduser().resolve())
    if not manifest_rows:
        raise ValueError("no manifest rows loaded")

    mask_root = Path(args.mask_root).expanduser().resolve()
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    fill_rgb = _parse_fill(args.mask_fill)
    random_seeds = [int(part.strip()) for part in args.random_seeds.split(",") if part.strip()]

    unsupported = [condition for condition in conditions if not (condition in {"clean", "answer_mask", "relate_mask", "union_mask"} or condition.startswith("random_control_"))]
    if unsupported:
        raise ValueError(f"unsupported conditions: {unsupported}")

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
    from huggingface_hub import hf_hub_download
    import yaml

    if args.model_name:
        model_name = args.model_name
    else:
        config_path = hf_hub_download(repo_id=args.transcoder_set, filename="config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model_name = (cfg or {}).get("model_name", "")
        if not model_name:
            raise ValueError(f"model_name missing in {args.transcoder_set}/config.yaml")

    print(f"[info] loading model={model_name} transcoder_set={args.transcoder_set} dtype={dtype}")
    model = ReplacementModel.from_pretrained(
        model_name,
        args.transcoder_set,
        dtype=dtype,
        lazy_encoder=True,
        lazy_decoder=True,
    )
    tokenizer = model.processor.tokenizer

    if args.max_rows > 0:
        manifest_rows = manifest_rows[: args.max_rows]

    results: list[dict[str, str]] = []

    for row in manifest_rows:
        image_path = (row.get("image_path") or "").strip()
        if not image_path:
            print(f"[skip] sample={row.get('sample_id','')} run={row.get('run','')} missing local image path")
            continue
        image_file = Path(image_path)
        if not image_file.exists():
            print(f"[skip] sample={row.get('sample_id','')} run={row.get('run','')} missing image file {image_file}")
            continue

        target_token_id = _safe_int(row.get("target_token_id"))
        layer = _safe_int(row.get("feature_layer"))
        pos = _safe_int(row.get("feature_pos"))
        feature_id = _safe_int(row.get("feature_id"))
        if target_token_id is None or layer is None or pos is None or feature_id is None:
            print(f"[skip] sample={row.get('sample_id','')} run={row.get('run','')} incomplete feature metadata")
            continue

        original_image = Image.open(image_file).convert("RGB")
        answer_mask = _mask_array(_load_mask(mask_root, image_path, "answer"))
        relate_mask = _mask_array(_load_mask(mask_root, image_path, "relate"))
        union_mask = np.logical_or(answer_mask, relate_mask)

        random_controls: dict[str, tuple[np.ndarray, float, bool, int, int]] = {}
        for idx, seed in enumerate(random_seeds, start=1):
            candidate, actual_iou, valid = _make_random_control_mask(
                answer_mask,
                union_mask,
                seed=seed,
                avoid_overlap_iou=args.avoid_overlap_iou,
                max_tries=args.random_max_tries,
            )
            random_controls[f"random_control_{idx}"] = (candidate, actual_iou, valid, idx, seed)

        question = row.get("question", "")
        assistant_prefix = row.get("assistant_prefix", "")
        condition_to_mask = {
            "clean": (None, "", "", ""),
            "answer_mask": (answer_mask, "", "", ""),
            "relate_mask": (relate_mask, "", "", ""),
            "union_mask": (union_mask, "", "", ""),
        }
        for key, (mask_bool, actual_iou, valid, index, seed) in random_controls.items():
            condition_to_mask[key] = (
                mask_bool,
                f"{actual_iou:.10g}",
                _bool_str(valid),
                f"{index}:{seed}",
            )

        for condition in conditions:
            mask_bool, actual_iou_str, random_valid_str, random_meta = condition_to_mask[condition]
            condition_image = original_image if mask_bool is None else _apply_mask(original_image, mask_bool, fill_rgb)

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
                    f"[skip] sample={row.get('sample_id','')} run={row.get('run','')} condition={condition} "
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

            random_index = ""
            random_seed = ""
            if random_meta:
                random_index, random_seed = random_meta.split(":")

            results.append(
                {
                    "sample_id": row.get("sample_id", ""),
                    "run": row.get("run", ""),
                    "prompt_name": row.get("prompt_name", ""),
                    "node_role": row.get("node_role", ""),
                    "node_source": row.get("node_source", ""),
                    "assistant_prefix": assistant_prefix,
                    "condition": condition,
                    "question": question,
                    "image_path": image_path,
                    "remote_image_path": row.get("remote_image_path", ""),
                    "feature_layer": str(layer),
                    "feature_pos": str(pos),
                    "feature_id": str(feature_id),
                    "target_token_id": str(target_token_id),
                    "reasoning_operation": row.get("reasoning_operation", ""),
                    "visual_structure": row.get("visual_structure", ""),
                    "image_dependence": row.get("image_dependence", ""),
                    "selection_tier": row.get("selection_tier", ""),
                    "source_bucket": row.get("source_bucket", ""),
                    "match_mode": row.get("match_mode", ""),
                    "sampled_match_label": row.get("sampled_match_label", ""),
                    "control_draw_idx": row.get("control_draw_idx", ""),
                    "mask_fill_rgb": args.mask_fill,
                    "avoid_overlap_iou": f"{args.avoid_overlap_iou:.10g}",
                    "random_control_index": random_index,
                    "random_control_seed": random_seed,
                    "random_control_valid": random_valid_str,
                    "random_control_actual_iou": actual_iou_str,
                    "answer_area_px": str(int(answer_mask.sum())),
                    "relate_area_px": str(int(relate_mask.sum())),
                    "union_area_px": str(int(union_mask.sum())),
                    "mask_area_px": "" if mask_bool is None else str(int(mask_bool.sum())),
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
                f"[done] sample={row.get('sample_id','')} run={row.get('run','')} prompt={row.get('prompt_name','')} "
                f"role={row.get('node_role','')} source={row.get('node_source','')} condition={condition} "
                f"delta_target_logit={delta_target_logit:.4f} random_iou={actual_iou_str or 'na'}"
            )

    out_path = Path(args.out_csv).expanduser().resolve()
    fieldnames = [
        "sample_id",
        "run",
        "prompt_name",
        "node_role",
        "node_source",
        "assistant_prefix",
        "condition",
        "question",
        "image_path",
        "remote_image_path",
        "feature_layer",
        "feature_pos",
        "feature_id",
        "target_token_id",
        "reasoning_operation",
        "visual_structure",
        "image_dependence",
        "selection_tier",
        "source_bucket",
        "match_mode",
        "sampled_match_label",
        "control_draw_idx",
        "mask_fill_rgb",
        "avoid_overlap_iou",
        "random_control_index",
        "random_control_seed",
        "random_control_valid",
        "random_control_actual_iou",
        "answer_area_px",
        "relate_area_px",
        "union_area_px",
        "mask_area_px",
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
    ]
    _write_csv(out_path, results, fieldnames)
    print(f"[done] out_csv={out_path}")
    return 0


def _bool_str(value: bool) -> str:
    return "True" if value else "False"


if __name__ == "__main__":
    raise SystemExit(main())
