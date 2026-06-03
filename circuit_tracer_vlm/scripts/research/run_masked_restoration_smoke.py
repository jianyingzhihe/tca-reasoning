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


def _fmt(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.10g}"


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


def _node_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str, str]:
    return (
        row.get("bucket", ""),
        row.get("sample_id", ""),
        row.get("run", ""),
        row.get("node_role", ""),
        row.get("feature_layer", ""),
        row.get("feature_pos", ""),
        row.get("feature_id", ""),
    )


def _parse_mask_fill(mask_fill: str | None, default_fill: str) -> tuple[int, int, int]:
    raw = (mask_fill or default_fill or "").strip()
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"invalid mask fill: {raw!r}")
    rgb = tuple(int(part) for part in parts)
    if any(v < 0 or v > 255 for v in rgb):
        raise ValueError(f"mask fill values must be in [0, 255], got {rgb}")
    return rgb


def _apply_mask(
    image: Image.Image,
    *,
    mask_fraction: float,
    mask_fill_rgb: tuple[int, int, int],
) -> Image.Image:
    masked = image.copy()
    width, height = masked.size
    mask_w = max(1, int(round(width * mask_fraction)))
    mask_h = max(1, int(round(height * mask_fraction)))
    left = max(0, (width - mask_w) // 2)
    top = max(0, (height - mask_h) // 2)
    right = min(width, left + mask_w)
    bottom = min(height, top + mask_h)
    ImageDraw.Draw(masked).rectangle((left, top, right, bottom), fill=mask_fill_rgb)
    return masked


def _build_condition_image(
    corrupt_row: dict[str, str],
    original_image: Image.Image,
    *,
    corrupt_condition: str,
    default_mask_fraction: float,
    default_mask_fill: str,
) -> Image.Image:
    if corrupt_condition == "clean":
        return original_image
    if corrupt_condition == "no_image":
        return Image.new("RGB", original_image.size, color=(128, 128, 128))
    if corrupt_condition == "wrong_image":
        wrong_path = (corrupt_row.get("condition_image_path") or "").strip()
        if not wrong_path:
            raise ValueError("wrong_image row missing condition_image_path")
        return Image.open(wrong_path).convert("RGB")
    if corrupt_condition == "masked_image":
        mask_fraction = _safe_float(corrupt_row.get("mask_fraction"))
        if math.isnan(mask_fraction):
            mask_fraction = default_mask_fraction
        if not 0.0 < mask_fraction <= 1.0:
            raise ValueError(f"invalid mask_fraction={mask_fraction}")
        mask_fill_rgb = _parse_mask_fill(corrupt_row.get("mask_fill_rgb"), default_mask_fill)
        return _apply_mask(
            original_image,
            mask_fraction=mask_fraction,
            mask_fill_rgb=mask_fill_rgb,
        )
    raise ValueError(f"unsupported corrupt condition: {corrupt_condition}")


def _capture_feature_value_and_logits(
    model,
    batch: dict,
    *,
    layer: int,
    pos: int,
    feature_id: int,
) -> tuple[torch.Tensor, float]:
    feature_value: dict[str, float] = {}

    def _hook(acts, hook):
        transcoder_acts = (
            model.transcoders.encode_layer(acts, layer, apply_activation_function=True).detach().squeeze(0)
        )
        feature_value["value"] = float(transcoder_acts[pos, feature_id].item())

    hook_name = f"blocks.{layer}.{model.feature_input_hook}"
    with torch.inference_mode(), model.hooks([(hook_name, _hook)]):  # type: ignore[arg-type]
        logits = model.forward_from_batch(batch)
    if "value" not in feature_value:
        raise RuntimeError(f"failed to capture feature value for layer={layer} pos={pos} feature={feature_id}")
    return logits, feature_value["value"]


def _batch_seq_len(batch: dict) -> int | None:
    input_ids = batch.get("input_ids")
    if input_ids is None or not torch.is_tensor(input_ids):
        return None
    if input_ids.ndim < 2:
        return None
    return int(input_ids.shape[1])


def _target_stats(logits: torch.Tensor, target_token_id: int, tokenizer) -> dict[str, str]:
    last_pos = logits.shape[1] - 1
    target_logit = float(logits[0, last_pos, target_token_id].item())
    probs = torch.softmax(logits[0, last_pos], dim=-1)
    target_prob = float(probs[target_token_id].item())
    top1_id = int(torch.argmax(logits[0, last_pos]).item())
    return {
        "target_logit": _fmt(target_logit),
        "target_prob": _fmt(target_prob),
        "top1_id": str(top1_id),
        "top1_token": tokenizer.convert_ids_to_tokens([top1_id])[0],
    }


def _recovery_fraction(clean_value: float, corrupt_value: float, restored_value: float) -> float:
    denom = clean_value - corrupt_value
    if denom == 0.0:
        return math.nan
    return (restored_value - corrupt_value) / denom


def _select_candidates(
    pilot_rows: list[dict[str, str]],
    *,
    restore_condition: str,
    corrupt_condition: str,
    node_role: str,
    min_abs_clean_delta: float,
    min_effect_weakening: float,
    max_candidates: int,
    max_nodes_per_sample_run: int,
) -> list[tuple[float, dict[str, str], dict[str, str]]]:
    by_node: dict[tuple[str, ...], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in pilot_rows:
        by_node[_node_key(row)][row.get("condition", "")] = row

    scored: list[tuple[float, dict[str, str], dict[str, str]]] = []
    for key, by_condition in by_node.items():
        clean_row = by_condition.get(restore_condition)
        corrupt_row = by_condition.get(corrupt_condition)
        if clean_row is None or corrupt_row is None:
            continue
        role = clean_row.get("node_role", "")
        if node_role != "both" and role != node_role:
            continue

        clean_delta = _safe_float(clean_row.get("delta_target_logit"))
        corrupt_delta = _safe_float(corrupt_row.get("delta_target_logit"))
        if math.isnan(clean_delta) or math.isnan(corrupt_delta):
            continue
        if abs(clean_delta) < min_abs_clean_delta:
            continue

        if role == "support":
            if clean_delta >= 0:
                continue
            weakening = corrupt_delta - clean_delta
        elif role == "suppressor":
            if clean_delta <= 0:
                continue
            weakening = clean_delta - corrupt_delta
        else:
            continue

        if weakening < min_effect_weakening:
            continue
        scored.append((weakening, clean_row, corrupt_row))

    scored.sort(
        key=lambda item: (
            -item[0],
            -abs(_safe_float(item[1].get("delta_target_logit"))),
            item[1].get("bucket", ""),
            item[1].get("sample_id", ""),
            item[1].get("run", ""),
        )
    )

    selected: list[tuple[float, dict[str, str], dict[str, str]]] = []
    per_sample_run_count: dict[tuple[str, str, str], int] = defaultdict(int)
    for score, clean_row, corrupt_row in scored:
        sample_run_key = (
            clean_row.get("bucket", ""),
            clean_row.get("sample_id", ""),
            clean_row.get("run", ""),
        )
        if per_sample_run_count[sample_run_key] >= max_nodes_per_sample_run:
            continue
        selected.append((score, clean_row, corrupt_row))
        per_sample_run_count[sample_run_key] += 1
        if len(selected) >= max_candidates:
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Restore clean feature values into a corrupted condition for top weakened Stage 1 nodes."
    )
    parser.add_argument("--run-tag-base", required=True)
    parser.add_argument("--pilot-csv", required=True)
    parser.add_argument("--outputs-root", default="outputs/phase_ab/ab_answer_aligned")
    parser.add_argument("--transcoder-set", default="tianhux2/gemma3-4b-it-plt")
    parser.add_argument("--model-name", default="")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "bfloat16", "float16", "fp32", "bf16", "fp16"],
    )
    parser.add_argument("--restore-condition", default="clean")
    parser.add_argument("--corrupt-condition", default="masked_image")
    parser.add_argument("--node-role", default="support", choices=["support", "suppressor", "both"])
    parser.add_argument("--min-abs-clean-delta", type=float, default=0.25)
    parser.add_argument("--min-effect-weakening", type=float, default=0.125)
    parser.add_argument("--max-nodes", type=int, default=8)
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=0,
        help="If >0, score this many candidates first, then keep only the final accepted nodes.",
    )
    parser.add_argument("--max-nodes-per-sample-run", type=int, default=1)
    parser.add_argument(
        "--require-gap-direction",
        action="store_true",
        help="Require clean_feature_value > corrupt_feature_value, i.e. the corruption reduced the feature's activation before restoration.",
    )
    parser.add_argument("--default-mask-fraction", type=float, default=0.4)
    parser.add_argument("--default-mask-fill", default="128,128,128")
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    pilot_rows = _read_csv(Path(args.pilot_csv).expanduser().resolve())
    if not pilot_rows:
        raise ValueError("no pilot rows loaded")

    candidate_pool_size = args.candidate_pool_size if args.candidate_pool_size > 0 else args.max_nodes
    selected = _select_candidates(
        pilot_rows,
        restore_condition=args.restore_condition,
        corrupt_condition=args.corrupt_condition,
        node_role=args.node_role,
        min_abs_clean_delta=args.min_abs_clean_delta,
        min_effect_weakening=args.min_effect_weakening,
        max_candidates=candidate_pool_size,
        max_nodes_per_sample_run=args.max_nodes_per_sample_run,
    )
    if not selected:
        raise ValueError("no restoration candidates selected")

    outputs_root = Path(args.outputs_root).expanduser().resolve()
    buckets = sorted({clean_row.get("bucket", "") for _, clean_row, _ in selected})
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

    results: list[dict[str, str]] = []
    tokenizer = model.processor.tokenizer

    for weakening_score, clean_row, corrupt_row in selected:
        bucket = clean_row.get("bucket", "")
        sample_id = clean_row.get("sample_id", "")
        run = clean_row.get("run", "")
        role = clean_row.get("node_role", "")
        meta_map = meta_a_by_bucket[bucket] if run == "A" else meta_b_by_bucket[bucket]
        meta = meta_map.get(sample_id, {})

        question = meta.get("question", clean_row.get("question", ""))
        assistant_prefix = meta.get("assistant_prefix", "")
        target_token_id = _safe_int(meta.get("target_token_id") or clean_row.get("target_token_id"))
        image_path = (meta.get("image_path") or clean_row.get("image_path") or "").strip()
        if target_token_id is None or not image_path:
            print(f"[skip] bucket={bucket} sample={sample_id} run={run} missing target or image path")
            continue

        layer = int(clean_row.get("feature_layer", "0"))
        pos = int(clean_row.get("feature_pos", "0"))
        feature_id = int(clean_row.get("feature_id", "0"))

        original_image = Image.open(image_path).convert("RGB")
        corrupt_image = _build_condition_image(
            corrupt_row,
            original_image,
            corrupt_condition=args.corrupt_condition,
            default_mask_fraction=args.default_mask_fraction,
            default_mask_fill=args.default_mask_fill,
        )

        clean_batch = _build_multimodal_batch(
            model.processor,
            original_image,
            f"<start_of_image> {question}",
            assistant_prefix=assistant_prefix,
        )
        clean_batch["image"] = original_image
        clean_batch = _device_batch(model, clean_batch)

        corrupt_batch = _build_multimodal_batch(
            model.processor,
            corrupt_image,
            f"<start_of_image> {question}",
            assistant_prefix=assistant_prefix,
        )
        corrupt_batch["image"] = corrupt_image
        corrupt_batch = _device_batch(model, corrupt_batch)

        try:
            with torch.inference_mode():
                clean_logits, clean_feature_value = _capture_feature_value_and_logits(
                    model,
                    clean_batch,
                    layer=layer,
                    pos=pos,
                    feature_id=feature_id,
                )
                corrupt_logits, corrupt_feature_value = _capture_feature_value_and_logits(
                    model,
                    corrupt_batch,
                    layer=layer,
                    pos=pos,
                    feature_id=feature_id,
                )
                if args.require_gap_direction:
                    gap_ok = clean_feature_value > corrupt_feature_value
                    if not gap_ok:
                        print(
                            f"[skip] bucket={bucket} sample={sample_id} run={run} role={role} "
                            f"feature=L{layer}:P{pos}:F{feature_id} gap={clean_feature_value - corrupt_feature_value:.4f} "
                            f"failed require-gap-direction"
                        )
                        continue
                restored_logits, _ = model.feature_intervention(
                    corrupt_batch,
                    [(layer, pos, feature_id, clean_feature_value)],
                    freeze_attention=True,
                    apply_activation_function=True,
                    sparse=False,
                )
                zeroed_logits, _ = model.feature_intervention(
                    corrupt_batch,
                    [(layer, pos, feature_id, 0.0)],
                    freeze_attention=True,
                    apply_activation_function=True,
                    sparse=False,
                )
        except (IndexError, RuntimeError) as exc:
            clean_len = _batch_seq_len(clean_batch)
            corrupt_len = _batch_seq_len(corrupt_batch)
            print(
                f"[skip] bucket={bucket} sample={sample_id} run={run} role={role} "
                f"feature=L{layer}:P{pos}:F{feature_id} clean_seq_len={clean_len} corrupt_seq_len={corrupt_len} "
                f"error={type(exc).__name__}: {exc}"
            )
            continue

        clean_stats = _target_stats(clean_logits, target_token_id, tokenizer)
        corrupt_stats = _target_stats(corrupt_logits, target_token_id, tokenizer)
        restored_stats = _target_stats(restored_logits, target_token_id, tokenizer)
        zeroed_stats = _target_stats(zeroed_logits, target_token_id, tokenizer)

        clean_target_logit = float(clean_stats["target_logit"])
        corrupt_target_logit = float(corrupt_stats["target_logit"])
        restored_target_logit = float(restored_stats["target_logit"])
        zeroed_target_logit = float(zeroed_stats["target_logit"])

        results.append(
            {
                "bucket": bucket,
                "sample_id": sample_id,
                "run": run,
                "node_role": role,
                "restore_condition": args.restore_condition,
                "corrupt_condition": args.corrupt_condition,
                "question": question,
                "image_path": image_path,
                "condition_image_path": corrupt_row.get("condition_image_path", ""),
                "target_token_id": str(target_token_id),
                "feature_layer": str(layer),
                "feature_pos": str(pos),
                "feature_id": str(feature_id),
                "selected_effect_weakening": _fmt(weakening_score),
                "clean_selected_delta_target_logit": clean_row.get("delta_target_logit", ""),
                "corrupt_selected_delta_target_logit": corrupt_row.get("delta_target_logit", ""),
                "clean_feature_value": _fmt(clean_feature_value),
                "corrupt_feature_value": _fmt(corrupt_feature_value),
                "feature_value_gap": _fmt(clean_feature_value - corrupt_feature_value),
                "clean_target_logit": clean_stats["target_logit"],
                "corrupt_target_logit": corrupt_stats["target_logit"],
                "restored_target_logit": restored_stats["target_logit"],
                "zeroed_target_logit": zeroed_stats["target_logit"],
                "restore_minus_corrupt_target_logit": _fmt(restored_target_logit - corrupt_target_logit),
                "zero_minus_corrupt_target_logit": _fmt(zeroed_target_logit - corrupt_target_logit),
                "restore_minus_zero_target_logit": _fmt(restored_target_logit - zeroed_target_logit),
                "clean_corrupt_logit_gap": _fmt(clean_target_logit - corrupt_target_logit),
                "restore_fraction_of_clean_gap": _fmt(
                    _recovery_fraction(clean_target_logit, corrupt_target_logit, restored_target_logit)
                ),
                "clean_target_prob": clean_stats["target_prob"],
                "corrupt_target_prob": corrupt_stats["target_prob"],
                "restored_target_prob": restored_stats["target_prob"],
                "zeroed_target_prob": zeroed_stats["target_prob"],
                "clean_top1_id": clean_stats["top1_id"],
                "clean_top1_token": clean_stats["top1_token"],
                "corrupt_top1_id": corrupt_stats["top1_id"],
                "corrupt_top1_token": corrupt_stats["top1_token"],
                "restored_top1_id": restored_stats["top1_id"],
                "restored_top1_token": restored_stats["top1_token"],
                "zeroed_top1_id": zeroed_stats["top1_id"],
                "zeroed_top1_token": zeroed_stats["top1_token"],
            }
        )

        print(
            f"[done] bucket={bucket} sample={sample_id} run={run} role={role} "
            f"feature=L{layer}:P{pos}:F{feature_id} weaken={weakening_score:.4f} "
            f"restore_minus_corrupt={restored_target_logit - corrupt_target_logit:.4f} "
            f"zero_minus_corrupt={zeroed_target_logit - corrupt_target_logit:.4f}"
        )
        if len(results) >= args.max_nodes:
            break

    out_path = Path(args.out_csv).expanduser().resolve()
    _write_csv(
        out_path,
        results,
        [
            "bucket",
            "sample_id",
            "run",
            "node_role",
            "restore_condition",
            "corrupt_condition",
            "question",
            "image_path",
            "condition_image_path",
            "target_token_id",
            "feature_layer",
            "feature_pos",
            "feature_id",
            "selected_effect_weakening",
            "clean_selected_delta_target_logit",
            "corrupt_selected_delta_target_logit",
            "clean_feature_value",
            "corrupt_feature_value",
            "feature_value_gap",
            "clean_target_logit",
            "corrupt_target_logit",
            "restored_target_logit",
            "zeroed_target_logit",
            "restore_minus_corrupt_target_logit",
            "zero_minus_corrupt_target_logit",
            "restore_minus_zero_target_logit",
            "clean_corrupt_logit_gap",
            "restore_fraction_of_clean_gap",
            "clean_target_prob",
            "corrupt_target_prob",
            "restored_target_prob",
            "zeroed_target_prob",
            "clean_top1_id",
            "clean_top1_token",
            "corrupt_top1_id",
            "corrupt_top1_token",
            "restored_top1_id",
            "restored_top1_token",
            "zeroed_top1_id",
            "zeroed_top1_token",
        ],
    )
    print(f"[done] out_csv={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
