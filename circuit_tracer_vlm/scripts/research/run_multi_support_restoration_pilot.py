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


def _load_compare_nodes(outputs_root: Path, run_tag_base: str, bucket: str) -> list[dict[str, str]]:
    compare_dir = outputs_root / f"{run_tag_base}_{bucket}" / f"{run_tag_base}_{bucket}"
    path = compare_dir / "nodes_detailed_controlled.csv"
    if not path.exists():
        return []
    return _read_csv(path)


def _load_generic_features(path: Path | None) -> set[tuple[str, str, str]]:
    if path is None or not path.exists():
        return set()
    out: set[tuple[str, str, str]] = set()
    for row in _read_csv(path):
        if row.get("is_generic", "").lower() != "true":
            continue
        if row.get("node_type") != "feature":
            continue
        out.add((row.get("layer", ""), row.get("pos", ""), row.get("feature_or_token_id", "")))
    return out


def _build_multimodal_batch(processor, image, prompt: str, assistant_prefix: str = ""):
    prompt = (prompt or "").strip()
    assistant_prefix = assistant_prefix or ""

    if "<start_of_image>" in prompt:
        if assistant_prefix:
            text = f"{prompt.rstrip()} {assistant_prefix.lstrip()}".strip()
        else:
            text = prompt
        return processor(text=text, images=image, return_tensors="pt")

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if assistant_prefix:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_prefix}],
                }
            )
            add_generation_prompt = False
        else:
            add_generation_prompt = True

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if isinstance(inputs, dict) and "input_ids" in inputs:
            return inputs
    except Exception:
        pass

    if assistant_prefix:
        text = f"<start_of_image> {prompt}\n\n{assistant_prefix}".strip()
    else:
        text = f"<start_of_image> {prompt}".strip()
    return processor(text=text, images=image, return_tensors="pt")


def _apply_center_mask(
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


def _condition_image(
    condition: str,
    original_image: Image.Image,
    condition_image_path: str,
    *,
    mask_fraction: float,
    mask_fill_rgb: tuple[int, int, int],
) -> Image.Image:
    if condition == "clean":
        return original_image
    if condition == "wrong_image":
        return Image.open(condition_image_path).convert("RGB")
    if condition == "masked_image":
        return _apply_center_mask(original_image, mask_fraction=mask_fraction, mask_fill_rgb=mask_fill_rgb)
    if condition == "no_image":
        return Image.new("RGB", original_image.size, color=(128, 128, 128))
    raise ValueError(f"unknown condition: {condition}")


def _batch_seq_len(batch: dict) -> int:
    return int(batch["input_ids"].shape[1])


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan top compare features, pick support-like ready candidates, and run grouped restoration on a corruption condition."
    )
    parser.add_argument("--compare-run-tag-base", required=True)
    parser.add_argument("--ready-pilot-csv", required=True)
    parser.add_argument("--full-pilot-csv", required=True)
    parser.add_argument("--outputs-root", default="outputs/phase_ab/ab_answer_aligned")
    parser.add_argument("--transcoder-set", default="tianhux2/gemma3-4b-it-plt")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16", "fp32", "bf16", "fp16"])
    parser.add_argument("--corrupt-condition", default="wrong_image", choices=["wrong_image", "masked_image", "no_image"])
    parser.add_argument("--generic-nodes-csv", default="")
    parser.add_argument("--top-compare-features", type=int, default=6)
    parser.add_argument("--min-abs-clean-delta", type=float, default=0.125)
    parser.add_argument("--min-effect-weakening", type=float, default=0.0)
    parser.add_argument("--min-selected-features", type=int, default=2)
    parser.add_argument("--max-selected-features", type=int, default=3)
    parser.add_argument("--mask-fraction", type=float, default=0.4)
    parser.add_argument("--mask-fill", default="128,128,128")
    parser.add_argument("--out-candidate-csv", required=True)
    parser.add_argument("--out-group-csv", required=True)
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root).expanduser().resolve()
    ready_rows = _read_csv(Path(args.ready_pilot_csv).expanduser().resolve())
    full_pilot_rows = _read_csv(Path(args.full_pilot_csv).expanduser().resolve())
    if not ready_rows or not full_pilot_rows:
        raise ValueError("ready/full pilot rows missing")
    generic_features = _load_generic_features(
        Path(args.generic_nodes_csv).expanduser().resolve() if args.generic_nodes_csv else None
    )

    mask_fill_parts = [int(part.strip()) for part in args.mask_fill.split(",")]
    mask_fill_rgb = (mask_fill_parts[0], mask_fill_parts[1], mask_fill_parts[2])

    # allowlist sample/runs from ready support rows only
    sample_run_allowlist: set[tuple[str, str, str]] = set()
    for row in ready_rows:
        if row.get("condition") == "clean" and row.get("node_role") == "support":
            sample_run_allowlist.add((row.get("bucket", ""), row.get("sample_id", ""), row.get("run", "")))

    by_sample_run_condition: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in full_pilot_rows:
        key = (row.get("bucket", ""), row.get("sample_id", ""), row.get("run", ""), row.get("condition", ""))
        by_sample_run_condition[key] = row

    buckets = sorted({bucket for bucket, _, _ in sample_run_allowlist})
    meta_a_by_bucket = {bucket: _load_meta(outputs_root, args.compare_run_tag_base, bucket, "A") for bucket in buckets}
    meta_b_by_bucket = {bucket: _load_meta(outputs_root, args.compare_run_tag_base, bucket, "B") for bucket in buckets}
    compare_nodes_by_bucket = {bucket: _load_compare_nodes(outputs_root, args.compare_run_tag_base, bucket) for bucket in buckets}

    sample_run_candidates: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for bucket, rows in compare_nodes_by_bucket.items():
        for row in rows:
            if row.get("node_type") != "feature":
                continue
            feature_key = (row.get("layer", ""), row.get("pos", ""), row.get("feature_id", ""))
            if feature_key in generic_features:
                continue
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

    candidate_rows: list[dict[str, str]] = []
    group_rows: list[dict[str, str]] = []

    for bucket, sample_id, run in sorted(sample_run_allowlist):
        meta_map = meta_a_by_bucket[bucket] if run == "A" else meta_b_by_bucket[bucket]
        meta = meta_map.get(sample_id, {})
        question = meta.get("question", "")
        assistant_prefix = meta.get("assistant_prefix", "")
        target_token_id = _safe_int(meta.get("target_token_id"))
        image_path = (meta.get("image_path") or "").strip()
        corrupt_row = by_sample_run_condition.get((bucket, sample_id, run, args.corrupt_condition))
        if target_token_id is None or not image_path or corrupt_row is None:
            continue
        condition_image_path = (corrupt_row.get("condition_image_path") or image_path).strip()

        original_image = Image.open(image_path).convert("RGB")
        clean_image = _condition_image("clean", original_image, condition_image_path, mask_fraction=args.mask_fraction, mask_fill_rgb=mask_fill_rgb)
        corrupt_image = _condition_image(args.corrupt_condition, original_image, condition_image_path, mask_fraction=args.mask_fraction, mask_fill_rgb=mask_fill_rgb)

        clean_batch = _build_multimodal_batch(model.processor, clean_image, f"<start_of_image> {question}", assistant_prefix=assistant_prefix)
        clean_batch["image"] = clean_image
        clean_batch = _device_batch(model, clean_batch)

        corrupt_batch = _build_multimodal_batch(model.processor, corrupt_image, f"<start_of_image> {question}", assistant_prefix=assistant_prefix)
        corrupt_batch["image"] = corrupt_image
        corrupt_batch = _device_batch(model, corrupt_batch)

        clean_seq_len = _batch_seq_len(clean_batch)
        corrupt_seq_len = _batch_seq_len(corrupt_batch)

        compare_rows = list(sample_run_candidates.get((bucket, sample_id, run), []))
        compare_rows.sort(key=lambda row: _safe_float(row.get("path_mass_best")), reverse=True)
        compare_rows = compare_rows[: args.top_compare_features]

        scanned_candidates: list[dict[str, str]] = []
        selected_features: list[tuple[int, int, int, float]] = []
        selected_rows: list[dict[str, str]] = []

        for compare_row in compare_rows:
            layer = int(compare_row["layer"])
            pos = int(compare_row["pos"])
            feature_id = int(compare_row["feature_id"])
            if not (pos < clean_seq_len and pos < corrupt_seq_len):
                candidate_rows.append(
                    {
                        "bucket": bucket,
                        "sample_id": sample_id,
                        "run": run,
                        "corrupt_condition": args.corrupt_condition,
                        "feature_layer": str(layer),
                        "feature_pos": str(pos),
                        "feature_id": str(feature_id),
                        "path_mass_best": compare_row.get("path_mass_best", ""),
                        "clean_seq_len": str(clean_seq_len),
                        "corrupt_seq_len": str(corrupt_seq_len),
                        "ready_for_condition": "false",
                        "support_like": "",
                        "clean_delta_target_logit": "",
                        "corrupt_delta_target_logit": "",
                        "effect_weakening": "",
                        "clean_feature_value": "",
                        "corrupt_feature_value": "",
                        "feature_value_gap": "",
                        "selected_for_group": "false",
                    }
                )
                continue

            try:
                with torch.inference_mode():
                    clean_logits, clean_feature_value = _capture_feature_value_and_logits(
                        model, clean_batch, layer=layer, pos=pos, feature_id=feature_id
                    )
                    corrupt_logits, corrupt_feature_value = _capture_feature_value_and_logits(
                        model, corrupt_batch, layer=layer, pos=pos, feature_id=feature_id
                    )
                    clean_zero_logits, _ = model.feature_intervention(
                        clean_batch,
                        [(layer, pos, feature_id, 0.0)],
                        freeze_attention=True,
                        apply_activation_function=True,
                        sparse=False,
                    )
                    corrupt_zero_logits, _ = model.feature_intervention(
                        corrupt_batch,
                        [(layer, pos, feature_id, 0.0)],
                        freeze_attention=True,
                        apply_activation_function=True,
                        sparse=False,
                    )
            except (IndexError, RuntimeError):
                continue

            clean_target_logit = float(clean_logits[0, clean_logits.shape[1] - 1, target_token_id].item())
            clean_zero_target_logit = float(clean_zero_logits[0, clean_zero_logits.shape[1] - 1, target_token_id].item())
            corrupt_target_logit = float(corrupt_logits[0, corrupt_logits.shape[1] - 1, target_token_id].item())
            corrupt_zero_target_logit = float(corrupt_zero_logits[0, corrupt_zero_logits.shape[1] - 1, target_token_id].item())

            clean_delta = clean_zero_target_logit - clean_target_logit
            corrupt_delta = corrupt_zero_target_logit - corrupt_target_logit
            weakening = corrupt_delta - clean_delta
            gap = clean_feature_value - corrupt_feature_value

            support_like = (
                clean_delta < -args.min_abs_clean_delta
                and weakening >= args.min_effect_weakening
                and gap > 0.0
            )

            row = {
                "bucket": bucket,
                "sample_id": sample_id,
                "run": run,
                "corrupt_condition": args.corrupt_condition,
                "feature_layer": str(layer),
                "feature_pos": str(pos),
                "feature_id": str(feature_id),
                "path_mass_best": compare_row.get("path_mass_best", ""),
                "clean_seq_len": str(clean_seq_len),
                "corrupt_seq_len": str(corrupt_seq_len),
                "ready_for_condition": "true",
                "support_like": "true" if support_like else "false",
                "clean_delta_target_logit": _fmt(clean_delta),
                "corrupt_delta_target_logit": _fmt(corrupt_delta),
                "effect_weakening": _fmt(weakening),
                "clean_feature_value": _fmt(clean_feature_value),
                "corrupt_feature_value": _fmt(corrupt_feature_value),
                "feature_value_gap": _fmt(gap),
                "selected_for_group": "false",
            }
            scanned_candidates.append(row)
            candidate_rows.append(row)

        support_candidates = [row for row in scanned_candidates if row["support_like"] == "true"]
        support_candidates.sort(
            key=lambda row: (
                -_safe_float(row.get("effect_weakening")),
                -abs(_safe_float(row.get("clean_delta_target_logit"))),
                -_safe_float(row.get("path_mass_best")),
            )
        )
        selected_rows = support_candidates[: args.max_selected_features]
        if len(selected_rows) < args.min_selected_features:
            continue

        for row in candidate_rows:
            if row["bucket"] == bucket and row["sample_id"] == sample_id and row["run"] == run:
                for chosen in selected_rows:
                    if (
                        row["feature_layer"] == chosen["feature_layer"]
                        and row["feature_pos"] == chosen["feature_pos"]
                        and row["feature_id"] == chosen["feature_id"]
                    ):
                        row["selected_for_group"] = "true"

        interventions_clean: list[tuple[int, int, int, float]] = []
        interventions_zero: list[tuple[int, int, int, float]] = []
        for row in selected_rows:
            interventions_clean.append(
                (
                    int(row["feature_layer"]),
                    int(row["feature_pos"]),
                    int(row["feature_id"]),
                    float(row["clean_feature_value"]),
                )
            )
            interventions_zero.append(
                (
                    int(row["feature_layer"]),
                    int(row["feature_pos"]),
                    int(row["feature_id"]),
                    0.0,
                )
            )

        with torch.inference_mode():
            clean_logits = model.forward_from_batch(clean_batch)
            corrupt_logits = model.forward_from_batch(corrupt_batch)
            clean_zero_group_logits, _ = model.feature_intervention(
                clean_batch,
                interventions_zero,
                freeze_attention=True,
                apply_activation_function=True,
                sparse=False,
            )
            corrupt_zero_group_logits, _ = model.feature_intervention(
                corrupt_batch,
                interventions_zero,
                freeze_attention=True,
                apply_activation_function=True,
                sparse=False,
            )
            corrupt_restore_group_logits, _ = model.feature_intervention(
                corrupt_batch,
                interventions_clean,
                freeze_attention=True,
                apply_activation_function=True,
                sparse=False,
            )

        clean_stats = _target_stats(clean_logits, target_token_id, tokenizer)
        corrupt_stats = _target_stats(corrupt_logits, target_token_id, tokenizer)
        clean_zero_group_stats = _target_stats(clean_zero_group_logits, target_token_id, tokenizer)
        corrupt_zero_group_stats = _target_stats(corrupt_zero_group_logits, target_token_id, tokenizer)
        corrupt_restore_group_stats = _target_stats(corrupt_restore_group_logits, target_token_id, tokenizer)

        clean_target_logit = float(clean_stats["target_logit"])
        corrupt_target_logit = float(corrupt_stats["target_logit"])
        clean_zero_group_target_logit = float(clean_zero_group_stats["target_logit"])
        corrupt_zero_group_target_logit = float(corrupt_zero_group_stats["target_logit"])
        corrupt_restore_group_target_logit = float(corrupt_restore_group_stats["target_logit"])

        group_rows.append(
            {
                "bucket": bucket,
                "sample_id": sample_id,
                "run": run,
                "corrupt_condition": args.corrupt_condition,
                "selected_feature_count": str(len(selected_rows)),
                "selected_features": ";".join(
                    f"L{row['feature_layer']}:P{row['feature_pos']}:F{row['feature_id']}" for row in selected_rows
                ),
                "selected_weakenings": ";".join(row["effect_weakening"] for row in selected_rows),
                "clean_target_logit": clean_stats["target_logit"],
                "corrupt_target_logit": corrupt_stats["target_logit"],
                "clean_zero_group_target_logit": clean_zero_group_stats["target_logit"],
                "corrupt_zero_group_target_logit": corrupt_zero_group_stats["target_logit"],
                "corrupt_restore_group_target_logit": corrupt_restore_group_stats["target_logit"],
                "clean_zero_group_minus_clean": _fmt(clean_zero_group_target_logit - clean_target_logit),
                "corrupt_zero_group_minus_corrupt": _fmt(corrupt_zero_group_target_logit - corrupt_target_logit),
                "corrupt_restore_group_minus_corrupt": _fmt(corrupt_restore_group_target_logit - corrupt_target_logit),
                "corrupt_restore_group_minus_zero_group": _fmt(
                    corrupt_restore_group_target_logit - corrupt_zero_group_target_logit
                ),
                "clean_corrupt_logit_gap": _fmt(clean_target_logit - corrupt_target_logit),
            }
        )
        print(
            f"[done] {bucket} {sample_id} run={run} selected={len(selected_rows)} "
            f"restore_minus_corrupt={corrupt_restore_group_target_logit - corrupt_target_logit:.4f} "
            f"restore_minus_zero={corrupt_restore_group_target_logit - corrupt_zero_group_target_logit:.4f}"
        )

    _write_csv(
        Path(args.out_candidate_csv).expanduser().resolve(),
        candidate_rows,
        [
            "bucket",
            "sample_id",
            "run",
            "corrupt_condition",
            "feature_layer",
            "feature_pos",
            "feature_id",
            "path_mass_best",
            "clean_seq_len",
            "corrupt_seq_len",
            "ready_for_condition",
            "support_like",
            "clean_delta_target_logit",
            "corrupt_delta_target_logit",
            "effect_weakening",
            "clean_feature_value",
            "corrupt_feature_value",
            "feature_value_gap",
            "selected_for_group",
        ],
    )
    _write_csv(
        Path(args.out_group_csv).expanduser().resolve(),
        group_rows,
        [
            "bucket",
            "sample_id",
            "run",
            "corrupt_condition",
            "selected_feature_count",
            "selected_features",
            "selected_weakenings",
            "clean_target_logit",
            "corrupt_target_logit",
            "clean_zero_group_target_logit",
            "corrupt_zero_group_target_logit",
            "corrupt_restore_group_target_logit",
            "clean_zero_group_minus_clean",
            "corrupt_zero_group_minus_corrupt",
            "corrupt_restore_group_minus_corrupt",
            "corrupt_restore_group_minus_zero_group",
            "clean_corrupt_logit_gap",
        ],
    )
    print(f"[done] candidate_rows={len(candidate_rows)} group_rows={len(group_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
