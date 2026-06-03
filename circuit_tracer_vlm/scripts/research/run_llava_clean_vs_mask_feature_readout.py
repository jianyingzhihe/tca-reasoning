#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageChops, ImageDraw


SAMPLES = {
    "okvqa_val_2847255": {
        "image": "COCO_val2014_000000284725.jpg",
        "question": "What country might this be based on the writing on the bus?",
        "answer": "china",
    },
    "okvqa_val_4157235": {
        "image": "COCO_val2014_000000415723.jpg",
        "question": "Who is playing this sport?",
        "answer": "dog",
    },
    "okvqa_val_3605295": {
        "image": "COCO_val2014_000000360529.jpg",
        "question": "What place is this?",
        "answer": "store",
    },
    "okvqa_val_3658865": {
        "image": "COCO_val2014_000000365886.jpg",
        "question": "What brand of phone is this?",
        "answer": "samsung",
    },
    "okvqa_val_4739195": {
        "image": "COCO_val2014_000000473919.jpg",
        "question": "What language is the sign on the bear in?",
        "answer": "spanish",
    },
    "okvqa_val_5334645": {
        "image": "COCO_val2014_000000533464.jpg",
        "question": "What time of day is it?",
        "answer": "afternoon",
    },
}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage2f-llava-mask] {message}", flush=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _env_presence() -> dict[str, bool]:
    keys = [
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "HF_ENDPOINT",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
    ]
    return {key: bool(os.environ.get(key)) for key in keys}


def _gpu_info() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    free, total = torch.cuda.mem_get_info()
    return {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "free_gb": round(free / (1024**3), 3),
        "total_gb": round(total / (1024**3), 3),
    }


def _first_param_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_layers(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _prompt(question: str, prompt_name: str) -> str:
    if prompt_name == "D_visual_only":
        return (
            f"{question} Use visual evidence, then reply with only one short sentence "
            "in exactly this format: The answer is <short answer>."
        )
    if prompt_name == "B_direct":
        return (
            f"{question} Reply with only one short sentence in exactly this format: "
            "The answer is <short answer>."
        )
    raise ValueError(f"Unknown prompt_name: {prompt_name}")


def _processor_patch(processor, config) -> dict[str, Any]:
    vision_config = getattr(config, "vision_config", None)
    patch_size = getattr(processor, "patch_size", None)
    if patch_size is None and vision_config is not None:
        patch_size = getattr(vision_config, "patch_size", None)
        if patch_size is not None:
            setattr(processor, "patch_size", patch_size)
    strategy = getattr(processor, "vision_feature_select_strategy", None)
    if strategy is None:
        strategy = getattr(config, "vision_feature_select_strategy", "default")
        setattr(processor, "vision_feature_select_strategy", strategy)
    additional = getattr(processor, "num_additional_image_tokens", None)
    if additional in (None, 0):
        additional = 1
        setattr(processor, "num_additional_image_tokens", additional)
    return {
        "patch_size": patch_size,
        "vision_feature_select_strategy": strategy,
        "num_additional_image_tokens": additional,
    }


def _find_file(root: Path, rel_candidates: list[str]) -> Path | None:
    for rel in rel_candidates:
        path = root / rel
        if path.exists():
            return path
    return None


def _labelme_mask(json_path: Path, label: str, size: tuple[int, int]) -> Image.Image | None:
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text(encoding="utf-8"))
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    found = False
    for shape in data.get("shapes", []):
        if str(shape.get("label", "")).lower() != label:
            continue
        points = [(float(x), float(y)) for x, y in shape.get("points", [])]
        if len(points) < 2:
            continue
        if shape.get("shape_type", "polygon") == "rectangle" and len(points) >= 2:
            (x1, y1), (x2, y2) = points[:2]
            draw.rectangle([x1, y1, x2, y2], fill=255)
        else:
            draw.polygon(points, fill=255)
        found = True
    return mask if found else None


def _load_masks(sample: dict[str, str], roots: list[Path], work_dir: Path) -> dict[str, Any]:
    image_name = sample["image"]
    stem = Path(image_name).stem
    out: dict[str, Any] = {
        "status": "missing",
        "answer_mask": None,
        "union_mask": None,
        "source": "",
        "diagnostics": {},
    }
    image_path = None
    for root in roots:
        image_path = _find_file(root, [f"images/{image_name}", image_name])
        if image_path is not None:
            break
    if image_path is None:
        out["diagnostics"]["image_missing"] = image_name
        return out
    image = Image.open(image_path).convert("RGB")

    for root in roots:
        answer_path = root / "exported_masks" / stem / "answer.png"
        relate_path = root / "exported_masks" / stem / "relate.png"
        if answer_path.exists():
            answer = Image.open(answer_path).convert("L").resize(image.size)
            if relate_path.exists():
                relate = Image.open(relate_path).convert("L").resize(image.size)
                union = ImageChops.lighter(answer, relate)
            else:
                union = answer.copy()
            out.update(
                {
                    "status": "ok",
                    "answer_mask": answer,
                    "union_mask": union,
                    "source": str(answer_path.parent),
                    "image_path": image_path,
                }
            )
            return out

        json_path = root / "images" / f"{stem}.json"
        if json_path.exists():
            answer = _labelme_mask(json_path, "answer", image.size)
            relate = _labelme_mask(json_path, "relate", image.size)
            if answer is not None:
                union = ImageChops.lighter(answer, relate) if relate is not None else answer.copy()
                export_dir = work_dir / "exported_masks" / stem
                export_dir.mkdir(parents=True, exist_ok=True)
                answer.save(export_dir / "answer.png")
                union.save(export_dir / "union.png")
                out.update(
                    {
                        "status": "ok",
                        "answer_mask": answer,
                        "union_mask": union,
                        "source": str(json_path),
                        "image_path": image_path,
                    }
                )
                return out
    out["image_path"] = image_path
    return out


def _apply_mask(image: Image.Image, mask: Image.Image, fill: tuple[int, int, int]) -> Image.Image:
    fill_img = Image.new("RGB", image.size, fill)
    return Image.composite(fill_img, image, mask)


def _find_subsequence(seq: list[int], candidates: list[list[int]]) -> tuple[int | None, int | None]:
    for cand in candidates:
        if not cand:
            continue
        n = len(cand)
        for start in range(0, len(seq) - n + 1):
            if seq[start : start + n] == cand:
                return start, start + n
    return None, None


def _bucket_positions(
    input_ids: list[int],
    token_texts: list[str],
    tokenizer,
    question: str,
    image_token_id: int,
) -> dict[str, list[int]]:
    question_candidates = []
    for variant in [question, " " + question, "\n" + question]:
        encoded = tokenizer(variant, add_special_tokens=False)
        question_candidates.append(list(encoded["input_ids"]))
    q_start, q_end = _find_subsequence(input_ids, question_candidates)

    assistant_candidates = []
    for variant in ["ASSISTANT:", "\nASSISTANT:", " ASSISTANT:"]:
        encoded = tokenizer(variant, add_special_tokens=False)
        assistant_candidates.append(list(encoded["input_ids"]))
    a_start, _a_end = _find_subsequence(input_ids, assistant_candidates)

    image_positions = [idx for idx, token_id in enumerate(input_ids) if token_id == image_token_id]
    image_end = max(image_positions) + 1 if image_positions else None
    if a_start is None:
        assistant_text_positions = [
            idx for idx, tok in enumerate(token_texts) if "ASS" in tok.upper() or "istant" in tok
        ]
        a_start = min(assistant_text_positions) if assistant_text_positions else None

    buckets = {
        "image_token_span": image_positions,
        "question": list(range(q_start, q_end)) if q_start is not None and q_end is not None else [],
        "post_image_text": (
            list(range(image_end, a_start if a_start is not None else len(input_ids)))
            if image_end is not None
            else []
        ),
        "assistant_prefix": list(range(a_start, len(input_ids))) if a_start is not None else [],
        "last_prompt_token": [len(input_ids) - 1] if input_ids else [],
    }
    return buckets


def _download_transcoder(repo_id: str, layer: int) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=f"transcoder_L{layer}.pt",
            local_files_only=False,
        )
    )


def _load_encoder(path: Path, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {path}, got {type(obj).__name__}")
    state_dict = obj.get("state_dict", obj)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected state_dict dict in {path}, got {type(state_dict).__name__}")
    weight = state_dict.get("_orig_mod.enc.1.weight")
    bias = state_dict.get("_orig_mod.enc.1.bias")
    if weight is None or bias is None:
        raise KeyError(f"Missing encoder tensors in {path}")
    out: dict[str, Any] = {
        "W_enc": weight.to(device=device, dtype=dtype),
        "b_enc": bias.to(device=device, dtype=dtype),
        "hidden_dim": int(weight.shape[1]),
        "feature_dim": int(weight.shape[0]),
        "weight_dtype": str(weight.dtype),
        "bias_dtype": str(bias.dtype),
    }
    ln_weight = state_dict.get("_orig_mod.enc.0.weight")
    ln_bias = state_dict.get("_orig_mod.enc.0.bias")
    if ln_weight is not None and ln_bias is not None:
        out["ln_weight"] = ln_weight.to(device=device, dtype=dtype)
        out["ln_bias"] = ln_bias.to(device=device, dtype=dtype)
    return out


def _encode_features(hidden: torch.Tensor, encoder: dict[str, Any]) -> torch.Tensor:
    encoded_input = hidden
    if "ln_weight" in encoder and "ln_bias" in encoder:
        encoded_input = F.layer_norm(
            hidden,
            (hidden.shape[-1],),
            encoder["ln_weight"],
            encoder["ln_bias"],
        )
    return F.relu(F.linear(encoded_input, encoder["W_enc"], encoder["b_enc"]))


def _feature_table(
    clean_features: torch.Tensor,
    masked_features: torch.Tensor,
    positions: list[int],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not positions:
        return [], {"status": "empty_bucket", "position_count": 0}
    pos = [idx for idx in positions if 0 <= idx < clean_features.shape[1]]
    if not pos:
        return [], {"status": "positions_out_of_range", "position_count": 0}
    pos_tensor = torch.tensor(pos, device=clean_features.device, dtype=torch.long)
    clean = clean_features[:, pos_tensor, :].detach()
    masked = masked_features[:, pos_tensor, :].detach()
    clean_max = clean.amax(dim=(0, 1))
    masked_max = masked.amax(dim=(0, 1))
    topk = min(top_k, clean_max.numel())
    vals, idxs = torch.topk(clean_max, k=topk)
    rows = []
    for value, feature_id in zip(vals.float().cpu().tolist(), idxs.cpu().tolist(), strict=False):
        masked_value = float(masked_max[feature_id].float().detach().cpu().item())
        rows.append(
            {
                "feature_id": int(feature_id),
                "clean_activation": float(value),
                "masked_activation": masked_value,
                "drop": float(value) - masked_value,
            }
        )
    masked_top = set(torch.topk(masked_max, k=topk).indices.detach().cpu().tolist())
    clean_top = set(idxs.detach().cpu().tolist())
    jaccard = len(clean_top & masked_top) / len(clean_top | masked_top) if clean_top | masked_top else 1.0
    summary = {
        "status": "ok",
        "position_count": len(pos),
        "topk_jaccard_change": 1.0 - jaccard,
        "bucket_mean_shift": float(clean.float().mean().item() - masked.float().mean().item()),
        "mean_topk_drop": float(sum(row["drop"] for row in rows) / len(rows)) if rows else 0.0,
    }
    return rows, summary


def _build_inputs(processor, image: Image.Image, question: str, device: torch.device) -> dict[str, Any]:
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2F LLaVA clean vs evidence-mask CLT readout.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--transcoder-repo", default="KokosDev/llava15-7b-clt")
    parser.add_argument("--annotation-roots", required=True, help="Comma-separated annotation roots.")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--samples", default="okvqa_val_2847255,okvqa_val_4157235,okvqa_val_3658865")
    parser.add_argument("--prompts", default="B_direct,D_visual_only")
    parser.add_argument("--layers", default="0")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-gpu-free-gb", type=float, default=18.0)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-summary-csv", required=True)
    args = parser.parse_args()

    layers = _parse_layers(args.layers)
    sample_ids = _parse_csv(args.samples)
    prompt_names = _parse_csv(args.prompts)
    roots = [Path(p) for p in _parse_csv(args.annotation_roots)]
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "created_at": _now(),
        "model_name": args.model_name,
        "transcoder_repo": args.transcoder_repo,
        "samples": sample_ids,
        "prompts": prompt_names,
        "layers": layers,
        "env_presence": _env_presence(),
        "gpu_before": _gpu_info(),
        "processor": {},
        "config": {},
        "model_load": {},
        "encoder_load": {},
        "mask_assets": {},
        "runs": [],
        "decision": {},
    }
    gpu = payload["gpu_before"]
    if not gpu.get("available") or float(gpu.get("free_gb", 0.0)) < args.min_gpu_free_gb:
        payload["decision"] = {"status": "partial", "reason": "insufficient_gpu_free_memory"}
        _write_json(Path(args.out_json), payload)
        return 0

    try:
        from transformers import AutoConfig, AutoProcessor, LlavaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(args.model_name)
        config = AutoConfig.from_pretrained(args.model_name)
        patch_info = _processor_patch(processor, config)
        tokenizer = processor.tokenizer
        text_config = getattr(config, "text_config", None)
        payload["processor"] = {
            "status": "ok",
            "processor_class": type(processor).__name__,
            "tokenizer_class": type(tokenizer).__name__,
            **patch_info,
        }
        payload["config"] = {
            "status": "ok",
            "model_type": getattr(config, "model_type", ""),
            "text_hidden_size": getattr(text_config, "hidden_size", "") if text_config else "",
            "text_num_hidden_layers": (
                getattr(text_config, "num_hidden_layers", "") if text_config else ""
            ),
            "image_token_index": getattr(config, "image_token_index", 32000),
        }
    except Exception as exc:  # noqa: BLE001
        payload["processor"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:3000],
        }
        payload["decision"] = {"status": "blocked", "reason": "processor_or_config_failed"}
        _write_json(Path(args.out_json), payload)
        return 0

    _log("loading LLaVA base model")
    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()
        device = _first_param_device(model)
        payload["model_load"] = {"status": "ok", "class": type(model).__name__, "device": str(device)}
    except Exception as exc:  # noqa: BLE001
        payload["model_load"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
        }
        payload["decision"] = {"status": "partial", "reason": "base_model_load_failed"}
        _write_json(Path(args.out_json), payload)
        return 0

    encoders: dict[int, dict[str, Any]] = {}
    for layer in layers:
        try:
            transcoder_path = _download_transcoder(args.transcoder_repo, layer)
            encoders[layer] = _load_encoder(transcoder_path, device=device, dtype=torch.float16)
            payload["encoder_load"][str(layer)] = {
                "status": "ok",
                "transcoder_path": str(transcoder_path),
                "hidden_dim": encoders[layer]["hidden_dim"],
                "feature_dim": encoders[layer]["feature_dim"],
            }
        except Exception as exc:  # noqa: BLE001
            payload["encoder_load"][str(layer)] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:3000],
            }
    if not encoders:
        payload["decision"] = {
            "status": "blocked",
            "reason": "no_llava_clt_encoders_loaded",
            "claim_boundary": "LLaVA clean-vs-mask readout did not run because no requested CLT layers loaded.",
        }
        _write_json(Path(args.out_json), payload)
        return 0

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    usable_samples = 0
    skipped_samples: list[str] = []
    image_token_id = int(getattr(config, "image_token_index", 32000))

    for sample_id in sample_ids:
        sample = SAMPLES[sample_id]
        mask_info = _load_masks(sample, roots, work_dir)
        payload["mask_assets"][sample_id] = {
            "status": mask_info["status"],
            "source": mask_info.get("source", ""),
            "image_path": str(mask_info.get("image_path", "")),
        }
        if mask_info["status"] != "ok":
            skipped_samples.append(sample_id)
            continue
        usable_samples += 1
        image_path = Path(mask_info["image_path"])
        clean_image = Image.open(image_path).convert("RGB")
        condition_images = {
            "clean": clean_image,
            "answer_mask": _apply_mask(clean_image, mask_info["answer_mask"], (128, 128, 128)),
            "union_mask": _apply_mask(clean_image, mask_info["union_mask"], (128, 128, 128)),
        }

        for prompt_name in prompt_names:
            question = _prompt(sample["question"], prompt_name)
            clean_inputs = _build_inputs(processor, clean_image, question, device)
            input_ids = clean_inputs["input_ids"][0].detach().cpu().tolist()
            token_texts = tokenizer.convert_ids_to_tokens(input_ids)
            buckets = _bucket_positions(input_ids, token_texts, tokenizer, question, image_token_id)

            condition_features: dict[str, dict[int, torch.Tensor]] = {}
            condition_shapes = {}
            for condition, image in condition_images.items():
                inputs = _build_inputs(processor, image, question, device)
                condition_shapes[condition] = {
                    key: list(value.shape) for key, value in inputs.items() if hasattr(value, "shape")
                }
                with torch.inference_mode():
                    outputs = model(**inputs, output_hidden_states=True, use_cache=False)
                hidden_states = outputs.hidden_states
                condition_features[condition] = {}
                for layer in layers:
                    if layer not in encoders:
                        continue
                    hidden_idx = layer + 1
                    if hidden_idx >= len(hidden_states):
                        continue
                    hidden = hidden_states[hidden_idx].to(device=device, dtype=torch.float16)
                    with torch.inference_mode():
                        condition_features[condition][layer] = _encode_features(hidden, encoders[layer]).detach()
                    del hidden
                del outputs

            payload["runs"].append(
                {
                    "sample_id": sample_id,
                    "prompt_name": prompt_name,
                    "condition_input_shapes": condition_shapes,
                    "bucket_counts": {key: len(value) for key, value in buckets.items()},
                }
            )

            for layer in layers:
                if layer not in encoders:
                    summary_rows.append(
                        {
                            "sample_id": sample_id,
                            "prompt_name": prompt_name,
                            "condition": "encoder_missing",
                            "layer": layer,
                            "bucket": "",
                            "status": "encoder_load_failed",
                        }
                    )
                    continue
                if layer not in condition_features.get("clean", {}):
                    summary_rows.append(
                        {
                            "sample_id": sample_id,
                            "prompt_name": prompt_name,
                            "condition": "layer_missing",
                            "layer": layer,
                            "bucket": "",
                            "status": "hidden_state_index_out_of_range",
                        }
                    )
                    continue
                clean_features = condition_features["clean"][layer]
                for condition in ["answer_mask", "union_mask"]:
                    masked_features = condition_features[condition][layer]
                    for bucket, positions in buckets.items():
                        rows, summary = _feature_table(
                            clean_features,
                            masked_features,
                            positions,
                            args.top_k,
                        )
                        summary_rows.append(
                            {
                                "sample_id": sample_id,
                                "prompt_name": prompt_name,
                                "condition": condition,
                                "layer": layer,
                                "bucket": bucket,
                                **summary,
                            }
                        )
                        for row in rows:
                            detail_rows.append(
                                {
                                    "sample_id": sample_id,
                                    "prompt_name": prompt_name,
                                    "condition": condition,
                                    "layer": layer,
                                    "bucket": bucket,
                                    **row,
                                }
                            )
            del condition_features
            torch.cuda.empty_cache()

    payload["gpu_after"] = _gpu_info()
    if usable_samples == len(sample_ids):
        status = "pass_mask_readout"
    elif usable_samples > 0:
        status = "partial_mask_readout"
    else:
        status = "blocked_no_usable_masks"
    payload["decision"] = {
        "status": status,
        "usable_samples": usable_samples,
        "requested_samples": len(sample_ids),
        "skipped_samples": skipped_samples,
        "claim_boundary": "LLaVA clean-vs-mask CLT feature readout only; not attribution, intervention, or cross-model mechanism replication.",
    }

    detail_fields = [
        "sample_id",
        "prompt_name",
        "condition",
        "layer",
        "bucket",
        "feature_id",
        "clean_activation",
        "masked_activation",
        "drop",
    ]
    summary_fields = [
        "sample_id",
        "prompt_name",
        "condition",
        "layer",
        "bucket",
        "status",
        "position_count",
        "topk_jaccard_change",
        "bucket_mean_shift",
        "mean_topk_drop",
    ]
    _write_csv(Path(args.out_csv), detail_rows, detail_fields)
    _write_csv(Path(args.out_summary_csv), summary_rows, summary_fields)
    _write_json(Path(args.out_json), payload)
    _log(f"done status={status} usable_samples={usable_samples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
