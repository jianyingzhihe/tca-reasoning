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
    "okvqa_val_3658865": {
        "image": "COCO_val2014_000000365886.jpg",
        "question": "What brand of phone is this?",
        "answer": "samsung",
    },
}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage2g-intervention] {message}", flush=True)


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


def _parse_float_csv(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


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
    if prompt_name == "C_step_only":
        return (
            f"{question} Think step by step internally, then reply with only one short sentence "
            "in exactly this format: The answer is <short answer>."
        )
    if prompt_name == "A_step_visual":
        return (
            f"{question} Think step by step from visual evidence internally, then reply with only one short sentence "
            "in exactly this format: The answer is <short answer>."
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
    out: dict[str, Any] = {"status": "missing", "diagnostics": {}}
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
        shifted_path = root / "exported_masks" / stem / "shifted.png"
        shuffled_path = root / "exported_masks" / stem / "shuffled.png"
        if answer_path.exists():
            answer = Image.open(answer_path).convert("L").resize(image.size)
            relate = Image.open(relate_path).convert("L").resize(image.size) if relate_path.exists() else None
            union = ImageChops.lighter(answer, relate) if relate is not None else answer.copy()
            shifted = Image.open(shifted_path).convert("L").resize(image.size) if shifted_path.exists() else None
            shuffled = Image.open(shuffled_path).convert("L").resize(image.size) if shuffled_path.exists() else None
            out.update(
                {
                    "status": "ok",
                    "answer_mask": answer,
                    "union_mask": union,
                    "shifted_mask": shifted,
                    "shuffled_mask": shuffled,
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


def _qwen_inputs(processor, image: Image.Image, image_path: str, question: str, answer_prefix: str, device: torch.device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + answer_prefix
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _qwen_bucket_positions(input_ids: list[int], token_texts: list[str]) -> dict[str, list[int]]:
    image_pos = [
        idx
        for idx, token in enumerate(token_texts)
        if "image" in token.lower() or "vision" in token.lower() or token == "<image>"
    ]
    return {"image_marker_or_span": list(range(min(image_pos), max(image_pos) + 1)) if image_pos else []}


def _llava_inputs(processor, image: Image.Image, question: str, answer_prefix: str, device: torch.device):
    prompt = f"USER: <image>\n{question}\nASSISTANT: {answer_prefix}"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _llava_bucket_positions(input_ids: list[int], image_token_id: int) -> dict[str, list[int]]:
    return {"image_token_span": [idx for idx, token_id in enumerate(input_ids) if token_id == image_token_id]}


def _target_candidates(tokenizer, answer: str) -> list[dict[str, Any]]:
    variants = [answer, " " + answer, answer.capitalize(), " " + answer.capitalize()]
    out = []
    seen = set()
    for variant in variants:
        ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        token_id = int(ids[0])
        if token_id in seen:
            continue
        seen.add(token_id)
        try:
            token_text = tokenizer.decode([token_id])
        except Exception:  # noqa: BLE001
            token_text = str(token_id)
        out.append({"variant": variant, "token_id": token_id, "token_text": token_text})
    return out


def _rank_and_top(logits: torch.Tensor, tokenizer, candidate_ids: list[int]) -> dict[str, Any]:
    row = logits[0, -1].float()
    probs = torch.softmax(row, dim=-1)
    best = None
    for token_id in candidate_ids:
        token_logit = float(row[token_id].item())
        rank = int((row > row[token_id]).sum().item() + 1)
        item = {
            "target_token_id": int(token_id),
            "target_token": tokenizer.decode([int(token_id)]),
            "target_logit": token_logit,
            "target_prob": float(probs[token_id].item()),
            "target_rank": rank,
        }
        if best is None or rank < best["target_rank"]:
            best = item
    top_vals, top_ids = torch.topk(row, k=5)
    assert best is not None
    best["top1_token_id"] = int(top_ids[0].item())
    best["top1_token"] = tokenizer.decode([int(top_ids[0].item())])
    best["top1_logit"] = float(top_vals[0].item())
    best["top5"] = [
        {
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)]),
            "logit": float(value),
        }
        for token_id, value in zip(top_ids.cpu().tolist(), top_vals.float().cpu().tolist(), strict=False)
    ]
    return best


def _replace_hidden(output: Any, new_hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (new_hidden, *output[1:])
    return new_hidden


def _select_features(
    clean_features: torch.Tensor,
    masked_features: torch.Tensor,
    positions: list[int],
    top_k: int,
    control_pool_size: int,
) -> dict[str, Any]:
    if not positions:
        return {"status": "empty_positions"}
    pos = torch.tensor(positions, device=clean_features.device, dtype=torch.long)
    clean = clean_features[:, pos, :].detach()
    masked = masked_features[:, pos, :].detach()
    clean_max = clean.amax(dim=(0, 1)).float()
    masked_max = masked.amax(dim=(0, 1)).float()
    drop = clean_max - masked_max
    active = clean_max > 0
    if not active.any():
        return {"status": "no_active_features"}
    active_scores = drop.clone()
    active_scores[~active] = -float("inf")
    k = min(top_k, int(active.sum().item()))
    evidence_vals, evidence_ids = torch.topk(active_scores, k=k)
    evidence_ids = evidence_ids.detach().cpu()
    evidence_mean_activation = float(clean_max[evidence_ids.to(clean_max.device)].mean().item())
    evidence_mean_drop = float(evidence_vals.float().mean().item())

    pool_k = min(control_pool_size, clean_max.numel())
    _pool_vals, pool_ids = torch.topk(clean_max, k=pool_k)
    evidence_set = {int(x) for x in evidence_ids.tolist()}
    candidates = []
    denom_drop = max(abs(evidence_mean_drop), 1e-6)
    denom_activation = max(abs(evidence_mean_activation), 1e-6)
    for feat_id in pool_ids.detach().cpu().tolist():
        feat_id = int(feat_id)
        if feat_id in evidence_set:
            continue
        score = (
            abs(float(drop[feat_id].item())) / denom_drop
            + abs(float(clean_max[feat_id].item()) - evidence_mean_activation) / denom_activation
        )
        candidates.append((score, feat_id))
    candidates.sort()
    control_ids = [feat_id for _score, feat_id in candidates[:k]]
    if len(control_ids) < k:
        return {"status": "insufficient_control_features"}

    def feature_payload(feat_ids: list[int]) -> list[dict[str, Any]]:
        rows = []
        for feat_id in feat_ids:
            rows.append(
                {
                    "feature_id": int(feat_id),
                    "clean_activation": float(clean_max[feat_id].item()),
                    "masked_activation": float(masked_max[feat_id].item()),
                    "drop": float(drop[feat_id].item()),
                }
            )
        return rows

    return {
        "status": "ok",
        "positions": positions,
        "evidence_feature_ids": [int(x) for x in evidence_ids.tolist()],
        "control_feature_ids": control_ids,
        "evidence_features": feature_payload([int(x) for x in evidence_ids.tolist()]),
        "control_features": feature_payload(control_ids),
    }


def _patch_tensor(
    hidden: torch.Tensor,
    *,
    positions: list[int],
    feature_ids: list[int],
    feature_acts: torch.Tensor,
    decoder_vectors: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if not positions or not feature_ids:
        return hidden
    pos_tensor = torch.tensor(positions, device=hidden.device, dtype=torch.long)
    feat_tensor = torch.tensor(feature_ids, device=feature_acts.device, dtype=torch.long)
    acts = feature_acts[:, pos_tensor, :][:, :, feat_tensor].to(hidden.device, dtype=hidden.dtype)
    vectors = decoder_vectors.to(hidden.device, dtype=hidden.dtype)
    patch = torch.einsum("bpf,fd->bpd", acts, vectors)
    new_hidden = hidden.clone()
    new_hidden[:, pos_tensor, :] = new_hidden[:, pos_tensor, :] - scale * patch
    return new_hidden


def _score_with_hook(
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
            patched = _patch_tensor(hidden, **intervention)
            return _replace_hidden(output, patched)

        handle = module.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=False, use_cache=False)
    finally:
        if handle is not None:
            handle.remove()
    return _rank_and_top(outputs.logits, tokenizer, target_ids)


def _load_llava_encoder_decoder(path: Path, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu")
    state_dict = obj.get("state_dict", obj)
    enc_w = state_dict["_orig_mod.enc.1.weight"]
    enc_b = state_dict["_orig_mod.enc.1.bias"]
    ln_w = state_dict.get("_orig_mod.enc.0.weight")
    ln_b = state_dict.get("_orig_mod.enc.0.bias")
    dec_w = state_dict["_orig_mod.dec.weight"]
    hidden_dim = int(enc_w.shape[1])
    feature_dim = int(enc_w.shape[0])
    out = {
        "W_enc": enc_w.to(device=device, dtype=dtype),
        "b_enc": enc_b.to(device=device, dtype=dtype),
        "hidden_dim": hidden_dim,
        "feature_dim": feature_dim,
        "dec_weight": dec_w.to(device=device, dtype=dtype),
    }
    if ln_w is not None and ln_b is not None:
        out["ln_weight"] = ln_w.to(device=device, dtype=dtype)
        out["ln_bias"] = ln_b.to(device=device, dtype=dtype)
    return out


def _llava_encode(hidden: torch.Tensor, encoder: dict[str, Any]) -> torch.Tensor:
    x = hidden
    if "ln_weight" in encoder and "ln_bias" in encoder:
        x = F.layer_norm(hidden, (hidden.shape[-1],), encoder["ln_weight"], encoder["ln_bias"])
    return F.relu(F.linear(x, encoder["W_enc"], encoder["b_enc"]))


def _llava_decoder_vectors(encoder: dict[str, Any], feature_ids: list[int]) -> torch.Tensor:
    ids = torch.tensor(feature_ids, device=encoder["dec_weight"].device, dtype=torch.long)
    dec_w = encoder["dec_weight"]
    hidden_dim = int(encoder["hidden_dim"])
    feature_dim = int(encoder["feature_dim"])
    if dec_w.shape == (hidden_dim, feature_dim):
        return dec_w[:, ids].T.contiguous()
    if dec_w.shape == (feature_dim, hidden_dim):
        return dec_w[ids].contiguous()
    raise ValueError(f"Unexpected LLaVA decoder weight shape: {tuple(dec_w.shape)}")


def _download_llava_transcoder(repo_id: str, layer: int) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=f"transcoder_L{layer}.pt", local_files_only=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2G cross-model feature-direction intervention smoke.")
    parser.add_argument("--model-family", choices=["qwen", "llava"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--transcoder-ref", required=True)
    parser.add_argument("--annotation-roots", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--samples", default="okvqa_val_2847255,okvqa_val_4157235,okvqa_val_3658865")
    parser.add_argument("--prompts", default="B_direct,D_visual_only")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--top-k-features", type=int, default=4)
    parser.add_argument("--control-pool-size", type=int, default=512)
    parser.add_argument("--scales", default="1.0")
    parser.add_argument("--signed-probe", action="store_true")
    parser.add_argument("--answer-prefix", default="The answer is ")
    parser.add_argument("--min-gpu-free-gb", type=float, default=18.0)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

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
        "transcoder_ref": args.transcoder_ref,
        "layer": args.layer,
        "scales": scales,
        "signed_probe": bool(args.signed_probe),
        "samples": sample_ids,
        "prompts": prompt_names,
        "env_presence": _env_presence(),
        "gpu_before": _gpu_info(),
        "selection": [],
        "decision": {},
        "claim_boundary": "Feature-direction ablation smoke only; not source tracing, not matched nearest control, not causal route replication.",
    }
    gpu = payload["gpu_before"]
    if not gpu.get("available") or float(gpu.get("free_gb", 0.0)) < args.min_gpu_free_gb:
        payload["decision"] = {"status": "partial", "reason": "insufficient_gpu_free_memory"}
        _write_json(Path(args.out_json), payload)
        return 0

    if args.model_family == "qwen":
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

        processor = AutoProcessor.from_pretrained(args.model_name, local_files_only=True)
        tokenizer = processor.tokenizer
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        dtype = torch.bfloat16
        model.eval()
        device = _first_param_device(model)
        transcoders, config = load_transcoder_from_hub(
            args.transcoder_ref,
            device=device,
            dtype=dtype,
            lazy_encoder=True,
            lazy_decoder=True,
        )
        target_module = model.language_model.layers[args.layer]

        def build_inputs(image: Image.Image, image_path: Path, question: str):
            return _qwen_inputs(processor, image, str(image_path), question, args.answer_prefix, device)

        def bucket_positions(input_ids: list[int], token_texts: list[str]) -> tuple[str, list[int]]:
            buckets = _qwen_bucket_positions(input_ids, token_texts)
            return "image_marker_or_span", buckets["image_marker_or_span"]

        def encode_features(hidden):
            return transcoders.encode_layer(hidden.to(device), args.layer, apply_activation_function=True)

        def decoder_vectors(feature_ids: list[int]):
            ids = torch.tensor(feature_ids, device=device, dtype=torch.long)
            vectors = transcoders._get_decoder_vectors(args.layer, ids)
            if vectors.ndim == 3:
                vectors = vectors[:, 0, :]
            return vectors.to(device=device, dtype=dtype)

        hidden_index = args.layer
        payload["transcoder"] = {
            "type": type(transcoders).__name__,
            "config_model_kind": config.get("model_kind", ""),
            "hook_approximation": "subtract feature activation times decoder offset 0 at native language layer output",
        }
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
        dtype = torch.float16
        model.eval()
        device = _first_param_device(model)
        transcoder_path = _download_llava_transcoder(args.transcoder_ref, args.layer)
        llava_encoder = _load_llava_encoder_decoder(transcoder_path, device=device, dtype=dtype)
        target_module = model.language_model.layers[args.layer]
        image_token_id = int(getattr(config, "image_token_index", 32000))

        def build_inputs(image: Image.Image, image_path: Path, question: str):
            return _llava_inputs(processor, image, question, args.answer_prefix, device)

        def bucket_positions(input_ids: list[int], token_texts: list[str]) -> tuple[str, list[int]]:
            del token_texts
            buckets = _llava_bucket_positions(input_ids, image_token_id)
            return "image_token_span", buckets["image_token_span"]

        def encode_features(hidden):
            return _llava_encode(hidden.to(device=device, dtype=dtype), llava_encoder)

        def decoder_vectors(feature_ids: list[int]):
            return _llava_decoder_vectors(llava_encoder, feature_ids).to(device=device, dtype=dtype)

        hidden_index = args.layer + 1
        payload["transcoder"] = {
            "type": "KokosDev/llava15-7b-clt custom pt",
            "path": str(transcoder_path),
            "processor_patch": patch_info,
            "hook_approximation": "subtract feature activation times decoder vector at native language layer output",
        }

    rows: list[dict[str, Any]] = []
    usable_runs = 0
    skipped: list[dict[str, Any]] = []

    for sample_id in sample_ids:
        sample = SAMPLES[sample_id]
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
            bucket_name, positions = bucket_positions(input_ids, token_texts)
            target_candidates = _target_candidates(tokenizer, sample["answer"])
            target_ids = [item["token_id"] for item in target_candidates]
            if not positions or not target_ids:
                skipped.append({"sample_id": sample_id, "prompt_name": prompt_name, "reason": "positions_or_target_missing"})
                continue

            with torch.inference_mode():
                clean_outputs = model(**clean_inputs, output_hidden_states=True, use_cache=False)
                union_outputs = model(**union_inputs, output_hidden_states=True, use_cache=False)
            clean_hidden = clean_outputs.hidden_states[hidden_index].to(device=device, dtype=dtype)
            union_hidden = union_outputs.hidden_states[hidden_index].to(device=device, dtype=dtype)
            clean_features = encode_features(clean_hidden).detach()
            union_features = encode_features(union_hidden).detach()
            selection = _select_features(
                clean_features,
                union_features,
                positions,
                top_k=args.top_k_features,
                control_pool_size=args.control_pool_size,
            )
            payload["selection"].append(
                {
                    "sample_id": sample_id,
                    "prompt_name": prompt_name,
                    "bucket": bucket_name,
                    "status": selection["status"],
                    "evidence_features": selection.get("evidence_features", []),
                    "control_features": selection.get("control_features", []),
                }
            )
            if selection["status"] != "ok":
                skipped.append({"sample_id": sample_id, "prompt_name": prompt_name, "reason": selection["status"]})
                continue

            baseline = _rank_and_top(clean_outputs.logits, tokenizer, target_ids)
            chosen_target_id = int(baseline["target_token_id"])
            feature_acts = clean_features.detach()
            evidence_ids = selection["evidence_feature_ids"]
            control_ids = selection["control_feature_ids"]
            interventions: dict[str, dict[str, Any] | None] = {"baseline": None}
            groups = {
                "evidence_top1": evidence_ids[:1],
                "evidence_topk": evidence_ids,
                "control_topk": control_ids,
            }
            for scale in scales:
                scale_label = str(scale).replace(".", "p").replace("-", "m")
                for group_name, group_ids in groups.items():
                    interventions[f"{group_name}_subtract_s{scale_label}"] = {
                        "positions": positions,
                        "feature_ids": group_ids,
                        "feature_acts": feature_acts,
                        "decoder_vectors": decoder_vectors(group_ids),
                        "scale": scale,
                    }
                    if args.signed_probe:
                        interventions[f"{group_name}_add_s{scale_label}"] = {
                            "positions": positions,
                            "feature_ids": group_ids,
                            "feature_acts": feature_acts,
                            "decoder_vectors": decoder_vectors(group_ids),
                            "scale": -scale,
                        }

            usable_runs += 1
            for condition, intervention in interventions.items():
                if condition == "baseline":
                    score = baseline
                else:
                    score = _score_with_hook(
                        model,
                        clean_inputs,
                        tokenizer,
                        [chosen_target_id],
                        target_module,
                        intervention,
                    )
                rows.append(
                    {
                        "model_family": args.model_family,
                        "sample_id": sample_id,
                        "prompt_name": prompt_name,
                        "layer": args.layer,
                        "bucket": bucket_name,
                        "condition": condition,
                        "target_answer": sample["answer"],
                        "target_token_id": score["target_token_id"],
                        "target_token": score["target_token"],
                        "target_logit": score["target_logit"],
                        "target_prob": score["target_prob"],
                        "target_rank": score["target_rank"],
                        "baseline_target_logit": baseline["target_logit"],
                        "baseline_target_rank": baseline["target_rank"],
                        "delta_logit_vs_baseline": score["target_logit"] - baseline["target_logit"],
                        "rank_damage_vs_baseline": score["target_rank"] - baseline["target_rank"],
                        "top1_token": score["top1_token"],
                        "baseline_top1_token": baseline["top1_token"],
                        "top1_changed": score["top1_token_id"] != baseline["top1_token_id"],
                        "evidence_feature_ids": "|".join(str(x) for x in evidence_ids),
                        "control_feature_ids": "|".join(str(x) for x in control_ids),
                        "target_candidates": json.dumps(target_candidates, ensure_ascii=False),
                    }
                )
            del clean_outputs, union_outputs, clean_features, union_features, feature_acts
            torch.cuda.empty_cache()

    payload["gpu_after"] = _gpu_info()
    payload["decision"] = {
        "status": "pass_intervention_smoke" if usable_runs else "blocked_no_usable_runs",
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
        "condition",
        "target_answer",
        "target_token_id",
        "target_token",
        "target_logit",
        "target_prob",
        "target_rank",
        "baseline_target_logit",
        "baseline_target_rank",
        "delta_logit_vs_baseline",
        "rank_damage_vs_baseline",
        "top1_token",
        "baseline_top1_token",
        "top1_changed",
        "evidence_feature_ids",
        "control_feature_ids",
        "target_candidates",
    ]
    _write_csv(Path(args.out_csv), rows, fields)
    _write_json(Path(args.out_json), payload)
    _log(f"done status={payload['decision']['status']} usable_runs={usable_runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
