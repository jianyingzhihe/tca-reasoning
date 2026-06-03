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
    print(f"[stage2f-qwen-mask] {message}", flush=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


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


def _parse_layers(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_samples(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


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
        shape_type = shape.get("shape_type", "polygon")
        if shape_type == "rectangle" and len(points) >= 2:
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
        if image_path:
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
                union = Image.eval(ImageChops.lighter(answer, relate), lambda x: x)  # type: ignore[name-defined]
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
                union = answer.copy()
                if relate is not None:
                    union = Image.eval(ImageChops.lighter(answer, relate), lambda x: x)  # type: ignore[name-defined]
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


def _build_inputs(processor, image: Image.Image, image_path: str, question: str, device: torch.device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _bucket_positions(input_ids: list[int], token_texts: list[str], tokenizer, question: str) -> dict[str, list[int]]:
    def encode(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    def find(candidates: list[list[int]]) -> tuple[int | None, int | None]:
        for cand in candidates:
            n = len(cand)
            if n == 0:
                continue
            for start in range(0, len(input_ids) - n + 1):
                if input_ids[start : start + n] == cand:
                    return start, start + n
        return None, None

    q_start, q_end = find([encode(question), encode(" " + question), encode("\n" + question)])
    image_pos = [
        i
        for i, tok in enumerate(token_texts)
        if "image" in tok.lower() or "vision" in tok.lower() or tok == "<image>"
    ]
    assistant_pos = [i for i, tok in enumerate(token_texts) if "assistant" in tok.lower()]
    if q_end is not None:
        assistant_pos = [i for i in assistant_pos if i >= q_end]
    assistant_start = min(assistant_pos) if assistant_pos else None
    buckets = {
        "question": list(range(q_start, q_end)) if q_start is not None and q_end is not None else [],
        "image_marker_or_span": list(range(min(image_pos), max(image_pos) + 1)) if image_pos else [],
        "assistant_prefix": list(range(assistant_start, len(input_ids))) if assistant_start is not None else [],
        "last_prompt_token": [len(input_ids) - 1],
        "position_2_diagnostic": [2] if len(input_ids) > 2 else [],
    }
    return buckets


def _feature_table(
    clean_features: torch.Tensor,
    masked_features: torch.Tensor,
    positions: list[int],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not positions:
        return [], {"status": "empty_bucket", "position_count": 0}
    pos = torch.tensor(positions, device=clean_features.device, dtype=torch.long)
    clean = clean_features[:, pos, :].detach()
    masked = masked_features[:, pos, :].detach()
    clean_max = clean.amax(dim=(0, 1))
    masked_max = masked.amax(dim=(0, 1))
    topk = min(top_k, clean_max.numel())
    vals, idxs = torch.topk(clean_max, k=topk)
    rows = []
    for value, feature_id in zip(vals.cpu().tolist(), idxs.cpu().tolist(), strict=False):
        masked_value = float(masked_max[feature_id].detach().cpu().item())
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
        "position_count": len(positions),
        "topk_jaccard_change": 1.0 - jaccard,
        "bucket_mean_shift": float(clean.float().mean().item() - masked.float().mean().item()),
        "mean_topk_drop": float(sum(r["drop"] for r in rows) / len(rows)) if rows else 0.0,
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2F Qwen clean vs evidence-mask feature readout.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--transcoder-set", default="KokosDev/qwen2p5vl-7b-clt")
    parser.add_argument("--annotation-roots", required=True, help="Comma-separated annotation roots.")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--samples", default="okvqa_val_2847255,okvqa_val_4157235,okvqa_val_3605295")
    parser.add_argument("--prompts", default="B_direct,D_visual_only")
    parser.add_argument("--layers", default="0,13,26")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-gpu-free-gb", type=float, default=18.0)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-summary-csv", required=True)
    args = parser.parse_args()

    layers = _parse_layers(args.layers)
    sample_ids = _parse_samples(args.samples)
    prompt_names = _parse_samples(args.prompts)
    roots = [Path(p) for p in _parse_samples(args.annotation_roots)]
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "created_at": _now(),
        "model_name": args.model_name,
        "transcoder_set": args.transcoder_set,
        "samples": sample_ids,
        "prompts": prompt_names,
        "layers": layers,
        "env_presence": _env_presence(),
        "gpu_before": _gpu_info(),
        "mask_assets": {},
        "runs": [],
        "decision": {},
    }
    gpu = payload["gpu_before"]
    if not gpu.get("available") or float(gpu.get("free_gb", 0.0)) < args.min_gpu_free_gb:
        payload["decision"] = {"status": "partial", "reason": "insufficient_gpu_free_memory"}
        _write_json(Path(args.out_json), payload)
        return 0

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

    processor = AutoProcessor.from_pretrained(args.model_name, local_files_only=True)
    tokenizer = processor.tokenizer
    _log("loading Qwen base model")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = _first_param_device(model)
    transcoders, _config = load_transcoder_from_hub(
        args.transcoder_set,
        device=device,
        dtype=torch.bfloat16,
        lazy_encoder=True,
        lazy_decoder=True,
    )

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    usable_samples = 0
    skipped_samples = []

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
            clean_inputs = _build_inputs(processor, clean_image, str(image_path), question, device)
            input_ids = clean_inputs["input_ids"][0].detach().cpu().tolist()
            token_texts = tokenizer.convert_ids_to_tokens(input_ids)
            buckets = _bucket_positions(input_ids, token_texts, tokenizer, question)

            condition_features: dict[str, dict[int, torch.Tensor]] = {}
            condition_shapes = {}
            for condition, image in condition_images.items():
                inputs = _build_inputs(processor, image, str(image_path), question, device)
                condition_shapes[condition] = list(inputs["input_ids"].shape)
                with torch.inference_mode():
                    outputs = model(**inputs, output_hidden_states=True, use_cache=False)
                hidden_states = outputs.hidden_states
                condition_features[condition] = {}
                for layer in layers:
                    with torch.inference_mode():
                        condition_features[condition][layer] = transcoders.encode_layer(
                            hidden_states[layer].to(device),
                            layer,
                            apply_activation_function=True,
                        ).detach()

            run_summary = {
                "sample_id": sample_id,
                "prompt_name": prompt_name,
                "condition_input_shapes": condition_shapes,
                "bucket_counts": {k: len(v) for k, v in buckets.items()},
            }
            payload["runs"].append(run_summary)

            for layer in layers:
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
                for condition in list(condition_features):
                    del condition_features[condition][layer]
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
        "claim_boundary": "Qwen clean-vs-mask CLT feature readout only; not attribution, intervention, or cross-model mechanism replication.",
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
