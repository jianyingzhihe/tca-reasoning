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
from PIL import Image


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage2f-qwen-pos] {message}", flush=True)


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
        "free_bytes": free,
        "total_bytes": total,
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


def _input_summary(inputs: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for key, value in inputs.items():
        if hasattr(value, "shape"):
            summary[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device) if hasattr(value, "device") else "",
            }
        else:
            summary[key] = str(type(value).__name__)
    return summary


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
    *,
    input_ids: list[int],
    token_texts: list[str],
    tokenizer,
    question: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    question_candidates = []
    for variant in [question, " " + question, "\n" + question]:
        encoded = tokenizer(variant, add_special_tokens=False)
        question_candidates.append(list(encoded["input_ids"]))
    q_start, q_end = _find_subsequence(input_ids, question_candidates)

    image_positions = [
        i
        for i, tok in enumerate(token_texts)
        if "image" in tok.lower() or "vision" in tok.lower() or tok in {"<image>"}
    ]
    if image_positions:
        image_start = min(image_positions)
        image_end = max(image_positions) + 1
    else:
        image_start = image_end = None

    assistant_positions = [i for i, tok in enumerate(token_texts) if "assistant" in tok.lower()]
    if q_end is not None:
        assistant_positions_after_question = [i for i in assistant_positions if i >= q_end]
    else:
        assistant_positions_after_question = assistant_positions
    assistant_start = (
        min(assistant_positions_after_question) if assistant_positions_after_question else None
    )

    rows: list[dict[str, Any]] = []
    for pos, (token_id, token_text) in enumerate(zip(input_ids, token_texts, strict=False)):
        labels: list[str] = []
        if pos == 2:
            labels.append("position_2_diagnostic")
        if image_start is not None and image_start <= pos < image_end:
            labels.append("image_marker_or_span")
        if q_start is not None and q_end is not None and q_start <= pos < q_end:
            labels.append("question")
        if assistant_start is not None and pos >= assistant_start:
            labels.append("assistant_prefix")
        if pos == len(input_ids) - 1:
            labels.append("last_prompt_token")
        if token_text.startswith("<|") or token_text.endswith("|>"):
            labels.append("special/template")
        if not labels:
            labels.append("other_text_or_template")
        rows.append(
            {
                "position": pos,
                "token_id": int(token_id),
                "token_text": token_text,
                "bucket": "|".join(labels),
                "primary_bucket": labels[0],
            }
        )

    diagnostics = {
        "question_span": [q_start, q_end] if q_start is not None and q_end is not None else None,
        "image_span": (
            [image_start, image_end] if image_start is not None and image_end is not None else None
        ),
        "assistant_start": assistant_start,
        "last_prompt_token": len(input_ids) - 1,
        "position_2": rows[2] if len(rows) > 2 else None,
    }
    return rows, diagnostics


def _bucket_index(token_rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    buckets = {
        "special/template": [],
        "image_marker_or_span": [],
        "question": [],
        "assistant_prefix": [],
        "last_prompt_token": [],
        "position_2_diagnostic": [],
        "other_text_or_template": [],
    }
    for row in token_rows:
        labels = str(row["bucket"]).split("|")
        pos = int(row["position"])
        for label in labels:
            if label in buckets:
                buckets[label].append(pos)
    return buckets


def _topk_for_positions(features: torch.Tensor, positions: list[int], k: int) -> dict[str, Any]:
    if not positions:
        return {"status": "empty_bucket", "position_count": 0}
    pos_tensor = torch.tensor(positions, device=features.device, dtype=torch.long)
    selected = features[:, pos_tensor, :].detach()
    batch, seq, feature_dim = selected.shape
    flat = selected.reshape(-1)
    topk = min(k, flat.numel())
    vals, idxs = torch.topk(flat, k=topk)
    rows = []
    for value, flat_idx in zip(vals.cpu().tolist(), idxs.cpu().tolist(), strict=False):
        pos_flat = flat_idx // feature_dim
        feature_id = flat_idx % feature_dim
        batch_idx = pos_flat // seq
        bucket_pos_idx = pos_flat % seq
        rows.append(
            {
                "batch_index": int(batch_idx),
                "position_index": int(positions[bucket_pos_idx]),
                "feature_id": int(feature_id),
                "activation": float(value),
            }
        )
    return {
        "status": "ok",
        "position_count": len(positions),
        "shape": [int(batch), int(seq), int(feature_dim)],
        "active_positive_count": int((selected > 0).sum().item()),
        "max_activation": float(selected.max().item()) if selected.numel() else 0.0,
        "mean_activation": float(selected.float().mean().item()) if selected.numel() else 0.0,
        "topk": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2F Qwen token/position mapping smoke.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--transcoder-set", default="KokosDev/qwen2p5vl-7b-clt")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--layers", default="0,13,26")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--min-gpu-free-gb", type=float, default=18.0)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-token-csv", default="")
    parser.add_argument("--out-bucket-csv", default="")
    args = parser.parse_args()

    layers = _parse_layers(args.layers)
    payload: dict[str, Any] = {
        "created_at": _now(),
        "model_name": args.model_name,
        "transcoder_set": args.transcoder_set,
        "image_path": args.image_path,
        "question": args.question,
        "layers": layers,
        "env_presence": _env_presence(),
        "gpu_before": _gpu_info(),
        "processor": {},
        "model_load": {},
        "transcoder_load": {},
        "forward": {},
        "position_mapping": {},
        "bucket_feature_readout": {},
        "decision": {},
    }
    gpu = payload["gpu_before"]
    if not gpu.get("available") or float(gpu.get("free_gb", 0.0)) < args.min_gpu_free_gb:
        payload["decision"] = {
            "status": "partial",
            "reason": "skipped_forward_because_gpu_free_memory_below_threshold",
        }
        _write_json(Path(args.out_json), payload)
        return 0

    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(args.model_name, local_files_only=True)
        tokenizer = processor.tokenizer
        payload["processor"] = {
            "status": "ok",
            "processor_class": type(processor).__name__,
            "tokenizer_class": type(tokenizer).__name__,
        }
    except Exception as exc:  # noqa: BLE001
        payload["processor"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
        }
        payload["decision"] = {"status": "blocked", "reason": "processor_failed"}
        _write_json(Path(args.out_json), payload)
        return 0

    _log("loading Qwen base model")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
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
            "error": str(exc)[:3000],
        }
        payload["decision"] = {"status": "partial", "reason": "base_model_load_failed"}
        _write_json(Path(args.out_json), payload)
        return 0

    try:
        from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

        transcoders, config = load_transcoder_from_hub(
            args.transcoder_set,
            device=device,
            dtype=torch.bfloat16,
            lazy_encoder=True,
            lazy_decoder=True,
        )
        payload["transcoder_load"] = {
            "status": "ok",
            "class": type(transcoders).__name__,
            "n_layers": int(getattr(transcoders, "n_layers", 0)),
            "d_transcoder": int(getattr(transcoders, "d_transcoder", 0)),
            "config_hidden_dim": config.get("hidden_dim", ""),
            "config_feature_dim": config.get("feature_dim", ""),
        }
    except Exception as exc:  # noqa: BLE001
        payload["transcoder_load"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:3000],
        }
        payload["decision"] = {"status": "blocked", "reason": "transcoder_load_failed"}
        _write_json(Path(args.out_json), payload)
        return 0

    image = Image.open(args.image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image_path},
                {"type": "text", "text": args.question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    payload["forward"]["input_summary"] = _input_summary(inputs)

    input_ids = inputs["input_ids"][0].detach().cpu().tolist()
    token_texts = tokenizer.convert_ids_to_tokens(input_ids)
    token_rows, diagnostics = _bucket_positions(
        input_ids=input_ids,
        token_texts=token_texts,
        tokenizer=tokenizer,
        question=args.question,
    )
    buckets = _bucket_index(token_rows)
    payload["position_mapping"] = {
        "status": "ok",
        "sequence_length": len(input_ids),
        "diagnostics": diagnostics,
        "bucket_counts": {key: len(value) for key, value in buckets.items()},
    }

    _log("running forward and CLT bucket readout")
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden_states = getattr(outputs, "hidden_states", None)
    payload["forward"].update(
        {
            "status": "ok",
            "logits_shape": list(outputs.logits.shape) if hasattr(outputs, "logits") else "",
            "hidden_states_count": len(hidden_states) if hidden_states is not None else 0,
        }
    )

    bucket_rows: list[dict[str, Any]] = []
    readout: dict[str, Any] = {}
    assert hidden_states is not None
    for layer in layers:
        hidden = hidden_states[layer].to(device)
        with torch.inference_mode():
            features = transcoders.encode_layer(hidden, layer, apply_activation_function=True)
        layer_out = {}
        for bucket_name, positions in buckets.items():
            summary = _topk_for_positions(features, positions, args.top_k)
            layer_out[bucket_name] = summary
            if summary.get("status") == "ok":
                for item in summary["topk"]:
                    bucket_rows.append(
                        {
                            "layer": layer,
                            "bucket": bucket_name,
                            "position_count": summary["position_count"],
                            "feature_id": item["feature_id"],
                            "position_index": item["position_index"],
                            "activation": item["activation"],
                        }
                    )
        readout[str(layer)] = layer_out
        del features
    payload["bucket_feature_readout"] = readout
    payload["gpu_after"] = _gpu_info()

    question_ok = diagnostics.get("question_span") is not None
    last_ok = bool(buckets.get("last_prompt_token"))
    if question_ok and last_ok:
        status = "pass_position_mapping"
    else:
        status = "partial_position_mapping_pass"
    payload["decision"] = {
        "status": status,
        "reason": "question_and_last_prompt_position_mapped" if status.startswith("pass") else "feature_readout_ok_but_some_position_spans_unresolved",
        "claim_boundary": "Token/position mapping and CLT readout smoke only; not attribution, intervention, or cross-model replication.",
    }

    if args.out_token_csv:
        _write_csv(
            Path(args.out_token_csv),
            token_rows,
            ["position", "token_id", "token_text", "bucket", "primary_bucket"],
        )
    if args.out_bucket_csv:
        _write_csv(
            Path(args.out_bucket_csv),
            bucket_rows,
            ["layer", "bucket", "position_count", "feature_id", "position_index", "activation"],
        )
    _write_json(Path(args.out_json), payload)
    _log(f"done status={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
