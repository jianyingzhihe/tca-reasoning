#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    print(f"[stage2f-qwen-readout] {message}", flush=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


def _parse_layers(raw: str) -> list[int]:
    layers: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        layers.append(int(part))
    return layers


def _topk_summary(features: torch.Tensor, k: int) -> dict[str, Any]:
    # features shape is expected to be [batch, seq, feature_dim].
    feat = features.detach()
    if feat.ndim == 2:
        feat = feat.unsqueeze(0)
    if feat.ndim != 3:
        return {"status": "failed", "reason": f"unexpected_features_ndim_{feat.ndim}"}

    batch, seq_len, feature_dim = feat.shape
    flat = feat.reshape(-1)
    topk = min(k, flat.numel())
    vals, idxs = torch.topk(flat, k=topk)
    rows = []
    for value, flat_idx in zip(vals.cpu().tolist(), idxs.cpu().tolist(), strict=False):
        pos_flat = flat_idx // feature_dim
        feature_id = flat_idx % feature_dim
        batch_idx = pos_flat // seq_len
        pos_idx = pos_flat % seq_len
        rows.append(
            {
                "batch_index": int(batch_idx),
                "position_index": int(pos_idx),
                "feature_id": int(feature_id),
                "activation": float(value),
            }
        )

    active_positive = int((feat > 0).sum().item())
    active_gt_1e_4 = int((feat > 1e-4).sum().item())
    max_value = float(feat.max().item()) if feat.numel() else 0.0
    mean_positive = float(feat[feat > 0].float().mean().item()) if active_positive else 0.0
    return {
        "status": "ok",
        "shape": [int(batch), int(seq_len), int(feature_dim)],
        "dtype": str(feat.dtype),
        "device": str(feat.device),
        "active_positive_count": active_positive,
        "active_gt_1e_4_count": active_gt_1e_4,
        "max_activation": max_value,
        "mean_positive_activation": mean_positive,
        "topk": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 2F Qwen native hidden-state to CLT feature-readout smoke."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--transcoder-set", default="KokosDev/qwen2p5vl-7b-clt")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--question", default="What does the sign say?")
    parser.add_argument("--layers", default="0,13,26")
    parser.add_argument("--hidden-state-offset", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--min-gpu-free-gb", type=float, default=18.0)
    args = parser.parse_args()

    layers = _parse_layers(args.layers)
    payload: dict[str, Any] = {
        "created_at": _now(),
        "model_name": args.model_name,
        "transcoder_set": args.transcoder_set,
        "image_path": args.image_path,
        "question": args.question,
        "layers": layers,
        "hidden_state_offset": args.hidden_state_offset,
        "env_presence": _env_presence(),
        "gpu_before": _gpu_info(),
        "processor": {},
        "model_load": {},
        "transcoder_load": {},
        "forward": {},
        "feature_readout": {},
        "decision": {},
    }

    gpu = payload["gpu_before"]
    if not gpu.get("available") or float(gpu.get("free_gb", 0.0)) < args.min_gpu_free_gb:
        payload["decision"] = {
            "status": "partial",
            "reason": "skipped_forward_because_gpu_free_memory_below_threshold",
            "min_gpu_free_gb": args.min_gpu_free_gb,
        }
        _write_json(Path(args.out_json), payload)
        _log("skipped: insufficient GPU free memory")
        return 0

    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(args.model_name, local_files_only=True)
        payload["processor"] = {
            "status": "ok",
            "processor_class": type(processor).__name__,
            "has_tokenizer": hasattr(processor, "tokenizer"),
        }
    except Exception as exc:  # noqa: BLE001
        payload["processor"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
        }
        payload["decision"] = {"status": "blocked", "reason": "processor_failed"}
        _write_json(Path(args.out_json), payload)
        _log("blocked: processor failed")
        return 0

    _log("loading Qwen2.5-VL model")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()
        model_device = _first_param_device(model)
        payload["model_load"] = {
            "status": "ok",
            "class": type(model).__name__,
            "device": str(model_device),
        }
    except Exception as exc:  # noqa: BLE001
        payload["model_load"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:3000],
        }
        payload["decision"] = {
            "status": "partial",
            "reason": "base_model_load_failed_or_insufficient_memory",
        }
        _write_json(Path(args.out_json), payload)
        _log("partial: model load failed")
        return 0

    _log("loading CLT/transcoder set with lazy decoder/encoder")
    try:
        from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

        transcoders, transcoder_config = load_transcoder_from_hub(
            args.transcoder_set,
            device=model_device,
            dtype=torch.bfloat16,
            lazy_encoder=True,
            lazy_decoder=True,
        )
        payload["transcoder_load"] = {
            "status": "ok",
            "class": type(transcoders).__name__,
            "n_layers": int(getattr(transcoders, "n_layers", len(getattr(transcoders, "transcoders", [])))),
            "d_transcoder": int(getattr(transcoders, "d_transcoder", 0)),
            "feature_input_hook": getattr(transcoders, "feature_input_hook", ""),
            "feature_output_hook": getattr(transcoders, "feature_output_hook", ""),
            "config_model_kind": transcoder_config.get("model_kind", ""),
            "config_hidden_dim": transcoder_config.get("hidden_dim", ""),
            "config_feature_dim": transcoder_config.get("feature_dim", ""),
        }
    except Exception as exc:  # noqa: BLE001
        payload["transcoder_load"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:3000],
        }
        payload["decision"] = {"status": "blocked", "reason": "transcoder_load_failed"}
        _write_json(Path(args.out_json), payload)
        _log("blocked: transcoder load failed")
        return 0

    try:
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
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{args.question}<|im_end|>\n<|im_start|>assistant\n"
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {key: value.to(model_device) if hasattr(value, "to") else value for key, value in inputs.items()}
        payload["forward"]["input_summary"] = _input_summary(inputs)
    except Exception as exc:  # noqa: BLE001
        payload["forward"] = {
            "status": "failed_input_processing",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
        }
        payload["decision"] = {"status": "partial", "reason": "input_processing_failed"}
        _write_json(Path(args.out_json), payload)
        _log("partial: input processing failed")
        return 0

    _log("running forward with output_hidden_states=True")
    try:
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        hidden_states = getattr(outputs, "hidden_states", None)
        payload["forward"].update(
            {
                "status": "ok",
                "logits_shape": list(outputs.logits.shape) if hasattr(outputs, "logits") else "",
                "hidden_states_count": len(hidden_states) if hidden_states is not None else 0,
                "hidden_state_shapes_for_layers": {
                    str(layer): (
                        list(hidden_states[layer + args.hidden_state_offset].shape)
                        if hidden_states is not None
                        and 0 <= layer + args.hidden_state_offset < len(hidden_states)
                        else ""
                    )
                    for layer in layers
                },
            }
        )
    except Exception as exc:  # noqa: BLE001
        payload["forward"].update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:3000],
            }
        )
        payload["decision"] = {"status": "blocked", "reason": "native_qwen_forward_failed"}
        _write_json(Path(args.out_json), payload)
        _log("blocked: forward failed")
        return 0

    readout: dict[str, Any] = {}
    all_ok = True
    try:
        assert hidden_states is not None
        for layer in layers:
            hidden_idx = layer + args.hidden_state_offset
            if hidden_idx < 0 or hidden_idx >= len(hidden_states):
                readout[str(layer)] = {
                    "status": "failed",
                    "reason": f"hidden_state_index_out_of_range_{hidden_idx}",
                }
                all_ok = False
                continue
            hidden = hidden_states[hidden_idx].to(model_device)
            with torch.inference_mode():
                features = transcoders.encode_layer(hidden, layer, apply_activation_function=True)
            readout[str(layer)] = {
                "hidden_state_index": hidden_idx,
                "hidden_shape": list(hidden.shape),
                "summary": _topk_summary(features, args.top_k),
            }
            del features
        payload["feature_readout"] = readout
    except Exception as exc:  # noqa: BLE001
        payload["feature_readout"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
        }
        payload["decision"] = {
            "status": "partial",
            "reason": "hidden_state_available_but_clt_encode_failed",
            "claim_boundary": "Feature readout smoke only; not attribution, intervention, or cross-model replication.",
        }
        _write_json(Path(args.out_json), payload)
        _log("partial: CLT encode failed")
        return 0

    payload["gpu_after"] = _gpu_info()
    if payload["forward"].get("status") == "ok" and all_ok:
        status = "pass_adapter_readout"
        reason = "native_qwen_hidden_states_encoded_by_qwen_clt"
    else:
        status = "partial"
        reason = "some_layers_failed_feature_readout"
    payload["decision"] = {
        "status": status,
        "reason": reason,
        "claim_boundary": "Qwen hidden-state to CLT feature-readout feasibility only; not attribution, intervention, or cross-model mechanism replication.",
    }
    _write_json(Path(args.out_json), payload)
    _log(f"done status={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
