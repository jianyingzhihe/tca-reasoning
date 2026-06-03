#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage2f-hook] {message}", flush=True)


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


def _shape_of(obj: Any) -> Any:
    if hasattr(obj, "shape"):
        return list(obj.shape)
    if isinstance(obj, (tuple, list)):
        return [_shape_of(x) for x in obj[:3]]
    return str(type(obj).__name__)


def _input_summary(inputs: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for key, value in inputs.items():
        if hasattr(value, "shape"):
            summary[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        else:
            summary[key] = str(type(value).__name__)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2F-2 Qwen2.5-VL native hook/forward smoke.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--question", default="What does the sign say?")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--min-gpu-free-gb", type=float, default=18.0)
    parser.add_argument("--layer-index", type=int, default=0)
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "created_at": _now(),
        "model_name": args.model_name,
        "image_path": args.image_path,
        "question": args.question,
        "env_presence": _env_presence(),
        "gpu_before": _gpu_info(),
        "processor": {},
        "model_load": {},
        "forward": {},
        "native_hook": {},
        "replacement_model_compatibility": {},
        "decision": {},
    }

    gpu = payload["gpu_before"]
    if not gpu.get("available") or float(gpu.get("free_gb", 0.0)) < args.min_gpu_free_gb:
        payload["decision"] = {
            "status": "partial",
            "reason": "Skipped forward because GPU free memory is below threshold.",
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
        payload["model_load"] = {
            "status": "ok",
            "class": type(model).__name__,
            "device": str(_first_param_device(model)),
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
    try:
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        device = _first_param_device(model)
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
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

    candidate_modules = [
        name
        for name, _module in model.named_modules()
        if re.search(rf"(?:^|\.)layers\.{args.layer_index}$", name)
    ]
    hook_capture: dict[str, Any] = {}
    hook_handle = None
    if candidate_modules:
        module_name = candidate_modules[0]
        module = dict(model.named_modules())[module_name]

        def _hook(_module, hook_inputs, hook_outputs):
            hook_capture["module_name"] = module_name
            hook_capture["input_shape"] = _shape_of(hook_inputs)
            hook_capture["output_shape"] = _shape_of(hook_outputs)

        hook_handle = module.register_forward_hook(_hook)
    payload["native_hook"] = {
        "candidate_layer_modules": candidate_modules[:10],
        "registered_module": candidate_modules[0] if candidate_modules else "",
    }

    _log("running forward with output_hidden_states=True")
    try:
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        if hook_handle is not None:
            hook_handle.remove()
        hidden_states = getattr(outputs, "hidden_states", None)
        payload["forward"].update(
            {
                "status": "ok",
                "logits_shape": list(outputs.logits.shape) if hasattr(outputs, "logits") else "",
                "hidden_states_count": len(hidden_states) if hidden_states is not None else 0,
                "selected_hidden_shape": (
                    list(hidden_states[min(args.layer_index + 1, len(hidden_states) - 1)].shape)
                    if hidden_states is not None and len(hidden_states)
                    else ""
                ),
            }
        )
        payload["native_hook"].update(hook_capture)
    except Exception as exc:  # noqa: BLE001
        if hook_handle is not None:
            hook_handle.remove()
        payload["forward"].update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:3000],
            }
        )

    payload["gpu_after"] = _gpu_info()
    payload["replacement_model_compatibility"] = {
        "status": "blocked_for_current_main_pipeline",
        "reason": "Current ReplacementModel path is Gemma3ForConditionalGeneration-oriented; this smoke uses native Qwen transformers, not circuit_tracer ReplacementModel.",
        "transformer_lens_hook_names_available": False,
        "native_hidden_states_available": payload["forward"].get("status") == "ok",
    }
    if payload["forward"].get("status") == "ok":
        status = "partial"
        reason = "native_qwen_forward_ok_but_no_replacement_model_adapter"
    else:
        status = "blocked"
        reason = "native_qwen_forward_failed"
    payload["decision"] = {
        "status": status,
        "reason": reason,
        "claim_boundary": "Native Qwen forward smoke only; not attribution, intervention, or cross-model replication.",
    }
    _write_json(Path(args.out_json), payload)
    _log(f"done status={status} forward={payload['forward'].get('status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
