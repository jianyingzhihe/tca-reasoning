#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage2f-llava-hook] {message}", flush=True)


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


def _disk_info(path: str) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": path,
        "total_gb": round(usage.total / (1024**3), 3),
        "used_gb": round(usage.used / (1024**3), 3),
        "free_gb": round(usage.free / (1024**3), 3),
    }


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
                "device": str(value.device) if hasattr(value, "device") else "",
            }
        else:
            summary[key] = str(type(value).__name__)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2F LLaVA base loader and hook-forward smoke.")
    parser.add_argument("--model-name", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--question", default="What does the sign say?")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--disk-path", default="/root/autodl-tmp")
    parser.add_argument("--min-free-gb", type=float, default=40.0)
    parser.add_argument("--min-gpu-free-gb", type=float, default=18.0)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--skip-base-forward", action="store_true")
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "created_at": _now(),
        "model_name": args.model_name,
        "image_path": args.image_path,
        "question": args.question,
        "env_presence": _env_presence(),
        "disk_before": _disk_info(args.disk_path),
        "gpu_before": _gpu_info(),
        "processor": {},
        "config": {},
        "model_load": {},
        "forward": {},
        "native_hook": {},
        "decision": {},
    }

    try:
        from transformers import AutoConfig, AutoProcessor

        processor = AutoProcessor.from_pretrained(args.model_name)
        config = AutoConfig.from_pretrained(args.model_name)
        text_config = getattr(config, "text_config", None)
        vision_config = getattr(config, "vision_config", None)
        # Recent Transformers versions expand LLaVA image tokens in the processor.
        # Some converted checkpoints do not persist these attributes, so mirror
        # the values from config before calling the processor on images.
        processor_patch_size = getattr(processor, "patch_size", None)
        if processor_patch_size is None and vision_config is not None:
            processor_patch_size = getattr(vision_config, "patch_size", None)
            if processor_patch_size is not None:
                setattr(processor, "patch_size", processor_patch_size)
        processor_vision_feature_select_strategy = getattr(
            processor, "vision_feature_select_strategy", None
        )
        if processor_vision_feature_select_strategy is None:
            processor_vision_feature_select_strategy = getattr(
                config, "vision_feature_select_strategy", "default"
            )
            setattr(
                processor,
                "vision_feature_select_strategy",
                processor_vision_feature_select_strategy,
            )
        processor_num_additional_image_tokens = getattr(
            processor, "num_additional_image_tokens", None
        )
        if processor_num_additional_image_tokens in (None, 0):
            processor_num_additional_image_tokens = 1
            setattr(processor, "num_additional_image_tokens", processor_num_additional_image_tokens)
        payload["processor"] = {
            "status": "ok",
            "processor_class": type(processor).__name__,
            "has_tokenizer": hasattr(processor, "tokenizer"),
            "patch_size": processor_patch_size,
            "vision_feature_select_strategy": processor_vision_feature_select_strategy,
            "num_additional_image_tokens": processor_num_additional_image_tokens,
        }
        payload["config"] = {
            "status": "ok",
            "model_type": getattr(config, "model_type", ""),
            "architectures": getattr(config, "architectures", []),
            "hidden_size": getattr(config, "hidden_size", ""),
            "text_hidden_size": getattr(text_config, "hidden_size", "") if text_config else "",
            "text_num_hidden_layers": (
                getattr(text_config, "num_hidden_layers", "") if text_config else ""
            ),
            "vision_config_type": type(vision_config).__name__,
            "vision_patch_size": getattr(vision_config, "patch_size", "") if vision_config else "",
            "vision_feature_select_strategy": getattr(
                config, "vision_feature_select_strategy", ""
            ),
        }
    except Exception as exc:  # noqa: BLE001
        payload["processor"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:3000],
        }
        payload["decision"] = {"status": "blocked", "reason": "processor_or_config_failed"}
        _write_json(Path(args.out_json), payload)
        _log("blocked: processor/config failed")
        return 0

    if args.skip_base_forward:
        payload["decision"] = {
            "status": "partial_base_asset_pass",
            "reason": "processor_config_ok_but_base_forward_skipped_after_slow_hf_xet_download",
            "claim_boundary": "LLaVA processor/config smoke only; base weights were not fully loaded and no hook-forward was run.",
        }
        payload["disk_after"] = _disk_info(args.disk_path)
        payload["gpu_after"] = _gpu_info()
        _write_json(Path(args.out_json), payload)
        _log("partial: base forward skipped by request")
        return 0

    disk = payload["disk_before"]
    gpu = payload["gpu_before"]
    if float(disk.get("free_gb", 0.0)) < args.min_free_gb:
        payload["decision"] = {
            "status": "partial_base_asset_pass",
            "reason": "processor_config_ok_but_disk_below_threshold_for_base_weights",
            "claim_boundary": "LLaVA processor/config smoke only; no base forward.",
        }
        _write_json(Path(args.out_json), payload)
        _log("partial: disk below threshold")
        return 0
    if not gpu.get("available") or float(gpu.get("free_gb", 0.0)) < args.min_gpu_free_gb:
        payload["decision"] = {
            "status": "partial_base_asset_pass",
            "reason": "processor_config_ok_but_gpu_free_memory_below_threshold",
            "claim_boundary": "LLaVA processor/config smoke only; no base forward.",
        }
        _write_json(Path(args.out_json), payload)
        _log("partial: gpu below threshold")
        return 0

    _log("loading LLaVA base model")
    try:
        from transformers import LlavaForConditionalGeneration

        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()
        device = _first_param_device(model)
        payload["model_load"] = {
            "status": "ok",
            "class": type(model).__name__,
            "device": str(device),
        }
    except Exception as exc:  # noqa: BLE001
        payload["model_load"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
        }
        payload["decision"] = {
            "status": "partial_base_asset_pass",
            "reason": "processor_config_ok_but_base_model_load_failed",
            "claim_boundary": "LLaVA base load failed; not hook-forward or replication.",
        }
        _write_json(Path(args.out_json), payload)
        _log("partial: model load failed")
        return 0

    image = Image.open(args.image_path).convert("RGB")
    prompt = f"USER: <image>\n{args.question}\nASSISTANT:"
    try:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        payload["forward"]["input_summary"] = _input_summary(inputs)
    except Exception as exc:  # noqa: BLE001
        payload["forward"] = {
            "status": "failed_input_processing",
            "error_type": type(exc).__name__,
            "error": str(exc)[:3000],
        }
        payload["decision"] = {"status": "partial", "reason": "input_processing_failed"}
        _write_json(Path(args.out_json), payload)
        return 0

    candidate_modules = [
        name
        for name, _module in model.named_modules()
        if re.search(rf"(?:^|\.)(?:layers|model\.layers)\.{args.layer_index}$", name)
    ]
    # Prefer language-model layers over vision tower layers when both are present.
    candidate_modules = sorted(
        candidate_modules,
        key=lambda n: (0 if "language" in n or "model.layers" in n else 1, len(n)),
    )
    hook_capture: dict[str, Any] = {}
    hook_handle = None
    module_name = candidate_modules[0] if candidate_modules else ""
    if module_name:
        module = dict(model.named_modules())[module_name]

        def _hook(_module, hook_inputs, hook_outputs):
            hook_capture["module_name"] = module_name
            hook_capture["input_shape"] = _shape_of(hook_inputs)
            hook_capture["output_shape"] = _shape_of(hook_outputs)

        hook_handle = module.register_forward_hook(_hook)
    payload["native_hook"] = {
        "candidate_layer_modules": candidate_modules[:20],
        "registered_module": module_name,
    }

    _log("running LLaVA forward")
    try:
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        if hook_handle is not None:
            hook_handle.remove()
        hidden_states = getattr(outputs, "hidden_states", None)
        selected_hidden_shape = ""
        if hidden_states is not None and len(hidden_states):
            idx = min(args.layer_index + 1, len(hidden_states) - 1)
            selected_hidden_shape = list(hidden_states[idx].shape)
        payload["forward"].update(
            {
                "status": "ok",
                "logits_shape": list(outputs.logits.shape) if hasattr(outputs, "logits") else "",
                "hidden_states_count": len(hidden_states) if hidden_states is not None else 0,
                "selected_hidden_shape": selected_hidden_shape,
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
                "error": str(exc)[:4000],
            }
        )

    payload["gpu_after"] = _gpu_info()
    payload["disk_after"] = _disk_info(args.disk_path)
    if payload["forward"].get("status") == "ok":
        status = "pass_hook_forward"
        reason = "llava_native_forward_and_hook_ok"
    else:
        status = "partial_base_asset_pass"
        reason = "base_loaded_but_forward_or_hook_failed"
    payload["decision"] = {
        "status": status,
        "reason": reason,
        "claim_boundary": "LLaVA base/hook-forward smoke only; not feature readout, intervention, or cross-model replication.",
    }
    _write_json(Path(args.out_json), payload)
    _log(f"done status={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
