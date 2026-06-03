#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage2f-llava-readout] {message}", flush=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
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


def _shape(value: Any) -> Any:
    if hasattr(value, "shape"):
        return list(value.shape)
    if isinstance(value, (tuple, list)):
        return [_shape(x) for x in value]
    return str(type(value).__name__)


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


def _download_transcoder(repo_id: str, layer: int) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=f"transcoder_L{layer}.pt",
            local_files_only=False,
        )
    )


def _load_encoder(path: Path, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
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
    out = {
        "W_enc": weight.to(device=device, dtype=dtype),
        "b_enc": bias.to(device=device, dtype=dtype),
        "raw_keys": sorted(str(k) for k in obj.keys()),
        "state_dict_keys": sorted(str(k) for k in state_dict.keys()),
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


def _bucket_positions(input_ids: torch.Tensor, image_token_id: int) -> dict[str, list[int]]:
    ids = input_ids.detach().cpu().tolist()
    image_positions = [idx for idx, token_id in enumerate(ids) if token_id == image_token_id]
    out = {
        "image_token_span": image_positions,
        "last_prompt_token": [len(ids) - 1] if ids else [],
    }
    if image_positions:
        last_image = max(image_positions)
        question_positions = list(range(last_image + 1, max(last_image + 1, len(ids) - 1)))
        out["post_image_text"] = question_positions
    else:
        out["post_image_text"] = list(range(max(0, len(ids) - 1)))
    return out


def _topk_for_bucket(features: torch.Tensor, positions: list[int], top_k: int) -> dict[str, Any]:
    if not positions:
        return {"status": "empty_bucket"}
    pos = [idx for idx in positions if 0 <= idx < features.shape[1]]
    if not pos:
        return {"status": "positions_out_of_range"}
    selected = features[:, pos, :].detach()
    max_per_feature = selected.amax(dim=(0, 1))
    k = min(top_k, max_per_feature.numel())
    vals, ids = torch.topk(max_per_feature, k=k)
    rows = [
        {"feature_id": int(feature_id), "activation": float(value)}
        for feature_id, value in zip(ids.cpu().tolist(), vals.float().cpu().tolist(), strict=False)
    ]
    return {
        "status": "ok",
        "position_count": len(pos),
        "feature_shape": list(selected.shape),
        "active_positive_count": int((selected > 0).sum().item()),
        "max_activation": float(selected.max().float().item()),
        "mean_positive_activation": (
            float(selected[selected > 0].float().mean().item()) if (selected > 0).any() else 0.0
        ),
        "topk": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2F LLaVA hidden-state to public CLT readout smoke.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--transcoder-repo", default="KokosDev/llava15-7b-clt")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--question", default="What does the sign say?")
    parser.add_argument("--layers", default="0")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--min-gpu-free-gb", type=float, default=18.0)
    args = parser.parse_args()

    layers = _parse_layers(args.layers)
    payload: dict[str, Any] = {
        "created_at": _now(),
        "model_name": args.model_name,
        "transcoder_repo": args.transcoder_repo,
        "image_path": args.image_path,
        "question": args.question,
        "layers": layers,
        "env_presence": _env_presence(),
        "gpu_before": _gpu_info(),
        "processor": {},
        "config": {},
        "model_load": {},
        "forward": {},
        "token_buckets": {},
        "transcoder_readout": {},
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
        from transformers import AutoConfig, AutoProcessor, LlavaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(args.model_name)
        config = AutoConfig.from_pretrained(args.model_name)
        patch_info = _processor_patch(processor, config)
        payload["processor"] = {
            "status": "ok",
            "processor_class": type(processor).__name__,
            **patch_info,
        }
        text_config = getattr(config, "text_config", None)
        payload["config"] = {
            "status": "ok",
            "model_type": getattr(config, "model_type", ""),
            "text_hidden_size": getattr(text_config, "hidden_size", "") if text_config else "",
            "text_num_hidden_layers": (
                getattr(text_config, "num_hidden_layers", "") if text_config else ""
            ),
            "image_token_index": getattr(config, "image_token_index", ""),
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
        payload["model_load"] = {
            "status": "ok",
            "class": type(model).__name__,
            "device": str(device),
        }
    except Exception as exc:  # noqa: BLE001
        payload["model_load"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:3000],
        }
        payload["decision"] = {"status": "partial", "reason": "model_load_failed"}
        _write_json(Path(args.out_json), payload)
        return 0

    try:
        image = Image.open(args.image_path).convert("RGB")
        prompt = f"USER: <image>\n{args.question}\nASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        image_token_id = int(getattr(config, "image_token_index", 32000))
        buckets = _bucket_positions(inputs["input_ids"][0], image_token_id)
        payload["token_buckets"] = {key: len(value) for key, value in buckets.items()}
    except Exception as exc:  # noqa: BLE001
        payload["forward"] = {
            "status": "failed_input_processing",
            "error_type": type(exc).__name__,
            "error": str(exc)[:3000],
        }
        payload["decision"] = {"status": "partial", "reason": "input_processing_failed"}
        _write_json(Path(args.out_json), payload)
        return 0

    _log("running forward")
    try:
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        hidden_states = getattr(outputs, "hidden_states", None)
        payload["forward"] = {
            "status": "ok",
            "input_shapes": {key: _shape(value) for key, value in inputs.items()},
            "logits_shape": list(outputs.logits.shape),
            "hidden_states_count": len(hidden_states) if hidden_states is not None else 0,
        }
    except Exception as exc:  # noqa: BLE001
        payload["forward"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
        }
        payload["decision"] = {"status": "partial", "reason": "forward_failed"}
        _write_json(Path(args.out_json), payload)
        return 0

    rows: list[dict[str, Any]] = []
    readout: dict[str, Any] = {}
    all_ok = True
    assert hidden_states is not None
    for layer in layers:
        layer_key = str(layer)
        readout[layer_key] = {}
        hidden_idx = layer + 1
        if hidden_idx >= len(hidden_states):
            readout[layer_key] = {
                "status": "failed",
                "reason": f"hidden_state_index_out_of_range_{hidden_idx}",
            }
            all_ok = False
            continue
        try:
            transcoder_path = _download_transcoder(args.transcoder_repo, layer)
            encoder = _load_encoder(transcoder_path, device=device, dtype=torch.float16)
            hidden = hidden_states[hidden_idx].to(device=device, dtype=torch.float16)
            with torch.inference_mode():
                encoded_input = hidden
                if "ln_weight" in encoder and "ln_bias" in encoder:
                    encoded_input = F.layer_norm(
                        hidden,
                        (hidden.shape[-1],),
                        encoder["ln_weight"],
                        encoder["ln_bias"],
                    )
                features = F.relu(F.linear(encoded_input, encoder["W_enc"], encoder["b_enc"]))
            layer_payload = {
                "status": "ok",
                "hidden_state_index": hidden_idx,
                "hidden_shape": list(hidden.shape),
                "transcoder_path": str(transcoder_path),
                "hidden_dim": encoder["hidden_dim"],
                "feature_dim": encoder["feature_dim"],
                "weight_dtype": encoder["weight_dtype"],
                "bias_dtype": encoder["bias_dtype"],
                "bucket_readout": {},
            }
            for bucket, positions in buckets.items():
                bucket_summary = _topk_for_bucket(features, positions, args.top_k)
                layer_payload["bucket_readout"][bucket] = bucket_summary
                for item in bucket_summary.get("topk", []):
                    rows.append(
                        {
                            "layer": layer,
                            "bucket": bucket,
                            "feature_id": item["feature_id"],
                            "activation": item["activation"],
                            "position_count": bucket_summary.get("position_count", ""),
                            "max_activation": bucket_summary.get("max_activation", ""),
                            "active_positive_count": bucket_summary.get(
                                "active_positive_count", ""
                            ),
                        }
                    )
            readout[layer_key] = layer_payload
            del features, hidden, encoder
        except Exception as exc:  # noqa: BLE001
            readout[layer_key] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:4000],
            }
            all_ok = False

    payload["transcoder_readout"] = readout
    payload["gpu_after"] = _gpu_info()
    payload["decision"] = {
        "status": "pass_feature_readout" if all_ok else "partial_feature_readout",
        "reason": (
            "llava_hidden_states_encoded_by_public_clt_assets"
            if all_ok
            else "some_layers_failed_feature_readout"
        ),
        "claim_boundary": "LLaVA CLT feature readout smoke only; not attribution, intervention, or cross-model mechanism replication.",
    }
    _write_json(Path(args.out_json), payload)
    _write_csv(
        Path(args.out_csv),
        rows,
        [
            "layer",
            "bucket",
            "feature_id",
            "activation",
            "position_count",
            "max_activation",
            "active_positive_count",
        ],
    )
    _log(f"done status={payload['decision']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
