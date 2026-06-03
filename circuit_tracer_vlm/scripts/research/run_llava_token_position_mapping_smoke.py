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
from PIL import Image


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage2f-llava-pos] {message}", flush=True)


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


def _parse_layers(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _shape(value: Any) -> Any:
    if hasattr(value, "shape"):
        return list(value.shape)
    if isinstance(value, (tuple, list)):
        return [_shape(item) for item in value[:3]]
    return str(type(value).__name__)


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


def _find_subsequence(seq: list[int], candidates: list[list[int]]) -> tuple[int | None, int | None]:
    for cand in candidates:
        if not cand:
            continue
        n = len(cand)
        for start in range(0, len(seq) - n + 1):
            if seq[start : start + n] == cand:
                return start, start + n
    return None, None


def _token_rows(
    *,
    input_ids: list[int],
    token_texts: list[str],
    tokenizer,
    question: str,
    image_token_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    question_candidates = []
    for variant in [question, " " + question, "\n" + question]:
        encoded = tokenizer(variant, add_special_tokens=False)
        question_candidates.append(list(encoded["input_ids"]))
    q_start, q_end = _find_subsequence(input_ids, question_candidates)

    assistant_candidates = []
    for variant in ["ASSISTANT:", "\nASSISTANT:", " ASSISTANT:"]:
        encoded = tokenizer(variant, add_special_tokens=False)
        assistant_candidates.append(list(encoded["input_ids"]))
    a_start, a_end = _find_subsequence(input_ids, assistant_candidates)

    image_positions = [idx for idx, token_id in enumerate(input_ids) if token_id == image_token_id]
    if image_positions:
        image_start = min(image_positions)
        image_end = max(image_positions) + 1
    else:
        image_start = image_end = None

    if a_start is None:
        assistant_text_positions = [
            idx for idx, tok in enumerate(token_texts) if "ASS" in tok.upper() or "istant" in tok
        ]
        a_start = min(assistant_text_positions) if assistant_text_positions else None
        a_end = len(input_ids) if a_start is not None else None

    rows: list[dict[str, Any]] = []
    for pos, (token_id, token_text) in enumerate(zip(input_ids, token_texts, strict=False)):
        labels: list[str] = []
        if image_start is not None and image_start <= pos < image_end:
            labels.append("image_token_span")
        if q_start is not None and q_end is not None and q_start <= pos < q_end:
            labels.append("question")
        if image_end is not None and image_end <= pos and (a_start is None or pos < a_start):
            labels.append("post_image_text")
        if a_start is not None and pos >= a_start:
            labels.append("assistant_prefix")
        if pos == len(input_ids) - 1:
            labels.append("last_prompt_token")
        if token_text.startswith("<") and token_text.endswith(">"):
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
        "assistant_span": [a_start, a_end] if a_start is not None and a_end is not None else None,
        "last_prompt_token": len(input_ids) - 1 if input_ids else None,
        "image_token_count": len(image_positions),
    }
    return rows, diagnostics


def _bucket_index(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    buckets = {
        "image_token_span": [],
        "post_image_text": [],
        "question": [],
        "assistant_prefix": [],
        "last_prompt_token": [],
        "special/template": [],
        "other_text_or_template": [],
    }
    for row in rows:
        labels = str(row["bucket"]).split("|")
        pos = int(row["position"])
        for label in labels:
            if label in buckets:
                buckets[label].append(pos)
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
    out: dict[str, torch.Tensor | int | str] = {
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
    return out  # type: ignore[return-value]


def _encode_features(hidden: torch.Tensor, encoder: dict[str, torch.Tensor]) -> torch.Tensor:
    encoded_input = hidden
    if "ln_weight" in encoder and "ln_bias" in encoder:
        encoded_input = F.layer_norm(
            hidden,
            (hidden.shape[-1],),
            encoder["ln_weight"],
            encoder["ln_bias"],
        )
    return F.relu(F.linear(encoded_input, encoder["W_enc"], encoder["b_enc"]))


def _topk_for_positions(features: torch.Tensor, positions: list[int], top_k: int) -> dict[str, Any]:
    if not positions:
        return {"status": "empty_bucket", "position_count": 0}
    pos = [idx for idx in positions if 0 <= idx < features.shape[1]]
    if not pos:
        return {"status": "positions_out_of_range", "position_count": 0}
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
        "max_activation": float(selected.max().float().item()) if selected.numel() else 0.0,
        "mean_positive_activation": (
            float(selected[selected > 0].float().mean().item()) if (selected > 0).any() else 0.0
        ),
        "topk": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2F LLaVA token/position mapping smoke.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--transcoder-repo", default="KokosDev/llava15-7b-clt")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--layers", default="0")
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
        from transformers import AutoConfig, AutoProcessor, LlavaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(args.model_name)
        config = AutoConfig.from_pretrained(args.model_name)
        patch_info = _processor_patch(processor, config)
        tokenizer = processor.tokenizer
        payload["processor"] = {
            "status": "ok",
            "processor_class": type(processor).__name__,
            "tokenizer_class": type(tokenizer).__name__,
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

    try:
        image = Image.open(args.image_path).convert("RGB")
        prompt = f"USER: <image>\n{args.question}\nASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        payload["forward"]["input_summary"] = _input_summary(inputs)
        image_token_id = int(getattr(config, "image_token_index", 32000))
        input_ids = inputs["input_ids"][0].detach().cpu().tolist()
        token_texts = tokenizer.convert_ids_to_tokens(input_ids)
        token_rows, diagnostics = _token_rows(
            input_ids=input_ids,
            token_texts=token_texts,
            tokenizer=tokenizer,
            question=args.question,
            image_token_id=image_token_id,
        )
        buckets = _bucket_index(token_rows)
        payload["position_mapping"] = {
            "status": "ok",
            "sequence_length": len(input_ids),
            "diagnostics": diagnostics,
            "bucket_counts": {key: len(value) for key, value in buckets.items()},
        }
    except Exception as exc:  # noqa: BLE001
        payload["forward"] = {
            "status": "failed_input_processing",
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
        }
        payload["decision"] = {"status": "partial", "reason": "input_processing_failed"}
        _write_json(Path(args.out_json), payload)
        return 0

    _log("running forward and CLT bucket readout")
    try:
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
    except Exception as exc:  # noqa: BLE001
        payload["forward"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
        }
        payload["decision"] = {"status": "partial", "reason": "forward_failed"}
        _write_json(Path(args.out_json), payload)
        return 0

    bucket_rows: list[dict[str, Any]] = []
    readout: dict[str, Any] = {}
    all_layers_ok = True
    assert hidden_states is not None
    for layer in layers:
        layer_key = str(layer)
        hidden_idx = layer + 1
        if hidden_idx >= len(hidden_states):
            readout[layer_key] = {
                "status": "failed",
                "reason": f"hidden_state_index_out_of_range_{hidden_idx}",
            }
            all_layers_ok = False
            continue
        try:
            transcoder_path = _download_transcoder(args.transcoder_repo, layer)
            encoder = _load_encoder(transcoder_path, device=device, dtype=torch.float16)
            hidden = hidden_states[hidden_idx].to(device=device, dtype=torch.float16)
            with torch.inference_mode():
                features = _encode_features(hidden, encoder)
            layer_payload = {
                "status": "ok",
                "hidden_state_index": hidden_idx,
                "hidden_shape": list(hidden.shape),
                "transcoder_path": str(transcoder_path),
                "hidden_dim": encoder["hidden_dim"],
                "feature_dim": encoder["feature_dim"],
                "bucket_readout": {},
            }
            for bucket_name, positions in buckets.items():
                summary = _topk_for_positions(features, positions, args.top_k)
                layer_payload["bucket_readout"][bucket_name] = summary
                for item in summary.get("topk", []):
                    bucket_rows.append(
                        {
                            "layer": layer,
                            "bucket": bucket_name,
                            "position_count": summary.get("position_count", ""),
                            "feature_id": item["feature_id"],
                            "activation": item["activation"],
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
            all_layers_ok = False

    payload["bucket_feature_readout"] = readout
    payload["gpu_after"] = _gpu_info()
    question_ok = payload["position_mapping"].get("diagnostics", {}).get("question_span") is not None
    image_ok = payload["position_mapping"].get("diagnostics", {}).get("image_span") is not None
    last_ok = bool(buckets.get("last_prompt_token"))
    if question_ok and image_ok and last_ok and all_layers_ok:
        status = "pass_position_mapping"
    elif image_ok and last_ok:
        status = "partial_position_mapping_pass"
    else:
        status = "partial_position_mapping"
    payload["decision"] = {
        "status": status,
        "reason": (
            "image_question_and_last_prompt_positions_mapped"
            if status == "pass_position_mapping"
            else "feature_readout_ok_but_some_position_spans_or_layers_unresolved"
        ),
        "claim_boundary": "LLaVA token/position mapping and CLT readout smoke only; not attribution, intervention, or cross-model replication.",
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
            ["layer", "bucket", "position_count", "feature_id", "activation"],
        )
    _write_json(Path(args.out_json), payload)
    _log(f"done status={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
