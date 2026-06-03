#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from run_cross_model_feature_intervention_smoke import (
    _env_presence,
    _first_param_device,
    _gpu_info,
    _prompt,
    _qwen_inputs,
    _rank_and_top,
    _replace_hidden,
)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage4-qwen-intervention] {message}", flush=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _sample_lookup(path: Path) -> dict[str, dict[str, str]]:
    return {row["sample_id"].strip(): row for row in _read_csv(path) if row.get("sample_id", "").strip()}


def _run_lookup(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    out = {}
    for row in _read_csv(path):
        sid = row.get("sample_id", "").strip()
        prompt = row.get("prompt_name", "").strip()
        if sid and prompt:
            out[(sid, prompt)] = row
    return out


def _meta_lookup(path: Path, run_name: str) -> dict[tuple[str, str], dict[str, str]]:
    out = {}
    for row in _read_csv(path):
        if row.get("status") != "ok":
            continue
        sid = row.get("sample_id", "").strip()
        if sid:
            out[(run_name, sid)] = row
    return out


def _image_path(row: dict[str, str], image_root: str) -> Path:
    if image_root:
        return Path(image_root) / Path(row["image_filename"]).name
    for key in ["image_path", "local_image_path"]:
        raw = row.get(key, "")
        if raw:
            return Path(raw)
    return Path(row["image_filename"])


def _select_node_rows(nodes_csv: Path, max_samples: int, top_features_per_sample: int) -> list[dict[str, str]]:
    rows = [
        row
        for row in _read_csv(nodes_csv)
        if row.get("node_type") == "feature" and row.get("feature_id", "") != ""
    ]
    rows.sort(
        key=lambda row: (
            row.get("sample_id", ""),
            row.get("run", ""),
            -float(row.get("path_mass_best") or 0.0),
        )
    )
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("sample_id", ""), row.get("run", ""))].append(row)
    selected = []
    for key in sorted(grouped):
        if max_samples > 0 and len({row.get("sample_id") for row in selected}) >= max_samples and key[0] not in {
            row.get("sample_id") for row in selected
        }:
            continue
        selected.extend(grouped[key][: max(1, top_features_per_sample)])
    return selected


def _decoder_vector(transcoders, layer: int, feature_id: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ids = torch.tensor([int(feature_id)], device=device, dtype=torch.long)
    vectors = transcoders._get_decoder_vectors(layer, ids)
    if vectors.ndim == 3:
        vectors = vectors[:, 0, :]
    return vectors[0].to(device=device, dtype=dtype)


def _score_with_zeroing(
    *,
    model,
    inputs: dict[str, Any],
    tokenizer,
    transcoders,
    module,
    layer: int,
    pos: int,
    feature_id: int,
    target_token_id: int,
    scale: float,
    zeroing_mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with torch.inference_mode():
        original = model(**inputs, output_hidden_states=False, use_cache=False)
    original_score = _rank_and_top(original.logits, tokenizer, [target_token_id])
    vector = _decoder_vector(transcoders, layer, feature_id, device, dtype)

    def _hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if pos < 0 or pos >= hidden.shape[1]:
            return output
        with torch.no_grad():
            acts = transcoders.encode_layer(hidden.to(device), layer, apply_activation_function=True)
            activation = acts[:, pos, int(feature_id)].to(hidden.device, dtype=hidden.dtype)
            patch = activation[:, None] * vector.to(hidden.device, dtype=hidden.dtype)[None, :]
            new_hidden = hidden.clone()
            if zeroing_mode == "subtract":
                new_hidden[:, pos, :] = new_hidden[:, pos, :] - scale * patch
            elif zeroing_mode == "add":
                new_hidden[:, pos, :] = new_hidden[:, pos, :] + scale * patch
            else:
                raise ValueError(f"unknown zeroing_mode: {zeroing_mode}")
        return _replace_hidden(output, new_hidden)

    handle = module.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            intervened = model(**inputs, output_hidden_states=False, use_cache=False)
    finally:
        handle.remove()
    intervened_score = _rank_and_top(intervened.logits, tokenizer, [target_token_id])
    return original_score, intervened_score


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage4 Qwen answer-aligned feature-node zeroing smoke.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--transcoder-ref", required=True)
    parser.add_argument("--sample-manifest", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--nodes-csv", required=True)
    parser.add_argument("--meta-a", required=True)
    parser.add_argument("--meta-b", required=True)
    parser.add_argument("--image-root", default="")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--layer", type=int, default=26)
    parser.add_argument("--answer-prefix", default="The answer is ")
    parser.add_argument("--run-map", default="A:D_visual_only,B:B_direct")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--top-features-per-sample", type=int, default=2)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--zeroing-modes", default="subtract")
    parser.add_argument("--min-gpu-free-gb", type=float, default=12.0)
    args = parser.parse_args()

    run_map = {}
    for item in args.run_map.split(","):
        if ":" not in item:
            continue
        run, prompt = item.split(":", 1)
        run_map[run.strip()] = prompt.strip()
    selected_nodes = _select_node_rows(Path(args.nodes_csv), args.max_samples, args.top_features_per_sample)
    payload: dict[str, Any] = {
        "created_at": _now(),
        "model_name": args.model_name,
        "transcoder_ref": args.transcoder_ref,
        "layer": args.layer,
        "requested_nodes": len(selected_nodes),
        "env_presence": _env_presence(),
        "gpu_before": _gpu_info(),
        "decision": {},
    }
    gpu = payload["gpu_before"]
    if not gpu.get("available") or float(gpu.get("free_gb", 0.0)) < args.min_gpu_free_gb:
        payload["decision"] = {"status": "blocked", "reason": "insufficient_gpu_free_memory"}
        _write_json(Path(args.summary_json), payload)
        return 0

    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    sample_rows = _sample_lookup(Path(args.sample_manifest))
    run_rows = _run_lookup(Path(args.run_manifest))
    meta = {}
    meta.update(_meta_lookup(Path(args.meta_a), "A"))
    meta.update(_meta_lookup(Path(args.meta_b), "B"))

    processor = AutoProcessor.from_pretrained(args.model_name, local_files_only=True)
    tokenizer = processor.tokenizer
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = _first_param_device(model)
    dtype = torch.bfloat16
    transcoders, _config = load_transcoder_from_hub(
        args.transcoder_ref,
        device=device,
        dtype=dtype,
        lazy_encoder=True,
        lazy_decoder=True,
    )
    module = model.language_model.layers[args.layer]

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for node in selected_nodes:
        sample_id = node.get("sample_id", "").strip()
        run_name = node.get("run", "").strip()
        prompt_name = run_map.get(run_name, "")
        sample = sample_rows.get(sample_id)
        meta_row = meta.get((run_name, sample_id))
        if not sample or not prompt_name or not meta_row:
            skipped.append({"sample_id": sample_id, "run": run_name, "reason": "missing_sample_prompt_or_meta"})
            continue
        run_row = run_rows.get((sample_id, prompt_name), {})
        question_text = run_row.get("question_text") or sample.get("question_text", "")
        image_path = _image_path(sample, args.image_root)
        try:
            target_token_id = int(meta_row["target_token_id"])
            pos = int(node["pos"])
            feature_id = int(node["feature_id"])
            image = Image.open(image_path).convert("RGB")
            question = _prompt(question_text, prompt_name)
            inputs = _qwen_inputs(processor, image, str(image_path), question, args.answer_prefix, device)
            for zeroing_mode in [part.strip() for part in args.zeroing_modes.split(",") if part.strip()]:
                original_score, intervened_score = _score_with_zeroing(
                    model=model,
                    inputs=inputs,
                    tokenizer=tokenizer,
                    transcoders=transcoders,
                    module=module,
                    layer=args.layer,
                    pos=pos,
                    feature_id=feature_id,
                    target_token_id=target_token_id,
                    scale=args.scale,
                    zeroing_mode=zeroing_mode,
                    device=device,
                    dtype=dtype,
                )
                delta_logit = float(intervened_score["target_logit"]) - float(original_score["target_logit"])
                delta_prob = float(intervened_score["target_prob"]) - float(original_score["target_prob"])
                delta_rank = int(intervened_score["target_rank"]) - int(original_score["target_rank"])
                rows.append(
                    {
                        "sample_id": sample_id,
                        "run": run_name,
                        "prompt_name": prompt_name,
                        "zeroing_mode": zeroing_mode,
                        "node_id": node.get("node_id", ""),
                        "layer": args.layer,
                        "pos": pos,
                        "feature_id": feature_id,
                        "depth_from_target": node.get("depth_from_target", ""),
                        "path_mass_best": node.get("path_mass_best", ""),
                        "target_token_id": target_token_id,
                        "target_token": original_score["target_token"],
                        "original_target_logit": original_score["target_logit"],
                        "intervened_target_logit": intervened_score["target_logit"],
                        "delta_target_logit": delta_logit,
                        "original_target_prob": original_score["target_prob"],
                        "intervened_target_prob": intervened_score["target_prob"],
                        "delta_target_prob": delta_prob,
                        "original_target_rank": original_score["target_rank"],
                        "intervened_target_rank": intervened_score["target_rank"],
                        "delta_target_rank": delta_rank,
                        "original_top1_token": original_score["top1_token"],
                        "intervened_top1_token": intervened_score["top1_token"],
                        "status": "ok",
                        "error_message": "",
                    }
                )
            if len(rows) % 10 == 0:
                _log(f"intervention rows complete: {len(rows)}/{len(selected_nodes)}")
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            skipped.append({"sample_id": sample_id, "run": run_name, "reason": repr(exc)})

    fieldnames = [
        "sample_id",
        "run",
        "prompt_name",
        "zeroing_mode",
        "node_id",
        "layer",
        "pos",
        "feature_id",
        "depth_from_target",
        "path_mass_best",
        "target_token_id",
        "target_token",
        "original_target_logit",
        "intervened_target_logit",
        "delta_target_logit",
        "original_target_prob",
        "intervened_target_prob",
        "delta_target_prob",
        "original_target_rank",
        "intervened_target_rank",
        "delta_target_rank",
        "original_top1_token",
        "intervened_top1_token",
        "status",
        "error_message",
    ]
    _write_csv(Path(args.out_csv), rows, fieldnames)
    negative = sum(1 for row in rows if float(row["delta_target_logit"]) < 0)
    payload["gpu_after"] = _gpu_info()
    payload["decision"] = {
        "status": "ok" if rows else "blocked_no_intervention_rows",
        "intervention_rows": len(rows),
        "requested_nodes": len(selected_nodes),
        "frac_negative_delta_target_logit": negative / len(rows) if rows else 0.0,
        "skipped": skipped[:100],
        "claim_boundary": "Qwen feature-node zeroing smoke; source-control verdict requires analyzer and controls.",
    }
    _write_json(Path(args.summary_json), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
