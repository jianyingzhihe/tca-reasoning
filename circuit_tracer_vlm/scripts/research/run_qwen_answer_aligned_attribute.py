#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image

from run_cross_model_feature_intervention_smoke import (
    _env_presence,
    _first_param_device,
    _gpu_info,
    _prompt,
    _qwen_bucket_positions,
    _qwen_inputs,
    _rank_and_top,
    _target_candidates,
)
from run_cross_model_hidden_position_patch_smoke import _answer_adjacent_positions


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage4-qwen-attribute] {message}", flush=True)


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sample_lookup(sample_manifest: Path) -> dict[str, dict[str, str]]:
    return {row["sample_id"].strip(): row for row in _read_csv(sample_manifest) if row.get("sample_id", "").strip()}


def _run_rows(sample_manifest: Path, run_manifest: Path, prompt_name: str) -> list[dict[str, str]]:
    samples = _sample_lookup(sample_manifest)
    rows = []
    for row in _read_csv(run_manifest):
        if row.get("prompt_name", "").strip() != prompt_name:
            continue
        sid = row.get("sample_id", "").strip()
        if not sid or sid not in samples:
            continue
        merged = dict(samples[sid])
        merged.update({k: v for k, v in row.items() if v != ""})
        rows.append(merged)
    return rows


def _image_path(row: dict[str, str], image_root: str) -> Path:
    if image_root:
        return Path(image_root) / Path(row["image_filename"]).name
    for key in ["image_path", "local_image_path"]:
        raw = row.get(key, "")
        if raw:
            return Path(raw)
    return Path(row["image_filename"])


def _decoder_vectors(transcoders, layer: int, feature_ids: list[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ids = torch.tensor(feature_ids, device=device, dtype=torch.long)
    vectors = transcoders._get_decoder_vectors(layer, ids)
    if vectors.ndim == 3:
        vectors = vectors[:, 0, :]
    return vectors.to(device=device, dtype=dtype)


def _select_feature_position_nodes(
    *,
    features: torch.Tensor,
    transcoders,
    layer: int,
    target_direction: torch.Tensor,
    max_feature_nodes: int,
    candidate_pool_size: int,
    node_sign: str,
    allowed_positions: list[int] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    # features: [1, seq, n_features]
    acts = features[0].detach()
    if acts.numel() == 0:
        return {"status": "empty_features"}
    score_acts = acts.float().abs()
    if allowed_positions is not None:
        allowed = sorted({int(pos) for pos in allowed_positions if 0 <= int(pos) < int(acts.shape[0])})
        if not allowed:
            return {"status": "empty_allowed_positions"}
        mask = torch.zeros(int(acts.shape[0]), device=score_acts.device, dtype=torch.bool)
        mask[torch.tensor(allowed, device=score_acts.device, dtype=torch.long)] = True
        score_acts = score_acts.masked_fill(~mask[:, None], 0)
    flat_abs = score_acts.flatten()
    pool_n = min(int(candidate_pool_size), int(flat_abs.numel()))
    if pool_n <= 0:
        return {"status": "empty_candidate_pool"}
    vals, flat_idx = torch.topk(flat_abs, k=pool_n)
    if float(vals.max().item()) <= 0:
        return {"status": "no_active_features"}

    n_feat = int(acts.shape[1])
    positions = torch.div(flat_idx, n_feat, rounding_mode="floor").detach().cpu().tolist()
    feature_ids = (flat_idx % n_feat).detach().cpu().tolist()
    unique_feature_ids = sorted({int(fid) for fid in feature_ids})
    vectors = _decoder_vectors(transcoders, layer, unique_feature_ids, device, dtype)
    target = target_direction.to(device=vectors.device, dtype=vectors.dtype)
    contrib_by_feature = torch.mv(vectors.float(), target.float()).detach().cpu()
    contrib_lookup = {fid: float(contrib_by_feature[i].item()) for i, fid in enumerate(unique_feature_ids)}

    scored: list[dict[str, Any]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for pos, feat_id in zip(positions, feature_ids, strict=False):
        pos_i = int(pos)
        feat_i = int(feat_id)
        key = (pos_i, feat_i)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        activation = float(acts[pos_i, feat_i].float().item())
        target_contribution = contrib_lookup.get(feat_i, 0.0)
        direct_effect = activation * target_contribution
        scored.append(
            {
                "layer": int(layer),
                "pos": pos_i,
                "feature_id": feat_i,
                "activation": activation,
                "target_contribution": target_contribution,
                "direct_effect": direct_effect,
                "abs_direct_effect": abs(direct_effect),
            }
        )
    if node_sign == "support":
        signed = [item for item in scored if float(item["direct_effect"]) > 0]
        if signed:
            scored = signed
    elif node_sign != "both":
        raise ValueError(f"unsupported node_sign: {node_sign}")
    scored.sort(key=lambda item: (item["abs_direct_effect"], abs(item["activation"])), reverse=True)
    selected = scored[: max(1, int(max_feature_nodes))]
    if not selected:
        return {"status": "no_selected_features"}
    return {"status": "ok", "nodes": selected, "unique_feature_count": len(unique_feature_ids), "candidate_pair_count": len(scored)}


def _build_graph_dict(
    *,
    input_string: str,
    input_ids: torch.Tensor,
    selected_nodes: list[dict[str, Any]],
    n_layers: int,
    target_token_id: int,
    target_prob: float,
    compact_positions: bool,
) -> dict[str, Any]:
    full_input_tokens = input_ids.detach().cpu().long()
    if compact_positions:
        original_positions = sorted({int(node["pos"]) for node in selected_nodes})
        if not original_positions:
            original_positions = list(range(int(full_input_tokens.numel())))
        position_to_compact = {pos: idx for idx, pos in enumerate(original_positions)}
        input_tokens = torch.tensor([int(full_input_tokens[pos].item()) for pos in original_positions], dtype=torch.long)
        position_map = torch.tensor(original_positions, dtype=torch.long)
    else:
        position_to_compact = {pos: pos for pos in range(int(full_input_tokens.numel()))}
        input_tokens = full_input_tokens
        position_map = None
    n_pos = int(input_tokens.numel())
    n_features = len(selected_nodes)
    n_errors = int(n_layers) * n_pos
    n_logits = 1
    token_start = n_features + n_errors
    logit_start = token_start + n_pos
    total_nodes = logit_start + n_logits
    adjacency = torch.zeros((total_nodes, total_nodes), dtype=torch.float32)

    active_features = []
    activation_values = []
    for node_idx, node in enumerate(selected_nodes):
        layer = int(node["layer"])
        original_pos = int(node["pos"])
        pos = int(position_to_compact[original_pos])
        feat = int(node["feature_id"])
        active_features.append([layer, pos, feat])
        activation = float(node["activation"])
        activation_values.append(activation)
        if 0 <= pos < n_pos:
            adjacency[node_idx, token_start + pos] = activation
        adjacency[logit_start, node_idx] = float(node["direct_effect"])

    graph = {
        "input_string": input_string,
        "input_tokens": input_tokens,
        "active_features": torch.tensor(active_features, dtype=torch.long),
        "selected_features": torch.arange(n_features, dtype=torch.long),
        "activation_values": torch.tensor(activation_values, dtype=torch.float32),
        "logit_tokens": torch.tensor([int(target_token_id)], dtype=torch.long),
        "logit_probabilities": torch.tensor([float(target_prob)], dtype=torch.float32),
        "adjacency_matrix": adjacency,
        "cfg": SimpleNamespace(n_layers=int(n_layers)),
        "scan": "stage4_qwen_answer_aligned_plt",
    }
    if position_map is not None:
        graph["position_map"] = position_map
    return graph


def _hook_hidden(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage4 Qwen answer-aligned PLT graph adapter.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--transcoder-ref", required=True)
    parser.add_argument("--sample-manifest", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--prompt-name", required=True, choices=["B_direct", "D_visual_only"])
    parser.add_argument("--image-root", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--meta-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--layer", type=int, default=26)
    parser.add_argument("--answer-prefix", default="The answer is ")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--max-feature-nodes", type=int, default=96)
    parser.add_argument("--candidate-pool-size", type=int, default=4096)
    parser.add_argument("--node-sign", choices=["support", "both"], default="support")
    parser.add_argument(
        "--position-filter",
        choices=["visual_answer", "visual_only", "answer_adjacent_only", "all"],
        default="visual_answer",
    )
    parser.add_argument("--compact-positions", action="store_true", default=True)
    parser.add_argument("--max-n-pos", type=int, default=512)
    parser.add_argument("--min-gpu-free-gb", type=float, default=12.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _run_rows(Path(args.sample_manifest), Path(args.run_manifest), args.prompt_name)
    if args.max_runs > 0:
        rows = rows[: args.max_runs]

    payload: dict[str, Any] = {
        "created_at": _now(),
        "model_name": args.model_name,
        "transcoder_ref": args.transcoder_ref,
        "prompt_name": args.prompt_name,
        "layer": args.layer,
        "requested_rows": len(rows),
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
    transcoders, config = load_transcoder_from_hub(
        args.transcoder_ref,
        device=device,
        dtype=dtype,
        lazy_encoder=True,
        lazy_decoder=True,
    )
    n_layers = int(len(model.language_model.layers))
    module = model.language_model.layers[args.layer]
    output_weight = model.get_output_embeddings().weight

    meta_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        sample_id = row.get("sample_id", "").strip()
        question_text = row.get("question_text", "").strip()
        answer_text = row.get("answer_text", "").strip()
        image_path = _image_path(row, args.image_root)
        meta: dict[str, Any] = {
            "sample_id": sample_id,
            "prompt_name": args.prompt_name,
            "question": question_text,
            "image_path": str(image_path),
            "answer_text": answer_text,
            "assistant_prefix": args.answer_prefix,
            "graph_output_path": str(out_dir / f"{sample_id}.pt"),
            "status": "error",
            "error_message": "",
        }
        try:
            if not image_path.exists():
                raise FileNotFoundError(f"image not found: {image_path}")
            image = Image.open(image_path).convert("RGB")
            question = _prompt(question_text, args.prompt_name)
            inputs = _qwen_inputs(processor, image, str(image_path), question, args.answer_prefix, device)
            input_ids = inputs["input_ids"][0].detach().cpu()
            if int(input_ids.numel()) > args.max_n_pos:
                raise ValueError(f"n_pos {int(input_ids.numel())} exceeds --max-n-pos {args.max_n_pos}")
            target_candidates = _target_candidates(tokenizer, answer_text)
            target_ids = [int(item["token_id"]) for item in target_candidates]
            if not target_ids:
                raise ValueError("no target token candidates")
            captured: dict[str, torch.Tensor] = {}

            def _capture_layer_output(_module, _inputs, output):
                captured["hidden"] = _hook_hidden(output).detach()
                return output

            handle = module.register_forward_hook(_capture_layer_output)
            try:
                with torch.inference_mode():
                    outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            finally:
                handle.remove()
            hidden = captured.get("hidden")
            hidden_capture_source = f"language_model.layers[{args.layer}]_forward_hook_output"
            if hidden is None:
                # HF hidden_states often include embeddings at index 0; fall back conservatively
                # only if the hook did not fire, and record that path in metadata.
                fallback_idx = min(args.layer + 1, len(outputs.hidden_states) - 1)
                hidden = outputs.hidden_states[fallback_idx].detach()
                hidden_capture_source = f"hidden_states[{fallback_idx}]_fallback"
            clean_score = _rank_and_top(outputs.logits, tokenizer, target_ids)
            target_token_id = int(clean_score["target_token_id"])
            token_texts = tokenizer.convert_ids_to_tokens(input_ids.detach().cpu().tolist())
            visual_positions = _qwen_bucket_positions(input_ids.detach().cpu().tolist(), token_texts).get("image_marker_or_span", [])
            answer_positions = _answer_adjacent_positions(int(input_ids.numel()), set(visual_positions), 4)
            allowed_positions = None
            if args.position_filter == "visual_answer":
                allowed_positions = sorted(set(int(pos) for pos in visual_positions + answer_positions))
            elif args.position_filter == "visual_only":
                allowed_positions = sorted(set(int(pos) for pos in visual_positions))
            elif args.position_filter == "answer_adjacent_only":
                allowed_positions = sorted(set(int(pos) for pos in answer_positions))
            hidden = hidden.to(device=device, dtype=dtype).detach()
            features = transcoders.encode_layer(hidden.to(device), args.layer, apply_activation_function=True).detach()
            target_direction = output_weight[target_token_id].detach().to(device=device, dtype=dtype)
            selection = _select_feature_position_nodes(
                features=features,
                transcoders=transcoders,
                layer=args.layer,
                target_direction=target_direction,
                max_feature_nodes=args.max_feature_nodes,
                candidate_pool_size=args.candidate_pool_size,
                node_sign=args.node_sign,
                allowed_positions=allowed_positions,
                device=device,
                dtype=dtype,
            )
            if selection["status"] != "ok":
                raise RuntimeError(f"feature selection failed: {selection['status']}")
            graph = _build_graph_dict(
                input_string=question + args.answer_prefix,
                input_ids=input_ids,
                selected_nodes=selection["nodes"],
                n_layers=n_layers,
                target_token_id=target_token_id,
                target_prob=float(clean_score["target_prob"]),
                compact_positions=args.compact_positions,
            )
            graph_path = out_dir / f"{sample_id}.pt"
            torch.save(graph, graph_path)
            meta.update(
                {
                    "target_token_id": target_token_id,
                    "target_token_text": clean_score["target_token"],
                    "target_rank": clean_score["target_rank"],
                    "target_logit": clean_score["target_logit"],
                    "target_prob": clean_score["target_prob"],
                    "top1_token": clean_score["top1_token"],
                    "top1_token_id": clean_score["top1_token_id"],
                    "n_pos": int(input_ids.numel()),
                    "graph_n_pos": int(graph["input_tokens"].numel()),
                    "visual_position_count": len(visual_positions),
                    "answer_adjacent_position_count": len(answer_positions),
                    "position_filter": args.position_filter,
                    "n_layers": n_layers,
                    "hidden_capture_source": hidden_capture_source,
                    "hidden_states_tuple_len": len(outputs.hidden_states) if outputs.hidden_states is not None else "",
                    "selected_feature_nodes": len(selection["nodes"]),
                    "candidate_pair_count": selection.get("candidate_pair_count", ""),
                    "unique_feature_count": selection.get("unique_feature_count", ""),
                    "node_sign": args.node_sign,
                    "graph_output_path": str(graph_path),
                    "status": "ok",
                    "error_message": "",
                }
            )
            if index % 10 == 0 or index == len(rows):
                _log(f"{args.prompt_name}: {index}/{len(rows)} graphs complete")
            del outputs, features, hidden
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            meta["status"] = "error"
            meta["error_message"] = repr(exc)
            _log(f"{args.prompt_name}: {sample_id} failed: {exc!r}")
        meta_rows.append(meta)

    fieldnames = [
        "sample_id",
        "prompt_name",
        "question",
        "image_path",
        "answer_text",
        "assistant_prefix",
        "target_token_id",
        "target_token_text",
        "target_rank",
        "target_logit",
        "target_prob",
        "top1_token",
        "top1_token_id",
        "n_pos",
        "graph_n_pos",
        "visual_position_count",
        "answer_adjacent_position_count",
        "position_filter",
        "n_layers",
        "hidden_capture_source",
        "hidden_states_tuple_len",
        "selected_feature_nodes",
        "candidate_pair_count",
        "unique_feature_count",
        "node_sign",
        "graph_output_path",
        "status",
        "error_message",
    ]
    _write_csv(Path(args.meta_csv), meta_rows, fieldnames)
    ok = sum(1 for row in meta_rows if row.get("status") == "ok")
    payload["gpu_after"] = _gpu_info()
    payload["decision"] = {
        "status": "ok" if ok else "blocked_no_graphs",
        "graph_rows_ok": ok,
        "graph_rows_requested": len(meta_rows),
        "graph_success_rate": ok / len(meta_rows) if meta_rows else 0.0,
        "claim_boundary": "Schema-compatible Qwen answer-aligned graph; not yet full Gemma-style replication without compare/intervention controls.",
    }
    _write_json(Path(args.summary_json), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
