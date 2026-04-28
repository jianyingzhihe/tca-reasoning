#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: str | None) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except Exception:
        return math.nan


def _infer_model_name_from_transcoder_set(repo_id: str) -> str:
    from huggingface_hub import hf_hub_download
    import yaml

    config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_name = (cfg or {}).get("model_name", "")
    if not model_name:
        raise ValueError(f"model_name missing in {repo_id}/config.yaml")
    return model_name


def _load_generic_features(path: Path | None) -> set[tuple[str, str, str]]:
    if path is None or not path.exists():
        return set()
    out = set()
    for row in _read_csv(path):
        if row.get("is_generic", "").lower() != "true":
            continue
        if row.get("node_type") != "feature":
            continue
        out.add((row.get("layer", ""), row.get("pos", ""), row.get("feature_or_token_id", "")))
    return out


def _load_selected_sample_ids(path: Path, sample_id_col: str) -> list[str]:
    rows = _read_csv(path)
    out = []
    for row in rows:
        sample_id = (row.get(sample_id_col) or "").strip()
        if sample_id:
            out.append(sample_id)
    return out


def _select_samples(
    compare_rows: list[dict[str, str]],
    *,
    rank_metric: str,
    descending: bool,
    max_samples: int,
) -> list[str]:
    ranked = sorted(compare_rows, key=lambda r: _safe_float(r.get(rank_metric)), reverse=descending)
    return [r["sample_id"] for r in ranked[:max_samples]]


def _device_batch(model, batch: dict) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(model.cfg.device) if torch.is_tensor(value) else value
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run feature-ablation smoke tests on answer-aligned graphs.")
    parser.add_argument("--run-root", required=True, help="Bucket run root containing meta csvs and pt dirs.")
    parser.add_argument("--compare-dir", required=True, help="Directory containing compare csv outputs.")
    parser.add_argument("--bucket", default="", help="Optional bucket label (for easier downstream aggregation).")
    parser.add_argument("--run", choices=["A", "B", "both"], default="both")
    parser.add_argument("--transcoder-set", default="tianhux2/gemma3-4b-it-plt")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16", "fp32", "bf16", "fp16"])
    parser.add_argument("--sample-rank-metric", default="edge_overlap_jaccard")
    parser.add_argument("--sample-rank-desc", action="store_true")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--sample-ids-csv", default="", help="Optional CSV listing sample_id values to use directly.")
    parser.add_argument("--sample-id-col", default="sample_id")
    parser.add_argument("--require-same-target", action="store_true")
    parser.add_argument("--top-features-per-sample", type=int, default=2)
    parser.add_argument("--generic-nodes-csv", default="")
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    compare_dir = Path(args.compare_dir).expanduser().resolve()
    meta_a = {r["sample_id"]: r for r in _read_csv(run_root / "answer_aligned_meta_a.csv")}
    meta_b = {r["sample_id"]: r for r in _read_csv(run_root / "answer_aligned_meta_b.csv")}
    sample_compare = _read_csv(compare_dir / "sample_compare_controlled.csv")
    nodes = _read_csv(compare_dir / "nodes_detailed_controlled.csv")
    generic_features = _load_generic_features(
        Path(args.generic_nodes_csv).expanduser().resolve() if args.generic_nodes_csv else None
    )

    filtered_compare = sample_compare
    if args.require_same_target:
        filtered_compare = [
            row
            for row in filtered_compare
            if (row.get("a_target_token_id", "") == row.get("b_target_token_id", ""))
        ]
    if args.sample_ids_csv:
        selected_samples = _load_selected_sample_ids(
            Path(args.sample_ids_csv).expanduser().resolve(),
            args.sample_id_col,
        )
        if args.max_samples > 0:
            selected_samples = selected_samples[: args.max_samples]
    else:
        selected_samples = _select_samples(
            filtered_compare,
            rank_metric=args.sample_rank_metric,
            descending=args.sample_rank_desc,
            max_samples=args.max_samples,
        )
    selected_set = set(selected_samples)

    candidate_rows = []
    runs = ["A", "B"] if args.run == "both" else [args.run]
    for run in runs:
        for row in nodes:
            if row.get("run") != run:
                continue
            if row.get("node_type") != "feature":
                continue
            sample_id = row.get("sample_id", "")
            if sample_id not in selected_set:
                continue
            key = (row.get("layer", ""), row.get("pos", ""), row.get("feature_id", ""))
            if key in generic_features:
                continue
            candidate_rows.append(row)

    candidate_rows.sort(
        key=lambda r: (
            r.get("sample_id", ""),
            r.get("run", ""),
            -_safe_float(r.get("path_mass_best")),
        )
    )

    top_candidates: list[dict[str, str]] = []
    per_sample_run_count: dict[tuple[str, str], int] = {}
    for row in candidate_rows:
        key = (row["sample_id"], row["run"])
        count = per_sample_run_count.get(key, 0)
        if count >= args.top_features_per_sample:
            continue
        top_candidates.append(row)
        per_sample_run_count[key] = count + 1

    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    dtype = dtype_map[args.dtype]

    from circuit_tracer import ReplacementModel
    from circuit_tracer.attribution.attribute import _build_multimodal_batch

    model_name = args.model_name or _infer_model_name_from_transcoder_set(args.transcoder_set)
    print(f"[info] loading model={model_name} transcoder_set={args.transcoder_set} dtype={dtype}")
    model = ReplacementModel.from_pretrained(
        model_name,
        args.transcoder_set,
        dtype=dtype,
        lazy_encoder=True,
        lazy_decoder=True,
    )

    results: list[dict[str, str]] = []
    for row in top_candidates:
        sample_id = row["sample_id"]
        run = row["run"]
        meta = meta_a[sample_id] if run == "A" else meta_b[sample_id]
        question = meta["question"]
        assistant_prefix = meta["assistant_prefix"]
        target_token_id = int(meta["target_token_id"])
        image_path = meta["image_path"]
        image = Image.open(image_path).convert("RGB")
        batch = _build_multimodal_batch(
            model.processor,
            image,
            f"<start_of_image> {question}",
            assistant_prefix=assistant_prefix,
        )
        batch["image"] = image
        batch = _device_batch(model, batch)

        layer = int(row["layer"])
        pos = int(row["pos"])
        feature_id = int(row["feature_id"])
        path_mass_best = _safe_float(row.get("path_mass_best"))

        with torch.inference_mode():
            original_logits = model.forward_from_batch(batch)
            intervened_logits, _ = model.feature_intervention(
                batch,
                [(layer, pos, feature_id, 0.0)],
                freeze_attention=True,
                apply_activation_function=True,
                sparse=False,
            )

        last_pos = original_logits.shape[1] - 1
        original_target_logit = float(original_logits[0, last_pos, target_token_id].item())
        intervened_target_logit = float(intervened_logits[0, last_pos, target_token_id].item())
        delta_target_logit = intervened_target_logit - original_target_logit

        original_probs = torch.softmax(original_logits[0, last_pos], dim=-1)
        intervened_probs = torch.softmax(intervened_logits[0, last_pos], dim=-1)
        original_target_prob = float(original_probs[target_token_id].item())
        intervened_target_prob = float(intervened_probs[target_token_id].item())
        delta_target_prob = intervened_target_prob - original_target_prob

        top1_before = int(torch.argmax(original_logits[0, last_pos]).item())
        top1_after = int(torch.argmax(intervened_logits[0, last_pos]).item())

        results.append(
            {
                "bucket": args.bucket,
                "sample_id": sample_id,
                "run": run,
                "question": question,
                "image_path": image_path,
                "target_token_id": str(target_token_id),
                "feature_layer": str(layer),
                "feature_pos": str(pos),
                "feature_id": str(feature_id),
                "path_mass_best": f"{path_mass_best:.10g}" if not math.isnan(path_mass_best) else "",
                "original_target_logit": f"{original_target_logit:.10g}",
                "intervened_target_logit": f"{intervened_target_logit:.10g}",
                "delta_target_logit": f"{delta_target_logit:.10g}",
                "original_target_prob": f"{original_target_prob:.10g}",
                "intervened_target_prob": f"{intervened_target_prob:.10g}",
                "delta_target_prob": f"{delta_target_prob:.10g}",
                "top1_before_id": str(top1_before),
                "top1_after_id": str(top1_after),
                "top1_before_token": model.processor.tokenizer.convert_ids_to_tokens([top1_before])[0],
                "top1_after_token": model.processor.tokenizer.convert_ids_to_tokens([top1_after])[0],
            }
        )
        print(
            f"[done] sample={sample_id} run={run} feature=L{layer}:P{pos}:F{feature_id} "
            f"delta_target_logit={delta_target_logit:.4f} delta_target_prob={delta_target_prob:.6f}"
        )

    _write_csv(
        Path(args.out_csv).expanduser().resolve(),
        results,
        [
            "bucket",
            "sample_id",
            "run",
            "question",
            "image_path",
            "target_token_id",
            "feature_layer",
            "feature_pos",
            "feature_id",
            "path_mass_best",
            "original_target_logit",
            "intervened_target_logit",
            "delta_target_logit",
            "original_target_prob",
            "intervened_target_prob",
            "delta_target_prob",
            "top1_before_id",
            "top1_after_id",
            "top1_before_token",
            "top1_after_token",
        ],
    )
    print(f"[done] out_csv={Path(args.out_csv).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
