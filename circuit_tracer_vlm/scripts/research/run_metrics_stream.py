#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
import importlib
from pathlib import Path
from statistics import mean

import torch


def _load_done_ids(metrics_csv: Path) -> set[str]:
    if not metrics_csv.exists():
        return set()
    done = set()
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sid = (row.get("sample_id") or "").strip()
            status = (row.get("status") or "").strip()
            if sid and status in {"ok", "failed"}:
                done.add(sid)
    return done


def _append_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_summary(metrics_csv: Path, summary_path: Path) -> None:
    rows = []
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("status") or "").strip() != "ok":
                continue
            rows.append(r)
    if not rows:
        summary_path.write_text("rows=0\n", encoding="utf-8")
        return

    rep = sorted(float(r["replacement_score"]) for r in rows)
    comp = sorted(float(r["completeness_score"]) for r in rows)

    def q(vals: list[float], p: float) -> float:
        i = int((len(vals) - 1) * p)
        return vals[i]

    lines = [
        f"rows={len(rows)}",
        f"mean_replacement={mean(rep)}",
        f"mean_completeness={mean(comp)}",
        f"replacement_p25/p50/p75={q(rep,0.25)}/{q(rep,0.5)}/{q(rep,0.75)}",
        f"completeness_p25/p50/p75={q(comp,0.25)}/{q(comp,0.5)}/{q(comp,0.75)}",
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_okvqa_manifest(
    manifest_path: Path,
    okvqa_root: Path,
    split: str,
    limit: int,
    seed: int,
) -> None:
    q_file = (
        "OpenEnded_mscoco_train2014_questions.json"
        if split == "train"
        else "OpenEnded_mscoco_val2014_questions.json"
    )
    a_file = (
        "mscoco_train2014_annotations.json" if split == "train" else "mscoco_val2014_annotations.json"
    )
    img_subdir = "train2014" if split == "train" else "val2014"
    img_prefix = "COCO_train2014_" if split == "train" else "COCO_val2014_"

    q_path = okvqa_root / "questions" / q_file
    a_path = okvqa_root / "annotations" / a_file
    img_root = okvqa_root / "images" / img_subdir

    if not q_path.exists():
        raise FileNotFoundError(f"missing questions file: {q_path}")
    if not a_path.exists():
        raise FileNotFoundError(f"missing annotations file: {a_path}")
    if not img_root.exists():
        raise FileNotFoundError(f"missing image folder: {img_root}")

    with q_path.open("r", encoding="utf-8") as f:
        q_data = json.load(f)["questions"]
    with a_path.open("r", encoding="utf-8") as f:
        a_data = json.load(f)["annotations"]

    ans_map = {}
    for a in a_data:
        qid = int(a["question_id"])
        ans = ""
        if a.get("answers"):
            ans = (a["answers"][0].get("answer") or "").strip()
        ans_map[qid] = ans

    rows = []
    for item in q_data:
        qid = int(item["question_id"])
        iid = int(item["image_id"])
        question = (item.get("question") or "").strip()
        if not question:
            continue
        rows.append(
            {
                "sample_id": "",
                "image_path": str(img_root / f"{img_prefix}{iid:012d}.jpg"),
                "question": question,
                "split": split,
                "trace_source": "none",
                "question_type": "okvqa",
                "notes": f"qid={qid};image_id={iid};answer={ans_map.get(qid,'')}",
            }
        )

    import random

    rng = random.Random(seed)
    rng.shuffle(rows)
    if limit > 0:
        rows = rows[:limit]
    for i, row in enumerate(rows, start=1):
        row["sample_id"] = f"okvqa_{split}_{i:05d}"

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "image_path", "question", "split", "trace_source", "question_type", "notes"]
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[auto] built manifest: {manifest_path} rows={len(rows)}")


def _normalize_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "fp32": "float32",
        "bf16": "bfloat16",
        "fp16": "float16",
    }
    name = mapping.get(dtype_str, dtype_str)
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"unsupported dtype: {dtype_str}") from exc


def _ensure_transformer_lens_with_vl() -> None:
    """Ensure we import a TransformerLens build that contains HookedVLTransformer."""
    script_path = Path(__file__).resolve()
    candidate_paths = [
        script_path.parents[3] / "third_party" / "TransformerLens",  # repo_root/third_party/TransformerLens
        script_path.parents[2] / "third_party" / "TransformerLens",  # circuit_tracer_vlm/third_party/TransformerLens
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)

    # If site-packages transformer_lens was imported earlier in this process,
    # force a clean re-import from the updated sys.path.
    for name in list(sys.modules.keys()):
        if name == "transformer_lens" or name.startswith("transformer_lens."):
            del sys.modules[name]

    lens = importlib.import_module("transformer_lens")

    if not hasattr(lens, "HookedVLTransformer"):
        raise ImportError(
            "transformer_lens does not provide HookedVLTransformer. "
            "Use the vendored third_party/TransformerLens fork or disable --reuse-model."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run attribution per manifest row, compute graph metrics, optionally delete .pt."
    )
    parser.add_argument("--manifest", required=True, help="Path to sample_manifest.csv")
    parser.add_argument("--metrics-output", required=True, help="Output CSV path")
    parser.add_argument("--summary-output", default="", help="Output summary txt path")
    parser.add_argument("--temp-pt-dir", required=True, help="Temp directory for per-sample .pt files")
    parser.add_argument(
        "--model",
        default="",
        help="Optional model name (if empty, infer from transcoder config).",
    )
    parser.add_argument("--transcoder-set", required=True, help="Transcoder set repo id")
    parser.add_argument("--dtype", default="bfloat16", help="float16/bfloat16/float32")
    parser.add_argument("--max-feature-nodes", type=int, default=96)
    parser.add_argument("--max-n-logits", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--offload", default="cpu", choices=["cpu", "disk", "none"])
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--encoder-cpu", action="store_true", default=True)
    parser.add_argument("--keep-pt", action="store_true", help="Keep .pt files instead of deleting")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip sample_ids already in metrics CSV")
    parser.add_argument("--log-every", type=int, default=10, help="Print progress every N samples")
    parser.add_argument(
        "--okvqa-root",
        default="~/tca-reasoning/data/okvqa",
        help="Root path for OK-VQA data (used when manifest is missing).",
    )
    parser.add_argument(
        "--okvqa-split",
        choices=["train", "val"],
        default="val",
        help="Split for auto-generated manifest when manifest is missing.",
    )
    parser.add_argument(
        "--okvqa-limit",
        type=int,
        default=1000,
        help="Sample count for auto-generated manifest when manifest is missing.",
    )
    parser.add_argument(
        "--okvqa-seed",
        type=int,
        default=42,
        help="Random seed for auto-generated manifest when manifest is missing.",
    )
    parser.add_argument(
        "--reuse-model",
        action="store_true",
        help="Load model/transcoder once and reuse across all samples (much faster).",
    )
    args = parser.parse_args()

    manifest = Path(args.manifest)
    metrics_output = Path(args.metrics_output)
    summary_output = Path(args.summary_output) if args.summary_output else metrics_output.with_suffix(".summary.txt")
    temp_pt_dir = Path(args.temp_pt_dir)
    temp_pt_dir.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    if not manifest.exists():
        okvqa_root = Path(args.okvqa_root).expanduser().resolve()
        print(f"[auto] manifest missing: {manifest}")
        print(
            f"[auto] trying to build from OK-VQA root={okvqa_root}, split={args.okvqa_split}, limit={args.okvqa_limit}"
        )
        _build_okvqa_manifest(
            manifest_path=manifest,
            okvqa_root=okvqa_root,
            split=args.okvqa_split,
            limit=args.okvqa_limit,
            seed=args.okvqa_seed,
        )

    with manifest.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    done_ids = _load_done_ids(metrics_output) if args.resume else set()
    fieldnames = [
        "sample_id",
        "replacement_score",
        "completeness_score",
        "n_nodes_total",
        "n_nonzero_edges",
        "status",
        "error_message",
    ]

    env = os.environ.copy()
    env["CIRCUIT_TRACER_TOPK"] = str(args.topk)
    env["CIRCUIT_TRACER_ENCODER_CPU"] = "1" if args.encoder_cpu else "0"
    os.environ["CIRCUIT_TRACER_TOPK"] = str(args.topk)
    os.environ["CIRCUIT_TRACER_ENCODER_CPU"] = "1" if args.encoder_cpu else "0"

    total = len(rows)
    processed = 0
    started = time.time()
    offload_value = None if args.offload == "none" else args.offload

    model_instance = None
    if args.reuse_model:
        _ensure_transformer_lens_with_vl()
        from circuit_tracer import ReplacementModel
        from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

        dtype = _normalize_dtype(args.dtype)
        print("[init] loading transcoder once...")
        transcoder, config = load_transcoder_from_hub(
            args.transcoder_set,
            dtype=dtype,
            lazy_encoder=False,
            lazy_decoder=True,
        )
        model_name = args.model.strip() or config.get("model_name", "")
        if not model_name:
            raise ValueError("--model is required when model_name is absent in transcoder config")
        print(f"[init] loading model once: {model_name}")
        model_instance = ReplacementModel.from_pretrained_and_transcoders(
            model_name,
            transcoder,
            dtype=dtype,
        )
        print("[init] model ready (reuse-model enabled)")

    from circuit_tracer.graph import Graph, compute_graph_scores

    for i, row in enumerate(rows, start=1):
        sample_id = (row.get("sample_id") or "").strip()
        image_path = (row.get("image_path") or "").strip()
        question = (row.get("question") or "").strip()
        if not sample_id or not image_path or not question:
            print(f"[skip] {i}/{total} invalid row")
            continue
        if sample_id in done_ids:
            processed += 1
            continue

        output_pt = temp_pt_dir / f"{sample_id}.pt"
        output_pt.unlink(missing_ok=True)

        try:
            print(f"[run] {i}/{total} sample={sample_id}")
            if args.reuse_model:
                from circuit_tracer import attribute

                assert model_instance is not None
                g = attribute(
                    prompt=f"<start_of_image> {question}",
                    model=model_instance,
                    max_n_logits=args.max_n_logits,
                    desired_logit_prob=0.95,
                    batch_size=args.batch_size,
                    max_feature_nodes=args.max_feature_nodes,
                    offload=offload_value,
                    verbose=False,
                    image_path=image_path,
                )
                if args.keep_pt:
                    g.to_pt(str(output_pt))
            else:
                cmd = [
                    "circuit-tracer",
                    "attribute",
                    "--prompt",
                    f"<start_of_image> {question}",
                    "--transcoder_set",
                    args.transcoder_set,
                    "--image",
                    image_path,
                    "--graph_output_path",
                    str(output_pt),
                    "--batch_size",
                    str(args.batch_size),
                    "--max_n_logits",
                    str(args.max_n_logits),
                    "--max_feature_nodes",
                    str(args.max_feature_nodes),
                    "--dtype",
                    args.dtype,
                ]
                if args.offload != "none":
                    cmd.extend(["--offload", args.offload])
                proc = subprocess.run(cmd, env=env, check=False)
                if proc.returncode != 0 or not output_pt.exists():
                    raise RuntimeError(f"attribute_return_code={proc.returncode}")
                g = Graph.from_pt(str(output_pt), map_location="cpu")

            rep, comp = compute_graph_scores(g)
            _append_row(
                metrics_output,
                {
                    "sample_id": sample_id,
                    "replacement_score": rep,
                    "completeness_score": comp,
                    "n_nodes_total": int(g.adjacency_matrix.shape[0]),
                    "n_nonzero_edges": int((g.adjacency_matrix != 0).sum().item()),
                    "status": "ok",
                    "error_message": "",
                },
                fieldnames,
            )
        except Exception as exc:  # noqa: BLE001
            _append_row(
                metrics_output,
                {
                    "sample_id": sample_id,
                    "replacement_score": "",
                    "completeness_score": "",
                    "n_nodes_total": "",
                    "n_nonzero_edges": "",
                    "status": "failed",
                    "error_message": f"{type(exc).__name__}:{exc}",
                },
                fieldnames,
            )
        finally:
            if not args.keep_pt:
                output_pt.unlink(missing_ok=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        done_ids.add(sample_id)
        processed += 1
        if processed % max(1, args.log_every) == 0 or processed == total:
            elapsed = max(time.time() - started, 1e-9)
            rate = processed / elapsed
            eta_min = (total - processed) / rate / 60.0 if rate > 0 else float("inf")
            print(
                f"[progress] {processed}/{total} ({processed/total*100:.1f}%) "
                f"rate={rate:.4f} sample/s eta={eta_min:.1f}m"
            )
            _write_summary(metrics_output, summary_output)

    _write_summary(metrics_output, summary_output)
    print(f"[done] metrics csv: {metrics_output}")
    print(f"[done] summary txt: {summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
