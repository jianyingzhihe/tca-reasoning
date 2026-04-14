#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run circuit-tracer attribute for all manifest rows.")
    parser.add_argument("--manifest", required=True, help="Path to sample_manifest.csv")
    parser.add_argument("--output-dir", required=True, help="Directory to store .pt graphs")
    parser.add_argument("--transcoder-set", required=True, help="Transcoder set repo id")
    parser.add_argument("--dtype", default="bfloat16", help="float16/bfloat16/float32")
    parser.add_argument("--max-feature-nodes", type=int, default=256)
    parser.add_argument("--max-n-logits", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--offload", default="cpu", choices=["cpu", "disk", "none"])
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--encoder-cpu", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CIRCUIT_TRACER_TOPK"] = str(args.topk)
    env["CIRCUIT_TRACER_ENCODER_CPU"] = "1" if args.encoder_cpu else "0"

    with manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        sample_id = row["sample_id"].strip()
        image_path = row["image_path"].strip()
        question = row["question"].strip()
        if not sample_id or not image_path or not question:
            print(f"[skip] invalid row: {row}")
            continue

        output_pt = out_dir / f"{sample_id}.pt"
        if output_pt.exists():
            print(f"[skip] exists: {output_pt}")
            continue

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

        print("[run]", " ".join(shlex.quote(x) for x in cmd))
        if args.dry_run:
            continue

        proc = subprocess.run(cmd, env=env, check=False)
        if proc.returncode != 0:
            print(f"[error] failed sample: {sample_id}")
            return proc.returncode

    print("[done] batch attribute completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

