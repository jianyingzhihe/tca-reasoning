#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import time
from pathlib import Path
from statistics import mean

from circuit_tracer.graph import Graph, compute_graph_scores


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run attribution per manifest row, compute graph metrics, optionally delete .pt."
    )
    parser.add_argument("--manifest", required=True, help="Path to sample_manifest.csv")
    parser.add_argument("--metrics-output", required=True, help="Output CSV path")
    parser.add_argument("--summary-output", default="", help="Output summary txt path")
    parser.add_argument("--temp-pt-dir", required=True, help="Temp directory for per-sample .pt files")
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
    args = parser.parse_args()

    manifest = Path(args.manifest)
    metrics_output = Path(args.metrics_output)
    summary_output = Path(args.summary_output) if args.summary_output else metrics_output.with_suffix(".summary.txt")
    temp_pt_dir = Path(args.temp_pt_dir)
    temp_pt_dir.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

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

    total = len(rows)
    processed = 0
    started = time.time()

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

        print(f"[run] {i}/{total} sample={sample_id}")
        proc = subprocess.run(cmd, env=env, check=False)
        if proc.returncode != 0 or not output_pt.exists():
            _append_row(
                metrics_output,
                {
                    "sample_id": sample_id,
                    "replacement_score": "",
                    "completeness_score": "",
                    "n_nodes_total": "",
                    "n_nonzero_edges": "",
                    "status": "failed",
                    "error_message": f"attribute_return_code={proc.returncode}",
                },
                fieldnames,
            )
            done_ids.add(sample_id)
            processed += 1
            continue

        try:
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
                    "error_message": f"metrics_exception={type(exc).__name__}",
                },
                fieldnames,
            )
        finally:
            if not args.keep_pt:
                output_pt.unlink(missing_ok=True)

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
