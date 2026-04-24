#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def _read_selected_ids(selected_csv: Path, sample_id_col: str) -> set[str]:
    with selected_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        (r.get(sample_id_col) or "").strip()
        for r in rows
        if (r.get(sample_id_col) or "").strip()
    }


def _extract_answer_prefix(generated_text: str, answer_text: str) -> str:
    txt = (generated_text or "").strip()
    ans = (answer_text or "").strip()
    if not txt:
        raise ValueError("generated_text is empty")

    m = re.search(r"(?is)^(.*?the answer is\s*[:\-]?\s*)", txt)
    if m:
        prefix = m.group(1)
        if prefix and not prefix.endswith((" ", "\t", "\n")):
            prefix += " "
        return prefix

    if ans:
        idx = txt.lower().find(ans.lower())
        if idx >= 0:
            prefix = txt[:idx]
            if prefix and not prefix.endswith((" ", "\t", "\n")):
                prefix += " "
            return prefix

    raise ValueError("unable to locate answer prefix inside generated_text")


def _first_answer_token_id(tokenizer, assistant_prefix: str, answer_text: str) -> tuple[int, str]:
    prefix = assistant_prefix or ""
    answer = (answer_text or "").strip()
    if not answer:
        raise ValueError("answer_text is empty")

    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prefix + answer, add_special_tokens=False)["input_ids"]
    if len(full_ids) <= len(prefix_ids):
        raise ValueError("tokenization produced no answer token after prefix")

    token_id = int(full_ids[len(prefix_ids)])
    token_text = tokenizer.convert_ids_to_tokens([token_id])[0]
    return token_id, token_text


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run answer-token-aligned attribution by reusing eval outputs. "
            "Each graph targets the first token of the chosen answer at the answer position."
        )
    )
    parser.add_argument("--eval-csv", required=True, help="Input eval csv (promptA_eval / promptB_eval)")
    parser.add_argument("--output-dir", required=True, help="Directory to store .pt graphs")
    parser.add_argument("--transcoder-set", required=True, help="Transcoder set repo id")
    parser.add_argument("--selected-csv", default="", help="Optional csv restricting sample_ids to process")
    parser.add_argument("--sample-id-col", default="sample_id")
    parser.add_argument(
        "--answer-source",
        default="predicted",
        choices=["predicted", "gold", "majority"],
        help="Which answer column to align to",
    )
    parser.add_argument("--metadata-csv", default="", help="Optional output csv with prefix/token metadata")
    parser.add_argument("--dtype", default="bfloat16", help="float16/bfloat16/float32")
    parser.add_argument("--max-feature-nodes", type=int, default=112)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--offload", default="cpu", choices=["cpu", "disk", "none"])
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from transformers import AutoProcessor

    eval_csv = Path(args.eval_csv).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv = (
        Path(args.metadata_csv).expanduser().resolve()
        if args.metadata_csv
        else out_dir / "answer_aligned_metadata.csv"
    )

    selected_ids = None
    if args.selected_csv:
        selected_ids = _read_selected_ids(Path(args.selected_csv).expanduser().resolve(), args.sample_id_col)

    model_name = _infer_model_name_from_transcoder_set(args.transcoder_set)
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer

    env = os.environ.copy()
    env["CIRCUIT_TRACER_TOPK"] = str(args.topk)
    env["CIRCUIT_TRACER_ENCODER_CPU"] = env.get("CIRCUIT_TRACER_ENCODER_CPU", "1")
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    repo_root = Path(__file__).resolve().parents[2]
    venv_python = repo_root / ".venv" / "bin" / "python"
    runner_python = env.get("TCA_PYTHON") or (str(venv_python) if venv_python.exists() else sys.executable)

    with eval_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    t0 = time.time()
    processed = 0
    skipped = 0
    meta_rows: list[dict] = []

    answer_col = {
        "predicted": "predicted_answer",
        "gold": "gold_answer",
        "majority": "majority_answer",
    }[args.answer_source]

    for idx, row in enumerate(rows, start=1):
        sample_id = (row.get("sample_id") or "").strip()
        if not sample_id:
            skipped += 1
            continue
        if selected_ids is not None and sample_id not in selected_ids:
            skipped += 1
            continue

        image_path = (row.get("image_path") or "").strip()
        question = (row.get("question") or "").strip()
        generated_text = (row.get("generated_text") or "").strip()
        answer_text = (row.get(answer_col) or "").strip()
        error_message = ""
        assistant_prefix = ""
        target_token_id = ""
        target_token_text = ""

        output_pt = out_dir / f"{sample_id}.pt"
        if output_pt.exists():
            processed += 1
            meta_rows.append(
                {
                    "sample_id": sample_id,
                    "question": question,
                    "image_path": image_path,
                    "generated_text": generated_text,
                    "answer_text": answer_text,
                    "assistant_prefix": "",
                    "target_token_id": "",
                    "target_token_text": "",
                    "graph_output_path": str(output_pt),
                    "status": "exists",
                    "error_message": "",
                }
            )
            continue

        try:
            if not image_path or not question:
                raise ValueError("missing image_path or question")
            if not answer_text:
                raise ValueError(f"missing {answer_col}")
            assistant_prefix = _extract_answer_prefix(generated_text, answer_text)
            token_id, token_text = _first_answer_token_id(tokenizer, assistant_prefix, answer_text)
            target_token_id = str(token_id)
            target_token_text = token_text

            cmd_preview = [
                runner_python,
                "-m",
                "circuit_tracer",
                "attribute",
                "--prompt",
                f"<start_of_image> {question}",
                "--assistant-prefix",
                assistant_prefix,
                "--target-logit-ids",
                target_token_id,
                "--transcoder_set",
                args.transcoder_set,
                "--image",
                image_path,
                "--graph_output_path",
                str(output_pt),
                "--batch_size",
                str(args.batch_size),
                "--max_n_logits",
                "1",
                "--max_feature_nodes",
                str(args.max_feature_nodes),
                "--dtype",
                args.dtype,
            ]
            if args.offload != "none":
                cmd_preview.extend(["--offload", args.offload])

            print(f"[run] {idx}/{total}", " ".join(shlex.quote(x) for x in cmd_preview))
            if not args.dry_run:
                proc = subprocess.run(
                    cmd_preview,
                    env=env,
                    check=False,
                    text=True,
                    capture_output=True,
                )
                if proc.returncode != 0:
                    stderr = (proc.stderr or "").strip()
                    stdout = (proc.stdout or "").strip()
                    detail = stderr or stdout
                    if detail:
                        detail = detail.replace("\n", " | ")
                        raise RuntimeError(
                            f"attribute command failed with exit code {proc.returncode}: {detail[-800:]}"
                        )
                    raise RuntimeError(f"attribute command failed with exit code {proc.returncode}")
            status = "ok"
            processed += 1
        except Exception as exc:  # noqa: BLE001
            status = "error"
            error_message = f"{type(exc).__name__}: {exc}"

        meta_rows.append(
            {
                "sample_id": sample_id,
                "question": question,
                "image_path": image_path,
                "generated_text": generated_text,
                "answer_text": answer_text,
                "assistant_prefix": assistant_prefix,
                "target_token_id": target_token_id,
                "target_token_text": target_token_text,
                "graph_output_path": str(output_pt),
                "status": status,
                "error_message": error_message,
            }
        )

        elapsed = max(time.time() - t0, 1e-9)
        done = processed + skipped
        rate = done / elapsed
        eta_min = (total - done) / rate / 60.0 if rate > 0 else float("inf")
        print(
            f"[progress] done={done}/{total} processed={processed} skipped={skipped} "
            f"rate={rate:.4f} item/s eta={eta_min:.1f}m"
        )

    fieldnames = [
        "sample_id",
        "question",
        "image_path",
        "generated_text",
        "answer_text",
        "assistant_prefix",
        "target_token_id",
        "target_token_text",
        "graph_output_path",
        "status",
        "error_message",
    ]
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    with metadata_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(meta_rows)

    print(f"[done] metadata -> {metadata_csv}")
    print(f"[done] graphs dir -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
