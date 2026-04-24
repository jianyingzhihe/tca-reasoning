#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
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


def _dtype_from_name(name: str):
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype: {name}")
    return mapping[name]


def _retry_feature_limits(base: int, retry_spec: str) -> list[int]:
    values = [base]
    for raw in (retry_spec or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        val = int(raw)
        if val > 0:
            values.append(val)
    dedup = []
    seen = set()
    for v in values:
        if v not in seen:
            dedup.append(v)
            seen.add(v)
    return dedup


def _cleanup_cuda():
    try:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        gc.collect()


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _log(message: str) -> None:
    print(message, flush=True)


def _memory_snapshot() -> str:
    parts: list[str] = []
    try:
        import psutil

        rss_gib = psutil.Process().memory_info().rss / (1024**3)
        parts.append(f"rss={rss_gib:.2f}GiB")
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            alloc_gib = torch.cuda.memory_allocated(dev) / (1024**3)
            reserved_gib = torch.cuda.memory_reserved(dev) / (1024**3)
            parts.append(f"cuda[{dev}] alloc={alloc_gib:.2f}GiB reserved={reserved_gib:.2f}GiB")
    except Exception:
        pass

    return ", ".join(parts)


def _log_memory(prefix: str) -> None:
    snapshot = _memory_snapshot()
    if snapshot:
        _log(f"{prefix} | {snapshot}")
    else:
        _log(prefix)


def _write_metadata_rows(metadata_csv: Path, meta_rows: list[dict], fieldnames: list[str]) -> None:
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    with metadata_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(meta_rows)


def _stream_subprocess(
    cmd: list[str],
    *,
    log_path: Path,
    env: dict[str, str],
) -> tuple[int, bool]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    saw_stage_log = False

    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_f.write(line)
            log_f.flush()
            text = line.rstrip()
            if any(
                marker in text
                for marker in (
                    "Loading HF model",
                    "HF model loaded",
                    "HookedVLTransformer ready",
                    "Replacement model configured",
                    "Phase 0:",
                    "Phase 1:",
                    "Phase 2:",
                    "Phase 3:",
                    "Phase 4:",
                )
            ):
                saw_stage_log = True
            _log(f"[child] {text}")

        return proc.wait(), saw_stage_log


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
    parser.add_argument("--no-lazy-encoder", action="store_false", dest="lazy_encoder")
    parser.add_argument("--no-lazy-decoder", action="store_false", dest="lazy_decoder")
    parser.set_defaults(lazy_encoder=True, lazy_decoder=True)
    parser.add_argument(
        "--retry-feature-nodes",
        default="64,48,32",
        help="Fallback max_feature_nodes values to try after the primary setting.",
    )
    parser.add_argument(
        "--exec-mode",
        default=os.environ.get("ANSWER_ATTR_EXEC_MODE", "subprocess"),
        choices=["inproc", "subprocess"],
        help="Run each attempt in-process or in a subprocess. Subprocess mode survives SIGKILL/OOM better.",
    )
    parser.add_argument(
        "--attempt-log-dir",
        default=os.environ.get("ANSWER_ATTR_ATTEMPT_LOG_DIR", ""),
        help="Optional directory for per-attempt logs. Defaults to <output-dir>/_attempt_logs.",
    )
    parser.add_argument(
        "--verbose-attribution",
        action="store_true",
        default=_bool_env("ANSWER_ATTR_VERBOSE_ATTRIBUTION", False),
        help="Emit detailed model-loading and attribution phase logs.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately after the first sample error.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from transformers import AutoProcessor

    eval_csv = Path(args.eval_csv).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    attempt_log_dir = (
        Path(args.attempt_log_dir).expanduser().resolve()
        if args.attempt_log_dir
        else out_dir / "_attempt_logs"
    )
    attempt_log_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv = (
        Path(args.metadata_csv).expanduser().resolve()
        if args.metadata_csv
        else out_dir / "answer_aligned_metadata.csv"
    )

    selected_ids = None
    if args.selected_csv:
        selected_ids = _read_selected_ids(Path(args.selected_csv).expanduser().resolve(), args.sample_id_col)

    model_name = _infer_model_name_from_transcoder_set(args.transcoder_set)
    _log(f"[init] inferred model_name={model_name}")
    _log_memory("[init] before processor load")
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer
    _log_memory("[init] after processor load")

    os.environ["CIRCUIT_TRACER_TOPK"] = str(args.topk)
    os.environ["CIRCUIT_TRACER_ENCODER_CPU"] = os.environ.get("CIRCUIT_TRACER_ENCODER_CPU", "1")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    dtype = _dtype_from_name(args.dtype)
    retry_limits = _retry_feature_limits(args.max_feature_nodes, args.retry_feature_nodes)

    with eval_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    t0 = time.time()
    processed = 0
    skipped = 0
    errors = 0
    meta_rows: list[dict] = []
    fieldnames = [
        "sample_id",
        "question",
        "image_path",
        "generated_text",
        "answer_text",
        "assistant_prefix",
        "target_token_id",
        "target_token_text",
        "used_max_feature_nodes",
        "graph_output_path",
        "attempt_log_path",
        "status",
        "error_message",
    ]

    answer_col = {
        "predicted": "predicted_answer",
        "gold": "gold_answer",
        "majority": "majority_answer",
    }[args.answer_source]

    if args.exec_mode == "inproc":
        from circuit_tracer import ReplacementModel, attribute

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
        used_max_feature_nodes = ""
        attempt_log_path = ""

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
                    "used_max_feature_nodes": "",
                    "graph_output_path": str(output_pt),
                    "attempt_log_path": "",
                    "status": "exists",
                    "error_message": "",
                }
            )
            _write_metadata_rows(metadata_csv, meta_rows, fieldnames)
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
            last_exc = None
            for attempt_idx, feature_limit in enumerate(retry_limits, start=1):
                cmd_preview = [
                    sys.executable,
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
                    str(feature_limit),
                    "--dtype",
                    args.dtype,
                ]
                if args.offload != "none":
                    cmd_preview.extend(["--offload", args.offload])
                if args.verbose_attribution:
                    cmd_preview.append("--verbose")

                _log(
                    f"[run] {idx}/{total} attempt={attempt_idx}/{len(retry_limits)} "
                    + " ".join(shlex.quote(x) for x in cmd_preview)
                )
                if args.dry_run:
                    used_max_feature_nodes = str(feature_limit)
                    last_exc = None
                    break

                model_instance = None
                graph = None
                try:
                    _cleanup_cuda()
                    _log_memory(f"[attempt] sample={sample_id} attempt={attempt_idx} before execution")

                    if args.exec_mode == "subprocess":
                        attempt_log = attempt_log_dir / f"{sample_id}.attempt{attempt_idx}.log"
                        attempt_log_path = str(attempt_log)
                        child_env = os.environ.copy()
                        child_env["PYTHONUNBUFFERED"] = "1"
                        returncode, saw_stage_log = _stream_subprocess(
                            cmd_preview,
                            log_path=attempt_log,
                            env=child_env,
                        )
                        if returncode != 0:
                            oom_hint = ""
                            if returncode in {-9, 137}:
                                oom_hint = " possible_oom_or_sigkill=1"
                            if returncode in {-9, 137} and not saw_stage_log:
                                oom_hint += " likely_died_during_model_load=1"
                            raise RuntimeError(
                                f"attribute_subprocess_failed returncode={returncode}.{oom_hint}".strip()
                            )
                        if not output_pt.exists():
                            raise RuntimeError(
                                f"attribute_subprocess_succeeded_but_graph_missing log={attempt_log}"
                            )
                    else:
                        model_instance = ReplacementModel.from_pretrained(
                            model_name,
                            args.transcoder_set,
                            dtype=dtype,
                            lazy_encoder=args.lazy_encoder,
                            lazy_decoder=args.lazy_decoder,
                        )
                        graph = attribute(
                            prompt=f"<start_of_image> {question}",
                            model=model_instance,
                            max_n_logits=1,
                            desired_logit_prob=0.95,
                            batch_size=args.batch_size,
                            max_feature_nodes=feature_limit,
                            offload=None if args.offload == "none" else args.offload,
                            verbose=args.verbose_attribution,
                            image_path=image_path,
                            assistant_prefix=assistant_prefix,
                            target_logit_ids=[token_id],
                        )
                        graph.to_pt(str(output_pt))

                    used_max_feature_nodes = str(feature_limit)
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if output_pt.exists():
                        try:
                            output_pt.unlink()
                        except OSError:
                            pass
                finally:
                    if graph is not None:
                        del graph
                    if model_instance is not None:
                        try:
                            model_instance.to("cpu")
                        except Exception:
                            pass
                        del model_instance
                    _cleanup_cuda()
                    _log_memory(f"[attempt] sample={sample_id} attempt={attempt_idx} after execution")

            if last_exc is not None:
                raise last_exc
            status = "ok"
            processed += 1
        except Exception as exc:  # noqa: BLE001
            status = "error"
            error_message = f"{type(exc).__name__}: {exc}"
            errors += 1
            _log(f"[error] sample={sample_id} {error_message}")
            if attempt_log_path:
                _log(f"[error] attempt_log={attempt_log_path}")

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
                "used_max_feature_nodes": used_max_feature_nodes,
                "graph_output_path": str(output_pt),
                "attempt_log_path": attempt_log_path,
                "status": status,
                "error_message": error_message,
            }
        )
        _write_metadata_rows(metadata_csv, meta_rows, fieldnames)

        elapsed = max(time.time() - t0, 1e-9)
        done = processed + skipped
        rate = done / elapsed
        eta_min = (total - done) / rate / 60.0 if rate > 0 else float("inf")
        _log(
            f"[progress] done={done}/{total} processed={processed} skipped={skipped} "
            f"rate={rate:.4f} item/s eta={eta_min:.1f}m"
        )
        if status == "error" and args.stop_on_error:
            break

    _write_metadata_rows(metadata_csv, meta_rows, fieldnames)

    _log(f"[done] metadata -> {metadata_csv}")
    _log(f"[done] graphs dir -> {out_dir}")
    _log(f"[summary] processed={processed} skipped={skipped} errors={errors}")
    if processed == 0 and errors > 0:
        _log("[fatal] no graphs were produced")
        return 2
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
