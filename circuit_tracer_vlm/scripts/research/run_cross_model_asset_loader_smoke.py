#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_INDEX_RE = re.compile(r"(?:^|_)(\d+)\.safetensors$")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage2f] {message}", flush=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(str(value)))
    except Exception:
        return None


def _layer_index(name: str) -> int | None:
    match = _INDEX_RE.search(Path(name).name)
    if not match:
        return None
    return int(match.group(1))


def _status_from_steps(config_ok: bool, files_ok: bool, lazy_status: str) -> str:
    if not config_ok or not files_ok:
        return "blocked"
    if lazy_status == "ok":
        return "pass"
    return "partial"


def _remote_file_size(api, repo_id: str, filename: str, revision: str | None) -> int | None:
    try:
        info = api.get_paths_info(repo_id=repo_id, paths=[filename], revision=revision)
        if info:
            return getattr(info[0], "size", None)
    except Exception:
        return None
    return None


def _try_cached_snapshot(repo_id: str, revision: str | None) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=["*.safetensors"],
        local_files_only=True,
    )


def _try_cached_file(repo_id: str, filename: str, revision: str | None) -> str | None:
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_files_only=True,
        )
    except Exception:
        return None


def _local_safetensors_in_snapshot(snapshot_path: str) -> list[str]:
    if not snapshot_path:
        return []
    root = Path(snapshot_path)
    if not root.exists():
        return []
    return sorted(str(p.relative_to(root)).replace("\\", "/") for p in root.glob("*.safetensors") if p.is_file())


def _inspect_safetensor(path: str) -> tuple[str, str]:
    from safetensors import safe_open

    tensors: list[str] = []
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            try:
                shape = list(f.get_slice(key).get_shape())
            except Exception:
                shape = []
            tensors.append(f"{key}:{shape}")
    return "|".join(tensors), ""


def _replacement_model_backend_check() -> dict[str, Any]:
    try:
        from circuit_tracer.replacement_model import ReplacementModel

        src = inspect.getsource(ReplacementModel.from_pretrained_and_transcoders)
        return {
            "replacement_model_import_ok": True,
            "uses_gemma3_for_conditional_generation": "Gemma3ForConditionalGeneration" in src,
            "uses_auto_model": "AutoModel" in src or "AutoModelFor" in src,
            "qwen_adapter_present": "Qwen2" in src or "Qwen/Qwen2.5" in src,
            "note": (
                "ReplacementModel.from_pretrained_and_transcoders is Gemma3-oriented"
                if "Gemma3ForConditionalGeneration" in src
                else "ReplacementModel backend is not obviously Gemma3-only"
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "replacement_model_import_ok": False,
            "uses_gemma3_for_conditional_generation": "",
            "uses_auto_model": "",
            "qwen_adapter_present": "",
            "note": f"{type(exc).__name__}:{exc}",
        }


def _processor_local_smoke(model_name: str) -> dict[str, Any]:
    if not model_name:
        return {"attempted": False, "status": "skipped_no_model_name"}
    try:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(model_name, local_files_only=True)
        return {
            "attempted": True,
            "status": "ok",
            "processor_class": type(proc).__name__,
            "has_tokenizer": hasattr(proc, "tokenizer"),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "attempted": True,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 2F-1 lightweight cross-model CLT asset loader smoke."
    )
    parser.add_argument("--transcoder-set", default="KokosDev/qwen2p5vl-7b-clt")
    parser.add_argument("--revision", default="")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--max-shape-downloads", type=int, default=0)
    parser.add_argument("--max-shape-download-mb", type=float, default=256.0)
    parser.add_argument("--allow-full-transcoder-download", action="store_true")
    parser.add_argument("--try-processor-local", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    repo_id = args.transcoder_set
    revision = args.revision.strip() or None
    api = HfApi()

    payload: dict[str, Any] = {
        "created_at": _now(),
        "transcoder_set": repo_id,
        "revision": revision or "main",
        "config": {},
        "repo_files": {},
        "local_cache": {},
        "lazy_clt_load": {},
        "processor_local_smoke": {},
        "replacement_model_backend": _replacement_model_backend_check(),
        "decision": {},
    }

    config_ok = False
    files_ok = False
    config: dict[str, Any] = {}
    config_error = ""
    _log(f"reading config.yaml from {repo_id}")
    try:
        config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml", revision=revision)
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        config_ok = True
        payload["config"] = {
            "status": "ok",
            "path": config_path,
            "model_kind": config.get("model_kind", ""),
            "architecture": config.get("architecture", ""),
            "model_name": config.get("model_name", ""),
            "n_layers": config.get("n_layers", config.get("num_layers", "")),
            "hidden_dim": config.get("hidden_dim", config.get("d_model", "")),
            "feature_dim": config.get("feature_dim", config.get("d_transcoder", "")),
            "feature_input_hook": config.get("feature_input_hook", ""),
            "feature_output_hook": config.get("feature_output_hook", ""),
            "file_pattern": config.get("file_pattern", ""),
            "layers": config.get("layers", ""),
            "raw_keys": sorted(str(k) for k in config.keys()),
        }
    except Exception as exc:  # noqa: BLE001
        config_error = f"{type(exc).__name__}:{exc}"
        payload["config"] = {"status": "failed", "error": config_error}
    _log(f"config status={payload['config'].get('status')}")

    remote_files: list[str] = []
    repo_error = ""
    try:
        _log("listing repository files")
        remote_files = api.list_repo_files(repo_id=repo_id, revision=revision)
        safetensors_files = sorted(x for x in remote_files if x.endswith(".safetensors"))
        enc_files = [x for x in safetensors_files if Path(x).name.startswith("W_enc_")]
        dec_files = [x for x in safetensors_files if Path(x).name.startswith("W_dec_")]
        layer_files = [x for x in safetensors_files if Path(x).name.startswith("layer_")]
        payload["repo_files"] = {
            "status": "ok",
            "total_files": len(remote_files),
            "safetensors_count": len(safetensors_files),
            "w_enc_count": len(enc_files),
            "w_dec_count": len(dec_files),
            "layer_file_count": len(layer_files),
            "first_safetensors_files": safetensors_files[:10],
        }
        files_ok = bool(safetensors_files)
    except Exception as exc:  # noqa: BLE001
        repo_error = f"{type(exc).__name__}:{exc}"
        safetensors_files = []
        enc_files = []
        dec_files = []
        layer_files = []
        payload["repo_files"] = {"status": "failed", "error": repo_error}
    _log(f"repo file status={payload['repo_files'].get('status')} safetensors={payload['repo_files'].get('safetensors_count', 0)}")

    local_snapshot = ""
    local_snapshot_status = "missing"
    if safetensors_files:
        try:
            _log("checking local safetensors snapshot cache only")
            local_snapshot = _try_cached_snapshot(repo_id, revision)
            local_snapshot_status = "ok"
        except Exception as exc:  # noqa: BLE001
            local_snapshot_status = f"failed:{type(exc).__name__}:{str(exc)[:500]}"
    payload["local_cache"] = {
        "safetensors_snapshot_status": local_snapshot_status,
        "safetensors_snapshot_path": local_snapshot,
    }
    local_safetensors = _local_safetensors_in_snapshot(local_snapshot)
    payload["local_cache"].update(
        {
            "local_safetensors_count": len(local_safetensors),
            "local_safetensors_files": local_safetensors[:10],
            "safetensors_snapshot_complete": bool(
                safetensors_files and len(local_safetensors) >= len(safetensors_files)
            ),
        }
    )

    expected_layers = _safe_int(payload.get("config", {}).get("n_layers"))
    indexed_files = enc_files or layer_files or safetensors_files
    observed_layers = sorted({idx for f in indexed_files if (idx := _layer_index(f)) is not None})
    missing_layers: list[int] = []
    extra_layers: list[int] = []
    if expected_layers is not None and observed_layers:
        expected = set(range(expected_layers))
        observed = set(observed_layers)
        missing_layers = sorted(expected - observed)
        extra_layers = sorted(observed - expected)
    payload["repo_files"].update(
        {
            "expected_layers": expected_layers if expected_layers is not None else "",
            "observed_indexed_layers": observed_layers,
            "missing_layers": missing_layers,
            "extra_layers": extra_layers,
        }
    )

    shape_rows: list[dict[str, Any]] = []
    download_budget = max(0, args.max_shape_downloads)
    _log(f"building shape CSV rows; max_shape_downloads={download_budget}")
    preferred_for_download = enc_files or layer_files or safetensors_files
    for filename in safetensors_files:
        row: dict[str, Any] = {
            "repo_id": repo_id,
            "filename": filename,
            "file_kind": (
                "W_enc"
                if Path(filename).name.startswith("W_enc_")
                else "W_dec"
                if Path(filename).name.startswith("W_dec_")
                else "layer"
                if Path(filename).name.startswith("layer_")
                else "other"
            ),
            "layer_index": "" if _layer_index(filename) is None else _layer_index(filename),
            "remote_size_bytes": "",
            "local_path": "",
            "shape_status": "not_inspected",
            "tensor_shapes": "",
            "shape_error": "",
        }
        size = _remote_file_size(api, repo_id, filename, revision)
        if size is not None:
            row["remote_size_bytes"] = size
        local_file = _try_cached_file(repo_id, filename, revision)
        if local_file is None and filename in preferred_for_download[:download_budget]:
            max_bytes = int(args.max_shape_download_mb * 1024 * 1024)
            if size is None:
                row["shape_status"] = "skipped_unknown_size"
                row["shape_error"] = "remote_size_bytes unavailable; refusing shape download in lightweight smoke"
            elif size > max_bytes:
                row["shape_status"] = "skipped_too_large"
                row["shape_error"] = f"remote_size_bytes {size} > max {max_bytes}"
            else:
                try:
                    local_file = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
                    download_budget -= 1
                except Exception as exc:  # noqa: BLE001
                    row["shape_status"] = "download_failed"
                    row["shape_error"] = f"{type(exc).__name__}:{str(exc)[:500]}"
        if local_file is not None:
            row["local_path"] = local_file
            try:
                row["tensor_shapes"], row["shape_error"] = _inspect_safetensor(local_file)
                row["shape_status"] = "ok"
            except Exception as exc:  # noqa: BLE001
                row["shape_status"] = "inspect_failed"
                row["shape_error"] = f"{type(exc).__name__}:{str(exc)[:500]}"
        shape_rows.append(row)

    lazy_status = "skipped"
    lazy_payload: dict[str, Any] = {"status": "skipped"}
    clt_like = bool(
        config_ok
        and (
            config.get("model_kind") == "cross_layer_transcoder"
            or config.get("architecture") == "cross_layer_transcoder"
            or str(config.get("file_pattern", "")).startswith("layer_")
        )
    )
    if clt_like:
        can_try_lazy = bool(
            args.allow_full_transcoder_download
            or payload["local_cache"].get("safetensors_snapshot_complete")
        )
        if can_try_lazy:
            _log("attempting lazy CLT load")
            old_offline = os.environ.get("HF_HUB_OFFLINE")
            if not args.allow_full_transcoder_download:
                os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

                transcoder, loaded_config = load_transcoder_from_hub(
                    repo_id,
                    device=torch.device("cpu"),
                    dtype=torch.bfloat16,
                    lazy_encoder=True,
                    lazy_decoder=True,
                )
                lazy_status = "ok"
                lazy_payload = {
                    "status": "ok",
                    "class": type(transcoder).__name__,
                    "n_layers": getattr(transcoder, "n_layers", ""),
                    "d_transcoder": getattr(transcoder, "d_transcoder", ""),
                    "d_model": getattr(transcoder, "d_model", ""),
                    "feature_input_hook": getattr(transcoder, "feature_input_hook", ""),
                    "feature_output_hook": getattr(transcoder, "feature_output_hook", ""),
                    "loaded_config_model_kind": loaded_config.get("model_kind", ""),
                }
            except Exception as exc:  # noqa: BLE001
                lazy_status = "failed"
                lazy_payload = {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1500],
                }
            finally:
                if not args.allow_full_transcoder_download:
                    if old_offline is None:
                        os.environ.pop("HF_HUB_OFFLINE", None)
                    else:
                        os.environ["HF_HUB_OFFLINE"] = old_offline
        else:
            lazy_status = "skipped_not_cached"
            lazy_payload = {
                "status": lazy_status,
                "reason": (
                    "Full transcoder safetensors snapshot is not cached and "
                    "--allow-full-transcoder-download is false."
                ),
                "local_safetensors_count": payload["local_cache"].get("local_safetensors_count"),
                "remote_safetensors_count": len(safetensors_files),
            }
    elif config_ok:
        lazy_status = "skipped_not_clt"
        lazy_payload = {
            "status": lazy_status,
            "reason": f"model_kind={config.get('model_kind')}",
        }
    payload["lazy_clt_load"] = lazy_payload

    model_name = str(config.get("model_name", "") or payload.get("config", {}).get("model_name", ""))
    if args.try_processor_local:
        payload["processor_local_smoke"] = _processor_local_smoke(model_name)
    else:
        payload["processor_local_smoke"] = {
            "attempted": False,
            "status": "skipped_by_default",
            "reason": "Base VLM processor/local model check is optional and does not download the full base model.",
        }

    files_complete = files_ok and not missing_layers
    final_status = _status_from_steps(config_ok, files_complete, lazy_status)
    payload["decision"] = {
        "status": final_status,
        "config_ok": config_ok,
        "files_ok": files_ok,
        "files_complete_for_expected_layers": files_complete,
        "lazy_clt_load_status": lazy_status,
        "replacement_model_gemma3_only": payload["replacement_model_backend"].get(
            "uses_gemma3_for_conditional_generation"
        ),
        "can_run_qwen_full_pipeline_now": False,
        "next_step": (
            "Write a Qwen adapter / HookedVLTransformer compatibility smoke before attribution."
            if final_status in {"pass", "partial"}
            else "Try backup Qwen2.5-VL PLT config/shape smoke or inspect HF/cache access."
        ),
        "claim_boundary": "This smoke does not replicate any mechanism result and does not complete a Qwen adapter.",
    }

    _write_json(Path(args.out_json).expanduser().resolve(), payload)
    _write_csv(
        Path(args.out_csv).expanduser().resolve(),
        shape_rows,
        [
            "repo_id",
            "filename",
            "file_kind",
            "layer_index",
            "remote_size_bytes",
            "local_path",
            "shape_status",
            "tensor_shapes",
            "shape_error",
        ],
    )
    print(
        f"[done] status={final_status} config_ok={config_ok} files_ok={files_ok} "
        f"lazy_status={lazy_status} out_json={args.out_json}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
