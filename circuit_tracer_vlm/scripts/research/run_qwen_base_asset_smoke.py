#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage2f-base] {message}", flush=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _disk_usage(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser()
    probe = p if p.exists() else p.parent
    usage = shutil.disk_usage(probe)
    return {
        "path": str(probe),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "free_gb": round(usage.free / (1024**3), 3),
    }


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


def _count_snapshot_files(snapshot_path: str) -> dict[str, Any]:
    if not snapshot_path:
        return {"exists": False, "file_count": 0, "total_bytes": 0}
    root = Path(snapshot_path)
    files = [p for p in root.rglob("*") if p.is_file()]
    weight_files = [p for p in files if p.suffix in {".safetensors", ".bin"}]
    return {
        "exists": root.exists(),
        "file_count": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "weight_file_count": len(weight_files),
        "weight_total_bytes": sum(p.stat().st_size for p in weight_files),
        "first_weight_files": [str(p.relative_to(root)).replace("\\", "/") for p in weight_files[:10]],
    }


def _processor_smoke(model_name: str) -> dict[str, Any]:
    try:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(model_name, local_files_only=True)
        return {
            "status": "ok",
            "processor_class": type(proc).__name__,
            "has_tokenizer": hasattr(proc, "tokenizer"),
            "tokenizer_class": type(getattr(proc, "tokenizer", None)).__name__,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
        }


def _config_and_meta_model_smoke(model_name: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, local_files_only=True)
        result["config"] = {
            "status": "ok",
            "model_type": getattr(config, "model_type", ""),
            "architectures": getattr(config, "architectures", ""),
            "hidden_size": getattr(config, "hidden_size", ""),
            "num_hidden_layers": getattr(config, "num_hidden_layers", ""),
            "vocab_size": getattr(config, "vocab_size", ""),
        }
    except Exception as exc:  # noqa: BLE001
        result["config"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
        }
        return result

    try:
        from accelerate import init_empty_weights
        from transformers import Qwen2_5_VLForConditionalGeneration

        with init_empty_weights():
            model = Qwen2_5_VLForConditionalGeneration(config)
        result["meta_model"] = {
            "status": "ok",
            "class": type(model).__name__,
            "parameter_count_meta": sum(p.numel() for p in model.parameters()),
        }
    except Exception as exc:  # noqa: BLE001
        result["meta_model"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2F-2 Qwen2.5-VL base asset download smoke.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--revision", default="")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--min-free-gb-for-base", type=float, default=40.0)
    parser.add_argument("--skip-full-base", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    model_name = args.model_name
    revision = args.revision.strip() or None
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    payload: dict[str, Any] = {
        "created_at": _now(),
        "model_name": model_name,
        "revision": revision or "main",
        "input_is_local_path": Path(model_name).expanduser().exists(),
        "env_presence": _env_presence(),
        "disk_before": _disk_usage(hf_home),
        "processor_snapshot": {},
        "full_snapshot": {},
        "processor_local_smoke": {},
        "model_local_smoke": {},
        "decision": {},
    }

    local_model_path = Path(model_name).expanduser()
    if local_model_path.exists():
        snapshot_path = str(local_model_path.resolve())
        snapshot_counts = _count_snapshot_files(snapshot_path)
        payload["processor_snapshot"] = {
            "status": "ok_local_path",
            "snapshot_path": snapshot_path,
            **snapshot_counts,
        }
        payload["processor_local_smoke"] = _processor_smoke(snapshot_path)
        full_status = "ok" if snapshot_counts.get("weight_file_count", 0) >= 5 else "failed_incomplete_local_path"
        payload["full_snapshot"] = {
            "status": full_status,
            "snapshot_path": snapshot_path,
            **snapshot_counts,
        }
        payload["model_local_smoke"] = _config_and_meta_model_smoke(snapshot_path)
        payload["disk_after"] = _disk_usage(hf_home)

        processor_ok = payload["processor_local_smoke"].get("status") == "ok"
        full_ok = payload["full_snapshot"].get("status") == "ok"
        meta_ok = payload["model_local_smoke"].get("meta_model", {}).get("status") == "ok"
        if processor_ok and full_ok and meta_ok:
            status = "pass"
        elif processor_ok and full_ok:
            status = "partial"
        else:
            status = "blocked"
        payload["decision"] = {
            "status": status,
            "processor_ok": processor_ok,
            "full_base_assets_ok": full_ok,
            "meta_model_ok": meta_ok,
            "claim_boundary": "Base asset smoke only; not attribution, intervention, or cross-model replication.",
        }
        _write_json(Path(args.out_json), payload)
        _log(f"done status={status} local_path=True processor={processor_ok} full={full_ok} meta={meta_ok}")
        return 0

    small_patterns = [
        "*.json",
        "*.txt",
        "*.model",
        "*.tiktoken",
        "*.jinja",
        "*.py",
        "README*",
        "LICENSE*",
    ]
    _log("downloading processor/tokenizer/config assets")
    try:
        small_snapshot = snapshot_download(
            repo_id=model_name,
            revision=revision,
            allow_patterns=small_patterns,
            max_workers=8,
        )
        payload["processor_snapshot"] = {
            "status": "ok",
            "snapshot_path": small_snapshot,
            **_count_snapshot_files(small_snapshot),
        }
    except Exception as exc:  # noqa: BLE001
        payload["processor_snapshot"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
        }

    payload["processor_local_smoke"] = _processor_smoke(model_name)

    free_gb = payload["disk_before"]["free_gb"]
    if args.skip_full_base or free_gb < args.min_free_gb_for_base:
        payload["full_snapshot"] = {
            "status": "skipped_low_disk_or_flag",
            "free_gb": free_gb,
            "min_free_gb_for_base": args.min_free_gb_for_base,
            "skip_full_base": args.skip_full_base,
        }
    else:
        _log("downloading full base model assets")
        full_patterns = small_patterns + ["*.safetensors", "*.bin", "*.index.json"]
        try:
            full_snapshot = snapshot_download(
                repo_id=model_name,
                revision=revision,
                allow_patterns=full_patterns,
                max_workers=8,
            )
            payload["full_snapshot"] = {
                "status": "ok",
                "snapshot_path": full_snapshot,
                **_count_snapshot_files(full_snapshot),
            }
        except Exception as exc:  # noqa: BLE001
            payload["full_snapshot"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
            }

    payload["model_local_smoke"] = _config_and_meta_model_smoke(model_name)
    payload["disk_after"] = _disk_usage(hf_home)

    processor_ok = payload["processor_local_smoke"].get("status") == "ok"
    full_ok = payload["full_snapshot"].get("status") == "ok"
    meta_ok = payload["model_local_smoke"].get("meta_model", {}).get("status") == "ok"
    if processor_ok and full_ok and meta_ok:
        status = "pass"
    elif processor_ok and (full_ok or payload["full_snapshot"].get("status") == "skipped_low_disk_or_flag"):
        status = "partial"
    else:
        status = "blocked"
    payload["decision"] = {
        "status": status,
        "processor_ok": processor_ok,
        "full_base_assets_ok": full_ok,
        "meta_model_ok": meta_ok,
        "claim_boundary": "Base asset smoke only; not attribution, intervention, or cross-model replication.",
    }
    _write_json(Path(args.out_json), payload)
    _log(f"done status={status} processor={processor_ok} full={full_ok} meta={meta_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
