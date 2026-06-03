#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_LAYER_RE = re.compile(r"layer_(\d+)\.safetensors$")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage2f-download] {message}", flush=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def _layer_index(name: str) -> int | None:
    match = _LAYER_RE.search(Path(name).name)
    return int(match.group(1)) if match else None


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _inspect_safetensor(path: Path) -> tuple[str, str, str]:
    from safetensors import safe_open

    tensor_bits: list[str] = []
    metadata: dict[str, str] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        for key in f.keys():
            shape: list[int] | str = ""
            dtype = ""
            try:
                sl = f.get_slice(key)
                shape = list(sl.get_shape())
                get_dtype = getattr(sl, "get_dtype", None)
                if callable(get_dtype):
                    dtype = str(get_dtype())
            except Exception as exc:  # noqa: BLE001
                shape = f"shape_error:{type(exc).__name__}:{str(exc)[:120]}"
            tensor_bits.append(f"{key}:shape={shape}:dtype={dtype}")
    return "|".join(tensor_bits), json.dumps(metadata, ensure_ascii=False, sort_keys=True), ""


def _remote_sizes(api, repo_id: str, filenames: list[str], revision: str | None) -> dict[str, int | None]:
    sizes: dict[str, int | None] = {}
    for name in filenames:
        try:
            info = api.get_paths_info(repo_id=repo_id, paths=[name], revision=revision)
            sizes[name] = getattr(info[0], "size", None) if info else None
        except Exception:
            sizes[name] = None
    return sizes


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2F-2 full Qwen CLT download and lazy-load smoke.")
    parser.add_argument("--repo-id", default="KokosDev/qwen2p5vl-7b-clt")
    parser.add_argument("--revision", default="")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    parser.add_argument("--inspect-layers", default="0,13,26")
    parser.add_argument("--skip-sha256", action="store_true")
    parser.add_argument("--skip-lazy-load", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    repo_id = args.repo_id
    revision = args.revision.strip() or None
    api = HfApi()
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    inspect_layers = {
        int(x.strip()) for x in args.inspect_layers.split(",") if x.strip()
    }

    payload: dict[str, Any] = {
        "created_at": _now(),
        "repo_id": repo_id,
        "revision": revision or "main",
        "env_presence": _env_presence(),
        "disk_before": _disk_usage(hf_home),
        "download": {},
        "config": {},
        "lazy_load": {},
        "decision": {},
    }

    if payload["disk_before"]["free_gb"] < args.min_free_gb:
        payload["download"] = {
            "status": "skipped_low_disk",
            "min_free_gb": args.min_free_gb,
        }
        payload["decision"] = {
            "status": "blocked",
            "reason": "Not enough free disk space for CLT download.",
        }
        _write_json(Path(args.out_json), payload)
        _write_csv(Path(args.out_csv), [], ["repo_id", "filename"])
        _log("blocked: low disk")
        return 0

    _log(f"listing files for {repo_id}")
    remote_files = api.list_repo_files(repo_id=repo_id, revision=revision)
    layer_files = sorted(x for x in remote_files if _LAYER_RE.search(Path(x).name))
    remote_size_map = _remote_sizes(api, repo_id, layer_files, revision)
    payload["download"]["remote_layer_count"] = len(layer_files)
    payload["download"]["remote_total_layer_bytes"] = sum(v or 0 for v in remote_size_map.values())

    _log("downloading config.yaml and all layer_*.safetensors")
    download_status = "ok"
    snapshot_path = ""
    error = ""
    try:
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=["config.yaml", "layer_*.safetensors"],
            max_workers=8,
        )
    except Exception as exc:  # noqa: BLE001
        download_status = "failed"
        error = f"{type(exc).__name__}:{str(exc)[:2000]}"

    payload["download"].update(
        {
            "status": download_status,
            "snapshot_path": snapshot_path,
            "error": error,
        }
    )

    rows: list[dict[str, Any]] = []
    config_ok = False
    files_complete = False
    inspected_ok = False
    if snapshot_path:
        root = Path(snapshot_path)
        config_path = root / "config.yaml"
        if config_path.exists():
            try:
                config = _load_yaml(config_path)
                config_ok = True
                payload["config"] = {
                    "status": "ok",
                    "model_kind": config.get("model_kind", ""),
                    "architecture": config.get("architecture", ""),
                    "model_name": config.get("model_name", ""),
                    "n_layers": config.get("n_layers", ""),
                    "hidden_dim": config.get("hidden_dim", ""),
                    "feature_dim": config.get("feature_dim", ""),
                    "feature_input_hook": config.get("feature_input_hook", ""),
                    "feature_output_hook": config.get("feature_output_hook", ""),
                    "file_pattern": config.get("file_pattern", ""),
                }
            except Exception as exc:  # noqa: BLE001
                payload["config"] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}:{str(exc)[:1000]}",
                }
        else:
            payload["config"] = {"status": "missing"}

        local_layers = sorted(root.glob("layer_*.safetensors"), key=lambda p: _layer_index(p.name) or -1)
        observed = sorted(x for x in (_layer_index(p.name) for p in local_layers) if x is not None)
        expected_n = int(payload.get("config", {}).get("n_layers") or len(layer_files) or 0)
        missing = sorted(set(range(expected_n)) - set(observed)) if expected_n else []
        files_complete = bool(expected_n and not missing and len(local_layers) >= expected_n)
        selected_ok: list[bool] = []
        for path in local_layers:
            idx = _layer_index(path.name)
            remote_size = remote_size_map.get(path.name, remote_size_map.get(str(path.name)))
            local_size = path.stat().st_size
            inspect = idx in inspect_layers
            tensor_shapes = ""
            tensor_metadata = ""
            inspect_status = "not_selected"
            inspect_error = ""
            if inspect:
                try:
                    tensor_shapes, tensor_metadata, inspect_error = _inspect_safetensor(path)
                    inspect_status = "ok"
                    selected_ok.append(True)
                except Exception as exc:  # noqa: BLE001
                    inspect_status = "failed"
                    inspect_error = f"{type(exc).__name__}:{str(exc)[:1000]}"
                    selected_ok.append(False)
            checksum = ""
            if not args.skip_sha256:
                checksum = _sha256(path)
            rows.append(
                {
                    "repo_id": repo_id,
                    "filename": path.name,
                    "layer_index": idx if idx is not None else "",
                    "remote_size_bytes": remote_size if remote_size is not None else "",
                    "local_size_bytes": local_size,
                    "size_match": "" if remote_size is None else str(local_size == remote_size),
                    "local_path": str(path),
                    "sha256": checksum,
                    "inspect_status": inspect_status,
                    "tensor_shapes": tensor_shapes,
                    "tensor_metadata": tensor_metadata,
                    "inspect_error": inspect_error,
                }
            )
        inspected_ok = bool(selected_ok) and all(selected_ok)
        payload["download"].update(
            {
                "local_layer_count": len(local_layers),
                "observed_layers": observed,
                "expected_layers": expected_n,
                "missing_layers": missing,
                "files_complete": files_complete,
                "inspected_layers": sorted(inspect_layers),
                "inspected_ok": inspected_ok,
            }
        )

    lazy_status = "skipped"
    lazy_payload: dict[str, Any] = {"status": "skipped"}
    if snapshot_path and not args.skip_lazy_load:
        _log("attempting load_transcoder_from_hub lazy load")
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
                "error": str(exc)[:2000],
            }
    payload["lazy_load"] = lazy_payload
    payload["disk_after"] = _disk_usage(hf_home)

    if not snapshot_path or not files_complete:
        final_status = "blocked"
    elif lazy_status == "ok" and inspected_ok:
        final_status = "pass"
    else:
        final_status = "partial"
    payload["decision"] = {
        "status": final_status,
        "config_ok": config_ok,
        "files_complete": files_complete,
        "inspected_ok": inspected_ok,
        "lazy_load_status": lazy_status,
        "claim_boundary": "Downloaded CLT smoke only; not a cross-model mechanism replication.",
    }

    _write_json(Path(args.out_json), payload)
    _write_csv(
        Path(args.out_csv),
        rows,
        [
            "repo_id",
            "filename",
            "layer_index",
            "remote_size_bytes",
            "local_size_bytes",
            "size_match",
            "local_path",
            "sha256",
            "inspect_status",
            "tensor_shapes",
            "tensor_metadata",
            "inspect_error",
        ],
    )
    _log(f"done status={final_status} layers={len(rows)} lazy={lazy_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
