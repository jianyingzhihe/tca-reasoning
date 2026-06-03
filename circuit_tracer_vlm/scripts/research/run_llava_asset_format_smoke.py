#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[stage2f-llava-format] {message}", flush=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


def _file_size(path: str | Path) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def _summarize_object(obj: Any, max_items: int = 40) -> dict[str, Any]:
    if torch.is_tensor(obj):
        return {
            "type": "tensor",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "numel": int(obj.numel()),
        }
    if isinstance(obj, dict):
        items = []
        for index, (key, value) in enumerate(obj.items()):
            if index >= max_items:
                break
            items.append({"key": str(key), "value_summary": _summarize_object(value, max_items=8)})
        return {
            "type": "dict",
            "len": len(obj),
            "items": items,
        }
    if isinstance(obj, (list, tuple)):
        return {
            "type": type(obj).__name__,
            "len": len(obj),
            "items": [_summarize_object(value, max_items=8) for value in obj[: min(len(obj), 8)]],
        }
    return {"type": type(obj).__name__, "repr": repr(obj)[:500]}


def _pick_file(files: list[str], prefixes: list[str], layer: int) -> str | None:
    layer_strings = [f"L{layer}", f"_{layer}", f"layer_{layer}", f"{layer}"]
    candidates = []
    for file_name in files:
        base = Path(file_name).name
        if not any(base.startswith(prefix) for prefix in prefixes):
            continue
        if not base.endswith((".pt", ".pth", ".bin")):
            continue
        if any(token in base for token in layer_strings):
            candidates.append(file_name)
    if candidates:
        return sorted(candidates)[0]

    fallback = [
        file_name
        for file_name in files
        if Path(file_name).name.startswith(tuple(prefixes))
        and Path(file_name).suffix in {".pt", ".pth", ".bin"}
    ]
    return sorted(fallback)[0] if fallback else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2F LLaVA CLT asset format smoke.")
    parser.add_argument("--repo-id", default="KokosDev/llava15-7b-clt")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--download-sample", action="store_true")
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "created_at": _now(),
        "repo_id": args.repo_id,
        "layer_index": args.layer_index,
        "env_presence": _env_presence(),
        "repo": {},
        "selected_files": {},
        "download": {},
        "torch_load": {},
        "decision": {},
    }

    try:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        info = api.model_info(args.repo_id, files_metadata=True)
        siblings = info.siblings or []
        files = [s.rfilename for s in siblings]
        size_by_file = {
            s.rfilename: getattr(s, "size", None)
            for s in siblings
            if getattr(s, "rfilename", None)
        }
        payload["repo"] = {
            "status": "ok",
            "file_count": len(files),
            "files": files[:200],
            "pt_file_count": sum(1 for f in files if f.endswith((".pt", ".pth"))),
            "has_config_yaml": "config.yaml" in files,
            "has_readme": any(Path(f).name.lower() == "readme.md" for f in files),
        }
    except Exception as exc:  # noqa: BLE001
        payload["repo"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:3000],
        }
        payload["decision"] = {"status": "blocked", "reason": "repo_file_list_failed"}
        _write_json(Path(args.out_json), payload)
        _log("blocked: file list failed")
        return 0

    transcoder_file = _pick_file(files, ["transcoder", "clt", "sae"], args.layer_index)
    mapping_file = _pick_file(files, ["mapping", "map"], args.layer_index)
    payload["selected_files"] = {
        "transcoder_file": transcoder_file,
        "transcoder_size": size_by_file.get(transcoder_file) if transcoder_file else None,
        "mapping_file": mapping_file,
        "mapping_size": size_by_file.get(mapping_file) if mapping_file else None,
    }

    if not transcoder_file:
        payload["decision"] = {
            "status": "partial",
            "reason": "repo_readable_but_no_transcoder_pt_file_selected",
            "claim_boundary": "LLaVA asset survey only; not loader/intervention replication.",
        }
        _write_json(Path(args.out_json), payload)
        _log("partial: no transcoder file selected")
        return 0

    if not args.download_sample:
        payload["decision"] = {
            "status": "partial_format_list_only",
            "reason": "file_list_readable_sample_download_not_requested",
            "claim_boundary": "LLaVA asset file-list smoke only; tensor format not inspected yet.",
        }
        _write_json(Path(args.out_json), payload)
        _log("partial: list only")
        return 0

    downloaded: dict[str, Any] = {}
    try:
        for label, filename in [("transcoder", transcoder_file), ("mapping", mapping_file)]:
            if not filename:
                continue
            _log(f"downloading {label}: {filename}")
            local_path = hf_hub_download(args.repo_id, filename=filename)
            downloaded[label] = {
                "filename": filename,
                "local_path": local_path,
                "local_size_bytes": _file_size(local_path),
            }
        payload["download"] = {"status": "ok", **downloaded}
    except Exception as exc:  # noqa: BLE001
        payload["download"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:3000],
            **downloaded,
        }
        payload["decision"] = {
            "status": "partial",
            "reason": "repo_readable_but_sample_download_failed",
        }
        _write_json(Path(args.out_json), payload)
        _log("partial: download failed")
        return 0

    load_results: dict[str, Any] = {}
    all_loaded = True
    for label, item in downloaded.items():
        local_path = item["local_path"]
        try:
            _log(f"torch.load {label}")
            try:
                obj = torch.load(local_path, map_location="cpu", weights_only=True)
            except TypeError:
                obj = torch.load(local_path, map_location="cpu")
            load_results[label] = {
                "status": "ok",
                "summary": _summarize_object(obj),
            }
        except Exception as exc:  # noqa: BLE001
            all_loaded = False
            load_results[label] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:3000],
            }
    payload["torch_load"] = load_results

    if all_loaded and load_results:
        status = "pass_format_readable"
        reason = "sample_pt_files_downloaded_and_torch_loaded"
    else:
        status = "partial_format_readable"
        reason = "some_sample_pt_files_failed_torch_load"
    payload["decision"] = {
        "status": status,
        "reason": reason,
        "claim_boundary": "LLaVA asset format smoke only; not native forward, feature readout, intervention, or cross-model replication.",
    }
    _write_json(Path(args.out_json), payload)
    _log(f"done status={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
