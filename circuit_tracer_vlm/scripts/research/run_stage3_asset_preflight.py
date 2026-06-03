#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


ASSETS = [
    {
        "asset_id": "gemma3_plt",
        "base_model": "google/gemma-3-4b-it",
        "transcoder_ref": "tianhux2/gemma3-4b-it-plt",
        "asset_axis": "PLT",
        "expected_format": "config_yaml_layer_safetensors",
        "expected_vlm": True,
    },
    {
        "asset_id": "qwen2p5vl_plt",
        "base_model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "transcoder_ref": "KokosDev/qwen2p5vl-7b-plt",
        "asset_axis": "PLT",
        "expected_format": "config_yaml_layer_safetensors",
        "expected_vlm": True,
    },
    {
        "asset_id": "qwen35_plt",
        "base_model": "Qwen/Qwen3.5-4B",
        "transcoder_ref": "KokosDev/qwen35-4b-plt",
        "asset_axis": "PLT",
        "expected_format": "custom_pt",
        "expected_vlm": "unknown_high_risk",
    },
    {
        "asset_id": "qwen2p5vl_clt",
        "base_model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "transcoder_ref": "KokosDev/qwen2p5vl-7b-clt",
        "asset_axis": "CLT",
        "expected_format": "config_yaml_layer_safetensors",
        "expected_vlm": True,
    },
    {
        "asset_id": "llava15_clt",
        "base_model": "llava-hf/llava-1.5-7b-hf",
        "transcoder_ref": "KokosDev/llava15-7b-clt",
        "asset_axis": "CLT",
        "expected_format": "custom_pt_mapping",
        "expected_vlm": True,
    },
]


def _env_presence() -> dict[str, bool]:
    return {
        key: bool(os.environ.get(key))
        for key in [
            "HF_HOME",
            "HF_ENDPOINT",
            "HUGGINGFACE_HUB_CACHE",
            "http_proxy",
            "https_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
        ]
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _summarize_repo(asset: dict[str, Any], *, local_files_only: bool) -> dict[str, Any]:
    row: dict[str, Any] = dict(asset)
    row.update(
        {
            "config_observed": False,
            "model_kind": "",
            "feature_input_hook": "",
            "feature_output_hook": "",
            "layer_file_count": 0,
            "custom_pt_count": 0,
            "mapping_pt_count": 0,
            "missing_layer_note": "",
            "loader_status": "not_attempted",
            "preflight_status": "unknown",
            "preflight_note": "",
        }
    )
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as exc:  # noqa: BLE001
        row["preflight_status"] = "blocked_dependency"
        row["preflight_note"] = f"huggingface_hub unavailable: {type(exc).__name__}: {exc}"
        return row

    repo_id = asset["transcoder_ref"]
    try:
        info = HfApi().model_info(repo_id)
        siblings = [item.rfilename for item in info.siblings]
    except Exception as exc:  # noqa: BLE001
        row["preflight_status"] = "blocked_repo_metadata"
        row["preflight_note"] = f"{type(exc).__name__}: {exc}"
        return row

    layer_files = sorted(name for name in siblings if name.startswith("layer_") and name.endswith(".safetensors"))
    custom_pts = sorted(name for name in siblings if name.startswith("transcoder_L") and name.endswith(".pt"))
    mapping_pts = sorted(name for name in siblings if name.startswith("mapping_L") and name.endswith(".pt"))
    row["layer_file_count"] = len(layer_files)
    row["custom_pt_count"] = len(custom_pts)
    row["mapping_pt_count"] = len(mapping_pts)

    if layer_files:
        try:
            layer_nums = sorted(int(Path(name).stem.split("_")[1]) for name in layer_files)
            expected = set(range(max(layer_nums) + 1))
            missing = sorted(expected - set(layer_nums))
            row["missing_layer_note"] = ",".join(str(x) for x in missing)
        except Exception:
            row["missing_layer_note"] = "could_not_parse_layer_numbers"
    elif custom_pts:
        try:
            layer_nums = sorted(int(Path(name).stem.split("_L")[1]) for name in custom_pts)
            expected = set(range(max(layer_nums) + 1))
            missing = sorted(expected - set(layer_nums))
            row["missing_layer_note"] = ",".join(str(x) for x in missing)
        except Exception:
            row["missing_layer_note"] = "could_not_parse_custom_pt_layers"

    if "config.yaml" in siblings:
        row["config_observed"] = True
        try:
            import yaml

            config_path = hf_hub_download(
                repo_id=repo_id,
                filename="config.yaml",
                local_files_only=local_files_only,
            )
            config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
            row["model_kind"] = config.get("model_kind", "")
            row["feature_input_hook"] = config.get("feature_input_hook", "")
            row["feature_output_hook"] = config.get("feature_output_hook", "")
        except Exception as exc:  # noqa: BLE001
            row["preflight_note"] = f"config_read_failed: {type(exc).__name__}: {exc}"

    if asset["asset_id"] == "qwen35_plt":
        if row["missing_layer_note"]:
            row["preflight_status"] = "partial_missing_layer_or_custom_loader"
            row["preflight_note"] = "custom .pt PLT with missing layers; base VLM capability must be checked"
        else:
            row["preflight_status"] = "partial_custom_loader"
            row["preflight_note"] = "custom .pt PLT; needs dedicated loader and VLM forward smoke"
    elif asset["asset_id"] == "llava15_clt":
        row["preflight_status"] = "partial_custom_clt"
        row["preflight_note"] = "custom .pt + mapping format; keep as CLT auxiliary"
    elif row["config_observed"] and row["layer_file_count"] > 0:
        row["preflight_status"] = "asset_format_pass"
        row["preflight_note"] = "config + per-layer safetensors observed"
    else:
        row["preflight_status"] = "blocked_unexpected_format"
        row["preflight_note"] = "expected config/layer files not observed"

    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage3 asset preflight for PLT/CLT dual-track experiments.")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    rows = [_summarize_repo(asset, local_files_only=args.local_files_only) for asset in ASSETS]
    payload = {
        "claim_boundary": "Asset preflight only. It does not establish cross-model mechanism replication.",
        "env_presence": _env_presence(),
        "rows": rows,
        "status_counts": {status: sum(row["preflight_status"] == status for row in rows) for status in sorted({row["preflight_status"] for row in rows})},
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(
        args.out_csv,
        rows,
        [
            "asset_id",
            "base_model",
            "transcoder_ref",
            "asset_axis",
            "expected_format",
            "expected_vlm",
            "config_observed",
            "model_kind",
            "feature_input_hook",
            "feature_output_hook",
            "layer_file_count",
            "custom_pt_count",
            "mapping_pt_count",
            "missing_layer_note",
            "preflight_status",
            "preflight_note",
        ],
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

