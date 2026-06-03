#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
OUT = ROOT / "doc" / "experiments" / "stage3" / "cross_model" / "stage3_qwen35_base_feasibility.json"


def main() -> int:
    base_model = "Qwen/Qwen3.5-4B"
    payload: dict[str, Any] = {
        "base_model": base_model,
        "claim_boundary": "Base/config feasibility only; no weights are loaded.",
        "repo_status": "unknown",
        "config_status": "not_attempted",
        "processor_status": "not_attempted",
        "is_vlm_candidate": "unknown",
        "notes": [],
    }
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(base_model)
        siblings = sorted(item.rfilename for item in info.siblings)
        payload["repo_status"] = "pass"
        payload["repo_file_count"] = len(siblings)
        payload["has_image_processor"] = any("image_processor" in name for name in siblings)
        payload["has_processor_config"] = "processor_config.json" in siblings
        payload["has_preprocessor_config"] = "preprocessor_config.json" in siblings
        payload["sample_files"] = siblings[:30]
    except Exception as exc:  # noqa: BLE001
        payload["repo_status"] = "blocked"
        payload["repo_error"] = f"{type(exc).__name__}: {exc}"

    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        payload["config_status"] = "pass"
        payload["model_type"] = getattr(config, "model_type", "")
        payload["architectures"] = getattr(config, "architectures", [])
        payload["vision_config_present"] = bool(getattr(config, "vision_config", None))
        payload["image_token_id_present"] = hasattr(config, "image_token_id") or hasattr(config, "image_token_index")
    except Exception as exc:  # noqa: BLE001
        payload["config_status"] = "blocked"
        payload["config_error"] = f"{type(exc).__name__}: {exc}"

    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        payload["processor_status"] = "pass"
        payload["processor_type"] = type(processor).__name__
        payload["processor_has_image_processor"] = bool(getattr(processor, "image_processor", None))
    except Exception as exc:  # noqa: BLE001
        payload["processor_status"] = "blocked"
        payload["processor_error"] = f"{type(exc).__name__}: {exc}"

    vlm_signals = [
        bool(payload.get("has_image_processor")),
        bool(payload.get("has_processor_config")),
        bool(payload.get("has_preprocessor_config")),
        bool(payload.get("vision_config_present")),
        bool(payload.get("image_token_id_present")),
        bool(payload.get("processor_has_image_processor")),
    ]
    payload["is_vlm_candidate"] = "yes" if any(vlm_signals) else "no"
    if payload["is_vlm_candidate"] == "no":
        payload["notes"].append("No processor/vision config signal found without loading weights.")
        payload["stage3_status"] = "blocked_for_vlm_mainline"
    else:
        payload["stage3_status"] = "partial_vlm_candidate_needs_forward"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
