#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DEFAULT_CONDITIONS = ["clean", "answer_mask", "union_mask"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mask_array(mask_image: Image.Image) -> np.ndarray:
    return np.asarray(mask_image, dtype=np.uint8) > 0


def _load_mask(mask_root: Path, image_path: str, label: str) -> Image.Image:
    stem = Path(image_path).stem
    path = mask_root / stem / f"{label}.png"
    if not path.exists():
        raise FileNotFoundError(f"missing mask {path}")
    return Image.open(path).convert("L")


def _apply_mask(original: Image.Image, mask_bool: np.ndarray, fill_rgb: tuple[int, int, int]) -> Image.Image:
    out = np.asarray(original.convert("RGB")).copy()
    out[mask_bool] = np.array(fill_rgb, dtype=np.uint8)
    return Image.fromarray(out, mode="RGB")


def _parse_fill(fill: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in fill.split(",")]
    if len(parts) != 3:
        raise ValueError("--mask-fill must be r,g,b")
    return tuple(parts)


def _safe_int(value: object) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(str(value)))
    except Exception:
        return None


def _device_batch(model, batch: dict) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(model.cfg.device) if torch.is_tensor(value) else value
    return out


def _extract_answer(generated: str) -> str:
    txt = (generated or "").strip()
    if not txt:
        return ""
    match = re.search(r"the answer is\s*[:\-]?\s*(.+)", txt, flags=re.IGNORECASE | re.DOTALL)
    ans = match.group(1).strip() if match else txt
    ans = re.split(r"[\n\r]", ans)[0].strip()
    ans = re.split(r"[.!?]", ans)[0].strip()
    return ans.strip("\"'` ")


def _dedupe_manifest_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            (row.get("sample_id") or "").strip(),
            (row.get("run") or "").strip(),
            (row.get("node_role") or "").strip(),
            (row.get("node_source") or "").strip(),
        )
        if all(key) and key not in out:
            out[key] = row
    return list(out.values())


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test decoded generation under region masks with source-node zeroing.")
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--transcoder-set", default="tianhux2/gemma3-4b-it-plt")
    parser.add_argument("--model", default="")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--sample-ids", default="")
    parser.add_argument("--node-role", default="support", choices=["support", "suppressor"])
    parser.add_argument("--node-source", default="source", choices=["source", "nearest_control"])
    parser.add_argument("--mask-fill", default="128,128,128")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    from circuit_tracer import ReplacementModel
    from circuit_tracer.attribution.attribute import _build_multimodal_batch
    from huggingface_hub import hf_hub_download
    import yaml

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    if args.model.strip():
        model_name = args.model.strip()
    else:
        config_path = hf_hub_download(repo_id=args.transcoder_set, filename="config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model_name = (cfg or {}).get("model_name", "")
        if not model_name:
            raise ValueError(f"model_name missing in {args.transcoder_set}/config.yaml")

    print(f"[info] loading model={model_name} transcoder_set={args.transcoder_set} dtype={dtype}", flush=True)
    model = ReplacementModel.from_pretrained(model_name, args.transcoder_set, dtype=dtype, lazy_encoder=True, lazy_decoder=True)

    rows = _dedupe_manifest_rows(_read_csv(Path(args.manifest_csv).expanduser().resolve()))
    sample_ids = {x.strip() for x in args.sample_ids.split(",") if x.strip()}
    rows = [
        row
        for row in rows
        if row.get("node_role") == args.node_role
        and row.get("node_source") == args.node_source
        and (not sample_ids or row.get("sample_id") in sample_ids)
    ]
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    if not rows:
        raise ValueError("no rows selected")

    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    fill_rgb = _parse_fill(args.mask_fill)
    mask_root = Path(args.mask_root).expanduser().resolve()
    results: list[dict[str, str]] = []

    for row in rows:
        image_path = (row.get("image_path") or "").strip()
        original = Image.open(image_path).convert("RGB")
        answer_mask = _mask_array(_load_mask(mask_root, image_path, "answer"))
        relate_mask = _mask_array(_load_mask(mask_root, image_path, "relate"))
        union_mask = np.logical_or(answer_mask, relate_mask)
        masks = {
            "clean": None,
            "answer_mask": answer_mask,
            "relate_mask": relate_mask,
            "union_mask": union_mask,
        }
        layer = _safe_int(row.get("feature_layer"))
        pos = _safe_int(row.get("feature_pos"))
        feature_id = _safe_int(row.get("feature_id"))
        if layer is None or pos is None or feature_id is None:
            print(f"[skip] incomplete feature metadata for {row.get('sample_id')} {row.get('run')}", flush=True)
            continue
        interventions = [(layer, pos, feature_id, 0.0)]

        for condition in conditions:
            if condition not in masks:
                raise ValueError(f"unsupported condition: {condition}")
            mask = masks[condition]
            image = original if mask is None else _apply_mask(original, mask, fill_rgb)
            question = row.get("question", "")
            assistant_prefix = row.get("assistant_prefix", "")
            out = {
                "sample_id": row.get("sample_id", ""),
                "run": row.get("run", ""),
                "prompt_name": row.get("prompt_name", ""),
                "node_role": row.get("node_role", ""),
                "node_source": row.get("node_source", ""),
                "condition": condition,
                "feature_layer": str(layer),
                "feature_pos": str(pos),
                "feature_id": str(feature_id),
                "baseline_generation": "",
                "intervention_generation": "",
                "baseline_answer": "",
                "intervention_answer": "",
                "answer_changed_by_intervention": "",
                "error_message": "",
            }
            try:
                batch = _build_multimodal_batch(
                    model.processor,
                    image,
                    f"<start_of_image> {question}",
                    assistant_prefix=assistant_prefix,
                )
                batch["image"] = image
                batch = _device_batch(model, batch)
                seq_len = int(batch["input_ids"].shape[1])
                if pos >= seq_len:
                    raise ValueError(f"position_out_of_range pos={pos} seq_len={seq_len}")

                baseline = model.generate(
                    batch,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    verbose=False,
                    use_past_kv_cache=True,
                )
                intervened, _, _ = model.feature_intervention_generate(
                    batch,
                    interventions,
                    freeze_attention=True,
                    apply_activation_function=True,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    verbose=False,
                    use_past_kv_cache=True,
                )
                out["baseline_generation"] = str(baseline)
                out["intervention_generation"] = str(intervened)
                out["baseline_answer"] = _extract_answer(str(baseline))
                out["intervention_answer"] = _extract_answer(str(intervened))
                out["answer_changed_by_intervention"] = str(out["baseline_answer"] != out["intervention_answer"])
            except Exception as exc:  # noqa: BLE001
                out["error_message"] = f"{type(exc).__name__}:{exc}\n{traceback.format_exc()}"
            results.append(out)
            print(
                f"[done] sample={row.get('sample_id','')} run={row.get('run','')} "
                f"condition={condition} changed={out['answer_changed_by_intervention']} err={out['error_message']}",
                flush=True,
            )

    fieldnames = [
        "sample_id",
        "run",
        "prompt_name",
        "node_role",
        "node_source",
        "condition",
        "feature_layer",
        "feature_pos",
        "feature_id",
        "baseline_generation",
        "intervention_generation",
        "baseline_answer",
        "intervention_answer",
        "answer_changed_by_intervention",
        "error_message",
    ]
    _write_csv(Path(args.out_csv).expanduser().resolve(), results, fieldnames)
    print(f"[done] out_csv={args.out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
