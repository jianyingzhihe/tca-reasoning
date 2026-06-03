#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def _safe_int(value: object) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(str(value)))
    except Exception:
        return None


def _parse_fill(fill: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in fill.split(",")]
    if len(parts) != 3:
        raise ValueError("--mask-fill must be r,g,b")
    return tuple(parts)


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


def _device_batch(model, batch: dict) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(model.cfg.device) if torch.is_tensor(value) else value
    return out


def _token_rank(logits_1d: torch.Tensor, token_id: int) -> int:
    target_logit = logits_1d[token_id]
    return int((logits_1d > target_logit).sum().item()) + 1


def _decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id])
    except Exception:
        return tokenizer.convert_ids_to_tokens([token_id])[0]


def _first_token_stats(logits: torch.Tensor, target_token_id: int, tokenizer) -> dict[str, str]:
    next_logits = logits[0, -1, :]
    probs = torch.softmax(next_logits.float(), dim=-1)
    top1_id = int(torch.argmax(next_logits).item())
    target_logit = float(next_logits[target_token_id].item())
    target_prob = float(probs[target_token_id].item())
    return {
        "target_logit": f"{target_logit:.10g}",
        "target_prob": f"{target_prob:.10g}",
        "target_rank": str(_token_rank(next_logits, target_token_id)),
        "top1_id": str(top1_id),
        "top1_token": _decode_token(tokenizer, top1_id),
        "target_token": _decode_token(tokenizer, target_token_id),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2B smoke: compare first answer token distribution under source/nearest "
            "feature zeroing for multimodal region conditions."
        )
    )
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--transcoder-set", default="tianhux2/gemma3-4b-it-plt")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16", "fp32", "bf16", "fp16"])
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--node-source", default="", choices=["", "source", "nearest_control"])
    parser.add_argument("--node-role", default="support", choices=["support", "suppressor"])
    parser.add_argument("--mask-fill", default="128,128,128")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    from circuit_tracer import ReplacementModel
    from circuit_tracer.attribution.attribute import _build_multimodal_batch
    from huggingface_hub import hf_hub_download
    import yaml

    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    dtype = dtype_map[args.dtype]

    if args.model_name.strip():
        model_name = args.model_name.strip()
    else:
        config_path = hf_hub_download(repo_id=args.transcoder_set, filename="config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model_name = (cfg or {}).get("model_name", "")
        if not model_name:
            raise ValueError(f"model_name missing in {args.transcoder_set}/config.yaml")

    rows = _read_csv(Path(args.manifest_csv).expanduser().resolve())
    rows = [row for row in rows if row.get("node_role") == args.node_role]
    if args.node_source:
        rows = [row for row in rows if row.get("node_source") == args.node_source]
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    if not rows:
        raise ValueError("no rows selected")

    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    fill_rgb = _parse_fill(args.mask_fill)
    mask_root = Path(args.mask_root).expanduser().resolve()

    print(f"[info] loading model={model_name} transcoder_set={args.transcoder_set} dtype={dtype}", flush=True)
    model = ReplacementModel.from_pretrained(model_name, args.transcoder_set, dtype=dtype, lazy_encoder=True, lazy_decoder=True)
    tokenizer = model.processor.tokenizer

    results: list[dict[str, str]] = []

    for row in rows:
        image_path = (row.get("image_path") or "").strip()
        original = Image.open(image_path).convert("RGB")
        answer_mask = _mask_array(_load_mask(mask_root, image_path, "answer"))
        relate_mask = _mask_array(_load_mask(mask_root, image_path, "relate"))
        union_mask = np.logical_or(answer_mask, relate_mask)
        condition_masks = {
            "clean": None,
            "answer_mask": answer_mask,
            "relate_mask": relate_mask,
            "union_mask": union_mask,
        }

        layer = _safe_int(row.get("feature_layer"))
        pos = _safe_int(row.get("feature_pos"))
        feature_id = _safe_int(row.get("feature_id"))
        target_token_id = _safe_int(row.get("target_token_id"))
        if layer is None or pos is None or feature_id is None or target_token_id is None:
            print(f"[skip] incomplete metadata sample={row.get('sample_id')} pair={row.get('pair_id')}", flush=True)
            continue

        for condition in conditions:
            if condition not in condition_masks:
                raise ValueError(f"unsupported condition: {condition}")
            mask = condition_masks[condition]
            image = original if mask is None else _apply_mask(original, mask, fill_rgb)
            question = row.get("question", "")
            assistant_prefix = row.get("assistant_prefix", "")
            out = {
                "pair_id": row.get("pair_id", ""),
                "sample_id": row.get("sample_id", ""),
                "run": row.get("run", ""),
                "prompt_name": row.get("prompt_name", ""),
                "node_role": row.get("node_role", ""),
                "node_source": row.get("node_source", ""),
                "condition": condition,
                "feature_layer": str(layer),
                "feature_pos": str(pos),
                "feature_id": str(feature_id),
                "target_token_id": str(target_token_id),
                "target_token": "",
                "baseline_target_logit": "",
                "intervention_target_logit": "",
                "delta_target_logit": "",
                "baseline_target_prob": "",
                "intervention_target_prob": "",
                "delta_target_prob": "",
                "baseline_target_rank": "",
                "intervention_target_rank": "",
                "rank_damage_by_intervention": "",
                "baseline_top1_id": "",
                "intervention_top1_id": "",
                "baseline_top1_token": "",
                "intervention_top1_token": "",
                "top1_changed_by_intervention": "",
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
                with torch.inference_mode():
                    baseline_logits = model.forward_from_batch(batch)
                    intervention_logits, _ = model.feature_intervention(
                        batch,
                        [(layer, pos, feature_id, 0.0)],
                        freeze_attention=True,
                        apply_activation_function=True,
                        sparse=False,
                    )
                baseline = _first_token_stats(baseline_logits, target_token_id, tokenizer)
                intervention = _first_token_stats(intervention_logits, target_token_id, tokenizer)
                out["target_token"] = baseline["target_token"]
                out["baseline_target_logit"] = baseline["target_logit"]
                out["intervention_target_logit"] = intervention["target_logit"]
                out["delta_target_logit"] = f"{float(intervention['target_logit']) - float(baseline['target_logit']):.10g}"
                out["baseline_target_prob"] = baseline["target_prob"]
                out["intervention_target_prob"] = intervention["target_prob"]
                out["delta_target_prob"] = f"{float(intervention['target_prob']) - float(baseline['target_prob']):.10g}"
                out["baseline_target_rank"] = baseline["target_rank"]
                out["intervention_target_rank"] = intervention["target_rank"]
                out["rank_damage_by_intervention"] = str(int(intervention["target_rank"]) - int(baseline["target_rank"]))
                out["baseline_top1_id"] = baseline["top1_id"]
                out["intervention_top1_id"] = intervention["top1_id"]
                out["baseline_top1_token"] = baseline["top1_token"]
                out["intervention_top1_token"] = intervention["top1_token"]
                out["top1_changed_by_intervention"] = str(baseline["top1_id"] != intervention["top1_id"])
            except Exception as exc:  # noqa: BLE001
                out["error_message"] = f"{type(exc).__name__}:{exc}"

            results.append(out)
            print(
                f"[done] sample={out['sample_id']} run={out['run']} source={out['node_source']} "
                f"condition={condition} rank_damage={out['rank_damage_by_intervention']} "
                f"top1_changed={out['top1_changed_by_intervention']} err={out['error_message']}",
                flush=True,
            )

    fieldnames = [
        "pair_id",
        "sample_id",
        "run",
        "prompt_name",
        "node_role",
        "node_source",
        "condition",
        "feature_layer",
        "feature_pos",
        "feature_id",
        "target_token_id",
        "target_token",
        "baseline_target_logit",
        "intervention_target_logit",
        "delta_target_logit",
        "baseline_target_prob",
        "intervention_target_prob",
        "delta_target_prob",
        "baseline_target_rank",
        "intervention_target_rank",
        "rank_damage_by_intervention",
        "baseline_top1_id",
        "intervention_top1_id",
        "baseline_top1_token",
        "intervention_top1_token",
        "top1_changed_by_intervention",
        "error_message",
    ]
    _write_csv(Path(args.out_csv).expanduser().resolve(), results, fieldnames)
    print(f"[done] out_csv={args.out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
