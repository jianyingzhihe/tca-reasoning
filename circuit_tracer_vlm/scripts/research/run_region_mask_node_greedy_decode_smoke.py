#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
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


def _init_stream_csv(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _append_stream_csv(path: Path, row: dict[str, str], fieldnames: list[str]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)
        f.flush()


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


def _append_token(batch: dict, token_id: int) -> dict:
    out = dict(batch)
    device = out["input_ids"].device
    token = torch.tensor([[token_id]], dtype=out["input_ids"].dtype, device=device)
    out["input_ids"] = torch.cat([out["input_ids"], token], dim=1)
    if "attention_mask" in out and torch.is_tensor(out["attention_mask"]):
        attn = torch.ones((out["attention_mask"].shape[0], 1), dtype=out["attention_mask"].dtype, device=out["attention_mask"].device)
        out["attention_mask"] = torch.cat([out["attention_mask"], attn], dim=1)
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


def _decode_ids(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def _greedy_decode(
    model,
    batch: dict,
    tokenizer,
    *,
    max_new_tokens: int,
    intervention: tuple[int, int, int, float] | None = None,
) -> tuple[list[int], str]:
    cur = dict(batch)
    generated: list[int] = []
    eos_id = tokenizer.eos_token_id
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            if intervention is None:
                logits = model.forward_from_batch(cur)
            else:
                logits, _ = model.feature_intervention(
                    cur,
                    [intervention],
                    freeze_attention=True,
                    apply_activation_function=True,
                    sparse=False,
                )
            next_id = int(torch.argmax(logits[0, -1, :]).item())
            if eos_id is not None and next_id == int(eos_id):
                break
            generated.append(next_id)
            cur = _append_token(cur, next_id)
    return generated, _decode_ids(tokenizer, generated)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 2B smoke: hand-written greedy decoded generation under node zeroing for VLM batches."
    )
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--transcoder-set", default="tianhux2/gemma3-4b-it-plt")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16", "fp32", "bf16", "fp16"])
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--pair-ids", default="")
    parser.add_argument("--sample-ids", default="")
    parser.add_argument("--node-role", default="support", choices=["support", "suppressor"])
    parser.add_argument("--node-source", default="", choices=["", "source", "nearest_control"])
    parser.add_argument("--mask-fill", default="128,128,128")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument(
        "--stream-output",
        action="store_true",
        help="Write each row as soon as it finishes, so partial results survive timeouts.",
    )
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
    pair_ids = {x.strip() for x in args.pair_ids.split(",") if x.strip()}
    sample_ids = {x.strip() for x in args.sample_ids.split(",") if x.strip()}
    rows = [row for row in rows if row.get("node_role") == args.node_role]
    if pair_ids:
        rows = [row for row in rows if row.get("pair_id") in pair_ids]
    if sample_ids:
        rows = [row for row in rows if row.get("sample_id") in sample_ids]
    rows = [row for row in rows if row.get("node_source") in {"source", "nearest_control"}]
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
        "assistant_prefix",
        "baseline_token_ids",
        "intervention_token_ids",
        "baseline_continuation",
        "intervention_continuation",
        "baseline_generated_text",
        "intervention_generated_text",
        "baseline_answer",
        "intervention_answer",
        "answer_changed_by_intervention",
        "error_message",
    ]
    out_csv = Path(args.out_csv).expanduser().resolve()
    results: list[dict[str, str]] = []
    if args.stream_output:
        _init_stream_csv(out_csv, fieldnames)

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
        if layer is None or pos is None or feature_id is None:
            print(f"[skip] incomplete metadata sample={row.get('sample_id')} pair={row.get('pair_id')}", flush=True)
            continue
        intervention = (layer, pos, feature_id, 0.0)

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
                "assistant_prefix": assistant_prefix,
                "baseline_token_ids": "",
                "intervention_token_ids": "",
                "baseline_continuation": "",
                "intervention_continuation": "",
                "baseline_generated_text": "",
                "intervention_generated_text": "",
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

                baseline_ids, baseline_cont = _greedy_decode(
                    model,
                    batch,
                    tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    intervention=None,
                )
                intervention_ids, intervention_cont = _greedy_decode(
                    model,
                    batch,
                    tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    intervention=intervention,
                )
                baseline_text = f"{assistant_prefix}{baseline_cont}".strip() if assistant_prefix else baseline_cont
                intervention_text = f"{assistant_prefix}{intervention_cont}".strip() if assistant_prefix else intervention_cont
                baseline_answer = _extract_answer(baseline_text)
                intervention_answer = _extract_answer(intervention_text)
                out["baseline_token_ids"] = " ".join(str(x) for x in baseline_ids)
                out["intervention_token_ids"] = " ".join(str(x) for x in intervention_ids)
                out["baseline_continuation"] = baseline_cont
                out["intervention_continuation"] = intervention_cont
                out["baseline_generated_text"] = baseline_text
                out["intervention_generated_text"] = intervention_text
                out["baseline_answer"] = baseline_answer
                out["intervention_answer"] = intervention_answer
                out["answer_changed_by_intervention"] = str(baseline_answer != intervention_answer)
            except Exception as exc:  # noqa: BLE001
                out["error_message"] = f"{type(exc).__name__}:{exc}"
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            results.append(out)
            if args.stream_output:
                _append_stream_csv(out_csv, out, fieldnames)
            print(
                f"[done] sample={out['sample_id']} run={out['run']} source={out['node_source']} "
                f"condition={condition} changed={out['answer_changed_by_intervention']} "
                f"base={out['baseline_answer']!r} int={out['intervention_answer']!r} err={out['error_message']}",
                flush=True,
            )

    if not args.stream_output:
        _write_csv(out_csv, results, fieldnames)
    print(f"[done] out_csv={args.out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
