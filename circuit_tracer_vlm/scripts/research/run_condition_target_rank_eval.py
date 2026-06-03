#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _load_selected_ids(path: Path) -> list[str]:
    rows = _read_csv(path)
    out: list[str] = []
    for row in rows:
        sample_id = (row.get("sample_id") or "").strip()
        if sample_id:
            out.append(sample_id)
    return out


def _build_target_meta_map(path: Path, default_run_slot: str) -> dict[tuple[str, str], dict[str, str]]:
    rows = _read_csv(path)
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        sample_id = (row.get("sample_id") or "").strip()
        if not sample_id:
            continue

        run = (row.get("run") or "").strip()
        token_id_str = (row.get("target_token_id") or "").strip()
        assistant_prefix = (row.get("assistant_prefix") or "").strip()
        if run and token_id_str:
            key = (sample_id, run)
            entry = {"target_token_id": str(int(token_id_str)), "assistant_prefix": assistant_prefix}
            prev = out.get(key)
            if prev is not None and prev.get("target_token_id") != entry["target_token_id"]:
                raise ValueError(f"inconsistent target_token_id for {key}: {prev['target_token_id']} vs {entry['target_token_id']}")
            if prev is None or assistant_prefix:
                out[key] = entry

        if (not run) and token_id_str:
            key = (sample_id, default_run_slot)
            entry = {"target_token_id": str(int(token_id_str)), "assistant_prefix": assistant_prefix}
            prev = out.get(key)
            if prev is not None and prev.get("target_token_id") != entry["target_token_id"]:
                raise ValueError(f"inconsistent target_token_id for {key}: {prev['target_token_id']} vs {entry['target_token_id']}")
            if prev is None or assistant_prefix:
                out[key] = entry

        a_token_id = (row.get("a_target_token_id") or "").strip()
        if a_token_id:
            key = (sample_id, "A")
            entry = {"target_token_id": str(int(a_token_id)), "assistant_prefix": ""}
            prev = out.get(key)
            if prev is not None and prev.get("target_token_id") != entry["target_token_id"]:
                raise ValueError(f"inconsistent target_token_id for {key}: {prev['target_token_id']} vs {entry['target_token_id']}")
            if prev is None:
                out[key] = entry

        b_token_id = (row.get("b_target_token_id") or "").strip()
        if b_token_id:
            key = (sample_id, "B")
            entry = {"target_token_id": str(int(b_token_id)), "assistant_prefix": ""}
            prev = out.get(key)
            if prev is not None and prev.get("target_token_id") != entry["target_token_id"]:
                raise ValueError(f"inconsistent target_token_id for {key}: {prev['target_token_id']} vs {entry['target_token_id']}")
            if prev is None:
                out[key] = entry
    return out


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


def _build_multimodal_inputs(processor, image, question: str, assistant_prefix: str = ""):
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if isinstance(inputs, dict) and "input_ids" in inputs:
            if assistant_prefix:
                prefix_ids = processor.tokenizer(assistant_prefix, add_special_tokens=False, return_tensors="pt")
                if prefix_ids["input_ids"].numel() > 0:
                    inputs["input_ids"] = inputs["input_ids"].to(prefix_ids["input_ids"].dtype)
                    inputs["input_ids"] = __import__("torch").cat([inputs["input_ids"], prefix_ids["input_ids"]], dim=1)
                    inputs["attention_mask"] = __import__("torch").cat(
                        [inputs["attention_mask"], prefix_ids["attention_mask"].to(inputs["attention_mask"].dtype)], dim=1
                    )
            return inputs
    except Exception:
        pass

    prompt = f"<start_of_image> {question}".strip()
    if assistant_prefix:
        prompt = f"{prompt}{assistant_prefix}"
    return processor(text=prompt, images=image, return_tensors="pt")


def _condition_image(condition: str, original_image, wrong_image, *, mask_fraction: float, mask_fill_rgb: tuple[int, int, int]):
    from PIL import Image, ImageDraw

    if condition == "clean":
        return original_image
    if condition == "no_image":
        return Image.new("RGB", original_image.size, color=(128, 128, 128))
    if condition == "wrong_image":
        return wrong_image
    if condition == "masked_image":
        masked = original_image.copy()
        width, height = masked.size
        mask_w = max(1, int(round(width * mask_fraction)))
        mask_h = max(1, int(round(height * mask_fraction)))
        left = max(0, (width - mask_w) // 2)
        top = max(0, (height - mask_h) // 2)
        right = min(width, left + mask_w)
        bottom = min(height, top + mask_h)
        ImageDraw.Draw(masked).rectangle((left, top, right, bottom), fill=mask_fill_rgb)
        return masked
    raise ValueError(f"unknown condition: {condition}")


def _decode_token(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)])
    return text.replace("\n", "\\n")


def _compact_topk(values: list[tuple[int, float, float, str]]) -> tuple[str, str, str, str]:
    ids = ",".join(str(v[0]) for v in values)
    logits = ",".join(f"{v[1]:.10g}" for v in values)
    probs = ",".join(f"{v[2]:.10g}" for v in values)
    toks = " || ".join(v[3] for v in values)
    return ids, logits, probs, toks


def _target_stats(logits, target_token_id: int, tokenizer) -> dict[str, str]:
    import torch

    last_pos = logits.shape[1] - 1
    row = logits[0, last_pos].detach().float().cpu()
    probs = torch.softmax(row, dim=-1)
    target_logit = float(row[target_token_id].item())
    target_prob = float(probs[target_token_id].item())

    sorted_ids = torch.argsort(row, descending=True)
    target_rank = int((sorted_ids == target_token_id).nonzero(as_tuple=False)[0].item()) + 1

    competitor_id = -1
    competitor_logit = float("nan")
    competitor_prob = float("nan")
    for candidate in sorted_ids.tolist():
        if int(candidate) != int(target_token_id):
            competitor_id = int(candidate)
            competitor_logit = float(row[competitor_id].item())
            competitor_prob = float(probs[competitor_id].item())
            break
    margin = target_logit - competitor_logit

    top5 = []
    for candidate in sorted_ids[:5].tolist():
        candidate = int(candidate)
        top5.append((candidate, float(row[candidate].item()), float(probs[candidate].item()), _decode_token(tokenizer, candidate)))
    top5_ids, top5_logits, top5_probs, top5_tokens = _compact_topk(top5)

    return {
        "target_logit": f"{target_logit:.10g}",
        "target_prob": f"{target_prob:.10g}",
        "target_rank": str(target_rank),
        "target_in_top1": "1" if target_rank == 1 else "0",
        "target_in_top5": "1" if target_rank <= 5 else "0",
        "target_token": _decode_token(tokenizer, target_token_id),
        "competitor_token_id": str(competitor_id),
        "competitor_token": _decode_token(tokenizer, competitor_id),
        "competitor_logit": f"{competitor_logit:.10g}",
        "competitor_prob": f"{competitor_prob:.10g}",
        "target_vs_competitor_margin": f"{margin:.10g}",
        "top5_token_ids": top5_ids,
        "top5_logits": top5_logits,
        "top5_probs": top5_probs,
        "top5_tokens": top5_tokens,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate target token rank/prob/margin under clean/masked/wrong/no-image conditions.")
    parser.add_argument("--eval-csv", required=True)
    parser.add_argument("--sample-ids-csv", required=True)
    parser.add_argument("--target-meta-source-csv", required=True)
    parser.add_argument("--run-slot", required=True, choices=["A", "B"])
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--transcoder-set", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conditions", default="clean,masked_image,wrong_image,no_image")
    parser.add_argument("--mask-fraction", type=float, default=0.4)
    parser.add_argument("--mask-fill", default="128,128,128")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    eval_csv = Path(args.eval_csv).expanduser().resolve()
    sample_ids_csv = Path(args.sample_ids_csv).expanduser().resolve()
    target_meta_source_csv = Path(args.target_meta_source_csv).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        output_csv.unlink()

    conditions = [part.strip() for part in args.conditions.split(",") if part.strip()]
    selected_ids = _load_selected_ids(sample_ids_csv)
    selected_set = set(selected_ids)
    eval_rows = [row for row in _read_csv(eval_csv) if (row.get("sample_id") or "").strip() in selected_set]
    eval_by_id = {(row.get("sample_id") or "").strip(): row for row in eval_rows}
    ordered_rows = [eval_by_id[sid] for sid in selected_ids if sid in eval_by_id]
    if len(ordered_rows) < 2:
        raise ValueError("need at least 2 selected rows for wrong-image mapping")

    target_meta_map = _build_target_meta_map(target_meta_source_csv, args.run_slot)
    missing_targets = [sid for sid in selected_ids if (sid, args.run_slot) not in target_meta_map]
    if missing_targets:
        raise ValueError(f"missing target_token_id for run={args.run_slot}: {missing_targets[:5]}")

    if args.model.strip():
        model_name = args.model.strip()
    else:
        if not args.transcoder_set.strip():
            raise ValueError("provide --model or --transcoder-set")
        model_name = _infer_model_name_from_transcoder_set(args.transcoder_set.strip())

    wrong_image_map: dict[str, tuple[str, str]] = {}
    for idx, row in enumerate(ordered_rows):
        donor = ordered_rows[(idx + 1) % len(ordered_rows)]
        wrong_image_map[(row.get("sample_id") or "").strip()] = (
            (donor.get("sample_id") or "").strip(),
            (donor.get("image_path") or "").strip(),
        )

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    print(
        f"[init] model={model_name} device={device} dtype={dtype} rows={len(ordered_rows)} run_slot={args.run_slot} conditions={conditions}",
        flush=True,
    )
    model = Gemma3ForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)

    fieldnames = [
        "sample_id",
        "run_slot",
        "condition",
        "wrong_image_sample_id",
        "question_id",
        "image_id",
        "question",
        "image_path",
        "condition_image_path",
        "assistant_prefix",
        "target_token_id",
        "target_token",
        "target_logit",
        "target_prob",
        "target_rank",
        "target_in_top1",
        "target_in_top5",
        "competitor_token_id",
        "competitor_token",
        "competitor_logit",
        "competitor_prob",
        "target_vs_competitor_margin",
        "top5_token_ids",
        "top5_logits",
        "top5_probs",
        "top5_tokens",
        "error_message",
    ]

    mask_fill_rgb = tuple(int(part.strip()) for part in args.mask_fill.split(","))
    started = time.time()
    written = 0
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(ordered_rows, start=1):
            sample_id = (row.get("sample_id") or "").strip()
            question = (row.get("question") or "").strip()
            image_path = (row.get("image_path") or "").strip()
            qid = (row.get("question_id") or "").strip()
            image_id = (row.get("image_id") or "").strip()
            target_meta = target_meta_map[(sample_id, args.run_slot)]
            target_token_id = int(target_meta["target_token_id"])
            assistant_prefix = target_meta.get("assistant_prefix", "")
            wrong_sample_id, wrong_image_path = wrong_image_map[sample_id]

            original_image = Image.open(image_path).convert("RGB")
            wrong_image = Image.open(wrong_image_path).convert("RGB")

            for condition in conditions:
                conditioned = _condition_image(
                    condition,
                    original_image,
                    wrong_image,
                    mask_fraction=args.mask_fraction,
                    mask_fill_rgb=mask_fill_rgb,
                )
                out = {
                    "sample_id": sample_id,
                    "run_slot": args.run_slot,
                    "condition": condition,
                    "wrong_image_sample_id": wrong_sample_id if condition == "wrong_image" else "",
                    "question_id": qid,
                    "image_id": image_id,
                    "question": question,
                    "image_path": image_path,
                    "condition_image_path": wrong_image_path if condition == "wrong_image" else image_path,
                    "target_token_id": str(target_token_id),
                    "assistant_prefix": assistant_prefix,
                    "target_token": "",
                    "target_logit": "",
                    "target_prob": "",
                    "target_rank": "",
                    "target_in_top1": "",
                    "target_in_top5": "",
                    "competitor_token_id": "",
                    "competitor_token": "",
                    "competitor_logit": "",
                    "competitor_prob": "",
                    "target_vs_competitor_margin": "",
                    "top5_token_ids": "",
                    "top5_logits": "",
                    "top5_probs": "",
                    "top5_tokens": "",
                    "error_message": "",
                }
                try:
                    inputs = _build_multimodal_inputs(processor, conditioned, question, assistant_prefix=assistant_prefix)
                    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
                    with torch.no_grad():
                        logits = model(**inputs).logits
                    out.update(_target_stats(logits, target_token_id, processor.tokenizer))
                except Exception as exc:  # noqa: BLE001
                    out["error_message"] = f"{type(exc).__name__}:{exc}"

                writer.writerow(out)
                written += 1

            if idx % max(1, args.log_every) == 0 or idx == len(ordered_rows):
                elapsed = max(time.time() - started, 1e-9)
                print(f"[progress] samples={idx}/{len(ordered_rows)} rows={written} elapsed_s={elapsed:.1f}", flush=True)

    print(f"[done] output_csv={output_csv} rows={written}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
