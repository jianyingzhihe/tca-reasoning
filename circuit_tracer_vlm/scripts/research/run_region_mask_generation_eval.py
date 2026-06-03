#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DEFAULT_CONDITIONS = [
    "clean",
    "answer_mask",
    "relate_mask",
    "union_mask",
    "random_control_1",
    "random_control_2",
    "random_control_3",
    "random_control_4",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_fill(fill: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in fill.split(",")]
    if len(parts) != 3:
        raise ValueError("--mask-fill must be r,g,b")
    return tuple(parts)


def _mask_array(mask_image: Image.Image) -> np.ndarray:
    arr = np.asarray(mask_image, dtype=np.uint8)
    return arr > 0


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


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    if union <= 0:
        return 0.0
    return inter / union


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _build_area_matched_rectangle(reference_mask: np.ndarray, x0: int, y0: int) -> np.ndarray | None:
    target_area = int(reference_mask.sum())
    if target_area <= 0:
        return np.zeros_like(reference_mask, dtype=bool)

    h, w = reference_mask.shape
    bbox = _bbox_from_mask(reference_mask)
    if bbox is None:
        return np.zeros_like(reference_mask, dtype=bool)
    bx0, by0, bx1, by1 = bbox
    bbox_w = max(1, bx1 - bx0)
    bbox_h = max(1, by1 - by0)
    aspect = bbox_w / bbox_h
    rect_w = max(1, int(round((target_area * aspect) ** 0.5)))
    rect_h = max(1, int(np.ceil(target_area / rect_w)))
    rect_w = min(rect_w, w)
    rect_h = min(rect_h, h)
    if x0 < 0 or y0 < 0 or x0 + rect_w > w or y0 + rect_h > h:
        return None

    out = np.zeros_like(reference_mask, dtype=bool)
    full_rows = target_area // rect_w
    rem = target_area % rect_w
    used_rows = min(full_rows, rect_h)
    if used_rows > 0:
        out[y0 : y0 + used_rows, x0 : x0 + rect_w] = True
    if rem > 0 and used_rows < rect_h:
        out[y0 + used_rows, x0 : x0 + rem] = True
    return out


def _make_random_control_mask(
    reference_mask: np.ndarray,
    avoid_mask: np.ndarray,
    seed: int,
    avoid_overlap_iou: float,
    max_tries: int,
) -> tuple[np.ndarray, float, bool]:
    target_area = int(reference_mask.sum())
    if target_area <= 0:
        return np.zeros_like(reference_mask, dtype=bool), 0.0, False

    h, w = reference_mask.shape
    bbox = _bbox_from_mask(reference_mask)
    if bbox is None:
        return np.zeros_like(reference_mask, dtype=bool), 0.0, False
    bx0, by0, bx1, by1 = bbox
    bbox_w = max(1, bx1 - bx0)
    bbox_h = max(1, by1 - by0)
    aspect = bbox_w / bbox_h
    rect_w = max(1, int(round((target_area * aspect) ** 0.5)))
    rect_h = max(1, int(np.ceil(target_area / rect_w)))
    rect_w = min(rect_w, w)
    rect_h = min(rect_h, h)

    rng = np.random.default_rng(seed)
    best_mask = None
    best_iou = None
    for _ in range(max_tries):
        x0 = int(rng.integers(0, max(1, w - rect_w + 1)))
        y0 = int(rng.integers(0, max(1, h - rect_h + 1)))
        candidate = _build_area_matched_rectangle(reference_mask, x0, y0)
        if candidate is None:
            continue
        iou = _mask_iou(candidate, avoid_mask)
        if best_iou is None or iou < best_iou:
            best_iou = iou
            best_mask = candidate
        if iou <= avoid_overlap_iou:
            return candidate, iou, True

    if best_mask is None:
        return np.zeros_like(reference_mask, dtype=bool), 0.0, False
    return best_mask, float(best_iou), False


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
    question = (question or "").strip()
    assistant_prefix = assistant_prefix or ""
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
        if assistant_prefix:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_prefix}],
                }
            )
        add_generation_prompt = not bool(assistant_prefix)
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if isinstance(inputs, dict) and "input_ids" in inputs:
            return inputs
    except Exception:
        pass
    prompt = f"<start_of_image> {question}".strip()
    if assistant_prefix:
        prompt = f"{prompt} {assistant_prefix.lstrip()}".strip()
    return processor(text=prompt, images=image, return_tensors="pt")


def _device_batch(batch: dict, device: str) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
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
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        sample_id = (row.get("sample_id") or "").strip()
        run = (row.get("run") or "").strip()
        if sample_id and run and (sample_id, run) not in out:
            out[(sample_id, run)] = row
    return list(out.values())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run short-answer generation under evidence-region masks.")
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--transcoder-set", default="tianhux2/gemma3-4b-it-plt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--mask-fill", default="128,128,128")
    parser.add_argument("--avoid-overlap-iou", type=float, default=0.05)
    parser.add_argument("--random-seeds", default="101,202,303,404")
    parser.add_argument("--random-max-tries", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    manifest_rows = _dedupe_manifest_rows(_read_csv(Path(args.manifest_csv).expanduser().resolve()))
    if args.max_rows > 0:
        manifest_rows = manifest_rows[: args.max_rows]
    if not manifest_rows:
        raise ValueError("no manifest rows loaded")

    model_name = args.model.strip() or _infer_model_name_from_transcoder_set(args.transcoder_set)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    print(f"[init] model={model_name} device={device} dtype={dtype} rows={len(manifest_rows)}", flush=True)

    model = Gemma3ForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)

    mask_root = Path(args.mask_root).expanduser().resolve()
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    random_seeds = [int(x.strip()) for x in args.random_seeds.split(",") if x.strip()]
    fill_rgb = _parse_fill(args.mask_fill)
    results: list[dict[str, str]] = []
    started = time.time()

    with torch.no_grad():
        for row_idx, row in enumerate(manifest_rows, start=1):
            image_path = row.get("image_path", "")
            image = Image.open(image_path).convert("RGB")
            answer_mask = _mask_array(_load_mask(mask_root, image_path, "answer"))
            relate_mask = _mask_array(_load_mask(mask_root, image_path, "relate"))
            union_mask = np.logical_or(answer_mask, relate_mask)
            random_masks = {}
            for idx, seed in enumerate(random_seeds, start=1):
                mask, actual_iou, valid = _make_random_control_mask(
                    answer_mask,
                    union_mask,
                    seed,
                    args.avoid_overlap_iou,
                    args.random_max_tries,
                )
                random_masks[f"random_control_{idx}"] = (mask, idx, seed, actual_iou, valid)

            condition_masks = {
                "clean": (None, "", "", "", ""),
                "answer_mask": (answer_mask, "", "", "", ""),
                "relate_mask": (relate_mask, "", "", "", ""),
                "union_mask": (union_mask, "", "", "", ""),
            }
            for condition, (mask, idx, seed, actual_iou, valid) in random_masks.items():
                condition_masks[condition] = (mask, str(idx), str(seed), f"{actual_iou:.10g}", str(bool(valid)))

            for condition in conditions:
                if condition not in condition_masks:
                    raise ValueError(f"unsupported condition: {condition}")
                mask_bool, random_index, random_seed, actual_iou, random_valid = condition_masks[condition]
                condition_image = image if mask_bool is None else _apply_mask(image, mask_bool, fill_rgb)
                assistant_prefix = row.get("assistant_prefix", "")
                out = {
                    "sample_id": row.get("sample_id", ""),
                    "run": row.get("run", ""),
                    "prompt_name": row.get("prompt_name", ""),
                    "assistant_prefix": assistant_prefix,
                    "condition": condition,
                    "question": row.get("question", ""),
                    "image_path": image_path,
                    "reasoning_operation": row.get("reasoning_operation", ""),
                    "visual_structure": row.get("visual_structure", ""),
                    "image_dependence": row.get("image_dependence", ""),
                    "mask_fill_rgb": args.mask_fill,
                    "random_control_index": random_index,
                    "random_control_seed": random_seed,
                    "random_control_valid": random_valid,
                    "random_control_actual_iou": actual_iou,
                    "answer_area_px": str(int(answer_mask.sum())),
                    "relate_area_px": str(int(relate_mask.sum())),
                    "union_area_px": str(int(union_mask.sum())),
                    "mask_area_px": "" if mask_bool is None else str(int(mask_bool.sum())),
                    "generated_continuation": "",
                    "generated_text": "",
                    "predicted_answer": "",
                    "error_message": "",
                }
                try:
                    batch = _device_batch(
                        _build_multimodal_inputs(processor, condition_image, row["question"], assistant_prefix),
                        device,
                    )
                    gen_ids = model.generate(
                        **batch,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        pad_token_id=processor.tokenizer.eos_token_id,
                    )
                    in_len = int(batch["input_ids"].shape[1])
                    new_ids = gen_ids[0, in_len:] if gen_ids.shape[1] > in_len else gen_ids[0]
                    continuation = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                    generated_text = f"{assistant_prefix}{continuation}".strip() if assistant_prefix else continuation
                    out["generated_continuation"] = continuation
                    out["generated_text"] = generated_text
                    out["predicted_answer"] = _extract_answer(generated_text)
                except Exception as exc:  # noqa: BLE001
                    out["error_message"] = f"{type(exc).__name__}:{exc}"

                results.append(out)
                print(
                    f"[done] sample={row.get('sample_id','')} run={row.get('run','')} "
                    f"condition={condition} pred={out['predicted_answer']!r}",
                    flush=True,
                )

            elapsed = max(time.time() - started, 1e-9)
            print(
                f"[progress] manifest_rows={row_idx}/{len(manifest_rows)} "
                f"output_rows={len(results)} elapsed_s={elapsed:.1f}",
                flush=True,
            )

    fieldnames = [
        "sample_id",
        "run",
        "prompt_name",
        "assistant_prefix",
        "condition",
        "question",
        "image_path",
        "reasoning_operation",
        "visual_structure",
        "image_dependence",
        "mask_fill_rgb",
        "random_control_index",
        "random_control_seed",
        "random_control_valid",
        "random_control_actual_iou",
        "answer_area_px",
        "relate_area_px",
        "union_area_px",
        "mask_area_px",
        "generated_continuation",
        "generated_text",
        "predicted_answer",
        "error_message",
    ]
    out_path = Path(args.out_csv).expanduser().resolve()
    _write_csv(out_path, results, fieldnames)
    print(f"[done] out_csv={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
