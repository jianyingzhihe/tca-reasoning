#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path


_NUM_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

_ARTICLES = {"a", "an", "the"}


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


def _vqa_normalize(text: str) -> str:
    s = (text or "").lower().strip()
    s = s.replace("\n", " ").replace("\t", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    toks = []
    for t in s.split():
        t = _NUM_MAP.get(t, t)
        if t in _ARTICLES:
            continue
        toks.append(t)
    return " ".join(toks)


def _extract_answer(generated: str) -> str:
    txt = (generated or "").strip()
    if not txt:
        return ""
    m = re.search(r"the answer is\s*[:\-]?\s*(.+)", txt, flags=re.IGNORECASE | re.DOTALL)
    ans = m.group(1).strip() if m else txt
    ans = re.split(r"[\n\r]", ans)[0].strip()
    ans = re.split(r"[.!?]", ans)[0].strip()
    ans = ans.strip("\"'` ")
    return ans


def _vqa_soft_score(pred_norm: str, gt_norms: list[str]) -> float:
    if not gt_norms:
        return 0.0
    n = len(gt_norms)
    accs = []
    for i in range(n):
        others = gt_norms[:i] + gt_norms[i + 1 :]
        matches = sum(1 for a in others if a == pred_norm)
        accs.append(min(1.0, matches / 3.0))
    return float(sum(accs) / len(accs))


def _majority_answer(gt_norms: list[str]) -> str:
    if not gt_norms:
        return ""
    return Counter(gt_norms).most_common(1)[0][0]


def _load_okvqa_ann_map(annotations_json: str) -> dict[str, list[str]]:
    if not annotations_json:
        return {}
    p = Path(annotations_json).expanduser().resolve()
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    ann = obj.get("annotations", [])
    out: dict[str, list[str]] = {}
    for item in ann:
        qid = str(item.get("question_id", "")).strip()
        answers = item.get("answers", []) or []
        out[qid] = [str(x.get("answer", "")).strip() for x in answers if str(x.get("answer", "")).strip()]
    return out


def _to_binary_correct(
    rule: str,
    strict_gold_correct: bool,
    strict_majority_correct: bool,
    vqa_score: float,
) -> str:
    if rule == "strict_gold":
        return "1" if strict_gold_correct else "0"
    if rule == "majority":
        return "1" if strict_majority_correct else "0"
    if rule == "vqa_0.3":
        return "1" if vqa_score >= 0.3 else "0"
    if rule == "vqa_0.6":
        return "1" if vqa_score >= 0.6 else "0"
    if rule == "vqa_1.0":
        return "1" if vqa_score >= 1.0 else "0"
    return "0"


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


def _build_multimodal_inputs(processor, image, question: str):
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
            return inputs
    except Exception:
        pass

    prompt = f"<start_of_image> {question}".strip()
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run short-answer generation under clean/masked/wrong/no-image conditions.")
    parser.add_argument("--eval-csv", required=True)
    parser.add_argument("--sample-ids-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--transcoder-set", default="")
    parser.add_argument("--annotations-json", required=True)
    parser.add_argument("--correct-rule", default="vqa_0.3", choices=["strict_gold", "majority", "vqa_0.3", "vqa_0.6", "vqa_1.0"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=16)
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

    if args.model.strip():
        model_name = args.model.strip()
    else:
        if not args.transcoder_set.strip():
            raise ValueError("provide --model or --transcoder-set")
        model_name = _infer_model_name_from_transcoder_set(args.transcoder_set.strip())

    ann_map = _load_okvqa_ann_map(args.annotations_json)
    mask_fill_rgb = tuple(int(part.strip()) for part in args.mask_fill.split(","))

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

    print(f"[init] model={model_name} device={device} dtype={dtype} rows={len(ordered_rows)} conditions={conditions}", flush=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)

    fieldnames = [
        "sample_id",
        "condition",
        "wrong_image_sample_id",
        "question_id",
        "image_id",
        "gold_answer",
        "majority_answer",
        "predicted_answer",
        "gold_answer_norm",
        "majority_answer_norm",
        "predicted_answer_norm",
        "strict_gold_correct",
        "strict_majority_correct",
        "vqa_score",
        "correct",
        "generated_text",
        "question",
        "image_path",
        "condition_image_path",
        "error_message",
    ]

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
            gold = (row.get("gold_answer") or "").strip()
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
                    "condition": condition,
                    "wrong_image_sample_id": wrong_sample_id if condition == "wrong_image" else "",
                    "question_id": qid,
                    "image_id": image_id,
                    "gold_answer": gold,
                    "majority_answer": "",
                    "predicted_answer": "",
                    "gold_answer_norm": "",
                    "majority_answer_norm": "",
                    "predicted_answer_norm": "",
                    "strict_gold_correct": "",
                    "strict_majority_correct": "",
                    "vqa_score": "",
                    "correct": "",
                    "generated_text": "",
                    "question": question,
                    "image_path": image_path,
                    "condition_image_path": wrong_image_path if condition == "wrong_image" else image_path,
                    "error_message": "",
                }
                try:
                    inputs = _build_multimodal_inputs(processor, conditioned, question)
                    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
                    with torch.no_grad():
                        gen_ids = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            top_k=None,
                            eos_token_id=processor.tokenizer.eos_token_id,
                            pad_token_id=processor.tokenizer.eos_token_id,
                        )
                    in_len = int(inputs["input_ids"].shape[1])
                    new_ids = gen_ids[0, in_len:] if gen_ids.shape[1] > in_len else gen_ids[0]
                    generated = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                    pred = _extract_answer(generated)
                    pred_norm = _vqa_normalize(pred)
                    gold_norm = _vqa_normalize(gold)
                    gt_answers = ann_map.get(qid, [])
                    gt_norms = [_vqa_normalize(x) for x in gt_answers if _vqa_normalize(x)]
                    majority = _majority_answer(gt_norms)
                    vqa_score = _vqa_soft_score(pred_norm, gt_norms) if gt_norms else 0.0
                    strict_gold_correct = pred_norm == gold_norm if gold_norm else False
                    strict_majority_correct = pred_norm == majority if majority else False
                    binary_correct = _to_binary_correct(args.correct_rule, strict_gold_correct, strict_majority_correct, vqa_score)

                    out["generated_text"] = generated
                    out["predicted_answer"] = pred
                    out["gold_answer_norm"] = gold_norm
                    out["majority_answer_norm"] = majority
                    out["majority_answer"] = majority
                    out["predicted_answer_norm"] = pred_norm
                    out["strict_gold_correct"] = "1" if strict_gold_correct else "0"
                    out["strict_majority_correct"] = "1" if strict_majority_correct else "0"
                    out["vqa_score"] = f"{vqa_score:.6f}"
                    out["correct"] = binary_correct
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
