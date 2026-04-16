#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path


PROMPT_A_TEMPLATE = (
    "{question} Think step by step from visual evidence, then reply exactly in the format: "
    "The answer is <short answer>."
)

PROMPT_B_TEMPLATE = "{question} Reply exactly in the format: The answer is <short answer>."

PROMPT_C_TEMPLATE = (
    "Below are an instruction that describes a task along with a reference answer. "
    "Using the reference answer as a guide, write your own response.\n"
    "### Example Instruction:\n"
    "{example_instruction}\n"
    "### Example Response:\n"
    "{example_response}\n"
    "### Instruction:\n"
    "{question}\n"
    "### Response:"
)

PROMPT_VARIANTS = ("A", "B", "C")

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


def _build_val_image_path(image_root: Path, image_id: int) -> Path:
    return image_root / "val2014" / f"COCO_val2014_{image_id:012d}.jpg"


def _load_questions(path: Path, limit: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: list[dict[str, str]] = []
    for item in data.get("questions", []):
        qid = int(item["question_id"])
        image_id = int(item["image_id"])
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        out.append(
            {
                "question_id": str(qid),
                "image_id": str(image_id),
                "question": question,
                "sample_id": f"okvqa_val_q{qid}",
            }
        )
    out.sort(key=lambda x: int(x["question_id"]))
    if limit > 0:
        out = out[:limit]
    return out


def _load_annotations(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    ann_map: dict[str, list[str]] = {}
    for item in data.get("annotations", []):
        qid = str(item.get("question_id", "")).strip()
        answers = item.get("answers", []) or []
        ann_map[qid] = [str(x.get("answer", "")).strip() for x in answers if str(x.get("answer", "")).strip()]
    return ann_map


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
    return ans.strip("\"'` ")


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


def _to_binary_correct(
    rule: str,
    strict_majority_correct: bool,
    vqa_score: float,
) -> str:
    if rule == "majority":
        return "1" if strict_majority_correct else "0"
    if rule == "vqa_0.3":
        return "1" if vqa_score >= 0.3 else "0"
    if rule == "vqa_0.6":
        return "1" if vqa_score >= 0.6 else "0"
    if rule == "vqa_1.0":
        return "1" if vqa_score >= 1.0 else "0"
    return "0"


def _render_prompt(variant: str, question: str, example_instruction: str, example_response: str) -> str:
    if variant == "A":
        return PROMPT_A_TEMPLATE.format(question=question).strip()
    if variant == "B":
        return PROMPT_B_TEMPLATE.format(question=question).strip()
    if variant == "C":
        return PROMPT_C_TEMPLATE.format(
            question=question,
            example_instruction=example_instruction,
            example_response=example_response,
        ).strip()
    raise ValueError(f"unknown prompt variant: {variant}")


def _output_csv_path(output_dir: Path, variant: str) -> Path:
    mapping = {"A": "promptA_eval.csv", "B": "promptB_eval.csv", "C": "promptC_eval.csv"}
    return output_dir / mapping[variant]


def _read_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sid = (row.get("sample_id") or "").strip()
            if sid:
                out.add(sid)
    return out


def _append_row(path: Path, row: dict[str, str], fieldnames: list[str]) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run OKVQA val inference for three prompts (A/B/C) with Gemma and write three eval csv files. "
            "This script intentionally does only this one task."
        )
    )
    parser.add_argument("--questions", required=True, help="OpenEnded_mscoco_val2014_questions.json")
    parser.add_argument("--annotations", required=True, help="mscoco_val2014_annotations.json")
    parser.add_argument("--image-root", required=True, help="COCO image root; expects val2014 under it")
    parser.add_argument("--output-dir", required=True, help="Directory for promptA/B/C eval csv outputs")
    parser.add_argument("--model", default="", help="HF model id; if empty, infer via --transcoder-set")
    parser.add_argument(
        "--transcoder-set",
        default="",
        help="Optional transcoder set repo id (used only when --model is empty)",
    )
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit over val questions")
    parser.add_argument(
        "--correct-rule",
        default="vqa_0.3",
        choices=["majority", "vqa_0.3", "vqa_0.6", "vqa_1.0"],
        help="Rule used for binary correct column",
    )
    parser.add_argument("--example-instruction", default="What color is the bus?")
    parser.add_argument("--example-response", default="The answer is yellow.")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from existing promptA/B/C csv files (default: on)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore existing output files and write fresh rows",
    )
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    questions_path = Path(args.questions).expanduser().resolve()
    annotations_path = Path(args.annotations).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model.strip():
        model_name = args.model.strip()
    else:
        if not args.transcoder_set.strip():
            raise ValueError("Provide --model or --transcoder-set.")
        model_name = _infer_model_name_from_transcoder_set(args.transcoder_set.strip())

    questions = _load_questions(questions_path, args.limit)
    if not questions:
        raise ValueError(f"no valid question rows in {questions_path}")
    ann_map = _load_annotations(annotations_path)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    print(f"[init] model={model_name}", flush=True)
    print(f"[init] device={device} dtype={dtype}", flush=True)
    print(f"[init] rows={len(questions)} resume={args.resume} correct_rule={args.correct_rule}", flush=True)

    model = Gemma3ForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)

    fieldnames = [
        "sample_id",
        "question_id",
        "image_id",
        "prompt_variant",
        "prompt_text",
        "majority_answer",
        "predicted_answer",
        "majority_answer_norm",
        "predicted_answer_norm",
        "strict_majority_correct",
        "vqa_score",
        "correct",
        "generated_text",
        "question",
        "image_path",
        "error_message",
    ]

    out_csv = {v: _output_csv_path(output_dir, v) for v in PROMPT_VARIANTS}
    done_ids = {v: _read_done_ids(out_csv[v]) if args.resume else set() for v in PROMPT_VARIANTS}

    sample_processed = 0
    rows_written = {v: 0 for v in PROMPT_VARIANTS}
    rows_skipped_resume = {v: 0 for v in PROMPT_VARIANTS}
    rows_error = {v: 0 for v in PROMPT_VARIANTS}
    started = time.time()

    for idx, item in enumerate(questions, start=1):
        sid = item["sample_id"]
        qid = item["question_id"]
        image_id = int(item["image_id"])
        question = item["question"]

        pending = [v for v in PROMPT_VARIANTS if sid not in done_ids[v]]
        if not pending:
            for v in PROMPT_VARIANTS:
                rows_skipped_resume[v] += 1
            sample_processed += 1
            continue

        image_path = _build_val_image_path(image_root, image_id)
        image_obj = None
        image_error = ""
        if not image_path.exists():
            image_error = f"FileNotFoundError:{image_path}"
        else:
            try:
                image_obj = Image.open(image_path).convert("RGB")
            except Exception as exc:  # noqa: BLE001
                image_error = f"{type(exc).__name__}:{exc}"

        gt_answers = ann_map.get(qid, [])
        gt_norms = [_vqa_normalize(x) for x in gt_answers if _vqa_normalize(x)]
        majority = _majority_answer(gt_norms)

        for variant in pending:
            prompt_text = _render_prompt(
                variant,
                question,
                args.example_instruction.strip(),
                args.example_response.strip(),
            )

            out = {
                "sample_id": sid,
                "question_id": qid,
                "image_id": str(image_id),
                "prompt_variant": variant,
                "prompt_text": prompt_text,
                "majority_answer": majority,
                "predicted_answer": "",
                "majority_answer_norm": majority,
                "predicted_answer_norm": "",
                "strict_majority_correct": "",
                "vqa_score": "",
                "correct": "",
                "generated_text": "",
                "question": question,
                "image_path": str(image_path),
                "error_message": "",
            }

            try:
                if image_error:
                    raise FileNotFoundError(image_error)
                inputs = processor(text=prompt_text, images=image_obj, return_tensors="pt")
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
                new_ids = gen_ids[0, in_len:]
                generated = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                pred = _extract_answer(generated)
                pred_norm = _vqa_normalize(pred)
                vqa_score = _vqa_soft_score(pred_norm, gt_norms) if gt_norms else 0.0
                strict_majority_correct = pred_norm == majority if majority else False
                correct = _to_binary_correct(args.correct_rule, strict_majority_correct, vqa_score)

                out["generated_text"] = generated
                out["predicted_answer"] = pred
                out["predicted_answer_norm"] = pred_norm
                out["strict_majority_correct"] = "1" if strict_majority_correct else "0"
                out["vqa_score"] = f"{vqa_score:.6f}"
                out["correct"] = correct
            except Exception as exc:  # noqa: BLE001
                out["error_message"] = f"{type(exc).__name__}:{exc}"
                rows_error[variant] += 1

            _append_row(out_csv[variant], out, fieldnames)
            done_ids[variant].add(sid)
            rows_written[variant] += 1

        sample_processed += 1
        if sample_processed % max(1, args.log_every) == 0 or sample_processed == len(questions):
            elapsed = max(time.time() - started, 1e-9)
            rate = sample_processed / elapsed
            eta_min = (len(questions) - sample_processed) / rate / 60.0 if rate > 0 else float("inf")
            print(
                f"[progress] {sample_processed}/{len(questions)} ({sample_processed/len(questions)*100:.1f}%) "
                f"rate={rate:.4f} sample/s eta={eta_min:.1f}m "
                f"written(A/B/C)={rows_written['A']}/{rows_written['B']}/{rows_written['C']} "
                f"errors(A/B/C)={rows_error['A']}/{rows_error['B']}/{rows_error['C']}",
                flush=True,
            )

    summary_path = output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"model={model_name}\n")
        f.write(f"questions={questions_path}\n")
        f.write(f"annotations={annotations_path}\n")
        f.write(f"image_root={image_root}\n")
        f.write(f"rows={len(questions)}\n")
        f.write(f"correct_rule={args.correct_rule}\n")
        for v in PROMPT_VARIANTS:
            f.write(f"{v}_csv={out_csv[v]}\n")
            f.write(f"{v}_written={rows_written[v]}\n")
            f.write(f"{v}_resume_skipped={rows_skipped_resume[v]}\n")
            f.write(f"{v}_errors={rows_error[v]}\n")

    print(f"[done] summary: {summary_path}", flush=True)
    for v in PROMPT_VARIANTS:
        print(f"[done] prompt{v}: {out_csv[v]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
