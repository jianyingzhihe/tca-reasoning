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


def _parse_gold_answer(notes: str) -> str:
    if not notes:
        return ""
    for part in notes.split(";"):
        part = part.strip()
        if part.startswith("answer="):
            return part.split("=", 1)[1].strip()
    return ""


def _parse_ids_from_notes(notes: str) -> tuple[str, str]:
    qid = ""
    image_id = ""
    if not notes:
        return qid, image_id
    for part in notes.split(";"):
        s = part.strip()
        if s.startswith("qid="):
            qid = s.split("=", 1)[1].strip()
        elif s.startswith("image_id="):
            image_id = s.split("=", 1)[1].strip()
    return qid, image_id


def _normalize_answer(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


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


def _read_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            sid = (r.get("sample_id") or "").strip()
            if sid:
                out.add(sid)
    return out


def _append_row(path: Path, row: dict, fieldnames: list[str]) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic VLM generation on a manifest and write eval csv. "
            "Supports OKVQA-style soft score if annotations json is provided."
        )
    )
    parser.add_argument("--manifest", required=True, help="Input manifest csv")
    parser.add_argument("--output-csv", required=True, help="Eval output csv")
    parser.add_argument("--model", default="", help="HF model id, e.g. google/gemma-3-4b-it")
    parser.add_argument(
        "--transcoder-set",
        default="",
        help="Optional transcoder repo id; if --model is empty, infer model_name from config.yaml.",
    )
    parser.add_argument(
        "--annotations-json",
        default="",
        help="Optional OKVQA annotations json (official soft score).",
    )
    parser.add_argument(
        "--correct-rule",
        default="vqa_0.3",
        choices=["strict_gold", "majority", "vqa_0.3", "vqa_0.6", "vqa_1.0"],
        help="Rule used to emit binary `correct` column.",
    )
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    manifest = Path(args.manifest).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.model.strip():
        model_name = args.model.strip()
    else:
        if not args.transcoder_set.strip():
            raise ValueError("provide --model or --transcoder-set")
        model_name = _infer_model_name_from_transcoder_set(args.transcoder_set.strip())

    ann_map = _load_okvqa_ann_map(args.annotations_json)

    with manifest.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit > 0:
        rows = rows[: args.limit]
    total = len(rows)
    if total == 0:
        raise ValueError(f"empty manifest: {manifest}")

    done_ids = _read_done_ids(output_csv) if args.resume else set()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    print(f"[init] model={model_name}")
    print(f"[init] device={device} dtype={dtype}")
    print(f"[init] annotations={'on' if ann_map else 'off'} correct_rule={args.correct_rule}")

    model = Gemma3ForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)

    fieldnames = [
        "sample_id",
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
        "error_message",
    ]

    started = time.time()
    processed = 0
    for i, row in enumerate(rows, start=1):
        sid = (row.get("sample_id") or "").strip()
        question = (row.get("question") or "").strip()
        image_path = (row.get("image_path") or "").strip()
        notes = (row.get("notes") or "").strip()
        qid_note, iid_note = _parse_ids_from_notes(notes)
        qid = (row.get("question_id") or qid_note or "").strip()
        image_id = (row.get("image_id") or iid_note or "").strip()
        gold = (row.get("gold_answer") or "").strip() or _parse_gold_answer(notes)
        if not sid or not question or not image_path:
            continue
        if sid in done_ids:
            processed += 1
            continue

        out = {
            "sample_id": sid,
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
            "error_message": "",
        }

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(text=question, images=image, return_tensors="pt")
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
            gold_norm = _vqa_normalize(gold)
            gt_answers = ann_map.get(qid, [])
            gt_norms = [_vqa_normalize(x) for x in gt_answers if _vqa_normalize(x)]
            majority = _majority_answer(gt_norms)
            vqa_score = _vqa_soft_score(pred_norm, gt_norms) if gt_norms else 0.0
            strict_gold_correct = pred_norm == gold_norm if gold_norm else False
            strict_majority_correct = pred_norm == majority if majority else False
            binary_correct = _to_binary_correct(
                args.correct_rule, strict_gold_correct, strict_majority_correct, vqa_score
            )

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

        _append_row(output_csv, out, fieldnames)
        done_ids.add(sid)
        processed += 1

        if processed % max(1, args.log_every) == 0 or processed == total:
            elapsed = max(time.time() - started, 1e-9)
            rate = processed / elapsed
            eta_min = (total - processed) / rate / 60.0 if rate > 0 else float("inf")
            print(
                f"[progress] {processed}/{total} ({processed/total*100:.1f}%) "
                f"rate={rate:.4f} sample/s eta={eta_min:.1f}m"
            )

    print(f"[done] eval csv: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
