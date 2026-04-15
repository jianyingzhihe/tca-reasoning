#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re


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


def _build_inputs(processor, image, question: str):
    # Preferred: multimodal chat template path
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
            return inputs, "chat_template_tokenized"
    except Exception:
        pass

    # Fallback: explicit image token style
    prompt = f"<start_of_image> {question}".strip()
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    return inputs, "start_of_image_fallback"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one deterministic VLM inference for debugging.")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--question", required=True, help="Question text")
    parser.add_argument(
        "--append-answer-format",
        action="store_true",
        help="Append: 'Reply exactly in the format: The answer is <short answer>.'",
    )
    parser.add_argument("--model", default="", help="HF model id, e.g. google/gemma-3-4b-it")
    parser.add_argument("--transcoder-set", default="tianhux2/gemma3-4b-it-plt")
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    model_name = args.model.strip() or _infer_model_name_from_transcoder_set(args.transcoder_set.strip())

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    question = args.question.strip()
    if args.append_answer_format:
        question = f"{question} Reply exactly in the format: The answer is <short answer>."

    print(f"[init] model={model_name}")
    print(f"[init] device={device} dtype={dtype}")
    print(f"[input] image={args.image}")
    print(f"[input] question={question}")

    model = Gemma3ForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)

    image = Image.open(args.image).convert("RGB")
    inputs, prompt_mode = _build_inputs(processor, image, question)
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    in_len = int(inputs["input_ids"].shape[1])
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

    if gen_ids.shape[1] > in_len:
        new_ids = gen_ids[0, in_len:]
    else:
        # Some model/processor combos may already return only newly generated ids.
        new_ids = gen_ids[0]

    generated_new = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    generated_full = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
    predicted_answer = _extract_answer(generated_new)

    print(f"[debug] prompt_mode={prompt_mode}")
    print(f"[debug] input_token_len={in_len}")
    print(f"[debug] generated_token_len={int(gen_ids.shape[1])}")
    print("----- GENERATED (NEW TOKENS) -----")
    print(generated_new)
    print("----- GENERATED (FULL SEQUENCE) -----")
    print(generated_full)
    print("----- EXTRACTED ANSWER -----")
    print(predicted_answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

