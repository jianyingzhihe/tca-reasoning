# OKVQA Val Prompt A/B/C Inference

This folder contains one script for one task:
run OKVQA val inference with Prompt A/B/C (including one-shot C) using Gemma, and write three eval csv files.

## Script

- `run_okvqa_val_prompt_abc_infer.py`

## Usage

```powershell
python scripts/research/okvqa_val_prompt_abc_infer/run_okvqa_val_prompt_abc_infer.py `
  --questions D:\path\to\OpenEnded_mscoco_val2014_questions.json `
  --annotations D:\path\to\mscoco_val2014_annotations.json `
  --image-root D:\path\to\okvqa\images `
  --output-dir D:\path\to\outputs\okvqa_val_prompt_abc_eval `
  --model google/gemma-3-4b-it `
  --correct-rule vqa_0.3 `
  --max-new-tokens 16
```

If you prefer the existing project setup, replace `--model` with:

```powershell
--transcoder-set tianhux2/gemma3-4b-it-plt
```

## Outputs

- `promptA_eval.csv`
- `promptB_eval.csv`
- `promptC_eval.csv`
- `summary.txt`

## Prompt templates used in the script

- A: `{question} Think step by step from visual evidence, then reply exactly in the format: The answer is <short answer>.`
- B: `{question} Reply exactly in the format: The answer is <short answer>.`
- C (one-shot):
  - Below are an instruction that describes a task along with a reference answer. Using the reference answer as a guide, write your own response.
  - ### Example Instruction:
  - `{example_instruction}`
  - ### Example Response:
  - `{example_response}`
  - ### Instruction:
  - `{question}`
  - ### Response:
