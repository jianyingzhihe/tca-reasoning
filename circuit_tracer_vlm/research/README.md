# Research Workspace (Phase A+B)

## Quick start

```bash
python scripts/research/setup_phase_ab.py --root .
```

This creates:
- `research/work/repro_config.yaml`
- `research/work/sample_manifest.csv`
- `research/work/dataset_with_trace.jsonl`
- `research/work/step_alignment.csv`
- `outputs/phase_ab/{raw_pt,graph_json,metrics,figures,logs}`

## Batch run (attribute only)

```bash
python scripts/research/run_batch_attribute.py \
  --manifest research/work/sample_manifest.csv \
  --output-dir outputs/phase_ab/raw_pt \
  --transcoder-set tianhux2/gemma3-4b-it-plt \
  --dtype bfloat16 \
  --max-feature-nodes 256 \
  --offload cpu
```

## Build manifest from OK-VQA

```bash
python scripts/research/okvqa_to_manifest.py \
  --questions ~/data/okvqa/questions/OpenEnded_mscoco_val2014_questions.json \
  --annotations ~/data/okvqa/annotations/mscoco_val2014_annotations.json \
  --image-root ~/data/okvqa/images \
  --split val \
  --output research/work/sample_manifest_okvqa_val200.csv \
  --limit 200 \
  --seed 42
```

Then run attribution:

```bash
python scripts/research/run_batch_attribute.py \
  --manifest research/work/sample_manifest_okvqa_val200.csv \
  --output-dir outputs/phase_ab/raw_pt_okvqa_val200 \
  --transcoder-set tianhux2/gemma3-4b-it-plt \
  --dtype bfloat16 \
  --max-feature-nodes 256 \
  --offload cpu
```

## Long run with low disk usage (metrics-only stream)

This mode computes metrics per sample and deletes temporary `.pt` files by default.
It supports resume and progress/ETA logs.
If `--manifest` does not exist, the script auto-builds it from OK-VQA files under
`~/tca-reasoning/data/okvqa` by default.

```bash
python scripts/research/run_metrics_stream.py \
  --manifest research/work/sample_manifest_okvqa_val1000.csv \
  --metrics-output outputs/phase_ab/metrics/okvqa_val1000_metrics_stream.csv \
  --summary-output outputs/phase_ab/metrics/okvqa_val1000_summary.txt \
  --temp-pt-dir outputs/phase_ab/tmp_pt \
  --transcoder-set tianhux2/gemma3-4b-it-plt \
  --dtype bfloat16 \
  --max-feature-nodes 96 \
  --max-n-logits 2 \
  --batch-size 1 \
  --offload cpu \
  --topk 8 \
  --log-every 10 \
  --reuse-model \
  --okvqa-split val \
  --okvqa-limit 1000
```

Run in background:

```bash
nohup python scripts/research/run_metrics_stream.py \
  --manifest research/work/sample_manifest_okvqa_val1000.csv \
  --metrics-output outputs/phase_ab/metrics/okvqa_val1000_metrics_stream.csv \
  --summary-output outputs/phase_ab/metrics/okvqa_val1000_summary.txt \
  --temp-pt-dir outputs/phase_ab/tmp_pt \
  --transcoder-set tianhux2/gemma3-4b-it-plt \
  --dtype bfloat16 \
  --max-feature-nodes 96 \
  --max-n-logits 2 \
  --batch-size 1 \
  --offload cpu \
  --topk 8 \
  --log-every 10 \
  > outputs/phase_ab/logs/night_metrics_stream.log 2>&1 &
```

## Three-case analysis (low/mid/high)

1) Pick three cases from metrics and generate a mini manifest:

```bash
python scripts/research/analyze_three_cases.py \
  --metrics-csv outputs/phase_ab/metrics/okvqa_val1000_metrics_stream.csv \
  --manifest-csv research/work/sample_manifest_okvqa_val1000.csv \
  --output-dir outputs/phase_ab/case3 \
  --select-by replacement_score
```

2) Re-run only those 3 samples to get `.pt` files:

```bash
python scripts/research/run_batch_attribute.py \
  --manifest outputs/phase_ab/case3/selected_manifest.csv \
  --output-dir outputs/phase_ab/case3/raw_pt \
  --transcoder-set tianhux2/gemma3-4b-it-plt \
  --dtype bfloat16 \
  --max-feature-nodes 128 \
  --max-n-logits 2 \
  --batch-size 1 \
  --offload cpu \
  --topk 8
```

3) Generate feature/error/token contribution tables for those 3 cases:

```bash
python scripts/research/analyze_three_cases.py \
  --metrics-csv outputs/phase_ab/metrics/okvqa_val1000_metrics_stream.csv \
  --manifest-csv research/work/sample_manifest_okvqa_val1000.csv \
  --output-dir outputs/phase_ab/case3 \
  --select-by replacement_score \
  --pt-dir outputs/phase_ab/case3/raw_pt
```

## Plan document

See:
- `research/PHASE_AB_PLAN.md`

## Prompt A/B circuit comparison workflow

This is the recommended pipeline for your current research question:
"same model + same dataset + different prompts => how circuits differ".

Follow-up tests after Stage 1 summaries live in:
- `research/STAGE1_NEXT_TESTS.md`

### 1) Build paired A/B manifests

```bash
python scripts/research/build_prompt_ab_manifests.py \
  --base-manifest research/work/sample_manifest_okvqa_val200.csv \
  --out-manifest-a research/work/manifest_ab_promptA.csv \
  --out-manifest-b research/work/manifest_ab_promptB.csv \
  --prompt-a-template "{question} Think step by step from visual evidence, then reply exactly in the format: The answer is <short answer>." \
  --prompt-b-template "{question} Reply exactly in the format: The answer is <short answer>." \
  --copy-gold-from-notes
```

Notes:
- keep `sample_id` aligned between A/B (script does this automatically)
- enforce a consistent output style in both prompts (for token-position alignment)

### 2) Run deterministic eval first (accuracy + correct flags)

```bash
python scripts/research/run_batch_eval.py \
  --manifest research/work/manifest_ab_promptA.csv \
  --output-csv outputs/phase_ab/eval/promptA_eval.csv \
  --transcoder-set tianhux2/gemma3-4b-it-plt \
  --annotations-json ~/tca-reasoning/data/okvqa/annotations/mscoco_val2014_annotations.json \
  --correct-rule vqa_0.3 \
  --max-new-tokens 16 \
  --log-every 20
```

```bash
python scripts/research/run_batch_eval.py \
  --manifest research/work/manifest_ab_promptB.csv \
  --output-csv outputs/phase_ab/eval/promptB_eval.csv \
  --transcoder-set tianhux2/gemma3-4b-it-plt \
  --annotations-json ~/tca-reasoning/data/okvqa/annotations/mscoco_val2014_annotations.json \
  --correct-rule vqa_0.3 \
  --max-new-tokens 16 \
  --log-every 20
```

Notes:
- output csv keeps `question_id` + `image_id` (parsed from manifest notes)
- keeps `generated_text` + extracted `predicted_answer`
- includes VQA-style `vqa_score` and binary `correct` by `--correct-rule`

### 3) Build A/B bucket file (A1_B0, A0_B1, ...)

```bash
python scripts/research/build_ab_bucket_csv.py \
  --run-a-csv outputs/phase_ab/eval/promptA_eval.csv \
  --run-b-csv outputs/phase_ab/eval/promptB_eval.csv \
  --correct-col correct \
  --require-same-pred \
  --out-bucket-csv outputs/phase_ab/metrics/prompt_ab_bucket.csv \
  --out-summary-txt outputs/phase_ab/metrics/prompt_ab_bucket_summary.txt
```

### 4) Optional: export bucket examples for manual focus

```bash
python scripts/research/export_bucket_examples.py \
  --bucket-csv outputs/phase_ab/metrics/prompt_ab_bucket.csv \
  --manifest-csv research/work/manifest_ab_promptA.csv \
  --out-csv outputs/phase_ab/metrics/prompt_ab_focus_examples.csv \
  --per-bucket 25 \
  --buckets A1_B0,A0_B1
```

### 5) Run attribution for A and B

```bash
python scripts/research/run_batch_attribute.py \
  --manifest research/work/manifest_ab_promptA.csv \
  --output-dir outputs/phase_ab/raw_pt_promptA \
  --transcoder-set tianhux2/gemma3-4b-it-plt \
  --dtype bfloat16 \
  --max-feature-nodes 128 \
  --max-n-logits 2 \
  --batch-size 1 \
  --offload cpu \
  --topk 8
```

```bash
python scripts/research/run_batch_attribute.py \
  --manifest research/work/manifest_ab_promptB.csv \
  --output-dir outputs/phase_ab/raw_pt_promptB \
  --transcoder-set tianhux2/gemma3-4b-it-plt \
  --dtype bfloat16 \
  --max-feature-nodes 128 \
  --max-n-logits 2 \
  --batch-size 1 \
  --offload cpu \
  --topk 8
```

### 6) Compare circuits (sample-level + bucket-level)

```bash
python scripts/research/compare_prompt_ab_circuits.py \
  --pt-dir-a outputs/phase_ab/raw_pt_promptA \
  --pt-dir-b outputs/phase_ab/raw_pt_promptB \
  --bucket-csv outputs/phase_ab/metrics/prompt_ab_bucket.csv \
  --out-sample-csv outputs/phase_ab/metrics/prompt_ab_circuit_sample.csv \
  --out-summary-csv outputs/phase_ab/metrics/prompt_ab_circuit_summary.csv \
  --target-logit-rank 0 \
  --parents-per-node 8 \
  --max-depth 48 \
  --log-every 20
```

Main metrics included:
- structure: `traced_max_depth`, `traced_nodes`, `traced_edges`
- concentration: `target_top1/top3/top10_concentration`
- composition: `target_error_ratio`, `target_feature_ratio`, `target_token_ratio`
- interpretability: `replacement_score`, `completeness_score`
- stability (A vs B overlap): `node_overlap_jaccard`, `edge_overlap_jaccard`

### 7) Deterministic sanity check (same prompt, run twice)

Run the same manifest twice into two dirs (e.g. `raw_pt_run1`, `raw_pt_run2`) and compare:

```bash
python scripts/research/compare_prompt_ab_circuits.py \
  --pt-dir-a outputs/phase_ab/raw_pt_run1 \
  --pt-dir-b outputs/phase_ab/raw_pt_run2 \
  --out-sample-csv outputs/phase_ab/metrics/repeat_sample.csv \
  --out-summary-csv outputs/phase_ab/metrics/repeat_summary.csv \
  --skip-graph-scores
```

Expected (deterministic setup):
- overlaps close to 1
- deltas close to 0

## Prompt A/B/C comparison (includes one-shot C)

If you want three prompts:
- A: structured reasoning cue
- B: weak instruction baseline
- C: one-shot style imitation

### 1) Build A/B/C manifests

```bash
python scripts/research/build_prompt_ab_manifests.py \
  --base-manifest research/work/sample_manifest_okvqa_val200.csv \
  --out-manifest-a research/work/manifest_promptA.csv \
  --out-manifest-b research/work/manifest_promptB.csv \
  --out-manifest-c research/work/manifest_promptC_oneshot.csv \
  --prompt-a-template "{question} Think step by step from visual evidence, then reply exactly in the format: The answer is <short answer>." \
  --prompt-b-template "{question} Reply exactly in the format: The answer is <short answer>." \
  --prompt-c-template "Below are an instruction that describes a task along with a reference answer. Using the reference answer as a guide, write your own response.\n### Example Instruction:\n{example_instruction}\n### Example Response:\n{example_response}\n### Instruction:\n{question}\n### Response:" \
  --example-instruction "What color is the bus?" \
  --example-response "The answer is yellow." \
  --copy-gold-from-notes
```

### One-command server run (A/B parallel, then C)

This runner builds A/B/C manifests and runs eval as:
- Prompt A + Prompt B in parallel (2 processes)
- Prompt C after A/B complete

Each process uses `run_batch_eval.py`, which loads the model once and reuses it for all samples.

```bash
bash scripts/server/run_okvqa_prompt_abc_eval.sh \
  research/work/sample_manifest_okvqa_val1000.csv
```

Optional env vars before run:

```bash
export RUN_TAG=okvqa_val1000_abc
export TRANSCODER_SET=tianhux2/gemma3-4b-it-plt
export OKVQA_ANN=~/tca-reasoning/data/okvqa/annotations/mscoco_val2014_annotations.json
```

### 2) Run three attribution batches

```bash
python scripts/research/run_batch_attribute.py --manifest research/work/manifest_promptA.csv --output-dir outputs/phase_ab/raw_pt_promptA --transcoder-set tianhux2/gemma3-4b-it-plt --dtype bfloat16 --max-feature-nodes 128 --max-n-logits 2 --batch-size 1 --offload cpu --topk 8
python scripts/research/run_batch_attribute.py --manifest research/work/manifest_promptB.csv --output-dir outputs/phase_ab/raw_pt_promptB --transcoder-set tianhux2/gemma3-4b-it-plt --dtype bfloat16 --max-feature-nodes 128 --max-n-logits 2 --batch-size 1 --offload cpu --topk 8
python scripts/research/run_batch_attribute.py --manifest research/work/manifest_promptC_oneshot.csv --output-dir outputs/phase_ab/raw_pt_promptC --transcoder-set tianhux2/gemma3-4b-it-plt --dtype bfloat16 --max-feature-nodes 128 --max-n-logits 2 --batch-size 1 --offload cpu --topk 8
```

### 3) Run three eval batches first

```bash
python scripts/research/run_batch_eval.py --manifest research/work/manifest_promptA.csv --output-csv outputs/phase_ab/eval/promptA_eval.csv --transcoder-set tianhux2/gemma3-4b-it-plt --annotations-json ~/tca-reasoning/data/okvqa/annotations/mscoco_val2014_annotations.json --correct-rule vqa_0.3 --max-new-tokens 16 --log-every 20
python scripts/research/run_batch_eval.py --manifest research/work/manifest_promptB.csv --output-csv outputs/phase_ab/eval/promptB_eval.csv --transcoder-set tianhux2/gemma3-4b-it-plt --annotations-json ~/tca-reasoning/data/okvqa/annotations/mscoco_val2014_annotations.json --correct-rule vqa_0.3 --max-new-tokens 16 --log-every 20
python scripts/research/run_batch_eval.py --manifest research/work/manifest_promptC_oneshot.csv --output-csv outputs/phase_ab/eval/promptC_eval.csv --transcoder-set tianhux2/gemma3-4b-it-plt --annotations-json ~/tca-reasoning/data/okvqa/annotations/mscoco_val2014_annotations.json --correct-rule vqa_0.3 --max-new-tokens 16 --log-every 20
```

### 4) Compare three prompts (AB/AC/BC automatically)

Prepare three evaluation csv files (A/B/C) with either:
- `sample_id + correct`, or
- `sample_id + predicted_answer + gold_answer`

Then:

```bash
python scripts/research/compare_prompt_abc.py \
  --pt-dir-a outputs/phase_ab/raw_pt_promptA \
  --pt-dir-b outputs/phase_ab/raw_pt_promptB \
  --pt-dir-c outputs/phase_ab/raw_pt_promptC \
  --eval-a-csv outputs/phase_ab/eval/promptA_eval.csv \
  --eval-b-csv outputs/phase_ab/eval/promptB_eval.csv \
  --eval-c-csv outputs/phase_ab/eval/promptC_eval.csv \
  --out-dir outputs/phase_ab/metrics/prompt_abc \
  --correct-col correct \
  --require-same-pred \
  --target-logit-rank 0 \
  --parents-per-node 8 \
  --max-depth 48 \
  --log-every 20
```

Key outputs:
- `outputs/phase_ab/metrics/prompt_abc/prompt_abc_overview.csv`
- `outputs/phase_ab/metrics/prompt_abc/buckets/*`
- `outputs/phase_ab/metrics/prompt_abc/pairwise/*`
