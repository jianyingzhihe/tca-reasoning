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
