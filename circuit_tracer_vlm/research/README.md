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

## Plan document

See:
- `research/PHASE_AB_PLAN.md`

