# Phase A+B Execution Plan (Week 1)

This plan is for your current goal: move from "pipeline is runnable" to "small reproducible study set with structured traces".

## Scope

- Phase A: reproducibility + experiment hygiene
- Phase B: small dataset with external trace (30-50 samples first)

## Week-1 Day-by-Day

### Day 1 - Freeze runtime config

Goals:
- lock a stable runtime config for all first-round experiments
- avoid changing multiple variables at once

Tasks:
- copy `research/templates/repro_config.yaml` to `research/work/repro_config.yaml`
- set model/transcoder/dtype/topk/offload in that file
- run 1 smoke sample and save outputs under `outputs/phase_ab`

Success criteria:
- one sample runs end-to-end with the frozen config

---

### Day 2 - Build sample manifest (30-50 items)

Goals:
- define the first controlled sample set

Tasks:
- fill `research/work/sample_manifest.csv` from template
- ensure each sample has: `sample_id,image_path,question`
- keep categories balanced (visual grounding / knowledge recall / relation reasoning)

Success criteria:
- manifest has at least 30 valid rows

---

### Day 3 - Build external trace file

Goals:
- create trace annotations per sample in one unified format

Tasks:
- fill `research/work/dataset_with_trace.jsonl`
- for each sample, include:
  - final answer
  - ordered `trace_steps`
  - step type labels

Success criteria:
- at least 30 samples contain parseable step lists

---

### Day 4 - Batch graph generation (small)

Goals:
- produce `.pt` and graph json for the small set

Tasks:
- run batch loop using `sample_manifest.csv`
- write outputs to:
  - `outputs/phase_ab/raw_pt`
  - `outputs/phase_ab/graph_json`
  - `outputs/phase_ab/logs`

Success criteria:
- >80% samples complete without crash

---

### Day 5 - Step candidate extraction (first pass)

Goals:
- map each external step to candidate internal subgraph

Tasks:
- for each sample + step:
  - compute top candidate nodes/edges
  - store in `research/work/step_alignment.csv`
- use `research/templates/step_alignment_template.csv` columns

Success criteria:
- each sample has at least one non-empty step mapping

---

### Day 6 - Quality check + error analysis

Goals:
- inspect failure modes before scaling

Tasks:
- compute quick stats:
  - graph size
  - error-node ratio
  - empty-step ratio
- manually inspect 5 success + 5 failure samples

Success criteria:
- one-page summary with actionable tuning decisions

---

### Day 7 - Freeze v1 baseline package

Goals:
- make a reusable baseline package for Phase C (causal tests)

Tasks:
- freeze:
  - config
  - sample list
  - trace file
  - first alignment csv
- write short experiment note (`research/work/week1_summary.md`)

Success criteria:
- a teammate can reproduce the same results with your files only

---

## Deliverables by end of Week 1

- `research/work/repro_config.yaml`
- `research/work/sample_manifest.csv` (>=30 rows)
- `research/work/dataset_with_trace.jsonl`
- `research/work/step_alignment.csv` (first pass)
- `research/work/week1_summary.md`

## What this unlocks next

After A+B, you can move to Phase C:
- step-level causal interventions (ablation/patching)
- bypass detection and mediation metrics
- structured-trace vs free-CoT comparisons

