# Stage6-009 Gemma Disk Rescue And Compact Rerun

## Purpose

This note records the Stage6 Gemma prompt/text/CoT engineering failure and the repaired rerun policy.

The `gemmaprompt_v1_mf32` full run is not a scientific negative result. It is an engineering-blocked run:

```text
status: blocked_by_disk_full
failed_step: attribute_B_baseline_gold_answer_unique
cause: /root/autodl-tmp reached 100% while writing Gemma attribution graph .pt files
```

## What Happened

The full run used `max_feature_nodes=32`. During A-side graph attribution, the run wrote roughly `54G` of `.pt` graph files before the B-side baseline attribution began. The remote disk then reached:

```text
/root/autodl-tmp: 350G used, 249M available, 100%
```

The B-side baseline graph save failed with `OSError: [Errno 28] No space left on device` and torch serialization stream errors. The compare CSVs were empty, so the run cannot support any Stage6 Gemma route-stability conclusion.

Local CSV/JSON evidence was fetched and preserved:

```text
decision.json
eval_A_condition.csv
eval_B_baseline.csv
valid_samples.csv
failure_manifest.csv
meta_a.csv
meta_b.csv
```

## Rescue Action

Only failed Stage6 Gemma graph material was removed remotely. Model caches, Stage3 Gemma main evidence, datasets, and older cross-model artifacts were not touched.

Cleaned path:

```text
/root/autodl-tmp/tca-reasoning/stage6_gemma_prompt_text_cot/source_tracing_full_gemmaprompt_v1_mf32/graphs_a_condition
```

The failed run directory was reduced from about `54G` to about `400K`, and remote free space recovered to about `54G`.

## Runner Fix

`scripts/local/run_stage6_gemma_prompt_text_cot_remote.py` now has hard disk safety gates:

```text
smoke: require >= 20G free
full compact, max_feature_nodes <= 8: require >= 80G free
full large graph, max_feature_nodes > 8: blocked unless --allow-large-graphs and require >= 160G free
```

The default `--max-feature-nodes` is now `8`, not `64`.

The runner also logs `df -h` and `du -sh` before/after major steps, and the B-baseline graph materialization no longer falls back to copying `.pt` files. It only permits hardlinks or symlinks; if both fail, the run is blocked rather than duplicating large graph files.

The runner now also supports sharded streaming full runs:

```text
--sharded-streaming
--shard-count 8
--cleanup-graphs-after-compare
```

In sharded mode, the full manifest is split by original sample, each shard is evaluated / attributed / compared independently, and graph `.pt` files are deleted after that shard's compare CSVs are produced. Sharded full with `max_feature_nodes <= 8` requires `>=35G` free disk; monolithic full still requires `>=80G`.

## Compact Rerun Policy

The old tag remains:

```text
gemmaprompt_v1_mf32: blocked_by_disk_full
```

The replacement tag is:

```text
gemmaprompt_v1_mf8
```

Rerun order:

```text
1. smoke with max_feature_nodes=8
2. fetch and analyze smoke
3. full compact only if remote free disk >= 80G
4. if free disk remains below 80G, do not start full compact
```

At the time of this note, Stage6 Gemma itself has been cleaned down to about `4M`, but the remote still has only about `54G` free because the remaining large directories are older assets:

```text
circuit_tracer_vlm/outputs/phase_ab: about 107G
stage3_gemma_paperpack: about 60G
data/hf_cache and datasets: large but intentionally untouched
```

Therefore a full compact run must either wait for explicit cleanup approval of older outputs, or be redesigned as streaming/sharded graph comparison that deletes `.pt` graphs after each shard is compared.

## Claim Boundary

This engineering failure does not affect the Stage4 main claim:

```text
Gemma has a sparse evidence-to-answer source-tracing route.
Qwen has hidden-level route support and strict-supported route-first feature nodes, while grouped feature route remains unresolved.
```

Stage6 remains exploratory. Until `gemmaprompt_v1_mf8` produces non-empty compare CSVs, Gemma Stage6 prompt/text/CoT robustness should be treated as pending rather than failed.

Superseded status note, 2026-06-02:

```text
This pending statement was a run-time boundary for the disk-rescue phase.
It has been superseded by the later `gemmaprompt_v1_mf8_sharded` and
`unified_v1_focus4` Stage6 analyses. Use Stage6-013 and Stage6-024 for the
current cross-model prompt/text symmetry status.
```

## Compact Smoke Result

`gemmaprompt_v1_mf8` smoke was launched after the rescue and passed end-to-end.

Remote decision:

```text
status: pass
eval_a_rows: 24
eval_b_rows: 24
valid_sample_rows: 24
failure_rows: 0
sample_compare_rows: 24
nodes_detailed_rows: 1007
edges_detailed_rows: 1631
graph_a_files: 24
graph_b_files: 24
graph_success_rate_vs_valid: 1.0
```

Local analyzer summary:

```text
status: gemma_stage6_source_tracing_prompt_text_cot_analyzed
rows: 24
unique_original_samples: 3
valid_rate: 1.0
overall_node_overlap_jaccard_mean: 0.2872
overall_edge_overlap_jaccard_mean: 0.2462
text_stability_hint: false
prompt_modulation_hint: true
```

The smoke graphs used about `12G` remotely. After fetching all CSV/JSON artifacts, those remote graph directories were safely cleaned and the smoke run directory was reduced to about `92K`.

The full compact run was not started because the remote disk returned to only about `54G` free, below the repaired `80G` hard gate. This is intentional: the smoke size shows that even `max_feature_nodes=8` can produce multi-GB graph directories, so full should not run monolithically on the current disk state.

Next valid choices:

```text
1. explicitly approve cleanup of older circuit_tracer_vlm/outputs/phase_ab artifacts, then run full compact;
2. implement and run a sharded/streaming full that compares one shard at a time and deletes .pt graphs after each shard;
3. keep Gemma Stage6 at smoke-only exploratory status and use Qwen Stage6 full for the current secondary claim.
```

## Sharded Full Launch

The sharded streaming option was implemented and launched on AutoDL/gpu1:

```text
tag: gemmaprompt_v1_mf8_sharded
mode: full
max_feature_nodes: 8
shard_count: 8
cleanup_graphs_after_compare: enabled
remote pid: 504875
remote log: /root/autodl-tmp/tca-reasoning/stage6_gemma_prompt_text_cot/logs/gemma_prompt_text_cot_full_20260530_195609.log
```

Initial status:

```text
remote free disk: 54G
sharded gate: 35G
shards built: 8
conditions: 144
original samples: 12
first active step: shard_00 eval_A_condition
```

This run is the official Gemma Stage6 full counterpart. `gemmaprompt_v1_mf32` remains engineering-blocked, and `gemmaprompt_v1_mf8` remains smoke-only.
