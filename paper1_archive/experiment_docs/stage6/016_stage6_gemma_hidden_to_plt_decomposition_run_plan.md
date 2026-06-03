# Stage6-016 Gemma Hidden-to-PLT Decomposition Run Plan

## Purpose

Test whether Gemma's confirmed hidden-level evidence-to-answer flow is captured by sparse PLT features, PLT reconstruction error, or the existing source-tracing route object.

This was designed to fill the symmetry counterpart with Qwen Stage4-038, which already showed that Qwen hidden effects are mostly retained in PLT reconstruction error rather than sparse topK reconstruction.

Completion note, 2026-06-02:

```text
This planned symmetry gap has been filled. The completed verdict is recorded in
Stage6-017 and the current cross-model completeness audit is Stage6-024.
```

## Design

Use Gemma's confirmed hidden gate:

```text
layer = 1
direction = restore
position group = visual+answer
mask condition = union_mask
```

For each prompt-run and mask condition, compare:

```text
hidden_residual
plt_topk_reconstruction
plt_reconstruction_error
source_tracing_route
```

`source_tracing_route` is imported from existing Stage3 Gemma source-tracing artifacts. It is not rerun here, so this experiment does not generate persistent `.pt` graph files.

## Metrics

Primary metrics:

```text
target_effect
rank_effect
correct_minus_wrong
real_minus_shifted
real_minus_shuffled
topk_over_hidden_retention
error_over_hidden_retention
```

TopK values:

```text
8,16,32,64,128
```

Position groups:

```text
visual+answer
top_hidden_delta_16
top_hidden_delta_32
```

## Acceptance

The experiment is interpretable if primary and strict both produce non-empty rows for all three operators and the analyzer can compare them to Qwen Stage4-038.

Decision statuses:

```text
gemma_hidden_flow_sparse_plt_captured
gemma_hidden_flow_route_captured_but_topk_partial
gemma_hidden_flow_error_heavy_like_qwen
gemma_hidden_to_plt_decomp_blocked
```

## Boundary

This is a decomposition experiment, not a new main-claim dependency. A mixed or blocked result does not weaken the existing Gemma PLT source-tracing route or the Gemma/Qwen hidden symmetric result.
