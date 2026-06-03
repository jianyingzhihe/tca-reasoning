# Stage6-017 Hidden-to-PLT Cross-Model Verdict

Updated: 2026-06-02 12:22:02

## Status

- Gemma decomposition status: `gemma_hidden_flow_error_heavy_like_qwen`.
- Qwen decomposition status: `qwen_route_may_live_in_plt_error`.
- Interpretation: under the shared hidden-to-PLT decomposition lens, both models show hidden-level evidence-to-answer flow, but sparse PLT topK reconstruction does not carry the main hidden effect. The effect is instead retained by the PLT reconstruction error / non-topK residual lens.

## Gemma Full Gates

Primary hidden residual gate:

```text
operator = hidden_residual
mask_condition = union_mask
position_group = visual+answer
n = 37
target_effect mean = 9.2736
CI low = 5.0034
CI high = 13.6182
positive_frac = 0.7297
```

Strict hidden residual gate:

```text
operator = hidden_residual
mask_condition = union_mask
position_group = visual+answer
n = 33
target_effect mean = 10.4735
CI low = 5.7841
CI high = 14.9621
positive_frac = 0.7576
```

Sparse PLT topK reconstruction:

```text
primary_topk_gate = null
strict_topk_gate = null
topk_retention_mean = -0.0040
```

PLT reconstruction error:

```text
primary_error_gate = passed
strict_error_gate = passed
best_top_k = 32
primary target_effect mean = 9.2196
primary CI low = 4.9628
strict target_effect mean = 10.6705
strict CI low = 6.0947
error_retention_mean = 0.9949
```

## Qwen Reference Gates

Qwen's matching hidden-to-PLT mediation result comes from Stage4-042. The strict full result uses the already-supported Qwen hidden route lens at layer 14.

Qwen hidden residual:

```text
operator = hidden_residual
mask_condition = answer_mask
n = 144
target_effect mean = 0.8205
CI low = 0.5860
positive_frac = 0.7639
```

Qwen sparse PLT topK reconstruction:

```text
topK = 8:   mean = -0.0050, CI low = -0.0162, positive_frac = 0.3403
topK = 16:  mean =  0.0010, CI low = -0.0103, positive_frac = 0.3333
topK = 32:  mean =  0.0005, CI low = -0.0104, positive_frac = 0.3056
topK = 64:  mean =  0.0013, CI low = -0.0110, positive_frac = 0.3125
topK = 128: mean =  0.0094, CI low = -0.0062, positive_frac = 0.3889
```

Qwen PLT reconstruction error:

```text
topK = 8:   mean = 0.8105, CI low = 0.5794, positive_frac = 0.7778
topK = 16:  mean = 0.7980, CI low = 0.5709, positive_frac = 0.7639
topK = 32:  mean = 0.7973, CI low = 0.5733, positive_frac = 0.7847
topK = 64:  mean = 0.7729, CI low = 0.5516, positive_frac = 0.7708
topK = 128: mean = 0.7777, CI low = 0.5546, positive_frac = 0.7708
```

Qwen therefore shows the same qualitative pattern as Gemma under this decomposition lens: hidden residual is supported, sparse PLT topK reconstruction is near zero and does not pass, while PLT reconstruction error nearly reproduces the hidden residual effect.

## Cross-Model Reading

The original expectation was that Gemma might differ from Qwen because Gemma's hidden flow would be more fully captured by sparse PLT topK features. The result does not support that topK-specific version:

```text
Gemma hidden residual: supported
Gemma PLT topK reconstruction: not supported
Gemma PLT reconstruction error: supported, nearly full retention
Qwen hidden residual: supported
Qwen PLT topK reconstruction: not supported
Qwen PLT reconstruction error: supported, nearly full retention
```

This does not weaken Gemma's source-tracing result. Gemma still has a strong sparse source-tracing route object at the graph level. The new conclusion is narrower: when we decompose a hidden residual patch into sparse PLT topK reconstruction versus reconstruction error, the answer-supporting hidden effect is mostly not captured by the local topK reconstruction operator.

## Boundary

This compares hidden-to-PLT decomposition lenses. It does not compare raw logit magnitudes across models, does not claim Gemma/Qwen sparse topology is identical, and does not say Gemma lacks a sparse source-tracing route. The safe wording is:

```text
Both Gemma and Qwen support hidden-level evidence-to-answer flow. Under the tested hidden-to-PLT decomposition, the local sparse topK reconstruction does not explain most of that flow; the reconstruction error lens carries the main effect. Gemma nevertheless remains the stronger sparse graph/source-tracing positive case.
```
