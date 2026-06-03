# Stage4-017 Qwen Expanded Source Tracing V2

## 目的

检查之前 Qwen automatic source tracing 不成立是否来自 topK/candidate-pool 太窄，而不是机制不存在。

## 输入

- `paperpack72_primary`
- `paperpack72_strict_sensitivity`
- Qwen layer `26`
- Sensitivity: layer `22`, layer `24`, `visual_only`, `answer_adjacent_only`

## 方法

Main settings:

- `max_feature_nodes=256`
- `candidate_pool_size=8192`
- `compare_topk_per_node=8`
- `top_features_per_sample=32`
- `position_filter=visual_answer`

Sensitivity:

- layer `22/24`, `top_features_per_sample=16`
- position filters `visual_only` and `answer_adjacent_only`, `top_features_per_sample=16`

## 结果

- `stage4_qwen_source_tracing_primary_full_expanded_v2_L26_top32_*`
- `stage4_qwen_source_tracing_strict_full_expanded_v2_L26_top32_*`
- sensitivity artifacts for L22/L24 and position-only filters

Main L26 top32:

- primary graph success: `140/144` prompt-runs, graph success rate `0.972222`
- primary intervention rows: `2240`
- primary decision: `qwen_source_tracing_not_supported`
- primary best negative target-logit fraction: `0.0991071`
- strict graph success: `140/144` prompt-runs, graph success rate `0.972222`
- strict intervention rows: `2240`
- strict decision: `qwen_source_tracing_not_supported`
- strict best negative target-logit fraction: `0.0866071`

Sensitivity:

- L22 primary top16: `qwen_source_tracing_not_supported`, best negative fraction `0.220536`
- L24 primary top16: `qwen_source_tracing_not_supported`, best negative fraction `0.187612`
- L26 visual-only top16: `qwen_source_tracing_not_supported`, best negative fraction `0.0875`
- L26 answer-adjacent-only top16: `qwen_source_tracing_not_supported`, best negative fraction `0.376229`

Important implementation note:

- The requested `top32` intervention was accepted by the runner, but `trace_compare_ab_controlled.py` still emitted about `8` feature nodes per prompt-run. This means automatic source tracing remains sparse; the bottleneck is route extraction/compare, not just intervention topK.

## 预期与实际偏差

- Disk initially filled during a larger graph-node run. We cleaned failed/already-fetched remote graph run roots and reran with graph selected nodes reduced to `128`, while preserving top32 intervention request.
- Two long-sequence samples exceeded the current `max_n_pos=512`, leaving `70/72` matched samples. This is acceptable for the current full-run threshold but should be reported.
- Expanded settings did not rescue automatic source tracing.

## 结论

Expanded Qwen automatic source tracing remains not supported under the current hook-aligned adapter.

Allowed wording:

`Expanded Qwen source tracing ran successfully, but automatic route selection still does not produce a robust Gemma-style source route.`

Not allowed:

`Qwen lacks evidence-sensitive mechanisms.`
