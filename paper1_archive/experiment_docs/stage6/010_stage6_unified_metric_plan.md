# Stage6-010: Gemma/Qwen Unified Metric Plan

## Summary

Stage6-010 fixes the earlier comparison problem: Qwen Stage6 measured fixed feature-node causal strength, while Gemma Stage6 measured sparse source-tracing graph overlap. This stage reports both models under two explicit lenses:

- Fixed-node causal strength: keep feature nodes fixed and test whether they still affect the answer token across text rewrites and prompt families.
- Route identity stability: rebuild or re-rank the route set for each prompt/text condition and compare top route identity with the `B_direct + original` baseline.

This is exploratory secondary-claim work. It does not replace Stage4 causal route evidence.

## Key Design

- Do not compare Qwen logit-effect numbers directly to Gemma Jaccard overlap numbers.
- Gemma old `mf8_sharded` graph-overlap rows are first split into aligned rows and format/target-token failures.
- Gemma fixed-node analysis only uses rows where `prefix_ok=1` and `target_token_same=1` for the main metric.
- Qwen route identity is Qwen-native: it compares top route-first feature sets, not Gemma edge graphs.

## Outputs

- `stage6_unified_fixednode_*`: Gemma fixed baseline feature-node ablation.
- `stage6_unified_routeidentity_*`: Qwen prompt/text route-first revalidation and topK overlap.
- `stage6_unified_crossmodel_*`: old-result reanalysis plus unified cross-model summary.

## Decision Boundary

Positive result supports: prompt/text/CoT modulates evidence-to-answer routes in both models under shared analysis lenses.

It does not support: Qwen and Gemma have identical sparse graphs, or Qwen fully replicates Gemma-style automatic source tracing.
