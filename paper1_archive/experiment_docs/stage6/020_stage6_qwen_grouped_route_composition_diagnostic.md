# Stage6-020 Qwen Grouped Route Composition Diagnostic

Updated: 2026-06-02

## Question

Qwen has:

```text
hidden-level route support;
strict-supported individual feature nodes;
grouped sparse feature-route closure unresolved.
```

This experiment asks why grouped routes do not close under the current operator.

It does not try to prove a Gemma-style Qwen graph. It diagnoses whether non-closure comes from mixed node direction, layer mixing, or topK aggregation.

## Inputs

Primary inputs:

```text
stage4_qwen_route_first_*_routefirst_v1_* artifacts
stage4_qwen_feature_route_*_featureroute_v1_* artifacts
stage6_unified_routeidentity_full_unified_v1_focus4_* artifacts
```

The preferred candidate pool is strict-supported route-first feature nodes, because these already passed the node-level evidence-to-answer lens.

## Conditions

### 1. Support-only grouping

Build route bundles from nodes whose individual direction is support-positive:

```text
restore_source_minus_controls > 0
real_minus_shifted > 0 or real_minus_shuffled > 0
correct_minus_wrong > 0
```

Compare against mixed topK bundles from the previous grouped route experiment.

### 2. Layer-wise grouping

Avoid mixing all L10-L17 nodes into one route bundle. Test:

```text
L10-L12
L13-L15
L16-L17
L14-near
```

The diagnostic question is whether one layer band is route-like even when the full mixed bundle is unstable.

### 3. TopK curve and leave-one-out

For 5-10 strong cases:

```text
top1
top2
top4
top8
top16
leave-one-out from top8 or top16
```

This checks whether adding nodes monotonically helps or whether some nodes drag down the bundle.

## Metrics

Use the same family of gates as the grouped route work:

```text
route_evidence_specificity
route_correct_minus_wrong
clean route zeroing source > controls
real restore > shifted/shuffled
rank effect
positive fraction
```

Do not require one fixed topK to pass everything for a useful diagnostic. The desired output is a decomposition of failure modes.

## Decision Labels

```text
qwen_grouped_route_support_only_improves
qwen_grouped_route_layer_band_improves
qwen_grouped_route_topk_nonmonotonic
qwen_grouped_route_still_unresolved
qwen_grouped_route_diagnostic_blocked
```

## Smoke Result

Generated artifacts:

```text
stage6_defensive_qwen_grouped_composition_smoke_defensive_v1_topk_summary.csv
stage6_defensive_qwen_grouped_composition_smoke_defensive_v1_layer_band_summary.csv
stage6_defensive_qwen_grouped_composition_smoke_defensive_v1_topk_nonmonotonic.csv
stage6_defensive_qwen_grouped_composition_smoke_defensive_v1_decision.json
```

Current smoke status:

```text
status = qwen_grouped_composition_artifact_smoke_ready
route_metric_rows = 8
candidate_rows = 1104
nonmonotonic_routes = 2
nonmonotonic_frac = 0.5
```

TopK smoke summary:

```text
strict topK4:
route_restore_source_minus_controls_mean = -0.0208
route_real_minus_shifted_mean = 0.0625
gate2_frac = 0.75
gate3_frac = 0.25
gate4_frac = 0.25
gate5_frac = 0.25
all_gates_frac = 0.25

strict topK8:
route_restore_source_minus_controls_mean = -0.0208
route_real_minus_shifted_mean = 0.0313
route_correct_minus_wrong_mean = 0.0313
gate2_frac = 0.25
gate3_frac = 0.25
gate4_frac = 0.0
gate5_frac = 0.5
all_gates_frac = 0.0
```

Layer-band smoke summary:

```text
strict L10-L12:
restore_source_minus_controls_mean = 0.0734
real_minus_shifted_mean = 0.1387
real_minus_shuffled_mean = 0.1167
route_first_evidence_gold_frac = 0.5238

strict L13-L15:
restore_source_minus_controls_mean = 0.0478
real_minus_shifted_mean = 0.1360
real_minus_shuffled_mean = 0.1066
route_first_evidence_gold_frac = 0.4706

strict L16-L17:
restore_source_minus_controls_mean = 0.0786
real_minus_shifted_mean = 0.1108
real_minus_shuffled_mean = 0.1051
route_first_evidence_gold_frac = 0.6818
```

Nonmonotonic examples:

```text
okvqa_val_00528::D_visual_only:
best_topk = 4
best_restore_source_minus_controls = 0.0417
topK8 restore_source_minus_controls = -0.0208

okvqa_val_00528::B_direct:
best_topk = 4
best_restore_source_minus_controls = 0.0
topK8 restore_source_minus_controls = -0.0417
```

This smoke already points toward a useful diagnostic: layer-band and node-level statistics stay positive, but grouped bundle behavior is fragile and can worsen when more nodes are added.

## Full Artifact Diagnostic Result

The full artifact-level diagnostic has now completed under `defensive_v1`.

Generated artifacts:

```text
stage6_defensive_qwen_grouped_composition_full_defensive_v1_topk_summary.csv
stage6_defensive_qwen_grouped_composition_full_defensive_v1_layer_band_summary.csv
stage6_defensive_qwen_grouped_composition_full_defensive_v1_topk_nonmonotonic.csv
stage6_defensive_qwen_grouped_composition_full_defensive_v1_decision.json
```

Full status:

```text
mode = full
route_metric_rows = 435
candidate_rows = 24970
topk_summary_rows = 5
layer_band_rows = 6
nonmonotonic_routes = 63
nonmonotonic_frac = 0.7241
```

The JSON status string still says `qwen_grouped_composition_artifact_smoke_ready`, but the `mode`, row counts, and full output filenames confirm this is the full analyzer pass.

Grouped topK full summary:

```text
strict topK4:
route_restore_source_minus_controls_mean = 0.0028
route_real_minus_shifted_mean = 0.0247
route_real_minus_shuffled_mean = 0.0194
route_correct_minus_wrong_mean = 0.0487
all_gates_frac = 0.0690

strict topK8:
route_restore_source_minus_controls_mean = 0.0125
route_real_minus_shifted_mean = 0.0430
route_real_minus_shuffled_mean = 0.0268
route_correct_minus_wrong_mean = 0.0550
all_gates_frac = 0.0920

strict topK64:
route_restore_source_minus_controls_mean = 0.0060
route_real_minus_shifted_mean = 0.0272
route_real_minus_shuffled_mean = 0.0189
route_correct_minus_wrong_mean = 0.0866
all_gates_frac = 0.1149
```

Layer-band full summary stays positive:

```text
strict L10-L12:
restore_source_minus_controls_mean = 0.0642
real_minus_shifted_mean = 0.1173
real_minus_shuffled_mean = 0.1103
route_first_evidence_gold_frac = 0.6622

strict L13-L15:
restore_source_minus_controls_mean = 0.0626
real_minus_shifted_mean = 0.1192
real_minus_shuffled_mean = 0.1053
route_first_evidence_gold_frac = 0.7198

strict L16-L17:
restore_source_minus_controls_mean = 0.0618
real_minus_shifted_mean = 0.1136
real_minus_shuffled_mean = 0.1089
route_first_evidence_gold_frac = 0.6456
```

Interpretation:

```text
The full diagnostic strengthens the smoke conclusion.
Qwen has positive node/layer-band evidence-to-answer signals, but grouped route bundles remain fragile.
The nonmonotonic fraction rises to 0.7241, meaning many route groups become weaker rather than stronger when more nodes are added.
This supports a composition-sensitive non-closure explanation rather than a simple absence-of-route explanation.
```

Boundary:

```text
This is still an artifact-level composition diagnostic, not a new proof that Qwen has a Gemma-style graph route.
It explains why Qwen can have strict-supported individual route-first nodes while failing to close as a stable grouped feature route.
```

## Paper-Ready Wording

If support-only or layer-wise grouping improves:

```text
Qwen's grouped route non-closure is not simple route absence. Individual route-first nodes are causally meaningful, but naive mixed topK grouping can combine heterogeneous directions or layer bands. This supports the interpretation that Qwen's feature-level route is more distributed and composition-sensitive than Gemma's source-tracing graph.
```

If nothing improves:

```text
Qwen remains supported at the hidden and individual feature-node levels, but current grouped patch operators do not recover a stable sparse route object. This strengthens the boundary that Qwen's visible mechanism is not yet a Gemma-style grouped sparse route.
```
