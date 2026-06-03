# Stage6-025 Gemma Grouped Composition Diagnostic

Updated: 2026-06-02

## Question

Stage6-020 diagnosed Qwen grouped-route non-closure as a composition-sensitive problem. To enforce cross-model completeness, this diagnostic asks the matching Gemma question:

```text
If Qwen gets an explicit grouped-composition diagnostic,
does Gemma also have a route-bundle composition diagnostic?
```

This is not a new proof that Gemma source-tracing routes exist. Stage3 already establishes the positive Gemma route. This diagnostic only tests how the existing Gemma route bundle behaves when viewed by topK route mass, sign mixture, and leave-one-edge concentration.

## Inputs

No new GPU run was needed. The analyzer reads existing Stage3 Gemma source-tracing artifacts:

```text
stage3_gemma_source_tracing_primary_full_edges_detailed_controlled.csv
stage3_gemma_source_tracing_primary_full_sample_compare_controlled.csv
stage3_gemma_source_tracing_strict_full_edges_detailed_controlled.csv
stage3_gemma_source_tracing_strict_full_sample_compare_controlled.csv
```

Analyzer:

```text
scripts/local/analyze_stage6_gemma_grouped_composition.py
```

Outputs:

```text
stage6_defensive_gemma_grouped_composition_full_defensive_v1_route_rows.csv
stage6_defensive_gemma_grouped_composition_full_defensive_v1_topk_summary.csv
stage6_defensive_gemma_grouped_composition_full_defensive_v1_route_stability.csv
stage6_defensive_gemma_grouped_composition_full_defensive_v1_compare_summary.csv
stage6_defensive_gemma_grouped_composition_full_defensive_v1_decision.json
```

## Metrics

For each Gemma route graph, edges are sorted by `path_mass`, then summarized at:

```text
topK = 1, 2, 4, 8, 16, 32
```

The diagnostic reports:

```text
path_mass_retention:
  how much of the route graph's traced path mass is captured by the topK bundle.

positive_path_mass_frac:
  how much topK path mass comes from positive-weight edges.

signed_path_mass_balance:
  how coherent the signed route mass is after positive/negative cancellation.
  1.0 means single-sign/coherent; lower values mean mixed-sign composition.

leave_one_top_edge_mass_frac:
  how much of the topK bundle is dominated by the single strongest edge.

node/edge overlap:
  A-condition vs B-baseline route identity overlap.
```

## Result

Decision:

```text
status = gemma_grouped_composition_artifact_ready
route_rows = 1704
topk_summary_rows = 24
route_stability_rows = 284
compare_summary_rows = 2
mixed_composition_frac_at_largest_topk = 0.7993
top_edge_dominated_frac_at_largest_topk = 0.0
```

TopK path-mass retention is high by topK32:

```text
primary A topK32 path_mass_retention_mean = 0.9759
primary B topK32 path_mass_retention_mean = 0.9719
strict A topK32 path_mass_retention_mean = 0.9765
strict B topK32 path_mass_retention_mean = 0.9719
```

The route is not dominated by one edge:

```text
primary A topK32 leave_one_top_edge_mass_frac_mean = 0.1380
primary B topK32 leave_one_top_edge_mass_frac_mean = 0.1343
strict A topK32 leave_one_top_edge_mass_frac_mean = 0.1381
strict B topK32 leave_one_top_edge_mass_frac_mean = 0.1345
top_edge_dominated_frac_at_largest_topk = 0.0
```

The bundle is compositionally mixed:

```text
primary A topK32 positive_path_mass_frac_mean = 0.5770
primary B topK32 positive_path_mass_frac_mean = 0.5904
strict A topK32 positive_path_mass_frac_mean = 0.5845
strict B topK32 positive_path_mass_frac_mean = 0.6046

primary A topK32 signed_path_mass_balance_mean = 0.2798
primary B topK32 signed_path_mass_balance_mean = 0.3012
strict A topK32 signed_path_mass_balance_mean = 0.2819
strict B topK32 signed_path_mass_balance_mean = 0.3147
```

Route identity overlap is moderate and stable across primary/strict:

```text
primary node_overlap_jaccard_mean = 0.3092
primary edge_overlap_jaccard_mean = 0.2094
strict node_overlap_jaccard_mean = 0.2999
strict edge_overlap_jaccard_mean = 0.1996
```

## Interpretation

The strict symmetric comparison with Qwen is:

```text
Qwen:
  grouped feature-route bundles are fragile/nonmonotonic under grouped restore.
  This explains why single feature nodes can pass but grouped route closure remains unresolved.

Gemma:
  the existing source-tracing graph route is compositionally mixed, but it is already a closed graph-level route object.
  TopK32 captures almost all traced path mass, and the route is not single-edge dominated.
```

So the updated conclusion is not:

```text
Gemma routes are compositionally simple and Qwen routes are compositionally complex.
```

The better conclusion is:

```text
Both models can have compositionally mixed route bundles.
The difference is that Gemma's mixed bundle is still organized by source-tracing into a usable sparse graph route,
whereas Qwen's grouped feature bundle remains non-closed under the current grouped intervention operator.
```

## Paper-Ready Wording

```text
To keep the grouped-route claim symmetric, we ran a Gemma route-bundle composition diagnostic on the existing source-tracing graphs. Gemma's route bundle is not trivially single-sign or single-edge dominated: topK32 contains mixed signed path mass. However, unlike Qwen's grouped feature route, the Gemma source-tracing bundle remains a closed graph-level route object with high path-mass retention and stable primary/strict graph statistics. This supports the interpretation that the cross-model difference is not simply "Gemma is simple, Qwen is complex"; rather, Gemma's route object is compositionally mixed but graph-closed, while Qwen's feature-level bundle is compositionally fragile and not yet graph-closed.
```

## Boundary

This is an artifact-level route-composition diagnostic. It mirrors the Qwen grouped-composition question, but it does not rerun new Gemma route interventions. That is acceptable for the current defensive purpose because Gemma's positive source-tracing intervention evidence already exists in Stage3.
