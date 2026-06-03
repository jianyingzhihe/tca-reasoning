# Stage6-023 Defensive Verdict

Updated: 2026-06-02

## Current Status

This verdict now includes completed Stage6-019, Stage6-020, Stage6-021, Stage6-022, and Stage6-025 defensive results.
It also includes the Stage6-024 cross-model symmetry audit, which controls which findings may be written as cross-model claims.

Known completed inputs:

```text
Stage6-016 Gemma hidden-to-PLT decomposition:
  gemma_hidden_flow_error_heavy_like_qwen

Stage6-017 cross-model hidden-to-PLT verdict:
  Gemma and Qwen are both error-heavy under tested local topK reconstruction.

Stage6-013 unified prompt/text verdict:
  prompt/text changes modulate route identity and strength, but do not erase evidence-to-answer mechanisms.
```

Local defensive preflight:

```text
tag = defensive_v1
status = ready_for_defensive_smoke
missing_required = []
Gemma hidden-to-PLT status = gemma_hidden_flow_error_heavy_like_qwen
Qwen grouped route status = qwen_hidden_route_feature_route_not_supported
```

## Defensive Claims To Resolve

### Claim A: Gemma error-heavy does not contradict source-tracing

Target wording:

```text
Gemma's sparse route claim is a graph-level route-object claim. It does not require local PLT topK reconstruction to mediate the entire hidden residual effect.
```

### Claim B: Qwen grouped non-closure is a composition problem, not route absence

Target wording:

```text
Qwen has hidden-level flow and individual route-first feature nodes. Current grouped route non-closure may reflect heterogeneous directions, layer-band composition, non-linear interaction, or PLT basis coverage limits.
```

### Claim C: Evidence masks are not pixel-perfect artifacts

Target wording:

```text
Evidence-region effects should not collapse into control-like behavior under modest morphology perturbations. The corrected current result supports coverage-sensitive robustness, not exact morphology invariance.
```

### Claim D: Decoded bridge is illustrative, not mandatory

Target wording:

```text
Case-level decoded bridges can make the mechanism legible, but target rank/margin evidence remains the primary quantitative bridge.
```

## Final Paper Integration

These experiments should appear as secondary/defensive results, not as main results:

```text
Result 1: evidence-sensitive answer routes exist.
Result 2: cross-model mechanisms exist but are visible through different route objects.
Result 3: prompt/text changes modulate route usage rather than erase it.
Result 4: defensive diagnostics constrain simple topK, grouping, and mask-artifact objections without overclaiming exact invariance.
```

## Status Table

```text
Stage6-019 Gemma case panel: completed, case_panel_ready
Stage6-020 Qwen grouped composition: gpu1 full artifact diagnostic completed
Stage6-021 mask robustness: gpu1 corrected partial and expanded Qwen mask16 completed
Stage6-022 decoded bridge: gpu1 symmetric mini-full completed
Stage6-024 cross-model symmetry audit: completed
Stage6-025 Gemma grouped composition diagnostic: completed
```

Preflight artifacts:

```text
stage6_defensive_defensive_v1_artifact_status.csv
stage6_defensive_defensive_v1_decision.json
stage6_defensive_gemma_case_panel_full_defensive_v1.csv
stage6_defensive_gemma_case_panel_full_defensive_v1_decision.json
```

## Stage6-019 Result

Gemma case panel is complete:

```text
status = case_panel_ready
raw_candidate_rows = 70
selected_rows = 6
```

The strongest example is `okvqa_val_2100995 / B_direct / wicker`:

```text
hidden_effect = 44.375
PLT topK reconstruction K=32 = -0.34375
PLT reconstruction error K=32 = 45.625
error_over_hidden = 1.0282
topk_over_hidden = -0.0077
```

Current defensive wording:

```text
Gemma's hidden-level flow is error-heavy under local hidden-to-PLT decomposition, but this does not contradict the sparse source-tracing route claim. The source-tracing result is a graph-level route-object result, not a claim that local sparse topK reconstruction explains the whole hidden residual effect.
```

## Stage6-020 Result

Qwen grouped composition full artifact diagnostic is complete:

```text
mode = full
route_metric_rows = 435
candidate_rows = 24970
topk_summary_rows = 5
layer_band_rows = 6
nonmonotonic_routes = 63
nonmonotonic_frac = 0.7241
```

Current reading:

```text
strict grouped topK bundles remain fragile;
strict layer-band summaries stay positive across L10-L12, L13-L15, and L16-L17;
adding nodes often makes grouped restore weaker instead of stronger.
```

This strengthens the composition interpretation:

```text
Qwen grouped-route non-closure is not simple route absence.
The visible issue is that bundle-level behavior is composition-sensitive and can be nonmonotonic even when layer-band and individual route-first signals are positive.
```

Key full numbers:

```text
strict topK8:
route_restore_source_minus_controls_mean = 0.0125
route_real_minus_shifted_mean = 0.0430
route_real_minus_shuffled_mean = 0.0268
route_correct_minus_wrong_mean = 0.0550
all_gates_frac = 0.0920

strict L13-L15:
restore_source_minus_controls_mean = 0.0626
real_minus_shifted_mean = 0.1192
real_minus_shuffled_mean = 0.1053
route_first_evidence_gold_frac = 0.7198
```

## Stage6-021 Result

Mask robustness corrected partial, expanded Qwen mask16, and Gemma morphology counterpart are complete:

```text
corrected partial status = mask_robustness_smoke_ready
corrected partial samples = 8
corrected partial evaluated candidates per variant = 48

expanded tag = mask16_defensive_v1
expanded status = mask_robustness_smoke_ready
expanded selected samples = 11
expanded evaluated candidates per variant = 132

cross-model tag = mask16_defensive_v1
cross-model status = crossmodel_mask_morphology_ready
Gemma usable prompt-runs per variant = 6
```

Important engineering note:

```text
The first exact original=dilate=erode smoke equality was invalidated by a
remote mask-path collision in the runner.
The runner was fixed by isolating remote asset roots per tag, and the current
reading below uses the corrected partial rerun.
```

Current corrected outputs:

```text
stage6_mask_robustness_assets_smoke_partial8_defensive_v1.csv
stage6_mask_robustness_assets_smoke_partial8_defensive_v1_decision.json
stage6_mask_robustness_smoke_partial8fix_defensive_v1_summary.csv
stage6_mask_robustness_smoke_partial8fix_defensive_v1_decision.json
```

Expanded outputs:

```text
stage6_mask_robustness_assets_smoke_mask16_defensive_v1.csv
stage6_mask_robustness_assets_smoke_mask16_defensive_v1_decision.json
stage6_mask_robustness_smoke_mask16_defensive_v1_summary.csv
stage6_mask_robustness_smoke_mask16_defensive_v1_decision.json
stage6_crossmodel_mask_robustness_smoke_mask16_defensive_v1_summary.csv
stage6_crossmodel_mask_robustness_smoke_mask16_defensive_v1_decision.json
```

Current corrected summary:

```text
original:
route_first_234_frac = 0.1250
restore_source_minus_controls_mean = 0.0148
real_minus_shifted_mean = 0.0098
real_minus_shuffled_mean = -0.0020
evidence_specificity_mean = 17.7469

dilate:
route_first_234_frac = 0.2500
restore_source_minus_controls_mean = 0.0150
real_minus_shifted_mean = 0.0938
real_minus_shuffled_mean = 0.0820
evidence_specificity_mean = 16.7127

erode:
route_first_234_frac = 0.1458
restore_source_minus_controls_mean = 0.0169
real_minus_shifted_mean = 0.0345
real_minus_shuffled_mean = 0.0228
evidence_specificity_mean = 14.7381
```

Decision deltas:

```text
original_vs_dilate_route234_delta = +0.1250
original_vs_erode_route234_delta = +0.0208
original_vs_dilate_specificity_delta = -1.0341
original_vs_erode_specificity_delta = -3.0087
```

Expanded Qwen mask16 summary:

```text
original:
n = 132
route_first_234_frac = 0.1439
route_first_evidence_gold_frac = 0.0909
restore_source_minus_controls_mean = 0.0268
real_minus_shifted_mean = 0.0464
real_minus_shuffled_mean = 0.0341
evidence_specificity_mean = 15.1441

dilate:
route_first_234_frac = 0.1667
route_first_evidence_gold_frac = 0.1136
restore_source_minus_controls_mean = 0.0197
real_minus_shifted_mean = 0.0587
real_minus_shuffled_mean = 0.0464
evidence_specificity_mean = 12.6312

erode:
route_first_234_frac = 0.1061
route_first_evidence_gold_frac = 0.0758
restore_source_minus_controls_mean = 0.0215
real_minus_shifted_mean = 0.0308
real_minus_shuffled_mean = 0.0185
evidence_specificity_mean = 12.1282
```

Expanded deltas:

```text
original_vs_dilate_route234_delta = +0.0227
original_vs_erode_route234_delta = -0.0379
original_vs_dilate_specificity_delta = -2.5129
original_vs_erode_specificity_delta = -3.0159
```

Current defensive wording:

```text
The corrected and expanded Qwen layer-14 morphology runs do not support a
simple pixel-perfect-boundary artifact story, but they also do not justify
claiming exact morphology invariance. Dilation and erosion modulate the
aggregate route-first statistics without erasing them. This is better described
as coverage-sensitive robustness than as boundary-insensitive robustness.
```

Gemma morphology counterpart:

```text
Gemma was evaluated with a hidden-lattice layer-1 visual+answer restore lens on
the same original/dilate/erode morphology question. This avoids generating new
large source-tracing graphs while testing the same defensive objection.

original:
n = 6
target_effect_union_mean = 3.2500
real_minus_shifted_mean = 3.5417
real_minus_shuffled_mean = 1.0833
correct_minus_wrong_mean = 4.6667

dilate:
target_effect_union_mean = 6.3750
real_minus_shifted_mean = 6.6667
real_minus_shuffled_mean = 4.2083
correct_minus_wrong_mean = 2.2292

erode:
target_effect_union_mean = 7.9271
real_minus_shifted_mean = 8.2188
real_minus_shuffled_mean = 5.7604
correct_minus_wrong_mean = 4.1771
```

Cross-model boundary:

```text
Stage6-021 is now symmetric as a defensive mask-artifact lens:
Qwen uses route-first layer-14 feature-node metrics;
Gemma uses hidden-lattice layer-1 visual+answer restore metrics.

The shared conclusion is not raw-magnitude equality. It is that modest
dilation/erosion modulates the evidence-to-answer readout but does not erase it.
This supports coverage-sensitive robustness rather than pixel-perfect
mask-boundary dependence.
```

## Stage6-022 Result

Node-to-generation bridge mini-full is complete under the symmetric v2 tag:

```text
status = decoded_bridge_full_ready
tag = defensive_v2_symmetric
Gemma rows = 90
Qwen hidden rows = 16
Qwen PLT rows = 156
Qwen CLT rows = 156
```

Important correction:

```text
The earlier defensive_v1 result was useful but asymmetric:
Gemma v1 used hidden-residual generation bridge only.
Qwen v1 used PLT/CLT multifeature generation bridge only.

The final Stage6-022 wording should use defensive_v2_symmetric.
```

Symmetric v2 lenses:

```text
Gemma hidden_residual
Gemma plt_topk_reconstruction
Gemma plt_reconstruction_error

Qwen hidden_residual
Qwen PLT multifeature
Qwen CLT multifeature
```

Gemma v2:

```text
hidden_residual restore:
case_count = 3
mean_oriented_sequence_gap = +18.0716
mean_oriented_first_token_gap = +27.6875
mean_oriented_rank_gap = +1601.3333
mean_oriented_margin_gap = +11.2708

plt_reconstruction_error restore:
case_count = 9
mean_oriented_sequence_gap = +19.1819
mean_oriented_first_token_gap = +26.8681
mean_oriented_rank_gap = +1362.8889
mean_oriented_margin_gap = +9.7014
decoded restore to clean = 2 / 9

plt_topk_reconstruction restore:
case_count = 9
mean_oriented_sequence_gap = -0.5111
mean_oriented_first_token_gap = -0.4462
mean_oriented_rank_gap = -585.3333
decoded restore to clean = 0
```

Qwen v2:

```text
hidden_residual restore:
case_count = 4
mean_oriented_sequence_gap = +8.6032
mean_oriented_first_token_gap = +7.0859
mean_oriented_rank_gap = +2568.75
mean_oriented_margin_gap = +2.25
decoded changed vs reference = 3 / 4

PLT multifeature restore:
case_count = 15
mean_oriented_sequence_gap = -0.0318
mean_oriented_first_token_gap = +0.0938
mean_oriented_rank_gap = +57.6667

CLT multifeature restore:
case_count = 15
mean_oriented_sequence_gap = +0.1269
mean_oriented_first_token_gap = +0.2896
mean_oriented_rank_gap = +500.85
```

Current defensive wording:

```text
Stage6-022 now provides symmetric illustrative case-level bridge evidence.
Hidden-residual interventions in both Gemma and Qwen move generation-side
rank, margin, and sequence scores. Gemma's generation bridge mirrors the
hidden-to-PLT decomposition: PLT reconstruction error carries the strong
generation-side effect, while sparse PLT topK reconstruction is weak or mixed.
For Qwen, hidden-residual interventions are stronger than the tested PLT/CLT
multifeature bridges, consistent with the interpretation that Qwen's evidence
flow is most visible in hidden/residual space and less cleanly closed as a
sparse feature bundle. Greedy decoded answer control remains case-level and
partial, not a required main-claim proof.
```

Core conclusion:

```text
Stage6-022 should be cited as a generation-side score bridge, not as stable
decoded-answer control. The symmetric v2 comparison shows that both Gemma and
Qwen hidden-residual interventions move answer-side sequence score, rank,
margin, and first-token score. The sparse feature-store bridge is weaker:
Gemma's PLT reconstruction error carries the strong generation-side effect,
while Gemma sparse PLT topK reconstruction is weak or mixed; Qwen hidden
interventions are stronger than the tested Qwen PLT/CLT multifeature bridges.
```

## Stage6-025 Result

Gemma grouped composition diagnostic is complete:

```text
status = gemma_grouped_composition_artifact_ready
route_rows = 1704
topk_summary_rows = 24
route_stability_rows = 284
compare_summary_rows = 2
mixed_composition_frac_at_largest_topk = 0.7993
top_edge_dominated_frac_at_largest_topk = 0.0
```

Key Gemma topK32 route-bundle numbers:

```text
primary A path_mass_retention_mean = 0.9759
primary B path_mass_retention_mean = 0.9719
strict A path_mass_retention_mean = 0.9765
strict B path_mass_retention_mean = 0.9719

primary A signed_path_mass_balance_mean = 0.2798
primary B signed_path_mass_balance_mean = 0.3012
strict A signed_path_mass_balance_mean = 0.2819
strict B signed_path_mass_balance_mean = 0.3147
```

Current defensive wording:

```text
The grouped-composition comparison is now symmetric. Qwen's grouped feature
route remains fragile/nonmonotonic under grouped restore. Gemma's route bundle
is also compositionally mixed, but it is graph-closed under source tracing:
topK32 captures almost all traced path mass and the bundle is not single-edge
dominated. The cross-model difference is therefore not "Gemma simple, Qwen
complex"; it is "Gemma mixed but graph-closed, Qwen mixed/fragile and not yet
grouped-route closed."
```

## Stage6-024 Symmetry Audit

The current cross-model completeness rule is:

```text
Every Qwen result used in a cross-model claim needs a Gemma counterpart.
Every Gemma result used in a cross-model claim needs a Qwen counterpart.
```

Completed symmetric lenses:

```text
hidden residual route
hidden-to-PLT decomposition
prompt/text fixed-node causal strength
prompt/text route identity stability
node-to-generation score bridge
mask morphology defensive lens
grouped composition diagnostic lens
```

Model-specific but tested:

```text
Gemma sparse source-tracing route is positive.
Qwen grouped sparse feature-route closure was tested and remains unresolved.
```

Remaining mandatory symmetry gaps:

```text
none for the current Stage6 claims.
```

Current reporting rule:

```text
Do not write single-model defensive results as cross-model claims.
Use complete symmetric lenses for cross-model conclusions.
Mask morphology is now complete at the defensive-lens level; label the readout
objects explicitly and do not compare raw magnitudes across models.
```
