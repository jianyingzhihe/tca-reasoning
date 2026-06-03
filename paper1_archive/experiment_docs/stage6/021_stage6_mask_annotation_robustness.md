# Stage6-021 Mask Annotation Robustness

Updated: 2026-06-02

## Question

The main route evidence depends on answer/union evidence masks. Reviewers may ask:

```text
Are the masks too subjective?
Are the regions tuned too precisely?
Would slightly different evidence regions preserve the result?
```

This experiment tests mask robustness without changing the main annotation assets.

## Mask Variants

For each selected sample:

```text
original_answer
original_union
dilated_answer
dilated_union
eroded_answer
eroded_union
shifted_mask
shuffled_mask
same_area_random
optional_second_annotator_answer
optional_second_annotator_union
```

If no second annotator masks exist, the first pass uses only morphology variants and existing controls.

## Sample Selection

Use 15-20 samples if full is affordable; smoke uses 3 samples.

Prioritize:

```text
visual_readout
symbol_text_reading
compact answer region
existing answer/union/shifted/shuffled masks
known source/control route artifacts
stable decoded answer
```

## Metrics

Primary metrics:

```text
source > controls
real > shifted/shuffled
answer/union weakening > same-area random
correct > wrong
decoded answer change rate
format failure rate
empty answer rate
```

The core comparison is not exact equality across masks. The question is whether morphology-perturbed evidence masks stay closer to original evidence masks than to shifted/shuffled/random controls.

## Decision Labels

```text
mask_robustness_supported
mask_robustness_morphology_partial
mask_robustness_boundary_sensitive
mask_robustness_blocked_missing_masks
```

## Interpretation Rules

Supported:

```text
original and dilated/eroded evidence masks preserve route weakening or restore direction,
and remain stronger than shifted/shuffled/random controls.
```

Partial:

```text
dilated masks preserve direction but eroded masks weaken it.
This means the route depends on sufficient evidence coverage, not on exact pixel-perfect annotation.
```

Boundary-sensitive:

```text
small morphology changes erase the effect.
This does not refute the main claim, but means the paper should emphasize compact evidence-region dependence and annotation sensitivity.
```

## Safety

Generated masks must be saved under a new Stage6 defensive output path. Do not overwrite the Stage3 paperpack mask directories.

## Initial Smoke Note

The first 3-sample smoke was useful for plumbing, but its exact
`original = dilate = erode` equality should not be interpreted scientifically.
We later found a remote asset collision in the route-first runner:
different mask variants were reusing the same remote `exported_masks/<image_stem>/`
paths and `put_if_missing` skipped overwriting them.

This was fixed by isolating remote asset roots by tag in
[`run_stage4_qwen_route_first_remote.py`](/E:/Bridging/scripts/local/run_stage4_qwen_route_first_remote.py).

## Corrected Partial Result

Generated assets:

```text
stage6_mask_robustness_assets_smoke_defensive_v1.csv
stage6_mask_robustness_assets_smoke_defensive_v1_decision.json
stage6_mask_robustness_original_smoke_defensive_v1_manifest.csv
stage6_mask_robustness_dilate_smoke_defensive_v1_manifest.csv
stage6_mask_robustness_erode_smoke_defensive_v1_manifest.csv
```

Corrected partial rerun assets:

```text
stage6_mask_robustness_assets_smoke_partial8_defensive_v1.csv
stage6_mask_robustness_assets_smoke_partial8_defensive_v1_decision.json
stage6_mask_robustness_smoke_partial8fix_defensive_v1_summary.csv
stage6_mask_robustness_smoke_partial8fix_defensive_v1_decision.json
```

Selected corrected partial samples:

```text
okvqa_val_01162
okvqa_val_03085
okvqa_val_1440035
okvqa_val_2078985
okvqa_val_2494045
okvqa_val_3713305
okvqa_val_4469905
okvqa_val_5186155
```

Remote evaluation used the Qwen route-first layer-14 smoke lens on `B_direct` and `D_visual_only`, producing 48 evaluated candidates per variant in the corrected partial rerun.

Corrected partial outputs:

```text
stage6_mask_robustness_smoke_partial8fix_defensive_v1_summary.csv
stage6_mask_robustness_smoke_partial8fix_defensive_v1_decision.json
status = mask_robustness_smoke_ready
```

Corrected partial summary:

```text
original:
n = 48
route_first_234_frac = 0.1250
route_first_evidence_gold_frac = 0.0833
restore_source_minus_controls_mean = 0.0148
real_minus_shifted_mean = 0.0098
real_minus_shuffled_mean = -0.0020
evidence_specificity_mean = 17.7469

dilate:
route_first_234_frac = 0.2500
route_first_evidence_gold_frac = 0.1667
restore_source_minus_controls_mean = 0.0150
real_minus_shifted_mean = 0.0938
real_minus_shuffled_mean = 0.0820
evidence_specificity_mean = 16.7127

erode:
route_first_234_frac = 0.1458
route_first_evidence_gold_frac = 0.1250
restore_source_minus_controls_mean = 0.0169
real_minus_shifted_mean = 0.0345
real_minus_shuffled_mean = 0.0228
evidence_specificity_mean = 14.7381
```

Corrected partial deltas:

```text
original_vs_dilate_route234_delta = +0.1250
original_vs_erode_route234_delta = +0.0208
original_vs_dilate_specificity_delta = -1.0341
original_vs_erode_specificity_delta = -3.0087
```

Current reading:

```text
After fixing the remote asset collision, morphology perturbations are not
exactly invariant. Dilation and erosion do change the aggregate route-first
summary, especially on route-first hit rate and evidence specificity.

However, the effect does not collapse under modest morphology changes:
restore_source_minus_controls_mean stays positive for all three variants,
and route_first_234_frac remains non-zero and even increases for dilate.

This supports a coverage-sensitive / morphology-partial reading more than
a pixel-perfect-boundary artifact reading.
```

Paper-ready wording at this stage:

```text
Corrected mask-robustness diagnostics do not support a simple pixel-perfect
artifact story, but they also do not justify claiming exact invariance.
In the current Qwen layer-14 partial rerun, dilation and erosion modulated
the aggregate route-first statistics without erasing them. This is more
consistent with evidence-coverage sensitivity than with a brittle one-pixel
annotation effect. A broader run is still needed before turning this into a
stronger paper-level claim.
```

## Expanded Qwen Mask16 Result

To strengthen the corrected partial result, we ran a broader Qwen route-first layer-14 morphology panel under `mask16_defensive_v1`.

Eligibility filtering produced 11 samples rather than the requested 16. This is acceptable because the stricter filter required compact masks, valid prompt runs, and layer-14 route-first candidate coverage.

Generated assets:

```text
stage6_mask_robustness_assets_smoke_mask16_defensive_v1.csv
stage6_mask_robustness_assets_smoke_mask16_defensive_v1_decision.json
stage6_mask_robustness_original_smoke_mask16_defensive_v1_manifest.csv
stage6_mask_robustness_dilate_smoke_mask16_defensive_v1_manifest.csv
stage6_mask_robustness_erode_smoke_mask16_defensive_v1_manifest.csv
stage6_mask_robustness_smoke_mask16_defensive_v1_summary.csv
stage6_mask_robustness_smoke_mask16_defensive_v1_decision.json
```

Expanded status:

```text
status = mask_robustness_smoke_ready
samples selected = 11
evaluated candidates per variant = 132
```

Expanded summary:

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
n = 132
route_first_234_frac = 0.1667
route_first_evidence_gold_frac = 0.1136
restore_source_minus_controls_mean = 0.0197
real_minus_shifted_mean = 0.0587
real_minus_shuffled_mean = 0.0464
evidence_specificity_mean = 12.6312

erode:
n = 132
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

Current expanded reading:

```text
The broader Qwen mask16 panel strengthens the corrected partial conclusion.
Morphology perturbations modulate the route-first metrics, but they do not
erase the evidence-to-answer signal. All three variants preserve positive
restore_source_minus_controls, positive real-minus-shifted/shuffled, and
non-zero route-first hit rates.
```

The most precise wording is still not "mask invariant". It is:

```text
Qwen route-first evidence effects are coverage-sensitive but not pixel-perfect
boundary artifacts under modest dilation/erosion in this layer-14 panel.
```

## Gemma Counterpart And Cross-Model Mask Lens

To satisfy the cross-model completeness rule, we added a Gemma morphology counterpart under the same `mask16_defensive_v1` defensive question.

Because Gemma source-tracing graph generation is disk-heavy, the counterpart uses the already validated Gemma hidden-lattice lens rather than rerunning large PLT source-tracing graphs:

```text
Qwen lens:
route-first layer-14 feature-node metrics
original / dilate / erode
n = 132 evaluated candidates per variant

Gemma lens:
hidden-lattice layer-1 visual+answer restore
original / dilate / erode
n = 6 usable prompt-runs per variant
```

Cross-model outputs:

```text
stage6_crossmodel_mask_robustness_smoke_mask16_defensive_v1_summary.csv
stage6_crossmodel_mask_robustness_smoke_mask16_defensive_v1_decision.json
status = crossmodel_mask_morphology_ready
```

Gemma summary:

```text
original:
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

Symmetric reading:

```text
The mask morphology defense is now symmetric at the scientific-question level.
Both models have original/dilate/erode perturbation results.

Qwen and Gemma should not be compared by raw effect magnitude here because the
readout objects differ. The shared defensive conclusion is that morphology
perturbations modulate the evidence-to-answer signal without erasing it.
```

Paper-ready wording:

```text
Across Qwen route-first and Gemma hidden-lattice defensive readouts, modest
dilation/erosion of the annotated evidence region changes the aggregate metrics
but does not collapse the evidence-to-answer signal into control-like behavior.
This supports a coverage-sensitive robustness interpretation rather than a
pixel-perfect mask-boundary artifact interpretation.
```
