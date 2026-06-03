# Stage6-024 Cross-Model Symmetry Audit

Updated: 2026-06-02

## Purpose

This audit enforces the rule that every Qwen result used in a cross-model claim must have a Gemma counterpart, and every Gemma result used in a cross-model claim must have a Qwen counterpart.

The symmetry standard is lens-level, not script-name-level:

```text
Same scientific question + comparable intervention/evaluation lens = symmetric.
Identical internal object names are not required, because Gemma and Qwen expose different route objects.
```

For example, Gemma's graph-level source-tracing route and Qwen's route-first feature set are not the same object. They can still answer the same route-identity question if the report labels the object type explicitly.

## Symmetry Matrix

| Lens / question | Gemma status | Qwen status | Symmetry verdict |
|---|---|---|---|
| Sparse source-tracing / grouped route closure | Complete positive: Gemma source-tracing graph route is the sparse baseline. | Attempted/diagnosed: Qwen source-tracing and grouped feature-route closure remain unresolved. | Symmetric as a tested question, asymmetric as a result. This is the main cross-model difference. |
| Hidden residual evidence-to-answer route | Complete: Stage6-014 Gemma hidden lattice primary/strict full. | Complete: Stage4 Qwen all-layer hidden route primary/strict. | Complete. |
| Hidden-to-PLT decomposition | Complete: Stage6-016 Gemma hidden residual / PLT topK / reconstruction error. | Complete: Stage4-038 Qwen hidden-to-PLT mediation. | Complete; both are error-heavy under local topK reconstruction. |
| Prompt/text/CoT fixed-node causal strength | Complete: Stage6-011/013 Gemma fixed-node full under `unified_v1_focus4`. | Complete: Stage6 Qwen prompt/text fixed-node reanalysis. | Complete. |
| Prompt/text/CoT route identity stability | Complete: Gemma graph-overlap / source-tracing route identity reanalysis. | Complete: Stage6-012 Qwen route rediscovery / topK overlap. | Complete, with object type labeled separately. |
| Node-to-generation bridge | Complete: Stage6-022 symmetric v2, Gemma hidden / PLT topK / PLT error. | Complete: Stage6-022 symmetric v2, Qwen hidden / PLT / CLT. | Complete; use rank/margin/sequence score as main bridge, decoded answer as illustrative. |
| Grouped composition diagnostic | Complete counterpart: Stage6-025 Gemma route-bundle topK path-mass / signed-mix / leave-one-edge diagnostic on existing source-tracing graphs. | Complete counterpart: Stage6-020 Qwen grouped topK / layer-band / nonmonotonic diagnostic under `defensive_v1`. | Complete as a composition-diagnostic lens. Gemma is graph-closed but compositionally mixed; Qwen is compositionally fragile and grouped-route closure remains unresolved. |
| Mask annotation morphology robustness | Complete counterpart: Gemma hidden-lattice layer-1 visual+answer restore on original/dilate/erode under `gemmamask_*_mask16_defensive_v1`. | Complete counterpart: Qwen route-first layer-14 original/dilate/erode under `mask16_defensive_v1`. | Complete as a defensive mask-artifact lens. Do not compare raw magnitudes because the readout objects differ. |

## Gap Status

No mandatory cross-model symmetry counterpart is currently missing for the Stage6 claims we intend to write. The sections below record gaps that were closed during the defensive pass.

## Recently Closed Gap: Grouped Composition

The grouped-composition gap is now closed at the artifact-diagnostic level:

```text
Qwen:
  Stage6-020
  grouped topK / layer-band / nonmonotonic diagnostic
  route_metric_rows = 435
  nonmonotonic_frac = 0.7241

Gemma:
  Stage6-025
  source-tracing route-bundle topK path-mass / signed-mix / leave-one-edge diagnostic
  route_rows = 1704
  topK32 path_mass_retention ~= 0.972 to 0.977
  mixed_composition_frac_at_largest_topk = 0.7993
  top_edge_dominated_frac_at_largest_topk = 0.0
```

The symmetric conclusion is:

```text
Both models can have compositionally mixed route bundles.
Gemma's route bundle is mixed but graph-closed under source tracing.
Qwen's feature bundle is mixed/fragile and grouped-route closure remains unresolved.
```

## Recently Closed Gap: Mask Morphology

The previous clearest symmetry gap was mask morphology. This is now closed at the defensive-lens level:

```text
Qwen:
  tag = mask16_defensive_v1
  lens = route-first layer-14 feature-node metrics
  variants = original / dilate / erode
  evaluated candidates per variant = 132

Gemma:
  tags = gemmamask_original/dilate/erode_mask16_defensive_v1
  lens = hidden-lattice layer-1 visual+answer restore
  variants = original / dilate / erode
  usable prompt-runs per variant = 6
```

Cross-model interpretation:

```text
Both models now have a morphology perturbation counterpart.
The comparable question is whether modest mask dilation/erosion erases the evidence-to-answer signal.
The comparable answer is no: morphology modulates the readout, but does not collapse it into control-like behavior.
Do not compare Qwen route-first raw values against Gemma hidden-lattice raw values.
```

## Reporting Rule

Use three labels in every synthesis table:

```text
Complete symmetric lens:
  both models tested under comparable scientific question.

Model-specific diagnostic:
  only one model needs this because it explains that model's failure/visibility mode.

Pending symmetry counterpart:
  do not write as a cross-model claim yet.
  At this point, no mandatory Stage6 claim is assigned this label.
```

Current safest paper wording:

```text
Gemma and Qwen are fully aligned for the major mechanism lenses: hidden residual route,
hidden-to-PLT decomposition, prompt/text perturbation, route identity, and generation-side bridge.
The mask morphology defense is aligned at the defensive-lens level: Qwen route-first
and Gemma hidden-lattice morphology panels both show coverage-sensitive modulation rather than
collapse under modest dilation/erosion. The grouped-composition defense is also aligned at
the artifact-diagnostic level: Gemma is graph-closed but mixed, while Qwen is mixed/fragile
and not yet grouped-route closed.
```
