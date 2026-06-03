# Stage6-018 Defensive Experiment Overview

Updated: 2026-06-02

## Purpose

Stage6 defensive experiments are not meant to expand the main claim. They are designed to answer likely reviewer objections around the already established mechanism story:

```text
Do the new hidden-to-PLT decomposition results contradict Gemma source-tracing?
Why does Qwen have hidden and feature-node support but no closed grouped route?
Are evidence masks too subjective?
Can node/route interventions connect to decoded generation behavior?
```

The goal is to make the paper more robust, not to create another required positive result.

## Current Baseline

The current main story is:

```text
Gemma3-PLT:
  complete sparse graph/source-tracing route object.

Qwen2.5-VL-PLT:
  hidden-level evidence-to-answer route supported;
  strict-supported individual feature nodes;
  grouped sparse feature-route closure unresolved.

Gemma and Qwen hidden-to-PLT decomposition:
  both are error-heavy under the tested local topK reconstruction lens.
```

The most important boundary is that `PLT topK reconstruction` and `graph-level source-tracing route` are not the same object. A local topK reconstruction can fail to mediate the full hidden effect while a graph-level route remains causally useful.

## Defensive Experiments

### Stage6-019: Gemma error-heavy vs source-route case panel

This is the highest-priority defensive writeup. It explains why Gemma can be `error-heavy` under hidden-to-PLT decomposition while still supporting a sparse source-tracing route.

Expected output:

```text
2-3 Gemma cases showing:
hidden_residual > 0
plt_topK_reconstruction near 0
plt_reconstruction_error approximately hidden_residual
source-tracing route gates pass
```

This is mostly an artifact-joining and explanation experiment. It should not rerun large graph generation.

### Stage6-020: Qwen grouped route composition diagnostic

This tests whether Qwen grouped feature-route non-closure is due to mixed node directions, layer mixing, or topK aggregation.

Conditions:

```text
support-only grouping
layer-wise grouping: L10-L12, L13-L15, L16-L17, L14-near
topK curve: 1, 2, 4, 8, 16
leave-one-out on strong cases
```

The goal is diagnostic. A mixed result is valuable if it explains why individual nodes can pass while bundles do not.

### Stage6-021: Mask annotation robustness

This defends against the objection that answer/union masks are subjective or tuned too precisely.

Conditions:

```text
original answer/union mask
dilated answer/union mask
eroded answer/union mask
shifted mask
shuffled mask
same-area random mask
optional second annotator mask
```

Main criterion: original and morphology-perturbed masks should not collapse into control-like behavior. Exact equality or boundary-insensitive invariance is not required.

Current smoke status:

```text
Initial small Qwen route-first layer-14 smoke was completed,
but the exact original=dilate=erode equality was later traced to a remote
mask-path collision in the runner.

Corrected partial rerun after asset-root isolation:
8 selected samples / 48 evaluated candidates per variant.
Morphology perturbations did not erase the signal, but they were not exactly invariant.

Expanded Qwen mask16 panel:
11 selected samples / 132 evaluated candidates per variant.

Gemma counterpart:
hidden-lattice layer-1 visual+answer restore on original/dilate/erode masks,
6 usable prompt-runs per variant.
```

This is not an exact morphology-invariance claim. The corrected cross-model readout is more informative than the initial artifact: modest dilation/erosion changed aggregate metrics, but did not collapse them, which is more consistent with coverage sensitivity than with a pixel-perfect mask artifact.

### Stage6-022: Node-to-generation bridge

This is optional and case-level. It checks whether strong node/route interventions can connect to decoded answer changes.

Gemma conditions:

```text
clean generation
source route zeroing generation
control route zeroing generation
masked image generation
source route restore generation
```

Qwen conditions:

```text
hidden restore generation
strong feature node restore/zeroing generation
control intervention generation
```

If decoded generation is too expensive or brittle, keep target rank / margin as the accepted bridge metric.

## Execution Order

1. Write Stage6-018 to Stage6-023 documents and local analyzer skeleton.
2. Run local artifact status analyzer.
3. Run gpu1 Qwen grouped composition smoke.
4. Run gpu1 mask robustness smoke.
5. Qwen grouped composition smoke completed and is diagnostic-positive.
6. Qwen grouped composition full artifact diagnostic completed and strengthens the composition-sensitive non-closure interpretation.
7. Mask robustness corrected partial completed; morphology modulates but does not erase the route-first signal.
8. Broader Qwen mask robustness `mask16_defensive_v1` completed on gpu1; only 11 samples passed the stricter eligibility filters.
9. Decoded bridge mini-full completed on a few strong cases; keep it illustrative and do not escalate to a large generation sweep unless specifically needed.
10. Cross-model symmetry audit added in `024_stage6_cross_model_symmetry_audit.md`.
11. Gemma mask morphology counterpart completed with a hidden-lattice layer-1 visual+answer restore lens, closing the mask-artifact symmetry gap at the defensive-lens level.
12. Gemma grouped-composition diagnostic completed in `025_stage6_gemma_grouped_composition_diagnostic.md`, closing the Qwen grouped-composition mirror at the artifact-diagnostic level.

## Symmetry Audit

Stage6 now uses an explicit cross-model completeness rule:

```text
Every Qwen result used in a cross-model claim needs a Gemma counterpart.
Every Gemma result used in a cross-model claim needs a Qwen counterpart.
```

The major mechanism lenses are already symmetric:

```text
hidden residual route
hidden-to-PLT decomposition
prompt/text fixed-node strength
prompt/text route identity
node-to-generation bridge
```

There are no remaining mandatory symmetry gaps for the current Stage6 claims:

```text
completed:
  hidden residual route
  hidden-to-PLT decomposition
  prompt/text fixed-node strength
  prompt/text route identity
  node-to-generation bridge
  mask morphology robustness
  grouped composition diagnostic
```

Mask morphology is no longer a pending symmetry counterpart. It is now complete at the defensive-lens level:

```text
Qwen = route-first layer-14 feature-node morphology panel.
Gemma = hidden-lattice layer-1 visual+answer morphology panel.
Conclusion = morphology modulates but does not erase evidence-to-answer signal.
```

The grouped-composition lens is also complete at the artifact-diagnostic level:

```text
Qwen = grouped topK / layer-band / nonmonotonic diagnostic.
Gemma = source-tracing route-bundle topK path-mass / signed-mix / leave-one-edge diagnostic.
Conclusion = Gemma is mixed but graph-closed; Qwen is mixed/fragile and not yet grouped-route closed.
```

## Safety

Do not delete model cache, OKVQA data, Stage3 Gemma strict assets, or existing Stage4/Qwen artifacts.

Do not generate persistent large `.pt` graph directories for Stage6 defensive work. If a rerun requires graph-like files, use sharded cleanup or keep it case-level.

Stage6 defensive failures must be reported as diagnostic boundaries, not as main-claim failures.
