# Stage4-011 Qwen Source-Tracing Critical Audit

## Purpose

This audit revisits the Stage4 Qwen2.5-VL-PLT source-tracing adapter with a deliberately skeptical lens. The main question is whether the current experiment really tests "all source-tracing nodes" or only a restricted subset, and whether the current negative verdict is safe to interpret.

## Short Answer

The current Stage4 Qwen adapter does **not** intervene on all weighted nodes in the model or all PLT layers. It tests a narrow, pre-registered but approximate route:

- model: `Qwen2.5-VL-7B-Instruct`
- transcoder: `KokosDev/qwen2p5vl-7b-plt`
- layer: `26`
- positions: visual span + answer-adjacent positions
- graph nodes: selected support features only
- compare export: top incoming graph nodes per target
- intervention: top 2 feature nodes per sample/prompt, with `subtract` and `add`

Therefore the current result can only say that this **specific Qwen answer-aligned adapter route** did not reproduce Gemma-style source tracing. It cannot say Qwen has no evidence-sensitive mechanism.

## Critical Findings

### 1. Layer-output alignment bug risk

Original implementation selected Qwen features from `outputs.hidden_states[args.layer]` but intervened on `model.language_model.layers[args.layer]`.

In most HuggingFace decoder models, `hidden_states[0]` is the embedding state and `hidden_states[layer + 1]` is the output of that layer. That means the attribution-side feature selection could have been off by one layer relative to the intervention hook.

Impact:

- This can make selected feature nodes fail intervention even if a real layer-26 route exists.
- The existing `qwen_source_tracing_not_supported` verdict should be treated as provisional until rerun after the fix.

Fix:

- `run_qwen_answer_aligned_attribute.py` now captures `language_model.layers[args.layer]` output via a forward hook and uses that tensor for PLT encoding.
- Metadata records `hidden_capture_source` and `hidden_states_tuple_len`.

### 2. The graph is schema-compatible, not a full Gemma ReplacementModel graph

The Qwen adapter builds a compact graph with:

- feature node to target-logit direct-effect edge
- token-position to feature activation bridge
- a single target-logit node

It does not reconstruct Gemma's full replacement graph with all intermediate feature/error/token routes.

Impact:

- `trace_compare_ab_controlled.py` can read the graph, but the graph is not mechanistically equivalent to Gemma source tracing.
- A negative result means this compact adapter route failed, not that all Qwen source routes failed.

### 3. Only layer 26 is tested

All feature nodes in the current full runs are layer 26:

- primary: 420 feature rows, all `layer=26`
- strict: 420 feature rows, all `layer=26`

Impact:

- If Qwen's source route lives at layer 22, 24, 28, or is distributed across layers, the current test can miss it.
- This is especially relevant because hidden-state bridge effects and feature/source-control effects may peak at different layers.

### 4. Only top support nodes are exported and intervened

Although each graph initially selects up to 96 feature nodes, controlled compare exports only the top incoming nodes under `--topk-per-node 3`. Intervention then uses only `--top-features-per-sample 2`.

Observed full-run coverage:

- 70 valid samples
- 2 prompts per sample
- 3 feature nodes exported per sample/prompt
- 2 feature nodes intervened per sample/prompt
- 560 intervention rows = 70 samples x 2 prompts x 2 features x 2 modes

Impact:

- This is a top-node smoke, not exhaustive node intervention.
- If the route is distributed across many medium-weight nodes, single-node zeroing can look negative.

### 5. Node selection favors positive direct-effect support features

The adapter uses `--node-sign support`, so it filters to features whose activation times decoder-vector dot target direction is positive.

Impact:

- Suppressor, inhibitory, or contrastive features are not part of the primary selected graph.
- Gemma mainline distinguishes support and suppressor routes; Qwen needs the same split before a final source-tracing verdict.

### 6. Direct-effect scoring is a proxy

The score uses `decoder_vector dot target_logit_direction`. This is a useful approximation, but Qwen's final logit passes through later layers and final normalization before unembedding.

Impact:

- Some selected "high direct-effect" layer-26 features may not actually control the final answer token.
- Better variants should use gradient-weighted, path-patched, or multi-feature restore/corrupt scores.

### 7. Controls are incomplete for full Gemma-style replication

The Stage4 Qwen source-tracing adapter currently has graph compare and feature-node zeroing, but it does not yet fully include all Gemma-style controls in the same true-tracing adapter:

- nearest/matched non-source controls
- random16 feature controls
- real mask vs shifted/shuffled under the adapter graph
- correct target vs wrong target under the adapter graph
- multi-feature restore/corrupt
- sequence-score or decoded behavior bridge

Impact:

- Even a positive rerun would first be "Qwen source-tracing adapter support", not immediately "full Gemma-style route replication."

## Revised Interpretation

Current Qwen wording should be:

`Qwen2.5-VL-PLT has paperpack evidence for approximate feature/source-control and hidden/first-token bridges. The first Stage4 compact answer-aligned source-tracing adapter did not show stable single-node target damage, but this result is provisional because the adapter had a layer-output alignment risk and only tested a narrow top-node layer-26 route.`

Do not write:

- `Qwen fully replicates Gemma-style source tracing.`
- `Qwen has no evidence-sensitive mechanism.`
- `Qwen source tracing is definitively negative.`

## Required Follow-Up Before Final Negative Claim

Before making a negative claim about Qwen full source tracing, rerun:

1. Hook-aligned Qwen source-tracing smoke.
2. Hook-aligned primary72 full.
3. Hook-aligned strict72 sensitivity.
4. Layer sweep for PLT source tracing: at minimum layers `22,24,26,28,30` if assets support them.
5. Multi-feature intervention: top `4/8/16/32`, not only top2 single-node zeroing.
6. Matched controls and negative controls inside the same adapter path.

Only if the adapter runs after these fixes and the route/controls still fail should the verdict become:

`Qwen does not support Gemma-style source tracing under the tested hook-aligned PLT adapter, layers, controls, and paperpack.`

## Follow-Up Completed

The hook-aligned rerun is documented in `012_qwen_source_tracing_hookfix_rerun.md`.

Summary:

- Hook-aligned layer 26 primary72 and strict72 full runs completed.
- Top8 primary72 and strict72 sensitivity completed.
- Layer 22 and layer 24 smoke runs completed and did not show supportive direction.
- Layer 27 was blocked by current PLT asset/loader index range.

Updated verdict:

`Qwen2.5-VL-PLT retains approximate feature/source-control evidence, but Gemma-style source tracing is not supported under the tested hook-aligned adapter and sensitivity settings.`
