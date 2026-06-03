# Stage6-014 Gemma Hidden-Layer Symmetric Probe

## Summary

This experiment mirrors the Qwen Stage4 all-layer hidden residual lattice on Gemma3. The question is not whether Gemma and Qwen have identical sparse graphs. The question is whether the same hidden-level causal lens can be applied to both models:

> If a real evidence mask damages the answer, can patching selected hidden residual positions from clean into masked recover the correct answer more than shifted/shuffled masks and wrong targets?

This is a symmetric hidden residual experiment. It does not replace Gemma sparse PLT source tracing.

## Fixed Design

- Model: `google/gemma-3-4b-it`
- Data: `paperpack72_primary_prompt_runs.csv` and `paperpack72_strict_sensitivity_prompt_runs.csv`
- Prompts: `B_direct`, `D_visual_only`, `C_step_only`, `A_step_visual`
- Masks: `answer_mask`, `union_mask`, `shifted_mask`, `shuffled_mask`
- Layers: all Gemma text language layers, `0..num_hidden_layers-1`
- Hook target: `model.model.language_model.layers[layer]`
- Hidden source index: `layer + 1`
- Position groups: `visual_span`, `answer_adjacent`, `visual+answer`, `top_hidden_delta`
- Directions: `restore` and `corrupt`
- Scale: `1.0`

`visual_span` is detected from `token_type_ids == 1` when available, with an image-token fallback. `top_hidden_delta` is selected within `visual+answer` positions using the norm of `clean_hidden - masked_hidden`.

## Metrics

The analyzer uses the same hidden gate metrics as the Qwen Stage4 analyzer:

- `hidden_effect`
- `hidden_real_minus_shifted`
- `hidden_real_minus_shuffled`
- `hidden_correct_minus_wrong`
- `hidden_rank_effect`

Pass rule:

```text
n >= 8
CI low > 0
positive_frac >= 0.6
rank_effect mean > 0
```

Primary full discovers the best gate. Strict full only confirms the exact primary-selected gate and keeps strict-best as diagnostic.

## Commands

Local compile:

```powershell
python -m py_compile scripts/local/run_stage6_gemma_hidden_causal_lattice_remote.py scripts/local/analyze_stage6_hidden_cross_model_symmetry.py
```

Smoke:

```powershell
python scripts/local/run_stage6_gemma_hidden_causal_lattice_remote.py --mode smoke --packs primary --layers 0,14,27,33 --max-prompt-runs 4 --tag symmetric_v1 --detach
python scripts/local/run_stage6_gemma_hidden_causal_lattice_remote.py --mode smoke --packs primary --tag symmetric_v1 --fetch-only
python scripts/local/analyze_stage6_hidden_cross_model_symmetry.py --mode smoke --tag symmetric_v1
```

Primary full:

```powershell
python scripts/local/run_stage6_gemma_hidden_causal_lattice_remote.py --mode full --packs primary --layers all --tag symmetric_v1 --detach
python scripts/local/run_stage6_gemma_hidden_causal_lattice_remote.py --mode full --packs primary --tag symmetric_v1 --fetch-only
python scripts/local/analyze_stage6_hidden_cross_model_symmetry.py --mode full --tag symmetric_v1
```

Strict full:

```powershell
python scripts/local/run_stage6_gemma_hidden_causal_lattice_remote.py --mode full --packs strict --layers all --tag symmetric_v1 --detach
python scripts/local/run_stage6_gemma_hidden_causal_lattice_remote.py --mode full --packs strict --tag symmetric_v1 --fetch-only
python scripts/local/analyze_stage6_hidden_cross_model_symmetry.py --mode full --tag symmetric_v1
```

## Decision Boundary

If Gemma hidden succeeds, we can write that both Gemma and Qwen support an evidence-to-answer causal route under the same hidden residual lens.

If Gemma hidden fails, we do not weaken the Gemma sparse PLT route claim. The result would mean Gemma's strongest visible route remains the sparse source-tracing graph, while Qwen's strongest visible route remains hidden residual / reconstruction error plus feature-node support.

Raw logit magnitudes are not compared across Gemma and Qwen. The cross-model table compares direction, CI sign, positive fraction, normalized layer location, and primary-to-strict retention.
