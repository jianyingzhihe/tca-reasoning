# Stage6-011: Gemma Fixed-Node Prompt Probe

## Goal

Make Gemma answer the same question Qwen Stage6 answered: if we freeze a feature node discovered in the baseline route, does that same node still causally support the answer under text rewrites and prompt/CoT variants?

## Candidate Definition

- Source nodes come from Gemma `B_direct + original` source-tracing graphs.
- Only traced feature nodes are selected.
- Default selection is top 4 feature nodes per original sample by `path_mass_best`.
- Controls are other high-path-mass feature nodes from the same baseline graph.

## Main Metrics

- `source_damage_target_logit`: baseline target logit minus target logit after zeroing the source feature.
- `source_minus_controls`: source damage minus mean control-node damage.
- `correct_minus_wrong`: source target damage minus source wrong-token damage.
- `prefix_ok`: attribution prefix was found.
- `target_token_same`: condition target token matches the baseline target token.

## Interpretation

Rows with `prefix_ok=0` or `target_token_same=0` are format/target-alignment failures, not mechanism failures. Main causal summaries filter to `prefix_ok=1 && target_token_same=1`.

## Coverage Accounting

Gemma fixed-node prompt probing uses three explicit denominators:

- `planned_rows`: all fixed baseline feature nodes scheduled for prompt/text probing.
- `planned_aligned_rows`: planned rows whose output format and target token are comparable (`prefix_ok=1 && target_token_same=1`).
- `usable_exact_rows`: planned-aligned rows that also execute successfully at the exact frozen `source_layer/source_pos/source_feature_id`.

Main fixed-node causal summaries use `usable_exact_rows`.

Rows with `source_pos_out_of_range` are not interpreted as evidence that the mechanism failed. They mean the exact frozen token position no longer exists under that prompt/text condition, so they are reported as coverage/alignment diagnostics. If these rows remain sparse, the exact fixed-node lens is considered usable. If they become common, especially within the main claim samples or a full prompt/text family, the experiment must be upgraded to a remapped-position or route-level lens before making a strong prompt-stability claim.
