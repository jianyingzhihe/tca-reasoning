# Stage6-013: Unified Cross-Model Verdict

## Current Status

Updated after the 2026-05-31 power-loss recovery.

- Old-result unified reanalysis completed for tag `unified_v1`.
- Gemma fixed-node smoke completed for tag `unified_v1`.
- Qwen route-identity smoke completed for tag `unified_v1`.
- Qwen focused route-identity full completed on AutoDL/gpu1 under tag `unified_v1_focus4`.
- Gemma fixed-node full was previously engineering-blocked by cgroup file-cache pressure; the cache-evict mitigation has now been validated, and the full rerun completed under tag `unified_v1_focus4`.

## Completed Smoke / Reanalysis

Old Qwen Stage6 fixed-node reanalysis:

- Source: `stage6_prompt_text_cot_full_prompttext_v1_candidate_metrics.csv`.
- Rows: `2304`.
- Main interpretation: report `restore_source_minus_controls`, `real_minus_shifted`, `real_minus_shuffled`, `restore_correct_minus_wrong`, and `pos234`; do not treat inherited `evidence_specificity` alone as prompt-stability evidence.

Old Gemma Stage6 graph-overlap reanalysis:

- Source prefix: `stage6_gemma_prompt_text_cot_full_gemmaprompt_v1_mf8_sharded`.
- Rows: `143`.
- Aligned rows: `119`.
- Prefix / target-token mismatch rows: `24`.
- Main interpretation: format and target-token failures are diagnostics, not mechanism failures.

Gemma fixed-node smoke:

- Tag: `unified_v1`.
- Rows: `12`.
- Aligned rows: `8`.
- Status: smoke passed; the fixed-node ablation path is technically valid on a small batch.

Gemma fixed-node post-fix probe:

- Tag: `unified_v1_focus4`.
- Rows: `24`.
- Aligned/ok rows: `12`.
- Skipped unaligned rows: `12`.
- Errors: `0`.
- Status: passed after targeted cgroup file-cache eviction.
- Note: these probe artifacts are temporary and will be overwritten by the full `unified_v1_focus4` rerun once it completes.

Qwen route-identity smoke:

- Tag: `unified_v1`.
- Candidate metrics: `48`.
- TopK overlap rows: `32`.
- Status: smoke passed across L10/L13/L15.

## Qwen Focused Full Result

Qwen focused route-identity full:

- Tag: `unified_v1_focus4`.
- Server: AutoDL/gpu1.
- Filter: top `4` candidates per `sample_id + question_variant + prompt_family + layer`.
- Filtered rows: `4608`.
- Layers: L10-L17.
- Rationale: the uncapped manifest has `49812` candidates and would likely take tens of hours; `focus4` keeps the complete `12 samples x 3 variants x 4 prompt families x 8 layers` grid while making TopK@32 feasible.
- Remote run completed with all eight layer raw/run JSON files present.
- Final remote disk stayed at about `160G` free.
- Analyzer output:
  - `stage6_unified_routeidentity_full_unified_v1_focus4_candidate_metrics.csv`
  - `stage6_unified_routeidentity_full_unified_v1_focus4_topk_overlap.csv`
  - `stage6_unified_routeidentity_full_unified_v1_focus4_summary_by_prompt.csv`
  - `stage6_unified_routeidentity_full_unified_v1_focus4_summary_overlap_by_prompt_topk.csv`

Main Qwen route-strength pattern:

- Prompt-family route strength is stable rather than dramatically changed by prompt wording.
- `pos234` by prompt: `A_step_visual=0.406`, `B_direct=0.395`, `C_step_only=0.396`, `D_visual_only=0.378`.
- `restore_source_minus_controls` by prompt stays positive: roughly `0.028-0.031`.
- `restore_correct_minus_wrong` by prompt stays positive: roughly `0.015-0.018`.
- Text variants are also close: `pos234 original=0.387`, `paraphrase_1=0.403`, `paraphrase_2=0.391`.

Main Qwen route-identity pattern:

- Top route identity is prompt/text sensitive even when route strength remains positive.
- Exact `layer:pos:feature` overlap versus `B_direct + original` baseline:
  - By prompt, Top4: `A_step_visual=0.071`, `B_direct=0.370`, `C_step_only=0.103`, `D_visual_only=0.083`.
  - By prompt, Top8: `A_step_visual=0.161`, `B_direct=0.425`, `C_step_only=0.165`, `D_visual_only=0.157`.
  - By prompt, Top16: `A_step_visual=0.365`, `B_direct=0.576`, `C_step_only=0.347`, `D_visual_only=0.349`.
  - By text variant, Top4: `original=0.311`, `paraphrase_1=0.089`, `paraphrase_2=0.071`.
- Top32 overlap is `1.0` by construction in the `focus4` run because each condition has only `4 candidates x 8 layers = 32` candidates; it should not be used as evidence of identity stability.

Interpretation:

> Qwen Stage6 supports the exploratory claim that prompt/text changes do not erase the evidence-to-answer route-first causal signal, but they do re-rank or swap which feature nodes appear among the strongest top route candidates.

## Gemma Fixed-Node Full Result

Gemma fixed-node full:

- Tag: `unified_v1_focus4`.
- Server: AutoDL/gpu1.
- Completion time: `2026-06-01 04:19 CST`.
- Remote run JSON: `status=ok`, `rows=524`, `ok=414`, `skipped=96`, `errors=14`, `aligned_only=true`.
- The `14` errors are all `source_pos_out_of_range` exact-position coverage failures, not mechanism failures.

Gemma fixed-node coverage:

- `planned_rows=524`.
- `planned_aligned_rows=428`.
- `usable_exact_rows=414`.
- `usable_exact_over_planned=0.790`.
- `usable_exact_over_planned_aligned=0.967`.
- `source_pos_out_of_range_rows=14`.
- `source_pos_out_of_range_over_attempted=0.027`.

This is below the redesign threshold. Exact fixed-node probing is therefore usable for the main diagnostic, with out-of-range rows reported as coverage limitations.

Gemma fixed-node causal pattern:

- Aggregate exact fixed-node strength is weak/near-zero rather than robustly positive.
- By prompt family, `source_minus_controls_mean` is close to zero:
  - `A_step_visual=0.011`, positive fraction `0.235`.
  - `B_direct=-0.009`, positive fraction `0.381`.
  - `C_step_only=0.009`, positive fraction `0.313`.
  - `D_visual_only=-0.005`, positive fraction `0.267`.
- By text variant, `source_minus_controls_mean` is also close to zero:
  - `original=-0.001`, positive fraction `0.368`.
  - `paraphrase_1=0.0003`, positive fraction `0.259`.
  - `paraphrase_2=0.0016`, positive fraction `0.295`.
- `correct_minus_wrong` is mostly not positive in fixed-node ablation, so this lens does not support a strong claim that exact frozen feature nodes remain gold-answer-specific across prompt/text rewrites.

Interpretation:

> Gemma's source-tracing graph-level route remains the stronger representation of its mechanism. Under an exact fixed-node prompt/text probe, frozen baseline nodes are measurable with good coverage, but their individual causal strength is sparse and not robustly positive across prompt families or paraphrases.

## Verdict Slots

- Gemma fixed-node causal strength: full completed; usable as a diagnostic, but not a strong positive fixed-node stability result.
- Gemma route identity after target-aligned filtering: old `mf8_sharded` aligned reanalysis available.
- Qwen fixed-node causal strength from old Stage6 reanalysis: available under unified causal metrics.
- Qwen route identity/topK overlap: focused full completed; Top4/Top8 identity is prompt/text sensitive, while aggregate route strength remains positive.

## Fixed-Node Coverage Rule

Gemma fixed-node prompt/text probing now reports three denominators:

- `planned_rows`: all baseline feature nodes scheduled for cross-prompt probing.
- `planned_aligned_rows`: rows with comparable format and target token (`prefix_ok=1 && target_token_same=1`).
- `usable_exact_rows`: aligned rows that also execute successfully at the exact frozen `source_layer/source_pos/source_feature_id`.

Main fixed-node causal summaries and cross-model fixed-node tables use `usable_exact_rows`.

`source_pos_out_of_range` rows are coverage/alignment diagnostics, not mechanism failures. The completed full run shows these errors are sparse and remain below the redesign threshold, so they do not require redesign for the current exploratory claim. The redesign trigger is:

- if `source_pos_out_of_range` exceeds roughly `10%`, report it prominently as an exact-coordinate coverage limitation;
- if it exceeds roughly `15-20%`, or concentrates in the main claim samples/prompt family/text variant, add a remapped-position or route-level rescue experiment before writing strong prompt-stability claims.

## Gemma Fixed-Node Full Kill Diagnosis

The Gemma fixed-node full failures were diagnosed on 2026-05-31.

Observed failures:

- `unified_v1` full was killed during `ReplacementModel.from_pretrained_and_transcoders(...)`, before any fixed-node loop.
- `unified_v1_probe24` was also killed at the same model-initialization point, even with only `24` rows.
- A minimal repro that only loaded Gemma `ReplacementModel` and did not read the fixed-node manifest also exited with `SIGKILL 137`.

System-level evidence:

- Host `free -h` was misleading because the machine has about `1TiB` physical RAM, but the container cgroup has `memory.max = 96636764160` bytes, roughly `90GiB`.
- Before the failing repro, cgroup `memory.current` was about `85.4GB`.
- cgroup `memory.stat` showed this was almost entirely file page cache, not live Python RSS:
  - `file ~= 84.45GB`
  - `active_file ~= 84.25GB`
  - `anon ~= 0.33GB`
- During minimal Gemma initialization, cgroup memory rose from about `85.4GB` to above `94GB` in the monitor sample window, then the process was killed with status `137`.
- GPU memory was only about `749MiB` at the kill point, so this was not CUDA OOM.

Root cause:

> The Gemma fixed-node full was killed because old model/data file page cache was charged against the container's roughly `90GiB` cgroup memory limit. Gemma model initialization needed additional CPU/cgroup memory, but the cgroup was already nearly full of reclaimable file cache, so the process received `SIGKILL 137`.

Validation:

- Global `drop_caches` was denied by container permissions.
- cgroup `memory.reclaim` was unavailable / read-only.
- A targeted `posix_fadvise(..., POSIX_FADV_DONTNEED)` over HF/model/stage6 files succeeded:
  - cgroup `memory.current` dropped from about `85.37GB` to about `1.91GB`.
  - cgroup `file` dropped from about `84.45GB` to about `1.18GB`.
- After this targeted file-cache eviction, the same minimal Gemma `ReplacementModel` initialization completed successfully.

Mitigation added:

- `run_stage6_gemma_fixed_node_prompt_probe_remote.py` now prints cgroup memory before/after a targeted file-cache eviction step.
- The eviction step does not delete model/cache files; it only asks Linux to discard cached file pages using `posix_fadvise(DONTNEED)`.
- The post-fix `unified_v1_focus4 --max-rows 24` probe completed successfully with `12` ok aligned rows and `0` errors.
- The full `unified_v1_focus4` Gemma fixed-node rerun completed with the same patched runner.

## Unified Scale Outputs

The analyzer now emits two cross-model unified tables:

- `stage6_unified_crossmodel_unified_v1_focus4_fixednode_unified_rows.csv`
  - Lens: fixed-node causal strength.
  - Qwen metric: `clean_source_minus_controls`.
  - Gemma metric: `source_minus_controls`.
  - Interpretation: compare direction, positive fraction, and within-model baseline delta/retention; do not compare raw logit magnitudes across models.
- `stage6_unified_crossmodel_unified_v1_focus4_routeidentity_unified_rows.csv`
  - Lens: route identity stability.
  - Qwen metric family: feature-set Jaccard overlap.
  - Gemma metric family: source-tracing graph node/edge Jaccard overlap.
  - Interpretation: compare Jaccard-style overlap under explicitly labeled route objects; Qwen focus4 Top32 is marked as constructed full-pool overlap and excluded from main identity claims.

## Claim Language

Use this wording now that both lenses have completed:

> Under unified prompt/text metrics, both Gemma and Qwen expose evidence-to-answer mechanisms, but the stable object differs by model and representation level. Gemma's strongest evidence remains graph/route-level source tracing rather than exact frozen single-node stability under prompt rewrites. Qwen's route-first causal strength remains positive across prompt/text conditions, while the exact top feature identities are prompt/text sensitive and should be evaluated by Qwen-native topK feature overlap rather than Gemma-style edge graphs.

Do not write a mechanism failure from rows marked as format failure, target-token mismatch, missing graph, OOM, or disk blocked.

## Current Claim Boundary

The Stage6-010 result remains exploratory, but the unified-scale completion is now done. The safe current wording is:

> Stage6 unified metrics are operational. Qwen focused route-identity full shows that route-first causal strength remains positive across prompt/text conditions, while exact top feature identities are prompt/text sensitive. Gemma fixed-node full has enough exact-position coverage, but individual frozen nodes are weak/near-zero on average under prompt/text rewrites; Gemma's stronger Stage6 evidence remains route/graph-level overlap rather than exact single-node stability.
