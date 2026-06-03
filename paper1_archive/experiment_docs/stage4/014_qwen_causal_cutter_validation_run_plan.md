# Stage4-014 Qwen Native Causal Cutter Validation Run Plan

## 目的

本实验不复用 Gemma 的节点地图。Gemma 只提供实验范式：`source/control`、`wrong-target`、`shifted/shuffled`、mask sensitivity，以及 rank/logit behavior bridge 的判据。

Qwen 候选节点必须从 Qwen2.5-VL-PLT 自己的 hook-aligned top8 intervention 结果中产生。换句话说，这里验证的是 “Qwen 自己地图里剪掉会伤 target 的节点是否是真因果节点”，不是把 Gemma 的路线贴到 Qwen 上。

## 输入

- Discovery source: `cross_model/stage4_qwen_source_tracing_primary_full_hookfix_top8_intervention.csv`
- Confirmation-only source: `cross_model/stage4_qwen_source_tracing_strict_full_hookfix_top8_intervention.csv`
- Candidate manifest: `cross_model/stage4_qwen_causal_cutter_candidate_manifest.csv`
- Model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Asset: `KokosDev/qwen2p5vl-7b-plt`
- Data: `paperpack72_primary` masks and images

Strict rows are used only as confirmation/sensitivity metadata. They are not used to select the main candidates.

## 候选规则

- Main rank condition: `original_target_rank <= 5`
- Sensitivity rank condition: `original_target_rank <= 10`
- Damage condition: `delta_target_logit <= -0.25` or `delta_target_rank >= 1`
- Exclusions: target token empty/special, top1 token special, malformed sample/prompt/feature/position fields
- Main candidate cap: at most 24 prompt-runs
- Numeric cap: at most 12 prompt-runs
- Non-numeric cap: at most 12 prompt-runs

Current manifest audit:

- Primary top8 intervention rows: `2240`
- Strict top8 intervention rows: `2240`
- Rank<=10 prompt-run candidates: `17`
- Main candidates: `12`
- Main numeric: `7`
- Main non-numeric: `5`
- Main candidates with strict confirmation: `10`

## 方法

For each Qwen-native candidate:

1. `source_zeroing`: intervene on the candidate source feature at the candidate source position.
2. `same_position_matched_feature_control`: same position, matched active feature, excluding known damaging discovery features where possible.
3. `same_feature_random_position_control`: same feature, different visual/answer-adjacent position.
4. `random_active_feature_control`: same position, random active non-source feature.
5. `wrong_target_zeroing`: score the strongest clean non-target token under the same interventions.
6. `answer_mask_restore` and `union_mask_restore`: restore clean-minus-mask feature contribution in masked runs.
7. `shifted_mask_restore` and `shuffled_mask_restore`: same restore test under spatial controls.

## 输出

- `cross_model/stage4_qwen_causal_cutter_validation_smoke_raw.csv`
- `cross_model/stage4_qwen_causal_cutter_validation_smoke_run.json`
- `cross_model/stage4_qwen_causal_cutter_validation_full_raw.csv`
- `cross_model/stage4_qwen_causal_cutter_validation_full_run.json`
- `cross_model/stage4_qwen_causal_cutter_validation_summary.csv`
- `cross_model/stage4_qwen_causal_cutter_validation_specificity.csv`
- `cross_model/stage4_qwen_causal_cutter_validation_case_table.csv`
- `cross_model/stage4_qwen_causal_cutter_validation_decision.json`

## 判据

- `causal_cutter_supported`: source clean zeroing stronger than matched controls, bootstrap CI does not cross 0, positive fraction stable.
- `evidence_linked_cutter_supported`: causal cutter supported, plus real mask restore stronger than shifted/shuffled, and correct target stronger than wrong target.
- `numeric_only_supported`: only numeric slice passes, written as a numeric-answer subphenomenon.
- `not_supported`: source is close to controls or strict/control specificity is unstable.
- `blocked`: runner, model, asset, or resource failure prevents interpretation.

## Claim Boundary

Allowed if positive:

`Qwen causal-screened PLT cutter nodes exist.`

Allowed if mask/target controls also pass:

`Qwen causal-screened evidence-linked cutter support.`

Not allowed:

- `Qwen fully replicates Gemma-style source tracing.`
- `Qwen has no evidence-sensitive mechanisms.`
- `Feature IDs are semantic object nodes.`

## 当前状态

Completed:

- Local scripts compile.
- Candidate manifest generated.
- 6-candidate smoke completed.
- 12-candidate main run completed.
- 17-candidate rank<=10 sensitivity completed.

Result is recorded in `015_qwen_causal_cutter_validation_results.md`.
