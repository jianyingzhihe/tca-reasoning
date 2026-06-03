# Stage4-015 Qwen Native Causal Cutter Validation Results

## 目的

记录 Stage4-014 的实际运行结果，判断 Qwen2.5-VL-PLT 中由 Qwen 自己 discovery 得到的 cutter candidates 是否通过 controls。

## 输入

- Candidate manifest: `cross_model/stage4_qwen_causal_cutter_candidate_manifest.csv`
- Main candidates: `12`
- Numeric candidates: `7`
- Non-numeric candidates: `5`
- Strict-confirmed main candidates: `10`

## 输出

- `cross_model/stage4_qwen_causal_cutter_validation_smoke_raw.csv`
- `cross_model/stage4_qwen_causal_cutter_validation_smoke_run.json`
- `cross_model/stage4_qwen_causal_cutter_validation_smoke_summary.csv`
- `cross_model/stage4_qwen_causal_cutter_validation_smoke_specificity.csv`
- `cross_model/stage4_qwen_causal_cutter_validation_smoke_case_table.csv`
- `cross_model/stage4_qwen_causal_cutter_validation_smoke_decision.json`
- `cross_model/stage4_qwen_causal_cutter_validation_full_raw.csv`
- `cross_model/stage4_qwen_causal_cutter_validation_full_run.json`
- `cross_model/stage4_qwen_causal_cutter_validation_summary.csv`
- `cross_model/stage4_qwen_causal_cutter_validation_specificity.csv`
- `cross_model/stage4_qwen_causal_cutter_validation_case_table.csv`
- `cross_model/stage4_qwen_causal_cutter_validation_decision.json`
- `cross_model/stage4_qwen_causal_cutter_validation_full_sensitivity_rank10_raw.csv`
- `cross_model/stage4_qwen_causal_cutter_validation_full_sensitivity_rank10_decision.json`

## 方法

Each candidate is rerun in Qwen2.5-VL-PLT with Qwen-native feature id and Qwen-native source position. The validation never reads Gemma node ids or Gemma feature maps.

Controls:

- source feature at source position
- same-position matched feature control
- same-feature random-position control
- random active feature control
- wrong target token
- answer/union evidence mask restore
- shifted/shuffled mask restore controls

## 结果

- `python -m py_compile` passed for the new builder, runner, analyzer, and research validation script.
- Candidate manifest generated successfully.
- Main candidate image and mask paths all exist locally.
- Remote smoke completed: 6/6 candidates, 240 raw rows, no skipped rows.
- Remote full main completed: 12/12 candidates, 480 raw rows, no skipped rows.
- Remote rank<=10 sensitivity completed: 17/17 candidates, 680 raw rows, no skipped rows.

Main result:

- Decision: `causal_cutter_supported`
- `clean_source_minus_controls`: mean `0.7951`, CI `[0.3819, 1.2743]`, positive fraction `0.9167`
- `clean_correct_minus_wrong`: mean `0.1094`, CI `[0.0625, 0.1615]`, positive fraction `0.6667`
- Numeric slice: mean source-control `1.2857`, CI `[0.8155, 1.9048]`
- Non-numeric slice: mean source-control `0.1083`, CI `[0.0417, 0.1750]`

Mask/evidence-link result:

- `answer_mask_restore_source_minus_controls`: mean `-0.0243`, CI `[-0.0556, 0.0]`
- `union_mask_restore_source_minus_controls`: mean `-0.0243`, CI `[-0.0764, 0.0243]`
- `answer real > shifted`: not supported
- `union real > shifted`: not supported
- `real > shuffled`: weak/inconsistent, not enough for evidence-linked claim

Rank<=10 sensitivity:

- Decision: `causal_cutter_supported`
- `clean_source_minus_controls`: mean `0.5748`, CI `[0.2672, 0.9424]`, positive fraction `0.8824`
- Numeric remains strong; non-numeric remains positive but small.
- Mask restore controls still do not support evidence-linked cutter.

## 预期与实际偏差

- Candidate selection is post-hoc relative to Qwen top8 intervention rows, so positive results support Qwen-native causal-screened cutter nodes, not automatic Gemma-style tracing.
- Non-numeric candidate count is only `5`; if only numeric passes, the correct conclusion is numeric-answer subphenomenon.
- Same-position matched controls exclude known damaging discovery features where possible, but they are still empirical controls rather than guaranteed null features.
- Actual result is stronger than numeric-only: non-numeric candidates also show positive source-control clean-zeroing, but the effect is much smaller than numeric.
- Actual result is weaker than evidence-linked: restore under answer/union masks does not beat shifted/shuffled controls.

## 结论

Supported:

`Qwen causal-screened PLT cutter nodes exist.`

Not supported by this experiment:

`Qwen causal-screened evidence-linked cutter support.`

Still not allowed:

`Qwen fully replicates Gemma-style source tracing.`

Plain-language interpretation:

Qwen has some PLT features that behave like real "cut wires": when we cut them, the target answer logit drops more than matched controls. But these cutter nodes are not yet tied back to the annotated evidence masks strongly enough, so they are causal-screened answer-support nodes rather than a full evidence-region route.
