# Stage4-018 Qwen Evidence-Linked Cutter V2

## 目的

Stage4-014 证明了 Qwen-native PLT cutter nodes 可以被剪伤 target，但没有证明这些节点和标注 evidence region 绑定。本实验用 expanded V2 candidates 做 activation-drop mediation 和 grouped restore。

## 输入

- `stage4_qwen_expanded_cutter_candidate_manifest.csv`
- Qwen2.5-VL-PLT layer `26`
- Masks: `answer_mask`, `union_mask`, `shifted_mask`, `shuffled_mask`

## 方法

- Candidate discovery 只来自 primary expanded source-tracing V2。
- Strict 只做 confirmation/sensitivity。
- 每个 sample/prompt 保留 top16 pool candidates。
- Grouped restore 使用 `top1/top4/top8/top16`。
- Controls:
  - source candidates
  - same-position matched feature controls
  - same-feature other-position controls
  - random-active controls

## 结果

- `stage4_qwen_evidence_linked_cutter_v2_smoke_raw.csv`
- `stage4_qwen_evidence_linked_cutter_v2_full_raw.csv`
- `stage4_qwen_mainline_v2_summary.csv`
- `stage4_qwen_mainline_v2_specificity.csv`
- `stage4_qwen_mainline_v2_decision.json`

Expanded candidate manifest:

- prompt-runs with candidates: `19`
- pool candidates: `43`
- main candidates: `31`
- main numeric: `16`
- main non-numeric: `15`
- strict-confirmed main: `26`

Expanded clean source/control validation:

- candidates: `31`
- decision: `causal_cutter_supported`
- `clean_source_minus_controls`: mean `0.4872`, CI `[0.2870, 0.7030]`, positive fraction `0.8065`
- `clean_correct_minus_wrong`: mean `0.0847`, CI `[0.0524, 0.1169]`
- numeric slice: mean `0.8958`, CI `[0.6250, 1.2135]`
- non-numeric slice: mean `0.0514`, CI `[0.0222, 0.0833]`

Grouped evidence-link V2:

- prompt-runs: `19`
- raw rows: `2432`
- decision: `qwen_cutter_only_supported`
- answer/union grouped restore did not beat source controls.
- answer/union grouped restore did not beat shifted/shuffled controls.
- union-mask activation-drop showed partial real-vs-shifted signal at top4/top8/top16, but not enough because real-vs-shuffled and restore specificity did not pass.

## 预期与实际偏差

If grouped restore succeeds but automatic tracing still fails, the correct claim is Qwen-native evidence-linked support, not Gemma-style full replication.

If grouped restore fails but clean zeroing remains strong, the correct claim remains cutter-only support.

Actual result follows the second case: expanded clean zeroing is robust, but grouped evidence-link restore does not close.

## 结论

Supported:

`Qwen expanded causal-screened PLT cutter nodes exist.`

Not supported:

`Qwen native evidence-linked cutter support.`

Not supported:

`Qwen Gemma-style automatic source tracing replication.`
