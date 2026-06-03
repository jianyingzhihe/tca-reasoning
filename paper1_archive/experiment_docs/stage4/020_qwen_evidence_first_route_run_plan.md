# Stage4-020 Qwen Evidence-First Route Validation Run Plan

## 目的

本轮专门补 Qwen2.5-VL-PLT 目前最薄的一环：`evidence-region specificity`。前面的 expanded cutter 证明了“有些 Qwen PLT 节点剪掉会伤答案”，但它们未必真的来自图像证据区域。本实验反过来先找“被真实 evidence mask 破坏、但不被 shifted/shuffled control 同等破坏”的 Qwen 自有 feature/position，再验证这些节点是否支撑目标答案。

## 输入

- 模型：`Qwen/Qwen2.5-VL-7B-Instruct`
- 资产：`KokosDev/qwen2p5vl-7b-plt`
- 层：主层 `26`
- 数据：`paperpack72_primary` 与 `paperpack72_strict_sensitivity`
- prompt：`B_direct`、`D_visual_only`
- mask 条件：`answer_mask`、`union_mask`、`shifted_mask`、`shuffled_mask`
- 位置组：默认 `visual_answer`，即 Qwen image span 与 answer-adjacent positions 的并集

## 输出

- `cross_model/stage4_qwen_evidence_first_primary_*`
- `cross_model/stage4_qwen_evidence_first_strict_*`
- `cross_model/stage4_qwen_adapter_v3_primary_*`
- `cross_model/stage4_qwen_adapter_v3_strict_*`
- 汇总：`stage4_qwen_evidence_first_route_summary.csv`
- specificity：`stage4_qwen_evidence_first_route_specificity.csv`
- 决策：`stage4_qwen_evidence_first_route_decision.json`

## 方法

1. Evidence-first discovery：对 clean、answer/union mask、shifted/shuffled mask 分别抽取 layer 26 PLT activation，筛选 `clean - real_mask` 大于 `clean - shifted/shuffled` 的 feature/position。
2. Target-attribution filter：只保留 decoder vector 对 target logit 方向为正，且不被 wrong-target contribution 吸收的候选。
3. Qwen-native validation：对 evidence-first candidates 跑 clean zeroing、matched controls、wrong-target、real-mask restore、shifted/shuffled restore。
4. Group intervention：对 `top1/top4/top8/top16` evidence-first candidates 做 grouped restore。
5. Adapter V3 route probe：把自动 source tracing 候选重新按 `path_mass * evidence_specificity * target_attribution` 打分，再跑同一套 validation。

## 结果

待运行。本轮先跑 6 prompt-run smoke；smoke 通过后跑 primary full，再跑 strict sensitivity。

## 预期与实际偏差

预期如果 Qwen 的答案支撑路线确实被 localized evidence region 驱动，evidence-first candidates 应该满足：real mask drop 强于 shifted/shuffled，source zeroing 强于 matched controls，correct target 效果强于 wrong target，并在 first-token/rank 或 sequence score 上有方向性。

如果 candidates 很少，记录为 evidence-sensitive PLT feature 稀疏限制；如果 clean zeroing 强但 restore 不强，继续保持 `cutter-only` 口径。

## 结论

本文件是 run plan，不给科学结论。允许的最终结论只有：

- `qwen_native_evidence_linked_supported`
- `qwen_adapter_v3_route_supported`
- `qwen_cutter_only_supported`
- `qwen_not_gemma_style_under_adapter`
- `blocked`

本轮不复用 Gemma node map，只复用 Gemma 的判据。
