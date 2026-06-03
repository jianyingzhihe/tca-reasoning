# Stage4-024/033 Qwen Decisive Route Verdict

## 目的

汇总 Qwen2.5-VL-PLT 的 broad Qwen-native route tests，判断此前失败更像哪一种：

- `layer_miss`
- `distributed_route`
- `plt_localization_failure`
- `route_absent_under_test`

本 verdict 只针对当前 paperpack72、公开 Qwen2.5-VL-PLT、现有 inference-time intervention 设定，不否定 Qwen 存在其他跨模态机制。

## 输入

- `cross_model/stage4_qwen_decisive_route_decision.json`
- `cross_model/stage4_qwen_decisive_route_summary.csv`
- `cross_model/stage4_qwen_decisive_route_specificity.csv`
- all-layer hidden raw artifacts with tag `alllayers`

## 输出

当前限定结论与下一步实验指向。

## 方法

Decision matrix：

- hidden primary + strict 成立：`qwen_hidden_route_supported`
- hidden 成立但 PLT feature/source route 失败：后续候选为 `plt_localization_failure`
- 单层失败、多层通过：后续候选为 `distributed_route`
- Adapter V4 通过：`qwen_adapter_v4_route_supported`
- hidden/PLT/multilayer/adapter 全失败：才可写 `qwen_gemma_style_sparse_route_not_supported under broad Qwen-native tests`

## 结果

Stage4-033 all-layer hidden retest 已完成。

Primary-selected gate：

- layer: `14`
- direction: `restore`
- position group: `top_hidden_delta`
- mask condition: `answer_mask`
- usable prompt-runs: `96`

Primary metrics：

- target logit effect mean: `1.7544`, CI low: `1.2872`
- real minus shifted mean: `1.1148`, CI low: `0.6851`
- real minus shuffled mean: `1.2932`, CI low: `0.8464`
- correct minus wrong mean: `0.3148`, CI low: `0.0424`
- target rank effect mean: `803.74`, CI low: `171.58`

Strict confirmation of the same primary-selected gate：

- usable prompt-runs: `101`
- target logit effect mean: `1.7242`, CI low: `1.2684`
- real minus shifted mean: `1.1144`, CI low: `0.7042`
- real minus shuffled mean: `1.2183`, CI low: `0.8010`
- correct minus wrong mean: `0.3311`, CI low: `0.0578`
- target rank effect mean: `766.53`, CI low: `151.23`

Decision status：

`qwen_hidden_route_supported`

## 预期与实际偏差

此前 layer 26-heavy 的 Qwen PLT/source-tracing 测试存在真实的 `layer_miss` 风险。全层 hidden retest 显示，当前最稳定的 hidden evidence-to-answer gate 在 layer 14，而不是 layer 26。

分析器也修正了一个重要口径问题：strict set 不能重新挑一个 strict-best gate 来替代 primary discovery gate。现在 strict gate 明确是对 primary-selected layer 14 gate 的 confirmation；strict-best gate 只作为 diagnostic。

## 结论

可以写：

`Qwen2.5-VL shows a replicated hidden-level evidence-region-sensitive answer-support route at layer 14 on primary and strict paperpack splits.`

不能写：

`Qwen fully replicates Gemma-style source tracing.`

也不能写：

`Qwen has no evidence-sensitive / cross-modal mechanism.`

当前最合理解释已经从“Qwen 可能没有 route”转向：

`Qwen has hidden-level route support, while current PLT/source-tracing localization has not yet captured a Gemma-style sparse feature route.`

下一步应围绕 layer 14 及邻近层做 PLT dense sweep、multi-layer route patch、Adapter V4 route probe，而不是继续固定 layer 26。

## Stage4-034 Adapter V4 Follow-up

已新增 Stage4-034/035/036/037 文档与 V4 执行计划。下一轮优先补 Adapter V4，而不是继续扩大 layer 26。

V4 固定主层为 layer `14`，score 为：

`positive(path_mass) * positive(evidence_specificity) * positive(target_attribution) * positive(zeroing_damage) * hidden_layer_weight`

主结论只允许来自 primary discovery + strict confirmation。同一套 scoring policy 必须冻结后再用于 strict，strict 不参与挑规则。
