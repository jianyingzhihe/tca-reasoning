# Stage4-035 Qwen Adapter V4 Route Design

## 目的

定义 Qwen Adapter V4 的自动 route scoring。V4 目标不是复制 Gemma 的节点，而是用 Qwen 自己的 layer 14 artifacts 生成可检验 route。

## 输入

- L14 source-tracing intervention rows
- L14 evidence-first PLT candidate rows
- hidden layer confirmation result from Stage4-033

## 输出

Adapter V4 manifest，字段复用 Qwen cutter/evidence-first manifest schema，并新增：

- `adapter_v4_score`
- `hidden_layer_weight`
- `zeroing_damage`
- `v4_match_level`

## 方法

V4 score 固定为：

`positive(path_mass) * positive(evidence_specificity) * positive(target_attribution) * positive(zeroing_damage) * hidden_layer_weight`

其中：

- `path_mass` 来自 Qwen source-tracing compare/intervention artifacts。
- `evidence_specificity` 来自 answer/union mask drop 相对 shifted/shuffled 的差异。
- `target_attribution` 来自 feature decoder direction 对 target token 的正贡献。
- `zeroing_damage` 来自 source-tracing subtract zeroing 对 target logit/rank 的损伤。
- `hidden_layer_weight` 主层 L14 为 `1.0`，邻层 sensitivity 按距离衰减。

primary 只用于冻结 scoring policy；strict 只按同一 policy 验证，不重新挑规则。

## 结果

待运行。

## 预期与实际偏差

严格主分析优先 `exact_pos_feature` match。若 exact match 稀疏，可跑 `same_feature` diagnostic，但不能把 fallback 结果写成完整自动 route 复现。

## 结论

待运行。
