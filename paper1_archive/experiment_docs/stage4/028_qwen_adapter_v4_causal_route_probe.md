# Stage4-024 Qwen Adapter V4 Causal Route Probe

## 目的

测试自动路线失败是不是因为 Adapter V2/V3 的 scoring 太像 Gemma，不适合 Qwen。Adapter V4 不训练，只用 frozen score。

## 输入

- Hidden mediation weight
- Evidence sensitivity
- Target attribution
- Zeroing damage

## 输出

- `stage4_qwen_decisive_route_adapter_v4_*`

## 方法

Adapter V4 score：

`route_score = hidden_mediation * evidence_sensitivity * target_attribution * zeroing_damage`

primary discovery 冻结 route map；strict confirmation 不参与选点。输出 Qwen-native route map，不复用 Gemma node ids。

## 结果

待前置实验完成后运行。

## 预期与实际偏差

如果 Adapter V4 通过，写 `Qwen-native source-tracing-like route support`，仍不写 full Gemma replication。如果 Adapter V4 失败，只否定当前 Qwen-native adapter route probe。

## 结论

待运行。
