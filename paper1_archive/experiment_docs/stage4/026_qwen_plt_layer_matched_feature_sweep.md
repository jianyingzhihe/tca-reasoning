# Stage4-024 Qwen PLT Layer-Matched Feature Sweep

## 目的

只在 hidden lattice 支持或接近支持的层上做 PLT feature sweep，避免把失败误归因于 layer 26。

## 输入

- Hidden Causal Lattice 的 top layers / near-pass layers
- Qwen2.5-VL-PLT
- paperpack72 primary / strict

## 输出

- `stage4_qwen_decisive_route_plt_layer_sweep_*`

## 方法

候选分数固定为：

`evidence_sensitivity * target_attribution * hidden_mediation_weight`

topK 为 `1,4,8,16,32,64`。对照包括 same-layer activation-matched、drop-matched、mask-insensitive 和 random-active。primary 只用于 discovery；strict 只用于 confirmation。

## 结果

待 hidden lattice 完成后运行。

## 预期与实际偏差

如果某些层通过，写 `Qwen route is layer-dependent`。如果 hidden 通过但 PLT 全失败，写 `Qwen has hidden-level evidence route, but PLT feature localization failed`。

## 结论

待运行。
