# Stage4-024 Qwen Decisive Route Test Run Plan

## 目的

本轮把 Qwen2.5-VL-PLT 的问题从“继续找正例”改成“可证伪排查”。不训练新 adapter，不复用 Gemma 节点地图，只用 Qwen 自己的 hidden-state、PLT feature 和 intervention 结果判断：失败到底来自层没打中、路线分布式、PLT localization 失败，还是当前测试范围下确实不支持 Gemma-style sparse PLT route。

## 输入

- `paperpack72_primary_prompt_runs.csv`
- `paperpack72_strict_sensitivity_prompt_runs.csv`
- Qwen base：`Qwen/Qwen2.5-VL-7B-Instruct`
- PLT：`KokosDev/qwen2p5vl-7b-plt`
- masks：`answer_mask`、`union_mask`、`shifted_mask`、`shuffled_mask`

## 输出

- `stage4_qwen_decisive_route_hidden_*`
- `stage4_qwen_decisive_route_summary.csv`
- `stage4_qwen_decisive_route_specificity.csv`
- `stage4_qwen_decisive_route_decision.json`
- 后续 PLT layer sweep、multi-layer patch、Adapter V4 artifacts 也统一使用 `stage4_qwen_decisive_route_*` 前缀。

## 方法

先跑 Hidden Causal Lattice Sweep，层为 `12,16,20,22,24,26,27`，位置组为 `visual_span`、`answer_adjacent`、`top_hidden_delta`、`visual+answer`。每个条件同时测试 restore 与 corrupt，并比较 real mask vs shifted/shuffled、correct target vs wrong target。

只有 hidden lattice 通过或接近通过的层，才进入 PLT layer-matched sweep、多层 route patch 和 Adapter V4。这样避免再盲目扩大 layer 26 的 PLT 搜索。

## 结果

待运行。

## 预期与实际偏差

如果 hidden 通过但 PLT 失败，说明更像 `plt_localization_failure`。如果单层 hidden/PLT 弱但多层 patch 通过，说明更像 `distributed_route`。如果所有 broad tests 都失败，才写当前 Qwen-native tests 下不支持 Gemma-style sparse PLT route。

## 结论

待运行。禁止写 `Qwen has no cross-modal mechanism`。
