# Stage4-038 Qwen Native Route Validation Run Plan

## 目的

验证 Qwen2.5-VL 的 evidence-to-answer route 是否能用 Qwen 自己的表示形态定位，而不是强行要求它复现 Gemma-style sparse PLT source tracing。

当前已知：

- hidden layer 14 route 在 primary/strict 上成立。
- Adapter V4 找到的 L14 PLT 节点 clean zeroing 会伤 target。
- Adapter V4 没有通过 evidence-mask restore、correct-vs-wrong、rank restore gates。

## 输入

- `paperpack72_primary_prompt_runs.csv`
- `paperpack72_strict_sensitivity_prompt_runs.csv`
- `stage4_qwen_adapter_v4_*_L14_layer14_*`
- `stage4_qwen_decisive_route_hidden_*_alllayers_*`
- base: `Qwen/Qwen2.5-VL-7B-Instruct`
- PLT: `KokosDev/qwen2p5vl-7b-plt`

## 输出

- `cross_model/stage4_qwen_native_route_*`
- `039_qwen_v4_failure_decomposition.md`
- `040_qwen_intervention_operator_sweep.md`
- `041_qwen_exact_position_plt_route.md`
- `042_qwen_hidden_to_plt_mediation.md`
- `043_qwen_native_route_verdict.md`

## 方法

本轮分四步：

1. V4 failure decomposition：用已有 primary/strict V4 artifacts 拆出失败来自 activation、operator、position mismatch 还是 feature insufficiency。
2. Hidden-to-PLT mediation：同一 prompt-run 比较 full hidden residual restore、PLT topK reconstruction restore、PLT reconstruction error restore。
3. Exact-position PLT route：禁用 same-feature fallback，只测试 exact position-feature match。
4. Operator sweep：在接近通过的候选上比较 decoder-add、scaled-add、topK reconstruction、residual patch。

## 结果

待运行。

## 预期与实际偏差

如果 hidden residual patch 成立但 PLT topK/error 均不成立，写 `PLT localization failed under current public PLT`。

如果 PLT topK 或 multi-position patch 成立，写 `Qwen-native distributed PLT route support`，不写 Gemma-style sparse route。

## 结论

待运行。
