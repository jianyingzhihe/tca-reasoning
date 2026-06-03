# Stage4-024 Qwen Hidden Causal Lattice

## 目的

先不碰 PLT feature，直接测试 hidden residual 层面是否存在 evidence-to-answer causal action。这个实验是后续 PLT 层扫的前置筛选器。

## 输入

- Qwen2.5-VL-7B-Instruct
- paperpack72 primary / strict
- 层：`12,16,20,22,24,26,27`
- 位置组：`visual_span`、`answer_adjacent`、`top_hidden_delta`、`visual+answer`

## 输出

- `stage4_qwen_decisive_route_hidden_primary_smoke_raw.csv`
- `stage4_qwen_decisive_route_hidden_primary_full_raw.csv`
- strict 对应 artifact
- analyzer summary / specificity / decision JSON

## 方法

对每个 prompt-run 先得到 clean hidden 与 mask hidden。restore 在 masked run 中把指定层/位置的 hidden residual 朝 clean 方向补回；corrupt 在 clean run 中朝 masked 方向扰动。主判据是 target logit/rank effect，同时要求 real mask 强于 shifted/shuffled，correct target 强于 wrong target。

## 结果

待运行。

## 预期与实际偏差

如果 hidden lattice 都不通过，后续 PLT feature route 很难成立；如果 hidden lattice 通过但 PLT route 失败，说明问题更可能出在 PLT feature localization 或 adapter scoring。

## 结论

待运行。
