# Stage4-034 Qwen Adapter V4 Layer-14 Run Plan

## 目的

本轮验证 Qwen 已确认的 hidden layer 14 evidence-to-answer route，是否能被 Qwen2.5-VL-PLT 的 feature/source route 自动捕获。

本实验不复用 Gemma 节点地图，只复用 Gemma 主线判据：source/control、evidence mask specificity、wrong-target specificity、rank/sequence behavior bridge。

## 输入

- base: `Qwen/Qwen2.5-VL-7B-Instruct`
- PLT: `KokosDev/qwen2p5vl-7b-plt`
- paperpack72 primary/strict prompt-runs
- confirmed hidden gate: layer `14`, `restore`, `top_hidden_delta`, `answer_mask`

## 输出

- `stage4_qwen_source_tracing_*_adapter_v4_L14_*`
- `stage4_qwen_decisive_route_plt_*_L14_*`
- `stage4_qwen_adapter_v4_*_L14_layer14_*`
- `stage4_qwen_adapter_v4_summary.csv`
- `stage4_qwen_adapter_v4_specificity.csv`
- `stage4_qwen_adapter_v4_decision.json`

## 方法

执行顺序：

1. L14 source-tracing smoke: primary 6 prompt-runs.
2. L14 PLT evidence/zeroing smoke: primary 6 prompt-runs.
3. Adapter V4 manifest smoke + intervention smoke.
4. L14 source-tracing primary full.
5. L14 PLT evidence/zeroing primary full.
6. Adapter V4 primary full validation.
7. Adapter V4 strict full validation.
8. 若 L14 V4 失败但候选充足，再补 `13/15` sensitivity。

所有长任务使用 detached/nohup，避免 SSH 断开导致远端任务中断。

## 结果

待运行。

## 预期与实际偏差

如果 L14 source tracing 或 PLT evidence candidates 为空，结论写 `qwen_hidden_route_supported_plt_unresolved`，不能写机制负结论。

如果 V4 跑通但 primary/strict 都失败，结论限定为 `qwen_not_gemma_style_under_v4_adapter`。

## 结论

待运行。
