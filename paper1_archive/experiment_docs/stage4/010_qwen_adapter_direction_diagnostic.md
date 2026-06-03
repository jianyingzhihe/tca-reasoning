# Stage4-010 Qwen Adapter Direction Diagnostic

## 目的

Stage4 Qwen source-tracing smoke 的 graph schema、A/B compare 和 intervention 工程已通过，但 direct-effect graph node zeroing 没有损伤 target。因此需要先诊断 Qwen adapter 的 source node 定义，再决定是否进入 primary72 full。

## 输入

- Stage4 primary smoke graph / nodes / intervention。
- Stage3 Qwen2.5-VL-PLT paperpack source-control positive rows。
- Qwen2.5-VL-PLT layer 26。

## 方法

比较三种候选 source node：

- `direct_effect_graph_node`：当前 adapter 使用的 `activation * decoder dot target_direction` top nodes。
- `zeroing_screened_node`：在 direct-effect candidate pool 内先做轻量 zeroing screen，只保留 zeroing 会损伤 target 的 nodes。
- `stage3_source_control_node`：Stage3 approximate source-control probe 中已经通过 source > matched-control 的 feature/position pairs。

每种候选都需要输出 graph-compatible rows，并用同一 intervention smoke 检查 target logit/rank。

## 判据

- 若 `direct_effect_graph_node` 继续失败，而 `zeroing_screened_node` 成功：只能写 `Qwen causal-screened route support`，不能写严格 Gemma-style source tracing replication。
- 若 `stage3_source_control_node` 成功但 graph direct node 失败：说明 Stage3 approximate route 是真实辅助证据，但 full source-tracing adapter 未闭合。
- 只有 direct graph node 在 smoke 和 full 中自然通过 intervention + controls，才进入 `qwen_full_source_tracing_supported`。

## 当前结果

已完成 sign diagnostic：`subtract` 与 `add` zeroing mode 都没有让 current direct-effect nodes 损伤 target。因此 primary72 full 暂不启动，先做 node-candidate diagnostic。

Update: primary72 full 与 strict72 full 已实际运行。两者均满足 graph success 门槛，但 intervention 方向不成立：

- primary best-mode negative `delta_target_logit` fraction: `12.14%`。
- strict best-mode negative `delta_target_logit` fraction: `11.07%`。

这说明失败不是 3-case smoke 偶然，也不是单纯 position compaction 或 sign convention 问题。

## 结论

当前状态是 `qwen_source_tracing_not_supported_under_current_adapter`。这不是 Qwen 机制负结论，也不是 full replication；它说明当前 direct answer-aligned PLT graph adapter 不能把 Stage3 的 approximate source/control evidence 升级为 Gemma-style source tracing replication。
