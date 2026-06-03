# Stage4-003 Qwen Source-Tracing Smoke

## 目的

先用 3 个 paperpack cases × 2 prompts 验证 Qwen adapter 的输入、target 对齐、graph schema、A/B compare 和 intervention smoke 全链路可运行。

## 输入

- `paperpack72_primary_manifest.csv`
- `paperpack72_primary_prompt_runs.csv`
- Qwen2.5-VL base + Qwen2.5-VL-PLT

## 输出

Smoke 输出放在 `doc/experiments/stage4/cross_model/`，文件名包含 `stage4_qwen_source_tracing_primary_smoke_*`。

## 方法

运行 `run_stage4_qwen_source_tracing_remote.py --pack primary --mode smoke`。

## 结果

已完成 primary smoke v1/v2。v2 修正了 smoke 行数和 support-only node selection 后，工程链路通过：

- A=`D_visual_only` graph: 3/3。
- B=`B_direct` graph: 3/3。
- valid matched samples: 3。
- compare rows: 3。
- node rows: 30，其中 feature node rows: 18。
- edge rows: 36。
- intervention rows: 12。

但科学方向未通过。加入 `subtract/add` 两种 zeroing convention 后仍然没有 target 损伤：

- subtract: 12 rows, mean `delta_target_logit = +0.0625`, negative fraction `0/12`。
- add: 12 rows, mean `delta_target_logit = +0.0521`, negative fraction `0/12`。

也就是说，失败不是简单 decoder sign / hook sign 问题；当前 direct-effect Qwen graph 的 top feature nodes 被 intervention 后没有损伤 target，不能作为 Qwen full source-tracing replication 的 smoke success。

## 预期与实际偏差

实际偏差是 graph schema / compare / intervention 工程都通过，但 direct-effect graph node 与 causal zeroing 方向不一致。下一步不能直接跑 primary72 full；应先做 adapter 诊断：

- decoder sign / hook sign 已做初步检查，`subtract/add` 都未损伤 target。
- 下一步应比较 direct-effect node、zeroing-screened node、Stage3 approximate source-control positive node 三种候选。
- 若 direct-effect adapter 仍失败，但 zeroing-screened route 成立，只能写 Qwen approximate/causal-screened route support，不能写严格 Gemma-style source tracing replication。

## 结论

当前结论：`qwen_adapter_schema_smoke_passed_but_direction_failed`。这不是 Qwen 机制负结论，但说明第一版 Qwen source-tracing adapter 还不能进入 paperpack72 full 主实验。
