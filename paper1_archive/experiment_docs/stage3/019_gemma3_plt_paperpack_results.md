# 019 Gemma3-PLT Paperpack Results

## 目的

在 `paperpack72_primary` / `paperpack72_strict_sensitivity` 上推进 Gemma3-PLT 主线，作为 PLT-only verdict 的 Gemma 侧基准。

Gemma 侧与 Qwen2.5-VL 侧的关键差异是：Gemma 已有 ReplacementModel / source-tracing pipeline，因此论文级 Gemma 结果应尽量包含 source tracing、controlled compare、node intervention、nearest/random controls、region-mask sensitivity 与 behavior bridge；Qwen 侧目前仍是 approximate source-control support，不能写成完整 Gemma-style source tracing 复现。

## 当前状态

```text
annotation_ready
paperpack72_primary_ready
paperpack72_strict_sensitivity_ready
historical_overlap_limited_only
source_tracing_smoke_passed
primary72_full_source_tracing_passed
strict72_graph_compare_passed_intervention_blocked
plt_only_verdict_ready_with_boundary
```

## 输入

```text
primary manifest:
  doc/experiments/stage3/paperpack72/paperpack72_primary_manifest.csv

strict manifest:
  doc/experiments/stage3/paperpack72/paperpack72_strict_sensitivity_manifest.csv

Gemma eval manifests:
  doc/experiments/stage3/cross_model/stage3_gemma_eval_primary_B_direct.csv
  doc/experiments/stage3/cross_model/stage3_gemma_eval_primary_D_visual_only.csv
  doc/experiments/stage3/cross_model/stage3_gemma_eval_strict_B_direct.csv
  doc/experiments/stage3/cross_model/stage3_gemma_eval_strict_D_visual_only.csv

Gemma3-PLT:
  tianhux2/gemma3-4b-it-plt

base:
  google/gemma-3-4b-it
```

## 已完成

### Manifest

Gemma eval manifest 已生成并检查：

```text
primary B_direct: 72 rows, missing_images=0, missing_gold_answers=0
primary D_visual_only: 72 rows, missing_images=0, missing_gold_answers=0
strict B_direct: 72 rows, missing_images=0, missing_gold_answers=0
strict D_visual_only: 72 rows, missing_images=0, missing_gold_answers=0
```

### Overlap Report

历史 overlap audit 结果：

```text
status: limited_calibration_only
historical_available_samples: 4
historical_available_primary_prompt_runs: 8
```

因此历史 Gemma rows 只能作为 limited calibration，不能替代 paperpack Gemma rerun。详见：

```text
doc/experiments/stage3/031_stage3_gemma_paperpack_overlap_report.md
```

### Source-Tracing Smoke

Gemma paperpack source-tracing smoke 已通过：

```text
status: pass
valid_smoke_sample_rows: 2
sample_compare_rows: 2
nodes_detailed_rows: 113
intervention_rows: 4
```

详见：

```text
doc/experiments/stage3/032_stage3_gemma_source_tracing_smoke.md
```

### Primary72 Full Source Tracing

Gemma3-PLT `paperpack72_primary` full source tracing 已完成：

```text
remote decision: pass
local analyzer status: primary_full_passed
eval A rows: 72
eval B rows: 72
valid samples: 71 / 72
graph A files: 71
graph B files: 71
graph success rate vs valid: 1.0
sample compare rows: 71
nodes detailed rows: 4126
edges detailed rows: 7071
intervention rows: 127
```

唯一 filtered sample 是 `okvqa_val_01162`：`D_visual_only` 空生成，`B_direct` 正常生成 `The answer is Tony Hawk.`。这是 eval/template 或行为诊断，不写作机制负结果。

详见：

```text
doc/experiments/stage3/033_stage3_gemma_primary72_source_tracing_full.md
```

### Strict72 Sensitivity Source Tracing

Gemma3-PLT `paperpack72_strict_sensitivity` full source-tracing sensitivity 已完成到 graph/compare 级别：

```text
remote decision: blocked
local analyzer status: strict_full_graph_compare_passed_intervention_blocked
eval A rows: 72
eval B rows: 72
valid samples: 71 / 72
graph A files: 71
graph B files: 71
graph success rate vs valid: 1.0
sample compare rows: 71
nodes detailed rows: 4095
edges detailed rows: 7003
intervention rows: 0
```

唯一 filtered sample 仍是 `okvqa_val_01162`：`D_visual_only` 空生成，`B_direct` 正常生成 `The answer is Tony Hawk.`。这与 primary72 一致，属于 eval/template 或行为诊断，不写作机制负结果。

strict 续跑中曾出现 `/root/autodl-tmp` 满盘，已通过清理远端 primary graph 中间目录后 `--resume` 补齐。随后 strict graph/compare 达到 `71/71`；但 strict intervention smoke 在当前远端状态下于 ReplacementModel / HookedVLTransformer 构造阶段被 `SIGKILL(137)` 打断。单独 load probe 复现同一现象，因此记录为工程/资源 blocked，而不是机制负结果。

详见：

```text
doc/experiments/stage3/034_stage3_gemma_strict72_source_tracing_sensitivity.md
```

## 预期与实际偏差

此前预期可以用历史 overlap 快速校准 Gemma paperpack；实际 overlap 只有 4 个样本 / 8 个 primary prompt-runs，因此必须新跑 Gemma source-tracing rerun。

Primary full 已通过，但 intervention smoke 仍暴露两个工程限制：部分 A-side metadata 缺 `target_token_id`，以及若干 selected features 位于末端 buffer 外而被 `position_buffer_exceeded` 跳过。这些不影响 graph/compare 验收，但如果后续要把 Gemma paperpack intervention 升级成论文主 endpoint，需要单独修复。

Strict sensitivity 进一步确认 graph/compare pipeline 不依赖 manual-review 样本；但 strict intervention smoke 当前被远端模型加载 `SIGKILL(137)` 阻塞。因此 strict 最强证据层级是 source-tracing graph/compare sensitivity，而不是 strict intervention replication。

## 结论

Gemma3-PLT paperpack 现在已经不再停留在 smoke：`paperpack72_primary` full source tracing 已通过，`paperpack72_strict_sensitivity` 也完成了 full graph/compare sensitivity。

当前可以进入 PLT-only verdict，但 verdict 必须区分：

```text
Gemma primary72: full source-tracing graph/compare + intervention smoke passed.
Gemma strict72: graph/compare sensitivity passed; intervention smoke engineering-blocked.
Qwen2.5-VL-PLT: approximate feature/source-control support, not Gemma-style source tracing.
```
