# 实验 002：Stage 2A-1 top24 pretrace 远端启动与输入修复

## 目的

本实验是 Stage 2A targeted replication pack 的第一个远端执行步骤。

它的目标不是直接证明最终机制 claim，而是先把新选出的 top24 replication candidates 跑过同一套 `B_direct` / `D_visual_only` 行为评估和 answer-aligned attribution tracing，筛出真正可进入 Stage 2A 复现实验的样本。

更具体地说，这一步要回答：

```text
1. top24 replication candidates 在 B_direct / D_visual_only 下是否能稳定生成可解析答案；
2. answer-aligned trace 是否能为这些样本产出 .pt graph；
3. 其中有多少样本后续可进入 support/suppressor source zeroing、nearest control 和 region-mask replication；
4. 是否需要再补一个 symbol_text_reading-heavy queue。
```

## 输入

本地输入目录：

```text
E:\Bridging\doc\experiments\stage2\stage2a_candidate_selection
```

本轮上传到服务器的文件：

```text
manifest_B_direct_stage2a_pretrace_top24.csv
manifest_D_visual_only_stage2a_pretrace_top24.csv
stage2a_trace_selected_ids_top24.csv
```

远端目标目录：

```text
/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm/doc/experiments/stage2/stage2a_candidate_selection
```

远端运行工作目录：

```text
/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm/research/work/stage2a_pretrace_top24
```

## 方法

远端执行分四段：

```text
1. B_direct eval
2. D_visual_only eval
3. B_direct answer-aligned trace
4. D_visual_only answer-aligned trace
```

eval 使用：

```text
scripts/research/run_batch_eval.py
model: google/gemma-3-4b-it local snapshot
correct_rule: vqa_0.3
max_new_tokens: 16
```

trace 使用：

```text
scripts/research/run_batch_answer_aligned_attribute.py
transcoder_set: tianhux2/gemma3-4b-it-plt
dtype: bfloat16
max_feature_nodes: 64
topk: 16
offload: disk
exec_mode: subprocess
retry_feature_nodes: 48,32
answer_source: predicted
```

## 首次启动问题

首次启动后，远端日志显示：

```text
eval_B_direct: manifest_rows=24, invalid_skip=24
eval_D_visual_only: manifest_rows=24, invalid_skip=24
```

随后 trace 阶段报错：

```text
FileNotFoundError: eval_B_direct.csv not found
```

原因不是模型生成失败，也不是 trace 失败，而是本地 CSV 文件头带有 UTF-8 BOM，并且 BOM 出现在第一列引号之前。远端 `csv.DictReader` 将第一列解析为：

```text
\ufeff"sample_id"
```

而不是：

```text
sample_id
```

因此 `run_batch_eval.py` 读不到 `sample_id`，全部 24 行被判定为 invalid row。

## 修复

已对三份 Stage 2A 输入 CSV 移除 UTF-8 BOM：

```text
manifest_B_direct_stage2a_pretrace_top24.csv
manifest_D_visual_only_stage2a_pretrace_top24.csv
stage2a_trace_selected_ids_top24.csv
```

修复后本地验证：

```text
fieldnames: sample_id, question, image_path, gold_answer, ...
first sample_id: okvqa_val_667695
```

修复后的文件已重新上传服务器，并重新启动远端任务。

## 远端重启状态

重启后的远端 run root：

```text
/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm/outputs/phase_ab/ab_answer_aligned/stage2a_pretrace_top24_20260519_202951
```

重启后 eval 结果正常：

```text
B_direct eval:
  manifest_rows = 24
  new = 24
  invalid_skip = 0
  empty_gen = 0

D_visual_only eval:
  manifest_rows = 24
  new = 24
  invalid_skip = 0
  empty_gen = 0
```

当前远端进度：

```text
B_direct answer-aligned trace 已开始
当前正在处理 okvqa_val_667695
```

## 输出预期

如果远端任务完整完成，应回传：

```text
eval_B_direct.csv
eval_D_visual_only.csv
answer_aligned_meta_b.csv
answer_aligned_meta_a.csv
pt_slotB_B_direct/*.pt
pt_slotA_D_visual_only/*.pt
logs/*.log
logs_b/*
logs_a/*
```

建议本地落点：

```text
E:\Bridging\remote_sync\2026-05-19_stage2a_pretrace_top24
```

## 预期与实际偏差

预期：

```text
top24 manifest 可以直接进入 eval。
```

实际：

```text
CSV BOM 导致首次 eval 全部 invalid。
```

偏差解释：

```text
这是数据文件编码 / schema 兼容问题，不是模型行为、trace 或机制结果问题。
```

修复后：

```text
B/D eval 均 24/24 成功，说明输入 schema 已恢复正常。
```

## 当前结论

Stage 2A-1 已经从“输入文件修复”阶段进入正式 trace 阶段。

目前可以确认：

```text
1. top24 manifest 内容本身是可用的；
2. B_direct / D_visual_only eval 能在服务器上跑通；
3. 远端已开始产出 answer-aligned trace；
4. 下一步需要等待 B/D trace 完成后同步结果，并构建 clean-core / support-source / nearest-control 筛选表。
```

## 对主 claim 的影响

本步骤不新增机制证据，但它是 targeted replication pack 的必要前置条件。

如果 top24 trace yield 足够，Stage 2A 可以继续推进到：

```text
source zeroing intervention screen
nearest node control screen
region-mask replication sample selection
```

如果 trace yield 不足，则按 run plan 进入后备路线：

```text
补一个 symbol_text_reading-heavy pretrace queue
```

## 下一步

按顺序继续：

```text
1. 轮询远端 B_direct trace 进度；
2. 轮询远端 D_visual_only trace 进度；
3. 完成后同步 run root 到 remote_sync；
4. 生成 003_stage2a_pretrace_readout.md；
5. 构建 Stage 2A region replication candidates。
```
