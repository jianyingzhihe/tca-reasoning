# 实验 006：Stage 2A-3 nearest matched-control clean-screen 读数

## 目的

本实验是 Stage 2A targeted replication 的第三步。

在实验 005 中，我们已经对 Stage 2A top24 pretrace pool 中的 traced source feature nodes 做了 clean-image zeroing，并得到：

```text
source zeroing successful rows = 87
support rows = 46
suppressor rows = 35
neutral rows = 6
existing mask + support samples = 17
clean_prompt + existing mask + support samples = 6
```

本实验继续为这些 non-neutral source rows 寻找 nearest matched non-source controls，并在 clean condition 下做同样的 zeroing。

核心目的有三个：

1. 判断 strict nearest-control 在 Stage 2A top24 pool 中的可用覆盖率。
2. 检查 source node 的 clean causal effect 是否强于 nearest non-source control。
3. 决定下一步 region-mask replication 应该使用哪一个分析池。

## 术语解释

`nearest matched-control`：节点对照。它不是 tracing 找到的 source node，但在同层、同位置或相近匹配标签下与 source node 尽量相似，用来排除“随便找一个附近 feature 也有同样效果”的解释。

`source effect`：按 node role 转成正向解释后的 effect。对 support node，source effect = `-source_delta_target_logit`；对 suppressor node，source effect = `source_delta_target_logit`。

`source-control gap`：`source_effect - control_effect`。正值表示 traced source node 比 nearest non-source control 更强。

`prompt-specific route`：每个 prompt 使用自己 trace 出来的 answer-adjacent route。它不要求 `B_direct` 和 `D_visual_only` 的 target token 完全相同，因此更符合当前主 claim：prompt 是路径调节/暴露变量，而不是行为优劣比较变量。

## 输入

远端 run root：

```text
/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm/outputs/phase_ab/ab_answer_aligned/stage2a_pretrace_top24_20260519_202951
```

nearest-control 输入：

```text
/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm/outputs/phase_ab/ab_answer_aligned/stage2a_pretrace_top24_20260519_202951/intervention/stage2a_source_zeroing_top4_pilot_clean_nonzero.csv
```

该输入由实验 005 的 source zeroing 输出过滤得到，只保留：

```text
condition = clean
node_role in {support, suppressor}
delta_target_logit != 0
```

本地同步输入与派生文件：

```text
E:\Bridging\remote_sync\2026-05-19_stage2a_pretrace_top24\stage2a_source_zeroing_top4.csv
E:\Bridging\remote_sync\2026-05-19_stage2a_pretrace_top24\stage2a_source_zeroing_top4_enriched.csv
E:\Bridging\remote_sync\2026-05-19_stage2a_pretrace_top24\stage2a_source_zeroing_top4_sample_summary.csv
```

## 输出

远端输出：

```text
/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm/outputs/phase_ab/ab_answer_aligned/stage2a_pretrace_top24_20260519_202951/intervention/stage2a_nearest_control_clean_top4_nonzero.csv
```

本地同步输出：

```text
E:\Bridging\remote_sync\2026-05-19_stage2a_pretrace_top24\stage2a_nearest_control_clean_top4_nonzero.csv
E:\Bridging\remote_sync\2026-05-19_stage2a_pretrace_top24\run_stage2a_nearest_control_clean.nohup.log
```

本地派生分析文件：

```text
E:\Bridging\remote_sync\2026-05-19_stage2a_pretrace_top24\stage2a_nearest_control_clean_top4_nonzero_enriched.csv
E:\Bridging\remote_sync\2026-05-19_stage2a_pretrace_top24\stage2a_nearest_control_clean_top4_nonzero_per_source_summary.csv
```

## 方法

远端脚本：

```text
scripts/research/run_modality_counterfactual_matched_control.py
```

关键参数：

```text
--run-tag-base stage2a_pretrace_top24_20260519_202951
--pilot-csv stage2a_source_zeroing_top4_pilot_clean_nonzero.csv
--match-mode nearest
--position-alignment strict
--out-csv stage2a_nearest_control_clean_top4_nonzero.csv
```

这一步只跑 `clean`，不跑 `wrong_image / masked_image / no_image`，因为目标不是重新做 modality counterfactual，而是为下一步 evidence-region mask 复现实验准备 node-control 清单。

## 运行状态

远端任务已完成。

```text
remote output lines = 28
data rows = 27
process status = finished
```

日志最后包含：

```text
[done] out_csv=.../stage2a_nearest_control_clean_top4_nonzero.csv
[done] nearest_control_clean 2026-05-19 23:47:55
```

## 主要结果

### 1. 覆盖率

从实验 005 的 `81` 条 non-neutral source rows 中，strict nearest-control 成功匹配并干预了 `27` 条。

```text
source non-neutral rows = 81
nearest matched rows = 27
matched samples = 10
```

按 role 统计：

| role | nearest rows | sample count |
|---|---:|---:|
| support | 21 | 10 |
| suppressor | 6 | 2 |

按已有区域标注过滤：

| 条件 | support sample count | suppressor sample count |
|---|---:|---:|
| existing answer/relate mask | 8 | 1 |
| clean_prompt + existing mask | 1 | 1 |

关键结论：

```text
existing mask + support + nearest-control samples = 8
clean_prompt + existing mask + support + nearest-control samples = 1
```

这说明 Stage 2A 可以继续做 prompt-specific route 的 strict region-mask replication，但不能把严格 B/D target-aligned prompt 对照作为主分析。

### 2. support source-control gap

按 per-source row 口径：

| pool | n rows | n samples | mean gap | median gap | positive rate | bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---|
| support all | 21 | 10 | +0.583 | +0.500 | 0.810 | [+0.262, +0.923] |
| support existing-mask | 16 | 8 | +0.516 | +0.438 | 0.812 | [+0.121, +0.941] |

按 sample mean 口径：

| pool | n samples | mean gap | median gap | positive sample rate | bootstrap 95% CI |
|---|---:|---:|---:|---:|---|
| support all | 10 | +0.431 | +0.594 | 0.800 | [+0.008, +0.827] |
| support existing-mask | 8 | +0.344 | +0.438 | 0.750 | [-0.149, +0.828] |

解释：

```text
per-source 口径下，support source > nearest control 是稳定正向的。
sample 聚合后，all-support pool 仍然略为正且 CI 刚好不跨 0。
但是 existing-mask support pool 只有 8 个样本，sample-level CI 跨 0。
```

因此，这一步支持继续推进 region-mask replication，但不能单独升级为“强统计复现已经完成”。

### 3. suppressor source-control gap

按 per-source row 口径：

| pool | n rows | n samples | mean gap | median gap | positive rate | bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---|
| suppressor all | 6 | 2 | +1.104 | +1.125 | 1.000 | [+0.875, +1.354] |
| suppressor existing-mask | 4 | 1 | +1.156 | +1.125 | 1.000 | [+0.875, +1.469] |

解释：

```text
suppressor source-control gap 方向很强，但 sample 数太少。
它只能作为 secondary / case-level signal，不能作为 Stage 2A 主成功标准。
```

## 可进入下一步 region-mask replication 的样本

`existing mask + support + nearest-control` 的 8 个样本是：

```text
okvqa_val_1740705
okvqa_val_1927165
okvqa_val_2131565
okvqa_val_2683965
okvqa_val_343215
okvqa_val_3794755
okvqa_val_5735275
okvqa_val_80655
```

每个样本的 support nearest row 数：

| sample_id | support nearest rows |
|---|---:|
| okvqa_val_1740705 | 2 |
| okvqa_val_1927165 | 3 |
| okvqa_val_2131565 | 2 |
| okvqa_val_2683965 | 3 |
| okvqa_val_343215 | 1 |
| okvqa_val_3794755 | 2 |
| okvqa_val_5735275 | 1 |
| okvqa_val_80655 | 2 |

其中 `okvqa_val_3794755` 还提供了 suppressor nearest-control rows，可作为 secondary suppressor case。

## 预期与实际偏差

预期：

```text
nearest-control 至少覆盖 8 个 support + mask 样本；
如果覆盖明显低于 8，则需要放宽 matching 或转向 random node controls。
```

实际：

```text
existing mask + support + nearest-control = 8 samples
```

这正好达到最低可推进门槛，但没有余量。

偏差：

```text
strict nearest-control row-level yield 只有 27/81。
很多 source rows 日志显示 no control candidate found。
这意味着当前 strict matching 过保守，但它也让 surviving controls 更干净。
```

## 结论

本实验给出两个结论。

第一，Stage 2A top24 pool 中，traced support source nodes 在 clean zeroing 下整体强于 nearest non-source controls。这个结果在 per-source 口径下较稳，在 sample-level 口径下仍有异质性。

第二，下一步 region-mask replication 应该固定为：

```text
primary pool = existing mask + support + nearest-control samples
sample count = 8
analysis framing = prompt-specific route replication
primary role = support
secondary role = suppressor
```

不能使用：

```text
strict B/D target-aligned clean_prompt comparison
```

原因是该池只剩 `1` 个 support + nearest + existing-mask 样本。

## 对主 claim 的影响

支持主 claim 的部分：

```text
source support nodes > nearest controls
```

这继续支撑“traced source nodes 不是任意附近 feature”的 specificity claim。

需要保守的部分：

```text
existing-mask support pool 的 sample-level CI 跨 0。
```

因此，nearest-control clean-screen 本身不是最终主证据，只是为下一步 evidence-region mask 复现实验准备干净样本池。

## 后续动作

下一步执行 Stage 2A-4：

```text
构建 8-sample nearest-supported region-mask replication pack
导出 answer / relate / union masks
生成 source + nearest-control manifest
在服务器跑 clean / answer_mask / relate_mask / union_mask / random_control_1..4
主读数：support answer/union weakening > random4，且 source weakening > nearest-control weakening
```

如果 Stage 2A-4 成功：

```text
它将成为 core24 之后的 targeted replication evidence。
```

如果 Stage 2A-4 失败：

```text
说明 core24 的 evidence-region sensitivity 可能依赖特定样本或特定节点筛选；
主 claim 需要降级为 focused-pilot，而不是 independent replication。
```
