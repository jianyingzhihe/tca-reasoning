# 实验 007：Stage 2A-4 region-mask replication pack 构建与远端启动

## 目的

本实验是 Stage 2A targeted replication 的第四步。

在实验 006 中，我们已经确认：

```text
existing mask + support + nearest-control samples = 8
support nearest-control pairs = 16
suppressor nearest-control pairs = 4
```

因此本实验把这些样本整理成独立 region-mask replication pack，并在服务器上启动 evidence-region mask counterfactual。

核心问题：

```text
在一批新的 prompt-specific route 样本中，
遮挡 answer / union evidence region 是否会削弱 support source route，
并且这种削弱是否强于 nearest node control 和 random region control？
```

## 输入

nearest-control clean-screen 输出：

```text
E:\Bridging\remote_sync\2026-05-19_stage2a_pretrace_top24\stage2a_nearest_control_clean_top4_nonzero_enriched.csv
```

answer-aligned metadata：

```text
E:\Bridging\remote_sync\2026-05-19_stage2a_pretrace_top24\answer_aligned_meta_a.csv
E:\Bridging\remote_sync\2026-05-19_stage2a_pretrace_top24\answer_aligned_meta_b.csv
```

复用的人工标注资产：

```text
E:\Bridging\annotation\okvqa_evidence_labelme_round4_core16_extra
E:\Bridging\annotation\okvqa_evidence_labelme_round4_ultraeasy16_fresh
E:\Bridging\annotation\okvqa_evidence_labelme_round4_core24_easy
```

## 输出

本地 pack：

```text
E:\Bridging\annotation\stage2a_region_replication_top24_nearest8
```

关键文件：

```text
annotation/stage2a_region_replication_top24_nearest8/region_experiment_manifest.csv
annotation/stage2a_region_replication_top24_nearest8/region_experiment_manifest_remote.csv
annotation/stage2a_region_replication_top24_nearest8/selection_summary.md
annotation/stage2a_region_replication_top24_nearest8/exported_masks/mask_export_summary.csv
```

远端 pack：

```text
/root/autodl-tmp/tca-reasoning/annotation/stage2a_region_replication_top24_nearest8
```

远端运行脚本：

```text
/root/autodl-tmp/tca-reasoning/annotation/stage2a_region_replication_top24_nearest8/run_stage2a_region_random4_remote.sh
```

远端日志：

```text
/root/autodl-tmp/tca-reasoning/annotation/stage2a_region_replication_top24_nearest8/run_stage2a_region_random4_remote.log
```

远端预期结果：

```text
/root/autodl-tmp/tca-reasoning/annotation/stage2a_region_replication_top24_nearest8/region_mask_stage2a_nearest8_random4.csv
```

## 方法

### 1. 构建 pack

新增本地脚本：

```text
E:\Bridging\scripts\local\build_stage2a_region_replication_pack.py
```

该脚本做四件事：

1. 从 nearest enriched CSV 中筛选 `existing_answer_relate_mask=True` 且 role 为 `support/suppressor` 的 rows。
2. 为每条 source-control pair 同时生成 `source` 和 `nearest_control` 两条 manifest row。
3. 从旧 annotation pack 中复制对应图片和 LabelMe JSON。
4. 同时写本地 manifest 和远端 manifest。

### 2. 导出 mask

使用：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/export_labelme_regions_to_masks.py
```

导出：

```text
answer.png
relate.png
```

如果同一张图中有多个 `answer` shapes，导出脚本会把它们绘制到同一张 `answer.png` 中，因此多个 answer 区域会被合并覆盖。

### 3. 远端 region-mask counterfactual

使用：

```text
scripts/research/run_region_mask_counterfactual_mainline.py
```

固定 conditions：

```text
clean
answer_mask
relate_mask
union_mask
random_control_1
random_control_2
random_control_3
random_control_4
```

随机区域控制：

```text
avoid_overlap_iou = 0.05
random_seeds = 101,202,303,404
```

## Pack 构建结果

```text
selected source-control pairs = 20
manifest rows = 40
sample count = 8
support sample count = 8
support pairs = 16
suppressor pairs = 4
```

support pairs by sample：

| sample_id | support pairs | annotation pack |
|---|---:|---|
| okvqa_val_1740705 | 2 | okvqa_evidence_labelme_round4_ultraeasy16_fresh |
| okvqa_val_1927165 | 3 | okvqa_evidence_labelme_round4_core16_extra |
| okvqa_val_2131565 | 2 | okvqa_evidence_labelme_round4_ultraeasy16_fresh |
| okvqa_val_2683965 | 3 | okvqa_evidence_labelme_round4_core16_extra |
| okvqa_val_343215 | 1 | okvqa_evidence_labelme_round4_core16_extra |
| okvqa_val_3794755 | 2 | okvqa_evidence_labelme_round4_core16_extra |
| okvqa_val_5735275 | 1 | okvqa_evidence_labelme_round4_core16_extra |
| okvqa_val_80655 | 2 | okvqa_evidence_labelme_round4_ultraeasy16_fresh |

mask export summary：

| image | labels | answer area px | relate area px |
|---|---|---:|---:|
| COCO_val2014_000000008065.jpg | answer,relate | 51660 | 25270 |
| COCO_val2014_000000034321.jpg | answer,relate | 42525 | 89100 |
| COCO_val2014_000000174070.jpg | answer,relate | 189318 | 39600 |
| COCO_val2014_000000192716.jpg | answer,relate | 115910 | 33573 |
| COCO_val2014_000000213156.jpg | answer,relate | 209192 | 130962 |
| COCO_val2014_000000268396.jpg | answer,relate | 10197 | 17655 |
| COCO_val2014_000000379475.jpg | answer,relate | 79200 | 92660 |
| COCO_val2014_000000573527.jpg | answer,relate | 12489 | 20670 |

注意：

```text
部分样本的 answer area 很大，说明它们虽然是 localized 标注，
但证据区域接近占据主体或大块区域。
后续分析中不能把它们解释成“小物体级局部 evidence”。
```

## 远端启动状态

启动时间：

```text
2026-05-20 00:01 Asia/Shanghai
```

远端 PID：

```text
838006
```

启动后初查：

```text
process status = running
model load = success
output csv = not yet created at first check
```

日志显示：

```text
[load_env] Loaded env from .../.env
[dev] Ready.
[info] loading model=google/gemma-3-4b-it transcoder_set=tianhux2/gemma3-4b-it-plt dtype=torch.bfloat16
Loaded pretrained model google/gemma-3-4b-it into HookedVLTransformer
```

这说明远端环境、模型加载和 pack 路径目前没有立刻失败。

## 预期结果

主成功标准：

```text
support source answer_mask weakening > random4_mean
support source union_mask weakening > random4_mean
support source weakening > nearest_control weakening
```

次要读数：

```text
relate_mask 位于 answer/union 与 random4 之间，若不稳定只做描述性结果。
suppressor 只作为 secondary / heterogeneous signal。
```

失败标准：

```text
answer/union mask 与 random4 没有方向差异；
source 与 nearest-control 的 evidence sensitivity 没有方向差异；
大量 position_out_of_range 或 mask 缺失导致 usable rows < 8 samples。
```

## 对主 claim 的影响

如果本实验复现 core24 的方向，则 Stage 2A 将从“原 core24 focused result”升级为“targeted replication evidence”。

如果本实验方向不稳，则主 claim 仍然可以保留为 core24 focused mechanism result，但不能声称 independent replication 已经成立。

## 后续动作

1. 继续监控远端 `region_mask_stage2a_nearest8_random4.csv`。
2. 完成后同步 CSV 和 log。
3. 用 `summarize_prefixfix_region_evidence.py` 或等价 Stage 2A summarizer 生成：
   - support source answer/union weakening
   - answer/union minus random4
   - source minus nearest-control
   - sample-level bootstrap CI
4. 把分析结果写入实验 008。
