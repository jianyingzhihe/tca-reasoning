# 实验 008：Stage 2A-5 nearest8 region-route 读数

## 目的

本实验读取实验 007 启动的 Stage 2A nearest8 region-mask counterfactual 结果。

核心问题是：

```text
在新构建的 8-sample targeted replication pack 中，
answer / union evidence-region mask 是否会削弱 support source route？
这种削弱是否强于 nearest-control node？
这种削弱是否强于 random same-area region controls？
```

这是 Stage 2A 的关键复现实验。它不再问 `D_visual_only` 是否比 `B_direct` 更好，而是检验 prompt-specific answer route 是否具有 evidence-region sensitivity。

## 输入

远端结果：

```text
/root/autodl-tmp/tca-reasoning/annotation/stage2a_region_replication_top24_nearest8/region_mask_stage2a_nearest8_random4.csv
```

本地同步结果：

```text
E:\Bridging\annotation\stage2a_region_replication_top24_nearest8\region_mask_stage2a_nearest8_random4.csv
E:\Bridging\annotation\stage2a_region_replication_top24_nearest8\run_stage2a_region_random4_remote.log
```

manifest：

```text
E:\Bridging\annotation\stage2a_region_replication_top24_nearest8\region_experiment_manifest.csv
```

分析脚本：

```text
E:\Bridging\scripts\local\summarize_stage2a_region_route_replication.py
```

## 输出

分析目录：

```text
E:\Bridging\annotation\stage2a_region_replication_top24_nearest8\analysis_route
```

关键 artifact：

```text
clean_calibration_summary.csv
metric_summary_iou0p05.csv
random_coverage_iou0p05.csv
route_weakening_iou0p05.csv
support_source_nearest_pairs_iou0p05.csv
STAGE2A_REGION_ROUTE_READOUT_iou0p05.md
STAGE2A_REGION_ROUTE_READOUT_iou0p2.md
STAGE2A_REGION_ROUTE_READOUT_iou1.md
```

其中：

```text
iou0p05 = run plan 中预注册的 strict random-control 口径
iou0p2  = relaxed diagnostic 口径
iou1    = all-random diagnostic 口径，不作为 strongest evidence
```

## 方法

### 1. 运行完成性检查

远端 job 完成：

```text
expected manifest rows = 40
conditions per row = 8
expected done rows = 320
actual done rows = 320
skip rows = 0
```

这说明：

```text
没有 mask 缺失；
没有 position_out_of_range；
没有模型运行中断；
所有 source 和 nearest_control rows 都完成了 clean / answer / relate / union / random1..4。
```

### 2. 读数口径

对每个 node row，先把 clean 与 masked condition 的 `delta_target_logit` 展开成 wide table。

support weakening：

```text
support weakening = masked_delta - clean_delta
```

原因：

```text
support node 的 clean delta 通常为负；
如果遮挡证据区域后 delta 变得不那么负，说明该 support route 变弱；
因此 masked_delta - clean_delta 为正。
```

suppressor weakening：

```text
suppressor weakening = clean_delta - masked_delta
```

原因：

```text
suppressor node 的 clean delta 通常为正；
如果遮挡后 suppressor effect 变小，则 clean_delta - masked_delta 为正。
```

source-nearest gap：

```text
source_minus_nearest_weakening = source_weakening - nearest_control_weakening
```

正值表示：

```text
traced source route 对 evidence-region mask 更敏感，
而不是 nearest non-source feature 同样敏感。
```

random4 口径：

```text
strict random4 只使用 random_control_actual_iou <= 0.05 的随机区域。
```

注意：本 pack 有若干 answer masks 面积很大，导致 same-area random region 很难避开 `answer ∪ relate`，因此 strict random4 覆盖率偏低。

## Clean calibration

clean calibration 完全通过：

| node_role | node_source | rows | samples | mean_abs_error | max_abs_error | exact_match_rate |
|---|---:|---:|---:|---:|---:|---:|
| support | nearest_control | 16 | 8 | 0 | 0 | 1.0 |
| support | source | 16 | 8 | 0 | 0 | 1.0 |
| suppressor | nearest_control | 4 | 1 | 0 | 0 | 1.0 |
| suppressor | source | 4 | 1 | 0 | 0 | 1.0 |

解释：

```text
region run 的 clean condition 与前一步 source/nearest clean-screen 完全一致。
因此本轮不是因为 manifest、target token 或 feature metadata 对不齐而产生结果。
```

## Strict random4 覆盖率

按预注册 `IoU <= 0.05` 口径：

| node_role | node_source | rows | samples | rows_with_random4 | samples_with_random4 | mean_valid_random_controls |
|---|---|---:|---:|---:|---:|---:|
| support | source | 16 | 8 | 4 | 2 | 4 |
| support | nearest_control | 16 | 8 | 4 | 2 | 4 |
| suppressor | source | 4 | 1 | 0 | 0 | NaN |
| suppressor | nearest_control | 4 | 1 | 0 | 0 | NaN |

关键判断：

```text
strict random4 对照在这个 pack 中覆盖不足。
它只能用于 diagnostic，不能支撑 strongest random-region specificity claim。
```

造成覆盖不足的主要原因不是脚本失败，而是几张图的 answer mask 面积太大，same-area random rectangle 很难在 `IoU<=0.05` 下避开 `answer ∪ relate`。

## Primary route results

### 1. support source weakening

| metric | unit | n | mean | median | positive_rate | bootstrap 95% CI |
|---|---|---:|---:|---:|---:|---|
| support source answer-mask weakening | pair | 16 | +0.480 | +0.438 | 0.625 | [+0.109, +0.852] |
| support source answer-mask weakening | sample_mean | 8 | +0.237 | +0.156 | 0.500 | [-0.281, +0.760] |
| support source union-mask weakening | pair | 16 | +0.574 | +0.500 | 0.625 | [+0.254, +0.902] |
| support source union-mask weakening | sample_mean | 8 | +0.536 | +0.313 | 0.625 | [+0.078, +1.049] |

解释：

```text
support source route 在 answer mask 和 union mask 下整体变弱。
union mask 的 sample-level CI 不跨 0，是本轮最干净的 route weakening 读数。
answer mask 在 pair-level 稳定为正，但 sample-level CI 跨 0，说明样本异质性较强。
```

### 2. source > nearest-control

| metric | unit | n | mean | median | positive_rate | bootstrap 95% CI |
|---|---|---:|---:|---:|---:|---|
| source-minus-nearest answer-mask weakening | pair | 16 | +0.266 | +0.406 | 0.625 | [-0.117, +0.613] |
| source-minus-nearest answer-mask weakening | sample_mean | 8 | +0.184 | +0.396 | 0.625 | [-0.254, +0.579] |
| source-minus-nearest union-mask weakening | pair | 16 | +0.164 | +0.188 | 0.563 | [-0.168, +0.473] |
| source-minus-nearest union-mask weakening | sample_mean | 8 | +0.064 | +0.156 | 0.625 | [-0.313, +0.409] |

解释：

```text
source > nearest-control 的方向为正，但 bootstrap CI 跨 0。
因此 Stage 2A nearest8 对 node specificity 的复现只能写成 partial / directional replication，
不能写成强统计复现。
```

### 3. answer/union > random4

strict `IoU<=0.05` 下：

| metric | unit | n | mean | median | positive_rate | bootstrap 95% CI |
|---|---|---:|---:|---:|---:|---|
| support source answer-mask minus random4 | pair | 4 | +0.344 | +0.406 | 0.750 | [-0.016, +0.641] |
| support source answer-mask minus random4 | sample_mean | 2 | +0.167 | +0.167 | 0.500 | [-0.188, +0.521] |
| support source union-mask minus random4 | pair | 4 | +0.359 | +0.406 | 0.750 | [+0.078, +0.594] |
| support source union-mask minus random4 | sample_mean | 2 | +0.219 | +0.219 | 0.500 | [-0.063, +0.500] |

解释：

```text
方向是正的，尤其 union-minus-random4 在 pair-level 为正；
但有效样本只有 2 个，不能作为主成功证据。
```

diagnostic `all-random` 口径下：

| metric | unit | n | mean | median | positive_rate | bootstrap 95% CI |
|---|---|---:|---:|---:|---:|---|
| support source answer-mask minus all-random | pair | 16 | +0.241 | +0.297 | 0.750 | [+0.023, +0.448] |
| support source answer-mask minus all-random | sample_mean | 8 | +0.114 | +0.094 | 0.500 | [-0.174, +0.397] |
| support source union-mask minus all-random | pair | 16 | +0.335 | +0.297 | 0.750 | [+0.055, +0.673] |
| support source union-mask minus all-random | sample_mean | 8 | +0.413 | +0.211 | 0.500 | [-0.005, +1.010] |

解释：

```text
all-random 结果方向支持 evidence-region sensitivity，
但因为允许较高 IoU，不能替代预注册 strict random4 主对照。
它可以放在 appendix / diagnostic，而不能写成 strongest evidence。
```

## Sample-level heterogeneity

support source sample table：

| sample_id | support_pairs | answer_weakening | union_weakening | strict random4 available |
|---|---:|---:|---:|---:|
| okvqa_val_1740705 | 2 | +0.500 | +0.500 | no |
| okvqa_val_1927165 | 3 | +1.542 | +1.271 | no |
| okvqa_val_2131565 | 2 | -0.188 | -0.250 | no |
| okvqa_val_2683965 | 3 | +1.042 | +1.021 | yes |
| okvqa_val_343215 | 1 | -1.063 | +1.813 | no |
| okvqa_val_3794755 | 2 | -0.063 | -0.063 | no |
| okvqa_val_5735275 | 1 | -0.250 | -0.125 | yes |
| okvqa_val_80655 | 2 | +0.375 | +0.125 | no |

关键观察：

```text
强正样本：okvqa_val_1927165, okvqa_val_2683965, okvqa_val_1740705。
混合样本：okvqa_val_343215 answer negative but union strongly positive。
弱或反向样本：okvqa_val_2131565, okvqa_val_5735275, okvqa_val_3794755。
```

这说明 Stage 2A nearest8 不是“所有 localized 样本都稳定成立”，而是“在 targeted replication 中复现了平均方向和若干强 case，但异质性仍然明显”。

## 预期与实际偏差

预期：

```text
answer/union weakening > 0
source weakening > nearest-control weakening
answer/union weakening > strict random4
```

实际：

```text
answer/union weakening > 0：基本支持，union 更稳。
source > nearest-control：方向支持，但 CI 跨 0，只能 partial。
answer/union > strict random4：覆盖不足，只能 diagnostic。
```

最大偏差：

```text
strict random4 有效样本只有 2 个。
```

这不是远端 job 失败，而是标注几何和 same-area non-overlap 随机控制之间的冲突：当 answer/relate 区域很大时，同面积随机遮挡很难避开证据区域。

## 结论

本实验给出一个 partial but useful replication。

可以说：

```text
Stage 2A nearest8 中，support source routes 在 answer/union evidence-region mask 下整体变弱；
union-mask weakening 最稳定；
source 比 nearest-control 更 evidence-sensitive 的方向复现，但统计上仍弱；
strict random-region specificity 在本 pack 中因为随机区域有效覆盖不足，不能作为 strongest evidence。
```

不能说：

```text
Stage 2A 已经强复现了 answer/union > random4；
Stage 2A 已经强证明 source > nearest-control；
所有 localized 样本都有稳定 evidence-region-sensitive support route。
```

## 对主 claim 的影响

对主 claim 的支持：

```text
support routes are evidence-region-sensitive
```

得到 Stage 2A targeted replication 的方向性支持，尤其是 union mask。

对主 claim 的限制：

```text
specificity over random region controls 仍主要依赖 core24；
Stage 2A nearest8 更像 focused targeted replication，而不是完全独立强复现。
```

因此当前总口径应写成：

```text
core24 provides the strongest evidence;
Stage 2A nearest8 directionally replicates support-route evidence sensitivity,
with stronger union-mask results and weaker random-control coverage.
```

## 后续动作

1. 补 Stage 2A behavior eval，检查 answer/union mask 是否导致 target rank / margin damage。
2. 对大 answer-mask 样本单独标记，避免把它们解释成 small localized evidence。
3. 如果要补 random-region specificity，下一轮需要改控制设计：
   - 对大 mask 样本使用 area-capped random controls；
   - 或按 available non-evidence area 采样；
   - 或只在 compact-answer subset 中做 strict same-area random controls。
4. 不建议马上扩大到更多样本；应先把 random-control 几何问题写清楚，再决定是否补 targeted compact subset。
