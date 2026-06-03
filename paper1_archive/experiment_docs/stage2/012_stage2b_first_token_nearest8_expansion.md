# 实验 012：Stage 2B nearest8 first-token bridge expansion

## 目的

实验 011 已经证明 first-token bridge 在工程上可行，并且在 4 个精选 case 中出现了一个很强的正向样本 `okvqa_val_2683965`。但 4-case smoke 还不能说明这是不是孤立现象。

本实验把 first-token bridge 扩大到 Stage 2A nearest8 region pack 中全部 support source-control pairs，目标是判断：

```text
support source node 清零是否整体上比 nearest-control node 更伤下一答案 token？
这种差异在 clean、answer_mask、union_mask 中哪个条件下最稳定？
```

这一步仍然不是完整 decoded generation。它回答的是更窄但更可控的问题：

```text
node intervention 是否会影响答案开头 token 的 logit / rank / top1？
```

---

## 输入

远端输入：

```text
/root/autodl-tmp/tca-reasoning/annotation/stage2a_region_replication_top24_nearest8/region_experiment_manifest_remote.csv
/root/autodl-tmp/tca-reasoning/annotation/stage2a_region_replication_top24_nearest8/exported_masks
```

本地同步输入/输出目录：

```text
E:\Bridging\annotation\stage2b_node_generation_smoke
```

运行脚本：

```text
E:\Bridging\vlm-circuit-tracing\circuit_tracer_vlm\scripts\research\run_region_mask_node_first_token_smoke.py
```

分析脚本：

```text
E:\Bridging\scripts\local\summarize_stage2b_first_token_smoke.py
```

---

## 输出

远端输出：

```text
/root/autodl-tmp/tca-reasoning/annotation/stage2b_node_generation_smoke/stage2b_first_token_support_nearest8.csv
/root/autodl-tmp/tca-reasoning/annotation/stage2b_node_generation_smoke/stage2b_first_token_support_nearest8.log
```

本地同步：

```text
E:\Bridging\annotation\stage2b_node_generation_smoke\stage2b_first_token_support_nearest8.csv
E:\Bridging\annotation\stage2b_node_generation_smoke\stage2b_first_token_support_nearest8.log
```

分析输出：

```text
E:\Bridging\annotation\stage2b_node_generation_smoke\analysis_first_token_nearest8\first_token_node_source_summary.csv
E:\Bridging\annotation\stage2b_node_generation_smoke\analysis_first_token_nearest8\first_token_source_nearest_gap_summary.csv
E:\Bridging\annotation\stage2b_node_generation_smoke\analysis_first_token_nearest8\first_token_pair_gap_table.csv
E:\Bridging\annotation\stage2b_node_generation_smoke\analysis_first_token_nearest8\first_token_case_table.csv
E:\Bridging\annotation\stage2b_node_generation_smoke\analysis_first_token_nearest8\first_token_bootstrap_summary.csv
E:\Bridging\annotation\stage2b_node_generation_smoke\analysis_first_token_nearest8\first_token_sample_run_gap_table.csv
```

---

## 方法

### 1. 实验对象

从 Stage 2A nearest8 manifest 中选择：

```text
node_role = support
node_source = source + nearest_control
```

规模：

```text
support source-control pairs = 16
node sources = 2
conditions = 3
expected rows = 16 x 2 x 3 = 96
```

实际输出：

```text
done rows = 96
csv rows including header = 97
error rows = 0
```

### 2. 条件

每个 source / nearest row 跑：

```text
clean
answer_mask
union_mask
```

没有跑 random controls，因为本实验的控制变量是 node control：

```text
source node vs nearest-control node
```

而不是区域控制。

### 3. first-token intervention

对每一行：

```python
baseline_logits = model.forward_from_batch(batch)
intervention_logits, _ = model.feature_intervention(
    batch,
    [(layer, pos, feature_id, 0.0)],
    freeze_attention=True,
    apply_activation_function=True,
    sparse=False,
)
```

然后读取最后位置的下一 token 分布：

```text
target_logit
target_prob
target_rank
top1_token
```

### 4. 主要指标

节点内指标：

```text
rank_damage_by_intervention = intervention_target_rank - baseline_target_rank
delta_target_logit = intervention_target_logit - baseline_target_logit
```

source-minus-nearest gap：

```text
rank_damage_gap = source_rank_damage - nearest_rank_damage
delta_logit_gap = source_delta_logit - nearest_delta_logit
```

解释：

```text
rank_damage_gap > 0:
    source 清零比 nearest-control 更伤目标 token 排名。

delta_logit_gap < 0:
    source 清零比 nearest-control 更降低目标 token logit。
```

### 5. 统计口径

使用两种 unit：

```text
pair-level:
    每个 source-control pair 是一个单位，n = 16。

sample-run-level:
    对同一个 sample_id x run x condition 内多个 pair 先平均，n = 9。
```

bootstrap CI：

```text
对对应 unit 重采样 10,000 次，报告 mean 和 95% CI。
```

---

## 结果一：node-source summary

| node_source | condition | n | mean_rank_damage | median_rank_damage | positive_rank_damage_rate | mean_delta_target_logit |
|---|---:|---:|---:|---:|---:|---:|
| source | clean | 16 | +3.25 | 0.00 | 0.4375 | -1.0273 |
| source | answer_mask | 16 | +16.06 | +1.00 | 0.5000 | -0.5469 |
| source | union_mask | 16 | +6.00 | +4.00 | 0.8750 | -0.4531 |
| nearest_control | clean | 16 | +0.625 | 0.00 | 0.3750 | -0.5117 |
| nearest_control | answer_mask | 16 | -2.94 | 0.00 | 0.3125 | -0.2969 |
| nearest_control | union_mask | 16 | +1.06 | 0.00 | 0.3125 | -0.1016 |

读法：

```text
source 清零在三个条件下平均都降低目标 token logit；
union_mask 下 source rank damage 最一致，positive rate = 0.875；
nearest-control 的 rank damage 更弱，尤其 answer_mask 下均值为负。
```

---

## 结果二：source-minus-nearest pair-level bootstrap

| condition | metric | n | mean | bootstrap 95% CI | status |
|---|---|---:|---:|---|---|
| clean | rank_damage_gap | 16 | +2.625 | [+0.375, +5.564] | positive |
| clean | delta_logit_gap | 16 | -0.516 | [-0.930, -0.117] | positive for source damage |
| answer_mask | rank_damage_gap | 16 | +19.000 | [+1.563, +47.377] | positive, strong-case driven |
| answer_mask | delta_logit_gap | 16 | -0.250 | [-0.770, +0.102] | weak |
| union_mask | rank_damage_gap | 16 | +4.938 | [+2.000, +8.563] | stable |
| union_mask | delta_logit_gap | 16 | -0.352 | [-0.766, -0.043] | stable |

关键判断：

```text
pair-level 下，union_mask 是最稳条件：
rank_damage_gap > 0 且 delta_logit_gap < 0，两个 CI 都不跨 0。
```

answer_mask 的 rank gap 也为正且 CI 不跨 0，但它的均值明显受强 case 影响，因此读法要保守。

---

## 结果三：source-minus-nearest sample-run bootstrap

| condition | metric | n | mean | bootstrap 95% CI | status |
|---|---|---:|---:|---|---|
| clean | rank_damage_gap | 9 | +2.611 | [+0.111, +6.167] | positive |
| clean | delta_logit_gap | 9 | -0.326 | [-0.762, +0.125] | weak |
| answer_mask | rank_damage_gap | 9 | +11.870 | [+0.185, +32.704] | positive, heterogeneous |
| answer_mask | delta_logit_gap | 9 | -0.170 | [-0.431, +0.038] | weak |
| union_mask | rank_damage_gap | 9 | +4.167 | [+1.833, +6.667] | stable |
| union_mask | delta_logit_gap | 9 | -0.277 | [-0.468, -0.097] | stable |

关键判断：

```text
sample-run-level 仍然支持 union_mask：
source 清零比 nearest-control 更伤 target rank，也更降低 target logit。
```

这比实验 011 更强，因为它不是单个 case，而是跨 9 个 sample-run 聚合后仍保留方向。

---

## 结果四：sample-run gap 表

| sample_id | run | condition | rank_damage_gap | delta_logit_gap | source_rank_damage | nearest_rank_damage |
|---|---|---|---:|---:|---:|---:|
| `okvqa_val_1740705` | A | answer_mask | +10.00 | -0.1563 | +22.50 | +12.50 |
| `okvqa_val_1740705` | A | union_mask | +10.00 | -0.1563 | +22.50 | +12.50 |
| `okvqa_val_1927165` | B | answer_mask | +0.67 | 0.0000 | +0.67 | 0.00 |
| `okvqa_val_1927165` | B | union_mask | +3.00 | -0.2708 | +3.00 | 0.00 |
| `okvqa_val_2131565` | B | answer_mask | 0.00 | -0.1250 | 0.00 | 0.00 |
| `okvqa_val_2131565` | B | union_mask | +0.50 | -0.3750 | +2.00 | +1.50 |
| `okvqa_val_2683965` | B | answer_mask | +93.67 | -1.0625 | +69.00 | -24.67 |
| `okvqa_val_2683965` | B | union_mask | +9.00 | -0.8750 | +8.33 | -0.67 |
| `okvqa_val_343215` | B | answer_mask | +2.00 | +0.3750 | +9.00 | +7.00 |
| `okvqa_val_343215` | B | union_mask | 0.00 | -0.2500 | -7.00 | -7.00 |
| `okvqa_val_3794755` | A | answer_mask | 0.00 | -0.1250 | 0.00 | 0.00 |
| `okvqa_val_3794755` | A | union_mask | +2.00 | -0.1250 | +2.00 | 0.00 |
| `okvqa_val_3794755` | B | answer_mask | +2.00 | -0.3750 | +2.00 | 0.00 |
| `okvqa_val_3794755` | B | union_mask | +6.00 | -0.3750 | +6.00 | 0.00 |
| `okvqa_val_5735275` | A | answer_mask | 0.00 | 0.0000 | 0.00 | 0.00 |
| `okvqa_val_5735275` | A | union_mask | 0.00 | +0.2500 | 0.00 | 0.00 |
| `okvqa_val_80655` | A | answer_mask | -1.50 | -0.0625 | -4.00 | -2.50 |
| `okvqa_val_80655` | A | union_mask | +7.00 | -0.3125 | +6.00 | -1.00 |

读法：

```text
union_mask 下多数 sample-run 为正，且没有明显强反向；
answer_mask 下 okvqa_val_2683965 很强，但 okvqa_val_80655 为反向；
okvqa_val_5735275 基本无效，是低信息样本。
```

---

## 关键 case

### 1. `okvqa_val_2683965`

这是目前最强 node-to-first-token bridge case。

```text
answer_mask:
source rank damage = +69.00
nearest rank damage = -24.67
rank_damage_gap = +93.67
delta_logit_gap = -1.0625

union_mask:
source rank damage = +8.33
nearest rank damage = -0.67
rank_damage_gap = +9.00
delta_logit_gap = -0.8750
```

解释：

```text
source node 清零显著伤害 target token；
nearest-control 不仅不伤害，有时还改善 rank。
```

这是后续 decoded generation loop 的第一候选。

### 2. `okvqa_val_1740705`

```text
answer_mask / union_mask rank_damage_gap = +10.00
source 和 nearest 都有 rank damage，但 source 更强。
```

这是比较稳的第二候选，但 clean 条件也有强 gap，说明 source 本身在 clean distribution 中已经更关键。

### 3. `okvqa_val_80655`

```text
answer_mask rank_damage_gap = -1.50
union_mask rank_damage_gap = +7.00
```

解释：

```text
它支持 union_mask bridge，但 answer_mask 反向。
```

适合作为“union 比 answer 更稳”的例子。

### 4. `okvqa_val_5735275`

```text
rank_damage_gap 基本为 0
```

这是一个低信息或无效样本，不适合做主图。

---

## 预期与实际偏差

### 预期

理想情况下：

```text
source 清零在 answer_mask 和 union_mask 下都比 nearest-control 更伤 target token；
clean 条件 gap 较小；
top1 change 更常出现在 source。
```

### 实际

实际结果更细：

```text
union_mask 最稳，pair-level 和 sample-run-level 的 rank/logit gap 都支持 source-specific damage；
answer_mask 的 rank gap 为正，但受 okvqa_val_2683965 强烈驱动，logit gap CI 跨 0；
clean 条件也有 source-minus-nearest gap，说明 traced support source 在 clean distribution 中本来就更关键；
top1 change 不是稳定主指标，仍应以 rank/logit gap 为主。
```

---

## 结论

实验 012 的正式判定：

```text
Stage 2B first-token bridge has directional replication on nearest8, strongest under union_mask.
```

中文：

```text
Stage 2B 的 first-token 桥接在 nearest8 上获得方向性复现，其中 union_mask 条件最稳定。
```

可以写：

```text
在 Stage 2A nearest8 的全部 support pairs 中，source node 清零比 nearest-control 更伤下一答案 token 的 rank 和 logit，尤其在 union evidence-region mask 下，pair-level 与 sample-run-level bootstrap CI 均支持同一方向。
```

不能写：

```text
node zeroing 已经稳定改变完整 decoded answer。
answer_mask 条件已经强统计复现。
所有样本都表现出 source-specific first-token damage。
top1 change 是 source-specific。
```

---

## 对主 claim 的影响

这一步让主链更完整：

```text
traced support route exists
-> evidence-region mask weakens route
-> evidence-region mask damages behavior
-> source node zeroing damages first-answer-token distribution more than nearest controls
```

它还不能完成最后一环：

```text
source node zeroing -> full decoded answer change
```

但已经比实验 011 更强，因为它不再只是一个 4-case smoke，而是在 Stage 2A nearest8 全 support pairs 上得到稳定的 first-token distribution 方向。

---

## 下一步

我建议下一步不要继续盲目扩大 first-token bridge，而是做一个更深但小的 decoded loop：

```text
Stage 2B-decoded-loop-1:
case = okvqa_val_2683965
conditions = clean, answer_mask, union_mask
nodes = strongest source pair + nearest-control
generation = hand-written greedy loop, max_new_tokens = 4 或 8
goal = 看 source zeroing 是否能改变短答案 token sequence，并强于 nearest-control
```

如果这个单 case 成功，再扩到：

```text
okvqa_val_1740705
okvqa_val_80655
```

如果失败，也不应解释为机制失败，而应记录为：

```text
first-token bridge does not yet imply full decoded-answer bridge.
```

---

## 一句话总结

实验 012 把 Stage 2B 从“工程可行 + 一个强 case”推进到“nearest8 上有稳定 first-token 方向性”：source support node 清零比 nearest-control 更伤下一答案 token，尤其在 union evidence-region mask 下最稳。下一步应转向 `okvqa_val_2683965` 的手写 decoded generation loop，而不是继续无目的扩大样本。
