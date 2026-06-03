# 实验 011：Stage 2B first-token node bridge smoke

## 目的

Stage 2B 的目标是补主链中最缺的一环：

```text
node intervention -> generation-side behavior
```

此前我们已经有两条链：

```text
evidence-region mask -> support route weakening
evidence-region mask -> target rank / decoded answer change
```

但还没有直接证明：

```text
清零某个 source node 会改变生成侧分布或最终答案。
```

本实验先做一个最小可行 smoke，不直接追求完整自然语言生成，而是比较第一答案 token 的分布：

```text
在 assistant_prefix = "The answer is " 已固定时，
下一 token 基本就是答案开头 token。
```

所以本轮问的是：

```text
清零 support source node 是否会伤害目标答案 token 的 logit / probability / rank？
这种伤害是否比 nearest-control node 更强？
清零是否会改变下一 token 的 top1？
```

---

## 为什么没有直接做完整 decoded generation

我们先尝试了旧脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_region_mask_node_generation_smoke.py
```

这个脚本调用：

```text
model.generate(batch, ...)
model.feature_intervention_generate(batch, ...)
```

实际失败原因：

```text
HookedVLTransformer.generate asserts input must be torch.Tensor or str.
当前多模态输入是包含 input_ids / attention_mask / image 的 batch dict。
```

具体 traceback：

```text
File ".../HookedVLTransformer.py", line 2370, in generate
    assert isinstance(tokens, torch.Tensor)
AssertionError
```

解释：

```text
这不是机制失败，而是现有 generation helper 对 VLM batch 的接口不兼容。
普通 region generation eval 能跑，是因为它直接使用 HuggingFace Gemma3ForConditionalGeneration.generate；
node intervention 必须走 ReplacementModel / HookedVLTransformer 的 hook 路径，而这个 generate 目前不支持 batch dict。
```

因此本轮改成 first-token bridge。它比完整 decoded generation 更窄，但工程上更可靠，也更直接对应 target token / rank 机制读数。

---

## 输入

本地 manifest：

```text
E:\Bridging\annotation\stage2b_node_generation_smoke\stage2b_node_generation_smoke_manifest.csv
E:\Bridging\annotation\stage2b_node_generation_smoke\stage2b_node_generation_smoke_manifest_remote.csv
```

远端 manifest：

```text
/root/autodl-tmp/tca-reasoning/annotation/stage2b_node_generation_smoke/stage2b_node_generation_smoke_manifest_remote.csv
```

mask root：

```text
/root/autodl-tmp/tca-reasoning/annotation/stage2a_region_replication_top24_nearest8/exported_masks
```

选入的 4 个 case / source-control pair：

| sample_id | run | pair_id | 选择原因 |
|---|---|---|---|
| `okvqa_val_1927165` | B | `003_okvqa_val_1927165_B_support_L11_P196_F151858` | route weakening 强，decoded answer 有明显变化 |
| `okvqa_val_2683965` | B | `010_okvqa_val_2683965_B_support_L26_P287_F39687` | answer / union route weakening 强，rank damage 强，是最适合 first-token bridge 的 visual-readout case |
| `okvqa_val_80655` | A | `019_okvqa_val_80655_A_support_L11_P196_F151858` | decoded answer 在动作描述上变化，适合看生成侧 |
| `okvqa_val_3794755` | A | `012_okvqa_val_3794755_A_support_L26_P293_F84060` | behavior damage 强但 route weakening 弱，作为 dissociation / heterogeneity case |

---

## 输出

远端输出：

```text
/root/autodl-tmp/tca-reasoning/annotation/stage2b_node_generation_smoke/stage2b_first_token_source_nearest_4case.csv
/root/autodl-tmp/tca-reasoning/annotation/stage2b_node_generation_smoke/stage2b_first_token_source_nearest_4case.log
```

本地同步：

```text
E:\Bridging\annotation\stage2b_node_generation_smoke\stage2b_first_token_source_nearest_4case.csv
E:\Bridging\annotation\stage2b_node_generation_smoke\stage2b_first_token_source_nearest_4case.log
```

分析输出：

```text
E:\Bridging\annotation\stage2b_node_generation_smoke\analysis_first_token\first_token_node_source_summary.csv
E:\Bridging\annotation\stage2b_node_generation_smoke\analysis_first_token\first_token_source_nearest_gap_summary.csv
E:\Bridging\annotation\stage2b_node_generation_smoke\analysis_first_token\first_token_pair_gap_table.csv
E:\Bridging\annotation\stage2b_node_generation_smoke\analysis_first_token\first_token_case_table.csv
E:\Bridging\annotation\stage2b_node_generation_smoke\analysis_first_token\STAGE2B_FIRST_TOKEN_SMOKE_SUMMARY.md
```

新增脚本：

```text
E:\Bridging\vlm-circuit-tracing\circuit_tracer_vlm\scripts\research\run_region_mask_node_first_token_smoke.py
E:\Bridging\scripts\local\summarize_stage2b_first_token_smoke.py
```

---

## 方法

### 1. 构建 4-case manifest

从 Stage 2A nearest8 manifest 中筛选 4 个 pair，并保留每个 pair 的两行：

```text
source
nearest_control
```

因此总 manifest 行数：

```text
4 cases x 2 node_source = 8 rows
```

每行包含：

```text
sample_id
run
prompt_name
assistant_prefix
question
image_path
feature_layer
feature_pos
feature_id
target_token_id
node_source
node_role
```

### 2. 运行条件

每行运行三个图像条件：

```text
clean
answer_mask
union_mask
```

因此总输出：

```text
8 manifest rows x 3 conditions = 24 rows
```

### 3. first-token 计算方式

对每个 `sample_id x run x node_source x condition`：

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

然后只看最后位置的 next-token logits：

```text
baseline_target_logit
intervention_target_logit
delta_target_logit = intervention - baseline

baseline_target_rank
intervention_target_rank
rank_damage_by_intervention = intervention_rank - baseline_rank

baseline_top1_token
intervention_top1_token
top1_changed_by_intervention
```

解释：

```text
rank_damage_by_intervention > 0:
    清零节点后目标答案 token 排名变差。

delta_target_logit < 0:
    清零节点后目标答案 token logit 下降。

top1_changed_by_intervention = True:
    清零节点改变了下一 token 的贪心选择。
```

### 4. source-minus-nearest gap

把同一个 `pair_id x condition` 的 source 和 nearest_control 合并：

```text
rank_damage_gap = source_rank_damage - nearest_rank_damage
delta_logit_gap = source_delta_logit - nearest_delta_logit
```

解释：

```text
rank_damage_gap > 0:
    source 清零比 nearest 更伤 target rank。

delta_logit_gap < 0:
    source 清零比 nearest 更降低 target logit。
```

---

## 运行完整性

远端 first-token smoke 成功完成：

```text
rows = 24
error rows = 0
conditions = clean / answer_mask / union_mask
node_source = source / nearest_control
```

这说明：

```text
ReplacementModel.forward_from_batch + feature_intervention 可以用于 VLM batch 的 first-token intervention。
```

这是 Stage 2B 的重要工程进展，因为之前完整 generation helper 卡在 batch dict 接口上。

---

## 总体结果

### Node-source summary

| node_source | condition | n | mean_rank_damage | median_rank_damage | positive_rank_damage_rate | mean_delta_target_logit | negative_delta_logit_rate | top1_changed_count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| source | clean | 4 | +0.25 | 0.00 | 0.25 | -1.0625 | 1.00 | 1 |
| source | answer_mask | 4 | +20.25 | -2.50 | 0.25 | -0.171875 | 0.75 | 1 |
| source | union_mask | 4 | +6.00 | +6.00 | 1.00 | -0.390625 | 0.75 | 1 |
| nearest_control | clean | 4 | 0.00 | 0.00 | 0.25 | -0.125 | 0.25 | 1 |
| nearest_control | answer_mask | 4 | -28.75 | -4.00 | 0.00 | +0.734375 | 0.00 | 1 |
| nearest_control | union_mask | 4 | -2.75 | -0.50 | 0.00 | +0.515625 | 0.00 | 1 |

读法：

```text
source 清零平均上更倾向于降低目标 token logit；
nearest 清零在 answer/union mask 下平均反而提高目标 token logit；
union_mask 下 source rank_damage 4/4 为正，是本轮最整齐的方向性结果。
```

但也要注意：

```text
top1_changed_count 在 source 和 nearest 中都出现；
top1 change 不能单独作为 source-specific evidence。
```

### Source-minus-nearest gap summary

| condition | n | mean_rank_damage_gap | median_rank_damage_gap | positive_rank_damage_gap_rate | mean_delta_logit_gap | negative_delta_logit_gap_rate |
|---|---:|---:|---:|---:|---:|---:|
| clean | 4 | +0.25 | 0.00 | 0.25 | -0.9375 | 1.00 |
| answer_mask | 4 | +49.00 | +1.50 | 0.50 | -0.90625 | 0.75 |
| union_mask | 4 | +8.75 | +6.50 | 1.00 | -0.90625 | 0.75 |

读法：

```text
union_mask 的 source-minus-nearest rank gap 最一致；
answer_mask 的均值很大，但主要由 okvqa_val_2683965 驱动；
delta_logit_gap 在三个条件下都为负，说明 source 清零相对 nearest 更降低目标 token logit。
```

---

## Case 结果

### 1. `okvqa_val_2683965`：最强正向 bridge case

目标 token：

```text
oval
```

关键结果：

| condition | source rank damage | nearest rank damage | rank gap | source delta logit | nearest delta logit | delta logit gap |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0 | 0 | 0 | -2.25 | +0.375 | -2.625 |
| answer_mask | +96 | -107 | +203 | -0.875 | +2.8125 | -3.6875 |
| union_mask | +10 | -10 | +20 | -0.9375 | +2.0625 | -3.000 |

读法：

```text
这是当前最干净的 node-to-first-token bridge case。
source node 清零降低目标 token logit，并在 answer/union mask 下伤害目标 rank；
nearest control 方向相反，甚至改善目标 rank。
```

这个 case 可作为 Stage 2B 的主图候选。

### 2. `okvqa_val_1927165`：有 top1 change，但 rank 方向不单调

目标 token：

```text
halt
```

关键结果：

| condition | source rank damage | nearest rank damage | source top1 change | nearest top1 change |
|---|---:|---:|---|---|
| clean | +1 | +1 | no | no |
| answer_mask | -10 | 0 | `a` -> `stop` | no |
| union_mask | +2 | 0 | no | no |

读法：

```text
source 清零能改变 answer_mask 下 top1 token；
但 target token rank 反而改善，因此不能简单写成“source 清零伤害答案”。
这个 case 更适合展示 first-token 分布会被 source intervention 改动，而不是展示干净支持路径。
```

### 3. `okvqa_val_80655`：source 和 nearest 都会影响 top1

目标 token：

```text
baseball
```

关键结果：

| condition | source rank damage | nearest rank damage | source top1 change | nearest top1 change |
|---|---:|---:|---|---|
| clean | 0 | -1 | no | no |
| answer_mask | -5 | -8 | no | no |
| union_mask | +10 | -1 | `sliding` -> `swinging` | `sliding` -> `swinging` |

读法：

```text
union_mask 下 source rank damage 为正；
但 source 和 nearest 都改变 top1，说明 top1 change 不是 source-specific。
这个 case 适合保留为“生成侧敏感但节点特异性不足”的例子。
```

### 4. `okvqa_val_3794755`：非特异或格式/竞争答案敏感

目标 token：

```text
laptop
```

关键结果：

| condition | source rank damage | nearest rank damage | source top1 change | nearest top1 change |
|---|---:|---:|---|---|
| clean | 0 | 0 | `a` -> `multiple` | `a` -> `multiple` |
| answer_mask | 0 | 0 | no | no |
| union_mask | +2 | 0 | no | no |

读法：

```text
clean 条件下 source 和 nearest 都改变 top1；
这个 case 不适合作为 source-specific bridge evidence。
它更像一个 dissociation / heterogeneity case。
```

---

## 预期与实际偏差

### 预期

理想情况下：

```text
source clean / answer / union 都应比 nearest 更明显降低目标 token logit；
source rank_damage 应为正；
source top1 change 应比 nearest 更常见。
```

### 实际

实际更复杂：

```text
source delta logit 相对 nearest 更负，这一点比较一致；
union_mask 下 source rank_damage gap 4/4 为正；
answer_mask 下均值强正，但只有 2/4 pair 为正，主要由 okvqa_val_2683965 驱动；
top1 change 在 source 和 nearest 都出现，不能作为干净主指标。
```

---

## 结论

本实验的正式判定：

```text
Stage 2B first-token bridge is technically feasible and mechanistically informative, but not yet a strong statistical result.
```

中文：

```text
Stage 2B 的第一 token 桥接在工程上已经跑通，并给出有机制信息的结果，但还不是强统计结论。
```

可以说：

```text
在 selected cases 中，support source node 清零可以改变下一答案 token 的 logit / rank / top1；
最强 case okvqa_val_2683965 呈现明确 source-specific rank damage；
union_mask 下 source-minus-nearest rank gap 在 4 个 case 中方向一致为正。
```

不能说：

```text
node zeroing 已经稳定改变完整 decoded answer；
source node 清零在所有 case 中都伤害答案；
top1 change 是 source-specific；
Stage 2B 已经完成 node-to-generation 的强因果闭环。
```

---

## 对主 claim 的影响

Stage 2B first-token smoke 对主 claim 的影响是：

```text
它把主链从 target-logit intervention 推进到 generation-side distribution；
证明这个方向工程上可行；
并找到了至少一个强正向 case，可用于后续更深 node-to-generation figure。
```

但当前仍需要保守：

```text
完整自然生成仍未被 node intervention 直接改变；
first-token bridge 只是最小桥，不是完整 decoded-answer bridge。
```

---

## 下一步

建议下一步分两条：

### 路线 1：扩大 first-token bridge

目标：

```text
从 4 cases 扩到 core24 / Stage2A 中所有 support+nearest 可用 positive cases。
```

筛选条件：

```text
support source
nearest_control available
assistant_prefix = "The answer is "
target_token_id complete
answer/union behavior damage 或 route weakening 有信号
```

成功标准：

```text
source delta_logit 比 nearest 更负；
source rank_damage 比 nearest 更正；
至少一个条件下 source-minus-nearest gap 方向稳定。
```

### 路线 2：手写 decoded generation loop

目标：

```text
绕开 HookedVLTransformer.generate 不支持 multimodal batch 的问题。
```

设计：

```text
第一个答案 token 使用 feature_intervention；
之后把选出的 token append 到 input_ids；
每一步用 forward_from_batch 重新跑 full multimodal forward；
先做 greedy max_new_tokens=4/8；
比较 source zeroing vs nearest zeroing 的 decoded short answer。
```

注意：

```text
这比 first-token 更接近最终行为，但工程风险更高；
应该先在 okvqa_val_2683965 一个 case 上做。
```

---

## 一句话总结

Stage 2B 第一轮已经把“节点清零只影响 target logit”的证据推进到“节点清零能影响下一答案 token 分布”。当前最强 case 是 `okvqa_val_2683965`，但整体结果仍异质；下一步要么扩大 first-token 统计，要么手写 decoded generation loop 来补完整自然生成桥。
