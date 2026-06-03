# Stage 2I-1 / 2I-2：Cross-Model 扩样本 decoded bridge 与 LLaVA bridge 结果

## 1. 实验目的

Stage 2H 已经证明了 3 个主 case 上的跨模型 hidden-position bridge：

```text
Qwen layer 26: top_hidden_delta_plus_answer_adjacent 能恢复 first-token target logit/rank，并有 partial decoded bridge。
LLaVA layer 15: top_hidden_delta_plus_answer_adjacent 能恢复 first-token target logit/rank，但 decoded generation 当时 underpowered。
```

Stage 2I 的目的，是回答这两个问题：

```text
1. Qwen 的 partial decoded-answer bridge 是否能从 3 个 case 扩到更多 localized samples？
2. LLaVA 之前 decoded generation underpowered 是因为样本不合适，还是因为模型上确实没有 decoded bridge 信号？
```

## 2. 输入

候选样本来自：

```text
doc/experiments/stage2/cross_model/stage2i_selected_12_manifest.csv
```

Qwen 使用 top 8：

```text
okvqa_val_1593205
okvqa_val_4739195
okvqa_val_4502065
okvqa_val_2954205
okvqa_val_1729795
okvqa_val_3959785
okvqa_val_1058855
okvqa_val_340155
```

LLaVA 使用 top 12：

```text
okvqa_val_1593205
okvqa_val_4739195
okvqa_val_4502065
okvqa_val_2954205
okvqa_val_1729795
okvqa_val_3959785
okvqa_val_1058855
okvqa_val_340155
okvqa_val_01514
okvqa_val_4043385
okvqa_val_4938465
okvqa_val_2708155
```

Prompt：

```text
B_direct
D_visual_only
```

模型与层：

```text
Qwen2.5-VL-7B-Instruct, layer 26
LLaVA-1.5-7B, layer 15
```

## 3. 方法

对每个 `sample_id x prompt`，先构造：

```text
clean image
union_mask image
```

然后比较以下 decoded generation 条件：

```text
baseline::clean
baseline::union_mask
restore::top_hidden_delta_plus_answer_adjacent
restore::answer_adjacent_text
restore::low_delta_control
restore::random_control_1
```

其中：

```text
top_hidden_delta_plus_answer_adjacent:
  从 union_mask run 中，把 top hidden-delta visual-like positions 加 answer-adjacent positions patch 回 clean hidden state。

answer_adjacent_text:
  只 patch answer-adjacent text positions，用来判断信号是不是主要由答案附近文本位置驱动。

low_delta_control:
  patch 低 hidden-delta positions，作为非敏感位置对照。

random_control_1:
  patch 随机 visual positions，作为随机位置对照。
```

生成设置：

```text
greedy decoding
max_new_tokens = 3
answer_prefix = "The answer is "
```

## 4. 输出

原始输出：

```text
doc/experiments/stage2/cross_model/stage2i_qwen_decoded_bridge_expansion.json
doc/experiments/stage2/cross_model/stage2i_qwen_decoded_bridge_expansion.csv
doc/experiments/stage2/cross_model/stage2i_llava_generation_gap_screen.json
doc/experiments/stage2/cross_model/stage2i_llava_generation_gap_screen.csv
doc/experiments/stage2/cross_model/stage2i_llava_decoded_bridge_expansion.json
doc/experiments/stage2/cross_model/stage2i_llava_decoded_bridge_expansion.csv
```

汇总输出：

```text
doc/experiments/stage2/cross_model/stage2i_cross_model_bridge_summary.csv
doc/experiments/stage2/cross_model/stage2i_cross_model_bridge_case_table.csv
doc/experiments/stage2/cross_model/stage2i_cross_model_bridge_decision.json
doc/experiments/stage2/cross_model/stage2i_bridge_bootstrap.csv
doc/experiments/stage2/cross_model/stage2i_bridge_bootstrap.json
```

## 5. 主要结果

### 5.1 Qwen decoded bridge expansion

判定：

```text
status = partial_decoded_bridge_expanded
total_sample_prompt_runs = 16
informative_clean_vs_union_runs = 15
best_group = top_hidden_delta_plus_answer_adjacent
```

核心数值：

```text
best_changed_away_from_union_informative_n = 11 / 15
best_same_as_clean_informative_n = 3 / 15
best_target_hit_n = 2 / 16
best_mean_logit_restore_vs_union = +5.109375
best_mean_rank_restore_vs_union = +1473.6875

low_delta_changed_away_from_union_informative_n = 1 / 15
random_changed_away_from_union_informative_n = 1 / 15
```

读法：

```text
Qwen 的 best bridge 明显比 low_delta/random control 更能把 decoded answer 从 union_mask answer 推开。
它也强烈恢复 target first-token logit/rank。
但它不经常把答案完整恢复到 clean/target，因此只能写 partial decoded bridge，而不是 full generation restoration。
```

### 5.2 LLaVA generation-gap screen

Stage 2H 中 LLaVA decoded generation underpowered，因为 3 个样本中 clean 和 union_mask 生成答案完全一样。

Stage 2I 扩到 12 个样本后：

```text
total_sample_prompt_runs = 24
informative_clean_vs_union_runs = 19
```

这说明：

```text
LLaVA 之前 decoded generation underpowered 主要是样本选择问题，不是 LLaVA decoded answer 对 union mask 永远不敏感。
```

### 5.3 LLaVA decoded bridge expansion

判定：

```text
status = partial_decoded_bridge_expanded
total_sample_prompt_runs = 24
informative_clean_vs_union_runs = 19
best_group = top_hidden_delta_plus_answer_adjacent
```

核心数值：

```text
best_changed_away_from_union_informative_n = 8 / 19
best_same_as_clean_informative_n = 3 / 19
best_target_hit_n = 5 / 24
best_mean_logit_restore_vs_union = +1.636556
best_mean_rank_restore_vs_union = +42.083333

low_delta_changed_away_from_union_informative_n = 0 / 19
random_changed_away_from_union_informative_n = 0 / 19
```

读法：

```text
LLaVA 现在不再只是 first-token/rank bridge。
在 selected-12 上，best bridge 能在一部分 informative rows 中改变 decoded answer，并且 controls 基本不能改变 decoded answer。
不过，best bridge 仍然很少完整恢复到 clean/target answer，因此仍是 partial decoded bridge。
```

## 6. Paired bootstrap 统计收束

bootstrap unit：

```text
sample_id x prompt
只统计 clean decoded answer 非空且 clean != union_mask 的 informative rows
```

Qwen best bridge vs low_delta/random controls：

```text
changed_away_from_union best - low_delta:
  mean = +0.666667
  95% CI = [+0.333333, +0.933333]
  status = stable_positive

changed_away_from_union best - random:
  mean = +0.666667
  95% CI = [+0.333333, +0.933333]
  status = stable_positive

logit_restore best - low_delta:
  mean = +5.297917
  95% CI = [+4.018750, +6.512500]
  status = stable_positive

logit_restore best - random:
  mean = +5.247917
  95% CI = [+3.958333, +6.400000]
  status = stable_positive
```

LLaVA best bridge vs low_delta/random controls：

```text
changed_away_from_union best - low_delta:
  mean = +0.421053
  95% CI = [+0.210526, +0.631579]
  status = stable_positive

changed_away_from_union best - random:
  mean = +0.421053
  95% CI = [+0.210526, +0.631579]
  status = stable_positive

logit_restore best - low_delta:
  mean = +1.832648
  95% CI = [+1.139186, +2.552015]
  status = stable_positive

logit_restore best - random:
  mean = +1.613487
  95% CI = [+0.903783, +2.333676]
  status = stable_positive
```

更强指标 `same_as_clean best - control`：

```text
Qwen 和 LLaVA 都是正均值，但 95% CI 触到 0。
因此“离开 union answer / 恢复 logit-rank”稳定；
“完整回到 clean answer”仍然只能写 weak / partial。
```

## 7. 预期与实际偏差

预期：

```text
Qwen 可能扩样本后仍有 partial decoded bridge；
LLaVA 可能继续 underpowered，因为 Stage 2H 中 clean/union decoded answer 没有变化。
```

实际：

```text
Qwen 符合预期，且 bridge-vs-control 差异很明显。
LLaVA 比预期更好：扩样本后出现 19/24 informative clean-vs-union rows，并且 best bridge 对 decoded answer 的影响强于 controls。
```

重要偏差：

```text
best bridge 并不明显强于 answer_adjacent_text bridge。
这说明跨模型 decoded bridge 可能包含较强的 answer-adjacent 汇聚成分，
不能解释成纯视觉 image-token source route。
```

## 8. 当前能说明什么

可以说：

```text
Qwen 和 LLaVA 都出现了多样本的 evidence-mask-sensitive hidden-state bridge。
这种 bridge 不只影响 first-token target logit/rank，也能在一部分样本中改变 decoded answer。
best bridge 相对 low-delta/random controls 的差异在 paired bootstrap 下为 stable_positive。
```

更保守但清晰的跨模型结论：

```text
Cross-model evidence now extends beyond readout sensitivity: Qwen and LLaVA both show partial decoded-answer bridge evidence at the hidden-state position level.
```

## 9. 仍然不能说明什么

不能说：

```text
Qwen/LLaVA 已经复现 Gemma 的 source-control causal route。
Qwen/LLaVA 的 CLT feature 是 causal source node。
这些 hidden positions 是对象级视觉语义节点。
best bridge 完整恢复了生成答案。
D_visual_only 比 B_direct 更好。
```

原因：

```text
Stage 2I 的干预对象是 hidden positions，而不是 traced CLT source nodes。
control 是 low-delta/random hidden position control，不是 Gemma 主线中的 nearest matched source-control。
decoded answer 改变稳定，但完整回到 clean/target 的比例仍有限。
answer_adjacent_text 也有强信号，说明机制可能是视觉信号在答案附近文本位置汇聚后的桥接，而不是纯 image-token 机制。
```

## 10. 对主 claim 的影响

Stage 2I 对 Gemma 主线是加分项，不是替代项。

主线仍然写：

```text
Gemma 上存在 source-traced, node-intervened, control-supported evidence-sensitive support routes。
```

跨模型辅助结论现在可以升级为：

```text
Qwen/LLaVA 也显示出 evidence-mask-sensitive hidden-state bridges；
在扩样本后，这种 bridge 对 decoded answer 的影响强于 low-delta/random hidden-position controls。
但这仍是 hidden-state-level bridge，不是 source-control route replication。
```

## 11. 下一步建议

最值得继续推进的三件事：

```text
1. 对 Stage 2I selected samples 跑 hidden-position restore+corrupt bidirectional expansion，
   检查 decoded bridge 的样本是否也满足双向 first-token/rank specificity。

2. 为 Qwen/LLaVA 设计更强的 activation-matched controls，
   比 random/low-delta controls 更接近 Gemma nearest-control 的精神。

3. 如果要冲更强跨模型 claim，下一阶段必须做模型内 source tracing 或 feature attribution，
   否则跨模型部分仍应保持 auxiliary。
```

