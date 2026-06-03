# Stage 2I-3：Cross-Model bidirectional hidden-position expansion

## 1. 实验目的

Stage 2I-1/2 已经看到 Qwen 和 LLaVA 在扩样本后都有 partial decoded bridge：

```text
Qwen:
  best hidden-position bridge 能让 decoded answer 离开 union_mask answer，强于 controls。

LLaVA:
  selected-12 后不再 underpowered，也出现 partial decoded bridge。
```

Stage 2I-3 的目的，是进一步验证这个 decoded bridge 背后是否也有双向 hidden-state 因果支持：

```text
restore: union_mask run 中 patch 回 clean hidden state，是否恢复 target answer logit/rank？
corrupt: clean run 中 patch 入 union hidden state，是否损伤 target answer logit/rank？
```

如果同一组 source-like positions 在 restore 和 corrupt 两个方向都强于 controls，则说明这不是单向偶然恢复，而是更接近 causal localization。

## 2. 证明层级

这一轮能支持：

```text
cross-model hidden-state-level causal localization
multi-case bidirectional hidden-position bridge
```

仍然不能支持：

```text
Qwen/LLaVA 已复现 Gemma 的 source-control causal route
Qwen/LLaVA CLT feature 是 causal source node
hidden positions 是对象级语义节点
纯视觉 token route 已经被完整证明
```

原因是：这一轮干预对象仍然是 hidden positions，而不是 Gemma 主线中 traced CLT source nodes；control 是 random/low-delta hidden-position control，而不是 nearest matched source-control node。

## 3. 输入

样本：

```text
doc/experiments/stage2/cross_model/stage2i_selected_12_manifest.csv
```

模型：

```text
Qwen2.5-VL-7B-Instruct, layer 26
LLaVA-1.5-7B, layer 15
```

Prompts：

```text
B_direct
D_visual_only
```

条件：

```text
clean
union_mask
restore groups
corrupt groups
```

位置组：

```text
top_hidden_delta_plus_answer_adjacent
top_hidden_delta
answer_adjacent_text
evidence_region
evidence_region_plus_answer_adjacent
low_delta_control
random_control_1..4
```

说明：

```text
Qwen 的 image grid 映射仍然不可靠，因此 Qwen 不强写 evidence_region token localization。
LLaVA 的 576 image token -> 24x24 grid 映射可用，因此 LLaVA 可以解释 evidence_region 相关位置。
```

## 4. 输出

原始输出：

```text
doc/experiments/stage2/cross_model/stage2i_qwen_hidden_position_patch.csv/json
doc/experiments/stage2/cross_model/stage2i_llava_hidden_position_patch.csv/json
```

分析输出：

```text
doc/experiments/stage2/cross_model/stage2i_hidden_position_patch_baseline_gap.csv
doc/experiments/stage2/cross_model/stage2i_hidden_position_patch_summary.csv
doc/experiments/stage2/cross_model/stage2i_hidden_position_patch_specificity.csv
doc/experiments/stage2/cross_model/stage2i_hidden_position_patch_case_table.csv
doc/experiments/stage2/cross_model/stage2i_hidden_position_patch_decision.json
doc/experiments/stage2/cross_model/stage2i_hidden_position_patch_bootstrap.csv/json
```

## 5. Baseline clean vs union gap

```text
Qwen:
  n_runs = 24
  mean_clean_union_logit_gap = +3.260417
  positive_logit_gap = 18/24
  mean_clean_union_rank_gap = +985.833333
  positive_rank_gap = 18/24

LLaVA:
  n_runs = 24
  mean_clean_union_logit_gap = +2.026530
  positive_logit_gap = 19/24
  mean_clean_union_rank_gap = +44.291667
  positive_rank_gap = 19/24
```

读法：

```text
union_mask 对两个模型都造成了稳定 target answer damage，因此 restore/corrupt patch 有可评估空间。
```

## 6. 主要结果：best group 的双向 specificity

Best group：

```text
top_hidden_delta_plus_answer_adjacent
```

Qwen：

```text
restore:
  source_minus_random_logit = +3.333659
  source_minus_random_rank = +875.364584
  positive_logit = 20/24

corrupt:
  source_minus_random_logit = +2.956380
  source_minus_random_rank = +410.885417
  positive_logit = 20/24

decision_status:
  supported_bidirectional_hidden_position_localization
```

LLaVA：

```text
restore:
  source_minus_random_logit = +1.220296
  source_minus_random_rank = +13.604167
  positive_logit = 21/24

corrupt:
  source_minus_random_logit = +0.577840
  source_minus_random_rank = +5.468749
  positive_logit = 18/24

decision_status:
  supported_bidirectional_hidden_position_localization
```

## 7. Paired bootstrap

Bootstrap unit：

```text
sample_id x prompt
n = 24 for each model-direction pair
```

Qwen best group vs random mean：

```text
restore effect_logit best - random_mean:
  mean = +3.333659
  95% CI = [+2.143880, +4.560547]
  status = stable_positive

corrupt effect_logit best - random_mean:
  mean = +2.956380
  95% CI = [+1.823568, +4.099609]
  status = stable_positive

restore effect_rank best - random_mean:
  mean = +875.364583
  95% CI = [+346.718750, +1518.593750]
  status = stable_positive

corrupt effect_rank best - random_mean:
  mean = +410.885417
  95% CI = [+47.916667, +957.802083]
  status = stable_positive
```

LLaVA best group vs random mean：

```text
restore effect_logit best - random_mean:
  mean = +1.220296
  95% CI = [+0.696615, +1.789998]
  status = stable_positive

corrupt effect_logit best - random_mean:
  mean = +0.577840
  95% CI = [+0.325724, +0.867676]
  status = stable_positive

restore effect_rank best - random_mean:
  mean = +13.604167
  95% CI = [+5.864583, +22.156250]
  status = stable_positive

corrupt effect_rank best - random_mean:
  mean = +5.468750
  95% CI = [+1.052083, +10.437500]
  status = stable_positive
```

## 8. 重要机制细节

### 8.1 Qwen：answer-adjacent 成分很强

Qwen summary：

```text
restore answer_adjacent_text:
  mean_effect_logit = +2.623698
  positive_logit = 20/24
  mean_effect_rank = +899.0

restore top_hidden_delta:
  mean_effect_logit = +0.367188
  positive_logit = 7/24
  mean_effect_rank = +227.375

restore top_hidden_delta_plus_answer_adjacent:
  mean_effect_logit = +3.442708
  positive_logit = 20/24
  mean_effect_rank = +986.167
```

读法：

```text
Qwen 的 strongest bridge 很大程度依赖 answer-adjacent positions。
这更像“视觉破坏后的答案附近 hidden-state 汇聚信号”被恢复，而不是纯 image-position route。
```

### 8.2 LLaVA：visual-only 与 evidence-region 也有明显信号

LLaVA summary：

```text
restore evidence_region:
  mean_effect_logit = +0.740072
  positive_logit = 22/24
  mean_effect_rank = +15.375

restore top_hidden_delta:
  mean_effect_logit = +1.045898
  positive_logit = 20/24
  mean_effect_rank = +30.375

restore answer_adjacent_text:
  mean_effect_logit = +0.438639
  positive_logit = 21/24
  mean_effect_rank = +20.708333

restore top_hidden_delta_plus_answer_adjacent:
  mean_effect_logit = +1.636556
  positive_logit = 21/24
  mean_effect_rank = +42.083333
```

读法：

```text
LLaVA 的 visual/evidence-region positions 本身就有可见 restore 信号。
但组合 answer-adjacent 后最强，说明视觉信号可能需要在答案附近位置汇聚后才最直接影响 target answer。
```

## 9. 预期与实际偏差

预期：

```text
如果 Stage 2I decoded bridge 不是偶然，hidden-position restore 应继续强于 controls；
corrupt 方向可能更弱，因为 clean hidden state 被破坏不一定线性影响输出。
```

实际：

```text
两个模型的 restore 和 corrupt 都稳定为正，并且 best group 在 paired bootstrap 中强于 random_mean / low_delta controls。
LLaVA 比 Stage 2H 更强：不仅 decoded bridge 不再 underpowered，hidden-position 双向定位也在 12 个样本上稳定。
```

主要偏差：

```text
best group 不是 pure visual group，而是 top_hidden_delta_plus_answer_adjacent。
这要求正文把跨模型结果写成 hidden-state bridge / evidence-to-answer convergence，
而不是写成已经定位到纯视觉 source route。
```

## 10. 结论

Stage 2I-3 支持：

```text
Qwen 和 LLaVA 在 selected-12 localized samples 上都有 bidirectional hidden-position causal localization。
同一组 source-like hidden positions 在 union→clean restore 与 clean→union corrupt 两个方向上都强于 random/low-delta controls。
这加强了跨模型 auxiliary evidence：不只是 Gemma 或单个 3-case smoke 才有 evidence-sensitive answer bridge。
```

保守主表述：

```text
Cross-model evidence now supports multi-case, bidirectional hidden-state bridge localization in Qwen and LLaVA.
```

必须保留的降级：

```text
This remains hidden-state-level localization, not CLT feature-level source tracing or Gemma-style source-control causal route replication.
```

## 11. 下一步

最合理的下一步是做更强的 control，而不是盲目继续扩大：

```text
1. activation-matched position controls:
   选取 hidden activation norm / delta norm 匹配但 mask-insensitive 的 positions，
   比 random/low-delta control 更强。

2. typed analysis:
   按 symbol_text_reading / visual_readout 分开读 Qwen/LLaVA bridge。

3. compact case panel:
   选 2-3 个 Qwen/LLaVA 最清楚的 case，做图文主图，展示 clean / union / bridge 生成变化与 hidden patch rank restore。
```

