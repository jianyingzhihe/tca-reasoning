# Stage 2M-3：Feature-Level Causal Bridge Smoke

## 1. 目的

本实验尝试把 Stage 2M Tier 1 的 hidden-state bridge 下沉到 CLT feature 层：

```text
如果 evidence-sensitive CLT features 真正承载 hidden bridge，那么用这些 features 做 clean→masked corruption 和 masked→clean restoration，应当比 matched feature controls 更能损伤或恢复 target logit/rank。
```

这一步仍然不是 source tracing，也不是 Gemma-style source-control route replication。

## 2. 输入

样本：

```text
沿用 Stage 2M-2 passing cases
Qwen: 8 unique samples, 16 prompt-runs
LLaVA: 6 unique samples, 12 prompt-runs
```

模型与 feature 资产：

```text
Qwen: Qwen2.5-VL-7B-Instruct, layer 26, KokosDev/qwen2p5vl-7b-clt
LLaVA: LLaVA-1.5-7B, layer 15, KokosDev/llava15-7b-clt
position group: top_hidden_delta_plus_answer_adjacent
top_k_features = 4
scale = 1.0
```

## 3. 方法

特征选择：

```text
在 top_hidden_delta_plus_answer_adjacent positions 上编码 clean / union_mask hidden states。
计算 feature_drop = clean_feature_activation - union_feature_activation。
选择 drop 最大的 active features 作为 evidence_topk。
```

feature controls：

```text
activation_matched_topk：clean activation 接近 evidence features 的控制特征。
drop_matched_topk：feature drop 接近 evidence features 的控制特征。
mask_insensitive_topk：clean active 但 clean-mask drop 很小的控制特征。
random_topk：同层 active feature 随机控制。
```

干预：

```text
masked→clean restoration:
  在 union_mask run 中加回 evidence/control feature 的 clean-minus-mask decoder contribution。

clean→masked corruption:
  在 clean run 中减去 evidence/control feature 的 clean-minus-mask decoder contribution。
```

指标：

```text
restore: logit_restore_vs_union, rank_restore_vs_union
corrupt: logit_damage_vs_clean, rank_damage_vs_clean
specificity: evidence_topk - matched/control feature groups
```

## 4. 输出

```text
doc/experiments/stage2/cross_model/stage2m_qwen_feature_bridge.csv/json
doc/experiments/stage2/cross_model/stage2m_llava_feature_bridge.csv/json
doc/experiments/stage2/cross_model/stage2m_feature_bridge_summary.csv
doc/experiments/stage2/cross_model/stage2m_feature_bridge_specificity.csv
doc/experiments/stage2/cross_model/stage2m_feature_bridge_decision.json
```

## 5. 可用性

```text
Qwen usable_runs = 16/16, rows = 192
LLaVA usable_runs = 12/12, rows = 144
```

## 6. 结果

Decision：

```text
overall_status = feature_bridge_not_established
Qwen status = feature_bridge_not_established
LLaVA status = feature_bridge_not_established
```

Qwen evidence_topk：

```text
corrupt:
  positive_logit_n = 6/16
  positive_rank_n = 4/16
  mean_logit_effect = -0.011719
  mean_rank_effect = +0.3125

restore:
  positive_logit_n = 7/16
  positive_rank_n = 8/16
  mean_logit_effect = +0.011719
  mean_rank_effect = +140.5625
```

Qwen matched/control 对比：

```text
restore evidence_minus_activation mean = +0.064453, positive_n = 9/16
restore evidence_minus_drop mean = +0.064453, positive_n = 8/16
restore evidence_minus_mask_insensitive mean = -0.044922, positive_n = 7/16
restore evidence_minus_random mean = -0.046875, positive_n = 5/16

corrupt evidence-minus-control 没有稳定超过 matched/control feature groups。
```

LLaVA evidence_topk：

```text
corrupt:
  positive_logit_n = 2/12
  positive_rank_n = 0/12
  mean_logit_effect = -0.016276
  mean_rank_effect = -0.083333

restore:
  positive_logit_n = 2/12
  positive_rank_n = 1/12
  mean_logit_effect = -0.010091
  mean_rank_effect = 0.0
```

LLaVA matched/control 对比：

```text
restore evidence_minus_activation mean = -0.013346, positive_n = 3/12
restore evidence_minus_drop mean = -0.006185, positive_n = 2/12
restore evidence_minus_mask_insensitive mean = -0.011393, positive_n = 3/12
restore evidence_minus_random mean = -0.013346, positive_n = 1/12

corrupt evidence-minus-control 也没有稳定超过 matched/control feature groups。
```

## 7. 解释

Qwen 有一个值得保留的弱信号：

```text
restore 的 rank effect 在 evidence_topk 上为正，mean_rank_effect = +140.5625。
```

但这个信号不能升级为 feature-level causal bridge：

```text
logit effect 很小；
restore 只稳定强于 activation-matched controls，不稳定强于 drop-matched / mask-insensitive / random controls；
corrupt 方向不成立；
因此不满足 Tier 2 成功标准。
```

LLaVA 在 feature 层没有形成可用信号：

```text
evidence_topk 的 restore/corrupt logit 与 rank 都不稳定，且不强于 controls。
```

## 8. 结论边界

可写：

```text
Stage 2M Tier 2 does not establish feature-level causal bridge. The cross-model replication currently remains strongest at hidden-state level.
```

不可写：

```text
Qwen/LLaVA CLT features reproduce Gemma source routes.
Qwen/LLaVA feature-level causal bridge is replicated.
```
