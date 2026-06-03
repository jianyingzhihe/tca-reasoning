# Stage 2O-1：Attribution-Weighted Multi-Feature Bridge

## 1. 目的

检查 Stage 2M 的 feature-level bridge 失败是否只是因为 feature 选择太粗。

Stage 2M 只按 `clean-mask drop` 选 feature；Stage 2O-1 改为选同时满足以下条件的 feature：

```text
1. clean activation 高；
2. evidence mask 后 activation/drop 发生变化；
3. decoder vector 对 target answer logit direction 有正贡献。
```

## 2. 输入

```text
Qwen: Stage 2N strongest 12 prompt-runs, layer 26
LLaVA: Stage 2N strongest 8 prompt-runs, layer 15
mask_condition = union_mask
position_groups:
  top_hidden_delta_plus_answer_adjacent
  top_hidden_delta
  answer_adjacent_text
```

## 3. 方法

Feature score：

```text
feature_score =
  ReLU(weighted_clean_minus_mask_feature_drop)
  * ReLU(decoder_vector · target_unembedding_vector)
```

干预：

```text
restore: masked run + selected feature clean-minus-mask decoder contribution
corrupt: clean run - selected feature clean-minus-mask decoder contribution
```

Controls：

```text
activation_matched_topk
drop_matched_topk
attribution_matched_mask_insensitive_topk
random_active_topk
```

## 4. 输出

```text
doc/experiments/stage2/cross_model/stage2o_qwen_attribution_feature_bridge.csv
doc/experiments/stage2/cross_model/stage2o_qwen_attribution_feature_bridge.json
doc/experiments/stage2/cross_model/stage2o_llava_attribution_feature_bridge.csv
doc/experiments/stage2/cross_model/stage2o_llava_attribution_feature_bridge.json
doc/experiments/stage2/cross_model/stage2o_feature_bridge_summary.csv
doc/experiments/stage2/cross_model/stage2o_feature_bridge_specificity.csv
doc/experiments/stage2/cross_model/stage2o_feature_bridge_decision.json
```

运行可用性：

```text
Qwen:
  requested prompt-runs = 12
  usable_runs = 12
  rows = 360

LLaVA:
  requested prompt-runs = 8
  usable_runs = 8
  rows = 240
```

## 5. 结果

总体判断：

```text
overall_status = partial_or_model_specific_feature_bridge_support
```

Qwen：

```text
status = feature_bridge_one_direction_supported
```

主位置组 `top_hidden_delta_plus_answer_adjacent`：

```text
restore:
  n_rows = 12
  positive_logit_n = 11/12
  mean_logit_effect = +1.057292
  95% CI = [+0.557292, +1.572917]
  effect_status = stable_positive
  mean_rank_effect = +126.666667
  above_all_controls_n = 10/12
  mean_positive_control_count = 3.833333 / 4

corrupt:
  n_rows = 12
  positive_logit_n = 10/12
  mean_logit_effect = +0.447917
  95% CI = [+0.223958, +0.708333]
  effect_status = stable_positive
  mean_rank_effect = +0.166667
  above_all_controls_n = 6/12
  mean_positive_control_count = 3.083333 / 4
```

解释：

```text
Qwen 的 restore 方向满足 Stage 2O 成功标准：
evidence_attribution_topk 不只自身为 stable_positive，也在 10/12 prompt-runs 中强于全部 matched controls。

Qwen 的 corrupt 方向自身 logit effect 稳定为正，但只在 6/12 prompt-runs 中强于全部 controls。
因此写成 one-direction feature bridge support，而不是 bidirectional full feature bridge。
```

LLaVA：

```text
status = feature_bridge_partial_or_weak
```

主位置组 `top_hidden_delta_plus_answer_adjacent`：

```text
restore:
  n_rows = 8
  positive_logit_n = 3/8
  mean_logit_effect = -0.002930
  95% CI = [-0.012695, +0.005859]
  effect_status = not_positive
  above_all_controls_n = 0/8

corrupt:
  n_rows = 8
  positive_logit_n = 4/8
  mean_logit_effect = +0.003906
  95% CI = [-0.018555, +0.028320]
  effect_status = weak_or_heterogeneous_positive
  above_all_controls_n = 2/8
```

解释：

```text
LLaVA 没有形成 feature-level causal bridge。
它最多支持 very weak / diagnostic feature signal，不能写成 feature bridge replication。
```

## 6. 结论

可以写：

```text
Stage 2O-1 provides model-specific feature-level bridge support for Qwen in the restoration direction.
```

中文：

```text
在 Qwen 上，使用 attribution-weighted feature 选择后，masked→clean feature restoration 能稳定恢复 target logit/rank，并且多数 prompt-runs 强于 matched feature controls。
这比 Stage 2M 的 top-drop feature smoke 明显更强。
```

不能写：

```text
Qwen 已经双向复现 feature-level causal bridge。
LLaVA 已经复现 feature-level causal bridge。
Qwen/LLaVA 已经复现 Gemma-style source-control routes。
```

预期与实际偏差：

```text
预期：
  attribution-weighted feature selection 应该比 Stage 2M 的 drop-only top-k 更接近 causal feature group。

实际：
  Qwen restore 明显成立，corrupt 只 partial；
  LLaVA 仍未成立。

解释：
  Qwen 的 CLT feature 与 hidden bridge 更对齐；
  LLaVA 的 hidden bridge 虽然存在，但当前 CLT feature 分解仍不能稳定承载该 bridge。
```
