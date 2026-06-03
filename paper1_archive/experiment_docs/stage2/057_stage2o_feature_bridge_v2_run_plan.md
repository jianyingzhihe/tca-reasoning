# Stage 2O：Feature-Level Bridge 与 Source-Control Route Probe Run Plan

## 1. 目标

Stage 2N 已经支持：

```text
Qwen/LLaVA 在 hidden-state 层存在 heldout-replicated evidence-region-sensitive bridge。
```

Stage 2O 不再扩大 hidden-state 样本，而是检查两条更硬机制链：

```text
实验一：Attribution-weighted multi-feature bridge
实验二：Approximate source-control route probe
```

## 2. 结论边界

可以升级的结论：

```text
如果实验一成立：Qwen/LLaVA 至少存在 feature-level causal bridge 的辅助证据。
如果实验二成立：Qwen/LLaVA 至少存在 approximate source-control route support。
```

不能直接写：

```text
Qwen/LLaVA 完整复现 Gemma-style source-control causal routes。
Qwen/LLaVA 的 feature 就是对象级语义节点。
D_visual_only 比 B_direct 更好。
```

## 3. 实验一设计

Stage 2M 的 feature bridge 使用 `clean-mask drop top-k features`，失败原因可能是：

```text
被 mask 破坏的 feature 未必对 target answer logit 有正贡献。
```

Stage 2O 改用：

```text
feature_score =
  positive(clean_activation - mask_activation)
  * positive(decoder_vector · target_logit_direction)
  * position_weight
```

主 feature group：

```text
evidence_attribution_topk
```

Controls：

```text
activation_matched_topk
drop_matched_topk
attribution_matched_mask_insensitive_topk
random_active_topk
```

干预：

```text
masked→clean feature restoration
clean→masked feature corruption
```

## 4. 实验二设计

实验二尝试构造近似 source-control pair：

```text
source feature:
  evidence visual / answer-adjacent positions 中 attribution-weighted score 高，
  mask drop 为正，
  target attribution 为正，
  zeroing 后会损伤 target logit。

matched control:
  同层同 bucket，
  activation/drop/target contribution 尽量接近，
  但 mask-insensitive 或 attribution 较弱。
```

干预：

```text
source_zeroing
matched_control_zeroing
source_restore
matched_control_restore
```

对照：

```text
source > matched control
correct target > wrong target
real evidence mask > shifted mask
answer_mask 与 union_mask 分开记录
```

## 5. 输入

```text
候选来自 Stage 2N hidden bridge 最强 rows。

Qwen:
  union_mask restore source_minus_random top 12 prompt-runs
  layer = 26
  CLT = KokosDev/qwen2p5vl-7b-clt

LLaVA:
  union_mask restore source_minus_random top 8 prompt-runs
  layer = 15
  CLT = KokosDev/llava15-7b-clt
```

## 6. 输出

```text
doc/experiments/stage2/cross_model/stage2o_qwen_attribution_feature_bridge.csv/json
doc/experiments/stage2/cross_model/stage2o_llava_attribution_feature_bridge.csv/json
doc/experiments/stage2/cross_model/stage2o_feature_bridge_summary.csv
doc/experiments/stage2/cross_model/stage2o_feature_bridge_specificity.csv
doc/experiments/stage2/cross_model/stage2o_feature_bridge_decision.json

doc/experiments/stage2/cross_model/stage2o_qwen_source_control_probe.csv/json
doc/experiments/stage2/cross_model/stage2o_llava_source_control_probe.csv/json
doc/experiments/stage2/cross_model/stage2o_source_control_summary.csv
doc/experiments/stage2/cross_model/stage2o_source_control_specificity.csv
doc/experiments/stage2/cross_model/stage2o_source_control_mask_specificity.csv
doc/experiments/stage2/cross_model/stage2o_source_control_decision.json
```

## 7. 成功标准

Feature-level bridge：

```text
Qwen: evidence_attribution_topk 在至少一个方向稳定强于 matched controls。
LLaVA: 若 effect 更小但方向一致，写 partial。
```

Source-control route probe：

```text
source > matched_control
correct > wrong
real mask > shifted mask
answer_mask 或 union_mask 至少一个稳定
```

只有全部成立时，才写：

```text
approximate source-control route support
```

仍不写：

```text
Gemma-style source-control route fully replicated
```
