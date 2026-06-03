# Stage 2O-2：Cross-Model Source-Control Route Probe

## 1. 目的

在 Qwen/LLaVA 中构造近似 source-control feature pair，检查是否能靠近 Gemma 主线的 source-control route evidence。

注意：

```text
本实验不是完整 source tracing。
它只能支持 approximate source-control route probe。
```

## 2. 输入

```text
Qwen: Stage 2N strongest 12 prompt-runs, layer 26
LLaVA: Stage 2N strongest 8 prompt-runs, layer 15
mask_conditions:
  answer_mask
  union_mask
position_group:
  top_hidden_delta_plus_answer_adjacent
```

## 3. 方法

Source feature 选择：

```text
从 evidence_attribution_topk 中选 zeroing 后 target logit damage 最大且为正的 feature。
```

Matched control 选择：

```text
从 attribution_matched_mask_insensitive / activation_matched / drop_matched / random active 中，
选择 activation/drop/target contribution 与 source 最接近的 feature。
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
source > matched_control
correct target > wrong target
real evidence mask > shifted mask
answer_mask / union_mask 分开分析
```

## 4. 输出

```text
doc/experiments/stage2/cross_model/stage2o_qwen_source_control_probe.csv
doc/experiments/stage2/cross_model/stage2o_qwen_source_control_probe.json
doc/experiments/stage2/cross_model/stage2o_llava_source_control_probe.csv
doc/experiments/stage2/cross_model/stage2o_llava_source_control_probe.json
doc/experiments/stage2/cross_model/stage2o_source_control_summary.csv
doc/experiments/stage2/cross_model/stage2o_source_control_specificity.csv
doc/experiments/stage2/cross_model/stage2o_source_control_mask_specificity.csv
doc/experiments/stage2/cross_model/stage2o_source_control_decision.json
```

运行可用性：

```text
Qwen:
  usable_pairs = 24
  rows = 144

LLaVA:
  usable_pairs = 16
  rows = 96
```

## 5. 结果

总体判断：

```text
overall_status = model_specific_approx_source_control_support
```

Qwen：

```text
status = approximate_source_control_route_supported
```

核心指标：

```text
source_control_restore:
  n = 24
  positive_n = 19/24
  mean source_minus_control_logit = +0.395833
  95% CI = [+0.158854, +0.677083]
  status = stable_positive

source_control_zeroing:
  n = 24
  positive_n = 24/24
  mean source_minus_control_logit = +0.489583
  95% CI = [+0.333333, +0.661458]
  status = stable_positive

real_minus_shuffled:
  n = 24
  positive_n = 17/24
  mean = +0.377604
  95% CI = [+0.135417, +0.664062]
  status = stable_positive

correct_minus_wrong:
  n = 4
  positive_n = 4/4
  mean = +0.406169
  95% CI = [+0.382812, +0.435384]
  status = stable_positive
```

Qwen 按 mask condition：

```text
answer_mask restore:
  n = 12
  source_minus_control mean = +0.414062
  positive_n = 10/12

answer_mask zeroing:
  n = 12
  source_minus_control mean = +0.473958
  positive_n = 12/12

union_mask restore:
  n = 12
  source_minus_control mean = +0.377604
  positive_n = 9/12

union_mask zeroing:
  n = 12
  source_minus_control mean = +0.505208
  positive_n = 12/12
```

解释：

```text
Qwen 同时满足：
1. source feature 强于 matched control；
2. source zeroing 比 control zeroing 更伤 target；
3. source restore 比 control restore 更恢复 target；
4. real evidence mask 强于 shifted mask；
5. correct target 强于 wrong target。

因此可以写 approximate source-control route support。
但由于 source 是通过 attribution-weighted probing 构造的，不是完整 graph tracing 得到的 source node，所以不能写 full Gemma-style route replication。
```

LLaVA：

```text
status = source_control_probe_partial
```

核心指标：

```text
source_control_restore:
  n = 16
  positive_n = 4/16
  mean source_minus_control_logit = -0.000732
  95% CI = [-0.002930, +0.001465]
  status = not_positive

source_control_zeroing:
  n = 16
  positive_n = 12/16
  mean source_minus_control_logit = +0.012207
  95% CI = [+0.005371, +0.019043]
  status = stable_positive

real_minus_shuffled:
  n = 16
  positive_n = 7/16
  mean = +0.003174
  95% CI = [+0.000488, +0.005859]
  status = stable_positive

correct_minus_wrong:
  n = 4
  positive_n = 4/4
  mean = +0.015472
  95% CI = [+0.003295, +0.027649]
  status = stable_positive
```

解释：

```text
LLaVA 有 source-control-like zeroing 信号，也有 target/location specificity。
但 restore 方向不成立，因此不能写 approximate route support，只能写 partial diagnostic support。
```

## 6. 结论

可以写：

```text
Qwen shows approximate source-control route support at the CLT feature-probe level.
LLaVA shows partial source-control-like evidence, mainly in zeroing and target/location controls, but lacks restoration specificity.
```

中文：

```text
Qwen 已经从 hidden-state bridge 推进到 approximate feature source-control probe：
构造出的 source feature 比 matched control 更能损伤/恢复 target，并且通过 wrong-target 与 shifted-mask 对照。

LLaVA 仍停留在 partial：zeroing 有方向，target/location 对照有方向，但 restore 不成立。
```

不能写：

```text
Qwen 完整复现 Gemma-style source-control route。
LLaVA 复现 source-control route。
Qwen/LLaVA 的 source feature 是对象级语义节点。
```

对主线的影响：

```text
Gemma 仍然是完整 source tracing + node intervention 主模型。
Qwen 现在提供更强的跨模型机制辅助证据：不仅 hidden bridge 成立，feature-level restore 与 approximate source-control probe 也有支持。
LLaVA 继续作为 smaller-effect / heterogeneous 复现线。
```
