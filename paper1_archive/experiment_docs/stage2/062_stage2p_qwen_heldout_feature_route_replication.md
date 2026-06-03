# Stage 2P-1：Qwen Heldout Feature/Route Replication

## 1. 目的

验证 Stage 2O 的 Qwen feature restoration 与 approximate source-control probe 是否能在 Stage 2O 未使用的 heldout prompt-runs 上复现。

## 2. 输入

```text
Qwen heldout prompt-runs = 24
sample 来源 = Stage 2N all52
排除 = Stage 2O 已使用的 10 个 sample_id
选择规则 = union_mask restore source_minus_random_logit 从高到低

模型：
  Qwen2.5-VL-7B-Instruct
  layer = 26
  CLT = KokosDev/qwen2p5vl-7b-clt

mask conditions:
  answer_mask
  union_mask
```

排除的 Stage 2O samples：

```text
okvqa_val_136595
okvqa_val_1593205
okvqa_val_2496585
okvqa_val_2683965
okvqa_val_343215
okvqa_val_3608785
okvqa_val_3918255
okvqa_val_4157235
okvqa_val_4502065
okvqa_val_4739195
```

## 3. 方法

```text
1. 从 Stage 2N all52 里排除 Stage 2O 用过的 sample_id；
2. 选择 Qwen hidden bridge 最强的 24 个 heldout prompt-runs；
3. 跑 answer_mask 与 union_mask 的 attribution-weighted feature bridge；
4. 跑 answer_mask / union_mask 的 approximate source-control probe；
5. 记录 shifted-mask 与 wrong-target controls。
```

## 4. 输出

```text
doc/experiments/stage2/cross_model/stage2p_qwen_feature_answer.csv/json
doc/experiments/stage2/cross_model/stage2p_qwen_feature_union.csv/json
doc/experiments/stage2/cross_model/stage2p_qwen_feature_bridge_summary.csv
doc/experiments/stage2/cross_model/stage2p_qwen_feature_bridge_specificity.csv
doc/experiments/stage2/cross_model/stage2p_qwen_feature_bridge_decision.json

doc/experiments/stage2/cross_model/stage2p_qwen_source_control_probe.csv/json
doc/experiments/stage2/cross_model/stage2p_qwen_source_control_summary.csv
doc/experiments/stage2/cross_model/stage2p_qwen_source_control_specificity.csv
doc/experiments/stage2/cross_model/stage2p_qwen_source_control_mask_specificity.csv
doc/experiments/stage2/cross_model/stage2p_qwen_source_control_decision.json
```

运行可用性：

```text
feature bridge answer_mask:
  usable_runs = 24
  rows = 720

feature bridge union_mask:
  usable_runs = 24
  rows = 720

source-control probe:
  usable_pairs = 48
  rows = 288
```

## 5. 结果

Feature bridge decision：

```text
Qwen status = feature_bridge_bidirectional_supported
```

Restore：

```text
answer_mask:
  n_rows = 24
  positive_logit_n = 24/24
  mean_logit_effect = +1.279297
  95% CI = [+0.867839, +1.729818]
  effect_status = stable_positive
  mean_rank_effect = +1052.041667
  mean_gap_closure = +0.212161

union_mask:
  n_rows = 24
  positive_logit_n = 23/24
  mean_logit_effect = +1.089844
  95% CI = [+0.630208, +1.617188]
  effect_status = stable_positive
  mean_rank_effect = +397.375
  mean_gap_closure = +0.169614

specificity:
  restore pooled n_rows = 48
  above_all_controls_n = 40/48
  mean_positive_control_count = 3.6875 / 4
```

Corrupt：

```text
answer_mask:
  n_rows = 24
  positive_logit_n = 23/24
  mean_logit_effect = +0.747396
  95% CI = [+0.434896, +1.132812]
  effect_status = stable_positive
  mean_rank_effect = +1.75
  mean_gap_closure = +0.122925

union_mask:
  n_rows = 24
  positive_logit_n = 20/24
  mean_logit_effect = +0.721354
  95% CI = [+0.348958, +1.161458]
  effect_status = stable_positive
  mean_rank_effect = +1.583333
  mean_gap_closure = +0.108007

specificity:
  corrupt pooled n_rows = 48
  above_all_controls_n = 32/48
  mean_positive_control_count = 3.270833 / 4
```

Source-control decision：

```text
Qwen status = approximate_source_control_route_supported
```

Aggregate：

```text
source_control_restore:
  n = 48
  positive_n = 41/48
  mean source_minus_control_logit = +0.476888
  95% CI = [+0.305664, +0.683268]
  status = stable_positive

source_control_zeroing:
  n = 48
  positive_n = 48/48
  mean source_minus_control_logit = +0.863281
  95% CI = [+0.713542, +1.022135]
  status = stable_positive

real_minus_shuffled:
  n = 48
  positive_n = 43/48
  mean = +0.455404
  95% CI = [+0.283203, +0.659180]
  status = stable_positive

correct_minus_wrong:
  n = 4
  positive_n = 4/4
  mean = +0.669886
  95% CI = [+0.452488, +0.887284]
  status = stable_positive
```

By mask condition：

```text
answer_mask restore:
  n = 24
  positive_n = 21/24
  mean source_minus_control_logit = +0.494141

answer_mask zeroing:
  n = 24
  positive_n = 24/24
  mean source_minus_control_logit = +0.867188

union_mask restore:
  n = 24
  positive_n = 20/24
  mean source_minus_control_logit = +0.459635

union_mask zeroing:
  n = 24
  positive_n = 24/24
  mean source_minus_control_logit = +0.859375
```

## 6. 结论

Stage 2P-1 明确加固了 Qwen 的跨模型 feature/source-control 证据：

```text
Qwen heldout feature bridge: bidirectional supported
Qwen heldout approximate source-control probe: supported
```

相对 Stage 2O 的推进：

```text
Stage 2O:
  Qwen feature bridge = one-direction supported

Stage 2P:
  Qwen heldout feature bridge = bidirectional supported
```

可以写：

```text
Qwen provides heldout-supported feature-level and approximate source-control auxiliary evidence.
```

仍不能写：

```text
Qwen fully replicates Gemma source tracing.
```

原因：

```text
Stage 2P 的 source/control 是 attribution-weighted probing 构造的 approximate pair，
不是完整 graph tracing 得到的 source node 与 nearest non-source control。
```
