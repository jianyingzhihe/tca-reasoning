# Stage 2N-2：Stricter Controls and Mask-Condition Analysis

## 1. 目的

Stage 2N-2 检查两个更严格的问题：

```text
1. hidden bridge 是否对 answer_mask 和 union_mask 都成立？
2. union_mask 下的 hidden bridge 是否 target-specific 和 evidence-location-specific？
```

这里的 stricter controls 仍然是 hidden-state-level controls，不是 feature-level 或 source-route controls。

## 2. Mask Condition

Stage 2N 主实验同时跑：

```text
answer_mask: 只遮挡人工标注的最小核心答案证据区域。
union_mask: 遮挡 answer + relate 的联合证据区域。
```

Mask condition summary：

```text
Qwen answer_mask:
  source_minus_random = +3.834926, stable_positive

Qwen union_mask:
  source_minus_random = +3.528669, stable_positive

LLaVA answer_mask:
  source_minus_random = +0.920284, stable_positive

LLaVA union_mask:
  source_minus_random = +0.808136, stable_positive
```

读法：

```text
answer_mask 和 union_mask 在两个模型上都成立。
这比只用 union_mask 更强，因为 answer_mask 更接近最小核心证据区域。
```

## 3. Wrong-Target Control

目的：

```text
验证 hidden patch 主要恢复/损伤正确 target，而不是任意 answer token。
```

方法：

```text
在 union_mask 条件下，固定 source-like group = top_hidden_delta_plus_answer_adjacent。
比较 correct target effect 与 wrong target effect。
```

输出：

```text
doc/experiments/stage2/cross_model/stage2n_wrong_target_summary.csv
doc/experiments/stage2/cross_model/stage2n_wrong_target_decision.json
```

结果：

```text
Qwen corrupt:
  correct_effect = +3.232701
  wrong_effect = +0.011370
  correct_minus_wrong = +3.221331
  status = stable_correct_gt_wrong

Qwen restore:
  correct_effect = +3.885324
  wrong_effect = +0.029663
  correct_minus_wrong = +3.855661
  status = stable_correct_gt_wrong

LLaVA corrupt:
  correct_effect = +0.795759
  wrong_effect = +0.107214
  correct_minus_wrong = +0.688545
  status = stable_correct_gt_wrong

LLaVA restore:
  correct_effect = +1.216239
  wrong_effect = +0.203217
  correct_minus_wrong = +1.013022
  status = stable_correct_gt_wrong
```

结论：

```text
两个模型在 restore/corrupt 两个方向都满足 correct > wrong。
这支持 target-specific hidden bridge，而不是任意 token-level 分布移动。
```

## 4. Mask-Shuffled Control

目的：

```text
验证 hidden bridge 是否来自真实证据区域遮挡，而不是任意遮挡或图像整体扰动。
```

方法：

```text
在 union_mask 条件下，比较真实 evidence mask 与同图平移后的 mask_shuffled。
source-like positions 固定从真实 evidence mask 的 clean-vs-mask hidden delta 得到。
```

输出：

```text
doc/experiments/stage2/cross_model/stage2n_mask_shuffled_summary.csv
doc/experiments/stage2/cross_model/stage2n_mask_shuffled_decision.json
```

结果：

```text
Qwen corrupt:
  real_effect = +3.232701
  shuffled_effect = +0.515625
  real_minus_shuffled = +2.717076
  status = stable_real_gt_shuffled

Qwen restore:
  real_effect = +3.885324
  shuffled_effect = +0.748326
  real_minus_shuffled = +3.136998
  status = stable_real_gt_shuffled

LLaVA corrupt:
  real_effect = +0.795759
  shuffled_effect = +0.105573
  real_minus_shuffled = +0.690186
  status = stable_real_gt_shuffled

LLaVA restore:
  real_effect = +1.216239
  shuffled_effect = +0.102120
  real_minus_shuffled = +1.114118
  status = stable_real_gt_shuffled
```

结论：

```text
两个模型在 restore/corrupt 两个方向都满足 real evidence mask > shuffled mask。
这支持 evidence-location specificity。
```

## 5. Decomposition

Stage 2N 同时保留：

```text
top_hidden_delta: visual source-like positions only
answer_adjacent_text: answer-adjacent text positions only
top_hidden_delta_plus_answer_adjacent: visual + answer-adjacent combo
```

读法：

```text
主结果继续以 combo group 为主。
这和 Stage 2L 的 evidence-to-answer bridge 解释一致：视觉证据信号可能需要在 answer-adjacent token 附近汇聚后影响 target answer。
```

## 6. 关键限制

```text
wrong-target 和 mask-shuffled controls 本轮只在 union_mask 条件下运行。
answer_mask 已在 matched/random hidden controls 下成立，但没有额外单独跑 answer-mask 的 wrong-target / shuffled controls。
```

因此最强表述应为：

```text
answer_mask and union_mask both support hidden bridge replication; union_mask additionally passes wrong-target and mask-shuffled stricter controls.
```

不应写成：

```text
answer_mask has passed every stricter negative control.
```
