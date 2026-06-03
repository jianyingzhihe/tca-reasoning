# 011 Stage3 First-Token / Rank Behavior Bridge

## 目的

把 Stage3 的 feature/source-control 结果和答案首 token 行为联系起来，检查内部干预是否不仅改变 feature/source-control 指标，也会改变目标答案第一个 token 的 logit 或 rank。

这里仍然不是 decoded generation：

```text
first-token/rank bridge:
  检查目标答案首 token 的 logit/rank 是否按预期变化。

decoded generation bridge:
  检查短 greedy generation 的最终文本答案是否改变。
```

本实验只完成 first-token/rank bridge。

## 输入

```text
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_plt_source_control.csv
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_clt_source_control.csv
doc/experiments/stage3/cross_model/stage3_llava15_clt_source_control.csv
```

## 输出

```text
doc/experiments/stage3/cross_model/stage3_behavior_first_token_comparisons.csv
doc/experiments/stage3/cross_model/stage3_behavior_first_token_summary.csv
doc/experiments/stage3/cross_model/stage3_behavior_first_token_decision.json
```

本地脚本：

```text
scripts/local/analyze_stage3_behavior_first_token_bridge.py
```

## 方法

复用 Stage3 source-control probe 中已经保存的字段：

```text
effect_logit:
  干预对目标答案首 token logit 的影响。

effect_rank:
  干预对目标答案首 token rank 的影响。
  rank 越小越好，因此脚本中已经按 restore/zeroing 方向转成“正数代表方向正确”。

source_minus_control:
  source feature 干预效果减 matched control 干预效果。

correct_minus_wrong:
  正确目标答案 token 的效果减错误目标答案 token 的效果。
```

只把以下组合作为 primary behavior bridge：

```text
mask_variant = real_mask
mask_condition = answer_mask, union_mask
intervention = restore, zeroing
position_group = top_hidden_delta_plus_answer_adjacent
```

专有名词解释：

```text
restore:
  在被遮挡运行中补回 clean run 的 source feature contribution，看目标答案 token 是否恢复。

zeroing:
  在 clean run 中移除 source feature contribution，看目标答案 token 是否受损。

matched control:
  与 source feature 激活或 attribution 相近，但不应具有同样证据敏感性的对照 feature。

gap closure:
  干预弥合 clean 与 mask 之间目标 logit gap 的比例。
```

## 结果

Primary first-token/rank summary：

| 资产 | mask | intervention | source-control logit | 95% CI | 正向数 | source-control rank |
|---|---|---|---:|---:|---:|---:|
| Qwen2.5-VL-PLT | answer | restore | +0.048 | [+0.012, +0.092] | 15/38 | -1.526 |
| Qwen2.5-VL-PLT | answer | zeroing | +0.232 | [+0.151, +0.344] | 35/38 | +0.868 |
| Qwen2.5-VL-PLT | union | restore | +0.056 | [+0.035, +0.075] | 22/37 | +29.946 |
| Qwen2.5-VL-PLT | union | zeroing | +0.179 | [+0.139, +0.221] | 31/37 | +1.189 |
| Qwen2.5-VL-CLT | answer | restore | +0.186 | [+0.128, +0.245] | 34/48 | +141.458 |
| Qwen2.5-VL-CLT | answer | zeroing | +0.750 | [+0.626, +0.887] | 48/48 | +7.479 |
| Qwen2.5-VL-CLT | union | restore | +0.199 | [+0.122, +0.283] | 34/48 | +77.375 |
| Qwen2.5-VL-CLT | union | zeroing | +0.714 | [+0.574, +0.870] | 48/48 | +7.938 |
| LLaVA-CLT | answer | restore | -0.000 | [-0.004, +0.004] | 13/45 | -0.089 |
| LLaVA-CLT | answer | zeroing | +0.051 | [+0.037, +0.066] | 39/45 | +0.644 |
| LLaVA-CLT | union | restore | +0.002 | [-0.002, +0.007] | 16/46 | +0.196 |
| LLaVA-CLT | union | zeroing | +0.033 | [+0.022, +0.046] | 37/46 | +0.370 |

Decision JSON 给出的状态：

| 资产 | 判定 |
|---|---|
| Qwen2.5-VL-CLT | supported_first_token_rank_bridge |
| Qwen2.5-VL-PLT | partial_first_token_rank_bridge |
| LLaVA-CLT | partial_first_token_rank_bridge |

## 预期与实际偏差

预期 Qwen-CLT 会最强，实际成立。Qwen-PLT 也有稳定 logit 方向，union restore 和两个 zeroing 项也有正向 rank；answer restore 的 logit 为正但 rank 均值略负，因此判为 partial，而不是 full supported。

预期 LLaVA-CLT 较弱，实际也成立。LLaVA zeroing 对 first-token 有小而稳定的正效应，但 restoration 不稳定，因此不能写 generation-level 或 route-level strong bridge。

## 结论

当前可以写：

```text
Qwen2.5-VL 的 source/control 差异可以桥接到目标答案首 token logit/rank；
CLT 版本最强，PLT 版本较弱但不是零信号。
```

当前不能写：

```text
Qwen/LLaVA 已经完成 decoded generation bridge。
Qwen/LLaVA 已经完整复现 Gemma-style source tracing route。
LLaVA 已经稳定完成 feature-level route localization。
```

下一步最自然的是：只对 Qwen2.5-VL-CLT 与 Qwen2.5-VL-PLT 中 source-control/first-token 同时通过的 rows，跑短 greedy decoded generation；如果 decoded answer 不变，就继续把行为桥接限制在 first-token/rank 层。
