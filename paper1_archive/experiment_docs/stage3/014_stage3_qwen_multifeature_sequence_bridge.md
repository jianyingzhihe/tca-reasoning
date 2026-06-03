# 014 Stage3 Qwen Multi-Feature Sequence Bridge

## 目的

本实验检查：在 Qwen2.5-VL 上，把单 feature 干预扩展成 attribution-weighted multi-feature group 后，是否能把已有 first-token/rank bridge 推进到更接近生成行为的 sequence-level bridge。

核心问题：

```text
source restore 是否比 matched controls 更能恢复目标答案序列 logprob？
source corruption 是否比 matched controls 更能损伤目标答案序列 logprob？
如果 sequence score 有改善，decoded answer 是否也回到 clean/target？
```

## 输入

manifest：

```text
doc/experiments/stage3/cross_model/stage3_qwen_generation_bridge_v2_manifest.csv
```

筛选后共 7 个 prompt-runs：

| 资产 | prompt-runs | 说明 |
|---|---:|---|
| Qwen2.5-VL-CLT | 4 | behavior-aware 条件下可用 rows 较多 |
| Qwen2.5-VL-PLT | 3 | PLT 信号较弱，严格条件下可用 rows 更少 |

主要 case 包括：

```text
okvqa_val_3959785 / kuwait airway
okvqa_val_4502065 / tortoise
okvqa_val_4739195 / spanish
```

## 输出

远端运行已完成并拉回：

```text
stage3_qwen2p5vl_plt_generation_bridge_v2.csv/json
stage3_qwen2p5vl_clt_generation_bridge_v2.csv/json
```

本地分析已完成：

```text
stage3_qwen_generation_bridge_v2_summary.csv
stage3_qwen_generation_bridge_v2_case_table.csv
stage3_qwen_generation_bridge_v2_decision.json
```

脚本：

```text
scripts/local/build_stage3_qwen_generation_bridge_v2_manifest.py
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_stage3_qwen_multifeature_sequence_bridge.py
scripts/local/run_stage3_qwen_generation_bridge_v2_remote.py
scripts/local/analyze_stage3_qwen_generation_bridge_v2.py
```

## 方法

每个 selected row 在 `top1/top4/top8/top16/top32` 上分别构造 source group 和 controls：

```text
source group:
  evidence_attribution_topk

controls:
  activation_matched_topk
  drop_matched_topk
  attribution_matched_mask_insensitive_topk
  random_active_topk
```

feature score 仍遵循 Stage3 feature bridge 的核心逻辑：

```text
feature_score =
  positive(clean_activation - mask_activation)
  * positive(decoder_vector · target_logit_direction)
  * position_weight
```

分析时计算：

```text
source_sequence_effect
control_sequence_mean
source_minus_control_sequence
source_first_token_effect
source_minus_control_first_token
source_first_rank_effect
source_minus_control_first_rank
source_decoded_to_clean
```

`source_minus_control_sequence > 0` 表示 source group 比 matched controls 更符合预期方向。restore 中表示更强恢复；corrupt 中表示更强损伤。

## 结果

### Qwen2.5-VL-CLT

CLT 的 restore 方向有明显 sequence-score 信号：

| topK | restore source-control sequence | 95% CI | positive |
|---:|---:|---|---:|
| 1 | +0.117 | [+0.084, +0.174] | 4/4 |
| 4 | +0.105 | [+0.022, +0.229] | 4/4 |
| 8 | +0.223 | [-0.054, +0.530] | 3/4 |
| 16 | +0.173 | [-0.084, +0.436] | 3/4 |
| 32 | +0.412 | [+0.029, +0.637] | 3/4 |

但 CLT 的 corrupt 方向没有同步成立：

| topK | corrupt source-control sequence | 95% CI | positive |
|---:|---:|---|---:|
| 1 | -0.074 | [-0.205, +0.037] | 1/4 |
| 4 | -0.139 | [-0.435, +0.078] | 1/4 |
| 8 | -0.089 | [-0.220, +0.041] | 2/4 |
| 16 | -0.182 | [-0.379, +0.012] | 1/4 |
| 32 | -0.133 | [-0.349, +0.083] | 2/4 |

first-token/rank 层仍有更强的正向信号，尤其 restore top8/top16/top32 的 first-token source-control gap 为正。但由于 sequence bridge 需要 restore 与 corrupt 两个方向同时成立，CLT 本轮判定为：

```text
first_token_only
```

这不是说 CLT 没有 sequence 相关信号，而是说它目前只有单向 restore sequence support，不能升级为双向 sequence-level causal bridge。

### Qwen2.5-VL-PLT

PLT 的 sequence-score 效应明显弱于 CLT。只有 top1 同时满足 restore/corrupt mean 为正：

| topK | restore source-control sequence | corrupt source-control sequence | 判定 |
|---:|---:|---:|---|
| 1 | +0.004 | +0.007 | partial_sequence_bridge |
| 4 | -0.041 | -0.016 | first_token_only |
| 8 | -0.041 | -0.088 | first_token_only |
| 16 | -0.001 | -0.335 | first_token_only |
| 32 | -0.080 | -0.593 | first_token_only |

top1 的两个方向都是微弱正值，但 CI 都跨 0：

```text
restore CI: [-0.061, +0.038]
corrupt CI: [-0.061, +0.046]
```

因此 PLT 本轮判定为：

```text
partial_sequence_bridge
```

这只能说明 PLT 在最小 top1 feature group 上有极弱的 sequence-level 方向性提示，不能写成稳定 sequence bridge。

### Decoded Answer

最重要的行为层结果是：restore 没有稳定把 decoded answer 拉回 clean answer。

| 资产 | restore rows per topK | source_decoded_to_clean |
|---|---:|---:|
| Qwen2.5-VL-CLT | 4 | 0/4 |
| Qwen2.5-VL-PLT | 3 | 0/3 |

这个结果在所有 topK 上都成立。也就是说，即便 CLT 在 sequence logprob 上有 restore 信号，短 greedy decoded answer 仍没有回到 clean/target。

## 预期与实际偏差

预期：多 feature restore 可能比单 feature restore 更容易改变 target answer sequence score，甚至带来少量 decoded answer 恢复。

实际：CLT 的 restore sequence score 确实更强，top1/top4/top32 的 bootstrap CI 不跨 0；但 corrupt 方向没有成立。PLT 只在 top1 有非常弱的双向 mean positive，且 CI 跨 0。decoded answer 仍然没有恢复。

主要偏差解释：

```text
1. Qwen decoded generation 对局部 feature patch 的阈值更高。
2. answer sequence logprob 比 greedy decoded answer 更敏感，因此能先看到弱行为层信号。
3. CLT restore 信号强但 corrupt 不成立，说明该 feature group 能补回部分答案倾向，
   但还不足以证明它是必要且特异的生成路径。
4. PLT top1 的 partial 信号太小，不能作为强结论。
```

## 结论

Stage3-13 加固了一个有限结论：

```text
Qwen2.5-VL-CLT 的 evidence-attribution multi-feature restore
能比 matched controls 更稳定提高 target answer sequence logprob；
但 corrupt 方向不成立，因此不能写成 supported sequence-level causal bridge。
```

同时也给出一个更保守的 PLT 结论：

```text
Qwen2.5-VL-PLT 只有 top1 的 partial sequence bridge hint；
主行为桥接仍应写在 first-token/rank 层。
```

本轮仍不能写：

```text
Qwen2.5-VL 已经完成 decoded generation-level causal bridge。
Qwen2.5-VL 已经完整复现 Gemma-style source tracing route。
D_visual_only 比 B_direct 更好。
```
