# 实验 032：Stage 2G Mask-to-Clean Feature Restoration Smoke

## 目的

本实验是 Stage 2G-5 的实际执行结果。

上一轮 `feature-direction dose/sign probe` 显示：

```text
Qwen/LLaVA 的 evidence-region-sensitive readout 成立；
但直接在 clean hidden 上 subtract/add evidence feature direction，
没有稳定建立 intervention replication。
```

因此本轮改用更自然的 restoration 设计：

```text
先看 union_mask 是否损伤 target answer；
再在 union_mask forward 中，
把 clean image 相比 union_mask 丢失的 evidence feature contribution patch 回去；
检查 target logit / target rank 是否恢复，并与 control feature restoration 比较。
```

这一步要回答的问题是：

```text
Qwen/LLaVA 中被 evidence mask 削弱的 readout features，
是否已经能作为 feature-level causal bridge 来恢复 target answer signal？
```

## 专有名词解释

`mask-to-clean restoration`：

```text
在 masked image 输入下运行模型；
但在某一层某些 token 位置，把 clean image 下更强、masked image 下更弱的 feature contribution 补回去。
如果补回 evidence feature 能恢复 target answer，就说明这些 feature 可能参与了答案行为。
```

`feature contribution`：

```text
这里用 CLT feature activation × decoder vector 近似表示某个 feature 对 hidden state 的贡献。
```

`evidence restore`：

```text
补回 clean - union_mask drop 最大的 evidence-sensitive features。
```

`control restore`：

```text
补回 clean activation 相近、但 evidence-mask drop 小的 control features。
```

## 输入

模型：

```text
Qwen2.5-VL-7B-Instruct:
  CLT = KokosDev/qwen2p5vl-7b-clt
  layer = 26
  bucket = image_marker_or_span

LLaVA-1.5-7B:
  CLT-like asset = KokosDev/llava15-7b-clt
  layer = 15
  bucket = image_token_span
```

样本：

```text
okvqa_val_2847255
okvqa_val_4157235
okvqa_val_3658865
```

prompt：

```text
B_direct
D_visual_only
```

每个模型：

```text
3 samples × 2 prompts = 6 usable runs
```

## 输出

原始输出：

```text
doc/experiments/stage2/cross_model/stage2g_qwen_feature_restoration_smoke.json
doc/experiments/stage2/cross_model/stage2g_qwen_feature_restoration_smoke.csv
doc/experiments/stage2/cross_model/stage2g_llava_feature_restoration_smoke.json
doc/experiments/stage2/cross_model/stage2g_llava_feature_restoration_smoke.csv
```

分析输出：

```text
doc/experiments/stage2/cross_model/stage2g_feature_restoration_baseline_gap.csv
doc/experiments/stage2/cross_model/stage2g_feature_restoration_summary.csv
doc/experiments/stage2/cross_model/stage2g_feature_restoration_specificity.csv
doc/experiments/stage2/cross_model/stage2g_feature_restoration_case_table.csv
doc/experiments/stage2/cross_model/stage2g_feature_restoration_decision.json
```

脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_cross_model_feature_restoration_smoke.py
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/analyze_stage2g_feature_restoration_smoke.py
scripts/local/run_stage2g_feature_restoration_remote.py
```

## 方法

### 1. 先确认 union_mask 是否真的损伤行为

对 clean 和 union_mask 都计算：

```text
target_logit
target_rank
top1_token
```

关键差值：

```text
clean_union_logit_gap = target_logit(clean) - target_logit(union_mask)
clean_union_rank_gap = target_rank(union_mask) - target_rank(clean)
```

如果 `clean_union_logit_gap > 0`，说明 union_mask 降低了 target logit。

如果 `clean_union_rank_gap > 0`，说明 union_mask 让 target rank 变差。

### 2. 选择 evidence/control features

在对应 bucket 上计算：

```text
feature_drop = clean_feature_activation - union_mask_feature_activation
```

选择：

```text
evidence_topk:
  feature_drop 最大的 top-k features

control_topk:
  clean activation 接近 evidence_topk，
  但 feature_drop 较小的 features
```

### 3. 在 union_mask forward 上做 restore

patch 定义：

```text
restore_patch = scale × feature_drop[feature_ids] × decoder_vectors[feature_ids]
union_hidden_restored = union_hidden + restore_patch
```

条件：

```text
clean
union_mask
evidence_top1_restore_s0.5 / s1.0 / s1.5
evidence_topk_restore_s0.5 / s1.0 / s1.5
control_topk_restore_s0.5 / s1.0 / s1.5
```

### 4. 恢复指标

```text
logit_restore_vs_union = target_logit(restored) - target_logit(union_mask)
rank_restore_vs_union = target_rank(union_mask) - target_rank(restored)
logit_gap_closure = logit_restore_vs_union / clean_union_logit_gap
```

解释：

```text
logit_restore_vs_union > 0 表示 target logit 被恢复；
rank_restore_vs_union > 0 表示 target rank 变好；
evidence_restore > control_restore 才能支持 feature-level specificity。
```

## 结果一：union_mask 本身确实造成强行为损伤

| 模型 | n runs | mean clean-union logit gap | positive logit gap | mean clean-union rank gap | positive rank gap | clean→union top1 changed |
|---|---:|---:|---:|---:|---:|---:|
| Qwen layer 26 | `6` | `+3.9844` | `6/6` | `+90.1667` | `4/6` | `4/6` |
| LLaVA layer 15 | `6` | `+2.9492` | `6/6` | `+54.0000` | `6/6` | `0/6` |

这个结果很重要：

```text
union_mask 不是无效 corruption；
它确实会显著降低 target answer logit，并常常让 target rank 变差。
```

因此如果 restoration 失败，不能解释成“mask 没有造成损伤”。更准确的解释是：

```text
当前 feature-level restoration 没有把 mask 造成的行为损伤救回来。
```

## 结果二：Qwen layer 26 restoration

关键条件：

| 条件 | mean logit restore | positive logit restore | mean rank restore | positive rank restore | mean gap closure |
|---|---:|---:|---:|---:|---:|
| evidence_topk_restore_s1.0 | `+0.0365` | `3/6` | `+0.3333` | `1/6` | `+0.0707` |
| control_topk_restore_s1.0 | `+0.0313` | `2/6` | `+1.3333` | `1/6` | `+0.0483` |
| evidence_topk_restore_s1.5 | `+0.0104` | `2/6` | `+1.5000` | `1/6` | `+0.0312` |
| control_topk_restore_s1.5 | `+0.0104` | `2/6` | `+1.3333` | `2/6` | `+0.0065` |

specificity：

| scale | evidence-control logit restore | evidence-control rank restore | evidence-control gap closure |
|---:|---:|---:|---:|
| `0.5` | `-0.0052` | `+1.6667` | `-0.0062` |
| `1.0` | `+0.0052` | `-1.0000` | `+0.0224` |
| `1.5` | `0.0000` | `+0.1667` | `+0.0247` |

Qwen 判定：

```text
Qwen 有 very weak / partial logit restoration：
evidence_topk_restore_s1.0 的 logit restore 略高于 control，
但幅度很小，rank restore 不强，positive rank restore 只有 1/6。
```

因此 Qwen 只能写：

```text
partial logit restoration only;
not feature-level causal bridge.
```

不能写：

```text
Qwen feature restoration 成功恢复答案行为。
```

## 结果三：LLaVA layer 15 restoration

关键条件：

| 条件 | mean logit restore | positive logit restore | mean rank restore | positive rank restore | mean gap closure |
|---|---:|---:|---:|---:|---:|
| evidence_topk_restore_s1.0 | `-0.0313` | `2/6` | `+1.1667` | `2/6` | `-0.0321` |
| control_topk_restore_s1.0 | `+0.0436` | `5/6` | `+4.3333` | `2/6` | `+0.0102` |
| evidence_topk_restore_s1.5 | `-0.0508` | `3/6` | `+1.5000` | `2/6` | `-0.0496` |
| control_topk_restore_s1.5 | `+0.0632` | `4/6` | `+7.1667` | `2/6` | `+0.0102` |

specificity：

| scale | evidence-control logit restore | evidence-control rank restore | evidence-control gap closure |
|---:|---:|---:|---:|
| `0.5` | `-0.0365` | `-1.1667` | `-0.0211` |
| `1.0` | `-0.0749` | `-3.1667` | `-0.0423` |
| `1.5` | `-0.1139` | `-5.6667` | `-0.0598` |

LLaVA 判定：

```text
LLaVA restoration 不支持 feature-level causal bridge。
evidence_restore 的 logit restore 平均为负，
control_restore 反而更强；
specificity 在所有 scale 上都是负的。
```

因此 LLaVA 当前仍只能写：

```text
readout replication supported;
feature-level restoration not supported.
```

## 预期与实际偏差

预期：

```text
如果 evidence-sensitive features 是 answer-support causal features，
那么在 union_mask input 上补回 clean-union feature drop，
应该能提升 target logit/rank，并且强于 control feature restore。
```

实际：

```text
Qwen:
  有非常弱的 logit restore signal，但 rank restore 不稳定，specificity 很弱。

LLaVA:
  evidence_restore 不如 control_restore，logit restore 甚至平均为负。
```

因此本轮没有达到 Stage 2G-5 成功标准。

## 结论

本实验得到的是一个清晰的负结果 / 边界结果：

```text
Qwen 和 LLaVA 的 union_mask 会强烈损伤 target answer logit/rank；
但把 evidence-mask-dropped CLT features patch 回去，
没有稳定恢复 target answer signal，也没有稳定强于 control restore。
```

这说明：

```text
跨模型 readout-level evidence sensitivity 仍然成立；
但当前 CLT feature-level restoration 还不能作为 cross-model causal bridge。
```

最终判定：

```text
Gemma3:
  full source/control causal route chain supported.

Qwen2.5-VL:
  readout replication supported;
  feature-direction intervention not supported;
  mask-to-clean feature restoration only partial logit signal;
  no causal route replication.

LLaVA:
  readout replication supported;
  feature-direction signed response partial;
  mask-to-clean feature restoration not supported;
  no causal route replication.
```

## 对“是否只能在 Gemma 上有效”的回答

仍然不能说：

```text
结论只能在 Gemma 上有效。
```

因为：

```text
Qwen/LLaVA 的 readout-level evidence-region sensitivity 是明确出现的；
union_mask 对 Qwen/LLaVA 的 target answer 行为也确实造成了强损伤。
```

但现在必须承认：

```text
Qwen/LLaVA 的 feature-level intervention/restoration 没有复现 Gemma 的 causal route chain。
```

最保守口径应更新为：

```text
Evidence-region-sensitive readouts and behavior damage under evidence masks are visible in Qwen and LLaVA,
but feature-level interventions/restorations do not yet establish cross-model causal route replication.
```

中文：

```text
Qwen 和 LLaVA 中可以看到对关键证据区域敏感的 feature readout，
而且证据区域遮挡会损伤目标答案行为；
但目前的 feature-level 干预和恢复实验还没有证明跨模型因果路径复现。
```

## 为什么这个负结果很有用

这个结果把跨模型故事的边界划得更清楚：

```text
readout-level external validity:
  有。

feature-level causal bridge:
  目前没有稳定支持。

source-control route-level causal replication:
  目前只有 Gemma 主线支持。
```

它也提示后续如果还要继续跨模型，应该优先考虑：

```text
1. hidden-state clean patch upper bound：
   先直接把 union_mask 的对应 layer/bucket hidden state 替换成 clean hidden state；
   看完整 hidden patch 是否能恢复答案。

2. answer-adjacent token patch：
   不只 patch image bucket，也 patch last prompt / assistant prefix 附近 hidden state。

3. 真正适配 Qwen/LLaVA 的 source tracing：
   不再只选 readout drop feature，而是寻找 answer-adjacent source nodes。
```

当前不建议继续扩大 feature restoration 样本，因为：

```text
核心问题不是样本数太少，
而是当前 readout feature selection + decoder patch 没有打通 causal bridge。
```
