# 实验 030：Stage 2G Feature Intervention Dose/Sign Probe

## 目的

本实验继续推进 Stage 2G，但仍然保持保守目标：

```text
不是证明 Qwen/LLaVA 已经复现 Gemma causal route；
而是检查 Qwen layer 26 和 LLaVA layer 15 的 evidence-sensitive readout feature，
在最小 feature-direction 干预下是否会影响 target answer logit/rank。
```

换句话说，本实验想从：

```text
readout replication
```

向：

```text
intervention replication
```

迈一步。只有当 feature 干预稳定损伤目标答案，并且强于 control feature，才允许继续进入 decoded answer bridge。否则就必须停在 readout-level 结论。

## 输入

模型与候选层：

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

每个模型共：

```text
3 samples × 2 prompts = 6 usable runs
```

## 输出

原始远端结果：

```text
doc/experiments/stage2/cross_model/stage2g_qwen_feature_intervention_dose_probe.json
doc/experiments/stage2/cross_model/stage2g_qwen_feature_intervention_dose_probe.csv
doc/experiments/stage2/cross_model/stage2g_llava_feature_intervention_dose_probe.json
doc/experiments/stage2/cross_model/stage2g_llava_feature_intervention_dose_probe.csv
```

本地分析结果：

```text
doc/experiments/stage2/cross_model/stage2g_feature_intervention_dose_summary.csv
doc/experiments/stage2/cross_model/stage2g_feature_intervention_dose_specificity.csv
doc/experiments/stage2/cross_model/stage2g_feature_intervention_dose_case_table.csv
doc/experiments/stage2/cross_model/stage2g_feature_intervention_dose_decision.json
```

分析脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/analyze_stage2g_feature_intervention_dose_probe.py
```

## 方法

### 1. evidence feature 选择

对每个 `sample_id × prompt`：

```text
clean image forward
union_mask image forward
读取候选层 hidden states
编码为 CLT feature activations
在 image bucket 上计算 clean_feature_activation - union_mask_feature_activation
选 drop 最大的 top-k features 作为 evidence-sensitive features
```

这里的 `evidence-sensitive features` 不是 Gemma 主线里的 `source nodes`。它们只是：

```text
在关键证据区域遮挡后 activation 下降明显的 readout features。
```

### 2. control feature 选择

control feature 不是 Gemma 的 nearest matched source-control。当前 Qwen/LLaVA 还没有完整 source tracing，所以这里只用临时近似 control：

```text
从 clean activation 较高的 feature pool 中，
选择 activation 接近 evidence feature、
但 mask drop 较小的 features。
```

因此这个 control 只能支持最小 specificity smoke，不能等同于 Gemma 的 nearest control。

### 3. feature-direction 干预

对 clean input 的目标层输出做 forward hook，使用 CLT decoder vector 构造 hidden patch。

干预条件：

```text
evidence_top1_subtract_s1/s3/s5
evidence_top1_add_s1/s3/s5
evidence_topk_subtract_s1/s3/s5
evidence_topk_add_s1/s3/s5
control_topk_subtract_s1/s3/s5
control_topk_add_s1/s3/s5
```

`subtract` 的含义：

```text
从 hidden state 中减去 feature activation × decoder vector。
这接近 feature ablation / scale-down。
```

`add` 的含义：

```text
向 hidden state 中加回同一 feature direction。
它是 signed-probe，用于检查方向和尺度效应；
它不是 primary ablation 成功标准。
```

### 4. 指标

只看 first answer token / target token：

```text
target_logit
target_rank
delta_logit_vs_baseline
rank_damage_vs_baseline
top1_changed
```

其中：

```text
rank_damage_vs_baseline > 0 表示 target rank 变差；
delta_logit_vs_baseline < 0 表示 target logit 下降；
logit_damage = -delta_logit_vs_baseline，正值表示 target 被损伤。
```

## 成功标准

按 Stage 2G run plan，primary intervention replication 要求：

```text
1. evidence_topk_subtract 至少 2/3 case 方向一致损伤 target rank 或 target logit；
2. evidence feature 的损伤强于 control feature；
3. 若 decoded answer 不变，只能写 first-token/rank bridge，不能写 generation-level causal explanation。
```

本实验中，我们把 `evidence_topk_subtract_s5p0` 作为最强 primary ablation 条件；`evidence_topk_add_s5p0` 只作为 signed high-dose probe。

## 结果一：Qwen layer 26

baseline 情况：

```text
usable runs = 6/6
okvqa_val_2847255: target = china, baseline rank = 1
okvqa_val_4157235: target = dog, baseline rank = 3
okvqa_val_3658865: target = samsung, baseline rank = 1
```

关键统计：

| 条件 | mean rank damage | rank damage > 0 | mean logit damage | logit damage > 0 | top1 changed |
|---|---:|---:|---:|---:|---:|
| evidence_topk_subtract_s5 | `0.0000` | `0/6` | `+0.1146` | `2/6` | `0/6` |
| control_topk_subtract_s5 | `0.0000` | `0/6` | `+0.5521` | `6/6` | `1/6` |
| evidence_topk_add_s5 | `0.0000` | `0/6` | `+0.9063` | `6/6` | `1/6` |
| control_topk_add_s5 | `+0.1667` | `1/6` | `+0.5208` | `4/6` | `1/6` |

specificity：

| 对比 | rank evidence-control | logit evidence-control |
|---|---:|---:|
| subtract_s5 | `0.0000` | `-0.4375` |
| add_s5 | `-0.1667` | `+0.3854` |

Qwen 判定：

```text
primary ablation 不成立。
evidence_topk_subtract_s5 没有造成 rank damage，logit damage 也弱于 control。
evidence_topk_add_s5 会降低 target logit，但这是 add direction，不是 ablation；
而且 rank 不变，因此只能记为 partial signed high-dose response。
```

Qwen 当前状态：

```text
readout replication: supported
intervention replication: not supported by primary ablation
source-control causal replication: not established
```

## 结果二：LLaVA layer 15

baseline 情况：

```text
usable runs = 6/6
okvqa_val_2847255: target = china, baseline rank = 2
okvqa_val_4157235: target = dog, baseline rank = 4/6
okvqa_val_3658865: target = samsung, baseline rank = 3
```

这里要注意：LLaVA 的 baseline target rank 本身不如 Qwen 稳，top1 常常不是目标答案。这会降低 decoded behavior bridge 的解释力。

关键统计：

| 条件 | mean rank damage | rank damage > 0 | mean logit damage | logit damage > 0 | top1 changed |
|---|---:|---:|---:|---:|---:|
| evidence_topk_subtract_s5 | `-0.3333` | `0/6` | `-0.0645` | `2/6` | `2/6` |
| control_topk_subtract_s5 | `+0.3333` | `2/6` | `+0.0690` | `4/6` | `0/6` |
| evidence_topk_add_s5 | `+0.8333` | `3/6` | `+0.5938` | `4/6` | `0/6` |
| control_topk_add_s5 | `0.0000` | `1/6` | `+0.0247` | `4/6` | `0/6` |

specificity：

| 对比 | rank evidence-control | logit evidence-control |
|---|---:|---:|
| subtract_s5 | `-0.6667` | `-0.1335` |
| add_s5 | `+0.8333` | `+0.5690` |

LLaVA 判定：

```text
primary ablation 不成立。
evidence_topk_subtract_s5 不但没有稳定损伤 target，平均 rank 还变好；
并且 control_subtract 的损伤更强。
```

但是：

```text
evidence_topk_add_s5 出现了较明显 high-dose signed response：
rank damage = +0.8333，3/6 run rank 变差；
logit damage = +0.5938，强于 control_add_s5。
```

这说明 LLaVA layer 15 的 evidence-sensitive directions 可能确实能影响 target distribution，但当前 signed direction 和 ablation 解释不稳定。因此只能写成：

```text
partial signed high-dose effect, not primary ablation success.
```

LLaVA 当前状态：

```text
readout replication: supported
intervention replication: partial signed response only
source-control causal replication: not established
```

## 预期与实际偏差

预期：

```text
如果 evidence-sensitive feature 真是 support feature，
那么 subtract/zeroing evidence_topk 应该降低 target logit 或损伤 target rank，
并且强于 control_topk。
```

实际：

```text
Qwen:
  subtract 条件没有 rank damage，且 logit damage 弱于 control。
  add 条件有 logit damage，但不是 ablation。

LLaVA:
  subtract 条件没有 primary success。
  add 条件出现更强 high-dose response，但只能说明 signed direction 可影响输出，
  不能说明 feature ablation 复现了 support route。
```

额外工程/数值现象：

```text
LLaVA 在 okvqa_val_2847255 的 subtract_s5 条件下出现 top1 = <s>，
并有 2 条 logit 为 NaN 的记录。
这说明 high-dose subtract 可能触发了非自然隐藏状态或数值异常，
不能把这部分当作正向机制证据。
```

## 结论

本实验的核心结论是：

```text
Qwen 和 LLaVA 的 evidence-region-sensitive readout 没有被推翻；
但 feature-direction ablation smoke 没有建立跨模型 causal route replication。
```

更精确地说：

```text
Qwen:
  layer 26 readout 很强；
  feature-direction intervention 目前只显示 partial signed high-dose logit response；
  不支持 primary ablation causal replication。

LLaVA:
  layer 15 readout 稳定；
  high-dose add 有较强 response；
  但 subtract/ablation 不成立，因此仍不能写 causal route replication。
```

因此 Stage 2G 后的证据阶梯更新为：

```text
Gemma3:
  full source/control causal route chain supported in localized strong-image-dependence cases.

Qwen2.5-VL:
  readout replication supported;
  intervention replication not established.

LLaVA:
  readout replication supported;
  partial signed high-dose response;
  intervention replication not established.
```

## 这是否说明结论只能在 Gemma 上有效？

不说明。

它说明的是：

```text
跨模型 readout 层面的现象已经出现：
Qwen 和 LLaVA 的内部 feature activation 都会对 answer/union evidence mask 作出反应。
```

但也说明：

```text
readout sensitivity 不能自动等价于 causal support route。
要把 Qwen/LLaVA 升级成跨模型因果复现，
必须补更接近 Gemma 主线的 source/control intervention。
```

所以当前最保守、最准确的英文口径仍然是：

```text
Evidence-region-sensitive feature readouts appear in Qwen and LLaVA as well,
but cross-model causal route replication remains unproven.
```

## 为什么这次 ablation 没过，不一定是机制失败？

有三个可能原因：

```text
1. readout feature 不是 source route
   它可能只是对图像区域变化敏感，但不是推动答案的因果节点。

2. decoder-vector patch 是近似干预
   Qwen/LLaVA 当前没有像 Gemma 那样完整接入 ReplacementModel 和 source tracing；
   我们是在 native layer output 上用 CLT decoder direction 做近似 patch。

3. sign / direction 可能不对应 support
   LLaVA 和 Qwen 都出现 add 比 subtract 更像“损伤”的情况；
   这提示 feature direction 的符号、层内残差几何或 decoder offset 可能不能直接按 support/suppressor 解释。
```

这就是为什么本实验应该被写成：

```text
useful negative/partial result
```

而不是：

```text
cross-model conclusion failed
```

## 下一步建议

不建议立刻扩大更多 readout case。现在真正缺的是更贴近因果链的 patch 设计。

下一步优先做 `Stage 2G-5: mask-to-clean feature restoration smoke`：

```text
1. 在 clean image 和 union_mask image 上分别读取同一层 evidence features；
2. 在 union_mask forward 中，把 clean - union 的 evidence feature contribution patch 回去；
3. 比较 masked baseline vs evidence_restore vs control_restore；
4. 指标仍先看 target logit / target rank；
5. 如果 evidence_restore 比 control_restore 更能恢复 target rank/logit，再进入 decoded generation bridge。
```

这个实验比当前 subtract/add 更合理，因为它直接利用：

```text
证据区域遮挡实际造成的 feature drop
```

而不是任意增减 clean hidden 中的 feature direction。

当前不进入 decoded answer bridge，原因是：

```text
primary ablation 没过；
decoded answer 如果此时变化，也很难归因到 evidence-sensitive support route。
```
