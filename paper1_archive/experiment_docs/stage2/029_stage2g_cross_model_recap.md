# 实验 029：Stage 2G Cross-Model 复盘表

## 目的

本实验不是新增跑数，而是把 `Gemma3 / Qwen2.5-VL / LLaVA` 三条线放到同一张证据阶梯里，避免把不同强度的证据混写。

这一步要回答的问题是：

```text
当前 cross-model 结果是否说明主结论只能在 Gemma 上成立？
如果不是，它到底说明了什么，又还缺什么？
```

结论先写清楚：

```text
当前结果不说明主结论只能在 Gemma 上有效。
更准确的判断是：Qwen 和 LLaVA 已经出现 evidence-region-sensitive feature readout；
但它们还没有完成 source node、matched control、causal intervention 这三件事，
所以不能写 cross-model causal route replication。
```

## 专有名词解释

`readout-level evidence-region sensitivity`：

```text
读取某一层某一组 token/bucket 的 CLT feature activation；
比较 clean image 和 answer/union evidence mask 后的 feature activation；
如果关键证据区域遮挡后 top feature activation 系统性下降，就叫 readout-level evidence-region sensitivity。
它说明模型内部表征对证据区域有反应，但还不是因果证明。
```

`intervention replication`：

```text
对 evidence-sensitive feature 或 hidden direction 做干预；
如果干预能稳定损伤 target answer logit/rank，并且强于 control feature，
才算跨模型进入 intervention replication。
```

`source-control causal replication`：

```text
像 Gemma 主线一样，先追踪 answer-adjacent source node，
再做 source node zeroing / nearest matched control / random control / corruption sensitivity；
只有这一整套成立，才能写强 cross-model causal route replication。
```

## 输入

主要输入来自 Stage 2F 已完成的 cross-model readout 结果：

```text
Qwen:
  doc/experiments/stage2/cross_model/stage2f_qwen_clean_vs_mask_feature_readout_3case_summary.csv

LLaVA:
  doc/experiments/stage2/cross_model/stage2f_llava_clean_vs_mask_feature_readout_3case_summary.csv
  doc/experiments/stage2/cross_model/stage2f_llava_clean_vs_mask_feature_readout_3case_layers15_30_summary.csv

Gemma:
  现有主线结果：source tracing / node zeroing / nearest control / random region control / wrong-image / region-mask / generation linkage
```

## 输出

```text
doc/experiments/stage2/029_stage2g_cross_model_recap.md
doc/experiments/stage2/cross_model/stage2g_cross_model_recap.csv
```

生成脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/collect_stage2g_cross_model_recap.py
```

## 方法

统一用三档证据强度整理：

```text
1. readout replication
   只说明 feature activation 对 evidence mask 有反应。

2. intervention replication
   说明 feature/direction 干预能影响 target answer logit/rank。

3. source-control causal replication
   说明 answer-adjacent source route 的因果作用强于 matched controls。
```

这样做的原因是：Qwen/LLaVA 的 CLT 资产和 Gemma 主 pipeline 不是同一个工程状态。Gemma 已经能做完整 route tracing；Qwen/LLaVA 当前主要能做 feature readout 和最小 hook-forward，因此不能把它们的 readout 直接升格成 causal route。

## 结果

| 模型 | 层 / bucket | 条件 | mean top-k drop | case 一致性 | 当前证据档位 | 因果状态 |
|---|---|---|---:|---:|---|---|
| Gemma3-4B-IT | multiple / answer-adjacent source routes | wrong image + region mask | 主线已支持 | 主线已支持 | source-control causal replication | 已在 localized strong-image-dependence cases 上支持 |
| Qwen2.5-VL-7B | layer 26 / image marker-or-span | answer mask | `+15.0573` | `4/6` run 为正 | readout replication | 未证明 intervention / source-control |
| Qwen2.5-VL-7B | layer 26 / image marker-or-span | union mask | `+18.4448` | `6/6` run 为正 | readout replication | 未证明 intervention / source-control |
| LLaVA-1.5-7B | layer 15 / image token span | answer mask | `+1.1018` | `4/6` run 为正 | readout replication | 未证明 intervention / source-control |
| LLaVA-1.5-7B | layer 15 / image token span | union mask | `+1.9326` | `6/6` run 为正 | readout replication | 未证明 intervention / source-control |
| LLaVA-1.5-7B | layer 30 / image token span | union mask | `-2.6451` | 异质 / 反转 | appendix only | 不作为主候选 |

## 判定

当前 cross-model 结论应该写成：

```text
Evidence-region-sensitive feature readouts appear in Qwen and LLaVA as well,
but cross-model causal route replication remains unproven.
```

中文表述：

```text
Qwen 和 LLaVA 中也出现了对关键证据区域遮挡敏感的 feature readout；
这说明现象不是 Gemma-only 的读出现象。
但跨模型的因果路径复现还没有完成，
因为 Qwen/LLaVA 尚未证明 source node 干预、matched control specificity 和 decoded behavior bridge。
```

## 保守边界

不能写：

```text
Qwen/LLaVA 已经复现 Gemma causal route。
Qwen/LLaVA 中 source routes beat controls。
LLaVA layer 30 是稳定候选。
D_visual_only 在跨模型上更好。
这些 feature 已经是对象级语义节点。
```

可以写：

```text
跨模型 readout 结果支持：关键证据区域遮挡会影响 Qwen/LLaVA 的内部 feature activation。
Gemma3 仍是目前唯一完成完整 source/control causal route chain 的模型。
下一步应从 readout 推进到 minimal feature intervention + control。
```

## 对主 claim 的影响

这一步没有削弱 Gemma 主线。它让主 claim 的外部有效性边界更清楚：

```text
Gemma:
  可以承担当前完整机制主结论。

Qwen/LLaVA:
  支持“不是 Gemma-only 的 readout-level 现象”；
  但目前只能作为 cross-model feasibility / external validity hint。
```

下一步 Stage 2G 的合理目标不是直接宣称跨模型复现，而是补：

```text
Qwen layer 26 minimal feature intervention smoke
LLaVA layer 15 minimal feature intervention smoke
evidence-sensitive feature vs mask-insensitive control feature
必要时再做 decoded answer bridge
```
