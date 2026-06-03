# Stage 2N：Hidden-State Replication Verdict

## 1. 最终判断

Stage 2N 的结论是：

```text
Qwen 和 LLaVA 的 cross-model hidden-state bridge 在独立 heldout localized samples 上复现成功。
```

这比 Stage 2M 更强，因为：

```text
Stage 2M: 24 samples, primary evidence from selected localized pack
Stage 2N: 28 heldout samples, explicitly excludes Stage 2M samples
All52 pooled: 52 localized samples
```

最稳英文表述：

```text
Evidence-region-sensitive hidden-state bridges replicate on heldout localized samples in both Qwen2.5-VL and LLaVA-1.5. Qwen shows larger effects; LLaVA shows smaller but stable target- and location-specific effects.
```

中文表述：

```text
跨模型 hidden-state 层面的证据区域敏感桥接现象不是 Gemma-only，也不是 Stage 2M 少数样本偶然结果；它在 Qwen 和 LLaVA 的独立 heldout localized 样本上均复现。
```

## 2. 证据链

### 2.1 Heldout source-like > random

```text
Qwen:
  answer_mask corrupt/restore: stable_positive
  union_mask corrupt/restore: stable_positive

LLaVA:
  answer_mask corrupt/restore: stable_positive
  union_mask corrupt/restore: stable_positive
```

关键均值：

```text
Qwen:
  answer_mask source_minus_random = +3.834926
  union_mask source_minus_random = +3.528669

LLaVA:
  answer_mask source_minus_random = +0.920284
  union_mask source_minus_random = +0.808136
```

### 2.2 Correct target > wrong target

```text
Qwen:
  corrupt correct_minus_wrong = +3.221331
  restore correct_minus_wrong = +3.855661
  both stable_correct_gt_wrong

LLaVA:
  corrupt correct_minus_wrong = +0.688545
  restore correct_minus_wrong = +1.013022
  both stable_correct_gt_wrong
```

### 2.3 Real mask > shuffled mask

```text
Qwen:
  corrupt real_minus_shuffled = +2.717076
  restore real_minus_shuffled = +3.136998
  both stable_real_gt_shuffled

LLaVA:
  corrupt real_minus_shuffled = +0.690186
  restore real_minus_shuffled = +1.114118
  both stable_real_gt_shuffled
```

### 2.4 All52 pooled

All52 pooled 继续稳定：

```text
Qwen union_mask:
  corrupt mean_effect_logit = +2.826472
  restore mean_effect_logit = +3.423528

LLaVA union_mask:
  corrupt mean_effect_logit = +0.730281
  restore mean_effect_logit = +1.322660
```

## 3. 与 Stage 2M 的关系

Stage 2M 已经证明：

```text
Qwen/LLaVA 有 hidden-state bridge。
Qwen 更强，LLaVA 更小但稳定。
feature-level causal bridge 未成立。
```

Stage 2N 新增证明：

```text
1. 该 hidden bridge 在 Stage 2M 之外的 28 个 heldout localized samples 上复现。
2. answer_mask 与 union_mask 均支持 hidden bridge。
3. union_mask 同时通过 wrong-target 与 mask-shuffled controls。
```

所以跨模型部分现在可以从：

```text
readout + hidden bridge auxiliary evidence
```

升级为：

```text
heldout-replicated hidden-state bridge auxiliary evidence
```

但仍不能升级为：

```text
cross-model feature-level source-route replication
```

## 4. 当前可写 Claim

推荐主文 claim：

```text
As cross-model auxiliary evidence, we find that evidence-region-sensitive hidden-state bridges replicate on heldout localized VQA samples in both Qwen2.5-VL and LLaVA-1.5. The effect is larger in Qwen and smaller but stable in LLaVA, and it passes target-specific and mask-location controls at the hidden-state level.
```

更保守中文：

```text
Qwen 和 LLaVA 都在 hidden-state 层出现了可复现的证据区域敏感桥接现象；这支持“视觉证据影响答案附近内部状态”的跨模型辅助证据。但我们还没有证明这些模型中存在与 Gemma 相同的 traced source-control causal routes。
```

## 5. 仍不能写什么

不能写：

```text
Qwen/LLaVA 完整复现 Gemma source-control causal routes。
Qwen/LLaVA 的 CLT features 已经形成 feature-level causal bridge。
Qwen/LLaVA 的 decoded answer 能稳定被 hidden patch 恢复。
D_visual_only 比 B_direct 更好。
```

原因：

```text
Stage 2M feature-level bridge 未成立。
Stage 2M decoded bridge 只是 partial smoke。
Stage 2N 没有做 source tracing / nearest non-source route intervention。
Prompt 在 Stage 2N 中只是 modulation factor；B 与 D 都复现 hidden bridge。
```

## 6. 下一步建议

最有价值的下一步不是继续扩大 hidden 样本，而是做机制下沉：

```text
1. 针对 Qwen 的 strong hidden cases 做 case-level source tracing adapter。
2. 尝试 attribution-weighted multi-feature patch，而不是 top-drop feature patch。
3. 对 LLaVA 做 smaller-effect case panel，展示 hidden bridge 但谨慎解释 feature failure。
4. 如果要进一步增强行为链，优先做 passing hidden cases 的 first-token/rank + short decoded answer paired table。
```

当前跨模型最稳落点：

```text
Hidden-state cross-model replication: supported
Decoded bridge: partial
Feature-level bridge: not established
Source-control route replication: not established
```
