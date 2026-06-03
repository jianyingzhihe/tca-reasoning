# Stage 2M：Cross-Model Full Replication Verdict

## 1. 总结判断

Stage 2M 的最终判断是：

```text
Qwen 和 LLaVA 都支持 hidden-state-level cross-model replication。
Decoded bridge 只有 partial smoke support。
Feature-level causal bridge 暂未成立。
Gemma-style source-control route replication 仍未完成。
```

最稳表述：

```text
The phenomenon is not Gemma-only at hidden-state level: Qwen provides strong large-effect hidden bridge evidence, and LLaVA provides smaller but target/location-specific hidden bridge evidence. However, feature-level causal bridge and source-control route replication remain unproven.
```

## 2. Tier 1：Hidden-State Full Replication

状态：

```text
supported
```

证据：

```text
24 localized samples
Qwen: 48/48 prompt-runs usable
LLaVA: 48/48 prompt-runs usable
matched controls, wrong-target controls, mask-shuffled controls 全部跑通
```

结果：

```text
overall_status = cross_model_hidden_replication_supported_qwen_stronger_llava_smaller

Qwen:
  tier1_status = tier1_hidden_mostly_supported
  wrong_target_status = supported
  mask_shuffled_status = supported
  matched_control_status = partial

LLaVA:
  tier1_status = tier1_hidden_full_supported
  wrong_target_status = supported
  mask_shuffled_status = supported
  matched_control_status = supported
```

读法：

```text
Qwen 的 effect size 更大，但部分 matched-control comparison 尤其 corrupt 方向较弱。
LLaVA 的 effect size 更小，但 target-specific 与 evidence-location-specific controls 稳定。
```

这说明：

```text
跨模型层面已经可以说 evidence-region-sensitive hidden bridge 不是 Gemma-only。
```

但不能说明：

```text
Qwen/LLaVA 已经复现 Gemma 的 traced source route。
```

## 3. Tier 1.5：Decoded Bridge

状态：

```text
partial_generation_bridge_smoke
```

Qwen：

```text
informative rows = 13
source restore target_hit = 4/16
union_mask target_hit = 2/16
source same_as_clean on informative rows = 3
source mean_logit_restore_vs_union = +5.255859
source mean_rank_restore_vs_union = +1247.875
```

LLaVA：

```text
informative rows = 11
source restore target_hit = 2/12
union_mask target_hit = 0/12
source same_as_clean on informative rows = 2
source mean_logit_restore_vs_union = +4.100260
source mean_rank_restore_vs_union = +45.25
```

读法：

```text
hidden patch 可以在部分样本上把 masked generation 往 clean/target answer 方向拉回。
但生成答案恢复不稳定，且 Qwen 的 matched controls 也能恢复不少 first-token/rank signal。
因此 decoded bridge 只能作为行为侧 smoke，不是完整 generation-level causal replication。
```

## 4. Tier 2：Feature-Level Causal Bridge

状态：

```text
not established
```

结果：

```text
Qwen:
  feature_bridge_not_established
  restore rank 有弱正信号，但 logit effect 小，controls 吸收明显，corrupt 方向不成立。

LLaVA:
  feature_bridge_not_established
  evidence_topk 在 restore/corrupt 上都不稳定，也不强于 controls。
```

解释：

```text
Stage 2M 的 hidden bridge 是跨模型存在的，但当前 CLT feature-level patch 没有把它可靠分解到少数 evidence features。
这可能意味着：
1. hidden bridge 是 distributed hidden-state effect；
2. 当前 CLT feature dictionary 不够对齐这个机制；
3. feature patch 的 decoder approximation 太粗；
4. feature 选择只按 clean-minus-mask drop，不等同 Gemma 的 source tracing；
5. 需要更精确的 source tracing / feature attribution / multi-feature group intervention。
```

## 5. Tier 3：Source-Control Route Replication

状态：

```text
not attempted / blocked by adapter and tracing gap
```

原因：

```text
Gemma 主线有完整 source tracing、node intervention、nearest non-source controls、region-mask sensitivity、rank/generation linkage。
Qwen/LLaVA 当前没有完成等价的 answer-adjacent source tracing，也没有 Gemma 风格的 source/control route intervention。
```

因此不能写：

```text
Qwen/LLaVA fully replicate Gemma source-control causal routes.
```

## 6. 最终可写 claim

建议主文写法：

```text
As auxiliary cross-model evidence, we find that evidence-region-sensitive hidden-state bridges appear in both Qwen2.5-VL and LLaVA-1.5. Qwen shows larger effects, while LLaVA shows smaller but target- and location-specific effects. However, our feature-level CLT patching does not yet establish a causal feature bridge, so the strongest source-control route claim remains specific to the Gemma pipeline.
```

中文口径：

```text
跨模型结果说明，这个现象不是只能在 Gemma 上看到；Qwen 和 LLaVA 都在 hidden-state 层出现了同类的证据区域敏感桥接现象。
但是，目前还没有证明 Qwen/LLaVA 的 CLT feature 能复现 Gemma 的 source route，也没有完成跨模型 source-control causal route replication。
```

## 7. 下一步

优先级：

```text
1. 不继续扩大 feature smoke，先分析为什么 feature bridge 被 controls 吸收。
2. 对 Qwen 的 restore-rank 正信号做 case-level drill-down，查看是否是少数 case/feature group 驱动。
3. 若要继续 Tier 2，改用 multi-feature group 或 attribution-weighted feature patch，而不是简单 top-drop feature。
4. 若要冲强 cross-model claim，需要实现 Qwen/LLaVA 的 answer-adjacent source tracing adapter。
```

当前最稳结论仍是：

```text
Gemma 主链证明 causal source routes；
Qwen/LLaVA 提供 hidden-state-level cross-model support；
feature-level cross-model causal route replication 暂未成立。
```
