# Stage 2P：Cross-Modal Feature Verdict

## 1. 当前状态

Stage 2P 已完成：

```text
Stage 2P-1 Qwen heldout feature/source-control replication
Stage 2P-2 LLaVA layer/top-k diagnostic sweep
```

总体结果：

```text
Qwen:
  heldout feature bridge = bidirectional supported
  heldout approximate source-control probe = supported

LLaVA:
  layer sweep = weak_or_partial
  feature-level bridge = not established
```

## 2. 可写结论

现在可以写：

```text
Qwen provides heldout-supported feature-level and approximate source-control auxiliary evidence.
```

可以写：

```text
LLaVA replicates hidden-state bridge, and layer sweep shows weak feature-level diagnostic signals, but current CLT feature localization remains unproven.
```

中文主结论：

```text
跨模型 feature/source-control 证据目前最强的是 Qwen。
Stage 2P 说明 Qwen 的 Stage 2O 正结果不是少数样本偶然现象，而是在 heldout prompt-runs 上复现并增强。

LLaVA 不能被写成“没有跨模态机制”。
它有 hidden-state bridge，也有少量 weak layer-dependent feature signals；
但这些信号没有稳定强于 matched controls，因此 feature-level route 仍未证成。
```

## 3. 不可写结论

无论 Stage 2P 结果如何，除非后续完成真正 source tracing adapter，否则不能写：

```text
Qwen/LLaVA 完整复现 Gemma-style source-control route。
```

如果 LLaVA sweep 失败，也不能写：

```text
LLaVA 没有跨模态 feature。
```

只能写：

```text
LLaVA 当前 CLT feature-level localization 未证成。
```

## 4. 对论文 Claim 的影响

推荐跨模型表述：

```text
Cross-model evidence is strongest for Qwen: Qwen shows heldout-supported feature-level restoration/corruption and approximate source-control probe effects. LLaVA replicates hidden-state bridge and shows weak layer-dependent feature signals, but feature-level route localization remains unproven.
```

中文：

```text
跨模型证据不是“没有”，而是分层很明显：
Qwen 已经有 heldout-supported feature/source-control 证据；
LLaVA 目前只有 hidden-state 复现与弱 feature 诊断信号。
```

最安全的层级：

```text
Gemma:
  full main causal route evidence

Qwen:
  heldout-supported cross-model feature/source-control auxiliary evidence

LLaVA:
  hidden-state replication + weak feature diagnostics, no feature route claim
```
