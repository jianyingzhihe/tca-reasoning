# Stage 2O：Feature-Level 与 Source-Control Verdict

## 1. 当前状态

Stage 2O 已完成两个实验：

```text
Stage 2O-1: Attribution-weighted multi-feature bridge
Stage 2O-2: Cross-model approximate source-control route probe
```

运行规模：

```text
Qwen:
  feature bridge usable_runs = 12/12
  source-control usable_pairs = 24

LLaVA:
  feature bridge usable_runs = 8/8
  source-control usable_pairs = 16
```

总体 verdict：

```text
Feature-level bridge:
  Qwen = one-direction supported, strongest in masked→clean restoration
  LLaVA = partial_or_weak / not established

Approximate source-control route probe:
  Qwen = supported
  LLaVA = partial
```

## 2. 可写结论

可以写：

```text
Qwen provides model-specific evidence that the cross-model hidden-state bridge can be partially localized to attribution-weighted CLT features.
```

可以写：

```text
In Qwen, attribution-weighted source-like CLT features outperform matched controls in both zeroing and restoration probes, and pass wrong-target and shifted-mask controls.
```

可以写：

```text
LLaVA continues to show smaller-effect evidence: source-like zeroing and target/location controls are positive, but feature restoration and full source-control specificity remain weak.
```

中文主口径：

```text
Stage 2O 把跨模型结果进一步分层：
Qwen 不仅有 hidden-state bridge，而且在 attribution-weighted feature 层和 approximate source-control probe 中也有支持；
LLaVA 的 hidden bridge 可以复现，但 feature/source-control 层仍然弱且异质。
```

## 3. 不可写结论

无论 Stage 2O 是否成功，除非后续完成真正 source tracing adapter，否则不能写：

```text
Qwen/LLaVA 完整复现 Gemma-style source-control route。
Qwen/LLaVA feature 是对象级语义节点。
```

还不能写：

```text
Qwen 已经完成 full source tracing。
LLaVA 已经存在 feature-level causal bridge。
这些 feature 是 dog/text/color 等具体语义节点。
hidden patch 或 feature patch 能稳定恢复完整 decoded answer。
```

## 4. 对主线影响

对 Gemma 主线：

```text
Gemma 主线不变。
Gemma 仍然是完整 causal route 证据链最强的模型：
source tracing、node intervention、nearest controls、region/wrong-image sensitivity、rank/generation linkage。
```

对跨模型部分：

```text
Stage 2N 已经说明 Qwen/LLaVA hidden-state bridge 不是 Gemma-only。
Stage 2O 进一步说明：Qwen 可以把 hidden bridge 下沉到 feature/source-control probe 层；
LLaVA 则支持 smaller-effect hidden/zeroing evidence，但不能升级到 feature/source-control route。
```

最稳英文 claim：

```text
Cross-model evidence is strongest for Qwen: beyond heldout hidden-state bridge replication, Qwen shows attribution-weighted feature restoration and approximate source-control probe support. LLaVA replicates the hidden-state bridge and shows partial source-like zeroing evidence, but feature-level route replication remains unproven.
```

最稳中文 claim：

```text
跨模型证据目前最强的是 Qwen：它不仅复现了 hidden-state bridge，还在 attribution-weighted feature restoration 和 approximate source-control probe 上获得支持。
LLaVA 说明现象不止出现在 Gemma/Qwen，但它主要停留在 hidden-state 与 zeroing 弱证据层，不能写成 feature-level route 复现。
```
