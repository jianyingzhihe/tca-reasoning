# Stage 2P：Cross-Modal Feature Evidence 加固与反例诊断 Run Plan

## 1. 目的

Stage 2P 继续推进跨模型/跨模态 feature-level evidence，但不盲目扩大 hidden-state bridge。

当前状态：

```text
Qwen:
  hidden-state bridge: supported
  attribution-weighted feature restoration: one-direction supported
  approximate source-control probe: supported

LLaVA:
  hidden-state bridge: supported
  feature-level bridge: not established
  source-control probe: partial
```

Stage 2P 目标：

```text
1. 用 heldout prompt-runs 加固 Qwen 的 feature/source-control 正结果；
2. 用 layer/top-k sweep 诊断 LLaVA feature-level 失败是否来自层选择或 feature 选择。
```

## 2. 结论边界

可以写：

```text
Qwen 有跨模型辅助的 feature/source-control 证据。
LLaVA hidden-state 现象复现，但 feature-level route 目前未证成。
```

不能写：

```text
没有跨模态 feature。
Qwen/LLaVA 完整复现 Gemma-style source-control route。
LLaVA 没有任何跨模态机制。
```

## 3. Stage 2P-1：Qwen Heldout Feature/Route Replication

样本：

```text
从 Stage 2N all52 中排除 Stage 2O 已用样本；
按 Qwen union_mask restore source_minus_random 从高到低选 24 个 prompt-runs。
```

实验：

```text
feature bridge:
  answer_mask
  union_mask

source-control probe:
  answer_mask
  union_mask
  shifted mask
  wrong target
```

成功标准：

```text
feature_restore > controls 稳定；
source_zeroing > matched_control 稳定；
source_restore > matched_control 稳定；
real mask > shifted mask；
correct target > wrong target。
```

## 4. Stage 2P-2：LLaVA Layer/Top-k Diagnostic

样本：

```text
复用 Stage 2O 的 8 个 LLaVA diagnostic prompt-runs。
```

扫层：

```text
layer = 12, 15, 18, 21
top_k = 1, 8, 32
```

方法：

```text
复用 attribution-weighted feature score。
比较 position groups:
  top_hidden_delta_plus_answer_adjacent
  top_hidden_delta
  answer_adjacent_text
```

判据：

```text
若某个 layer/top_k 出现 stable restore/corrupt 且强于 controls：
  写 LLaVA feature bridge 是 layer-sensitive / weaker than Qwen。

若所有配置都失败：
  写 current CLT feature-level localization remains unproven。
  不写 LLaVA has no cross-modal features。
```

## 5. 输出

```text
doc/experiments/stage2/062_stage2p_qwen_heldout_feature_route_replication.md
doc/experiments/stage2/063_stage2p_llava_layer_feature_diagnostic.md
doc/experiments/stage2/064_stage2p_cross_modal_feature_verdict.md

doc/experiments/stage2/cross_model/stage2p_qwen_feature_bridge_decision.json
doc/experiments/stage2/cross_model/stage2p_qwen_source_control_decision.json
doc/experiments/stage2/cross_model/stage2p_llava_layer_sweep_decision.json
```

## 6. 解释规则

```text
Qwen heldout 成功：
  heldout-supported feature/source-control auxiliary evidence。

Qwen heldout 失败：
  Stage 2O 是 strong selected-case evidence，不能升级为 heldout-supported。

LLaVA sweep 成功：
  LLaVA feature-level bridge 是 layer/top-k sensitive。

LLaVA sweep 失败：
  当前 CLT feature localization 未证成，不代表没有跨模态特征。
```
