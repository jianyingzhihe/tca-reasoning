# Stage 2P 到 Stage 2Q：当前结论、边界与下一步

## 1. 目的

本文档记录 Stage 2P 结束后的当前判断，并明确回答三个问题：

```text
1. 现在能不能下定论？
2. 如果能，能下什么层级的定论？
3. 如果不能，还缺哪些实验，尤其 Qwen / LLaVA 是否还需要做一条和 Gemma 类似的完整主线？
```

这里的关键是把证据分层，不把不同强度的结果混写：

```text
Gemma:
  full main causal-route evidence

Qwen:
  heldout-supported feature-level bridge
  heldout-supported approximate source-control probe
  not yet full Gemma-style source tracing

LLaVA:
  hidden-state bridge replicated
  weak layer-dependent feature diagnostic signals
  feature-level route not established
```

## 2. 术语解释

```text
hidden-state bridge：
  隐状态桥接。把 clean run 中某些位置的 hidden state patch 到 masked run，
  或反向从 masked patch 到 clean，观察 target token logit / rank 是否恢复或受损。
  它证明的是“这一层、这些位置携带了与视觉证据相关、且能影响目标答案的中间表示”。

feature-level bridge：
  特征层桥接。通过 CLT / sparse feature，把 hidden-state 效果进一步分解到 feature。
  如果 evidence-sensitive features 的 restore / corrupt 强于 matched feature controls，
  就能说明不是整块 hidden state 泛泛有效，而是某些 feature group 更特异地承载证据。

approximate source-control probe：
  近似 source-control 探针。因为 Qwen / LLaVA 目前还没有完全接入 Gemma 那套 source tracing adapter，
  所以先用“evidence-sensitive + target-attributing + zeroing 有效”的 feature 作为 source-like feature，
  再匹配 activation/drop/attribution 相近的 control feature。

Gemma-style full route replication：
  最完整的主线复现。要求在同一模型上完成 source tracing、source/control 干预、
  evidence region mask sensitivity、random / shifted / wrong-target controls、
  target rank 或 generation linkage。只有这条链都成立，才接近 Gemma 主线强度。
```

## 2.1 Transcoder 可比性 caveat

Qwen / LLaVA 的 transcoder 资产和 Gemma 主线不是完全同一种对象，因此跨模型结论必须写得更保守。

```text
Gemma:
  当前主线使用 tianhux2/gemma3-4b-it-plt。
  这是当前 ReplacementModel / Gemma3 pipeline 原生兼容的 PLT / transcoder set。
  它能进入完整 attribution graph、source tracing、node intervention、region-mask mainline。

Qwen:
  使用 KokosDev/qwen2p5vl-7b-clt。
  它有 config.yaml 与 layer_*.safetensors，config_model_kind = transcoder_set，
  hook 记录为 blocks.{layer}.hook_resid_pre -> blocks.{layer}.hook_resid_post。
  但 Qwen base model 不是当前 ReplacementModel 的原生后端，
  因此我们现在做的是 native Qwen hidden/feature patch 与 approximate source-control probe，
  不是完全等同 Gemma attribution graph 的 source tracing。

LLaVA:
  使用 KokosDev/llava15-7b-clt。
  它是 custom .pt + mapping_L*.pt 格式，没有标准 config.yaml。
  mapping 文件记录 MLP neuron 与 CLT feature 的对应关系，hook / loader 都是自定义路径。
  因此 LLaVA 的 feature-level 证据比 Qwen 更弱，更应该作为 diagnostic / hidden-state support。
```

因此：

```text
Qwen 的正结果可以支持“跨模型辅助证据”，尤其是 feature-level 与 approximate source-control。
LLaVA 的结果目前主要支持 hidden-state bridge。
二者都不能直接写成与 Gemma 完全同构的 source nodes。
```

## 3. Stage 2P 当前结果

### 3.1 Qwen：heldout feature bridge 成立

Stage 2P-1 在排除 Stage 2O 已用样本之后，重新选取 24 个 Qwen heldout prompt-runs，分别在 `answer_mask` 与 `union_mask` 上运行 attribution-weighted feature bridge。

结果：

```text
Qwen status = feature_bridge_bidirectional_supported
```

Restore 方向：

```text
answer_mask:
  n_rows = 24
  positive_logit_n = 24/24
  mean_logit_effect = +1.279297
  95% CI = [+0.867839, +1.729818]

union_mask:
  n_rows = 24
  positive_logit_n = 23/24
  mean_logit_effect = +1.089844
  95% CI = [+0.630208, +1.617188]

pooled specificity:
  above_all_controls_n = 40/48
```

Corrupt 方向：

```text
answer_mask:
  n_rows = 24
  positive_logit_n = 23/24
  mean_logit_effect = +0.747396
  95% CI = [+0.434896, +1.132812]

union_mask:
  n_rows = 24
  positive_logit_n = 20/24
  mean_logit_effect = +0.721354
  95% CI = [+0.348958, +1.161458]

pooled specificity:
  above_all_controls_n = 32/48
```

解释：

```text
Qwen 的 attribution-weighted evidence features 不只是读数会变。
在 masked run 中补回这些 feature contribution 可以恢复 target logit；
在 clean run 中移除这些 feature contribution 会损伤 target logit。
而且这个效果在 heldout prompt-runs 上复现，不是 Stage 2O 少数样本偶然结果。
```

### 3.2 Qwen：approximate source-control probe 成立

Stage 2P-1 同时运行 Qwen source-like feature 与 matched control feature 的对照。

结果：

```text
Qwen status = approximate_source_control_route_supported
```

核心指标：

```text
source_control_restore:
  n = 48
  positive_n = 41/48
  mean source_minus_control_logit = +0.476888
  95% CI = [+0.305664, +0.683268]

source_control_zeroing:
  n = 48
  positive_n = 48/48
  mean source_minus_control_logit = +0.863281
  95% CI = [+0.713542, +1.022135]

real_minus_shuffled:
  n = 48
  positive_n = 43/48
  mean = +0.455404
  95% CI = [+0.283203, +0.659180]

correct_minus_wrong:
  n = 4
  positive_n = 4/4
  mean = +0.669886
  95% CI = [+0.452488, +0.887284]
```

解释：

```text
Qwen 的 source-like feature 比 matched control 更能影响 target answer；
真实 evidence mask 比 shifted / shuffled mask 更有效；
correct target 比 wrong target 更相关。

这已经是很强的跨模型辅助证据。
但它仍然是 approximate source-control probe，不等同于 Gemma-style full source tracing。
```

### 3.3 LLaVA：hidden-state 成立，feature-level 未证成

Stage 2P-2 对 LLaVA 扫描了：

```text
layers = 12, 15, 18, 21
top_k = 1, 8, 32
mask_condition = union_mask
```

总体结果：

```text
status = llava_layer_sweep_weak_or_partial
strong_configs = []
```

最接近正结果的配置：

```text
layer 18, top_k 32, corrupt:
  n_rows = 8
  positive_logit_n = 7/8
  mean_logit_effect = +0.060547
  95% CI = [+0.004883, +0.103516]
  above_all_controls_n = 1/8
```

解释：

```text
LLaVA 并非完全没有信号。
它在 hidden-state bridge 层面已经有复现，在 feature sweep 中也有弱的层敏感信号。

但这些 feature-level 信号没有稳定强于 matched controls。
所以不能写 LLaVA feature-level route established。
也不能反过来写 LLaVA 没有跨模态 feature。
```

## 4. 当前能下什么定论

### 4.1 可以下的结论

可以写：

```text
在 localized、strong image-dependence、证据区域可标注的 VQA 样本中，
Gemma 的 answer-adjacent support route 对关键 evidence region 敏感，
并且这种敏感性强于 matched controls 与 random / shifted region controls。
```

可以写：

```text
该现象不是 Gemma-only 的读出现象。
Qwen 在 heldout prompt-runs 上表现出 feature-level restoration/corruption，
以及 approximate source-control probe support。
```

可以写：

```text
LLaVA 支持 hidden-state-level cross-model replication，
但当前 CLT feature-level route localization 仍未证成。
```

推荐英文表述：

```text
Cross-model evidence is strongest for Qwen:
Qwen shows heldout-supported feature-level restoration/corruption and approximate source-control probe effects.
LLaVA replicates the hidden-state bridge and shows weak layer-dependent feature signals,
but feature-level route localization remains unproven.
```

### 4.2 不能下的结论

不能写：

```text
Qwen / LLaVA 完整复现 Gemma-style source-control route。
```

不能写：

```text
LLaVA 没有跨模态 feature。
```

不能写：

```text
Qwen 的 feature 已经是对象级语义节点。
```

不能写：

```text
D_visual_only 比 B_direct 更好。
```

不能写：

```text
这些机制适用于所有 VQA 样本或所有视觉语言模型。
```

## 5. 是否还需要补实验

如果目标只是写一个收窄后的机制结论：

```text
可以下定论。
```

这个结论是：

```text
Gemma 上存在完整的 evidence-region-sensitive answer support route；
Qwen 提供了 heldout-supported feature/source-control 辅助证据；
LLaVA 提供 hidden-state 层面的跨模型辅助证据。
```

如果目标是写更强的跨模型结论：

```text
还不能下最终定论。
```

还缺：

```text
1. Qwen Gemma-style full route replication：
   从 source discovery / tracing 到 source-control intervention，
   再到 region-mask sensitivity 与 target rank / generation linkage。

2. LLaVA Gemma-style diagnostic route replication：
   即使 effect 更小，也要检查是否能在同一条链上出现 source > control、
   real > shifted、correct > wrong。

3. Qwen / LLaVA decoded answer bridge：
   如果 hidden / feature intervention 能改变 decoded answer，跨模型行为桥会更强。
   如果 decoded 不变，也可以保守写 first-token / rank bridge。

4. Qwen semantic case analysis：
   选 2-3 个高质量 case，展示 feature 是否和 answer evidence region、OCR / object region、
   top activating patches 有可解释对应。
```

## 6. 当前项目级判断

最稳版本：

```text
局部机制 claim 已经成立。
跨模型 readout / hidden-state claim 已经成立。
Qwen feature-level and approximate source-control auxiliary evidence 已经成立。
LLaVA feature-level route 尚未成立。
full cross-model Gemma-style route replication 尚未完成。
```

因此后续 Stage 2Q 的目标不是推翻现有结论，而是把跨模型证据从：

```text
Qwen auxiliary feature/source-control evidence
LLaVA hidden-state evidence
```

推进到：

```text
Qwen / LLaVA 是否能在同一条 Gemma-style 主线上复现。
```
