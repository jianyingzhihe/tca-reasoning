# Gemma / Qwen 机制综合报告

更新时间：2026-06-02

## 0. 最重要的一句话

现在最稳、最适合给合作者讲的主线是：

```text
Gemma3-PLT 上，我们已经看到一条完整、稀疏、可自动追踪的 evidence-to-answer 路线；
Qwen2.5-VL 上，我们也已经证明存在同类 evidence-to-answer 因果机制，
但它的可见形态不同：hidden 层最稳，PLT feature 单节点也已经 strict 成立，
只是这些 feature 节点目前还没有稳定闭合成 Gemma 那种 grouped sparse feature route。
```

更白话一点：

```text
Gemma 像是能画出一条清楚的小电路图；
Qwen 不是没有电路，而是电路长得更分布、更不像一条单纯稀疏小路。
我们已经能证明 Qwen 有一批关键 feature 节点，但还不能证明这些节点能简单串成 Gemma 式完整路线。
```

这就是目前最有价值的跨模型结论：

```text
同类 evidence-to-answer 机制存在，但机制在不同模型/表示里的可见形态不同。
```

## 1. 现在可以支撑的主线 claim

推荐写法：

```text
Gemma3-PLT provides a complete sparse source-tracing baseline for an evidence-region-sensitive answer route.
Qwen2.5-VL also supports an evidence-to-answer causal mechanism:
the hidden-level route is robust, and strict-confirmed route-first PLT feature nodes exist.
However, current public Qwen-PLT has not yet closed this mechanism as a Gemma-style grouped sparse feature route.
```

中文解释：

```text
Gemma 的路线图已经比较完整：证据区域会影响一组稀疏节点，这些节点能被自动 tracing 找出来，剪/补这些节点会影响答案，并且控制实验能排除“随便遮、随便剪也有效”。

Qwen 的机制也不是空的：hidden 层已经有很强的证据到答案因果流；PLT 里也已经找到一批严格复验的 feature 节点，它们满足答案支撑、证据 mask specificity、correct > wrong 等条件。

但 Qwen 还没有像 Gemma 那样，把这些 feature 节点稳定拼成一条 grouped sparse route。也就是说，Qwen 的 feature node-level 支持成立，feature route-level closure 还没成立。
```

## 2. Gemma 证明了什么

Gemma3-PLT 是目前最完整的正例。

它证明的不是“模型答对了”，也不是“某些节点和答案相关”，而是更强的链条：

```text
真实证据区域 -> 稀疏 source-tracing feature route -> 正确答案
```

这条链的特征是：

- 可以自动画出 route graph。
- 节点对真实 evidence mask 敏感。
- 剪 source nodes 会伤正确答案。
- 遮证据后补 source nodes 会救正确答案。
- 真实 evidence mask 的 restore 强于 shifted/shuffled mask。
- 对正确答案的影响强于 wrong target。
- primary 和 strict 都能复现 graph/compare 主结构。

所以 Gemma 可以作为我们的 sparse route baseline：

```text
Gemma = evidence-region-sensitive sparse source-tracing route 已经成立。
```

但也要注意，Gemma 的“完整”不是说从 layer 0 到最后一层每一层都画出同一条线，而是说在我们定义的 source-tracing 框架里，它能自动找出一条稀疏、可干预、可控制验证的路线。

## 3. Qwen 现在证明到了哪三层

Qwen 现在不是“失败”，而是证据被分成了三层。

### 3.1 Hidden-level route：已经成立

Hidden-level 指的是直接在 transformer hidden residual / hidden state 上做 patch，而不是先把它分解成 PLT feature。

我们在 all-layer hidden retest 里系统扫了 Qwen 层，不再只盯 layer 26。最终最强 gate 是：

```text
layer 14
restore
top_hidden_delta
answer_mask
```

strict confirmation 也复现了这个 gate。

关键结果：

```text
target logit effect mean = 1.7242, CI low = 1.2684
real minus shifted mean = 1.1144, CI low = 0.7042
real minus shuffled mean = 1.2183, CI low = 0.8010
correct minus wrong mean = 0.3311, CI low = 0.0578
target rank effect mean = 766.53, CI low = 151.23
```

白话解释：

```text
把真实证据遮掉以后，Qwen 的答案会受影响；
如果在 hidden layer 14 把关键 hidden 表示补回 clean 状态，正确答案会被救回来；
而且这种救回不是 shifted/shuffled mask 都能做到，也不是对 wrong target 一样强。
```

所以 hidden-level 结论是：

```text
Qwen has a robust hidden-level evidence-to-answer causal route.
```

### 3.2 Hidden-to-PLT mediation：route 很可能不在 sparse topK feature 里

我们进一步问：

```text
既然 hidden route 成立，当前公开 Qwen-PLT 的 sparse topK feature 能不能重构这条 route？
```

Stage4-038 的结果是：

- `hidden_residual` 成立。
- `plt_reconstruction_error` 几乎复现 hidden residual 的效果。
- `plt_topK_reconstruction` 没有通过 sparse feature mediation gate。

这说明：

```text
Qwen 的主要因果效果，很可能有相当一部分保留在 PLT reconstruction error / non-feature residual 里，
没有被当前 public Qwen-PLT 的 sparse topK features 完整捕捉。
```

白话说：

```text
Qwen 的路确实在模型内部，但当前这个 PLT 显微镜没有把它完整拆成一颗颗稀疏 feature。
```

### 3.3 Feature node-level route-first：已经 strict 成立

这是最新一条重要升级。

早期我们从 evidence-sensitive feature 出发，发现很多节点“遮证据会变”，但不一定“剪了会伤答案 / 补了能救答案”。这说明：

```text
先找 evidence-sensitive，不一定能找到真正的答案传输点。
```

后来 Stage4-060 反过来做：

```text
先找更像答案传输点的节点，也就是先满足 2+3+4，
再回头检查它是否满足 1 evidence-sensitive 和 5 correct > wrong。
```

五个 gate 是：

```text
1 = evidence-sensitive：
    遮真实证据区域时，节点 activation 变化明显，并且强于 shifted/shuffled。

2 = clean source > controls：
    clean 图像下剪 source node，比剪 matched/random controls 更伤正确答案。

3 = restore source > controls：
    遮证据后补 source node，比补 controls 更能救正确答案。

4 = real restore > shifted/shuffled：
    真实证据 mask 下的 restore 效果强于 shifted/shuffled mask。

5 = correct > wrong：
    对正确答案的影响强于 wrong target。
```

Stage4-060 最终结果：

```text
decision = qwen_route_first_full_supported

primary candidates = 23040
primary route_first_234 = 2029
primary route_first_gold = 1491
primary route_first_evidence_gold = 1334

strict candidates = 1930
strict route_first_234 = 1930
strict route_first_gold = 1447
strict route_first_evidence_gold = 1295
```

其中：

```text
route_first_234 = 先满足 2+3+4
route_first_gold = 满足 2+3+4+5
route_first_evidence_gold = 满足 1+2+3+4+5
```

这条线非常重要，因为它证明：

```text
Qwen-PLT 中确实存在一批 strict-confirmed route-first feature nodes。
这些节点不是只“看见证据会变”，而是同时满足答案支撑、restore、real-mask specificity、evidence sensitivity、correct specificity。
```

白话说：

```text
Qwen 里已经找到了很多单个关键开关。
这些开关不是随便亮一下，而是真的和“证据 -> 正确答案”这个方向有关。
```

## 4. 最新 Stage4-066：feature route-level 没有完全闭合

Stage4-066 做的是下一步：

```text
既然单个 feature nodes 已经成立，那把同一个 sample/prompt 里的这些节点打包成一条 route bundle，
整体剪/整体补，能不能得到 Gemma 那种 route-level circuit？
```

实验定义：

```text
route = 同一个 sample_id + prompt_name 下，由 Stage4-060 route-first nodes 组成的一组 feature nodes
topK = 4,8,16,32,64
```

验证方式：

```text
clean_route_zeroing:
  clean 图像下整体剪 route，看正确答案是否下降。

mask_route_restore:
  answer/union mask 下整体补 route，看正确答案是否恢复。

controls:
  same-size matched feature route
  same-feature random-position route
  random-active route
  shifted/shuffled mask
  wrong target
```

规模：

```text
primary grouped routes = 455
primary raw rows = 18200
primary unique samples = 50

strict grouped routes = 435
strict raw rows = 17400
strict unique samples = 48
strict missing fraction mean = 0.0
```

最终 decision：

```text
qwen_route_first_nodes_supported_route_unresolved
```

这句话的意思是：

```text
单个 route-first feature nodes 已经成立；
但是把这些 nodes 成组作为一条 feature route 来整体干预，目前还没有稳定通过全部 route-level gates。
```

## 5. Stage4-066 到底哪里过了，哪里没过

它不是全失败。它的结果很有结构。

稳定通过的部分：

- `route_evidence_specificity` 在 primary/strict、所有 topK 上都很强。
- `route_correct_minus_wrong` 在 primary/strict、所有 topK 上都稳定为正。
- topK `32/64` 的 `clean route zeroing source > controls` 在 primary/strict 都通过。
- topK `8` 的 `real restore > shifted/shuffled` 在 primary/strict 都通过。

卡住的部分：

- 没有任何一个预先固定 topK 同时通过 `1+2+3+4+5`。
- `restore source > controls` 整体偏弱，CI low 过不了。
- rank effect 平均值为正，但 positive fraction 低于预注册的 `0.5`。

关键数字举例：

```text
Primary topK 8:
  route_evidence_specificity mean = 107.34, CI low = 94.54, positive_frac = 1.00
  route_real_minus_shifted mean = 0.0439, CI low = 0.0251
  route_real_minus_shuffled mean = 0.0257, CI low = 0.0054
  route_correct_minus_wrong mean = 0.0540, CI low = 0.0388
  route_restore_source_minus_controls mean = 0.0138, CI low = -0.0008

Strict topK 8:
  route_evidence_specificity mean = 110.08, CI low = 97.46
  route_real_minus_shifted mean = 0.0430, CI low = 0.0223
  route_real_minus_shuffled mean = 0.0268, CI low = 0.0084
  route_correct_minus_wrong mean = 0.0550, CI low = 0.0393
  route_restore_source_minus_controls mean = 0.0125, CI low = -0.0023

Primary topK 64:
  route_zeroing_source_minus_controls mean = 0.0687, CI low = 0.0398
  route_evidence_specificity mean = 187.01, CI low = 154.81
  route_correct_minus_wrong mean = 0.0869, CI low = 0.0659
  route_restore_source_minus_controls mean = 0.0062, CI low = -0.0070

Strict topK 64:
  route_zeroing_source_minus_controls mean = 0.0670, CI low = 0.0366
  route_evidence_specificity mean = 191.23, CI low = 157.42
  route_correct_minus_wrong mean = 0.0866, CI low = 0.0654
  route_restore_source_minus_controls mean = 0.0060, CI low = -0.0075
```

白话解读：

```text
这些 grouped routes 确实带有证据区域信息，也确实更偏正确答案；
剪大一些的 route group 会伤答案；
topK 8 的真实 mask specificity 也不错；
但“遮证据后整体补回来，比补 controls 更能救答案”这个 route-level restore gate 没稳定过。
```

所以这轮不能写：

```text
Qwen feature route-level circuit fully supported.
```

只能写：

```text
Qwen route-first feature nodes are supported, but grouped feature-route closure remains unresolved.
```

## 6. 这说明 Qwen 和 Gemma 的区别是什么

最准确的区别不是“Gemma 有，Qwen 没有”。

而是：

```text
Gemma:
  evidence-sensitive nodes 里面，有一批能自然闭合成 sparse source-tracing route。
  自动 tracing、source/control、mask specificity、correct > wrong 比较一致。

Qwen:
  evidence-sensitive nodes 很多，但其中大量并不是 causal answer-support nodes。
  必须反过来先找 answer-support / restore / real-mask-specific nodes，再回看 evidence sensitivity。
  单 feature node 层面已经 strict 成立。
  但 grouped feature route 层面还没有稳定闭合。
```

这更像是“分布不同”或“可见形态不同”：

```text
Gemma 的 PLT route 更稀疏、更像一条线。
Qwen 的机制更分布、更隐藏，hidden/error 层最稳，feature nodes 能抓到，但 route bundle 不是简单相加就能闭合。
```

## 7. 为什么这仍然是成果

这轮结果不是把 Qwen 打回原点，反而把边界画清楚了。

现在我们能把 Qwen 分成三句话：

```text
1. Qwen hidden-level evidence-to-answer route: supported.

2. Qwen PLT route-first feature nodes: supported.

3. Qwen grouped sparse feature route: unresolved / not yet supported.
```

这比之前“Qwen 可能没找到”更清楚。

它说明：

```text
不是没有 evidence-to-answer mechanism；
不是没有 feature-level causal nodes；
问题集中在 route-level composition：这些节点怎么组成一条可整体补/剪的电路，目前还没有闭合。
```

这也是机制异质性的核心价值。

## 8. 我们是怎么保证严谨性的

这部分是给合作者解释“不是我们拍脑袋”的。

### 8.1 Primary / strict 分离

Primary 用来发现候选或冻结规则。

Strict 只做确认：

```text
不重新挑节点
不重新调 topK
不重新挑层
不重新换阈值
```

Stage4-060 的 strict 是 frozen exact confirmation。

Stage4-066 的 strict grouped route 也是从 primary frozen route nodes 来的，strict missing fraction mean 为 `0.0`。

### 8.2 证据 mask 对照

我们不是只遮一块图像。

我们比较：

```text
answer_mask / union_mask:
  真正标注的证据区域。

shifted_mask:
  把证据区域平移，控制“遮同样大小区域”。

shuffled_mask:
  打乱/换位 mask，控制“遮挡本身”。
```

如果真实证据 mask 比 shifted/shuffled 更强，才说明它更像 evidence-region-specific。

### 8.3 Source/control 对照

剪 source node 或补 source node 不够。

还要比较：

```text
same-position matched feature control
same-feature random-position control
random-active control
```

目的就是排除：

```text
是不是随便剪一个 active feature 都会伤答案？
是不是同一个 feature 换个位置也一样？
是不是只是 patch 规模导致的？
```

### 8.4 Correct/wrong target 对照

我们不只看正确答案 logit。

还看：

```text
对正确 target 的影响是否强于 wrong target。
```

这样可以排除：

```text
节点只是让模型整体变乱，而不是特异地支持正确答案。
```

### 8.5 不只看平均值

我们看：

```text
mean
bootstrap CI low
positive fraction
```

原因是平均值为正还不够。如果只有少数极强 case 拉高平均值，但多数样本方向不稳定，就不能写 full support。

Stage4-066 grouped route 就是典型例子：

```text
很多 route metric 的 mean 是正的；
但 positive fraction 或 CI low 没过预注册门槛；
所以不能硬写 route-level closure。
```

## 9. 现在能写什么，不能写什么

### 9.1 可以写

```text
Gemma3-PLT supports a complete sparse evidence-region-sensitive source-tracing route.
```

```text
Qwen2.5-VL supports a hidden-level evidence-to-answer causal route.
```

```text
Qwen-PLT contains strict-confirmed route-first feature nodes that satisfy evidence sensitivity, source/control, restore, real-vs-control, and correct-vs-wrong gates.
```

```text
Qwen grouped feature-route closure remains unresolved under the current public Qwen-PLT basis and grouped patch operator.
```

```text
Across models, the evidence-to-answer phenomenon appears shared, but its visible mechanistic form is representation-dependent.
```

### 9.2 不能写

```text
Qwen fully replicates Gemma-style source tracing.
```

不能写，因为 Qwen grouped feature route 和 automatic source-tracing graph 还没有闭合。

```text
Qwen has no evidence-to-answer mechanism.
```

不能写，因为 hidden-level route 和 feature node-level route-first 都已经成立。

```text
Qwen feature route is fully established.
```

不能写，因为 Stage4-066 grouped route 没有过全部 gates。

```text
所有模型机制一样。
```

不能写，因为 Gemma 和 Qwen 的可见形态明显不同。

## 10. 给合作者的白话版本

如果要口头讲，可以这样说：

```text
我们现在不是在证明模型只是“答对了”，而是在证明图像证据有没有真的进入模型内部，然后影响答案。

Gemma 这边很漂亮：它像一块能画出线路图的电路板。真实证据区域会影响一批稀疏 feature nodes，这些 nodes 可以被自动 tracing 找出来，剪掉会伤答案，遮证据后补回来能救答案，控制组也过。

Qwen 这边也不是没有路。Qwen 的 hidden layer 14 已经很稳地证明有证据到答案的因果流。进一步地，我们在 Qwen-PLT 里也找到了很多单个 feature 节点，这些节点严格通过了答案支撑、restore、真实证据 mask、correct > wrong 等 gate。

但是 Qwen 和 Gemma 的差别在于：Qwen 这些 feature 节点目前还没有被简单拼成一条完整路线。单个节点成立，成组 route 也有证据敏感和正确答案方向，但整体 restore 没有稳定超过 controls。

所以我们的结论是：同类 evidence-to-answer 机制在 Qwen 里也存在，但 Qwen 的可见形态更分布、更不像 Gemma 那种稀疏自动路线图。
```

## 11. 最终推荐主线 claim

最推荐写成：

```text
Gemma3-PLT establishes a complete sparse source-tracing route from evidence region to answer.
Qwen2.5-VL also exhibits an evidence-to-answer causal mechanism:
hidden-level restoration is robust, and strict-confirmed Qwen-PLT route-first feature nodes exist.
However, these Qwen feature nodes do not yet close into a stable grouped sparse feature route under the current public Qwen-PLT basis and intervention operator.
Thus, the cross-model phenomenon is shared, but the mechanistic visibility is representation-dependent.
```

中文短版：

```text
Gemma 是“能画出来的稀疏路线”；
Qwen 是“hidden 路线很稳，feature 节点也能严格找到，但还没拼成 Gemma 那种完整路线图”。
```

## 12. 下一步最自然的问题

Stage4-066 把问题定位到了 route-level composition。

下一步如果继续推进，不应该再问：

```text
Qwen 有没有单个 causal feature node？
```

这个已经支持。

更应该问：

```text
这些 Qwen feature nodes 为什么成组 patch 不闭合？
```

可能原因包括：

- grouped patch operator 太线性，不能模拟 Qwen 的真实组合方式。
- route 不是同 sample/prompt 内简单 topK feature 相加，而是需要跨层顺序或非线性交互。
- public Qwen-PLT feature basis 没有捕捉完整因果子空间。
- 真正 route 主要在 hidden residual / reconstruction error，feature nodes 只是可见投影。

所以后续最有价值的实验会是：

```text
operator sweep
cross-layer ordered patch
feature interaction / ablation synergy test
hidden-to-feature residual decomposition
```

但当前主线已经可以清楚写出来：

```text
Qwen 不是没有机制；Qwen 是有 hidden route 和 feature nodes，但 grouped sparse feature route 还没闭合。
```

## 13. 2026-06-02 综合更新：主结论与二级结论

这一节是截至 2026-06-02 的最新合并版本，把 Stage4 主线、Qwen route-first、Qwen grouped feature route、Stage6 prompt/text/CoT 探索、Gemma hidden 对称实验、hidden-to-PLT 分解，以及 Stage6 defensive mask robustness correction 放在同一个口径下。

### 13.1 当前最稳的总 claim

现在最适合写进论文主线的版本是：

```text
多模态模型内部存在从视觉证据区域到答案的 evidence-to-answer causal route。

Gemma3-PLT 给出了最完整的稀疏 source-tracing 正例：
证据区域敏感节点可以被自动 tracing 找出，剪掉会伤答案，遮证据后补回来能救答案，并且通过 shifted/shuffled、matched/random controls 和 correct-vs-wrong 控制。

Qwen2.5-VL 也支持同类 evidence-to-answer 机制，但可见形态不同：
hidden-level route 严格成立，route-first PLT feature 单节点严格成立；
但 grouped sparse feature route 在当前 public Qwen-PLT basis 和 grouped patch operator 下尚未闭合。
```

白话说法：

```text
Gemma 是“路线图能画出来”的模型；
Qwen 是“机制存在，但更分布、更不容易画成一条 Gemma 式稀疏线”的模型。
```

这不是一个弱结论。它说明跨模型共享的是 evidence-to-answer 因果现象，而不是共享同一种可解释形态。

### 13.2 Stage4 主线结论

Stage4 现在可以支撑四个层级的结论。

1. Gemma sparse route 成立。

Gemma3-PLT 是目前最完整的 positive control。它支持一条从真实视觉证据区域到正确答案的稀疏 source-tracing route。这个 route 同时满足 evidence sensitivity、clean source > controls、masked restore source > controls、real mask restore > shifted/shuffled、correct > wrong。

2. Qwen hidden route 成立。

Qwen all-layer hidden sweep 选出的主 gate 是：

```text
layer 14
direction = restore
position group = top_hidden_delta
mask = answer_mask
```

strict confirmation 的核心结果是：

```text
target logit effect mean = 1.7242, CI low = 1.2684
real minus shifted mean = 1.1144, CI low = 0.7042
real minus shuffled mean = 1.2183, CI low = 0.8010
correct minus wrong mean = 0.3311, CI low = 0.0578
rank effect mean = 766.53, CI low = 151.23
decision = qwen_hidden_route_supported
```

也就是说，Qwen 的 hidden residual 层级已经明确存在 evidence-to-answer 因果流。

3. Qwen route-first feature 单节点成立。

Stage4-060 反过来先找满足 `2+3+4` 的 Qwen-native route-first candidates，再回看 `1+5`。这个实验已经支持：

```text
decision = qwen_route_first_full_supported
```

含义是：Qwen-PLT 里确实存在一批单个 feature 节点，它们不仅能传递正确答案支持，而且在真实证据 mask、controls、correct-vs-wrong 上通过 strict confirmation。

4. Qwen grouped feature route 尚未闭合。

Stage4-066 把单节点推进到同一 sample/prompt 内的 grouped feature route bundle。结果不是“完全失败”，而是更精确地定位为：

```text
route-first nodes supported
grouped sparse feature-route closure unresolved
```

这说明单个节点确实有效，但把它们简单按 topK 聚成一组后，当前 grouped patch operator 没有稳定通过全部 gates。合理解释是：Qwen 的 feature-level route 可能更分布、更依赖跨层顺序、非线性交互、hidden residual 子空间，或者 public Qwen-PLT basis 对完整因果子空间覆盖不够。

### 13.3 Stage6 prompt/text/CoT 探索结论

Stage6 的定位不是重做主 claim，而是补论文里的二级结论：这个机制是否只是 prompt wording artifact？CoT/visual prompt 是否会改变 route？不同模型的稳定对象是否不同？

当前最稳的 Stage6 二级结论是：

```text
prompt/text 改写不会简单擦除 evidence-to-answer signal；
但它会改变哪些节点/路线最显著。
```

换句话说，知识主要来自图像证据时，换问法不会让机制消失；但模型内部“这次走哪条路”会变。

更重要的是，Stage6 不是用来解释“为什么 Gemma 和 Qwen 不一样”的根因实验。Gemma/Qwen 的核心差异在不换 prompt 时已经存在：

```text
Gemma:
能闭合成 sparse source-tracing route graph。

Qwen:
hidden route 强，route-first feature 单节点强，
但 grouped feature route 尚未闭合。
```

换 prompt/text/CoT 以后，这个差异没有消失，而是以类似形式再次出现：

```text
Gemma:
更适合用 route graph / source-tracing object 来看。

Qwen:
更适合用 hidden flow / route-first single feature nodes 来看。
```

所以 Stage6 的真正价值是防御一个质疑：

```text
这些 evidence-to-answer 节点/路线是不是只是在某个固定 prompt 模板下偶然出现的？
```

目前结果更支持：

```text
不是。prompt/text/CoT 会调节内部路线分配，但不会简单制造或抹掉主机制。
Gemma/Qwen 的差异更像模型和表示层级本身的可见形态差异，而不是 prompt wording 造成的假象。
```

### 13.4 Qwen Stage6：强度稳定，身份敏感

Qwen Stage6 unified route-identity full 已完成：

```text
tag = unified_v1_focus4
candidate metrics = 4608
layers = L10-L17
grid = 12 samples x 3 text variants x 4 prompt families x 8 layers
```

Qwen 的 route-first causal strength 在 prompt/text 条件下保持正向：

```text
pos234 by prompt:
A_step_visual = 0.406
B_direct = 0.395
C_step_only = 0.396
D_visual_only = 0.378

restore_source_minus_controls by prompt:
roughly 0.028-0.031

restore_correct_minus_wrong by prompt:
roughly 0.015-0.018
```

但 Qwen 的 top feature identity 对 prompt/text 很敏感。和 `B_direct + original` baseline 比较，exact `layer:pos:feature` overlap 大致是：

```text
Top4 overlap by prompt:
A_step_visual = 0.071
B_direct = 0.370
C_step_only = 0.103
D_visual_only = 0.083

Top8 overlap by prompt:
A_step_visual = 0.161
B_direct = 0.425
C_step_only = 0.165
D_visual_only = 0.157

Top16 overlap by prompt:
A_step_visual = 0.365
B_direct = 0.576
C_step_only = 0.347
D_visual_only = 0.349
```

解释：

```text
Qwen 的 route signal 稳定，但 top 节点身份会换。
```

这很适合作为一个二级结论：模型不是固定使用同一个 feature node answer wire，而是在同一视觉证据任务下，根据 prompt/text 重新分配或重排若干可用 route-first 节点。

### 13.5 Gemma Stage6 fixed-node：覆盖够，但 exact frozen node 不稳定

Gemma fixed-node full 已完成：

```text
tag = unified_v1_focus4
rows = 524
ok = 414
skipped = 96
errors = 14
usable_exact_over_planned = 0.790
usable_exact_over_planned_aligned = 0.967
source_pos_out_of_range_over_attempted = 0.027
```

这说明工程覆盖是够的：少量 `source_pos_out_of_range` 是位置对齐诊断，不是机制失败，也没有达到需要重设计的比例。

但 fixed-node causal strength 的结果很弱：

```text
source_minus_controls_mean by prompt:
A_step_visual = 0.011
B_direct = -0.009
C_step_only = 0.009
D_visual_only = -0.005

source_minus_controls_mean by text variant:
original = -0.001
paraphrase_1 = 0.0003
paraphrase_2 = 0.0016
```

解释：

```text
Gemma 的强证据不是“第一次找到的 exact 单个 frozen node 跨所有问法都稳定有效”，
而是“在每个条件下，source-tracing graph / route-level object 能捕捉到证据到答案的稀疏路线”。
```

这和 Gemma 主线不矛盾。它反而告诉我们：Gemma 的稳定对象更像 route graph，而不是跨 prompt 固定坐标的单个 node。

### 13.6 Gemma/Qwen 统一口径后怎么比较

现在不要把 Qwen 的 fixed feature-node strength 和 Gemma 的 graph overlap 直接混比。统一口径应该是两个 lens：

```text
Lens 1: fixed-node causal strength
固定 baseline 节点，跨 prompt/text 做同类节点干预。

Lens 2: route identity / topology stability
每个 prompt/text 条件重新找 route/top route set，再和 baseline 比较身份重叠。
```

在这个口径下，目前可以写：

```text
Qwen:
fixed/route-first causal strength 在 prompt/text 下保持正向；
但 top feature identities 明显 prompt/text sensitive。

Gemma:
exact frozen single-node strength 跨 prompt/text 不稳定；
但 graph/route-level source tracing 仍是它更强、更自然的机制表示。
```

不能写：

```text
Gemma 和 Qwen 的 raw logit effect 可以直接比较大小。
Gemma/Qwen route topology 完全同构。
Qwen fully replicates Gemma-style automatic source tracing。
Gemma fixed-node near-zero 说明 Gemma 主机制失败。
```

### 13.7 Gemma hidden 对称实验：已支持

为了让 hidden-level 口径也对称，Stage6-014 把 Qwen hidden residual lattice 的算法迁移到了 Gemma。这个实验现在已经完成 primary + strict。

最终状态：

```text
status = gemma_hidden_route_supported
mode = full
tag = symmetric_v1
hidden rows = 152320
hidden prompt-runs = 70
Gemma available layer count = 34
```

Primary gate：

```text
layer = 1
direction = restore
position group = visual+answer
mask condition = union_mask
n = 37
hidden_effect mean = 9.274
hidden_effect CI low = 5.139
positive_frac = 0.730
real_minus_shifted CI low = 2.267
real_minus_shuffled CI low = 0.872
correct_minus_wrong CI low = 2.662
rank_effect mean = 612.703
rank_effect CI low = 118.811
passed = true
```

Strict confirmation：

```text
layer = 1
direction = restore
position group = visual+answer
mask condition = union_mask
n = 33
hidden_effect mean = 10.473
hidden_effect CI low = 6.011
positive_frac = 0.758
real_minus_shifted CI low = 2.121
real_minus_shuffled CI low = 1.102
correct_minus_wrong CI low = 2.678
rank_effect mean = 686.848
rank_effect CI low = 130.970
passed = true
```

这意味着，在 hidden residual lens 下，Gemma 和 Qwen 都支持同类 evidence-to-answer causal route：

```text
Gemma hidden symmetric status = gemma_hidden_route_supported
Qwen existing hidden status = qwen_hidden_route_supported
```

这个结论很重要，因为它让跨模型比较更对称：Gemma 不只是 PLT/source-tracing graph 成立，在和 Qwen 相同的 hidden patch 口径下也成立。与此同时，边界仍然要写清楚：这不表示 Gemma/Qwen hidden topology 同构，也不比较两个模型 raw logit effect 的绝对大小；它只说明二者在同一个 hidden residual 干预镜头下，都存在视觉证据到答案的因果流。

### 13.8 论文式结果结构建议

如果整理成论文结果，可以写成：

```text
Result 1:
Gemma3-PLT establishes a complete sparse evidence-region-sensitive source-tracing route.

Result 2:
Qwen2.5-VL also contains an evidence-to-answer causal mechanism, strongest at hidden level and supported by strict-confirmed route-first PLT feature nodes.

Result 3:
The same cross-model phenomenon has representation-dependent visibility:
Gemma exposes a sparse route graph; Qwen exposes hidden route + single feature nodes, while grouped feature route remains unresolved.

Result 4:
Prompt/text/COT-style changes do not simply erase the evidence-to-answer signal, but they can change which internal nodes/routes become most salient.

Result 5:
Frozen exact node stability and route-level stability are different notions.
Gemma is stronger at graph/route-level source tracing than exact frozen-node prompt stability;
Qwen route-first strength is stable while top feature identity is prompt/text sensitive.

Result 6:
Defensive diagnostics constrain simple alternative explanations.
Qwen grouped-route non-closure is composition-sensitive rather than simple route absence;
Gemma error-heavy hidden-to-PLT decomposition does not contradict graph-level source tracing;
corrected mask robustness supports coverage-sensitive robustness rather than exact mask-shape invariance;
Stage6-022 gives illustrative node-to-generation bridge evidence at the rank/margin/sequence level, with limited decoded-answer change in corrupt cases.
```

### 13.9 最终可写与不可写

现在可以写：

```text
There is evidence for an internal visual-evidence-to-answer causal flow in multimodal models.
Gemma provides a complete sparse PLT source-tracing example.
Qwen provides hidden-level support and strict-confirmed feature-node support, but not yet a closed Gemma-style grouped sparse feature route.
Prompt/text changes modulate route identity more than they erase route strength.
The stable explanatory object differs across models and representation lenses.
```

现在不能写：

```text
Qwen fully replicates Gemma-style sparse source tracing.
Grouped Qwen feature route is solved.
Gemma and Qwen use the same route topology.
CoT necessarily improves visual grounding.
Gemma fixed-node near-zero means Gemma has no route.
Gemma and Qwen use the same hidden topology just because both pass the hidden residual lens.
```

最安全的总括句是：

```text
The shared phenomenon is an evidence-to-answer causal mechanism; the visible route object is model- and representation-dependent.
```

### 13.11 Cross-model hidden-to-PLT conclusion

The hidden-to-PLT decomposition now gives a clearer cross-model secondary conclusion:

```text
Both Gemma and Qwen have hidden-level evidence-to-answer causal flow.
In both models, the tested sparse PLT topK reconstruction does not carry the main hidden effect.
In both models, the PLT reconstruction error / non-topK residual lens carries most of the hidden effect.
```

For Qwen, Stage4-042 strict full showed:

```text
hidden_residual / answer_mask / target_effect:
mean = 0.8205
CI low = 0.5860
positive_frac = 0.7639

plt_topK_reconstruction:
topK 8..128 means are near zero, with CI lows below zero.

plt_reconstruction_error:
topK = 8 mean = 0.8105, CI low = 0.5794
topK = 16 mean = 0.7980, CI low = 0.5709
topK = 32 mean = 0.7973, CI low = 0.5733
topK = 64 mean = 0.7729, CI low = 0.5516
topK = 128 mean = 0.7777, CI low = 0.5546
```

For Gemma, Stage6-016 full showed:

```text
hidden_residual / union_mask / visual+answer:
primary mean = 9.2736, CI low = 5.0034
strict mean = 10.4735, CI low = 5.7841

plt_topK_reconstruction:
primary_topk_gate = null
strict_topk_gate = null
topk_retention_mean = -0.0040

plt_reconstruction_error:
primary mean = 9.2196, CI low = 4.9628
strict mean = 10.6705, CI low = 6.0947
error_retention_mean = 0.9949
```

This updates the earlier intuition. The difference between Gemma and Qwen is not that Gemma's hidden flow is locally captured by sparse PLT topK while Qwen's is not. Under the tested hidden-to-PLT decomposition, both are error-heavy.

The remaining cross-model difference is at the route-object level:

```text
Gemma:
hidden flow is error-heavy under local decomposition,
but graph-level source tracing still closes into a sparse route object.

Qwen:
hidden flow is error-heavy under local decomposition,
single route-first feature nodes are strict-supported,
but grouped sparse feature-route closure remains unresolved.
```

Safe paper wording:

```text
The shared mechanism is a hidden-level visual-evidence-to-answer causal flow.
Current PLT topK reconstruction does not locally mediate most of this flow in either Gemma or Qwen.
Gemma remains the stronger sparse graph/source-tracing positive case, while Qwen exposes the same mechanism most robustly through hidden residuals and strict-supported individual feature nodes.
```

### 13.12 Final cross-model hidden-to-PLT wording

This is the clean final version to use when summarizing the hidden-to-PLT result:

```text
Both Gemma and Qwen have a hidden-level visual-evidence-to-answer causal flow.
For both models, the tested sparse PLT topK reconstruction does not locally mediate the main hidden effect.
For both models, PLT reconstruction error / non-topK residual carries most of the hidden effect.
```

The numerical pattern is parallel:

```text
Qwen:
hidden_residual answer_mask target_effect mean = 0.8205, CI low = 0.5860
plt_topK_reconstruction topK 8..128 = near zero, CI lows below zero
plt_reconstruction_error topK 8 mean = 0.8105, CI low = 0.5794

Gemma:
hidden_residual union_mask visual+answer strict mean = 10.4735, CI low = 5.7841
plt_topK_reconstruction gate = null, topk_retention_mean = -0.0040
plt_reconstruction_error strict mean = 10.6705, CI low = 6.0947
error_retention_mean = 0.9949
```

The interpretation is:

```text
The Gemma/Qwen difference is not that Gemma's hidden effect is locally captured by sparse PLT topK while Qwen's is not.
Under this decomposition lens, both models are error-heavy.
The remaining difference is route-object closure:
Gemma has a graph-level sparse source-tracing route object;
Qwen has hidden-level route support and strict-supported single feature nodes, but grouped sparse feature-route closure remains unresolved.
```

Safe paper sentence:

```text
The shared mechanism is a hidden-level evidence-to-answer causal flow. Current PLT topK reconstruction does not locally mediate most of this flow in either Gemma or Qwen; Gemma remains the stronger sparse graph/source-tracing positive case, while Qwen exposes the same mechanism most robustly through hidden residuals and strict-supported individual feature nodes.
```

### 13.13 Stage6 defensive update: mask robustness correction

Updated on 2026-06-02.

This update corrects the earlier Stage6-021 mask robustness reading.

The earlier runs:

```text
defensive_v1
partial8_defensive_v1
```

showed exact equality across:

```text
original
dilate
erode
```

That equality should not be used as a scientific conclusion. It was traced to a runner-side remote asset collision: different mask variants reused the same remote `exported_masks/<image_stem>/answer.png` and `union.png` paths, and `put_if_missing` skipped overwriting files that were already present.

The runner has now been fixed by isolating remote asset roots per tag in:

```text
scripts/local/run_stage4_qwen_route_first_remote.py
```

The corrected Qwen layer-14 partial rerun is:

```text
tag = partial8fix_defensive_v1
samples = 8
candidates per variant = 48
model/lens = Qwen route-first layer-14 feature-node probe
variants = original, dilate, erode
```

Corrected summary:

```text
original:
route_first_234_frac = 0.1250
route_first_evidence_gold_frac = 0.0833
restore_source_minus_controls_mean = 0.0148
real_minus_shifted_mean = 0.0098
real_minus_shuffled_mean = -0.0020
evidence_specificity_mean = 17.7469

dilate:
route_first_234_frac = 0.2500
route_first_evidence_gold_frac = 0.1667
restore_source_minus_controls_mean = 0.0150
real_minus_shifted_mean = 0.0938
real_minus_shuffled_mean = 0.0820
evidence_specificity_mean = 16.7127

erode:
route_first_234_frac = 0.1458
route_first_evidence_gold_frac = 0.1250
restore_source_minus_controls_mean = 0.0169
real_minus_shifted_mean = 0.0345
real_minus_shuffled_mean = 0.0228
evidence_specificity_mean = 14.7381
```

Corrected interpretation:

```text
The mask robustness result is not exact morphology invariance.
Dilation and erosion do modulate route-first statistics.
But the signal does not collapse under modest morphology perturbation:
restore_source_minus_controls_mean remains positive across original, dilate, and erode,
and route_first_234_frac remains non-zero.
```

Safe paper wording:

```text
Corrected mask-robustness diagnostics do not support a simple pixel-perfect mask-boundary artifact story, but they also do not justify claiming exact morphology invariance. In the current Qwen layer-14 partial rerun, dilation and erosion modulate the aggregate route-first statistics without erasing them. The result is best described as coverage-sensitive robustness.
```

What should not be written:

```text
original / dilate / erode are identical.
Mask shape has no effect on route strength.
The current mask robustness result fully proves boundary-insensitive robustness.
```

### 13.10 Stage6-016 hidden-to-PLT 分解：Gemma 也呈现 error-heavy

Stage6-016 补齐了一个重要的对称口径：既然 Qwen 的 hidden-to-PLT mediation 显示主要 effect 落在 `plt_reconstruction_error`，那么 Gemma 的 hidden-level evidence-to-answer flow 是否会更容易被 sparse PLT topK reconstruction 捕捉？

最新 full 结果是：

```text
status = gemma_hidden_flow_error_heavy_like_qwen
primary raw rows = 9768
strict raw rows = 8712
gemma metric rows = 1056
```

Gemma hidden residual 本身成立：

```text
primary hidden gate:
operator = hidden_residual
mask_condition = union_mask
position_group = visual+answer
n = 37
target_effect mean = 9.2736
CI low = 5.0034
positive_frac = 0.7297

strict hidden gate:
operator = hidden_residual
mask_condition = union_mask
position_group = visual+answer
n = 33
target_effect mean = 10.4735
CI low = 5.7841
positive_frac = 0.7576
```

但 sparse PLT topK reconstruction 没有通过 gate：

```text
primary_topk_gate = null
strict_topk_gate = null
topk_retention_mean = -0.0040
```

相反，PLT reconstruction error 几乎复现 hidden residual effect：

```text
primary error gate:
operator = plt_reconstruction_error
top_k = 32
target_effect mean = 9.2196
CI low = 4.9628
positive_frac = 0.6757

strict error gate:
operator = plt_reconstruction_error
top_k = 32
target_effect mean = 10.6705
CI low = 6.0947
positive_frac = 0.7576

error_retention_mean = 0.9949
```

Qwen 的同口径 Stage4-042 strict full 结果是同一模式：

```text
Qwen hidden residual:
operator = hidden_residual
mask_condition = answer_mask
n = 144
target_effect mean = 0.8205
CI low = 0.5860
positive_frac = 0.7639

Qwen sparse PLT topK reconstruction:
topK = 8:   mean = -0.0050, CI low = -0.0162
topK = 16:  mean =  0.0010, CI low = -0.0103
topK = 32:  mean =  0.0005, CI low = -0.0104
topK = 64:  mean =  0.0013, CI low = -0.0110
topK = 128: mean =  0.0094, CI low = -0.0062

Qwen PLT reconstruction error:
topK = 8:   mean = 0.8105, CI low = 0.5794
topK = 16:  mean = 0.7980, CI low = 0.5709
topK = 32:  mean = 0.7973, CI low = 0.5733
topK = 64:  mean = 0.7729, CI low = 0.5516
topK = 128: mean = 0.7777, CI low = 0.5546
```

也就是说，Qwen 在这个 decomposition lens 下不是只有“hidden route 成立”这一层，而是更具体地显示：

```text
hidden residual 可以救答案；
sparse PLT topK reconstruction 基本接近 0；
PLT reconstruction error 几乎完整保留 hidden residual effect。
```

这说明，原先一个可能解释是“Gemma 的 hidden flow 更容易被 PLT sparse topK 捕捉，而 Qwen 更多残留在 reconstruction error”。Stage6-016 不支持这个具体版本。更准确的说法是：

```text
Gemma 和 Qwen 在 hidden-to-PLT decomposition lens 下都呈现 error-heavy；
但 Gemma 仍然有更强的 graph-level source-tracing route object。
```

这不推翻 Gemma sparse source-tracing 主结论。Gemma 的 source-tracing graph 仍然是强正例；新的边界只是：如果把 hidden patch 直接拆成 local PLT topK reconstruction 与 reconstruction error，主要答案支撑 effect 不在 local sparse topK reconstruction 里，而在 reconstruction error / non-topK residual lens 里。

因此，当前跨模型解释要更新为：

```text
两者都有 hidden-level evidence-to-answer flow；
两者在 tested hidden-to-PLT decomposition 下都不是 simple topK reconstruction captured；
Gemma 的优势在于 graph/source-tracing route 层级更闭合；
Qwen 的弱点在于 feature route bundle 尚未闭合。
```

### 13.14 Stage6-022 node-to-generation bridge symmetric mini-full

Stage6-022 adds a small illustrative bridge from internal interventions to generation-side behavior. This is not a new main claim and should not be treated as a requirement for the mechanism.

The first `defensive_v1` bridge was useful but asymmetric:

```text
Gemma v1 = hidden-residual generation bridge only.
Qwen v1 = PLT/CLT multifeature generation bridge only.
```

This was corrected in the symmetric v2 run:

```text
tag = defensive_v2_symmetric
status = decoded_bridge_full_ready

Gemma rows = 90
Qwen hidden rows = 16
Qwen PLT rows = 156
Qwen CLT rows = 156
```

The final Stage6-022 comparison uses these lenses:

```text
Gemma hidden_residual
Gemma plt_topk_reconstruction
Gemma plt_reconstruction_error

Qwen hidden_residual
Qwen PLT multifeature
Qwen CLT multifeature
```

Gemma v2:

```text
hidden_residual restore:
case_count = 3
mean_oriented_sequence_gap = +18.0716
mean_oriented_first_token_gap = +27.6875
mean_oriented_rank_gap = +1601.3333
mean_oriented_margin_gap = +11.2708

plt_reconstruction_error restore:
case_count = 9
mean_oriented_sequence_gap = +19.1819
mean_oriented_first_token_gap = +26.8681
mean_oriented_rank_gap = +1362.8889
mean_oriented_margin_gap = +9.7014
decoded restore to clean = 2 / 9

plt_topk_reconstruction restore:
case_count = 9
mean_oriented_sequence_gap = -0.5111
mean_oriented_first_token_gap = -0.4462
mean_oriented_rank_gap = -585.3333
decoded restore to clean = 0
```

Qwen v2:

```text
hidden_residual restore:
case_count = 4
mean_oriented_sequence_gap = +8.6032
mean_oriented_first_token_gap = +7.0859
mean_oriented_rank_gap = +2568.75
mean_oriented_margin_gap = +2.25
decoded changed vs reference = 3 / 4

PLT multifeature restore:
case_count = 15
mean_oriented_sequence_gap = -0.0318
mean_oriented_first_token_gap = +0.0938
mean_oriented_rank_gap = +57.6667

CLT multifeature restore:
case_count = 15
mean_oriented_sequence_gap = +0.1269
mean_oriented_first_token_gap = +0.2896
mean_oriented_rank_gap = +500.85
```

Current interpretation:

```text
Hidden-residual interventions in both Gemma and Qwen move generation-side rank, margin, and sequence scores.

Gemma's generation bridge mirrors the hidden-to-PLT decomposition: sparse PLT topK reconstruction is weak or mixed, while PLT reconstruction error carries the strong generation-side effect.

Qwen hidden-residual bridge is stronger than the tested Qwen PLT/CLT multifeature bridges in this small panel, consistent with the broader interpretation that Qwen's evidence flow is most visible in hidden/residual space and less cleanly closed as a sparse feature bundle.
```

Core conclusion:

```text
Stage6-022 is best understood as a generation-side score bridge. It shows that
the internal evidence-to-answer flow can affect answer-side scoring behavior,
but it does not establish stable decoded-answer control.

Under the symmetric v2 comparison, both Gemma and Qwen hidden-residual
interventions move sequence score, first-token score, rank, and margin. Sparse
feature-store interventions are weaker or partial: Gemma's PLT reconstruction
error carries the strong bridge while sparse PLT topK reconstruction remains
weak/mixed; Qwen hidden-residual bridge is stronger than the tested PLT/CLT
multifeature bridges.

This supports the final secondary conclusion: both models have evidence-to-answer
flow that reaches generation-side scoring, but the most direct visible carrier is
hidden/residual space. Sparse feature bundles are diagnostic but do not fully
mediate the generation-side effect in this panel.
```

Safe paper wording:

```text
As an illustrative symmetric bridge, hidden-residual interventions in both models move generation-side rank, margin, and sequence scores in selected cases. Gemma's PLT reconstruction-error bridge also carries strong generation-side effects, whereas Gemma sparse PLT topK reconstruction remains weak or mixed. Qwen hidden-residual interventions are stronger than the tested PLT/CLT feature-store bridges. Greedy decoded answer control remains partial and case-level, so this is defensive bridge evidence rather than a generation-level proof requirement.
```

What should not be written:

```text
Stage6-022 proves stable decoded generation control.
Gemma graph source-tracing route zeroing has been tested during generation.
Qwen restore reliably recovers masked decoded answers.
defensive_v1 is the final symmetric cross-model bridge.
```

### 13.15 Stage6 cross-model symmetry audit and defensive completeness

We now enforce a stricter completeness rule:

```text
Every Qwen result used in a cross-model claim needs a Gemma counterpart.
Every Gemma result used in a cross-model claim needs a Qwen counterpart.
```

This rule is lens-level, not script-name-level. The two models do not always expose the same route object, so the correct comparison is:

```text
same scientific question
same intervention/evaluation family where possible
explicitly labeled object type
no raw logit magnitude comparison across models unless normalized within model
```

Current symmetry status:

```text
Complete symmetric lenses:
  hidden residual evidence-to-answer route
  hidden-to-PLT decomposition
  prompt/text fixed-node causal strength
  prompt/text route identity stability
  node-to-generation score bridge

Model-specific but tested:
  Gemma sparse source-tracing graph route is positive.
  Qwen grouped sparse feature-route closure was tested and remains unresolved.

Defensive symmetry:
  Mask morphology is now complete at the defensive-lens level:
  Qwen has route-first layer-14 original/dilate/erode runs,
  and Gemma has hidden-lattice layer-1 visual+answer original/dilate/erode runs.
  Grouped composition is now complete at the artifact-diagnostic level:
  Qwen has grouped topK / layer-band / nonmonotonic diagnostics,
  and Gemma has source-tracing route-bundle topK path-mass / signed-mix
  diagnostics.
```

The latest Qwen grouped-composition full diagnostic strengthens the current Qwen interpretation:

```text
mode = full
route_metric_rows = 435
candidate_rows = 24970
nonmonotonic_routes = 63
nonmonotonic_frac = 0.7241
```

Layer-band signals remain positive:

```text
strict L10-L12:
restore_source_minus_controls_mean = 0.0642
route_first_evidence_gold_frac = 0.6622

strict L13-L15:
restore_source_minus_controls_mean = 0.0626
route_first_evidence_gold_frac = 0.7198

strict L16-L17:
restore_source_minus_controls_mean = 0.0618
route_first_evidence_gold_frac = 0.6456
```

But grouped topK bundles remain weak and nonmonotonic:

```text
strict topK8:
route_restore_source_minus_controls_mean = 0.0125
route_real_minus_shifted_mean = 0.0430
route_real_minus_shuffled_mean = 0.0268
route_correct_minus_wrong_mean = 0.0550
all_gates_frac = 0.0920
```

Plain-language conclusion:

```text
Qwen has many individually meaningful route-first switches, and layer-band
signals remain clearly positive. The problem is that bundling these switches
into one sparse feature route is composition-sensitive: adding more nodes often
does not monotonically improve the route and can weaken restore behavior.
```

This is why the final cross-model claim should remain:

```text
Gemma and Qwen both have evidence-to-answer causal flow.
Gemma's flow is currently most cleanly visible as a sparse source-tracing graph route.
Qwen's flow is visible at hidden and individual feature-node levels, but grouped
feature-route closure remains unresolved because feature bundles are composition-sensitive.
```

Gemma grouped-composition counterpart:

```text
status = gemma_grouped_composition_artifact_ready
route_rows = 1704
topk_summary_rows = 24
route_stability_rows = 284

topK32 path_mass_retention:
primary A = 0.9759
primary B = 0.9719
strict A = 0.9765
strict B = 0.9719

mixed_composition_frac_at_largest_topk = 0.7993
top_edge_dominated_frac_at_largest_topk = 0.0
```

Symmetric grouped-composition reading:

```text
Gemma is not compositionally trivial: its source-tracing bundle also contains
mixed signed path mass. But Gemma is graph-closed: topK32 captures almost all
traced path mass, and the route is not dominated by one edge.

Qwen is compositionally fragile: grouped topK restore remains weak and often
nonmonotonic, even though layer-band and individual route-first signals are
positive.

So the difference is not "Gemma simple, Qwen complex". It is "Gemma mixed but
graph-closed; Qwen mixed/fragile and not yet grouped-route closed."
```

Current defensive boundary:

```text
Do not write raw-magnitude equality across models for mask morphology.
But we can now write a fully symmetric defensive mask-artifact claim at the
scientific-question level: Qwen and Gemma both have original/dilate/erode
morphology counterparts, and in both cases modest morphology changes modulate
but do not erase the evidence-to-answer readout.
```

Latest cross-model mask morphology result:

```text
tag = mask16_defensive_v1
cross-model status = crossmodel_mask_morphology_ready

Qwen:
lens = route-first layer-14 feature-node metrics
selected samples = 11
evaluated candidates per variant = 132

original:
route_first_234_frac = 0.1439
route_first_evidence_gold_frac = 0.0909
restore_source_minus_controls_mean = 0.0268
real_minus_shifted_mean = 0.0464
real_minus_shuffled_mean = 0.0341
evidence_specificity_mean = 15.1441

dilate:
route_first_234_frac = 0.1667
restore_source_minus_controls_mean = 0.0197
real_minus_shifted_mean = 0.0587
real_minus_shuffled_mean = 0.0464
evidence_specificity_mean = 12.6312

erode:
route_first_234_frac = 0.1061
restore_source_minus_controls_mean = 0.0215
real_minus_shifted_mean = 0.0308
real_minus_shuffled_mean = 0.0185
evidence_specificity_mean = 12.1282

Gemma:
lens = hidden-lattice layer-1 visual+answer restore
usable prompt-runs per variant = 6

original:
target_effect_union_mean = 3.2500
real_minus_shifted_mean = 3.5417
real_minus_shuffled_mean = 1.0833
correct_minus_wrong_mean = 4.6667

dilate:
target_effect_union_mean = 6.3750
real_minus_shifted_mean = 6.6667
real_minus_shuffled_mean = 4.2083
correct_minus_wrong_mean = 2.2292

erode:
target_effect_union_mean = 7.9271
real_minus_shifted_mean = 8.2188
real_minus_shuffled_mean = 5.7604
correct_minus_wrong_mean = 4.1771
```

Interpretation:

```text
Across Qwen route-first and Gemma hidden-lattice defensive readouts, modest
dilation/erosion of the annotated evidence region changes aggregate metrics
but does not collapse the evidence-to-answer signal into control-like behavior.
This supports coverage-sensitive robustness rather than a pixel-perfect
mask-boundary artifact interpretation.
```
