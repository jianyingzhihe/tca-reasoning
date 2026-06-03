# Gemma 与 Qwen 机制实验阶段报告

更新时间：2026-05-28

## 1. 一句话结论

我们目前最稳的结论是：

```text
Gemma3-PLT 上已经形成完整的 evidence-region-sensitive source-tracing 主证据链。
Qwen2.5-VL 上也存在 evidence-to-answer 因果流，但它目前最稳定地出现在 hidden residual 层，而不是像 Gemma 那样被当前公开 Qwen-PLT 表示成一条稀疏、可自动追踪的 feature route。
```

换句话说：

```text
这个现象不是 Gemma-only。
但 Qwen 目前不是“完全复现 Gemma-style sparse source tracing”。
更合理的说法是：Qwen 有同类 evidence-to-answer 机制信号，但表示形态不同，可能更分布式，或主要落在 PLT reconstruction error / non-feature residual 中。
```

## 2. 给合作者的直白版解释

我们的核心问题是：vi

```text
模型答图像问题时，是否真的把“图像里的证据区域”传到了“答案 token”？
```

可以把模型想成一套电路：

- 图像证据区域：传感器。
- 内部节点：电路里的开关、导线或中间模块。
- 答案 token：最后亮起来的灯泡。
- source tracing：从灯泡倒着找，是哪些开关和导线把信号传过来的。
- intervention：手动剪掉或补回某个开关，看灯泡会不会变暗或恢复。

Gemma 的结果更像：

```text
我们能画出一张路线图，并且图上的一些节点被剪掉后，答案确实受影响。
遮掉图像证据区域时，这些路线也会变弱。
```

Qwen 的结果更像：

```text
我们确定模型内部有一股“证据到答案”的因果流，尤其在 layer 14 的 hidden residual 上很稳。
PLT 里也能找到很多对证据区域敏感的 feature，以及少数剪掉会伤答案的 cutter 节点。
但目前还没能把这些 PLT feature 串成 Gemma 那样稀疏、稳定、自动选路的完整路线图。
```

## 3. 术语表

### Evidence region

人工标注的图像证据区域。比如问题问“牌子上写了什么”，证据区域就是牌子文字；问题问“这个人拿着什么”，证据区域就是手里的物体。

### Mask

把图像某个区域涂成灰色 `(128,128,128)`，看模型内部状态和答案分数怎么变。

- `answer_mask`：遮核心答案证据区域。
- `relate_mask`：遮与答案相关的上下文区域。
- `union_mask`：`answer_mask + relate_mask`，遮更宽的证据区域。
- `shifted_mask`：把 `union_mask` 的形状搬到图像其他位置，尽量避开原证据区。
- `shuffled_mask`：生成面积相近、位置随机、尽量不重叠证据区的遮挡块。

如果 `answer/union mask` 的影响明显大于 `shifted/shuffled mask`，说明模型不是只对“遮挡”敏感，而是对“遮到了证据区域”敏感。

### Hidden residual

模型某一层的原始内部向量状态。它像一整束还没拆开的电线，信息很完整，但不容易解释成单个语义节点。

### PLT feature

PLT transcoder 把 hidden residual 拆成的一组 feature。可以粗略理解成“把一整束电线拆成很多可读的小开关”。这些 feature 更容易做节点级分析，但拆分不一定完美。

本文不强行展开 PLT 缩写，只按实验含义使用：`PLT` 指当前公开的、用于把模型内部状态表示为 sparse features 的 transcoder 资产。

### CLT

另一类 transcoder / feature 表示资产。它可以作为跨表示稳健性或异质性分析，但本报告主线集中在 PLT 与 hidden residual。

### Reconstruction error / non-feature residual

PLT feature 不能完整重构 hidden residual。剩下没被 sparse feature basis 捕获的部分就是 reconstruction error。  
如果因果效果主要在 reconstruction error 里，意思是：

```text
模型内部确实有这股信号，但当前 PLT feature 没有把它拆成少数可解释节点。
```

### Source node

source 不是“源代码”，而是“被认为向答案传递信号的内部节点”。在 Qwen-PLT 里常写成：

```text
F:L15:P155:ID5892
```

意思是：

```text
第 15 层，第 155 个位置，第 5892 个 PLT feature。
```

### Cutter node

剪掉后会伤害答案的节点。  
如果一个节点被 zeroing 以后正确答案 logit 或 rank 下降，就说明它像一根支持答案的线。

### Source tracing

从答案 token 反向追踪，自动生成一张 attribution graph，找出哪些 token、feature、error node、edge 对答案有贡献。

### Gemma-style source tracing

这是当前最严格的主线标准，不只是“有节点受证据影响”，而是需要一整套证据链：

- 能自动画出 answer-aligned attribution graph。
- 能找到 source nodes 和 controls。
- source nodes 比 matched / nearest / random controls 更影响答案。
- 遮真实证据区域比 shifted/shuffled controls 更影响路线。
- 对正确答案的影响强于 wrong target。
- first-token/rank/sequence score 至少有行为层面的桥接。
- primary discovery 和 strict confirmation 分开，避免在验证集上重新挑规则。

### Causal gate

我们判断一个节点是不是“因果节点”的门槛。最重要的几条是：

- `source > controls`：剪 source 比剪匹配控制节点更伤答案。
- `real > shifted/shuffled`：真实证据遮挡的效果强于空间控制遮挡。
- `correct > wrong`：对正确答案的影响强于错误目标 token。
- `rank/logit bridge`：target logit 或 rank 朝预期方向变化。

## 4. Gemma3-PLT 的结果

### 4.1 使用的模型和资产

```text
base model: google/gemma-3-4b-it
PLT asset: tianhux2/gemma3-4b-it-plt
data: paperpack72 primary + strict sensitivity
```

### 4.2 Primary72 full source tracing

Gemma primary72 是当前最完整的主证据包。

关键结果：

```text
valid samples: 71 / 72
graph A files: 71 / 71
graph B files: 71 / 71
graph success rate vs valid: 1.0
sample compare rows: 71
nodes detailed rows: 4126
edges detailed rows: 7071
intervention rows: 127
```

解释：

```text
Gemma 在 primary pack 上完整跑通了 eval、gold-answer aligned attribution graph、A/B controlled compare、node/edge details 和 intervention smoke。
```

这支持：

```text
Gemma3-PLT primary72 source-tracing pipeline passed.
```

### 4.3 Strict72 sensitivity

Strict pack 用于排除“是不是 primary 里少数人工审核样本带来的假象”。

关键结果：

```text
valid samples: 71 / 72
graph A files: 71 / 71
graph B files: 71 / 71
graph success rate vs valid: 1.0
sample compare rows: 71
nodes detailed rows: 4095
edges detailed rows: 7003
intervention rows: 0 in full strict run
```

Strict 的 graph/compare 完整复现，但 full intervention smoke 被远端 `SIGKILL(137)` 和模型加载资源状态阻塞。后面做过轻量 repair，说明这更像工程资源问题，不是机制负结果。

可写结论：

```text
Gemma3-PLT strict72 reproduces the source-tracing graph/compare pipeline on 71 valid samples.
```

不能写：

```text
Gemma strict full intervention completely passed.
```

### 4.4 Gemma 的总判断

Gemma 是目前证据链最强的模型：

```text
Gemma3-PLT 支持完整的 evidence-region-sensitive answer-support source-tracing 主线。
```

更严谨地说：

```text
Gemma3-PLT 在 primary72 上完成 full source tracing；strict72 复现 graph/compare sensitivity；strict full intervention 仍有工程阻塞，但不构成机制负证据。
```

## 5. Qwen2.5-VL-PLT 的结果

### 5.1 使用的模型和资产

```text
base model: Qwen/Qwen2.5-VL-7B-Instruct
PLT asset: KokosDev/qwen2p5vl-7b-plt
data: paperpack72 primary + strict sensitivity
```

### 5.2 Stage3 approximate PLT support

Stage3 里 Qwen 已经显示 PLT feature/source-control 近似支持。

Primary72：

```text
feature rows: 4320
feature prompt-runs: 144
source-control rows: 1452
source prompt-runs: 128
source usable pairs: 242
feature specificity positives: 8 / 8
source-control positives: 4 / 4
real-vs-shuffled positives: 4 / 8
```

Strict72：

```text
feature rows: 4320
feature prompt-runs: 144
source-control rows: 1446
source prompt-runs: 128
source usable pairs: 241
feature specificity positives: 8 / 8
source-control positives: 4 / 4
real-vs-shuffled positives: 4 / 8
```

这说明：

```text
Qwen2.5-VL-PLT 有 evidence-region-sensitive approximate feature/source-control support。
```

但这不是完整 Gemma-style source tracing，因为当时还没有真正等价的 Qwen ReplacementModel/source-tracing adapter。

### 5.3 Qwen causal cutter nodes

我们后来不再复用 Gemma 地图，而是用 Qwen 自己的 feature/position 找候选。

Stage4 causal cutter validation 结果：

```text
main candidates: 12
rank<=10 sensitivity candidates: 17
main decision: causal_cutter_supported
rank<=10 sensitivity decision: causal_cutter_supported
```

主结果：

```text
clean_source_minus_controls mean: 0.7951
CI: [0.3819, 1.2743]
positive fraction: 0.9167

clean_correct_minus_wrong mean: 0.1094
CI: [0.0625, 0.1615]
positive fraction: 0.6667
```

Sensitivity：

```text
clean_source_minus_controls mean: 0.5748
CI: [0.2672, 0.9424]
positive fraction: 0.8824
```

解释：

```text
Qwen-PLT 中确实存在一些“剪了会伤答案”的 cutter-like PLT 节点。
```

但 mask restore 不成立：

```text
answer_mask_restore_source_minus_controls mean: -0.0243
union_mask_restore_source_minus_controls mean: -0.0243
answer/union real > shifted: not supported
real > shuffled: weak/inconsistent
```

所以当前只能写：

```text
Qwen causal-screened PLT cutter nodes exist.
```

不能写：

```text
Qwen causal-screened evidence-linked cutter route is established.
```

### 5.4 Qwen all-layer hidden retest

这是 Qwen 目前最强、最重要的正结果。

我们检查了 Qwen language layers `0..27`，避免只看 layer 26 的偏差。

Primary selected gate：

```text
layer: 14
direction: masked -> clean restore
position group: top_hidden_delta
mask condition: answer_mask
usable prompt-runs: 96
```

Primary 指标：

```text
target logit effect mean: 1.7544
CI low: 1.2872

real minus shifted mean: 1.1148
CI low: 0.6851

real minus shuffled mean: 1.2932
CI low: 0.8464

correct minus wrong mean: 0.3148
CI low: 0.0424

target rank effect mean: 803.74
CI low: 171.58
```

Strict confirmation 同一 gate：

```text
usable prompt-runs: 101

target logit effect mean: 1.7242
CI low: 1.2684

real minus shifted mean: 1.1144
CI low: 0.7042

real minus shuffled mean: 1.2183
CI low: 0.8010

correct minus wrong mean: 0.3311
CI low: 0.0578

target rank effect mean: 766.53
CI low: 151.23
```

结论：

```text
Qwen2.5-VL has a replicated hidden-level evidence-region-sensitive answer-support route at layer 14.
```

这非常关键。它说明 Qwen 不是没有 evidence-to-answer 机制；相反，hidden 层证据很强。

### 5.5 Qwen hidden-to-PLT mediation

接下来我们问：

```text
既然 Qwen hidden layer 14 route 成立，它能不能被 PLT sparse features 重构出来？
```

比较三种 patch：

- `hidden_residual`：直接补回完整 hidden residual。
- `plt_topk_reconstruction`：只用 topK PLT features 重构补回。
- `plt_reconstruction_error`：只补回 PLT 没解释掉的 error / non-feature residual。

Primary 和 strict 都得到：

```text
decision: qwen_route_may_live_in_plt_error
```

Strict 关键指标：

```text
hidden_residual / answer_mask / target_effect mean: 0.8205
CI low: 0.5860

hidden_residual / answer_mask / real > shifted mean: 0.5204
CI low: 0.3007

hidden_residual / answer_mask / real > shuffled mean: 0.4943
CI low: 0.2643

plt_reconstruction_error / top8 / answer_mask / target_effect mean: 0.8105
CI low: 0.5794

plt_reconstruction_error / top8 / answer_mask / real > shifted mean: 0.5094
CI low: 0.2898

plt_reconstruction_error / top8 / answer_mask / real > shuffled mean: 0.4856
CI low: 0.2603
```

解释：

```text
Qwen 的 hidden route 很稳，而且 PLT reconstruction error 几乎复现了 hidden residual 的因果效果。
但 sparse topK PLT features 没有通过 gate。
```

这支持：

```text
Qwen route may live in PLT reconstruction error / non-feature residual.
```

### 5.6 Qwen all-layer bounded exhaustive 与中层 dense scan

我们进一步排除了“是不是漏层、漏节点”的问题。

Broad all-layer 结果显示：

```text
L10/L13/L15 附近有较明显 source-control 近通过信号。
L14 是 hidden route 最强层，但不是 sparse PLT near-pass 最强层。
L17-L20 后信号变弱。
```

因此后续中层扫描集中在 `L10-L17`。

### 5.7 Qwen evidence-specific allpool validation

最后我们专门找 evidence-specific PLT nodes，也就是：

```text
遮真实证据区域后明显变弱，且 shifted/shuffled controls 不能解释的 Qwen-PLT 节点。
```

最终修复 selection bug 后，真正全池覆盖：

```text
L10: 146 candidates
L11: 219 candidates
L12: 260 candidates
L13: 269 candidates
L14: 185 candidates
L15: 417 candidates
L16: 257 candidates
L17: 394 candidates
total: 2147 candidates
```

关键结果：

```text
answer_mask real > shifted:
mean = 0.0155
CI low = 0.0117
=> stable positive

union_mask real > shifted:
mean = 0.0097
CI low = 0.0056
=> stable positive
```

这说明：

```text
Qwen-PLT 里确实有大量 evidence-sensitive candidates。
```

但因果 gate 没过：

```text
clean source > controls:
mean = -0.00093
CI low = -0.00349
=> not supported

answer restore source > controls:
mean = -0.00053
CI low = -0.00272
=> not supported

correct > wrong:
mean = -0.00065
CI low = -0.00357
=> not supported
```

因此当前判定：

```text
qwen_evidence_sensitive_but_not_causal
```

更直白地说：

```text
这些 Qwen-PLT 节点确实很像“看见证据区域会变”的节点。
但把它们单独当 source 去剪或补，还不能稳定控制答案。
```

## 6. Gemma 与 Qwen 的对照

| 问题 | Gemma3-PLT | Qwen2.5-VL-PLT |
|---|---|---|
| 是否有 evidence-to-answer 信号 | 是 | 是 |
| 最强证据层级 | PLT source-tracing graph + node intervention | hidden residual layer 14 |
| 是否能自动画路线图 | 是，Gemma pipeline 原生支持 | 尚未闭合，Adapter V2/V3/V4 均未通过完整 gates |
| 是否有 sparse cutter nodes | 是，主线包含 intervention | 有，Qwen-native cutter candidates 成立 |
| 是否 evidence-linked | Gemma 支持更完整 | Qwen PLT 目前未稳定闭合 |
| 是否通过 strict | graph/compare strict 通过，full strict intervention 工程阻塞 | hidden route strict 通过，PLT sparse route 未通过 |
| 当前主结论 | 完整主线模型 | 机制存在但表示异质，PLT sparse localization 未成立 |

## 7. 现在可以对外怎么讲

推荐口径：

```text
We find strong evidence-region-sensitive answer-support routes in Gemma3-PLT with full source tracing. In Qwen2.5-VL, the same broad evidence-to-answer phenomenon is present at the hidden-state level and partially visible in PLT features, but current public Qwen-PLT does not localize it into a Gemma-like sparse source-tracing route.
```

中文口径：

```text
Gemma 上我们已经有完整的 source tracing 主链：能画路线、能做节点干预、能看到证据区域特异性。
Qwen 上不是没有机制，而是机制的可见形态不同：hidden 层的 evidence-to-answer 因果流很强，PLT 里也有证据敏感节点和 cutter 节点，但当前 sparse PLT feature basis 还没把它闭合成 Gemma 那样的稀疏路线。
```

更短版本：

```text
Gemma 是完整路线图。
Qwen 是同类因果流存在，但路线没有被当前 PLT 稀疏节点图完整捕获。
```

## 8. 现在不能对外怎么讲

不能写：

```text
Qwen fully replicates Gemma-style source tracing.
```

原因：

```text
Qwen 的 automatic source-tracing adapter 还没有通过 source/control、real-vs-shifted/shuffled、correct-vs-wrong、rank/sequence 全部 gates。
```

不能写：

```text
Qwen 没有 evidence-to-answer mechanism。
```

原因：

```text
Qwen hidden layer 14 的 primary + strict 结果非常强，已经支持 hidden-level evidence route。
```

不能写：

```text
Gemma 和 Qwen 的机制完全一样。
```

原因：

```text
Gemma 的机制在 PLT sparse source tracing 中清晰可见；Qwen 的主要因果效果目前更像在 hidden residual / PLT reconstruction error 中，而不是少数 sparse PLT features。
```

## 9. 方法学上我们已经修正过的坑

### 不复用 Gemma 地图

Qwen 后续实验没有拿 Gemma 节点 id 当 Qwen 节点。Gemma 只提供判据，不提供 Qwen 的路线图。

### 不只测 layer 26

早期 Qwen 结果有 layer 26 偏置风险。后续 all-layer hidden sweep 覆盖了 `0..27`，发现最强 confirmed hidden gate 在 layer 14。

### Primary 和 strict 分离

我们修正了一个分析风险：strict 不应重新挑 best gate。现在 strict 是确认 primary-selected gate，而不是在 strict 上重选规则。

### 全池验证 bug 已修复

`Stage4-052 --selection all` 最初有 runner bug：外层生成了全池 manifest，但 validation 仍硬编码 `main`，实际只跑每层 20 个。已修复并重新用 `allpool` 跑完真正全池，覆盖 `2147` 个候选。

## 10. 下一步建议

如果目标是继续推进 Qwen：

1. 不要再简单扩大 sparse node 数量。`2147` 个 evidence-specific candidates 已经说明“找 evidence-sensitive node”不是主要瓶颈。
2. 重点测试 intervention operator：当前 single-feature zeroing/restore 可能太弱，应该测 grouped feature patch、multi-position patch、scaled patch、residual-direction patch。
3. 继续围绕 hidden layer 14 和中层 `10-17`，不要回到 layer 26 默认假设。
4. 把 Qwen 的结论写成 representation-dependent mechanism：hidden route supported，current sparse PLT localization failed or unresolved。
5. CLT 可以作为 Stage5 异质性图谱，但不应覆盖 PLT/hidden 主线。

## 11. 当前最终摘要

```text
Gemma3-PLT:
  完整 source-tracing 主线成立。
  primary full 通过。
  strict graph/compare sensitivity 通过。
  strict full intervention 有工程阻塞，不是机制负结果。

Qwen2.5-VL:
  hidden-level evidence-to-answer route 成立，layer 14 primary + strict 强复现。
  PLT 中存在 evidence-sensitive candidates。
  PLT 中存在 causal-screened cutter nodes。
  但当前 sparse PLT source route 没有稳定闭合。
  hidden-to-PLT mediation 指向 reconstruction error / non-feature residual。

跨模型含义:
  evidence-to-answer 现象不是 Gemma-only。
  但不同模型的可解释表示形态不同。
  Gemma 是 sparse PLT source-tracing 成功案例。
  Qwen 是 hidden-level 成功、PLT sparse localization 未成立的机制异质性案例。
```

