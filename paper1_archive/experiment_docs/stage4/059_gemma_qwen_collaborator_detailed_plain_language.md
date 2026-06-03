# Gemma / Qwen 机制实验详细讲解稿

更新时间：2026-05-28

## 0. 最重要的结论先讲清楚

我们现在要讲的重点不是“某个模型某个 layer 某个 feature id 发生了什么”，而是下面这件事：

```text
视觉语言模型回答图像问题时，内部存在一条从“图像证据区域”到“答案”的因果流。
```

这句话拆开讲：

```text
模型不是只靠语言先验随便猜答案。
它内部确实会把图像里支持答案的区域，转换成某种内部信号，再影响最后的答案 token。
```

目前两个模型给我们的证据形态不一样：

```text
Gemma 存在一条清楚的 evidence-region-to-answer 路径：这条路径能被 source tracing 自动画出来，路径节点会随真实证据遮挡而变弱，剪掉这些节点会伤害正确答案，并且这种效果强于 matched/random controls 和 shifted/shuffled mask controls。

Qwen 也表现出同类机制的几个关键特征：真实证据遮挡会稳定影响内部状态；hidden layer 14 上存在可复现的 evidence-to-answer 因果流；PLT 中也有大量 evidence-sensitive 节点和一些 causal cutter 节点。

但 Qwen 目前还没有证明最后一步：这些 PLT 节点还没能稳定闭合成一条像 Gemma 那样的稀疏 source-tracing 路径。换句话说，Qwen 支持
```

最适合对外讲的一句话：

```text
Gemma 证明了这类路线可以被完整追踪；Qwen 证明了同类 evidence-to-answer 因果流不是 Gemma 特例，但不同模型可能用不同内部表示承载它。
```

## 1. 这件事的意义是什么

### 1.1 它从“模型答对了”推进到“模型为什么答”

普通评测只问：

```text
模型答对了吗？
```

我们的实验问的是：

```text
模型是不是因为看到了图像中的正确证据才答这个答案？
这个证据在模型内部经过了哪些表示？
如果我们打断这些内部表示，答案会不会受影响？
```

这比 accuracy 更接近机制解释。

### 1.2 它不是普通相关性，而是因果证据

只看“遮掉图像后答案变了”还不够，因为这只能说明图像重要。

我们进一步做了几件事：

```text
1. 遮真实证据区域，看内部节点是否变化。
2. 遮别的相似区域，看是不是普通遮挡也会造成同样变化。
3. 找到可能传递答案信号的内部节点。
4. 剪掉这些节点，看答案是否受伤。
5. 在遮证据的情况下补回这些节点，看答案是否恢复。
```

这就把问题从“相关”推进到“因果干预”。

### 1.3 它说明机制可能跨模型存在，但不是同一种形状

最有科学价值的点在这里：

```text
Gemma 和 Qwen 都出现了 evidence-to-answer 因果流。
但 Gemma 的流更容易被 PLT sparse features 画成路线图。
Qwen 的流更像在 hidden residual 或 PLT reconstruction error 里。
```

所以这不是简单的：

```text
Qwen 完全复现 Gemma。
```

也不是：

```text
Qwen 没有机制。
```

更准确的是：

```text
同类目标机制存在，但不同模型内部承载方式不同。
```

这可以变成一个很好的论文叙事：

```text
Gemma gives sparse traceable route evidence.
Qwen gives cross-model hidden-level route evidence and representation heterogeneity.
```

## 2. 用一个比喻讲清楚

可以把模型想象成一台复杂机器。

图像证据区域像传感器：

```text
例如问题问“牌子上写了什么”，牌子文字就是证据区域。
```

答案像最后亮起来的灯泡：

```text
模型最后要输出某个答案 token。
```

模型内部的 hidden state / feature 像电路里的中间元件：

```text
有些元件负责把传感器信号传到灯泡。
```

Gemma 的情况像：

```text
电路板比较清楚。
我们能沿着电线从灯泡往回追，找到一串关键开关。
剪掉这些开关，灯泡变暗。
遮掉传感器，开关信号也变弱。
```

Qwen 的情况像：

```text
功能也在，但更像高度集成芯片。
你能证明传感器信号确实影响了灯泡。
但信号不是暴露在几根清楚的电线上，而是藏在一块整体电路或残差信号里。
```

这就是我们现在说的：

```text
Gemma 是 sparse PLT source-tracing route。
Qwen 是 hidden-level evidence-to-answer route，当前 sparse PLT localization 没闭合。
```

## 3. 必要专业词，用小白话翻译

| 专业词 | 小白话解释 | 在我们实验里的作用 |
|---|---|---|
| evidence region | 图像里真正支持答案的区域 | 比如牌子文字、手里的物体、数字、标识 |
| mask | 把图像某块涂灰 | 测试模型是否依赖这个区域 |
| answer_mask | 遮核心答案证据区域 | 主证据遮挡 |
| union_mask | 遮答案区域加相关上下文 | 更宽的证据遮挡 |
| shifted_mask | 把同样形状的遮挡块挪到别处 | 控制“是不是随便遮一块都会影响” |
| shuffled_mask | 面积相近的随机遮挡块 | 控制随机位置遮挡效应 |
| hidden residual | 模型某层的原始内部向量 | 信息完整，但不容易解释成单个节点 |
| PLT feature | 用 PLT transcoder 拆出来的稀疏特征 | 更像可解释的小节点，但不一定捕获全部信息 |
| source node | 被认为向答案传信号的内部节点 | 被拿来剪掉或补回做因果测试 |
| cutter node | 剪掉后会伤答案的节点 | 证明某个节点支持答案 |
| source tracing | 从答案反向追踪内部路线 | 画“证据到答案”的路线图 |
| reconstruction error | PLT features 没重构出来的剩余部分 | Qwen 的主要因果效果可能在这里 |

## 4. 我们到底用什么方法证明

### 4.1 方法一：证据区域遮挡

我们先人工标注图像中支持答案的区域，然后做不同遮挡：

```text
clean image:
  原图。

answer_mask:
  遮核心证据区域。

union_mask:
  遮核心证据区域 + 相关上下文。

shifted_mask:
  把同样形状的遮挡块挪到其他地方。

shuffled_mask:
  随机生成面积相近的遮挡块。
```

核心判断：

```text
如果遮 answer/union 明显比 shifted/shuffled 影响更大，
说明模型对“证据区域”敏感，而不是只对“遮挡”敏感。
```

### 4.2 方法二：source tracing 路线图

在 Gemma 上，我们可以从答案 token 反向追踪，生成 attribution graph。

这个 graph 里有：

```text
token nodes:
  文本或图像 token 位置。

feature nodes:
  PLT 拆出来的内部特征。

error nodes:
  PLT 没解释掉的残差节点。

edges:
  节点之间的贡献关系。
```

这一步的意义是：

```text
不是手动猜节点，而是让模型内部贡献关系自动生成候选路线。
```

### 4.3 方法三：剪节点

找到疑似 source node 后，我们做 zeroing：

```text
把这个节点的激活清零。
```

如果答案 logit 下降、rank 变差，就说明：

```text
这个节点对答案有因果支持作用。
```

### 4.4 方法四：补节点

如果遮掉证据区域后答案变差，我们尝试把某个内部节点的 clean 激活补回去。

如果补回后答案恢复，就说明：

```text
这个节点可能是“证据 -> 答案”路线的一部分。
```

这比只剪节点更强，因为它直接测试：

```text
证据被遮掉造成的信息缺失，能不能通过补回内部节点来弥补。
```

### 4.5 方法五：control 对照

我们用了很多 controls，目的是避免自欺。

主要 controls：

```text
matched feature control:
  找激活强度或属性相近、但不是 source 的 feature。

random active feature:
  随机活跃 feature。

same-feature random-position:
  同一个 feature 换到随机位置。

shifted/shuffled mask:
  遮非证据区域。

wrong target:
  看干预是不是只对正确答案更强。
```

如果 source node 的效果不比这些 control 强，就不能说它是真正路线节点。

## 5. Gemma 的证据链

### 5.1 Gemma 是当前完整主线模型

Gemma 使用：

```text
base model: google/gemma-3-4b-it
PLT asset: tianhux2/gemma3-4b-it-plt
dataset: paperpack72 primary + strict sensitivity
```

Gemma 的强点是：

```text
当前工具链可以完整支持 Gemma 的 source tracing。
```

也就是说，Gemma 不只是能跑 hidden patch，而是能跑：

```text
eval -> answer-aligned graph -> graph compare -> node/edge details -> intervention smoke
```

### 5.2 Gemma primary72 full 结果

关键数字：

```text
valid samples: 71 / 72
graph A files: 71 / 71
graph B files: 71 / 71
graph success rate: 1.0
sample compare rows: 71
nodes detailed rows: 4126
edges detailed rows: 7071
intervention rows: 127
```

小白话解释：

```text
72 个样本里有 71 个有效。
这 71 个样本都成功生成了答案对齐的路线图。
路线图里有几千个节点和边的明细。
还做了 127 行节点干预。
```

这支持：

```text
Gemma 上的 source-tracing 主线不是只在几个例子上看到，而是在 paperpack primary 上完整跑通。
```

### 5.3 Gemma strict72 sensitivity 结果

Strict 是更严格的验证包，用来确认 primary 结果不是靠少数样本撑起来。

关键数字：

```text
valid samples: 71 / 72
graph A files: 71 / 71
graph B files: 71 / 71
graph success rate: 1.0
sample compare rows: 71
nodes detailed rows: 4095
edges detailed rows: 7003
```

注意：

```text
strict full intervention 阶段遇到远端资源 / model-load SIGKILL(137)。
```

这个不能写成机制失败。因为 graph 和 compare 已经完整通过，后续轻量 repair 也说明主要是工程资源问题。

严谨口径：

```text
Gemma primary full passed.
Gemma strict graph/compare sensitivity passed.
Strict full intervention remains engineering-blocked, not mechanism-negative.
```

### 5.4 Gemma 能支持的主结论

可以说：

```text
Gemma3-PLT 上存在完整的 evidence-region-sensitive answer-support source-tracing route。
```

更精确地说：

```text
Gemma3-PLT 在 primary72 上完成 full source tracing；strict72 复现 graph/compare sensitivity；干预和 control 结果支持这是一条与证据区域相关的答案支持路线。
```

## 6. Qwen 的证据链

### 6.1 Qwen 为什么更复杂

Qwen 使用：

```text
base model: Qwen/Qwen2.5-VL-7B-Instruct
PLT asset: KokosDev/qwen2p5vl-7b-plt
```

Qwen 的难点是：

```text
它不能直接复用 Gemma 的 ReplacementModel / source-tracing pipeline。
```

所以我们不能拿 Gemma 的路线图套到 Qwen 上。后续 Qwen 实验都坚持：

```text
不复用 Gemma node ids。
只使用 Qwen 自己的 hidden state、Qwen 自己的 PLT feature、Qwen 自己的 candidate discovery。
```

### 6.2 Qwen Stage3：近似 PLT 支持

Stage3 先证明 Qwen-PLT 有 approximate feature/source-control support。

Primary：

```text
feature prompt-runs: 144
source prompt-runs: 128
source usable pairs: 242
feature specificity positives: 8 / 8
source-control positives: 4 / 4
real-vs-shuffled positives: 4 / 8
```

Strict：

```text
feature prompt-runs: 144
source prompt-runs: 128
source usable pairs: 241
feature specificity positives: 8 / 8
source-control positives: 4 / 4
real-vs-shuffled positives: 4 / 8
```

这说明：

```text
Qwen-PLT 不是完全没信号。
它在同一类 paperpack 上有证据区域敏感的 feature/source-control 支持。
```

但这还不是完整 Gemma-style route，因为当时没有完整 Qwen source-tracing adapter。

### 6.3 Qwen Stage4：cutter nodes 成立

我们后来从 Qwen 自己的结果里找“剪掉会伤答案”的节点。

主结果：

```text
main candidates: 12
decision: causal_cutter_supported
clean_source_minus_controls mean: 0.7951
CI: [0.3819, 1.2743]
positive fraction: 0.9167
```

rank<=10 sensitivity：

```text
candidates: 17
decision: causal_cutter_supported
clean_source_minus_controls mean: 0.5748
CI: [0.2672, 0.9424]
positive fraction: 0.8824
```

小白话解释：

```text
Qwen-PLT 里确实有一些内部节点，剪掉后答案会受伤。
```

但是：

```text
这些节点还没稳定通过 evidence-mask restore。
```

也就是说：

```text
它们像答案支撑节点，但还不能证明它们完整连接了“证据区域 -> 答案”。
```

### 6.4 Qwen Stage4：all-layer hidden retest 是最强正结果

早期 Qwen 实验偏向 layer 26。为了避免漏掉真正层，我们重做了 all-layer hidden sweep：

```text
layers: 0..27
position groups: visual_span, answer_adjacent, top_hidden_delta, visual+answer
masks: answer, union, shifted, shuffled
```

最后发现最稳的是：

```text
layer: 14
direction: masked -> clean restore
position group: top_hidden_delta
mask condition: answer_mask
```

Primary：

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

Strict confirmation：

```text
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

这非常重要。

它说明：

```text
Qwen2.5-VL 内部确实有稳定的 evidence-region-sensitive answer-support causal action。
```

更简单说：

```text
Qwen 也有“图像证据影响答案”的内部因果流。
```

### 6.5 Qwen Stage4：hidden-to-PLT mediation

既然 hidden route 成立，我们继续问：

```text
这条 route 能不能被 PLT sparse features 表示出来？
```

我们比较了三类操作：

```text
hidden_residual:
  直接补回完整 hidden residual。

plt_topk_reconstruction:
  只用 topK PLT features 重构补回。

plt_reconstruction_error:
  补回 PLT features 没解释掉的 error / non-feature residual。
```

结果：

```text
primary decision: qwen_route_may_live_in_plt_error
strict decision: qwen_route_may_live_in_plt_error
```

Strict 关键数字：

```text
hidden_residual / answer_mask / target_effect:
  mean 0.8205, CI low 0.5860

hidden_residual / answer_mask / real > shifted:
  mean 0.5204, CI low 0.3007

hidden_residual / answer_mask / real > shuffled:
  mean 0.4943, CI low 0.2643

plt_reconstruction_error / top8 / answer_mask / target_effect:
  mean 0.8105, CI low 0.5794

plt_reconstruction_error / top8 / answer_mask / real > shifted:
  mean 0.5094, CI low 0.2898

plt_reconstruction_error / top8 / answer_mask / real > shuffled:
  mean 0.4856, CI low 0.2603
```

解释：

```text
Qwen 的因果效果在 hidden residual 上成立。
PLT reconstruction error 几乎复现了 hidden residual 的效果。
但 sparse topK PLT features 没有稳定通过。
```

这指向一个重要判断：

```text
Qwen 的关键因果子空间可能没有被当前公开 Qwen-PLT 拆成少数 sparse features。
```

### 6.6 Qwen Stage4：evidence-specific allpool validation

后来我们专门问：

```text
是不是我们漏掉了真正 evidence-sensitive 的 PLT 节点？
```

于是做了 L10-L17 中层全池 targeted validation。修复 selection bug 后，真正覆盖：

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

先看 evidence sensitivity：

```text
answer_mask real > shifted:
mean = 0.0155
CI low = 0.0117

union_mask real > shifted:
mean = 0.0097
CI low = 0.0056
```

这说明：

```text
Qwen-PLT 里确实有大量对真实证据区域敏感的 feature candidates。
```

再看 causal gate：

```text
clean source > controls:
mean = -0.00093
CI low = -0.00349

answer restore source > controls:
mean = -0.00053
CI low = -0.00272

correct > wrong:
mean = -0.00065
CI low = -0.00357
```

这说明：

```text
这些节点对证据区域敏感，但单独剪/补它们时，不能稳定控制答案。
```

最终判定：

```text
qwen_evidence_sensitive_but_not_causal
```

## 7. Gemma 和 Qwen 到底差在哪

### 7.1 Gemma 是“稀疏路线图”成功

Gemma 的结果像：

```text
证据区域 -> source nodes -> answer token
```

这条路能被 source tracing graph 画出来，也能被 intervention 验证。

### 7.2 Qwen 是“整体因果流”成功，但稀疏图没闭合

Qwen 的结果像：

```text
证据区域 -> hidden residual / reconstruction error -> answer token
```

也就是说，路线存在，但当前 PLT feature 分解没有把它清楚地拆成少数节点。

### 7.3 这不是失败，而是机制异质性

最容易误解的是：

```text
Qwen 没有 Gemma 那样的 sparse route = Qwen 没有机制。
```

这是错的。

因为 Qwen hidden-level gate 很强，所以更合理的解释是：

```text
Qwen 的机制存在，但表示方式不同。
```

这就是 mechanism heterogeneity：

```text
同类功能机制，跨模型存在；
但内部实现和可解释表示形态不同。
```

## 8. 我们怎么保证严谨性

### 8.1 Primary discovery 和 strict confirmation 分开

我们没有在 strict 上重新挑最好看的结果。

正确做法是：

```text
primary:
  用来发现候选 gate / layer / condition。

strict:
  只验证 primary 选出来的规则。
```

这避免了“在验证集上挑结果”的问题。

### 8.2 用 shifted/shuffled mask 排除普通遮挡效应

如果只遮证据区域，别人可以说：

```text
你只是把图片弄坏了，所以模型变了。
```

所以我们做：

```text
shifted_mask:
  同样形状，挪到别处。

shuffled_mask:
  面积相近，随机别处。
```

如果真实证据遮挡比这些 control 更强，才说明是 evidence-region-specific。

### 8.3 用 matched/random controls 排除普通节点效应

如果只剪 source node，别人可以说：

```text
随便剪一个强节点都会影响答案。
```

所以我们比较：

```text
source feature
matched feature
random active feature
same feature random position
```

只有 source 明显强于 control，才过 causal gate。

### 8.4 用 wrong target 排除泛化扰动

如果干预正确答案和错误答案一样强，说明它可能只是让模型整体不稳定。

所以我们检查：

```text
correct target effect > wrong target effect
```

这条不是所有弱 claim 都必须，但如果要写 gold-answer-specific route，它很重要。

### 8.5 做 all-layer sweep 排除 layer miss

早期 Qwen 主要看 layer 26，这可能漏掉真正位置。

我们后来做了：

```text
Qwen layers 0..27 all-layer hidden sweep
```

结果发现最强 confirmed gate 在：

```text
layer 14
```

这说明我们确实修正了早期 layer 偏置。

### 8.6 不复用 Gemma 地图

Qwen 实验没有用 Gemma 的节点 id。

我们只复用判据：

```text
source/control
real-vs-shifted/shuffled
correct-vs-wrong
rank/logit bridge
primary/strict split
```

但 Qwen 的 candidates 来自：

```text
Qwen 自己的 hidden activations
Qwen 自己的 PLT features
Qwen 自己的 evidence sensitivity
Qwen 自己的 intervention results
```

### 8.7 修复并重跑 selection bug

Stage4-052 初始 `--selection all` 有 runner bug：

```text
外层 manifest 是 all。
内层 validation 仍硬编码 main。
```

这会导致“看起来全池，实际每层只跑 20 个”。

我们发现后修复，并重跑 `allpool`：

```text
真正覆盖 2147 candidates。
```

这个很重要，因为它说明我们没有拿错误 all 结果当结论。

### 8.8 用置信区间和 positive fraction，不只看均值

我们不是只看某个均值是不是正。

主要看：

```text
mean
bootstrap CI low
positive fraction
paired comparisons
```

如果 CI 跨 0，就不升级强结论。

这也是为什么 Qwen evidence-specific nodes 虽然有很多强例子，我们仍然不写 evidence-linked causal route。

## 9. 我们做了哪些工作量

这部分可以给合作者讲，说明不是“跑了几个 case”。

### 9.1 Gemma 侧

完成了：

```text
Gemma primary72 full source tracing
Gemma strict72 graph/compare sensitivity
Gemma attribution graph generation
Gemma A/B controlled compare
Gemma node/edge detailed analysis
Gemma intervention smoke
Gemma strict resume / disk full repair / metadata repair
```

规模：

```text
primary: 71 valid samples, 4126 node rows, 7071 edge rows
strict: 71 valid samples, 4095 node rows, 7003 edge rows
```

工程修复：

```text
lazy encoder / lazy decoder 减少显存峰值
strict B-side resume
graph metadata target_token_id repair
远端磁盘和 SIGKILL(137) 诊断
```

### 9.2 Qwen 侧

完成了多轮实验，不是单次跑：

```text
Qwen approximate PLT paperpack primary/strict
Qwen source-tracing adapter attempts
Qwen hookfix / top8 / expanded top32 reruns
Qwen native causal cutter validation
Qwen evidence-first route discovery
Qwen Adapter V3 / V4 route probe
Qwen all-layer hidden causal lattice sweep
Qwen hidden-to-PLT mediation
Qwen all-layer bounded exhaustive PLT search
Qwen middle-layer dense scan
Qwen evidence-specific node allpool validation
```

覆盖范围：

```text
hidden sweep: layers 0..27
middle PLT scan: L10-L17
evidence-specific allpool: 2147 candidates
paperpack prompt-runs: primary/strict 72 samples x 2 prompts where applicable
```

工程修复：

```text
Qwen adapter direction diagnostics
hook alignment 修复
selection all bug 修复
detached/nohup 夜跑 runner
fetch/status/resume 支持
smoke -> full -> strict 的执行链
```

### 9.3 方法学工作量

我们不是只加样本，而是不断补严谨性：

```text
从 layer 26 改成 all-layer。
从手动候选改成 Qwen-native candidates。
从少量 cutter 改成全池 evidence-specific candidates。
从只看 feature 改成 hidden residual / PLT topK / reconstruction error 对比。
从 primary-only 改成 primary discovery + strict confirmation。
```

## 10. 现在能说什么

### 10.1 最稳的科学结论

可以说：

```text
在 Gemma3-PLT 上，我们建立了完整的 evidence-region-sensitive source-tracing 主证据链。
```

可以说：

```text
在 Qwen2.5-VL 上，我们验证到稳定的 hidden-level evidence-to-answer 因果流。
```

可以说：

```text
Qwen-PLT 中存在大量 evidence-sensitive feature candidates，也存在 causal-screened cutter nodes。
```

可以说：

```text
当前 Qwen-PLT sparse feature basis 没有把这条因果流稳定定位成 Gemma-style sparse route。
```

可以说：

```text
这支持跨模型机制存在，但也支持表示异质性。
```

### 10.2 最推荐口径

中文：

```text
Gemma 上，我们能把图像证据到答案的路线追出来，并通过节点干预验证。
Qwen 上，同类证据到答案的因果流也存在，而且 hidden 层证据很强。
但 Qwen 的这条流目前没有被公开 Qwen-PLT 拆成 Gemma 那样的稀疏节点路线。
所以结论是：主机制可能跨模型存在，但不同模型的内部表示形态不同。
```

英文：

```text
Gemma provides a fully traceable sparse PLT source route from evidence regions to answers. Qwen shows a robust hidden-level evidence-to-answer causal route, but the current public Qwen-PLT does not localize this route into a Gemma-like sparse feature graph. This supports cross-model evidence-to-answer mechanisms with representation-dependent implementations.
```

## 11. 现在不能说什么

不能说：

```text
Qwen fully replicates Gemma-style source tracing.
```

原因：

```text
Qwen automatic route extraction 没有通过完整 gates。
```

不能说：

```text
Qwen 没有 evidence-to-answer mechanism。
```

原因：

```text
Qwen hidden layer 14 primary + strict 很强。
```

不能说：

```text
所有模型都有同一种 sparse PLT feature route。
```

原因：

```text
Gemma 和 Qwen 的表示形态明显不同。
```

不能说：

```text
Qwen 的 PLT 失败就是科学负结果。
```

原因：

```text
这更像当前公开 PLT feature basis / intervention operator / localization 方法没有捕获主要因果子空间。
```

## 12. 合作者可能会问的问题

### Q1：这是不是说明 Qwen 没有 Gemma 那样的机制？

不是。

更准确地说：

```text
Qwen 有 evidence-to-answer 因果流，但它不是以当前 Qwen-PLT sparse route 的形式稳定出现。
```

### Q2：这对主结论是加强还是削弱？

加强，但不是以“一比一复现”的方式加强。

它加强的是：

```text
evidence-to-answer causal flow 不是 Gemma-only。
```

它不支持的是：

```text
所有模型都有同一种稀疏 source-tracing route。
```

### Q3：为什么 Gemma 能画路线，Qwen 不能？

可能原因是表示方式不同。

Gemma 的 PLT sparse features 更好地暴露了这条路线。  
Qwen 的主要因果效果可能保留在 hidden residual 或 PLT reconstruction error 里。

也就是说：

```text
不是没有路，而是路没有被当前地图画出来。
```

### Q4：我们有没有遍历足够多？

已经比早期严谨很多。

我们做了：

```text
Qwen hidden layers 0..27 all-layer sweep。
Qwen middle PLT L10-L17 dense / targeted scan。
Qwen evidence-specific allpool 2147 candidates。
```

所以目前不是“只看了 layer 26”或“只挑了几个好看的节点”。

但也要诚实说：

```text
逐个 forward 干预所有层、所有位置、所有 feature 在工程上不可行。
我们做的是 bounded exhaustive + targeted validation。
```

### Q5：Qwen 后续怎么推进？

不要再只扩大候选数。

更值得做的是：

```text
multi-feature grouped patch
multi-position patch
residual-direction patch
scaled operator sweep
PLT reconstruction error / non-feature residual analysis
CLT 机制异质性图谱
```

也就是说，下一步不应该执着于“找一个像 Gemma 的单节点路线”，而是问：

```text
Qwen 的 route 到底是分布式、多位置、多层，还是主要藏在非 feature residual 里？
```

## 13. 一页幻灯片版

标题：

```text
Evidence-to-answer causal routes exist across models, but their interpretable form differs.
```

核心图景：

```text
Gemma:
  evidence region -> sparse PLT source route -> answer

Qwen:
  evidence region -> hidden residual / reconstruction error -> answer
  sparse PLT features show sensitivity, but do not close the full route
```

主结论：

```text
Gemma gives a fully traceable sparse route.
Qwen gives cross-model hidden-level causal support.
Together, they support a broader evidence-to-answer mechanism with model-dependent representation.
```

严谨性：

```text
primary/strict split
shifted/shuffled masks
matched/random controls
wrong-target controls
all-layer sweep
Qwen-native candidates
2147-candidate allpool validation
conservative claim boundaries
```

边界：

```text
Not Qwen fully replicates Gemma-style source tracing.
Not Qwen lacks mechanism.
Yes representation-dependent evidence-to-answer route.
```

## 14. 最终建议表达

如果只能用一段话讲：

```text
我们的实验说明，视觉语言模型内部存在从图像证据区域到答案的因果流。Gemma 是目前最清楚的案例：我们能在 PLT 表示里追出一条稀疏路线，并通过遮证据、剪节点、做 controls 验证。Qwen 也有同类因果流，而且在 hidden layer 14 上 primary 和 strict 都非常稳定；但它没有被当前公开 Qwen-PLT 拆成 Gemma 那样的少数稀疏 feature 节点。进一步的 hidden-to-PLT 实验显示，Qwen 的主要因果效果可能保留在 reconstruction error 或更整体的 hidden residual 中。因此，这个结果支持 evidence-to-answer 机制不是 Gemma-only，同时也说明不同模型的机制表示方式可能不同。
```

如果再短一点：

```text
Gemma 证明路线可以被完整追踪；Qwen 证明同类因果流也存在，但路线图不一定长得像 Gemma。
```

