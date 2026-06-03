# 实验 010：Stage 2A targeted replication verdict

## 目的

本文件把 Stage 2A 的 targeted replication pack 做一次正式判定，避免继续在分散 CSV、单步 readout 和 case 观察之间反复改口径。

Stage 2A 的核心问题不是：

```text
D_visual_only 是否比 B_direct 更好？
```

而是：

```text
在一批新的 localized / strong-image-dependence / existing-mask 样本中，
answer-adjacent support routes 是否仍然表现出 evidence-region sensitivity？
这种敏感性是否强于 nearest-control node 和 random-region control？
这种区域遮挡是否也会伤害 target rank / margin / decoded answer？
```

最终判定口径：

```text
Stage 2A = main partial replication + behavior support
不是 full independent strong replication
```

这句话很重要。它说明 Stage 2A 确实补强了主线，但还不能替代 core24 作为 strongest evidence，也不能让我们把论文 claim 放大成“广义 VLM 视觉机制已被强证明”。

---

## 输入与上游实验

Stage 2A verdict 依赖以下上游实验：

| 编号 | 文档 | 作用 |
|---|---|---|
| 001 | `001_stage2a_candidate_selection.md` | 从已有标注和 current backbone 中构建 top24 targeted replication candidate pool |
| 002 | `002_stage2a_pretrace_remote_start.md` | 在服务器启动 top24 B/D eval 和 answer-aligned trace |
| 003 | `003_stage2a_pretrace_eval_partial_readout.md` | 读取 B/D 行为初筛，确认 D 不支持“行为更优”叙事 |
| 004 | `004_stage2a_pretrace_trace_compare_readout.md` | 确认 B/D trace 均 24/24 完成，产出 source zeroing 候选 |
| 005 | `005_stage2a_source_zeroing_top4_readout.md` | 得到 support / suppressor source nodes |
| 006 | `006_stage2a_nearest_control_clean_readout.md` | 得到 nearest matched-control clean-screen，可用 support+mask+nearest samples = 8 |
| 007 | `007_stage2a_region_replication_pack_and_remote_start.md` | 构建 nearest8 region-mask pack 并启动服务器 region run |
| 008 | `008_stage2a_region_route_readout.md` | 读取 route weakening、source>nearest、random4 coverage |
| 009 | `009_stage2a_behavior_generation_readout.md` | 读取 target rank / margin / decoded answer change 和 route-behavior linkage |

关键数据目录：

```text
E:\Bridging\remote_sync\2026-05-19_stage2a_pretrace_top24
E:\Bridging\annotation\stage2a_region_replication_top24_nearest8
E:\Bridging\annotation\stage2a_region_replication_top24_nearest8\analysis_route
E:\Bridging\annotation\stage2a_region_replication_top24_nearest8\analysis_behavior
```

---

## Stage 2A 数据链条

### 1. top24 pretrace pool

Stage 2A 首先从 current-mainline 相关样本中构建 top24 pretrace pool。

完成情况：

```text
B_direct eval valid = 24/24
D_visual_only eval valid = 24/24
B_direct format ok = 24/24
D_visual_only format ok = 18/24
B_direct correct = 8/24
D_visual_only correct = 5/24
B_direct trace = 24/24
D_visual_only trace = 24/24
```

解释：

```text
B/D 都能跑完整个 trace pipeline；
但 D_visual_only 在格式和正确率上没有稳定优于 B_direct；
因此 Stage 2A 不应写成 prompt winner 实验。
```

### 2. source zeroing

source zeroing 用来判定 traced source nodes 的方向：

```text
delta_target_logit < 0 -> support node
delta_target_logit > 0 -> suppressor node
delta_target_logit = 0 -> neutral node
```

完成情况：

```text
source zeroing rows = 87
support rows = 46
suppressor rows = 35
neutral rows = 6
existing mask + support samples = 17
clean_prompt + existing mask + support samples = 6
```

解释：

```text
Stage 2A 新 pool 里仍然出现 signed answer-adjacent routes；
support 和 suppressor 都存在；
但 strict clean-prompt aligned 可用样本只有 6 个，因此不能把 B/D target-aligned prompt comparison 写成 primary。
```

### 3. nearest-control clean-screen

nearest-control 的作用是排除：

```text
不是 traced source node 特异有效，
而是任意附近 feature 都会产生类似干预效果。
```

完成情况：

```text
source non-neutral rows = 81
nearest matched rows = 27
matched samples = 10
support nearest rows = 21
suppressor nearest rows = 6
existing mask + support + nearest samples = 8
clean_prompt + existing mask + support + nearest samples = 1
```

解释：

```text
严格同时要求 existing mask、support source 和 nearest control 后，样本数降到 8；
这足够启动一个小而严格的 targeted replication pack；
但不够支撑 broad statistical generalization。
```

### 4. nearest8 region pack

构建出的正式 region pack：

```text
samples = 8
source-control pairs = 20
manifest rows = 40
support pairs = 16
suppressor pairs = 4
support samples = 8
```

support samples：

```text
okvqa_val_1740705
okvqa_val_1927165
okvqa_val_2131565
okvqa_val_2683965
okvqa_val_343215
okvqa_val_3794755
okvqa_val_5735275
okvqa_val_80655
```

mask caveat：

```text
多个 answer masks 面积较大；
它们仍是 localized 标注，不是 diffuse 全图问题；
但 same-area random non-overlap controls 会因此很难生成有效 strict random4。
```

---

## Route verdict

### 1. 运行完整性

region-route job 完整跑完：

```text
expected rows = 40 manifest rows x 8 conditions = 320
actual done rows = 320
skip rows = 0
clean calibration exact match = 1.0
```

解释：

```text
本轮没有 position_out_of_range、mask 缺失、target 对齐失败或 clean calibration 漂移。
如果结果偏弱，应解释为机制异质性或样本几何问题，而不是 pipeline 崩溃。
```

### 2. support source weakening

| metric | unit | n | mean | median | positive_rate | bootstrap 95% CI | 判定 |
|---|---|---:|---:|---:|---:|---|---|
| support source answer-mask weakening | pair | 16 | +0.480 | +0.438 | 0.625 | [+0.109, +0.852] | 支持 |
| support source answer-mask weakening | sample_mean | 8 | +0.237 | +0.156 | 0.500 | [-0.281, +0.760] | 弱 / 异质 |
| support source union-mask weakening | pair | 16 | +0.574 | +0.500 | 0.625 | [+0.254, +0.902] | 支持 |
| support source union-mask weakening | sample_mean | 8 | +0.536 | +0.313 | 0.625 | [+0.078, +1.049] | 本轮最稳 |

判定：

```text
Stage 2A 方向性复现了 evidence-region-sensitive support route。
union mask 的复现比 answer mask 更稳。
answer mask 在 pair-level 稳定为正，但 sample-level 受异质性影响较大。
```

保守读法：

```text
新 pack 中，证据区域遮挡确实会削弱 support source routes；
但这个效应不是每个样本都成立，不能写成 universal。
```

### 3. source > nearest-control

| metric | unit | n | mean | median | positive_rate | bootstrap 95% CI | 判定 |
|---|---|---:|---:|---:|---:|---|---|
| source-minus-nearest answer-mask weakening | pair | 16 | +0.266 | +0.406 | 0.625 | [-0.117, +0.613] | 方向支持 |
| source-minus-nearest answer-mask weakening | sample_mean | 8 | +0.184 | +0.396 | 0.625 | [-0.254, +0.579] | partial |
| source-minus-nearest union-mask weakening | pair | 16 | +0.164 | +0.188 | 0.563 | [-0.168, +0.473] | weak |
| source-minus-nearest union-mask weakening | sample_mean | 8 | +0.064 | +0.156 | 0.625 | [-0.313, +0.409] | weak |

判定：

```text
source > nearest-control 的方向为正，但 bootstrap CI 跨 0。
Stage 2A 不能强声称 node specificity replicated。
```

保守读法：

```text
nearest8 没有推翻 source specificity；
但它只提供方向性支持，strongest node-specificity evidence 仍主要来自 core24 / previous mainline。
```

### 4. answer/union > random-region control

strict `IoU <= 0.05` random4 覆盖：

```text
support source valid rows = 4/16
support source valid samples = 2/8
```

判定：

```text
strict random4 在 Stage 2A nearest8 中 underpowered。
原因主要是若干 answer masks 面积大，same-area random rectangle 很难避开 answer ∪ relate。
```

strict random4 读数：

| metric | unit | n | mean | median | positive_rate | bootstrap 95% CI | 判定 |
|---|---|---:|---:|---:|---:|---|---|
| answer-mask minus random4 | pair | 4 | +0.344 | +0.406 | 0.750 | [-0.016, +0.641] | diagnostic |
| union-mask minus random4 | pair | 4 | +0.359 | +0.406 | 0.750 | [+0.078, +0.594] | diagnostic |

保守读法：

```text
方向与主 claim 一致；
但有效样本太少，不能把 Stage 2A 写成 random-region specificity 的强复现。
core24 仍是 random4/random16 specificity 的 strongest evidence。
```

---

## Behavior verdict

### 1. 运行完整性

behavior / generation job 完整：

```text
sample-runs = 9
conditions = 8
behavior rows = 72
generation rows = 72
empty rows = 0
error rows = 0
```

解释：

```text
区域遮挡没有导致格式崩坏、空答案或运行失败。
因此 decoded answer change 可以作为真实行为变化的读数，而不是 pipeline artifact。
```

### 2. target rank / margin

| metric | n | mean | median | positive_rate | bootstrap 95% CI | 判定 |
|---|---:|---:|---:|---:|---|---|
| answer-mask rank damage | 9 | +34.56 | +2 | 0.667 | [-0.11, +89.89] | 方向支持 |
| union-mask rank damage | 9 | +20.89 | +13 | 0.889 | [+6.56, +35.33] | 支持 |
| answer-mask margin drop | 9 | +2.69 | +1.75 | 0.778 | [-0.62, +6.53] | 方向支持 |
| union-mask margin drop | 9 | +3.92 | +3.13 | 0.778 | [+0.92, +7.00] | 支持 |

判定：

```text
union mask 稳定伤害 target rank 和 target-vs-competitor margin；
answer mask 方向一致，但 CI 跨 0。
```

### 3. decoded generation

| condition | rows | changed_rows | changed_rate | empty_rows | error_rows | 判定 |
|---|---:|---:|---:|---:|---:|---|
| answer_mask | 9 | 5 | 0.556 | 0 | 0 | 支持 |
| relate_mask | 9 | 4 | 0.444 | 0 | 0 | 描述性 |
| union_mask | 9 | 6 | 0.667 | 0 | 0 | 支持 |
| strict random valid rows | 8 | 2 | 0.250 | 0 | 0 | diagnostic |
| all random rows | 36 | 14 | 0.389 | 0 | 0 | appendix diagnostic |

判定：

```text
answer/union evidence-region masks 经常改变 decoded answer；
union mask 改变率最高；
没有 empty/error，因此不是格式崩坏解释。
```

### 4. route-behavior linkage

| route metric | behavior metric | n | Pearson | Spearman | 判定 |
|---|---|---:|---:|---:|---|
| answer weakening | answer rank damage | 9 | +0.481 | +0.828 | promising |
| union weakening | union rank damage | 9 | -0.354 | -0.167 | 不支持简单线性解释 |
| answer weakening | answer margin drop | 9 | +0.453 | +0.117 | weak |
| union weakening | union margin drop | 9 | -0.496 | -0.577 | 不支持简单线性解释 |

判定：

```text
answer-mask route weakening 与 answer-mask rank damage 有方向一致信号；
union-mask 行为损伤虽强，但不被当前 traced support route 的线性 weakening 简单解释。
```

保守读法：

```text
Stage 2A 支持“区域遮挡同时影响 route 与行为”，
但还不能说“当前这一组 traced nodes 完全解释了所有 union-mask 行为变化”。
```

---

## Case-level verdict

### 强正向或适合主图候选

| sample_id | 读法 |
|---|---|
| `okvqa_val_1927165` | answer weakening 强，decoded answer 从 `obey the law` 变成 stop-sign 相关替代表达；适合展示 evidence mask 会改变答案路线，但需要注意语义近邻答案 |
| `okvqa_val_2683965` | answer / union weakening 强，rank damage 强；clean `a rectangle`，answer mask 后 `a square`，是很好的视觉读出/形状 case |
| `okvqa_val_80655` | decoded answer 在 baseball action 上变化，适合做行为侧例子 |
| `okvqa_val_3794755` | union 行为伤害强，答案变为 `a television`；但 route weakening 近 0，适合作为“行为变化不总由当前 traced support route 线性解释”的反例或补充 case |

### 重要反例

| sample_id | 读法 |
|---|---|
| `okvqa_val_343215` | union route weakening 强，但 target rank 反而改善，说明 route 和行为关系不总是单调 |
| `okvqa_val_2131565` | answer / union route weakening 偏弱或反向，是 heterogeneity 的重要样本 |
| `okvqa_val_5735275` | strict random4 可用，但 route weakening 弱，适合解释为什么 Stage 2A 只能 partial |

---

## Success criteria 对照

| run plan criterion | Stage 2A evidence | status | conservative reading |
|---|---|---|---|
| support source `answer_mask` weakening 为正 | pair mean `+0.480`, CI `[+0.109,+0.852]`; sample mean `+0.237`, CI crosses 0 | partial success | answer-mask 在 pair-level 成立，但 sample-level 异质 |
| support source `union_mask` weakening 为正 | pair mean `+0.574`, CI `[+0.254,+0.902]`; sample mean `+0.536`, CI `[+0.078,+1.049]` | success | union evidence-region sensitivity 是 Stage 2A 最稳 route 结果 |
| source weakening > nearest-control weakening | answer sample mean `+0.184`, union sample mean `+0.064`, CI both cross 0 | partial / weak | 方向支持，但不能强声称 node specificity replicated |
| answer/union weakening > strict random4 | strict random4 support source samples `2/8`; pair-level direction positive | not strongly supported in Stage 2A | random specificity 仍依赖 core24；Stage 2A 只给 diagnostic |
| evidence mask harms rank / margin | union rank CI `[+6.56,+35.33]`, union margin CI `[+0.92,+7.00]` | success | union mask 行为伤害稳定 |
| evidence mask changes decoded answer without format collapse | answer `5/9`, union `6/9`, empty/error = 0 | success | 区域遮挡进入自然生成行为 |
| route weakening links to behavior | answer weakening vs answer rank damage Spearman `+0.828` | exploratory support | 方向 promising，但样本少，不能写成强预测模型 |

---

## 总判定

Stage 2A 的正式判定：

```text
main partial replication + behavior support
```

具体来说：

1. Stage 2A 在新 targeted pack 中复现了 support source route 的 evidence-region weakening，尤其是 `union_mask`。
2. Stage 2A 补上了行为侧证据：`union_mask` 稳定伤害 target rank / margin，`answer_mask` 和 `union_mask` 经常改变 decoded answer，且没有格式崩坏或空答案。
3. Stage 2A 的 node specificity 只方向性支持：source > nearest-control 为正但 CI 跨 0。
4. Stage 2A 的 strict random-region specificity 不足：random4 覆盖太低，不能作为 strongest evidence。
5. 因此 core24 仍是 strongest evidence，Stage 2A 是 targeted partial replication 和 behavior reinforcement。

---

## 能说什么

可以在主报告中写：

```text
在 Stage 2A targeted replication pack 中，support source routes 在证据区域遮挡下整体变弱，尤其 union mask 的 sample-level bootstrap CI 不跨 0。
同一批区域遮挡也会伤害目标答案 rank / margin，并经常改变 decoded answer。
这说明 core24 中观察到的 evidence-region sensitivity 不是孤立现象，但新 pack 仍表现出明显异质性。
```

也可以写：

```text
Stage 2A 支持主结论的收窄版本：localized strong-image-dependence VQA 中存在 evidence-region-sensitive support routes。
```

---

## 不能说什么

不能写：

```text
Stage 2A 已经强复现了 random-region specificity。
Stage 2A 已经强复现了 source > nearest-control。
D_visual_only 比 B_direct 更好。
这些 source nodes 是对象级语义节点。
node intervention 已经直接改变 decoded generation。
union-mask 行为变化完全由当前 traced support routes 解释。
```

这些说法要么证据不足，要么不是本轮实验回答的问题。

---

## 对 Stage 2 后续的影响

Stage 2A 之后，最应该推进的不是继续扩大同一个 nearest8 pack，而是补主链最缺的一环：

```text
Stage 2B: node-to-generation bridge
```

推荐先做 smoke，不直接上大实验：

```text
cases: okvqa_val_1927165, okvqa_val_2683965, okvqa_val_80655, okvqa_val_3794755
conditions: clean generation, support source zeroing generation, nearest-control zeroing generation
primary readout: first answer token distribution / target token rank / decoded short answer
success: source zeroing causes stronger target-rank or decoded-answer damage than nearest control
```

如果 Stage 2B 工程不可行，则转向 Stage 2C：

```text
选择 2-3 个强 case 做 semantic feature / region visualization，
证明 source route 至少在功能上与 answer mask 区域激活或遮挡相关，
但仍不把它写成 object-level semantic node。
```

---

## 一句话总结

Stage 2A 没有把主结论升级成“强独立复现”，但它完成了一个有价值的收束：新 pack 中 support route 的 evidence-region weakening 方向复现，union mask 最稳，并且区域遮挡会实际伤害 rank、margin 和 decoded answer。下一步必须补 `node intervention -> generation` 的直接桥，否则“路径解释最终回答”的 claim 仍只能保守写成关联性增强，而不是完整因果闭环。
