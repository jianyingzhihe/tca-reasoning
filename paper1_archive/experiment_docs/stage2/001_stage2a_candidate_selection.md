# 实验 001：Stage 2A Targeted Replication Candidate Selection

日期：2026-05-19  
状态：已完成 candidate selection；等待服务器 pre-trace  
对应计划：Stage 2A

---

## 1. 目的

Stage 2A 的目标是构造一个独立 targeted replication pack，用来验证 core24 的 evidence-region-sensitive support route 结果不是少数样本或选择偏差造成的。

本实验是 Stage 2A 的第一步，不直接产生机制结论，而是回答：

```text
1. 当前有没有足够多的 immediate-ready replication samples？
2. 如果没有，下一批应该优先 trace 哪些样本？
3. 哪些样本已经有 answer / relate mask，可以减少新的人工标注成本？
4. 哪些样本应该排除或降级，因为证据区域太弥散？
```

---

## 2. 输入

核心输入：

```text
doc/5.16/core24_prefixfix_region_analysis_2026-05-19/current_backbone_replication_candidate_rows.csv
doc/5.16/core24_prefixfix_region_analysis_2026-05-19/REGION_REPLICATION_FEASIBILITY_2026-05-19.md
doc/5.16/expanded_localized_discovery_screen20_readout_2026-05-18/annotation_candidate_sheet.csv
doc/5.16/expanded_localized_discovery_pool_2026-05-18/expanded_localized_discovery_pool_top.csv
doc/5.16/round400_localized_discovery_pool_2026-05-18/round400_localized_discovery_pool_all.csv
annotation/okvqa_evidence_labelme_round4_core24_easy/manifest.csv
annotation/okvqa_evidence_labelme_round5_expanded20_route7/manifest.csv
annotation/okvqa_evidence_labelme_round4_core16_extra/manifest.csv
annotation/okvqa_evidence_labelme_round4_ultraeasy16_fresh/manifest.csv
annotation/okvqa_evidence_labelme_round4_mainline16/manifest.csv
```

这些输入分成三类：

1. 当前 backbone 已经有 source/control 的样本；
2. 已经有人类 answer / relate 标注、但没有当前 source/control overlap 的样本；
3. localized discovery pool 中尚未标注、但视觉证据可能 compact 的样本。

---

## 3. 输出

输出目录：

```text
doc/experiments/stage2/stage2a_candidate_selection/
```

输出文件：

```text
stage2a_candidate_samples.csv
stage2a_candidate_tier_counts.csv
stage2a_immediate_ready.csv
stage2a_pretrace_queue_top40.csv
stage2a_trace_queue_top24.csv
stage2a_trace_selected_ids_top24.csv
manifest_B_direct_stage2a_pretrace_top24.csv
manifest_D_visual_only_stage2a_pretrace_top24.csv
REMOTE_STAGE2A_PRETRACE_COMMANDS.md
```

---

## 4. 方法

### 4.1 固定排除规则

主 replication pack 不使用：

```text
1. 已经进入 core24 主分析的样本；
2. 已经进入 route7 heterogeneous discovery run 的样本；
3. 用户明确指出为 diffuse/global 的样本，例如 okvqa_val_1235705；
4. terrain / tint / time-of-day / broad place / resort 这类容易全局弥散的题；
5. entity_linking 或高知识先验太强、证据区域不清楚的题。
```

### 4.2 分层候选

候选被分成四类：

```text
immediate_ready_backbone:
  当前已有 support source + nearest control，且未进入 core24/route7。

pretrace_existing_mask:
  已有 answer/relate mask，但之前没有当前 source/control overlap。
  这些样本需要先重新 trace；如果得到 support source + nearest，就可以不用再标注直接跑 region replication。

pretrace_discovery_pool:
  来自 localized discovery pool，尚未有人类 mask。
  这些样本需要先 trace，再决定是否送标注。

already_route7_reference:
  已经跑过 route7，不能作为独立 replication 主样本，只保留作参考。
```

### 4.3 Trace queue 筛选

`stage2a_trace_queue_top24.csv` 不是简单按分数前 24，而是额外加入了人工友好的启发式。

优先：

```text
brand / logo / sign / language
arrow / shape
dog / bear / bike / racket
visible action / object state
small appliance / electronic devices
已有 answer/relate mask 的样本
```

暂缓：

```text
wood type
female animal naming
cake flavor
usual color
occupation/job
generic material
global place / terrain / tint
```

这个选择是为了提高 pre-trace 后能得到 compact visual-evidence support route 的概率。

---

## 5. 结果

### 5.1 候选总量

生成的总候选池：

```text
already_route7_reference: 7
immediate_ready_backbone: 2
pretrace_discovery_pool: 50
pretrace_existing_mask: 31
```

### 5.2 Immediate-ready 结果

严格满足“当前 backbone + support source + nearest + 未进 core24/route7”的样本只有两个：

```text
okvqa_val_299845
question: Is this at a salt water beach or a lake?
answer: salt water beach
type: scene_inference / diffuse_global

okvqa_val_3605295
question: What place is this?
answer: store
type: scene_inference / diffuse_global
```

判断：

> 这两个样本不能直接组成 Stage 2A 主 replication pack。

原因：

1. 数量太少；
2. 都是 `diffuse_global`；
3. 和 Stage 2A 主目标的 localized evidence-region claim 不完全匹配；
4. 可以作为 stress / appendix case，但不适合作为主成功判据。

### 5.3 Pre-trace queue top24

最终生成的 `stage2a_trace_queue_top24.csv` 包含：

```text
24 samples
21 pretrace_existing_mask
3 pretrace_discovery_pool
21 visual_readout
3 symbol_text_reading
```

前 24 个样本：

| sample_id | tier | question | answer | type |
|---|---|---|---|---|
| okvqa_val_667695 | pretrace_existing_mask | What already happened to this food item? | bitten | visual_readout |
| okvqa_val_80655 | pretrace_existing_mask | What action is the baseball player doing in this scene? | bat | visual_readout |
| okvqa_val_3313665 | pretrace_existing_mask | What type of bike is this? | penny farthing | visual_readout |
| okvqa_val_1994425 | pretrace_existing_mask | The body hugging garment seen here is called a what? | wetsuit | visual_readout |
| okvqa_val_1083155 | pretrace_existing_mask | What type of dog is pictured? | dalmation | visual_readout |
| okvqa_val_2802115 | pretrace_existing_mask | What type of fence is used to enclose here? | chainlink | visual_readout |
| okvqa_val_3265105 | pretrace_existing_mask | What company makes the white car? | toyota | visual_readout |
| okvqa_val_3326275 | pretrace_existing_mask | What brand is the racket? | wilson | visual_readout |
| okvqa_val_2131565 | pretrace_existing_mask | What is this plate made from? | ceramic | visual_readout |
| okvqa_val_1740705 | pretrace_existing_mask | The stuffed animal in the photo is called what kind of bear? | teddy | visual_readout |
| okvqa_val_2496585 | pretrace_existing_mask | What type of product is being advertised on the bus? | paint | visual_readout |
| okvqa_val_2373185 | pretrace_existing_mask | How many sides do these signs normally have? | 8 | visual_readout |
| okvqa_val_5735275 | pretrace_existing_mask | What is the arrow indicating? | 1 way | visual_readout |
| okvqa_val_1083925 | pretrace_existing_mask | What type of pastries in this image that has a hole in the middle and are round? | donuts | visual_readout |
| okvqa_val_1927165 | pretrace_existing_mask | What does stop mean? | stop | visual_readout |
| okvqa_val_3794755 | pretrace_existing_mask | What electronic devices are pictured here? | laptop | visual_readout |
| okvqa_val_343215 | pretrace_existing_mask | What kind of bear? | teddy | visual_readout |
| okvqa_val_2683965 | pretrace_existing_mask | What shape is the sign? | octagon | visual_readout |
| okvqa_val_1996815 | pretrace_existing_mask | What small appliance is that stuffed animal inside? | microwave | visual_readout |
| okvqa_val_602025 | pretrace_existing_mask | What kind of dog is this? | sheep dog | visual_readout |
| okvqa_val_4033335 | pretrace_existing_mask | What kind of food is this? | dessert | visual_readout |
| okvqa_val_5291225 | pretrace_discovery_pool | What language is the counter sign here written in? | chinese | symbol_text_reading |
| okvqa_val_1985905 | pretrace_discovery_pool | Which brand of car is shown in this picture? | chevy | symbol_text_reading |
| okvqa_val_4549785 | pretrace_discovery_pool | What brand of bike is this? | honda | symbol_text_reading |

### 5.4 为什么这个 queue 合理

这个 queue 的最大优点是：

```text
21/24 已经有 answer / relate mask。
```

这意味着如果 pre-trace 找到 support source + nearest control，我们可以直接进入 region-mask replication，不一定需要立刻再让用户标一批图。

最大缺点是：

```text
visual_readout 占比过高，symbol_text_reading 只有 3 个。
```

所以如果第一轮 trace 后 symbol_text_reading yield 不够，需要再补一个 symbol-heavy pretrace queue。

---

## 6. 预期

成功预期：

```text
24 个 pretrace 样本中，至少 10-15 个 clean-core 可用；
至少 8-10 个样本有 support source；
至少 6-8 个样本有 nearest control；
其中已有 mask 的样本占多数，可以直接构造 Stage 2A region replication pack。
```

partial 预期：

```text
clean-core yield 足够，但 nearest control 少；
或者 support source 足够，但已有 mask 样本不足；
或者 visual_readout 成功，symbol_text_reading 不足。
```

失败预期：

```text
clean-core yield 很低；
support source 不足；
nearest control 不足；
大多数已有-mask 样本无法进入 answer-aligned trace。
```

---

## 7. 实际结果

本实验只完成 selection 和 manifest 准备，尚未运行服务器 pre-trace。

已生成：

```text
manifest_B_direct_stage2a_pretrace_top24.csv
manifest_D_visual_only_stage2a_pretrace_top24.csv
stage2a_trace_selected_ids_top24.csv
```

这三个文件是下一步服务器 eval + answer-aligned trace 的输入。

---

## 8. 预期与实际偏差

原始预期：

```text
可能能从 current backbone 中直接找到 10-15 个 support+nearest candidate。
```

实际：

```text
严格 immediate-ready 只有 2 个，而且都是 diffuse_global。
```

因此 Stage 2A 必须加入一个 pre-trace substage：

```text
Stage 2A-0: candidate selection
Stage 2A-1: B/D eval + answer-aligned pretrace
Stage 2A-2: intervention + nearest/random control screen
Stage 2A-3: region replication pack construction
```

这不是坏结果。它反而让 selection bias 更可控：我们不会把 diffuse immediate-ready 样本硬塞进 localized 主结论，而是先公开承认 immediate-ready 不足，再走 pre-trace。

---

## 9. 结论

本实验结论：

```text
Stage 2A candidate selection 已完成。
当前不能直接进入 region replication，因为 immediate-ready localized sample 不足。
下一步应该在 top24 pretrace queue 上跑 B/D eval + answer-aligned trace。
```

对主 claim 的影响：

```text
没有新增机制证据。
但它提高了 Stage 2A replication 的严谨性，因为 selection rule 已冻结，并且明确排除了 diffuse/global 样本作为主成功证据。
```

---

## 10. 后续动作

下一步按顺序做：

1. 把两个 manifest 和 selected ids 同步到服务器；
2. 分别跑 `B_direct` 和 `D_visual_only` eval；
3. 用 eval CSV 跑 answer-aligned attribution；
4. 生成 clean-core summary；
5. 对 clean-core 样本跑 source zeroing intervention；
6. 构造 nearest/random controls；
7. 从中筛出真正可进入 region replication 的 10-15 个样本；
8. 对已有 mask 样本直接跑 region pipeline，对没有 mask 但机制很强的样本再交给用户标注。

---

## 11. 当前最重要的接受标准

下一阶段成功标准：

```text
至少 8 个 localized samples 同时满足：
1. clean-core answer-aligned trace 成功；
2. support source node 可用；
3. nearest control 可用；
4. answer/relate mask 已有，或证据区域足够清楚可补标。
```

如果达不到：

```text
不要扩 claim；
补一个 symbol_text_reading-heavy pretrace queue；
或者降低 Stage 2A 为 partial replication attempt。
```

