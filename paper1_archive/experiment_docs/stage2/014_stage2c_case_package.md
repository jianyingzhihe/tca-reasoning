# Stage 2C：Semantic / Region Case Package

日期：2026-05-20

## 1. 实验目的

Stage 2C 的目的不是重新开一个大规模统计实验，而是把 Stage 2A 和 Stage 2B 已经跑出的关键结果整理成可以进入主报告或论文图的 case package。

这一步要回答的问题是：

```text
我们能不能选出 1-2 个高质量 case，把“证据区域遮挡 -> route weakening -> 行为变化 / generation bridge”的链条讲清楚？
```

这里的重点是“功能解释”，不是“对象语义解释”。也就是说，本实验可以支持：

```text
某些 answer-adjacent support routes 对人工标注的关键证据区域敏感，并且这种敏感性和生成侧变化有关。
```

但不能支持：

```text
某个 feature node 就是明确的 object detector / OCR detector / 语义概念节点。
```

## 2. 术语解释

`case package`：把单个样本的图像、人工 mask、route metric、behavior metric、node-to-generation metric 和解释性注释放在一起，供主图、正文表格或 appendix 使用。

`token-position mapping`：把 traced feature 的 `feature_pos` 映射回多模态输入序列中的 token 位置，判断它是在 image soft token 上，还是在文本 / assistant prefix token 上。这个映射只能说明位置，不等于说明语义。

`image_soft_token`：VLM 处理图像时插入到文本序列中的图像 token。这里它们是视觉信息进入语言模型序列的位置。

`source-specific decoded change`：清零 source node 后，decoded answer 发生变化；对应 nearest control node 清零时没有同样变化。这是 node-to-generation bridge 中较强的 case-level 证据。

## 3. 输入数据

本实验复用已有 Stage 2A / Stage 2B 产物，不新增远端大跑数。

主要输入：

```text
E:\Bridging\annotation\stage2a_region_replication_top24_nearest8\region_experiment_manifest.csv
E:\Bridging\annotation\stage2a_region_replication_top24_nearest8\analysis_route\route_weakening_iou1.csv
E:\Bridging\annotation\stage2a_region_replication_top24_nearest8\analysis_behavior\behavior_wide_iou0p05.csv
E:\Bridging\annotation\stage2a_region_replication_top24_nearest8\analysis_behavior\generation_cases_iou0p05.csv
E:\Bridging\annotation\stage2b_node_generation_smoke\analysis_first_token_nearest8\first_token_pair_gap_table.csv
E:\Bridging\annotation\stage2b_node_generation_smoke\stage2b_greedy_decode_1927165_pair003.csv
E:\Bridging\annotation\stage2b_node_generation_smoke\stage2b_greedy_decode_2683965_pair010.csv
```

新增远端轻量脚本：

```text
E:\Bridging\vlm-circuit-tracing\circuit_tracer_vlm\scripts\research\map_feature_positions_to_image_tokens.py
```

该脚本已通过本地语法检查：

```text
python -m py_compile E:\Bridging\vlm-circuit-tracing\circuit_tracer_vlm\scripts\research\map_feature_positions_to_image_tokens.py
```

## 4. 方法

### 4.1 选择 case

本轮固定两个 case，不再重新筛选：

```text
003_okvqa_val_1927165_B_support_L11_P196_F151858
010_okvqa_val_2683965_B_support_L26_P287_F39687
```

选择原因：

1. `okvqa_val_1927165` 是当前最干净的 source-specific decoded-answer positive case。
2. `okvqa_val_2683965` 是 first-token bridge 最强 case 之一，但 decoded loop 不呈现 source-specific decoded change，适合作为边界 / dissociation case。

这两个 case 放在一起，可以避免只展示“好看的正例”，也能说明我们对 claim 的边界是清楚的。

### 4.2 远端 token-position mapping

远端执行命令把两个 pair 的 source node 和 nearest control node 都映射到输入 token 序列：

```text
manifest:
/root/autodl-tmp/tca-reasoning/annotation/stage2a_region_replication_top24_nearest8/region_experiment_manifest_remote.csv

output:
/root/autodl-tmp/tca-reasoning/annotation/stage2b_node_generation_smoke/stage2c_position_map_main_cases.csv
```

同步回本地后得到：

```text
E:\Bridging\annotation\stage2b_node_generation_smoke\stage2c_position_map_main_cases.csv
E:\Bridging\annotation\stage2b_node_generation_smoke\stage2c_position_map_main_cases.log
E:\Bridging\annotation\stage2b_node_generation_smoke\run_stage2c_position_map_main_cases.sh
```

远端输出状态：

```text
remote_rc = 0
rows = 4
```

### 4.3 生成 case package

本地把 route、behavior、first-token、greedy decoded loop、token-position mapping 合并成主 case 表。

同时生成每个样本的图像包：

```text
original image
answer mask
relate mask
union mask
answer overlay
relate overlay
union overlay
```

输出目录：

```text
E:\Bridging\annotation\stage2c_case_package
```

## 5. 输出文件

主输出：

```text
E:\Bridging\annotation\stage2c_case_package\stage2c_main_case_summary.csv
E:\Bridging\annotation\stage2c_case_package\stage2c_support_pair_shortlist.csv
E:\Bridging\annotation\stage2c_case_package\stage2c_case_package_manifest.json
```

图像输出：

```text
E:\Bridging\annotation\stage2c_case_package\images\okvqa_val_1927165\original.jpg
E:\Bridging\annotation\stage2c_case_package\images\okvqa_val_1927165\answer_overlay.jpg
E:\Bridging\annotation\stage2c_case_package\images\okvqa_val_1927165\relate_overlay.jpg
E:\Bridging\annotation\stage2c_case_package\images\okvqa_val_1927165\union_overlay.jpg

E:\Bridging\annotation\stage2c_case_package\images\okvqa_val_2683965\original.jpg
E:\Bridging\annotation\stage2c_case_package\images\okvqa_val_2683965\answer_overlay.jpg
E:\Bridging\annotation\stage2c_case_package\images\okvqa_val_2683965\relate_overlay.jpg
E:\Bridging\annotation\stage2c_case_package\images\okvqa_val_2683965\union_overlay.jpg
```

## 6. 结果一：`okvqa_val_1927165`

### 6.1 基本信息

```text
sample_id: okvqa_val_1927165
prompt: B_direct
question: What does stop mean?
reasoning_operation: visual_readout
image_dependence: strong
source node: L11 / P196 / F151858
nearest control: L23 / P286 / F130599
```

### 6.2 Token-position mapping

```text
source node position:
feature_pos = 196
token_at_pos = <image_soft_token>
is_image_token_pos = True
image_token_index = 192

nearest control position:
feature_pos = 286
token_at_pos = <end_of_turn>
is_image_token_pos = False
```

解释：

这个 case 的 source node 位于 image soft token 上，而 nearest control 位于文本 / turn boundary token 上。这使它很适合做主图正例，因为“source node 的位置”和“证据区域遮挡敏感性”在叙事上比较一致。

注意：这里仍然不能说该 source node 就是 “stop sign detector”。它只能说明这个 source feature 发生在图像 token 位置，并且对 evidence-region mask 有功能敏感性。

### 6.3 Route weakening

```text
source answer-mask weakening = +1.8125
source union-mask weakening = +1.5000
source answer minus random4 = +0.9375
source union minus random4 = +0.6250

nearest answer-mask weakening = +1.0000
nearest union-mask weakening = +1.0000

source minus nearest answer weakening = +0.8125
source minus nearest union weakening = +0.5000
```

解释：

遮挡 answer / union evidence region 会明显削弱 source support route，而且 source 的削弱幅度大于 nearest control，也大于 random4 region control。

这是当前最适合放主图的正例。

### 6.4 Behavior / generation

区域遮挡行为侧：

```text
clean generated answer: obey the law
answer_mask generated answer: a signal to halt
union_mask generated answer: a warning signal

answer_mask changed from clean: True
union_mask changed from clean: True

target rank clean: 20
target rank answer_mask: 30
target rank union_mask: 32
answer rank damage: +10
union rank damage: +12
```

First-token bridge：

```text
answer_mask source-minus-nearest rank gap = 0
union_mask source-minus-nearest rank gap = +2
union_mask delta logit gap = -0.125
```

Decoded-loop node intervention：

```text
source clean:
baseline answer = to halt movement
intervention answer = to halt or cease movement
changed = True

source answer_mask:
baseline answer = a sign
intervention answer = stop
changed = True

nearest control clean:
changed = False

nearest control answer_mask:
changed = False
```

解释：

这个 case 同时满足三条链：

1. answer / union region mask 削弱 source support route；
2. region mask 改变 decoded answer 和 target rank；
3. source node intervention 在 decoded loop 中能改变答案，而 nearest control 没有同样效果。

因此它是 Stage 2C 目前最强的 figure-ready case。

## 7. 结果二：`okvqa_val_2683965`

### 7.1 基本信息

```text
sample_id: okvqa_val_2683965
prompt: B_direct
question: What shape is the sign?
reasoning_operation: visual_readout
image_dependence: strong
source node: L26 / P287 / F39687
nearest control: L11 / P196 / F151858
```

### 7.2 Token-position mapping

```text
source node position:
feature_pos = 287
token_at_pos = <end_of_turn>
is_image_token_pos = False

nearest control position:
feature_pos = 196
token_at_pos = <image_soft_token>
is_image_token_pos = True
image_token_index = 192
```

解释：

这个 case 很重要，因为它提醒我们：traced support route 不一定总在 image token 位置。一个 answer-adjacent route 可以在 assistant prefix / turn boundary 附近发挥作用，同时仍然对图像证据区域遮挡敏感。

因此它不适合被写成“视觉 token 上的语义 feature 正例”，但非常适合展示 route 与 generation 之间的 dissociation。

### 7.3 Route weakening

```text
source answer-mask weakening = +1.3750
source union-mask weakening = +1.3125
source answer minus random4 = +0.7500
source union minus random4 = +0.6875

nearest answer-mask weakening = +2.4375
nearest union-mask weakening = +1.6875

source minus nearest answer weakening = -1.0625
source minus nearest union weakening = -0.3750
```

解释：

这个 case 里 source route 对 answer / union region mask 是敏感的，而且强于 random4 region control；但是 source 不强于 nearest node control。也就是说，它支持“区域证据敏感”，但不支持“source-specific stronger than nearest”。

这正是它应该被写成 boundary case 的原因。

### 7.4 Behavior / generation

区域遮挡行为侧：

```text
clean generated answer: a rectangle
answer_mask generated answer: a square
union_mask generated answer: rectangle

answer_mask changed from clean: True
union_mask changed from clean: False

target rank clean: 2
target rank answer_mask: 243
target rank union_mask: 43
answer rank damage: +241
union rank damage: +41
answer margin drop: +15.6875
union margin drop: +11.8125
```

First-token bridge：

```text
answer_mask source-minus-nearest rank gap = +203
answer_mask delta logit gap = -3.6875
union_mask source-minus-nearest rank gap = +20
union_mask delta logit gap = -3.0000
```

Decoded-loop node intervention：

```text
source clean:
baseline answer = oval
intervention answer = oval
changed = False

source answer_mask:
baseline answer = square
intervention answer = square
changed = False

source union_mask:
baseline answer = rectangle
intervention answer = rectangle
changed = False

nearest control answer_mask:
baseline answer = square
intervention answer = rectangle
changed = True
```

解释：

这个 case 的 first-token damage 很强，但 decoded answer 没有出现 source-specific change。它说明：

```text
first-token distribution damage 不必然转化为最终 decoded answer change。
```

这对论文是有价值的边界结果。它能帮助我们避免把 node-to-generation bridge 说得过强。

## 8. 预期与实际偏差

### 8.1 符合预期

`okvqa_val_1927165` 符合 Stage 2C 预期：它把 evidence-region sensitivity、source > nearest、generation-side change 和 source-specific decoded change 放到了同一个 case 里。

这让它可以作为主图候选：

```text
image + answer/union mask
source and nearest nodes
route weakening bar
target rank / decoded answer change
node intervention decoded loop
```

### 8.2 偏离预期

`okvqa_val_2683965` 原本是 first-token 最强的 case 之一，但 greedy decoded loop 没有显示 source-specific decoded answer change；相反，nearest control 在 answer_mask 下改变了 decoded answer。

这不是坏结果。它说明 first-token bridge 和 decoded generation 之间存在非线性，也说明我们不能直接把 first-token rank damage 写成“足以改变最终答案”。

## 9. 当前结论

Stage 2C 产出了一组可以进入报告的 case package：

1. `okvqa_val_1927165` 是当前最强正例，支持“evidence-region-sensitive support route 与 decoded generation bridge 可以在同一 case 中对齐”。
2. `okvqa_val_2683965` 是边界 / dissociation case，支持“first-token damage 不必然导致 decoded answer change”，也提醒我们不能把 route 位置简单等同于视觉语义位置。
3. Token-position mapping 显示，source nodes 可以位于 image soft token，也可以位于 assistant / turn boundary 附近。因此后续文案应写成 “answer-adjacent route”，而不是统一写成 “image-token semantic node”。

## 10. 对主 claim 的影响

本实验加强：

```text
在 localized strong-image-dependence cases 中，support routes 可以表现出 evidence-region sensitivity，并且在个别高质量 case 中能连接到 generation-side 变化。
```

本实验不支持：

```text
所有 source nodes 都强于 nearest controls。
所有 first-token 伤害都会改变 decoded answer。
source node 都是 object-level visual semantic nodes。
```

因此 Stage 2C 的定位应为：

```text
figure-ready functional case package
```

而不是：

```text
object-level semantic interpretation proof
```

## 11. 后续建议

优先级最高的下一步：

1. 把 `okvqa_val_1927165` 做成主图草稿，包括原图、answer/union overlay、route weakening、decoded loop 对照。
2. 把 `okvqa_val_2683965` 放入 appendix 或正文 boundary paragraph，说明 first-token 与 decoded answer 的 dissociation。
3. 如果继续推进 Stage 2C，可以再对 `okvqa_val_1740705` 和 `okvqa_val_80655` 做 token-position mapping，形成 3-4 个 case 的小型 case panel。

不建议现在做：

```text
把 token-position mapping 扩展成 object-level semantic claim。
把 2683965 写成 decoded generation 正例。
用单个 case 替代 Stage 2A 的统计结论。
```

