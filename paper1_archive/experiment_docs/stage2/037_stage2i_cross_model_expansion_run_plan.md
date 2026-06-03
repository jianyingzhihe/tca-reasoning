# Stage 2I：Cross-Model 扩样本与补强实验 Run Plan

## 1. 这一阶段要回答什么

用户问题是：能不能通过增加更多样本，或者用其他实验方式，进一步证明 Qwen / LLaVA 里也存在和 Gemma 主线相近的视觉证据机制？

答案是：可以继续推进，但必须把证明层级分清楚。

当前 Gemma 主线已经有完整机制链：

```text
source tracing
node intervention
nearest / random controls
wrong-image sensitivity
region-mask sensitivity
rank / generation linkage
```

Qwen / LLaVA 当前还没有达到这个层级。它们已经支持：

```text
readout sensitivity
hidden-state bucket patch upper bound
hidden-position causal localization
Qwen partial decoded-answer bridge smoke
LLaVA first-token / rank bridge
```

Stage 2I 的目标不是把 Qwen / LLaVA 直接升级成“已复现 Gemma causal route”，而是做两件更稳的事：

1. **扩样本检验**：把 Stage 2H 的 3 个 case 扩到 8-12 个可定位证据样本，检查 Qwen / LLaVA 的 hidden-position bridge 是否仍然方向一致。
2. **筛出可生成评估样本**：尤其针对 LLaVA，先找 clean 与 union_mask decoded answer 确实不同的样本，再评估 bridge 是否能把答案推回 clean / target。

## 2. 核心结论边界

如果 Stage 2I 成功，可以写：

```text
Qwen / LLaVA 上的 evidence-mask sensitivity 与 hidden-state causal bridge 在更多 localized cases 上复现。
Qwen 的 decoded-answer bridge 从 3-case smoke 扩展到更大的 targeted set。
LLaVA 若能找到 clean-vs-union generation gap，则可以评估 generation bridge；否则继续保留为 first-token/rank bridge。
```

仍然不能写：

```text
Qwen / LLaVA 已复现 Gemma 的 source-control causal route。
Qwen / LLaVA 的 CLT feature 已经是 causal source node。
hidden positions / CLT features 是对象级语义节点。
D_visual_only 比 B_direct 更好。
跨模型机制已经被完整证明。
```

## 3. 为什么增加样本有价值

Stage 2H 的强点是因果定位清楚，但弱点是样本数只有 3 个。扩样本能解决三个问题：

1. **偶然性问题**：3 个 case 可能恰好适合 hidden patch；8-12 个 case 更能说明方向不是偶然。
2. **LLaVA underpowered 问题**：Stage 2H 中 LLaVA clean 和 union_mask 生成答案完全一样，导致 decoded answer bridge 无法评估。扩样本后先筛 generation gap，可以找到真正可评估的行。
3. **类型覆盖问题**：可以比较 symbol_text_reading、visual_readout、scene_inference 三类中哪类更容易出现跨模型桥接信号。

## 4. 样本来源

Stage 2I 默认不要求新标注，先复用已有 mask 包：

```text
annotation/okvqa_evidence_labelme_round4_core24_easy/exported_masks
annotation/okvqa_evidence_labelme_round4_core16_extra/exported_masks
annotation/okvqa_evidence_labelme_round4_ultraeasy16_fresh/exported_masks
annotation/okvqa_evidence_labelme_round5_expanded20_route7/exported_masks
annotation/stage2a_region_replication_top24_nearest8/exported_masks
```

旧 round2 / round3 也可以做方法回归，但不进入 Stage 2I 主读法，因为标注规范和当前 answer / relate 体系不完全一致。

## 5. 候选样本筛选规则

优先级固定如下：

1. 有 `answer.png` 与 `relate.png`。
2. 有明确 `question_text` 与 `answer_text`。
3. `image_dependence = strong` 或 legacy visual label 为 localized。
4. `answer` 区域不是弥漫性整图区域。
5. 优先类型：`symbol_text_reading > visual_readout > scene_inference`。
6. 优先已有 Gemma 路径证据或 nearest-control 证据的样本，但这不是 Qwen/LLaVA 的必要条件。

局部性过滤建议：

```text
answer_area_frac <= 0.45: primary eligible
0.45 < answer_area_frac <= 0.65: large-localized / secondary
answer_area_frac > 0.65: diffuse_or_fullscreen，默认不进主扩样本
```

## 6. 实验 2I-0：候选 manifest 构建

目的：

```text
把所有可复用 mask 包统一成一张 cross-model candidate manifest，避免手工挑样本导致选择偏差。
```

输入：

```text
已有 annotation manifest.csv
已有 exported_masks/COCO_val2014_xxx/answer.png
已有 images/COCO_val2014_xxx.jpg
已有 stage2a / core24 / route7 结果表
```

输出：

```text
doc/experiments/stage2/cross_model/stage2i_cross_model_candidate_manifest.csv
doc/experiments/stage2/cross_model/stage2i_selected_12_manifest.csv
doc/experiments/stage2/cross_model/stage2i_manifest_summary.json
```

判据：

```text
至少找到 12 个 primary eligible localized samples。
如果不足 12 个，允许加入 large-localized secondary，但必须在文档里标注。
如果不足 8 个，需要用户补标注。
```

## 7. 实验 2I-1：Qwen decoded bridge 扩样本

目的：

```text
检查 Stage 2H 中 Qwen 的 partial decoded-answer bridge 是否能在更多样本上复现。
```

模型与层：

```text
Qwen2.5-VL-7B-Instruct
layer = 26
group = top_hidden_delta_plus_answer_adjacent
controls = low_delta_control, random_control_1
```

样本：

```text
stage2i_selected_12_manifest.csv 中优先 8-12 个样本
prompts = B_direct, D_visual_only
conditions = clean, union_mask, best_bridge, low_delta_control, random_control_1
```

指标：

```text
clean_target_hit
union_target_hit
best_bridge_target_hit
informative_clean_vs_union_rows
best_bridge_changed_away_from_union
best_bridge_same_as_clean
best_bridge_logit_restore_vs_union
best_bridge_rank_restore_vs_union
control_changed_away_from_union
```

成功标准：

```text
best_bridge 在 informative rows 中比 low_delta/random control 更常离开 union answer。
best_bridge target hit 或 same_as_clean 明显高于 controls。
first-token/rank restore 方向在多数行一致。
```

结论边界：

```text
成功：Qwen hidden-position bridge has expanded decoded-answer support。
失败：Qwen hidden-position bridge 仍主要停留在 first-token/rank 层面。
无论成功失败，都不写 source-control route replication。
```

## 8. 实验 2I-2：LLaVA generation-gap screen

目的：

```text
解决 Stage 2H 中 LLaVA decoded answer underpowered 的问题。
先筛出 clean 与 union_mask 生成答案不同的样本，再进入 bridge 评估。
```

模型与层：

```text
LLaVA-1.5-7B
layer = 15
```

筛选方式：

```text
对 8-12 个 selected samples 先只跑 clean 与 union_mask greedy decoding。
如果 clean == union，则该行只保留 first-token/rank 读法，不纳入 decoded bridge 主评估。
如果 clean != union，则进入 bridge conditions。
```

成功标准：

```text
至少找到 4 个 informative clean-vs-union rows。
在 informative rows 上，best_bridge 比 controls 更常离开 union answer 或回到 clean/target。
```

结论边界：

```text
若没有足够 informative rows，则说明 LLaVA 在这些样本上 decoded answer 对 union mask 不敏感，不能说明 hidden bridge 不存在。
若 bridge 成功，只写 LLaVA decoded bridge smoke，不写 feature/source route replication。
```

## 9. 实验 2I-3：hidden-position patch 扩样本

目的：

```text
如果 decoded generation 太慢或太不稳定，先用 first-token / target-rank 层面的 hidden-position patch 扩样本。
这比完整 decoded answer 更接近 Stage 2H 的强结果，也更稳定。
```

设计：

```text
models = Qwen layer 26, LLaVA layer 15
samples = selected 8-12
prompts = B_direct, D_visual_only
directions = restore + corrupt
groups = top_hidden_delta_plus_answer_adjacent, low_delta_control, random_control_1..4
```

成功标准：

```text
restore: source-like group 的 logit/rank restore 大于 random controls。
corrupt: 同一 source-like group 对 clean run 的损伤也大于 controls。
positive direction >= 70% rows。
```

结论边界：

```text
这可以加强 hidden-state causal bridge。
仍不能替代 Gemma 的 CLT source tracing / nearest-control source node。
```

## 10. 实验 2I-4：需要新标注的触发条件

只有以下情况才需要用户继续标注：

```text
primary eligible localized samples < 8
LLaVA informative clean-vs-union rows < 4 且我们需要 generation-level LLaVA bridge
现有候选过多 large/fullscreen，缺少 compact answer evidence
```

如果需要新标注，优先给用户：

```text
symbol_text_reading / visual_readout
answer region compact
问题与答案明确
clean baseline 能答对或接近答对
```

## 11. 这一阶段的推荐执行顺序

```text
2I-0 构建 manifest
2I-1 Qwen 8-12 case decoded bridge expansion
2I-2 LLaVA generation-gap screen
2I-3 若时间允许，补 Qwen/LLaVA hidden-position patch expansion
2I-4 复盘：判断是否需要用户补标注
```

## 12. 最终判定模板

成功时写：

```text
Stage 2I expands cross-model support from a 3-case smoke to a targeted multi-case set. Qwen shows stronger decoded-answer bridge evidence, while LLaVA either shows generation-level bridge on informative rows or remains supported at the first-token/rank hidden-position level. These results strengthen cross-model external validity of evidence-sensitive hidden-state bridges, but do not establish full source-control causal route replication outside Gemma.
```

失败或弱结果时写：

```text
Stage 2I does not invalidate the Gemma main claim. It shows that current Qwen/LLaVA evidence is limited to readout and hidden-state bridge levels, and that decoded generation bridge is sample/model dependent. The cross-model story should remain auxiliary unless a later stage adds source tracing or stronger model-specific controls.
```

