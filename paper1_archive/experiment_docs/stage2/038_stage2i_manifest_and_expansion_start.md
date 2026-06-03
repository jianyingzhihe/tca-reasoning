# Stage 2I-0 / 2I-1：Cross-Model 扩样本候选构建与远端实验启动记录

## 1. 实验目的

这一阶段回答用户提出的核心问题：

```text
能否通过增加更多样本，或者其他实验方式，进一步证明 Qwen / LLaVA 里也有和 Gemma 主线相近的视觉证据机制？
```

本阶段的判断是：可以推进，但要分层证明。

Gemma 当前有完整主链：source tracing、node intervention、nearest/random controls、wrong-image/region-mask sensitivity、rank/generation linkage。Qwen / LLaVA 当前还没有 source tracing 与 matched source-control route，因此 Stage 2I 不把目标写成“跨模型完整复现”，而是补强：

```text
readout sensitivity -> hidden-state bridge -> decoded answer bridge
```

## 2. 方法概览

本阶段先不让用户重新标注，而是复用已有 answer/relate mask 包，自动构建统一候选表。

候选过滤的关键点：

```text
answer_area_frac <= 0.45:
  primary localized candidate

0.45 < answer_area_frac <= 0.65:
  large-localized secondary candidate

answer_area_frac > 0.65:
  diffuse/fullscreen，默认不进主扩样本
```

这个过滤吸收了前一轮标注中的经验：有些样本虽然人工标注了 answer 区域，但答案区域几乎占满屏幕，不适合当作“关键证据区域定位”的强证据。

## 3. 输入

使用的 mask 包：

```text
annotation/okvqa_evidence_labelme_round4_core24_easy/exported_masks
annotation/okvqa_evidence_labelme_round4_core16_extra/exported_masks
annotation/okvqa_evidence_labelme_round4_ultraeasy16_fresh/exported_masks
annotation/okvqa_evidence_labelme_round5_expanded20_route7/exported_masks
annotation/stage2a_region_replication_top24_nearest8/exported_masks
```

同时读取各 annotation pack 的 `manifest.csv`，补齐：

```text
sample_id
question_text
answer_text
reasoning_operation
visual_structure
image_dependence
```

## 4. 输出

候选构建脚本：

```text
scripts/local/build_stage2i_cross_model_candidate_manifest.py
```

输出文件：

```text
doc/experiments/stage2/cross_model/stage2i_cross_model_candidate_manifest.csv
doc/experiments/stage2/cross_model/stage2i_selected_12_manifest.csv
doc/experiments/stage2/cross_model/stage2i_manifest_summary.json
```

后续远端实验脚本：

```text
scripts/local/run_stage2i_cross_model_expansion_remote.py
```

Stage 2I 分析脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/analyze_stage2i_cross_model_expansion.py
```

## 5. 候选构建结果

总体统计：

```text
total_mask_rows = 68
unique_samples = 59
primary_eligible = 56
secondary_large_localized = 4
excluded_diffuse_or_fullscreen = 8
selected_count = 12
primary_answer_area_frac_mean = 0.209879
```

这说明当前已有标注资产足够启动扩样本实验，不需要马上新增人工标注。

## 6. 当前选出的 12 个 Stage 2I 候选

```text
1.  okvqa_val_1593205  answer=tokyo          type=symbol_text_reading  answer_frac=0.040781
2.  okvqa_val_4739195  answer=spanish        type=symbol_text_reading  answer_frac=0.098340
3.  okvqa_val_4502065  answer=tortoise       type=symbol_text_reading  answer_frac=0.138795
4.  okvqa_val_2954205  answer=move           type=symbol_text_reading  answer_frac=0.202109
5.  okvqa_val_1729795  answer=arrow          type=symbol_text_reading  answer_frac=0.203437
6.  okvqa_val_3959785  answer=kuwait airway  type=symbol_text_reading  answer_frac=0.262117
7.  okvqa_val_1058855  answer=fire hydrant   type=visual_readout       answer_frac=0.067962
8.  okvqa_val_340155   answer=racquet        type=visual_readout       answer_frac=0.081382
9.  okvqa_val_01514    answer=50 pounds      type=visual_readout       answer_frac=0.122931
10. okvqa_val_4043385  answer=german         type=symbol_text_reading  answer_frac=0.350000
11. okvqa_val_4938465  answer=schwinn        type=symbol_text_reading  answer_frac=0.362412
12. okvqa_val_2708155  answer=left           type=visual_readout       answer_frac=0.174114
```

## 7. 下一步远端实验设计

### 7.1 Qwen selected-8 decoded bridge expansion

目的：

```text
检查 Stage 2H 中 Qwen 的 partial decoded-answer bridge 是否能扩展到更多样本。
```

固定配置：

```text
model = Qwen2.5-VL-7B-Instruct
layer = 26
samples = selected top 8
prompts = B_direct, D_visual_only
conditions = clean, union_mask, top_hidden_delta_plus_answer_adjacent, answer_adjacent_text, low_delta_control, random_control_1
generation = greedy, max_new_tokens=3
```

成功读法：

```text
best bridge 在 informative clean-vs-union rows 上比 controls 更常离开 union answer。
best bridge 更常回到 clean/target answer。
first-token target logit/rank restore 方向多数一致。
```

### 7.2 LLaVA selected-12 generation-gap screen

目的：

```text
解决 Stage 2H 中 LLaVA decoded answer underpowered 的问题。
先筛 clean 与 union_mask 生成答案是否真的不同。
```

固定配置：

```text
model = LLaVA-1.5-7B
layer = 15
samples = selected top 12
prompts = B_direct, D_visual_only
conditions = clean, union_mask
generation = greedy, max_new_tokens=3
```

成功读法：

```text
如果 informative clean-vs-union rows >= 4，则下一步跑 LLaVA bridge conditions。
如果 informative rows 太少，只能说明这些候选里 LLaVA decoded generation 对 union mask 不够敏感，不能说明 hidden-state bridge 不存在。
```

## 8. 当前结论

```text
可以通过增加样本继续加强跨模型证据。
但 Stage 2I 的主要价值是扩大 Qwen/LLaVA 的 hidden-state / decoded bridge evidence，
而不是直接把它们升级成 Gemma 式 source-control causal route replication。
```

如果 Stage 2I 成功，最强可写结论是：

```text
Qwen/LLaVA show multi-case cross-model support for evidence-sensitive hidden-state bridges,
with Qwen showing stronger decoded-answer bridge evidence.
```

如果 Stage 2I 失败或结果弱，Gemma 主线不受影响；跨模型部分继续作为 auxiliary evidence。

