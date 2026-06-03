# 实验 022：Stage 2F Qwen Clean vs Evidence-Mask Feature Readout

日期：2026-05-20

## 1. 实验目的

本实验对应 Stage 2F-3 的 Qwen Q3。Q2 已经确认 Qwen 的 image span、question span、assistant prefix 和 last prompt token 可以稳定定位，因此本实验进一步检查：

```text
在 Qwen2.5-VL 上，人工 answer / union evidence mask 是否会改变 CLT feature activations？
```

注意：本实验仍然不是因果实验。它只比较 clean image 与 masked image 下的 feature readout，不做 attribution graph，不做 source node tracing，不做 feature intervention。

## 2. 输入

样本固定为计划中的 3 个主 case：

```text
okvqa_val_2847255
okvqa_val_4157235
okvqa_val_3605295
```

prompt 固定为：

```text
B_direct
D_visual_only
```

条件固定为：

```text
clean
answer_mask
union_mask
```

层固定为：

```text
0, 13, 26
```

mask 来源：

```text
/root/autodl-tmp/tca-reasoning/stage2f_cross_model/qwen_q3_assets/exported_masks
/root/autodl-tmp/tca-reasoning/annotation/okvqa_evidence_labelme_round4_core24_easy
/root/autodl-tmp/tca-reasoning/annotation/okvqa_evidence_labelme_round4_mainline16
```

## 3. 输出

```text
doc/experiments/stage2/022_stage2f_qwen_clean_vs_mask_feature_readout.md
doc/experiments/stage2/cross_model/stage2f_qwen_clean_vs_mask_feature_readout.json
doc/experiments/stage2/cross_model/stage2f_qwen_clean_vs_mask_feature_readout.csv
doc/experiments/stage2/cross_model/stage2f_qwen_clean_vs_mask_feature_readout_summary.csv
```

新增脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_qwen_clean_vs_mask_feature_readout.py
```

## 4. 方法

对每个可用样本、prompt、condition、layer 和 bucket：

1. 运行 Qwen native forward，读取 hidden states。
2. 用 Qwen CLT encoder 得到 feature activations。
3. 用 clean 条件下每个 bucket 的 top-k features 作为参照。
4. 对 answer/union mask 计算：

```text
drop = clean_activation - masked_activation
topk_jaccard_change = 1 - Jaccard(clean_topk, masked_topk)
bucket_mean_shift = mean(clean_bucket_features) - mean(masked_bucket_features)
```

bucket 固定为：

```text
image_marker_or_span
question
assistant_prefix
last_prompt_token
position_2_diagnostic
```

## 5. 结果

最终判定：

```text
decision.status = partial_mask_readout
usable_samples = 2 / 3
skipped_samples = [okvqa_val_3605295]
```

mask 可用性：

```text
okvqa_val_2847255 = ok
okvqa_val_4157235 = ok
okvqa_val_3605295 = missing
```

解释：

```text
3605295 本地/远端都只有图像，没有找到 answer/relate/exported mask 或 LabelMe JSON。
因此本轮 Qwen Q3 是 partial，不强行补假 mask，也不临时换样本。
```

输出规模：

```text
summary rows = 120
detail rows = 2400
```

### 5.1 关键 aggregate 读数

以下是 2 个可用样本 × 2 个 prompt 的均值。

Layer 0：

```text
answer_mask image bucket:
mean_topk_drop = +0.6805
bucket_mean_shift = +0.0127
topk_jaccard_change = 0.5926

union_mask image bucket:
mean_topk_drop = +0.5355
bucket_mean_shift = +0.0191
topk_jaccard_change = 0.5165

question / last_prompt / position_2:
基本为 0
```

Layer 13：

```text
answer_mask image bucket:
mean_topk_drop = -0.0563
bucket_mean_shift = +0.1597
topk_jaccard_change = 0.0476

union_mask image bucket:
mean_topk_drop = +1.3750
bucket_mean_shift = +0.3115
topk_jaccard_change = 0.0476
```

Layer 26：

```text
answer_mask image bucket:
mean_topk_drop = +23.0203
bucket_mean_shift = +0.0968
topk_jaccard_change = 0.7879

union_mask image bucket:
mean_topk_drop = +23.9453
bucket_mean_shift = +0.0360
topk_jaccard_change = 0.8225
```

对照诊断：

```text
position_2_diagnostic 在所有层和条件下 drop/shift 基本为 0。
这进一步支持 position 2 是模板位置，不是 evidence-sensitive visual readout 位置。
```

## 6. 预期与实际偏差

预期：

```text
3 个样本都能运行 clean / answer_mask / union_mask。
```

实际：

```text
2 个样本可运行，3605295 缺 mask，因此 Q3 判为 partial。
```

机制方向：

```text
Qwen readout-level evidence sensitivity 主要出现在 image span，尤其 layer 26。
question / last_prompt token 的变化较弱且不稳定。
position 2 基本不受 mask 影响。
```

## 7. 结论

本实验支持：

```text
Qwen2.5-VL 的 CLT feature readout 对人工 evidence mask 有可观察反应。
这种反应主要集中在 image token span，而不是 system/template position 2。
Layer 26 的 image span 对 answer/union mask 的 top feature drop 和 top-k 改组最明显。
```

本实验不支持：

```text
Qwen 已经出现 causal support route。
Qwen 已经 source > control。
Qwen 已经复现 Gemma3 的 evidence-region-sensitive support route。
```

最准确判定：

```text
Stage 2F Qwen Q3 = partial_mask_readout with readout-level positive signal。
```

## 8. 后续动作

Qwen 下一步可以进入两个方向：

```text
1. 补齐 3605295 或改用已有 mask 的第三个主 case，再复跑 Q3 到 3/3 usable。
2. 基于 layer 26 image-span sensitive features，做 Qwen minimal intervention adapter 设计。
```

在补齐之前，Qwen Q3 只能写成：

```text
readout-level evidence sensitivity in 2 localized cases
```

不能写成：

```text
cross-model causal replication
```
