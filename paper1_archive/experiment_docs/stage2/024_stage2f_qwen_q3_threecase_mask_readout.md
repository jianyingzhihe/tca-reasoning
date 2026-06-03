# 实验 024：Stage 2F Qwen Q3 三样本 Evidence-Mask Readout 复跑

日期：2026-05-20

## 1. 实验目的

本实验是在 `022_stage2f_qwen_clean_vs_mask_feature_readout.md` 之后的补强复跑。上一轮 Qwen Q3 计划跑 3 个主 case：

```text
okvqa_val_2847255
okvqa_val_4157235
okvqa_val_3605295
```

但 `okvqa_val_3605295` 没有找到现成 answer / relate / union mask 或 LabelMe JSON，因此 Q3 只能判为 `partial_mask_readout`。本轮不临时造 mask，也不引入新人工标注，而是用已有核心标注样本 `okvqa_val_3658865` 替代缺失的 `3605295`，把 Qwen clean-vs-mask readout 从 `2/3` 补到 `3/3`。

本实验仍然是 readout-level feasibility / sensitivity，不是 attribution、不是 feature intervention，也不是跨模型因果机制复现。

## 2. 输入

模型与资产：

```text
Qwen base:
/root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5

Qwen CLT:
KokosDev/qwen2p5vl-7b-clt
```

样本：

```text
okvqa_val_2847255
question = What country might this be based on the writing on the bus?
answer = china
image = COCO_val2014_000000284725.jpg

okvqa_val_4157235
question = Who is playing this sport?
answer = dog
image = COCO_val2014_000000415723.jpg

okvqa_val_3658865
question = What brand of phone is this?
answer = samsung
image = COCO_val2014_000000365886.jpg
```

prompt：

```text
B_direct
D_visual_only
```

conditions：

```text
clean
answer_mask
union_mask
```

layers：

```text
0
13
26
```

## 3. 输出

```text
doc/experiments/stage2/cross_model/stage2f_qwen_position_mapping_3658865.json
doc/experiments/stage2/cross_model/stage2f_qwen_position_mapping_3658865_tokens.csv
doc/experiments/stage2/cross_model/stage2f_qwen_position_mapping_3658865_buckets.csv

doc/experiments/stage2/cross_model/stage2f_qwen_clean_vs_mask_feature_readout_3case.json
doc/experiments/stage2/cross_model/stage2f_qwen_clean_vs_mask_feature_readout_3case.csv
doc/experiments/stage2/cross_model/stage2f_qwen_clean_vs_mask_feature_readout_3case_summary.csv
```

涉及脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_qwen_token_position_mapping_smoke.py
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_qwen_clean_vs_mask_feature_readout.py
scripts/local/run_stage2f4_cross_model_remote.py
```

## 4. 方法

第一步，对 `okvqa_val_3658865` 单独做 token / position mapping，确认：

```text
image span
question span
assistant prefix
last prompt token
position 2 diagnostic
```

第二步，复跑 Qwen Q3：

```text
3 samples × 2 prompts × 3 conditions × 3 layers
```

对每个 bucket 计算：

```text
mean_topk_drop = clean top-k feature activation - masked feature activation
topk_jaccard_change = 1 - Jaccard(clean_topk, masked_topk)
bucket_mean_shift = clean bucket feature mean - masked bucket feature mean
```

重点仍然看：

```text
image_marker_or_span
position_2_diagnostic
question
last_prompt_token
```

## 5. 实际结果

### 5.1 3658865 position mapping

判定：

```text
decision.status = pass_position_mapping
```

关键位置：

```text
sequence_length = 285
image_span = [14, 250]
question_span = [250, 280]
assistant_start = 283
last_prompt_token = 284
position_2 token_id = 198
position_2 token_text = Ċ
```

bucket count：

```text
special/template = 241
image_marker_or_span = 236
question = 30
assistant_prefix = 2
last_prompt_token = 1
position_2_diagnostic = 1
other_text_or_template = 11
```

解释：

```text
position 2 仍然是 system/template newline，不是视觉证据位置，也不是答案附近位置。
```

### 5.2 三样本 Q3 mask readout

判定：

```text
decision.status = pass_mask_readout
usable_samples = 3 / 3
skipped_samples = []
```

mask 可用性：

```text
okvqa_val_2847255 = ok
okvqa_val_4157235 = ok
okvqa_val_3658865 = ok
```

输出规模：

```text
summary rows = 180
detail rows = 3600
```

## 6. 关键读数

以下统计为：

```text
3 samples × 2 prompts
bucket = image_marker_or_span
```

Layer 0：

```text
answer_mask:
mean_topk_drop = +0.527
topk_jaccard_change = 0.586
bucket_mean_shift = +0.0165

union_mask:
mean_topk_drop = +1.444
topk_jaccard_change = 0.594
bucket_mean_shift = +0.0296
```

Layer 13：

```text
answer_mask:
mean_topk_drop = -0.663
topk_jaccard_change = 0.0317
bucket_mean_shift = +0.172

union_mask:
mean_topk_drop = +1.219
topk_jaccard_change = 0.0317
bucket_mean_shift = +0.501
```

Layer 26：

```text
answer_mask:
mean_topk_drop = +15.057
topk_jaccard_change = 0.689
bucket_mean_shift = +0.0733

union_mask:
mean_topk_drop = +18.445
topk_jaccard_change = 0.778
bucket_mean_shift = +0.0012
```

Layer 26 分样本读数：

```text
okvqa_val_2847255:
answer_mask drop = +13.8125
union_mask drop = +14.84375

okvqa_val_4157235:
answer_mask drop = +32.228125
union_mask drop = +33.046875

okvqa_val_3658865:
answer_mask drop = -0.9375 / -0.8000
union_mask drop = +7.4125 / +7.4750
```

其中 `3658865` 的 answer-only mask 对 layer 26 image top features 不稳定，但 union mask 仍为正。这说明 Qwen readout-level evidence sensitivity 在第三样本上更依赖更宽的 evidence/context union 区域。

## 7. 预期与实际偏差

预期：

```text
替换缺失 mask 的 3605295 后，Qwen Q3 应至少达到 3/3 usable。
```

实际：

```text
达到 3/3 usable，并且 Qwen layer 26 image bucket 继续保持最强 mask sensitivity。
```

偏差：

```text
第三样本 3658865 的 answer_mask readout 不稳定，union_mask 明显更强。
```

这个偏差不削弱 feasibility 结论，但提示后续 Qwen evidence-mask readout 不应只看最小 answer mask；对于品牌/文字类样本，relate/context 区域可能承载关键视觉证据。

## 8. 结论

本实验支持：

```text
Qwen Q3 已从 partial_mask_readout 升级为 pass_mask_readout。
Qwen2.5-VL 的 CLT readout 对人工 evidence mask 有稳定可见反应。
反应主要集中在 image token span，尤其 layer 26。
position 2 仍然是模板位置，不参与 strongest readout。
```

本实验不支持：

```text
Qwen 已经复现 Gemma3 causal support route；
Qwen 已经完成 source tracing / nearest control / feature intervention；
Qwen 已经证明 source > control；
Qwen 已经完成跨模型机制复现。
```

最准确判定：

```text
Stage 2F Qwen Q3 three-case rerun = pass_mask_readout with readout-level evidence-region sensitivity。
```

## 9. 对主 claim 的影响

不改变当前 Gemma3 主 claim。它把 Qwen 支线从：

```text
2-case partial readout
```

推进为：

```text
3-case readout-level evidence-mask sensitivity。
```

这让 Qwen 成为更可信的 cross-model feasibility 候选，但仍不能写成 causal replication。

## 10. 后续动作

下一步 Qwen 可以进入：

```text
1. layer 26 image-span sensitive feature 的 minimal intervention adapter；
2. clean vs answer/union mask 的 feature stability case panel；
3. 如果 adapter 成功，再做 1-3 case source/control causal smoke。
```

