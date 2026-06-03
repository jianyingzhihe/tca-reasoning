# Stage6-007 Initial Asset Analysis And Sample Strategy

## 目的

本文记录 Stage6 设计前的资产核对，避免探索性实验凭直觉挑样本。

核心发现：

```text
Stage5 已被 CLT heterogeneity docs 占用，所以 prompt/text/CoT exploratory line 使用 Stage6。
paperpack72 已经足够支持小规模 exploratory prompt/text 实验。
但 symbol_text_reading 样本很少，不能做均衡大样本结论。
```

## Paperpack72 分布

Primary prompt-runs：

```text
total prompt-runs = 144
samples = 72
prompts = B_direct / D_visual_only
```

Primary sample type：

```text
visual_readout = 44 samples
compact_scene_inference = 23 samples
symbol_text_reading = 4 samples
mixed_localized = 1 sample
```

Image dependence tier：

```text
strong = 64
moderate = 5
replacement = 3
```

## Qwen route-first evidence-gold 覆盖

Stage4-060 primary route_first_evidence_gold 覆盖：

```text
route_first_evidence_gold nodes = 1334
unique samples = 50
```

按类型覆盖的样本数：

```text
visual_readout = 32
compact_scene_inference = 15
symbol_text_reading = 2
mixed_localized = 1
```

这说明 Stage6 可以优先复用 Qwen route-first node-level evidence，但不能把 symbol_text_reading 写成大样本统计结论。

## 推荐样本策略

Stage6 full exploratory pack 推荐：

```text
visual_readout: 6-8 samples
compact_scene_inference: 4-6 samples
symbol_text_reading: 2-4 samples, case-level diagnostic
mixed_localized: optional 1 sample
```

总样本：

```text
12-16 samples
```

优先级：

```text
1. Stage4-060 route_first_evidence_gold 节点多。
2. image_dependence_tier = strong。
3. answer/union mask 不过度大、不完全覆盖整图。
4. clean decoded answer 稳定。
5. question rewrite 容易保持语义不变。
```

## 候选样本例子

### Visual readout

```text
okvqa_val_216885:
  evidence_gold_nodes = 72
  question = Is redwood or cedar more prevalent as a siding material?
  answer = cedar

okvqa_val_1499095:
  evidence_gold_nodes = 59
  question = What did the animal just finishing doing?
  answer = catch fish

okvqa_val_4943285:
  evidence_gold_nodes = 54
  question = What does this animal eat?
  answer = tuna

okvqa_val_4075245:
  evidence_gold_nodes = 47
  question = What is the plate made out of?
  answer = ceramic

okvqa_val_4837235:
  evidence_gold_nodes = 40
  question = What sport could they do with these?
  answer = surf
```

注意：

```text
部分 visual_readout 样本 answer/union mask 面积偏大，例如 okvqa_val_216885。
这些样本适合 route robustness，但不适合严格 pixel-localization claim。
```

### Symbol/text reading

```text
okvqa_val_03979:
  evidence_gold_nodes = 36
  question = A beverage is represented here what brand is its opposite in the cola wars?
  answer = pepsi

okvqa_val_4987585:
  evidence_gold_nodes = 29
  question = Which brand of bike is rided by the person in the photo?
  answer = suzuki
```

注意：

```text
symbol_text_reading 在 route-first evidence-gold 中只有 2 个强覆盖样本。
只能写 case-level diagnostic，不能写 universal type claim。
```

### Compact scene inference

```text
okvqa_val_00327:
  evidence_gold_nodes = 47
  question = What is this vehicle used for?
  answer = transportation

okvqa_val_1440035:
  evidence_gold_nodes = 45
  question = The red items on the pastry are a specific ingredient for what type of cake?
  answer = strawberry

okvqa_val_147735:
  evidence_gold_nodes = 44
  question = Why type of restaurant would serve this food?
  answer = diner

okvqa_val_03609:
  evidence_gold_nodes = 42
  question = What is the shower made of?
  answer = glass

okvqa_val_03085:
  evidence_gold_nodes = 34
  question = What is this machine used for?
  answer = work
```

注意：

```text
scene_inference 更容易混入常识/语言先验。
它适合作为 route visibility 更异质的对照组。
```

## Grill-Me 风险审查

### 风险 1: 把 Stage6 写成 CoT 提升论文

处理：

```text
Stage6 只写 prompt modulation，不写 CoT improves VQA。
```

### 风险 2: 等价改写实际改变答案语义

处理：

```text
每个 paraphrase 必须人工审查。
分析中记录 answer granularity change。
```

### 风险 3: CoT 改变 final answer token 位置

处理：

```text
所有 prompt 强制输出 The answer is <short answer>.
如果格式失败，标为 format_confounded，不纳入 route endpoint claim。
```

### 风险 4: type-sliced 过度结论

处理：

```text
symbol_text_reading 和 mixed_localized 只写 diagnostic。
视觉读出 vs 场景推断可以写 exploratory trend，不写 universal law。
```

### 风险 5: Qwen grouped feature route 已未闭合

处理：

```text
Stage6 Qwen 主分析用 hidden-level route 和 route-first feature node-level metrics。
不强行把 grouped feature route 当主 endpoint。
```

## 推荐下一步

先做最小 smoke：

```text
4 samples:
  2 visual_readout
  1 symbol_text_reading
  1 compact_scene_inference

2 question variants:
  original
  paraphrase_1

2 prompt families:
  B_direct
  A_step_visual

3 image conditions:
  clean
  answer_mask
  shuffled_mask
```

Smoke 目的：

```text
验证 question rewrite / prompt family / mask condition 能跑通；
检查 final answer format；
检查 route/node metrics 非空；
不做科学结论。
```

