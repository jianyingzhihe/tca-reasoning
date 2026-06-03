# Stage 2K：Matched-control 解释性复盘

## 1. 实验目的

Stage 2J 得到一个更精细的跨模型判断：

```text
Qwen:
  matched-control specificity 大多成立。

LLaVA:
  activation-matched 下仍有优势；
  delta-matched 下优势变弱，说明 clean-vs-union hidden delta magnitude 可能解释了相当一部分效果。
```

Stage 2K 不再新跑模型，而是对 Stage 2J 原始结果做解释性拆解，验证这个判断是否稳。

要回答的问题：

```text
1. Qwen 是否真的比 LLaVA 更像 source-specific hidden bridge？
2. LLaVA 的弱点是否主要来自 delta-matched controls？
3. 这种现象是否在 symbol_text_reading / visual_readout 两类题型中一致？
4. best bridge 的优势到底来自 visual source-like positions，还是 answer-adjacent text positions？
```

## 2. 证明层级

Stage 2K 能支持：

```text
对 Stage 2J matched-control 结果的解释；
判断 Qwen/LLaVA 跨模型辅助证据的强弱；
识别哪些题型或位置组更稳定。
```

Stage 2K 不能支持：

```text
Qwen/LLaVA 已经复现 Gemma source-control causal route；
Qwen/LLaVA CLT feature 是 causal source node；
hidden positions 是对象级语义节点；
LLaVA 强 specificity 已经成立。
```

## 3. 输入

```text
doc/experiments/stage2/cross_model/stage2j_qwen_matched_control_patch.csv
doc/experiments/stage2/cross_model/stage2j_llava_matched_control_patch.csv
doc/experiments/stage2/cross_model/stage2i_selected_12_manifest.csv
```

分析脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/analyze_stage2k_matched_control_explanation.py
```

## 4. 输出

```text
doc/experiments/stage2/cross_model/stage2k_matched_control_explanation_case.csv
doc/experiments/stage2/cross_model/stage2k_matched_control_explanation_model_summary.csv
doc/experiments/stage2/cross_model/stage2k_matched_control_explanation_typed_summary.csv
doc/experiments/stage2/cross_model/stage2k_matched_control_explanation_decision.json
```

## 5. 指标定义

核心指标 1：`combo_minus_delta_combo`

```text
top_hidden_delta_plus_answer_adjacent
-
delta_matched_plus_answer_adjacent
```

含义：

```text
固定 answer-adjacent text positions，只比较 source-like visual positions 是否强于 delta-matched visual controls。
如果该指标稳定为正，说明效果不只是 hidden delta magnitude 大。
```

核心指标 2：`combo_minus_activation_combo`

```text
top_hidden_delta_plus_answer_adjacent
-
activation_matched_plus_answer_adjacent
```

含义：

```text
固定 answer-adjacent text positions，只比较 source-like visual positions 是否强于 activation-matched visual controls。
如果该指标稳定为正，说明效果不只是 clean activation norm 大。
```

核心指标 3：visual-only 增量

```text
top_hidden_delta_plus_answer_adjacent - answer_adjacent_text
delta_matched_plus_answer_adjacent - answer_adjacent_text
activation_matched_plus_answer_adjacent - answer_adjacent_text
```

含义：

```text
检查 visual positions 在 answer-adjacent 基础上到底贡献了多少额外恢复/损伤。
```

## 6. Model-level 结果

### 6.1 Qwen

整体判定：

```text
status = specificity_explanation_supported
```

Restore：

```text
combo_minus_delta_combo:
  mean = +0.406250
  95% CI = [+0.101562, +0.778646]
  status = stable_positive

combo_minus_activation_combo:
  status = stable_positive
```

Corrupt：

```text
combo_minus_delta_combo:
  mean = +0.149089
  95% CI = [+0.044271, +0.273438]
  status = stable_positive

combo_minus_activation_combo:
  status = weak_or_heterogeneous_positive
```

读法：

```text
Qwen 在 restore 方向同时强于 delta-matched 和 activation-matched controls；
corrupt 方向仍强于 delta-matched，但 activation-matched 较弱。
总体支持 Qwen 是更强的 cross-model auxiliary line。
```

### 6.2 LLaVA

整体判定：

```text
status = delta_explanation_partially_supported
```

Restore：

```text
combo_minus_delta_combo:
  mean = +0.146484
  95% CI = [-0.269043, +0.549805]
  status = weak_or_heterogeneous_positive

combo_minus_activation_combo:
  status = stable_positive
```

Corrupt：

```text
combo_minus_delta_combo:
  mean = +0.193522
  95% CI = [-0.045736, +0.490560]
  status = weak_or_heterogeneous_positive

combo_minus_activation_combo:
  status = stable_positive
```

读法：

```text
LLaVA 明显强于 activation-matched controls；
但不稳定强于 delta-matched controls。
这支持“LLaVA 的 hidden bridge 有一部分可以被 clean-vs-union delta magnitude 解释”的判断。
```

## 7. Typed 结果

### 7.1 Qwen：强结果主要来自 symbol_text_reading

Symbol text reading：

```text
restore combo_minus_delta_combo:
  mean = +0.578125
  status = stable_positive

restore combo_minus_activation_combo:
  mean = +0.488281
  status = stable_positive

corrupt combo_minus_delta_combo:
  mean = +0.219727
  status = stable_positive

corrupt combo_minus_activation_combo:
  mean = +0.173828
  status = weak_or_heterogeneous_positive
```

Visual readout：

```text
restore combo_minus_delta_combo:
  mean = +0.062500
  status = stable_positive

restore combo_minus_activation_combo:
  mean = +0.039062
  status = weak_or_heterogeneous_positive

corrupt combo_minus_delta_combo:
  mean = +0.007812
  status = weak_or_heterogeneous_positive

corrupt combo_minus_activation_combo:
  mean = -0.007812
  status = not_positive
```

读法：

```text
Qwen 的 matched-control specificity 主要由 symbol_text_reading 样本支撑。
visual_readout 切片明显更弱，不能把 Qwen 的强结论泛化到所有 localized visual cases。
```

### 7.2 LLaVA：delta-matched 弱点跨类型存在

Symbol text reading：

```text
restore combo_minus_delta_combo:
  mean = +0.144287
  status = weak_or_heterogeneous_positive

restore combo_minus_activation_combo:
  mean = +1.375488
  status = stable_positive
```

Visual readout：

```text
restore combo_minus_delta_combo:
  mean = +0.150879
  status = stable_positive

restore combo_minus_activation_combo:
  mean = +0.537109
  status = stable_positive

corrupt combo_minus_delta_combo:
  mean = +0.090820
  status = weak_or_heterogeneous_positive

corrupt combo_minus_activation_combo:
  mean = +0.174805
  status = weak_or_heterogeneous_positive
```

读法：

```text
LLaVA 的 activation-matched 优势较明显；
delta-matched 弱点不是单一题型造成的，但 visual_readout restore 稍强。
总体仍应写成 partial / heterogeneous specificity。
```

## 8. 对“是否是这样”的回答

### 判断 1：Qwen 是更强的 cross-model auxiliary replication

结论：

```text
支持。
```

依据：

```text
Qwen 在 model-level restore/corrupt 下大多强于 delta/activation matched controls。
尤其 symbol_text_reading 切片稳定。
```

边界：

```text
Qwen visual_readout 切片弱；
仍不是 source-control route replication。
```

### 判断 2：LLaVA 的 bridge 有一部分被 delta magnitude 解释

结论：

```text
支持。
```

依据：

```text
LLaVA 强于 activation-matched controls；
但 combo_minus_delta_combo 在 restore/corrupt 下都只是 weak_or_heterogeneous_positive。
```

边界：

```text
这不等于 LLaVA 没有 bridge。
LLaVA 仍有 bidirectional hidden bridge 和 partial decoded bridge；
只是 matched-control specificity 弱于 Qwen。
```

### 判断 3：best bridge 不应写成 pure visual route

结论：

```text
支持。
```

依据：

```text
Qwen 的 answer-adjacent 成分很强；
LLaVA 的 best group 也是 top_hidden_delta_plus_answer_adjacent。
更合理的机制表述是 evidence-to-answer hidden-state bridge / convergence。
```

## 9. 对主 claim 的影响

Gemma 主线不变：

```text
Gemma 是完整机制链模型：
source tracing + node intervention + nearest/random controls + wrong-image/region-mask sensitivity + behavior linkage。
```

跨模型辅助结论进一步收窄：

```text
Qwen:
  stronger auxiliary replication；
  matched-control explanation mostly supports source-like hidden-position specificity；
  strongest in symbol_text_reading。

LLaVA:
  bidirectional hidden bridge + partial decoded bridge；
  delta-matched controls explain part of the effect；
  specificity should be written as partial / heterogeneous。
```

推荐论文表述：

```text
Cross-model analyses suggest that evidence-sensitive answer bridges are not unique to Gemma. Qwen provides the strongest auxiliary support, including matched-control evidence, especially in text/symbol-reading cases. LLaVA also shows hidden-state and decoded bridge signals, but its specificity is partly absorbed by delta-matched controls, so we treat it as partial rather than full cross-model causal replication.
```

## 10. 下一步是否还需要实验

可以继续，但方向要更精准。推荐优先级：

```text
1. Qwen symbol_text_reading case panel:
   选 2-3 个 Qwen matched-control 最稳 case，做主图/表格。

2. LLaVA delta-matched diagnostic case:
   选一个 LLaVA activation-matched 成立但 delta-matched 弱的 case，作为边界反例。

3. 如果要继续提升 cross-model 强度：
   需要 Qwen/LLaVA source tracing 或 feature-level attribution；
   单纯继续扩大 hidden patch 样本的边际收益开始下降。
```

