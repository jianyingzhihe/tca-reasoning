# Stage 2J：Cross-Model matched-control hidden-position specificity

## 1. 实验目的

Stage 2I-3 已经证明：

```text
Qwen / LLaVA selected-12 上存在 bidirectional hidden-position bridge。
top_hidden_delta_plus_answer_adjacent 在 restore 和 corrupt 两个方向都强于 random / low-delta controls。
```

Stage 2J 继续追问一个更严格的问题：

```text
best bridge 是否只是因为这些位置的 clean-vs-union hidden delta 更大，或者 clean activation norm 更大？
```

因此这一轮加入更强对照：

```text
delta_matched_control:
  从非 source visual positions 中，选择 hidden delta norm 与 top_hidden_delta source positions 最接近的位置。

activation_matched_control:
  从非 source visual positions 中，选择 clean hidden activation norm 与 top_hidden_delta source positions 最接近的位置。

delta_matched_plus_answer_adjacent:
  delta_matched visual positions + 同一组 answer-adjacent text positions。

activation_matched_plus_answer_adjacent:
  activation_matched visual positions + 同一组 answer-adjacent text positions。
```

其中 `*_plus_answer_adjacent` 是最关键的强对照，因为它固定了 answer-adjacent text positions，只替换 visual/source-like 部分。

## 2. 重要实现说明

第一次远端 smoke 中发现 matched-control 分数表的键值写反，导致 matched visual positions 为空，`*_plus_answer_adjacent` 退化成纯 `answer_adjacent_text`。

这个结果没有进入结论。随后修复：

```text
delta_scores = {position: score for score, position in _position_delta_norms(...)}
activation_scores = {position: score for score, position in _position_activation_norms(...)}
```

修复后重新跑 Stage 2J，确认：

```text
Qwen:
  top_delta_position_count = 32
  delta_matched_position_count = 32
  activation_matched_position_count = 32
  *_plus_answer_adjacent position_count = 36

LLaVA:
  top_delta_position_count = 64
  delta_matched_position_count = 64
  activation_matched_position_count = 64
  *_plus_answer_adjacent position_count = 68
```

以下所有结果都来自修复后的重跑。

## 3. 输入

样本：

```text
doc/experiments/stage2/cross_model/stage2i_selected_12_manifest.csv
```

模型：

```text
Qwen2.5-VL-7B-Instruct, layer 26
LLaVA-1.5-7B, layer 15
```

Prompt：

```text
B_direct
D_visual_only
```

方向：

```text
restore: union_mask -> clean hidden patch
corrupt: clean -> union hidden patch
```

## 4. 输出

原始输出：

```text
doc/experiments/stage2/cross_model/stage2j_qwen_matched_control_patch.csv/json
doc/experiments/stage2/cross_model/stage2j_llava_matched_control_patch.csv/json
```

分析输出：

```text
doc/experiments/stage2/cross_model/stage2j_matched_control_summary.csv
doc/experiments/stage2/cross_model/stage2j_matched_control_case_table.csv
doc/experiments/stage2/cross_model/stage2j_matched_control_bootstrap.csv/json
doc/experiments/stage2/cross_model/stage2j_matched_control_decision_readout.json
```

## 5. 主结果概览

### 5.1 Qwen：matched-control mostly supported

Qwen best combo vs matched controls：

```text
restore combo_vs_delta_matched_plus:
  mean logit delta = +0.406250
  95% CI = [+0.096354, +0.789062]
  status = stable_positive

restore combo_vs_activation_matched_plus:
  mean logit delta = +0.338542
  95% CI = [+0.044271, +0.710938]
  status = stable_positive

corrupt combo_vs_delta_matched_plus:
  mean logit delta = +0.149089
  95% CI = [+0.042969, +0.276693]
  status = stable_positive

corrupt combo_vs_activation_matched_plus:
  mean logit delta = +0.113281
  95% CI = [-0.006510, +0.259766]
  status = weak_or_heterogeneous_positive
```

Qwen visual-only source vs matched controls：

```text
restore visual_vs_delta_matched:
  mean logit delta = +0.278646
  95% CI = [+0.069010, +0.539062]
  status = stable_positive

restore visual_vs_activation_matched:
  mean logit delta = +0.277344
  95% CI = [-0.002604, +0.644531]
  status = weak_or_heterogeneous_positive

corrupt visual_vs_delta_matched:
  mean logit delta = +0.205729
  95% CI = [+0.028646, +0.424479]
  status = stable_positive

corrupt visual_vs_activation_matched:
  mean logit delta = +0.169271
  95% CI = [+0.010417, +0.356771]
  status = stable_positive
```

判定：

```text
Qwen matched-control specificity mostly supported。
source-like top_delta positions 不只是 activation/delta 大；在多数 matched-control 比较下仍然更有效。
但 corrupt combo vs activation_matched_plus 较弱，因此不写成绝对强结论。
```

### 5.2 LLaVA：activation-matched 支持，delta-matched 降级

LLaVA best combo vs matched controls：

```text
restore combo_vs_delta_matched_plus:
  mean logit delta = +0.146484
  95% CI = [-0.269043, +0.549805]
  status = weak_or_heterogeneous_positive

restore combo_vs_activation_matched_plus:
  mean logit delta = +1.096029
  95% CI = [+0.503906, +1.720052]
  status = stable_positive

corrupt combo_vs_delta_matched_plus:
  mean logit delta = +0.193522
  95% CI = [-0.046712, +0.493978]
  status = weak_or_heterogeneous_positive

corrupt combo_vs_activation_matched_plus:
  mean logit delta = +0.225911
  95% CI = [+0.007975, +0.503581]
  status = stable_positive
```

LLaVA visual-only source vs matched controls：

```text
restore visual_vs_delta_matched:
  mean logit delta = +0.114746
  95% CI = [-0.267904, +0.486328]
  status = weak_or_heterogeneous_positive

restore visual_vs_activation_matched:
  mean logit delta = +0.991862
  95% CI = [+0.507975, +1.508464]
  status = stable_positive

corrupt visual_vs_delta_matched:
  mean logit delta = +0.213216
  95% CI = [-0.037109, +0.501139]
  status = weak_or_heterogeneous_positive

corrupt visual_vs_activation_matched:
  mean logit delta = +0.200358
  95% CI = [-0.020182, +0.490234]
  status = weak_or_heterogeneous_positive
```

判定：

```text
LLaVA matched-control specificity is partial。
它明显强于 activation-matched controls，但 delta-matched controls 吸收了相当一部分效果。
因此 LLaVA 的 hidden-position bridge 不能写成强 matched-control specificity，只能写成 partial / heterogeneous specificity。
```

## 6. 为什么这个结果重要

Stage 2I 的 random/low-delta controls 已经能说明：

```text
source-like hidden positions 强于普通随机位置和低变化位置。
```

Stage 2J 进一步说明：

```text
Qwen:
  source-like positions 仍然强于 delta/activation matched controls，支持更强的 hidden-state specificity。

LLaVA:
  source-like positions 强于 activation-matched controls，但对 delta-matched controls 的优势不稳定。
  这说明 LLaVA 的一部分 bridge 可能由“clean-vs-union hidden delta 大”解释，而不是完全由 source-like specificity 解释。
```

这个结果不是失败，而是让跨模型结论更准确：

```text
Qwen cross-model hidden bridge specificity is stronger.
LLaVA cross-model hidden bridge exists, but matched-control specificity is weaker and more heterogeneous.
```

## 7. 对当前主 claim 的影响

Gemma 主线不变：

```text
Gemma 仍然是唯一拥有 source tracing + node intervention + nearest/random controls + wrong-image/region-mask sensitivity 的完整主链模型。
```

跨模型辅助结论更新为：

```text
Qwen:
  multi-case bidirectional hidden-state bridge + partial decoded bridge + mostly supported matched-control specificity。

LLaVA:
  multi-case bidirectional hidden-state bridge + partial decoded bridge；
  matched-control specificity partial，尤其 delta-matched control 下需要保守。
```

最终跨模型表述建议：

```text
Cross-model experiments show that evidence-mask-sensitive hidden-state bridges are not Gemma-only. Qwen provides the stronger auxiliary replication, including matched-control support. LLaVA also shows bidirectional hidden-state and partial decoded bridge evidence, but delta-matched controls absorb part of the effect, so LLaVA specificity should be described as partial and heterogeneous.
```

## 8. 不能写什么

仍然不能写：

```text
Qwen/LLaVA 已经复现 Gemma source-control causal routes。
Qwen/LLaVA 的 CLT features 是 causal source nodes。
LLaVA 强 matched-control specificity 已经成立。
hidden positions 是对象级语义节点。
D_visual_only 比 B_direct 更好。
```

## 9. 下一步建议

现在不建议立刻继续无脑扩大样本。更有价值的是：

```text
1. typed matched-control analysis:
   分开 symbol_text_reading / visual_readout，看看 LLaVA 的 delta-matched 弱点是不是集中在某类题。

2. case panel:
   为 Qwen 选 2-3 个 matched-control 最稳 case；
   为 LLaVA 选 1 个 activation-matched 成立但 delta-matched 弱的反例 case。

3. 如果要继续升级跨模型强 claim：
   需要 Qwen/LLaVA 内部的 source tracing 或 feature-level attribution，而不是继续只 patch hidden positions。
```

