# Stage 2M-1：Expanded Hidden-State Cross-Model Replication

## 1. 目的

本实验把 Stage 2L 的 12-sample hidden bridge 证据扩到 24 个 localized samples，目标是验证：

```text
Qwen 和 LLaVA 是否都存在 evidence-sensitive、target-specific、evidence-location-specific hidden-state bridge。
```

这一步只判断 Tier 1 hidden-state replication，不判断 feature-level causal bridge，也不判断 Gemma-style source-control route replication。

## 2. Manifest

```json
{
  "target_count": 24,
  "selected_count": 24,
  "available_eligible_count": 52,
  "type_quotas": {
    "symbol_text_reading": 12,
    "visual_readout": 8,
    "scene_inference": 4
  },
  "selected_type_counts": {
    "symbol_text_reading": 11,
    "visual_readout": 11,
    "scene_inference": 2
  },
  "quota_deficits": {
    "symbol_text_reading": 1,
    "scene_inference": 2
  },
  "usable_status": "pass_min20",
  "selected_samples": [
    "okvqa_val_1593205",
    "okvqa_val_4739195",
    "okvqa_val_4502065",
    "okvqa_val_2954205",
    "okvqa_val_1729795",
    "okvqa_val_3959785",
    "okvqa_val_4043385",
    "okvqa_val_4938465",
    "okvqa_val_136595",
    "okvqa_val_2847255",
    "okvqa_val_3658865",
    "okvqa_val_1058855",
    "okvqa_val_340155",
    "okvqa_val_01514",
    "okvqa_val_2708155",
    "okvqa_val_02444",
    "okvqa_val_1704425",
    "okvqa_val_02422",
    "okvqa_val_3774865",
    "okvqa_val_5334645",
    "okvqa_val_03149",
    "okvqa_val_5606265",
    "okvqa_val_512035",
    "okvqa_val_3608785"
  ],
  "claim_boundary": "Stage 2M manifest selects existing localized samples for cross-model replication. It does not establish replication by itself."
}
```

类型配比没有强行伪造：现有样本只有 11 个 `symbol_text_reading` 和 2 个 `scene_inference` 可用，因此缺口由高分 `visual_readout` 补齐。

## 3. Tier 1 Verdict

| model_family | tier1_status | matched_control_status | wrong_target_status | mask_shuffled_status |
| --- | --- | --- | --- | --- |
| qwen | tier1_hidden_mostly_supported | matched_specificity_partial | wrong_target_control_supported | mask_shuffled_control_supported |
| llava | tier1_hidden_full_supported | matched_specificity_supported | wrong_target_control_supported | mask_shuffled_control_supported |

Overall:

```text
cross_model_hidden_replication_supported_qwen_stronger_llava_smaller
```

可用性：

```text
Qwen hidden matched controls: 48/48 prompt-runs usable
LLaVA hidden matched controls: 48/48 prompt-runs usable
Qwen wrong-target control: 48/48 prompt-runs usable
LLaVA wrong-target control: 48/48 prompt-runs usable
Qwen mask-shuffled control: 48/48 prompt-runs usable
LLaVA mask-shuffled control: 48/48 prompt-runs usable
```

核心读法：

```text
Qwen 的 hidden bridge effect size 更大，但 matched-control 的 corrupt 方向只有 weak / heterogeneous positive，因此 Tier 1 写成 mostly supported。
LLaVA 的 effect size 更小，但 matched-control、wrong-target、mask-shuffled 三类控制更均衡，因此 Tier 1 可写成 hidden-state full supported。
这不是说 LLaVA 比 Qwen 机制更强，而是说在当前控制表中 LLaVA 的相对 specificity 更稳定；绝对 effect size 仍是 Qwen 更大。
```

## 4. Wrong-Target 与 Mask-Shuffled 负控制

| model_family | direction | group_name | n_rows | mean_correct_minus_wrong_logit | mean_real_minus_shuffled_logit | status |
| --- | --- | --- | --- | --- | --- | --- |
| llava | corrupt | top_hidden_delta_plus_answer_adjacent | 48 | 0.688402 |  | stable_correct_gt_wrong |
| llava | restore | top_hidden_delta_plus_answer_adjacent | 48 | 1.594312 |  | stable_correct_gt_wrong |
| qwen | corrupt | top_hidden_delta_plus_answer_adjacent | 48 | 2.689514 |  | stable_correct_gt_wrong |
| qwen | restore | top_hidden_delta_plus_answer_adjacent | 48 | 3.229248 |  | stable_correct_gt_wrong |
| llava | corrupt | top_hidden_delta_plus_answer_adjacent | 48 |  | 0.681885 | stable_real_gt_shuffled |
| llava | restore | top_hidden_delta_plus_answer_adjacent | 48 |  | 1.338013 | stable_real_gt_shuffled |
| qwen | corrupt | top_hidden_delta_plus_answer_adjacent | 48 |  | 2.228841 | stable_real_gt_shuffled |
| qwen | restore | top_hidden_delta_plus_answer_adjacent | 48 |  | 2.698568 | stable_real_gt_shuffled |

## 5. Matched-Control 读法

Matched-control model summary 文件：

```text
doc/experiments/stage2/cross_model/stage2m_matched_control_model_summary.csv
```

Model-level matched-control summary：

```text
Qwen:
  restore combo_minus_delta = +0.186198, stable_positive
  restore combo_minus_activation = +0.216146, stable_positive
  corrupt combo_minus_delta = +0.051758, weak_or_heterogeneous_positive
  corrupt combo_minus_activation = +0.037760, weak_or_heterogeneous_positive

LLaVA:
  restore combo_minus_delta = +0.187012, weak_or_heterogeneous_positive
  restore combo_minus_activation = +0.925293, stable_positive
  corrupt combo_minus_delta = +0.263753, stable_positive
  corrupt combo_minus_activation = +0.283447, stable_positive
```

类型化读法：

```text
Qwen: symbol_text_reading 最强；visual_readout 的 matched-control specificity 更弱。
LLaVA: visual_readout 和 symbol_text_reading 都有正向结果；scene_inference 样本只有 2 个，不能作强结论。
```

因此：

```text
Qwen 可写成 large-effect hidden bridge replication with partially stable matched specificity。
LLaVA 可写成 smaller-effect hidden bridge replication with stable target/location controls and stronger relative matched-control specificity。
```

两者都不能写成 Gemma-style source-control route replication。

## 6. 下一步候选

根据 Tier 1 passing case，已经生成 decoded bridge 候选：

```text
doc/experiments/stage2/cross_model/stage2m_decoded_candidate_prompt_rows.csv
doc/experiments/stage2/cross_model/stage2m_decoded_candidate_manifest.csv
doc/experiments/stage2/cross_model/stage2m_decoded_candidate_decision.json
```

候选规模：

```text
Qwen: 8 个 unique samples
LLaVA: 6 个 unique samples
合计: 10 个 unique samples，因为部分样本同时入选两个模型
```

这些样本只用于 Stage 2M-2 decoded bridge smoke，不代表新的人工标注包，也不代表 generation-level 复现已经成立。

## 7. 结论边界

可写：

```text
Qwen and LLaVA both show expanded hidden-state bridge replication with target-specific and evidence-location-specific controls; Qwen is stronger, while LLaVA shows a smaller but still positive hidden-state bridge.
```

不可写：

```text
Qwen/LLaVA fully replicate Gemma source-control causal routes.
```
