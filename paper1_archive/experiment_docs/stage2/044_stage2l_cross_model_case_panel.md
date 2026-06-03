# Stage 2L-1：Cross-Model Case Panel 与失败模式面板

## 1. 实验目的

这一步不重新跑 Qwen/LLaVA，而是复用 Stage 2I/2J/2K 的结果，挑出最适合进入正文或附录的跨模型 case，并把失败模式单独列出来。

核心问题是：

```text
哪些 case 最支持 Qwen 的 matched-control hidden bridge？
哪些 case 说明 LLaVA 的 bridge 仍然 partial / heterogeneous？
下一步扩大样本时应该优先扩大哪些类型？
```

## 2. 输入与输出

输入：

```text
doc/experiments/stage2/cross_model/stage2k_matched_control_explanation_case.csv
doc/experiments/stage2/cross_model/stage2i_selected_12_manifest.csv
doc/experiments/stage2/cross_model/stage2i_qwen_decoded_bridge_expansion.csv
doc/experiments/stage2/cross_model/stage2i_llava_decoded_bridge_expansion.csv
```

输出：

```text
doc/experiments/stage2/cross_model/stage2l_case_panel.csv
doc/experiments/stage2/cross_model/stage2l_case_panel_decision.json
doc/experiments/stage2/044_stage2l_cross_model_case_panel.md
```

## 3. 方法

对每个 `model x sample x prompt` 汇总 restore 与 corrupt 两个方向：

```text
restore_combo_minus_delta
restore_combo_minus_activation
corrupt_combo_minus_delta
corrupt_combo_minus_activation
```

同时记录：

```text
是否 above_both_matched_controls
是否被 delta-matched / activation-matched controls 吸收
decoded bridge 是否改变 union_mask 下的答案
题型、答案、问题、answer mask 面积比例
```

分类规则是保守的：

```text
Qwen strong-positive: matched-control 四项里多数为正，且至少一个方向 above_both。
LLaVA strong-positive: 同时强于 delta/activation matched controls。
LLaVA delta-diagnostic: activation control 下有信号，但 delta control 吸收明显。
weak/failure: 不足以作为正证据，只适合失败分析。
```

## 4. 总体统计

```json
{
  "llava_strong_positive": 12,
  "llava_activation_supported": 6,
  "llava_delta_diagnostic": 2,
  "llava_weak_or_failure": 4,
  "qwen_strong_positive": 5,
  "qwen_moderate_positive": 4,
  "qwen_weak_or_failure": 15
}
```

## 5. Qwen 正证据候选

| model_family | sample_id | prompt_name | answer_text | reasoning_operation | positive_matched_control_count | case_category | evidence_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen | okvqa_val_3959785 | B_direct | kuwait airway | symbol_text_reading | 4 | qwen_strong_positive | 12.0 |
| qwen | okvqa_val_3959785 | D_visual_only | kuwait airway | symbol_text_reading | 4 | qwen_strong_positive | 12.0 |
| qwen | okvqa_val_4502065 | B_direct | tortoise | symbol_text_reading | 4 | qwen_strong_positive | 12.0 |
| qwen | okvqa_val_4502065 | D_visual_only | tortoise | symbol_text_reading | 4 | qwen_strong_positive | 12.0 |
| qwen | okvqa_val_1058855 | B_direct | fire hydrant | visual_readout | 4 | qwen_strong_positive | 11.0 |
| qwen | okvqa_val_1593205 | D_visual_only | tokyo | symbol_text_reading | 2 | qwen_moderate_positive | 5.5 |

读法：

```text
Qwen 的正文/附录正证据应优先从这些 case 里选。
它们主要用于支持：Qwen 不是只有 readout sensitivity，而是存在更强的 hidden-state-level evidence-to-answer bridge。
仍然不能写成 Qwen 已经完成 Gemma-style source-control causal route replication。
```

## 6. LLaVA 候选与诊断

| model_family | sample_id | prompt_name | answer_text | reasoning_operation | positive_matched_control_count | case_category | evidence_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llava | okvqa_val_01514 | B_direct | 50 pounds | visual_readout | 4 | llava_strong_positive | 12.0 |
| llava | okvqa_val_01514 | D_visual_only | 50 pounds | visual_readout | 4 | llava_strong_positive | 12.0 |
| llava | okvqa_val_1058855 | B_direct | fire hydrant | visual_readout | 4 | llava_strong_positive | 12.0 |
| llava | okvqa_val_1593205 | D_visual_only | tokyo | symbol_text_reading | 4 | llava_strong_positive | 12.0 |
| llava | okvqa_val_1058855 | D_visual_only | fire hydrant | visual_readout | 4 | llava_strong_positive | 11.0 |
| llava | okvqa_val_1593205 | B_direct | tokyo | symbol_text_reading | 4 | llava_strong_positive | 11.0 |

读法：

```text
LLaVA 可以作为“不是 Gemma-only / Qwen-only”的辅助证据。
但它的 specificity 弱于 Qwen，尤其 delta-matched controls 会解释一部分效果。
因此 LLaVA 更适合放在 partial cross-model support 和 heterogeneity analysis。
```

## 7. 失败或诊断 case

| model_family | sample_id | prompt_name | answer_text | reasoning_operation | positive_matched_control_count | case_category | evidence_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llava | okvqa_val_3959785 | D_visual_only | kuwait airway | symbol_text_reading | 2 | llava_delta_diagnostic | 3.5 |
| llava | okvqa_val_2954205 | D_visual_only | move | symbol_text_reading | 1 | llava_delta_diagnostic | 0.5 |
| llava | okvqa_val_2708155 | B_direct | left | visual_readout | 0 | llava_weak_or_failure | -1.0 |
| llava | okvqa_val_1729795 | B_direct | arrow | symbol_text_reading | 0 | llava_weak_or_failure | -2.0 |
| llava | okvqa_val_1729795 | D_visual_only | arrow | symbol_text_reading | 0 | llava_weak_or_failure | -2.0 |
| llava | okvqa_val_2708155 | D_visual_only | left | visual_readout | 0 | llava_weak_or_failure | -2.0 |
| qwen | okvqa_val_1729795 | D_visual_only | arrow | symbol_text_reading | 1 | qwen_weak_or_failure | 2.0 |
| qwen | okvqa_val_01514 | D_visual_only | 50 pounds | visual_readout | 1 | qwen_weak_or_failure | 1.0 |

这些 case 的作用不是削弱主线，而是帮助我们防止过度叙事：

```text
如果一个 case 被 delta-matched control 吸收，就不能写成强 source-like specificity。
如果一个 case 只有 answer-adjacent text 位置有效，就更像 answer-local aggregation，而不是纯视觉位置路径。
如果 decoded answer 不动，只能写 first-token / rank bridge。
```

## 8. 当前结论

Stage 2L-1 支持继续推进，但也把边界画得更清楚：

```text
Qwen：适合继续做扩大样本、negative controls、decoded bridge only-on-passing-cases。
LLaVA：适合继续做 diagnostic replication，重点解释 delta-matched control 为什么吸收效果。
Gemma：仍然是唯一完整主链模型，cross-model 目前只作为 auxiliary support。
```

## 9. 下一步建议

推荐马上做两件事：

```text
1. Stage 2L-3 negative controls：mask-shuffled 与 wrong-target token。
2. Stage 2L-4 evidence-to-answer 汇聚位置分析：image-only / answer-adjacent-only / image+answer-adjacent。
```

如果这两步继续支持 Qwen，再扩到 24 / 36 个样本会更有意义；否则先不要盲目扩大样本。
