# Stage 2L-4：Evidence-to-Answer Bridge 位置分解

## 1. 实验目的

Stage 2I/2J 显示最强 bridge 往往是：

```text
top_hidden_delta visual positions + answer-adjacent text positions
```

这一步不重新跑模型，而是复用 Stage 2J patch CSV，把 bridge 拆成：

```text
image-only: top_hidden_delta
answer-adjacent-only: answer_adjacent_text
combo: top_hidden_delta_plus_answer_adjacent
```

核心问题是：

```text
视觉位置本身是否提供额外贡献？
答案附近位置是否是主要汇聚处？
combo 是否强于 image-only 和 answer-adjacent-only？
```

## 2. 输入与输出

输入：

```text
doc/experiments/stage2/cross_model/stage2j_qwen_matched_control_patch.csv
doc/experiments/stage2/cross_model/stage2j_llava_matched_control_patch.csv
```

输出：

```text
doc/experiments/stage2/cross_model/stage2l_bridge_decomposition_case_table.csv
doc/experiments/stage2/cross_model/stage2l_bridge_decomposition_summary.csv
doc/experiments/stage2/cross_model/stage2l_bridge_decomposition_decision.json
doc/experiments/stage2/046_stage2l_evidence_to_answer_bridge_decomposition.md
```

## 3. 总体结果

| model_family | direction | n_rows | mean_combo_minus_image | combo_minus_image_status | mean_combo_minus_answer | combo_minus_answer_status | mean_combo_minus_max_single | combo_minus_max_single_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llava | corrupt | 24 | 0.359863 | stable_positive | 0.225912 | stable_positive | 0.044596 | weak_positive |
| llava | restore | 24 | 0.590657 | stable_positive | 1.197917 | stable_positive | 0.419759 | stable_positive |
| qwen | corrupt | 24 | 2.758464 | stable_positive | 0.147786 | stable_positive | -0.013672 | heterogeneous |
| qwen | restore | 24 | 3.075521 | stable_positive | 0.81901 | stable_positive | 0.460938 | weak_positive |

## 4. 判定

```json
{
  "llava": {
    "verdict": "visual_plus_answer_bridge_supported",
    "combo_minus_answer_status_by_direction": {
      "corrupt": "stable_positive",
      "restore": "stable_positive"
    },
    "combo_minus_image_status_by_direction": {
      "corrupt": "stable_positive",
      "restore": "stable_positive"
    },
    "combo_minus_max_single_status_by_direction": {
      "corrupt": "weak_positive",
      "restore": "stable_positive"
    }
  },
  "qwen": {
    "verdict": "visual_plus_answer_bridge_supported",
    "combo_minus_answer_status_by_direction": {
      "corrupt": "stable_positive",
      "restore": "stable_positive"
    },
    "combo_minus_image_status_by_direction": {
      "corrupt": "stable_positive",
      "restore": "stable_positive"
    },
    "combo_minus_max_single_status_by_direction": {
      "corrupt": "heterogeneous",
      "restore": "weak_positive"
    }
  }
}
```

## 5. 读法

如果 `combo_minus_answer` 为正，说明在 answer-adjacent 位置之外，visual source-like positions 仍有额外贡献。

如果 `combo_minus_image` 为正，说明 answer-adjacent 位置在 image-only 的基础上也有额外贡献。

如果 `combo_minus_max_single` 为正，说明 combo 强于任一单独位置组，更符合 evidence-to-answer bridge / convergence 的解释。

这一步仍然只是 hidden-state-level decomposition，不是 feature-level source route replication。
