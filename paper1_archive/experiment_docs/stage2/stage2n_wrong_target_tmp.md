# Stage 2L-3：Wrong-Target Negative Control

## 1. 实验目的

这一步检验一个关键替代解释：

```text
hidden-position bridge 是否只是普遍推高/拉低任意答案 token，而不是更偏向当前样本的正确 target answer？
```

如果同一组 patch 对正确答案 target 的 restore/corrupt 效果强于错误 target，就能支持：

```text
cross-model hidden bridge 有 answer-specific 成分。
```

但这仍然不能升级为：

```text
Qwen/LLaVA 已经复现 Gemma source-control causal route。
```

## 2. 方法

每个样本保留原来的 correct target，同时从 manifest 的下一个不同样本中取 wrong target answer。

对同一组 hidden-position patch 分别计算：

```text
correct_effect_logit
wrong_effect_logit
correct_minus_wrong_logit
```

主组固定为：

```text
top_hidden_delta_plus_answer_adjacent
```

控制组包括：

```text
delta_matched_plus_answer_adjacent
activation_matched_plus_answer_adjacent
answer_adjacent_text
low_delta_control
random_control_1
```

## 3. 主组结果

| model_family | direction | group_name | n_rows | mean_correct_effect_logit | mean_wrong_effect_logit | mean_correct_minus_wrong_logit | ci95_low | ci95_high | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llava | corrupt | top_hidden_delta_plus_answer_adjacent | 56 | 0.795759 | 0.107214 | 0.688545 | 0.458863 | 0.912659 | stable_correct_gt_wrong |
| llava | restore | top_hidden_delta_plus_answer_adjacent | 56 | 1.216239 | 0.203217 | 1.013022 | 0.628557 | 1.396779 | stable_correct_gt_wrong |
| qwen | corrupt | top_hidden_delta_plus_answer_adjacent | 56 | 3.232701 | 0.01137 | 3.221331 | 2.398856 | 4.118164 | stable_correct_gt_wrong |
| qwen | restore | top_hidden_delta_plus_answer_adjacent | 56 | 3.885324 | 0.029663 | 3.855661 | 2.947213 | 4.880406 | stable_correct_gt_wrong |

## 4. 判定

```json
{
  "llava": {
    "verdict": "wrong_target_control_supported",
    "direction_statuses": {
      "corrupt": "stable_correct_gt_wrong",
      "restore": "stable_correct_gt_wrong"
    },
    "target_group_rows": [
      {
        "model_family": "llava",
        "direction": "corrupt",
        "group_name": "top_hidden_delta_plus_answer_adjacent",
        "n_rows": 56,
        "n_samples": 28,
        "mean_correct_effect_logit": 0.795759,
        "mean_wrong_effect_logit": 0.107214,
        "mean_correct_minus_wrong_logit": 0.688545,
        "ci95_low": 0.458863,
        "ci95_high": 0.912659,
        "correct_stronger_n": 51,
        "correct_positive_n": 51,
        "wrong_positive_n": 30,
        "status": "stable_correct_gt_wrong"
      },
      {
        "model_family": "llava",
        "direction": "restore",
        "group_name": "top_hidden_delta_plus_answer_adjacent",
        "n_rows": 56,
        "n_samples": 28,
        "mean_correct_effect_logit": 1.216239,
        "mean_wrong_effect_logit": 0.203217,
        "mean_correct_minus_wrong_logit": 1.013022,
        "ci95_low": 0.628557,
        "ci95_high": 1.396779,
        "correct_stronger_n": 50,
        "correct_positive_n": 53,
        "wrong_positive_n": 33,
        "status": "stable_correct_gt_wrong"
      }
    ]
  },
  "qwen": {
    "verdict": "wrong_target_control_supported",
    "direction_statuses": {
      "corrupt": "stable_correct_gt_wrong",
      "restore": "stable_correct_gt_wrong"
    },
    "target_group_rows": [
      {
        "model_family": "qwen",
        "direction": "corrupt",
        "group_name": "top_hidden_delta_plus_answer_adjacent",
        "n_rows": 56,
        "n_samples": 28,
        "mean_correct_effect_logit": 3.232701,
        "mean_wrong_effect_logit": 0.01137,
        "mean_correct_minus_wrong_logit": 3.221331,
        "ci95_low": 2.398856,
        "ci95_high": 4.118164,
        "correct_stronger_n": 46,
        "correct_positive_n": 46,
        "wrong_positive_n": 30,
        "status": "stable_correct_gt_wrong"
      },
      {
        "model_family": "qwen",
        "direction": "restore",
        "group_name": "top_hidden_delta_plus_answer_adjacent",
        "n_rows": 56,
        "n_samples": 28,
        "mean_correct_effect_logit": 3.885324,
        "mean_wrong_effect_logit": 0.029663,
        "mean_correct_minus_wrong_logit": 3.855661,
        "ci95_low": 2.947213,
        "ci95_high": 4.880406,
        "correct_stronger_n": 46,
        "correct_positive_n": 47,
        "wrong_positive_n": 30,
        "status": "stable_correct_gt_wrong"
      }
    ]
  }
}
```

## 5. 读法

如果 `top_hidden_delta_plus_answer_adjacent` 在 correct target 上稳定强于 wrong target，说明 bridge 更像 answer-specific bridge。

如果 wrong target 也被同等恢复，则说明现象可能更多是 general logit movement 或 answer-adjacent distribution shift。

本实验只验证 target-specificity，不验证 object-level semantic node，也不验证完整 source-control route replication。
