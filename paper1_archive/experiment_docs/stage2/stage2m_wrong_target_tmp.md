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
| llava | corrupt | top_hidden_delta_plus_answer_adjacent | 48 | 0.65389 | -0.034512 | 0.688402 | 0.370091 | 1.009811 | stable_correct_gt_wrong |
| llava | restore | top_hidden_delta_plus_answer_adjacent | 48 | 1.446818 | -0.147494 | 1.594312 | 1.053792 | 2.11364 | stable_correct_gt_wrong |
| qwen | corrupt | top_hidden_delta_plus_answer_adjacent | 48 | 2.352539 | -0.336975 | 2.689514 | 1.899679 | 3.462199 | stable_correct_gt_wrong |
| qwen | restore | top_hidden_delta_plus_answer_adjacent | 48 | 2.884766 | -0.344482 | 3.229248 | 2.322876 | 4.13147 | stable_correct_gt_wrong |

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
        "n_rows": 48,
        "n_samples": 24,
        "mean_correct_effect_logit": 0.65389,
        "mean_wrong_effect_logit": -0.034512,
        "mean_correct_minus_wrong_logit": 0.688402,
        "ci95_low": 0.370091,
        "ci95_high": 1.009811,
        "correct_stronger_n": 37,
        "correct_positive_n": 37,
        "wrong_positive_n": 27,
        "status": "stable_correct_gt_wrong"
      },
      {
        "model_family": "llava",
        "direction": "restore",
        "group_name": "top_hidden_delta_plus_answer_adjacent",
        "n_rows": 48,
        "n_samples": 24,
        "mean_correct_effect_logit": 1.446818,
        "mean_wrong_effect_logit": -0.147494,
        "mean_correct_minus_wrong_logit": 1.594312,
        "ci95_low": 1.053792,
        "ci95_high": 2.11364,
        "correct_stronger_n": 41,
        "correct_positive_n": 41,
        "wrong_positive_n": 24,
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
        "n_rows": 48,
        "n_samples": 24,
        "mean_correct_effect_logit": 2.352539,
        "mean_wrong_effect_logit": -0.336975,
        "mean_correct_minus_wrong_logit": 2.689514,
        "ci95_low": 1.899679,
        "ci95_high": 3.462199,
        "correct_stronger_n": 38,
        "correct_positive_n": 40,
        "wrong_positive_n": 16,
        "status": "stable_correct_gt_wrong"
      },
      {
        "model_family": "qwen",
        "direction": "restore",
        "group_name": "top_hidden_delta_plus_answer_adjacent",
        "n_rows": 48,
        "n_samples": 24,
        "mean_correct_effect_logit": 2.884766,
        "mean_wrong_effect_logit": -0.344482,
        "mean_correct_minus_wrong_logit": 3.229248,
        "ci95_low": 2.322876,
        "ci95_high": 4.13147,
        "correct_stronger_n": 42,
        "correct_positive_n": 40,
        "wrong_positive_n": 20,
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
