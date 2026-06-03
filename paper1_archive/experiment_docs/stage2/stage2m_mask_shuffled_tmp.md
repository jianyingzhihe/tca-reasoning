# Stage 2L-3b：Mask-Shuffled Negative Control

## 1. 实验目的

这一步检验另一个替代解释：

```text
hidden bridge 是否只是由任意遮挡造成，而不是由真实证据区域遮挡造成？
```

方法是在同一张图内把 union mask 平移到非原始位置，形成 `mask_shuffled` 条件。然后固定使用真实 evidence mask 导出的 source-like position group，比较：

```text
evidence_mask patch effect
mask_shuffled patch effect
real_minus_shuffled
```

## 2. 主组结果

| model_family | direction | group_name | n_rows | mean_real_mask_effect_logit | mean_shuffled_mask_effect_logit | mean_real_minus_shuffled_logit | ci95_low | ci95_high | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llava | corrupt | top_hidden_delta_plus_answer_adjacent | 48 | 0.65389 | -0.027995 | 0.681885 | 0.380046 | 0.982015 | stable_real_gt_shuffled |
| llava | restore | top_hidden_delta_plus_answer_adjacent | 48 | 1.446818 | 0.108805 | 1.338013 | 0.874959 | 1.808472 | stable_real_gt_shuffled |
| qwen | corrupt | top_hidden_delta_plus_answer_adjacent | 48 | 2.352539 | 0.123698 | 2.228841 | 1.371094 | 3.073242 | stable_real_gt_shuffled |
| qwen | restore | top_hidden_delta_plus_answer_adjacent | 48 | 2.884766 | 0.186198 | 2.698568 | 1.786458 | 3.66862 | stable_real_gt_shuffled |

## 3. 判定

```json
{
  "llava": {
    "verdict": "mask_shuffled_control_supported",
    "direction_statuses": {
      "corrupt": "stable_real_gt_shuffled",
      "restore": "stable_real_gt_shuffled"
    },
    "target_group_rows": [
      {
        "model_family": "llava",
        "direction": "corrupt",
        "group_name": "top_hidden_delta_plus_answer_adjacent",
        "n_rows": 48,
        "n_samples": 24,
        "mean_real_mask_effect_logit": 0.65389,
        "mean_shuffled_mask_effect_logit": -0.027995,
        "mean_real_minus_shuffled_logit": 0.681885,
        "ci95_low": 0.380046,
        "ci95_high": 0.982015,
        "real_stronger_n": 41,
        "status": "stable_real_gt_shuffled"
      },
      {
        "model_family": "llava",
        "direction": "restore",
        "group_name": "top_hidden_delta_plus_answer_adjacent",
        "n_rows": 48,
        "n_samples": 24,
        "mean_real_mask_effect_logit": 1.446818,
        "mean_shuffled_mask_effect_logit": 0.108805,
        "mean_real_minus_shuffled_logit": 1.338013,
        "ci95_low": 0.874959,
        "ci95_high": 1.808472,
        "real_stronger_n": 44,
        "status": "stable_real_gt_shuffled"
      }
    ]
  },
  "qwen": {
    "verdict": "mask_shuffled_control_supported",
    "direction_statuses": {
      "corrupt": "stable_real_gt_shuffled",
      "restore": "stable_real_gt_shuffled"
    },
    "target_group_rows": [
      {
        "model_family": "qwen",
        "direction": "corrupt",
        "group_name": "top_hidden_delta_plus_answer_adjacent",
        "n_rows": 48,
        "n_samples": 24,
        "mean_real_mask_effect_logit": 2.352539,
        "mean_shuffled_mask_effect_logit": 0.123698,
        "mean_real_minus_shuffled_logit": 2.228841,
        "ci95_low": 1.371094,
        "ci95_high": 3.073242,
        "real_stronger_n": 37,
        "status": "stable_real_gt_shuffled"
      },
      {
        "model_family": "qwen",
        "direction": "restore",
        "group_name": "top_hidden_delta_plus_answer_adjacent",
        "n_rows": 48,
        "n_samples": 24,
        "mean_real_mask_effect_logit": 2.884766,
        "mean_shuffled_mask_effect_logit": 0.186198,
        "mean_real_minus_shuffled_logit": 2.698568,
        "ci95_low": 1.786458,
        "ci95_high": 3.66862,
        "real_stronger_n": 38,
        "status": "stable_real_gt_shuffled"
      }
    ]
  }
}
```

## 4. 读法

如果真实 evidence mask 的 patch effect 强于 shifted mask，说明 bridge 对证据位置有一定特异性。

如果 shifted mask 同样强，说明现象可能更多来自遮挡强度、全局分布变化或 answer-adjacent 聚合，而不是严格 evidence-region specificity。

本实验仍然是 hidden-state-level negative control，不是 feature-level source route replication。
