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
| llava | corrupt | top_hidden_delta_plus_answer_adjacent | 24 | 0.555501 | 0.004069 | 0.551432 | 0.305013 | 0.839681 | stable_real_gt_shuffled |
| llava | restore | top_hidden_delta_plus_answer_adjacent | 24 | 1.636556 | 0.252604 | 1.383952 | 0.848958 | 1.963704 | stable_real_gt_shuffled |
| qwen | corrupt | top_hidden_delta_plus_answer_adjacent | 24 | 2.953776 | 0.234375 | 2.719401 | 1.63737 | 3.860026 | stable_real_gt_shuffled |
| qwen | restore | top_hidden_delta_plus_answer_adjacent | 24 | 3.442708 | 0.286458 | 3.15625 | 1.980469 | 4.38151 | stable_real_gt_shuffled |

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
        "n_rows": 24,
        "n_samples": 12,
        "mean_real_mask_effect_logit": 0.555501,
        "mean_shuffled_mask_effect_logit": 0.004069,
        "mean_real_minus_shuffled_logit": 0.551432,
        "ci95_low": 0.305013,
        "ci95_high": 0.839681,
        "real_stronger_n": 20,
        "status": "stable_real_gt_shuffled"
      },
      {
        "model_family": "llava",
        "direction": "restore",
        "group_name": "top_hidden_delta_plus_answer_adjacent",
        "n_rows": 24,
        "n_samples": 12,
        "mean_real_mask_effect_logit": 1.636556,
        "mean_shuffled_mask_effect_logit": 0.252604,
        "mean_real_minus_shuffled_logit": 1.383952,
        "ci95_low": 0.848958,
        "ci95_high": 1.963704,
        "real_stronger_n": 22,
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
        "n_rows": 24,
        "n_samples": 12,
        "mean_real_mask_effect_logit": 2.953776,
        "mean_shuffled_mask_effect_logit": 0.234375,
        "mean_real_minus_shuffled_logit": 2.719401,
        "ci95_low": 1.63737,
        "ci95_high": 3.860026,
        "real_stronger_n": 19,
        "status": "stable_real_gt_shuffled"
      },
      {
        "model_family": "qwen",
        "direction": "restore",
        "group_name": "top_hidden_delta_plus_answer_adjacent",
        "n_rows": 24,
        "n_samples": 12,
        "mean_real_mask_effect_logit": 3.442708,
        "mean_shuffled_mask_effect_logit": 0.286458,
        "mean_real_minus_shuffled_logit": 3.15625,
        "ci95_low": 1.980469,
        "ci95_high": 4.38151,
        "real_stronger_n": 20,
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
