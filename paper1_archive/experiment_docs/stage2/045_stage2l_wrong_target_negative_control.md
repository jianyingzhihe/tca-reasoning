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

## 2.1 执行与修复记录

第一次远端运行时，Qwen 正常完成，但 LLaVA 全部被跳过。诊断后发现不是 LLaVA 图像位置失败，因为 LLaVA 的 image token span 能稳定定位到 `576` 个 image tokens；真正原因是 wrong-target 选择规则过严。

LLaVA 使用的 SentencePiece tokenizer 经常让不同答案共享一个前导空格 token。旧规则只要 wrong target 的任意候选 token 和 correct target 重叠，就丢弃整个 wrong answer，导致所有 wrong target 都为空。

修复方式：

```text
不再因为任意 candidate overlap 就丢弃整个 wrong answer。
改为过滤掉与 correct target 重叠的候选 token，只保留非重叠候选 token。
```

修复后 LLaVA 重新运行成功：

```text
LLaVA usable_runs = 24 / 24
Qwen usable_runs = 24 / 24
```

## 3. 主组结果

| model_family | direction | group_name | n_rows | mean_correct_effect_logit | mean_wrong_effect_logit | mean_correct_minus_wrong_logit | ci95_low | ci95_high | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llava | corrupt | top_hidden_delta_plus_answer_adjacent | 24 | 0.555501 | 0.025532 | 0.52997 | 0.171885 | 0.987908 | stable_correct_gt_wrong |
| llava | restore | top_hidden_delta_plus_answer_adjacent | 24 | 1.636556 | 0.151321 | 1.485235 | 0.661184 | 2.312505 | stable_correct_gt_wrong |
| qwen | corrupt | top_hidden_delta_plus_answer_adjacent | 24 | 2.953776 | -0.531291 | 3.485067 | 2.478394 | 4.456096 | stable_correct_gt_wrong |
| qwen | restore | top_hidden_delta_plus_answer_adjacent | 24 | 3.442708 | -0.507324 | 3.950033 | 2.830078 | 5.07194 | stable_correct_gt_wrong |

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
        "n_rows": 24,
        "n_samples": 12,
        "mean_correct_effect_logit": 0.555501,
        "mean_wrong_effect_logit": 0.025532,
        "mean_correct_minus_wrong_logit": 0.52997,
        "ci95_low": 0.171885,
        "ci95_high": 0.987908,
        "correct_stronger_n": 16,
        "correct_positive_n": 18,
        "wrong_positive_n": 15,
        "status": "stable_correct_gt_wrong"
      },
      {
        "model_family": "llava",
        "direction": "restore",
        "group_name": "top_hidden_delta_plus_answer_adjacent",
        "n_rows": 24,
        "n_samples": 12,
        "mean_correct_effect_logit": 1.636556,
        "mean_wrong_effect_logit": 0.151321,
        "mean_correct_minus_wrong_logit": 1.485235,
        "ci95_low": 0.661184,
        "ci95_high": 2.312505,
        "correct_stronger_n": 19,
        "correct_positive_n": 21,
        "wrong_positive_n": 15,
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
        "n_rows": 24,
        "n_samples": 12,
        "mean_correct_effect_logit": 2.953776,
        "mean_wrong_effect_logit": -0.531291,
        "mean_correct_minus_wrong_logit": 3.485067,
        "ci95_low": 2.478394,
        "ci95_high": 4.456096,
        "correct_stronger_n": 19,
        "correct_positive_n": 20,
        "wrong_positive_n": 7,
        "status": "stable_correct_gt_wrong"
      },
      {
        "model_family": "qwen",
        "direction": "restore",
        "group_name": "top_hidden_delta_plus_answer_adjacent",
        "n_rows": 24,
        "n_samples": 12,
        "mean_correct_effect_logit": 3.442708,
        "mean_wrong_effect_logit": -0.507324,
        "mean_correct_minus_wrong_logit": 3.950033,
        "ci95_low": 2.830078,
        "ci95_high": 5.07194,
        "correct_stronger_n": 22,
        "correct_positive_n": 20,
        "wrong_positive_n": 7,
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

## 6. 结论边界

本实验支持：

```text
Qwen 和 LLaVA 的 hidden-position bridge 具有 target-specific 成分；
同一组 patch 对当前样本 correct target 的 restore/corrupt 效果明显强于 wrong target。
```

本实验不支持：

```text
Qwen/LLaVA 已经复现 Gemma 的 source-control causal route；
hidden positions 或 CLT features 已经是对象级语义节点；
LLaVA 的 matched-control specificity 已经和 Qwen 一样强。
```

和 Stage 2J/2K 合起来读：

```text
Qwen：matched-control specificity 与 wrong-target specificity 都较强，是目前更强的 auxiliary cross-model line。
LLaVA：wrong-target specificity 成立，说明不是纯 general target movement；但 delta-matched controls 仍会吸收部分效果，所以 specificity 仍写成 partial / heterogeneous。
```
