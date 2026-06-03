# Stage 2N-1：Heldout Hidden-State Bridge Replication

## 1. 目的

本实验用 Stage 2M 没有使用过的 localized mask 样本做独立复现，检验：

```text
Qwen/LLaVA 的 evidence-region-sensitive hidden-state bridge 是否不是少数样本偶然结果？
```

这一步只验证 hidden-state-level bridge，不验证 CLT feature-level causal bridge，也不验证 Gemma-style source-control route replication。

## 2. 样本

Manifest：

```text
doc/experiments/stage2/cross_model/stage2n_heldout_manifest.csv
doc/experiments/stage2/cross_model/stage2n_all52_manifest.csv
doc/experiments/stage2/cross_model/stage2n_manifest_summary.json
```

样本规模：

```text
eligible localized samples = 52
Stage 2M used samples = 24
Stage 2N heldout samples = 28
usable_status = pass_min20
```

Heldout 类型分布：

```text
visual_readout = 9
untyped_localized = 19
```

说明：

```text
untyped_localized 是已有 localized 标注中缺少 reasoning_operation 元数据的样本。
它们可以用于 heldout replication，但不能用于强类型化结论。
```

## 3. 输入与配置

模型：

```text
Qwen: Qwen2.5-VL-7B-Instruct, layer 26
LLaVA: LLaVA-1.5-7B, layer 15
```

Prompt：

```text
B_direct
D_visual_only
```

Mask condition：

```text
answer_mask
union_mask
```

主组：

```text
top_hidden_delta_plus_answer_adjacent
```

对照：

```text
random_control_1..4
low_delta_control
delta_matched_plus_answer_adjacent
activation_matched_plus_answer_adjacent
answer_adjacent_text
top_hidden_delta
```

## 4. 方法

对每个 `model x sample x prompt x mask_condition`：

```text
1. 跑 clean image forward。
2. 跑 masked image forward，其中 mask_condition = answer_mask 或 union_mask。
3. 在 visual bucket 中用 clean-vs-mask hidden delta 选 top_hidden_delta positions。
4. 加上 answer-adjacent text positions，形成 top_hidden_delta_plus_answer_adjacent。
5. 做 masked→clean restore 和 clean→masked corrupt。
6. 比较 source-like group 与 random / low-delta / matched controls。
```

核心指标：

```text
restore: masked run 中 patch 回 clean hidden 后，target logit/rank 是否恢复。
corrupt: clean run 中 patch 成 masked hidden 后，target logit/rank 是否受损。
source_minus_random_logit: source-like hidden bridge 是否强于随机 visual positions。
source_minus_delta/activation_matched_logit: source-like hidden bridge 是否强于 matched hidden controls。
```

## 5. 输出

Raw：

```text
doc/experiments/stage2/cross_model/stage2n_qwen_hidden_position_patch.csv/json
doc/experiments/stage2/cross_model/stage2n_llava_hidden_position_patch.csv/json
```

Summary：

```text
doc/experiments/stage2/cross_model/stage2n_heldout_hidden_summary.csv
doc/experiments/stage2/cross_model/stage2n_all52_hidden_summary.csv
doc/experiments/stage2/cross_model/stage2n_hidden_specificity_case.csv
doc/experiments/stage2/cross_model/stage2n_hidden_specificity_summary.csv
doc/experiments/stage2/cross_model/stage2n_hidden_typed_summary.csv
doc/experiments/stage2/cross_model/stage2n_hidden_mask_condition_summary.csv
doc/experiments/stage2/cross_model/stage2n_hidden_prompt_summary.csv
doc/experiments/stage2/cross_model/stage2n_hidden_replication_decision.json
```

## 6. 可用性

```text
Qwen usable_runs = 112 / 112
LLaVA usable_runs = 112 / 112

计算方式：
28 samples x 2 prompts x 2 mask_conditions = 112 runs per model
```

## 7. 主结果

Decision：

```text
overall_status = cross_model_heldout_hidden_replication_supported

Qwen:
  status = heldout_hidden_replication_supported
  source_random_stable_rows = 4/4
  matched_positive_rows = 2/4

LLaVA:
  status = heldout_hidden_replication_supported
  source_random_stable_rows = 4/4
  matched_positive_rows = 3/4
```

Qwen heldout：

```text
answer_mask corrupt:
  source_effect_logit = +3.606585, CI [+2.744420, +4.565290]
  source_minus_random = +3.602121, stable_positive

answer_mask restore:
  source_effect_logit = +4.149833, CI [+3.044364, +5.362444]
  source_minus_random = +4.067732, stable_positive

union_mask corrupt:
  source_effect_logit = +3.232701, CI [+2.445312, +4.167411]
  source_minus_random = +3.226981, stable_positive

union_mask restore:
  source_effect_logit = +3.885324, CI [+2.962612, +4.928571]
  source_minus_random = +3.830357, stable_positive
```

LLaVA heldout：

```text
answer_mask corrupt:
  source_effect_logit = +0.700474, CI [+0.519322, +0.894008]
  source_minus_random = +0.707947, stable_positive

answer_mask restore:
  source_effect_logit = +1.505301, CI [+1.159773, +1.898542]
  source_minus_random = +1.132621, stable_positive

union_mask corrupt:
  source_effect_logit = +0.795759, CI [+0.629813, +0.960484]
  source_minus_random = +0.781546, stable_positive

union_mask restore:
  source_effect_logit = +1.216239, CI [+0.960589, +1.513323]
  source_minus_random = +0.834726, stable_positive
```

## 8. Matched-Control 读法

Qwen：

```text
source_minus_random 全部 stable positive。
source_minus_delta/activation matched 只有 2/4 达到 positive 判据。
说明 Qwen 的 hidden bridge 很强，但 matched controls 能吸收一部分效应。
```

LLaVA：

```text
source_minus_random 全部 stable positive。
matched_positive_rows = 3/4。
union restore 的 delta_matched control 吸收较多效应，因此不写成完全 source-control specificity。
```

## 9. 类型与 Prompt

Heldout 类型：

```text
visual_readout:
  Qwen union source_minus_random restore = +5.852865, stable_positive
  LLaVA union source_minus_random restore = +0.735080, stable_positive

untyped_localized:
  Qwen union source_minus_random restore = +2.872327, stable_positive
  LLaVA union source_minus_random restore = +0.881926, stable_positive
```

Prompt：

```text
Qwen B_direct answer_mask source_minus_random = +4.038993, stable_positive
Qwen D_visual_only answer_mask source_minus_random = +3.630859, stable_positive
LLaVA B_direct answer_mask source_minus_random = +0.941354, stable_positive
LLaVA D_visual_only answer_mask source_minus_random = +0.899214, stable_positive
```

Prompt 读法：

```text
B_direct 与 D_visual_only 都复现 hidden bridge。
这支持 prompt 作为 modulation factor，而不是“D 一定更好”的行为 claim。
```

## 10. 结论

Stage 2N-1 明确加强了 Stage 2M：

```text
Qwen 和 LLaVA 在独立 heldout localized samples 上都复现了 evidence-mask-sensitive hidden bridge。
answer_mask 与 union_mask 均成立。
Qwen 效应更大；LLaVA 效应更小但稳定。
```

仍然不能写：

```text
Qwen/LLaVA 复现了 Gemma source-control causal route。
Qwen/LLaVA 的 CLT feature-level causal bridge 已成立。
```
