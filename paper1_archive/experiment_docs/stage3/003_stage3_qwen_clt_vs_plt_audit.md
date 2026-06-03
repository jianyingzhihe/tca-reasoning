# 003 Stage3 Qwen CLT vs PLT Audit

## 目的

审计 Stage2 Qwen 正结果是否依赖 CLT。Qwen2.5-VL 同时有公开 PLT 与 CLT，因此它是判断 `transcoder-type robustness` 的核心模型。

## 输入

```text
base model = Qwen/Qwen2.5-VL-7B-Instruct
PLT = KokosDev/qwen2p5vl-7b-plt
CLT = KokosDev/qwen2p5vl-7b-clt
manifest = stage3_aligned24_manifest.csv
run manifest = stage3_aligned48_prompt_runs.csv
prompts = B_direct, D_visual_only
layer = 26
mask condition = union_mask
position groups:
  top_hidden_delta_plus_answer_adjacent
  top_hidden_delta
  answer_adjacent_text
```

## 输出

```text
cross_model/stage3_qwen2p5vl_plt_feature_union.csv/json
cross_model/stage3_qwen2p5vl_clt_feature_union.csv/json
cross_model/stage3_smoke_summary_slices.csv
cross_model/stage3_smoke_comparisons.csv
```

## 方法

对同一批样本和 prompt-runs，分别加载 Qwen2.5-VL-PLT 与 Qwen2.5-VL-CLT。用 attribution-weighted feature 选择：

```text
feature_score =
  positive(clean_activation - mask_activation)
  * positive(decoder_vector · target_logit_direction)
  * position_weight
```

然后比较：

```text
evidence_attribution_topk
activation_matched_topk
drop_matched_topk
attribution_matched_mask_insensitive_topk
random_active_topk
```

干预方向：

```text
masked -> clean restoration
clean -> masked corruption
```

## Smoke 结果

首轮 smoke 每个资产跑 6 个 prompt-runs，每个输出 180 行。

| 资产 | 位置组 | 方向 | evidence 均值 | control 均值 | evidence-control | 正向数 |
|---|---|---|---:|---:|---:|---:|
| Qwen2.5-VL-PLT | answer_adjacent_text | restore | +0.286 | -0.008 | +0.294 | 6/6 |
| Qwen2.5-VL-PLT | answer_adjacent_text | corrupt | +0.208 | +0.021 | +0.188 | 6/6 |
| Qwen2.5-VL-PLT | top_hidden_delta_plus_answer_adjacent | restore | +0.083 | -0.001 | +0.085 | 5/6 |
| Qwen2.5-VL-PLT | top_hidden_delta_plus_answer_adjacent | corrupt | +0.104 | +0.018 | +0.086 | 5/6 |
| Qwen2.5-VL-CLT | answer_adjacent_text | restore | +0.823 | -0.081 | +0.904 | 6/6 |
| Qwen2.5-VL-CLT | answer_adjacent_text | corrupt | +0.990 | -0.031 | +1.021 | 6/6 |
| Qwen2.5-VL-CLT | top_hidden_delta_plus_answer_adjacent | restore | +0.453 | +0.003 | +0.451 | 5/6 |
| Qwen2.5-VL-CLT | top_hidden_delta_plus_answer_adjacent | corrupt | +0.458 | +0.034 | +0.424 | 6/6 |

## 预期与实际偏差

预期如果 PLT 与 CLT 都同方向，则说明 Qwen evidence 不是 CLT-only。实际 smoke 显示二者同方向，但 CLT 效应明显大于 PLT。

另一个偏差是：纯 `top_hidden_delta` 视觉位置组不稳定，主要信号来自 `answer_adjacent_text` 或 `top_hidden_delta_plus_answer_adjacent`。这提示 Qwen 的答案相关信号可能需要在答案邻近文本位置汇聚，而不是只在 visual token 位置直接体现。

## 当前结论

可以暂时写：

```text
Qwen2.5-VL 的 feature bridge 信号不是 CLT-only；PLT 中也出现同方向正信号。
但 CLT 效应更强，说明 evidence-sensitive feature bridge 的效应大小受 transcoder 类型影响。
```

不能写：

```text
Qwen2.5-VL-PLT 已完整复现 Gemma-style source tracing route。
```

原因是当前仍是 attribution-weighted feature patch 和 approximate source/control probe，不是真正的 ReplacementModel source tracing。

## Full aligned24 结果

Full run 使用 24 个样本、48 个 prompt-runs。Qwen2.5-VL-PLT 与 Qwen2.5-VL-CLT 都完成 48/48 usable runs，各输出 1440 行 feature bridge 结果。

Feature bridge 的主结果如下：

| 资产 | 位置组 | 方向 | evidence-control 均值 | 95% bootstrap CI | 正向数 |
|---|---|---|---:|---:|---:|
| Qwen2.5-VL-PLT | answer_adjacent_text | restore | +0.341 | [+0.244, +0.449] | 42/48 |
| Qwen2.5-VL-PLT | answer_adjacent_text | corrupt | +0.207 | [+0.116, +0.294] | 39/48 |
| Qwen2.5-VL-PLT | combined | restore | +0.071 | [+0.030, +0.113] | 36/48 |
| Qwen2.5-VL-PLT | combined | corrupt | +0.028 | [-0.012, +0.067] | 27/48 |
| Qwen2.5-VL-CLT | answer_adjacent_text | restore | +1.161 | [+0.970, +1.359] | 47/48 |
| Qwen2.5-VL-CLT | answer_adjacent_text | corrupt | +0.983 | [+0.793, +1.174] | 45/48 |
| Qwen2.5-VL-CLT | combined | restore | +0.470 | [+0.330, +0.621] | 40/48 |
| Qwen2.5-VL-CLT | combined | corrupt | +0.299 | [+0.164, +0.440] | 39/48 |

其中 `combined` 指 `top_hidden_delta_plus_answer_adjacent`。

## Full 结论

Qwen2.5-VL 的跨模型 feature bridge 已经不只是 smoke 现象。PLT 与 CLT 在 full aligned24 上同方向成立，且 PLT 的 restore 主项 CI 不跨 0。CLT 效应显著更大，因此更精确的口径是：

```text
Qwen2.5-VL evidence-sensitive feature bridge is robust in direction across PLT and CLT,
but strongly representation-dependent in effect size.
```

中文口径：

```text
Qwen2.5-VL 的证据区域敏感 feature bridge 不是 CLT-only；
PLT 中也存在同方向证据，但效应更小，说明该现象受 transcoder 类型影响。
```
