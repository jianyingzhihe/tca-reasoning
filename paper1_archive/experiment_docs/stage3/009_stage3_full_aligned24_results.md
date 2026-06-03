# 009 Stage3 Full Aligned24 Results

## 目的

在首轮 smoke 通过后，使用 Stage3 aligned24 manifest 完整跑 Qwen2.5-VL-PLT、Qwen2.5-VL-CLT 与 LLaVA-CLT，检查 feature bridge 与 approximate source-control probe 是否能在 48 个 prompt-runs 上稳定成立。

## 输入

```text
samples = 24 localized samples
prompt-runs = 48
prompts = B_direct, D_visual_only
mask conditions:
  feature bridge: union_mask
  source-control: answer_mask, union_mask

assets:
  Qwen2.5-VL-PLT = KokosDev/qwen2p5vl-7b-plt
  Qwen2.5-VL-CLT = KokosDev/qwen2p5vl-7b-clt
  LLaVA-CLT = KokosDev/llava15-7b-clt
```

## 输出

```text
stage3_qwen2p5vl_plt_feature_union.csv/json
stage3_qwen2p5vl_clt_feature_union.csv/json
stage3_llava15_clt_feature_union.csv/json
stage3_qwen2p5vl_plt_source_control.csv/json
stage3_qwen2p5vl_clt_source_control.csv/json
stage3_llava15_clt_source_control.csv/json
stage3_full_summary.json
stage3_full_feature_summary_slices.csv
stage3_full_source_control_summary_slices.csv
```

## 方法

Feature bridge 使用 attribution-weighted feature 选择：

```text
feature_score =
  positive(clean_activation - mask_activation)
  * positive(decoder_vector · target_logit_direction)
  * position_weight
```

Source-control probe 使用 approximate source/control pair：

```text
source feature:
  mask drop > 0
  target attribution > 0
  zeroing 损伤 target logit/rank

matched control:
  同层、同 bucket、activation/drop/attribution 接近
  但 mask-insensitive 或 target-attribution 更弱
```

重要边界：这仍然不是 Gemma-style full source tracing。它是跨模型近似 source-control probe。

## 结果

运行状态：

| 资产 | feature rows | source-control rows | source-control usable pairs |
|---|---:|---:|---:|
| Qwen2.5-VL-PLT | 1440 | 450 | 75 |
| Qwen2.5-VL-CLT | 1440 | 576 | 96 |
| LLaVA-CLT | 1440 | 546 | 91 |

Feature bridge 主结果：

| 资产 | 位置组 | 方向 | evidence-control | 95% CI | 正向数 |
|---|---|---|---:|---:|---:|
| Qwen2.5-VL-PLT | answer_adjacent_text | restore | +0.341 | [+0.244, +0.449] | 42/48 |
| Qwen2.5-VL-PLT | answer_adjacent_text | corrupt | +0.207 | [+0.116, +0.294] | 39/48 |
| Qwen2.5-VL-PLT | combined | restore | +0.071 | [+0.030, +0.113] | 36/48 |
| Qwen2.5-VL-PLT | combined | corrupt | +0.028 | [-0.012, +0.067] | 27/48 |
| Qwen2.5-VL-CLT | answer_adjacent_text | restore | +1.161 | [+0.970, +1.359] | 47/48 |
| Qwen2.5-VL-CLT | answer_adjacent_text | corrupt | +0.983 | [+0.793, +1.174] | 45/48 |
| Qwen2.5-VL-CLT | combined | restore | +0.470 | [+0.330, +0.621] | 40/48 |
| Qwen2.5-VL-CLT | combined | corrupt | +0.299 | [+0.164, +0.440] | 39/48 |
| LLaVA-CLT | answer_adjacent_text | restore | +0.018 | [+0.005, +0.030] | 30/48 |
| LLaVA-CLT | combined | restore | +0.006 | [-0.001, +0.015] | 28/48 |

Source-control 主结果：

| 资产 | mask | intervention | source-control | 95% CI | 正向数 |
|---|---|---|---:|---:|---:|
| Qwen2.5-VL-PLT | answer | restore | +0.048 | [+0.012, +0.093] | 15/38 |
| Qwen2.5-VL-PLT | answer | zeroing | +0.232 | [+0.153, +0.345] | 35/38 |
| Qwen2.5-VL-PLT | union | restore | +0.056 | [+0.035, +0.076] | 22/37 |
| Qwen2.5-VL-PLT | union | zeroing | +0.179 | [+0.137, +0.225] | 31/37 |
| Qwen2.5-VL-CLT | answer | restore | +0.186 | [+0.127, +0.244] | 34/48 |
| Qwen2.5-VL-CLT | answer | zeroing | +0.750 | [+0.634, +0.893] | 48/48 |
| Qwen2.5-VL-CLT | union | restore | +0.199 | [+0.123, +0.284] | 34/48 |
| Qwen2.5-VL-CLT | union | zeroing | +0.714 | [+0.574, +0.876] | 48/48 |
| LLaVA-CLT | answer | restore | -0.000 | [-0.004, +0.004] | 13/45 |
| LLaVA-CLT | answer | zeroing | +0.051 | [+0.037, +0.066] | 39/45 |
| LLaVA-CLT | union | restore | +0.002 | [-0.002, +0.007] | 16/46 |
| LLaVA-CLT | union | zeroing | +0.033 | [+0.021, +0.045] | 37/46 |

## 预期与实际偏差

预期 Qwen-CLT 会强于 Qwen-PLT，实际成立。更重要的是，Qwen-PLT 并没有失败：它在 feature restore、source zeroing、source restoration 上都有正向结果，说明 Qwen 的现象不是 CLT-only。

预期 LLaVA-CLT 较弱，实际成立。LLaVA 的 zeroing 有小而稳定的 source-control 正效应，但 restoration 基本不成立，因此不能写稳定 feature/source route replication。

## 结论

当前可以下的 Stage3 结论是：

```text
Qwen2.5-VL 在 PLT 与 CLT 中都出现 evidence-region-sensitive feature/source-control 支持；
CLT 效应更强，PLT 效应更小但方向一致。
因此该现象不是 Gemma-only，也不是 Qwen CLT-only。
```

LLaVA 结论必须保持保守：

```text
LLaVA-CLT 有弱 feature/zeroing 辅助信号，但 restoration/source route localization 未证成。
它支持跨模型异质性，而不是完整 route replication。
```

下一步如果要继续增强论文主线，优先做：

```text
1. Gemma3-PLT 在 Stage3 aligned24 上轻量校准。
2. Qwen2.5-VL-PLT/CLT passing rows 的 first-token/rank 与 short decoded behavior bridge。
3. 若要写“完整跨模型 source route”，实现真正 Qwen ReplacementModel/source tracing adapter。
```
