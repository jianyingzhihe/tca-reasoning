# 008 Stage3 Initial Smoke Results

## 目的

在跑完整 aligned24 之前，先确认 Stage3 的关键资产和脚本链路是否能跑通，并初步观察 Qwen2.5-VL 的 PLT 与 CLT 是否同方向。

## 输入

```text
manifest = stage3_aligned24_manifest.csv
run manifest = stage3_aligned48_prompt_runs.csv
smoke limit = first 6 prompt-runs

models/assets:
  Qwen2.5-VL-PLT: KokosDev/qwen2p5vl-7b-plt
  Qwen2.5-VL-CLT: KokosDev/qwen2p5vl-7b-clt
  LLaVA-CLT: KokosDev/llava15-7b-clt
```

## 输出

```text
stage3_asset_preflight.json/csv
stage3_qwen2p5vl_plt_feature_union.json/csv
stage3_qwen2p5vl_clt_feature_union.json/csv
stage3_llava15_clt_feature_union.json/csv
stage3_smoke_summary.json
stage3_smoke_summary_slices.csv
stage3_smoke_comparisons.csv
```

## 方法

远端环境：

```text
source scripts/server/dev.sh
source /etc/network_turbo
HF_HOME=/root/autodl-tmp/tca-reasoning/data/hf_cache
```

每条资产跑 attribution-weighted feature bridge：

```text
mask_condition = union_mask
position_groups =
  top_hidden_delta_plus_answer_adjacent
  top_hidden_delta
  answer_adjacent_text
directions = restore, corrupt
feature groups =
  evidence_attribution_topk
  activation_matched_topk
  drop_matched_topk
  attribution_matched_mask_insensitive_topk
  random_active_topk
```

## 结果

远端资源：

```text
disk free = 119GB
GPU = NVIDIA vGPU-48GB, free ~= 48GB
Qwen2.5-VL base model cache exists
LLaVA base model cache exists
```

运行结果：

```text
Qwen2.5-VL-PLT: pass, usable_runs=6, rows=180
Qwen2.5-VL-CLT: pass, usable_runs=6, rows=180
LLaVA-CLT: pass, usable_runs=6, rows=180
```

关键数值：

| 资产 | 位置组 | 方向 | evidence-control |
|---|---|---|---:|
| Qwen2.5-VL-PLT | answer_adjacent_text | restore | +0.294 |
| Qwen2.5-VL-PLT | answer_adjacent_text | corrupt | +0.188 |
| Qwen2.5-VL-PLT | combined | restore | +0.085 |
| Qwen2.5-VL-PLT | combined | corrupt | +0.086 |
| Qwen2.5-VL-CLT | answer_adjacent_text | restore | +0.904 |
| Qwen2.5-VL-CLT | answer_adjacent_text | corrupt | +1.021 |
| Qwen2.5-VL-CLT | combined | restore | +0.451 |
| Qwen2.5-VL-CLT | combined | corrupt | +0.424 |
| LLaVA-CLT | answer_adjacent_text | restore | +0.014 |
| LLaVA-CLT | answer_adjacent_text | corrupt | +0.010 |
| LLaVA-CLT | combined | restore | -0.001 |
| LLaVA-CLT | combined | corrupt | -0.018 |

其中 `combined` 指 `top_hidden_delta_plus_answer_adjacent`。

## 预期与实际偏差

预期 Qwen2.5-VL-PLT 若为正，说明 Qwen 证据不是 CLT-only。实际结果符合，但 PLT 明显弱于 CLT。

预期 LLaVA-CLT 可能较弱，实际结果也符合：只在 answer-adjacent text 位置有很小正效应，combined 主组不稳定。

## 结论

Stage3 smoke 通过。可以继续 full aligned24。

当前最稳的解释是：

```text
Qwen2.5-VL 存在跨 transcoder 类型同方向的 evidence-sensitive feature bridge，
但效应大小具有 representation dependence。
LLaVA-CLT 目前仍是 hidden/weak feature auxiliary，不是稳定 source-route 复现。
```
