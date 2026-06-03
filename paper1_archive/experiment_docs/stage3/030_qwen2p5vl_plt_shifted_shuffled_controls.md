# 030 Qwen2.5-VL-PLT Shifted/Shuffled Source-Control Controls

## 目的

补齐 Qwen2.5-VL-PLT source-control probe 的 spatial negative controls。此前 CSV 中的 `mask_shuffled` 历史变量名实际来自 `_shift_mask(...)`，更准确应视为 shifted spatial control。本实验修正 loader 和 source-control runner，使输出同时包含：

```text
real_mask
mask_shifted
mask_shuffled
```

其中 `mask_shifted` 和 `mask_shuffled` 来自 paperpack 导出的 control masks。

## 输入

```text
paperpack72_primary
paperpack72_strict_sensitivity
base: Qwen/Qwen2.5-VL-7B-Instruct
transcoder: KokosDev/qwen2p5vl-7b-plt
layer: 26
position group: top_hidden_delta_plus_answer_adjacent
mask conditions: answer_mask, union_mask
```

## 输出

```text
stage3_qwen2p5vl_plt_source_control_primary_controls_full.csv/json
stage3_qwen2p5vl_plt_source_control_strict_controls_full.csv/json
stage3_qwen2p5vl_plt_controls_full_specificity_summary.csv
```

## 方法

1. 扩展 mask loader，读取 `shifted.png` 和 `shuffled.png`。
2. 扩展 source-control probe，在 real restore 外额外跑 `mask_shifted` restore 和 `mask_shuffled` restore。
3. 不重跑 feature bridge，只补 source-control controls。
4. 比较 `source_minus_control(real)`、`source_minus_control(shifted)`、`source_minus_control(shuffled)`，以及 `real - shifted`、`real - shuffled`。

## 结果

运行规模：

| pack | usable pairs | rows |
|---|---:|---:|
| primary | 242 | 1936 |
| strict | 241 | 1928 |

核心 specificity：

| pack | mask | comparison | mean | 95% CI |
|---|---|---|---:|---|
| primary | answer_mask | real source-control restore | +0.0335 | [+0.0095, +0.0617] |
| primary | answer_mask | shifted source-control restore | +0.0058 | [-0.0086, +0.0209] |
| primary | answer_mask | real - shifted | +0.0278 | [+0.0036, +0.0560] |
| primary | answer_mask | real - shuffled | +0.0244 | [-0.0004, +0.0527] |
| strict | answer_mask | real source-control restore | +0.0358 | [+0.0122, +0.0625] |
| strict | answer_mask | shifted source-control restore | +0.0089 | [-0.0049, +0.0240] |
| strict | answer_mask | real - shifted | +0.0269 | [+0.0028, +0.0553] |
| strict | answer_mask | real - shuffled | +0.0234 | [-0.0017, +0.0512] |
| primary | union_mask | real - shifted | +0.0149 | [-0.0097, +0.0445] |
| strict | union_mask | real - shifted | +0.0162 | [-0.0091, +0.0452] |

## 预期与实际偏差

预期希望 `real > shifted` 和 `real > shuffled` 在 answer/union mask 上都稳定成立。实际结果更细：

```text
answer_mask: real > shifted 稳定成立，primary 与 strict CI 均不跨 0。
answer_mask: real > shuffled 方向为正，但 CI 贴近或略跨 0。
union_mask: real > shifted/shuffled 方向为正，但 CI 跨 0。
```

这说明 spatial control 支持不是“全条件强闭合”，而是主要集中在 answer_mask source-control restore 上。union_mask 更宽，可能引入 relate/background 区域，导致 source specificity 被稀释。

## 结论

可以写：

```text
Qwen2.5-VL-PLT answer-mask source-control restore is stronger on real evidence masks than on shifted spatial controls, and this holds in both primary and strict packs.
```

不能写：

```text
All shifted/shuffled controls are decisively weaker for both answer and union masks.
```

这一步加强了 Qwen-PLT 的 reviewer-proofing，但也要求论文口径更精确：最强 spatial-control 证据是 answer-mask real-vs-shifted，union 和 true shuffled 作为 weaker/partial support。
