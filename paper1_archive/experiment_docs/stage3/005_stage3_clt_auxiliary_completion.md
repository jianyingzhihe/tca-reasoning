# 005 Stage3 CLT Auxiliary Completion

## 目的

不浪费 Stage2 已有 CLT 结果，把 Qwen-CLT 与 LLaVA-CLT 补成完整辅助线。CLT 辅助线用于判断 robustness/heterogeneity，而不是替代 PLT 主线。

## 输入

```text
Qwen2.5-VL-CLT:
  base = Qwen/Qwen2.5-VL-7B-Instruct
  transcoder = KokosDev/qwen2p5vl-7b-clt
  layer = 26

LLaVA-CLT:
  base = llava-hf/llava-1.5-7b-hf
  transcoder = KokosDev/llava15-7b-clt
  layer = 15

manifest = stage3_aligned24_manifest.csv
prompts = B_direct, D_visual_only
```

## 输出

```text
cross_model/stage3_qwen2p5vl_clt_feature_union.csv/json
cross_model/stage3_llava15_clt_feature_union.csv/json
后续 full:
  stage3_qwen2p5vl_clt_source_control.csv/json
  stage3_llava15_clt_source_control.csv/json
```

## 方法

Qwen-CLT 与 LLaVA-CLT 都复用 attribution-weighted feature bridge：

```text
evidence_attribution_topk
activation_matched_topk
drop_matched_topk
attribution_matched_mask_insensitive_topk
random_active_topk
```

LLaVA 的目标不是追求 PLT 同构，而是判断：

```text
hidden-state bridge 是否存在？
CLT feature/source route 是否能稳定定位？
如果不能，是否可作为 feature localization 失败但 hidden bridge 存在的反例？
```

## Smoke 结果

Qwen2.5-VL-CLT：

```text
usable prompt-runs = 6/6
raw rows = 180
answer_adjacent_text restore evidence-control = +0.904
answer_adjacent_text corrupt evidence-control = +1.021
combined restore evidence-control = +0.451
combined corrupt evidence-control = +0.424
```

LLaVA-CLT：

```text
usable prompt-runs = 6/6
raw rows = 180
answer_adjacent_text restore evidence-control = +0.014
answer_adjacent_text corrupt evidence-control = +0.010
combined restore evidence-control = -0.001
combined corrupt evidence-control = -0.018
```

## 预期与实际偏差

Qwen-CLT 结果符合预期，且强于 Qwen-PLT。LLaVA-CLT 仍然只显示极弱、位置敏感的 feature 信号，combined 主组不成立。这和 Stage2 的结论一致：LLaVA 更像 hidden-state-level 支持，而不是稳定 feature/source route 支持。

## 当前结论

可以写：

```text
Qwen-CLT 是强 auxiliary positive line。
LLaVA-CLT 是 weak/heterogeneous auxiliary line；它支持 hidden bridge 可能跨模型存在，但当前 CLT feature route localization 未证成。
```

不能写：

```text
LLaVA 没有跨模态特征。
```

因为失败可能来自 CLT 分解、层选择、feature selection 或 adapter 近似，而不能直接推出模型内部没有相关机制。

## Full aligned24 结果

Qwen2.5-VL-CLT full feature bridge：

```text
usable prompt-runs = 48/48
raw feature rows = 1440
answer_adjacent_text restore evidence-control = +1.161, CI [+0.970, +1.359]
answer_adjacent_text corrupt evidence-control = +0.983, CI [+0.793, +1.174]
combined restore evidence-control = +0.470, CI [+0.330, +0.621]
combined corrupt evidence-control = +0.299, CI [+0.164, +0.440]
```

Qwen2.5-VL-CLT full approximate source-control：

```text
usable source-control pairs = 96
answer_mask restore source-control = +0.186, CI [+0.127, +0.244]
answer_mask zeroing source-control = +0.750, CI [+0.634, +0.893]
union_mask restore source-control = +0.199, CI [+0.123, +0.284]
union_mask zeroing source-control = +0.714, CI [+0.574, +0.876]
```

LLaVA-CLT full feature bridge：

```text
usable prompt-runs = 48/48
raw feature rows = 1440
answer_adjacent_text restore evidence-control = +0.018, CI [+0.005, +0.030]
answer_adjacent_text corrupt evidence-control = +0.002, CI [-0.015, +0.020]
combined restore evidence-control = +0.006, CI [-0.001, +0.015]
combined corrupt evidence-control = -0.009, CI [-0.025, +0.007]
```

LLaVA-CLT full approximate source-control：

```text
usable source-control pairs = 91
answer_mask restore source-control = -0.000, CI [-0.004, +0.004]
answer_mask zeroing source-control = +0.051, CI [+0.037, +0.066]
union_mask restore source-control = +0.002, CI [-0.002, +0.007]
union_mask zeroing source-control = +0.033, CI [+0.021, +0.045]
```

## Full 结论

Qwen-CLT 是 Stage3 最强的 auxiliary positive line：feature bridge、source zeroing、source restoration 都稳定为正，且 real mask restoration 远强于 mask_shuffled。

LLaVA-CLT 结果更复杂：zeroing 有稳定小正效应，但 restoration/source-control specificity 不稳定。因此它不能作为 feature-level route replication；更稳的写法仍然是：

```text
LLaVA provides weak CLT auxiliary evidence and stronger hidden-state-level evidence,
but stable CLT feature/source route localization remains unproven.
```
