# Stage 2P-2：LLaVA Layer/Feature Diagnostic

## 1. 目的

诊断 LLaVA 的 feature-level bridge 失败是否来自当前 layer 15 / top8 设置，而不是直接下结论“LLaVA 没有跨模态 feature”。

## 2. 输入

```text
LLaVA diagnostic prompt-runs = 8
来源 = Stage 2O LLaVA diagnostic rows

模型：
  LLaVA-1.5-7B
  CLT = KokosDev/llava15-7b-clt

layers:
  12, 15, 18, 21

top_k:
  1, 8, 32

mask_condition:
  union_mask
```

网络与资产：

```text
远端脚本已 source /etc/network_turbo。
L12 / L18 / L21 CLT 文件已成功下载。
所有 12 个 layer/top_k 配置均完成运行。
```

## 3. 方法

```text
固定 Stage 2O 的 8 个 LLaVA prompt-runs。

扫描：
  layer = 12, 15, 18, 21
  top_k = 1, 8, 32

position groups:
  top_hidden_delta_plus_answer_adjacent
  top_hidden_delta
  answer_adjacent_text
```

## 4. 输出

```text
doc/experiments/stage2/cross_model/stage2p_llava_layer{12,15,18,21}_top{1,8,32}_feature_bridge.csv/json
doc/experiments/stage2/cross_model/stage2p_llava_layer_sweep_summary.csv
doc/experiments/stage2/cross_model/stage2p_llava_layer_sweep_specificity.csv
doc/experiments/stage2/cross_model/stage2p_llava_layer_sweep_decision.json
```

运行可用性：

```text
12 configurations completed
每个配置 usable_runs = 8
每个配置 rows = 240
```

## 5. 结果

总体判断：

```text
status = llava_layer_sweep_weak_or_partial
strong_configs = []
observed_layers = 12, 15, 18, 21
missing_layers = []
```

最接近正结果的配置：

```text
layer 18, top_k 32, corrupt:
  n_rows = 8
  positive_logit_n = 7/8
  mean_logit_effect = +0.060547
  95% CI = [+0.004883, +0.103516]
  effect_status = stable_positive
  above_all_controls_n = 1/8
  mean_positive_control_count = 2.625 / 4
```

弱正配置：

```text
layer 15, top_k 32, corrupt:
  mean_logit_effect = +0.068359
  effect_status = weak_or_heterogeneous_positive
  above_all_controls_n = 4/8

layer 18, top_k 32, restore:
  mean_logit_effect = +0.020508
  effect_status = weak_or_heterogeneous_positive
  above_all_controls_n = 2/8

layer 21, top_k 1, restore:
  mean_logit_effect = +0.005859
  effect_status = weak_or_heterogeneous_positive
  above_all_controls_n = 3/8

layer 21, top_k 8, restore:
  mean_logit_effect = +0.004395
  effect_status = weak_or_heterogeneous_positive
  above_all_controls_n = 3/8
```

关键解释：

```text
LLaVA sweep 找到了一些 weak / partial signals，尤其 layer 18 top32 corrupt 自身 effect CI 不跨 0。
但是这些配置没有通过 strong specificity 标准：
  evidence_attribution_topk 没有在多数 rows 中强于全部 matched controls。

因此不能写 LLaVA feature bridge supported。
```

## 6. 结论

Stage 2P-2 没有证明 LLaVA feature-level bridge。

可以写：

```text
LLaVA shows weak layer-dependent feature-level signals, with the strongest diagnostic signal at layer 18 top32 corrupt, but matched controls absorb much of the effect.
```

中文：

```text
LLaVA 并非完全没有信号；layer sweep 显示有弱的层敏感现象。
但它没有达到“feature bridge 成立”的标准，因为 source-like feature 不能稳定强于 matched controls。
```

不能写：

```text
LLaVA 没有跨模态 feature。
```

只能写：

```text
LLaVA 当前 CLT feature-level localization remains unproven。
```
