# 010 Stage3 Gemma3-PLT Overlap Calibration

## 目的

检查 Stage3 aligned24 manifest 是否可以直接复用已有 Gemma3-PLT 主线 source/control 结果，作为 PLT-aligned cross-model 主线里的 Gemma 校准基线。

这里的核心问题不是重新证明 Gemma 主线，而是防止口径混淆：

```text
如果 Stage3 aligned24 与历史 Gemma source/control 元数据高度重叠，
  可以直接做同 manifest 的 Gemma baseline。
如果重叠不足，
  只能写 overlap calibration，不能写完整 Stage3 Gemma rerun。
```

## 输入

```text
Stage3 manifest:
  doc/experiments/stage3/cross_model/stage3_aligned24_manifest.csv

Gemma3-PLT historical prefix-fix mainline:
  remote_sync/2026-05-19_core24_prefixfix_random16/region_mask_mainline_prefixfix_random16.csv

Gemma support source-nearest summary:
  doc/5.16/core24_prefixfix_region_analysis_2026-05-19/random16_analysis/support_source_nearest_pairs_iou0p06.csv
```

## 输出

```text
doc/experiments/stage3/cross_model/stage3_gemma_overlap_manifest.csv
doc/experiments/stage3/cross_model/stage3_gemma_overlap_raw_rows.csv
doc/experiments/stage3/cross_model/stage3_gemma_overlap_support_pairs.csv
doc/experiments/stage3/cross_model/stage3_gemma_overlap_summary.csv
doc/experiments/stage3/cross_model/stage3_gemma_missing_source_samples.csv
doc/experiments/stage3/cross_model/stage3_gemma_overlap_decision.json
```

本地脚本：

```text
scripts/local/analyze_stage3_gemma_overlap_calibration.py
```

## 方法

对 Stage3 aligned24 的每个样本检查是否同时具备：

```text
support source clean rows
support nearest-control clean rows
answer_mask / union_mask / random16 condition rows
```

专有名词解释：

```text
PLT:
  Per-Layer Transcoder，逐层转码器。这里指每层独立学习的稀疏特征分解资产。

source node:
  在 Gemma 主线中由 attribution/source tracing 识别出的候选内部特征节点。

nearest control:
  与 source node 在层、位置、激活或其他匹配标准上接近，但不是 source 的对照节点。

weakening:
  证据区域遮挡后，节点干预对目标答案 logit 的影响减弱量。
  本分析中对 support node 使用：
  weakening = condition_delta_target_logit - clean_delta_target_logit

random16:
  16 个同面积随机区域遮挡对照，用来排除“只要遮住任意区域都会削弱路径”的解释。
```

## 结果

严格按 support source + nearest control 同时可用来算，Stage3 aligned24 中只有 4 个样本可作为 Gemma overlap calibration：

| sample_id | 类型 | 可用 prompt |
|---|---|---|
| `okvqa_val_02444` | visual_readout | B_direct, D_visual_only |
| `okvqa_val_2847255` | symbol_text_reading | D_visual_only |
| `okvqa_val_4739195` | symbol_text_reading | B_direct, D_visual_only |
| `okvqa_val_5334645` | scene_inference | B_direct, D_visual_only |

因此可用 comparison rows 为 7 条，而不是完整 48 prompt-runs。

主 summary：

| 条件 | 指标 | n | 均值 | 95% CI | 正向数 |
|---|---|---:|---:|---:|---:|
| answer_mask | source_minus_nearest | 7 | +0.973 | [+0.170, +1.875] | 5/7 |
| union_mask | source_minus_nearest | 7 | +1.143 | [+0.321, +2.071] | 6/7 |
| answer_mask | source_minus_random16 | 7 | +0.469 | [+0.061, +0.890] | 4/7 |
| union_mask | source_minus_random16 | 7 | +0.522 | [+0.122, +0.924] | 5/7 |

类型切片中，visual_readout 与 symbol_text_reading 方向较清楚；scene_inference 仍然小样本且异质。

## 预期与实际偏差

预期可能有 9 个 Stage3 样本与 core24 Gemma 结果重叠；实际只有 4 个样本同时具备 support source 与 nearest control，可用于严格 source-control calibration。

偏差原因是：部分 Stage3 样本虽然在历史 Gemma raw rows 中出现，但缺 nearest control，或只存在 source 行，不能支撑 source-control 对照。

## 结论

当前可以写：

```text
Gemma3-PLT historical prefix-fix results overlap with Stage3 on a small subset,
and this subset preserves the expected source > nearest and source > random16 direction.
```

当前不能写：

```text
Gemma3-PLT has been fully rerun on Stage3 aligned24.
Gemma3-PLT provides a 24-sample matched baseline for Stage3.
```

下一步如果要把 Gemma baseline 做完整，需要对 Stage3 aligned24 剩余样本重新做 Gemma source tracing / nearest-control construction，生成真正 run-ready manifest。
