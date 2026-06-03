# 029 Qwen2.5-VL-PLT Strict72 Sensitivity Results

## 目的

检查 `paperpack72_primary` 的 Qwen2.5-VL-PLT 正结果是否依赖 5 个 `manual_review_image_dependence` 样本。strict pack 排除这些 moderate/manual-review 样本，并用 replacement 样本补齐到 72。

## 输入

```text
paperpack: paperpack72_strict_sensitivity
samples: 72
prompt-runs: 144
base model: Qwen/Qwen2.5-VL-7B-Instruct
transcoder: KokosDev/qwen2p5vl-7b-plt
layer: 26
primary position group: top_hidden_delta_plus_answer_adjacent
```

## 输出

```text
stage3_qwen2p5vl_plt_feature_union_strict_full.csv/json
stage3_qwen2p5vl_plt_source_control_strict_full.csv/json
stage3_qwen2p5vl_plt_strict_full_feature_summary.csv
stage3_qwen2p5vl_plt_strict_full_feature_specificity.csv
stage3_qwen2p5vl_plt_strict_full_source_control_summary.csv
stage3_qwen2p5vl_plt_strict_full_source_control_specificity.csv
stage3_qwen2p5vl_plt_strict_full_case_table.csv
stage3_qwen2p5vl_plt_strict_full_decision.json
```

## 方法

完全复用 primary72 full 的 Qwen2.5-VL-PLT runner 和分析脚本，只替换 manifest 为 `paperpack72_strict_sensitivity`。这样 primary 与 strict 的差异主要来自样本包，而不是实验配置。

## 结果

自动 decision：

```text
qwen_plt_strict_support
```

运行规模：

| item | value |
|---|---:|
| remote exit status | 0 |
| feature rows | 4320 |
| feature prompt-runs | 144 |
| source-control rows | 1446 |
| source-control prompt-runs | 128 |
| usable source-control pairs | 241 |

Feature bridge 主位置组：

| comparison | direction | strict mean logit diff | 95% CI |
|---|---|---:|---|
| evidence > activation-matched | restore | +0.1368 | [+0.0917, +0.1885] |
| evidence > activation-matched | corrupt | +0.1400 | not copied here; see CSV |
| evidence > drop-matched | restore | +0.0563 | [+0.0260, +0.0903] |
| evidence > mask-insensitive matched | restore | +0.0515 | [+0.0152, +0.0924] |
| evidence > random-active | restore | +0.0506 | [+0.0208, +0.0846] |

Source-control real-mask probe：

| mask | intervention | source-control mean logit | 95% CI | positive frac |
|---|---|---:|---|---:|
| answer_mask | restore | +0.0358 | [+0.0122, +0.0625] | 0.422 |
| answer_mask | zeroing | +0.1849 | [+0.1578, +0.2149] | 0.926 |
| union_mask | restore | +0.0221 | [-0.0019, +0.0496] | 0.358 |
| union_mask | zeroing | +0.1988 | [+0.1557, +0.2561] | 0.925 |

Legacy real-vs-shifted restore：

| mask | source real-shifted mean logit | 95% CI |
|---|---:|---|
| answer_mask | +0.0561 | [+0.0302, +0.0852] |
| union_mask | +0.0499 | [+0.0241, +0.0790] |

## 预期与实际偏差

预期 strict pack 可能略弱，因为替换样本可能更难或标注质量不完全等同。实际结果没有变弱：feature restore、answer-mask source restore、legacy real-vs-shifted restore 都与 primary 同方向且略强或相近。

shifted + true shuffled 双控制已在 `030_qwen2p5vl_plt_shifted_shuffled_controls.md` 中补跑。补跑后最稳的是 answer-mask real > shifted；true shuffled 和 union mask 方向为正但更弱。

## 结论

strict sensitivity 支持 primary 结果：

```text
Qwen2.5-VL-PLT feature/source-control support is not driven by the five manual-review/moderate image-dependence samples.
```

因此 Qwen2.5-VL-PLT 侧现在可以写成：

```text
Qwen2.5-VL-PLT shows primary and strict-sensitivity approximate feature/source-control support on the paperpack72 heldout annotation pool.
```

但仍不能写成：

```text
Qwen2.5-VL fully replicates Gemma-style source tracing.
Qwen2.5-VL has decoded generation-level causal restoration.
```
