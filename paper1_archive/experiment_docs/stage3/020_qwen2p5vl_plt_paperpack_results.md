# 020 Qwen2.5-VL-PLT Paperpack Results

## 目的

在独立 `paperpack72_primary` 上检验 Qwen2.5-VL-PLT 是否存在和 Gemma 主线同方向的 evidence-region-sensitive feature/source-control 支持信号。

本实验只支持 Qwen 的 **approximate feature/source-control support**，不等同于 Gemma-style source tracing，因为 Qwen 还没有完整 `ReplacementModel` / source-tracing adapter。

## 当前状态

```text
primary72_full_done
strict72_sensitivity_done
gemma3_plt_primary_done_strict_graph_compare_done
plt_only_verdict_written
```

## 输入

```text
manifest: doc/experiments/stage3/paperpack72/paperpack72_primary_manifest.csv
prompt-runs: doc/experiments/stage3/paperpack72/paperpack72_primary_prompt_runs.csv
base: Qwen/Qwen2.5-VL-7B-Instruct
transcoder: KokosDev/qwen2p5vl-7b-plt
layer: 26
mask condition: union_mask for feature bridge; answer_mask + union_mask for source-control probe
prompts: B_direct, D_visual_only
```

## 输出

Raw artifacts:

```text
doc/experiments/stage3/cross_model/stage3_paperpack_asset_preflight_primary_full.csv/json
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_plt_feature_union_primary_full.csv/json
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_plt_source_control_primary_full.csv/json
```

Analysis artifacts:

```text
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_plt_primary_full_feature_summary.csv
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_plt_primary_full_feature_specificity.csv
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_plt_primary_full_source_control_summary.csv
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_plt_primary_full_source_control_specificity.csv
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_plt_primary_full_case_table.csv
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_plt_primary_full_key_slice_summary.csv
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_plt_primary_full_decision.json
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_plt_controls_full_specificity_summary.csv
```

## 方法

1. 远端 preflight 检查 PLT/CLT 资产格式、层数、hook metadata。
2. 在 72 个样本、144 个 prompt-runs 上运行 Qwen2.5-VL-PLT attribution-weighted multi-feature bridge。
3. 在同一 pack 上运行 approximate source-control probe，比较 source feature 与 matched control feature。
4. 本地分析 paired evidence-vs-control、source-vs-control、legacy real-vs-shifted、correct-vs-wrong，以及 manual-review slice；随后在 `030` 补跑 true shifted/shuffled controls。

## 结果

运行成功：

| item | value |
|---|---:|
| remote exit status | 0 |
| feature rows | 4320 |
| feature prompt-runs | 144 |
| source-control rows | 1452 |
| source-control prompt-runs | 128 |
| source-control usable pairs | 242 |

Feature bridge 主位置组 `top_hidden_delta_plus_answer_adjacent`：

| comparison | direction | mean logit diff | 95% CI |
|---|---|---:|---|
| evidence > activation-matched | restore | +0.1297 | [+0.0831, +0.1785] |
| evidence > activation-matched | corrupt | +0.1311 | [+0.0790, +0.1845] |
| evidence > drop-matched | restore | +0.0507 | [+0.0194, +0.0839] |
| evidence > drop-matched | corrupt | +0.0106 | [-0.0163, +0.0382] |
| evidence > mask-insensitive matched | restore | +0.0484 | [+0.0110, +0.0903] |
| evidence > random-active | restore | +0.0474 | [+0.0162, +0.0801] |

Source-control real-mask probe：

| mask | intervention | source-control mean logit | 95% CI | positive frac |
|---|---|---:|---|---:|
| answer_mask | restore | +0.0335 | [+0.0095, +0.0617] | 0.410 |
| answer_mask | zeroing | +0.1857 | [+0.1588, +0.2152] | 0.934 |
| union_mask | restore | +0.0195 | [-0.0053, +0.0477] | 0.342 |
| union_mask | zeroing | +0.2048 | [+0.1616, +0.2613] | 0.933 |

Legacy real-vs-shifted restore：此前历史变量名为 `mask_shuffled`，但脚本实际使用 shifted mask。

| role | mask | intervention | real-shifted mean logit | 95% CI |
|---|---|---|---:|---|
| source | answer_mask | restore | +0.0516 | [+0.0248, +0.0809] |
| source | union_mask | restore | +0.0501 | [+0.0246, +0.0775] |
| matched_control | answer_mask | restore | +0.0143 | [+0.0015, +0.0289] |
| matched_control | union_mask | restore | +0.0152 | [+0.0020, +0.0301] |

Slice 检查：

| slice | metric | mean |
|---|---|---:|
| all_primary | feature evidence minus mean controls, restore | +0.0690 |
| strong_only excluding manual-review | feature evidence minus mean controls, restore | +0.0763 |
| manual_review_only | feature evidence minus mean controls, restore | -0.0279 |
| all_primary | source-control answer zeroing | +0.1857 |
| strong_only excluding manual-review | source-control answer zeroing | +0.1875 |
| manual_review_only | source-control answer zeroing | +0.1632 |

## 预期与实际偏差

预期希望同时覆盖 `shifted_mask` 和 `mask_shuffled`。已在后续 source-control-only controls 补跑中完成。结果显示 answer-mask 的 `real > shifted` 在 primary 和 strict 中稳定成立；true shuffled 方向为正但 CI 贴近/略跨 0，union mask 控制更弱。

预期希望 source restore 和 source zeroing 都强成立。实际结果是 zeroing 很强，restore 较弱：answer mask restore CI 不跨 0，union mask restore CI 接近但略跨 0。

预期希望 manual-review 样本不破坏主结论。实际排除 manual-review 后主均值更强，说明这 5 个样本保留为 moderate slice 是合理的；最强口径应优先使用 strong-only sensitivity。

## 结论

当前可以写：

```text
On the independently annotated paperpack72 primary set, Qwen2.5-VL-PLT shows evidence-region-sensitive feature/source-control support at the approximate probe level.
```

当前不能写：

```text
Qwen2.5-VL fully replicates Gemma-style source tracing.
Qwen2.5-VL has a decoded generation-level causal bridge.
```

strict sensitivity 与 shifted/shuffled controls 已完成，方向与 primary 一致但 spatial-control 强度集中在 answer-mask。Gemma3-PLT primary full 与 strict graph/compare sensitivity 也已完成，因此 Qwen2.5-VL-PLT 结果已经进入 `022 PLT-only verdict`，但仍保持 approximate source-control 口径。
