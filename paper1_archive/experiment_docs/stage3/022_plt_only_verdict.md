# 022 PLT-Only Verdict

## 目的

在进入 CLT 辅助线之前，只基于 PLT 资产给出 Stage3 当前主线判定。

本 verdict 只合并：

```text
Gemma3-PLT: tianhux2/gemma3-4b-it-plt
Qwen2.5-VL-PLT: KokosDev/qwen2p5vl-7b-plt
Qwen35-PLT: KokosDev/qwen35-4b-plt feasibility record
```

它不使用 Qwen-CLT 或 LLaVA-CLT 结果，也不把 Qwen approximate probe 写成完整 Gemma-style source tracing。

## 输入

```text
019_gemma3_plt_paperpack_results.md
020_qwen2p5vl_plt_paperpack_results.md
021_qwen35_plt_feasibility_verdict.md
033_stage3_gemma_primary72_source_tracing_full.md
034_stage3_gemma_strict72_source_tracing_sensitivity.md
```

## 结果矩阵

| model | PLT asset | paperpack result | strongest supported level | boundary |
|---|---|---|---|---|
| Gemma3 | `tianhux2/gemma3-4b-it-plt` | primary72 full passed; strict72 graph/compare sensitivity passed | full Gemma source-tracing pipeline on primary; strict source-tracing graph/compare sensitivity | strict intervention smoke is engineering-blocked by model-load `SIGKILL(137)` |
| Qwen2.5-VL | `KokosDev/qwen2p5vl-7b-plt` | primary72 and strict72 support approximate feature/source-control evidence | PLT approximate feature/source-control support | no Qwen ReplacementModel/source-tracing adapter; no decoded generation bridge claim |
| Qwen35 | `KokosDev/qwen35-4b-plt` | pending/high-risk feasibility record | not part of main PLT claim | blocked/pending is not a negative mechanism result |

## Gemma3-PLT Summary

Primary72:

```text
status: primary_full_passed
valid samples: 71 / 72
graph A/B: 71 / 71
sample compare rows: 71
node rows: 4126
edge rows: 7071
intervention rows: 127
```

Strict72 sensitivity:

```text
status: strict_full_graph_compare_passed_intervention_blocked
valid samples: 71 / 72
graph A/B: 71 / 71
sample compare rows: 71
node rows: 4095
edge rows: 7003
intervention rows: 0
```

Interpretation:

```text
Gemma3-PLT provides the full source-tracing baseline on primary72.
Strict72 confirms the graph/compare layer after removing manual-review samples.
Strict intervention smoke is a resource/engineering block, not a mechanism negative.
```

## Qwen2.5-VL-PLT Summary

Primary72:

```text
status: qwen_plt_primary_support
feature rows: 4320
feature prompt-runs: 144
source-control rows: 1452
source prompt-runs: 128
source usable pairs: 242
primary feature specificity positives: 8 / 8
source-control positives: 4 / 4
real-vs-shuffled positives: 4 / 8
```

Strict72:

```text
status: qwen_plt_strict_support
feature rows: 4320
feature prompt-runs: 144
source-control rows: 1446
source prompt-runs: 128
source usable pairs: 241
primary feature specificity positives: 8 / 8
source-control positives: 4 / 4
real-vs-shuffled positives: 4 / 8
```

Interpretation:

```text
Qwen2.5-VL-PLT supports evidence-region-sensitive approximate feature/source-control probes on the same paperpack family.
This is cross-model PLT-aligned support, but not Gemma-style source tracing replication.
```

## Verdict

当前可以写：

```text
Gemma3-PLT and Qwen2.5-VL-PLT show PLT-aligned evidence-region-sensitive answer-support evidence on the independently annotated paperpack.
```

更精确的论文口径应写成：

```text
On localized, image-dependent VQA cases with annotated evidence regions, Gemma3-PLT provides a full source-tracing baseline on paperpack72 primary and strict graph/compare sensitivity, while Qwen2.5-VL-PLT provides independent approximate feature/source-control support on primary and strict packs.
```

当前不能写：

```text
Qwen2.5-VL fully replicates Gemma-style source tracing.
All VLMs have the same source-control route.
Qwen35-PLT is a negative result.
D_visual_only is better than B_direct.
CLT results strengthen or weaken this PLT-only verdict.
```

## 下一步

PLT-first 主线已经可以进入 CLT-second 阶段。下一轮可以按计划启动：

```text
1. Qwen2.5-VL-CLT paperpack completion
2. Qwen PLT-vs-CLT same-sample comparison
3. LLaVA-CLT auxiliary / heterogeneity diagnostic
```

如果要进一步加固 PLT-only claim，最有价值的补充不是重跑 Qwen，而是修复 Gemma strict intervention 的当前 model-load `SIGKILL(137)`，或实现 Qwen ReplacementModel/source-tracing adapter。
