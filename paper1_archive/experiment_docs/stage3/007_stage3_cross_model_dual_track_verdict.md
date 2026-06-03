# 007 Stage3 Cross-Model Dual-Track Verdict

> 论文级当前有效计划已更新为 `016_stage3_plt_first_paper_grade_run_plan.md`。本文件记录截至 Stage3-15 的 pilot/interim verdict；paper-grade final verdict 将在 `025_stage3_paper_grade_final_verdict.md` 中更新。

## 目的

输出 Stage3 当前能下的保守结论，并明确哪些结论还不能下。

## 输入

```text
Stage2 Gemma 主线结果
Stage2 Qwen/LLaVA CLT 结果
Stage3 asset preflight
Stage3 Qwen CLT vs PLT smoke
Stage3 CLT auxiliary smoke
Stage3 full aligned24/source-control 结果
Stage3 Gemma3-PLT overlap calibration
Stage3 first-token/rank behavior bridge
Stage3 Qwen generation bridge v2 sequence-score analysis
Stage3 paperpack72 candidate pool and PLT-first replan
```

## 当前证据矩阵

| 模型 | 资产 | 当前状态 | 可支持口径 |
|---|---|---|---|
| Gemma3 | PLT | 历史主线完整；Stage3 overlap calibration 只有 4 个严格可用样本 | full Gemma mainline evidence + limited Stage3 calibration |
| Qwen2.5-VL | PLT | full feature/source-control positive；first-token/rank partial positive；top1 有极弱 partial sequence hint；decoded restore 未成立 | PLT-aligned support |
| Qwen2.5-VL | CLT | full feature/source-control strong positive；first-token/rank supported；restore-only sequence score positive；decoded restore 未成立 | CLT auxiliary positive |
| LLaVA-1.5 | CLT | feature weak，zeroing small positive，restore weak；first-token/rank partial | hidden-state-only / weak CLT auxiliary |
| Qwen35 | PLT | custom `.pt`，缺 L1，base 有 Qwen3VLProcessor 但当前 transformers 不支持 config | partial/high-risk feasibility only |

## 当前能下的结论

可以写：

```text
Gemma3-PLT 的 Stage3 overlap 子集仍保留 source > nearest 与 source > random16 方向；
但这只是 overlap calibration，不是完整 Stage3 aligned24 Gemma rerun。
```

可以写：

```text
Qwen2.5-VL 的 evidence-sensitive feature bridge 不是 CLT-only；
在 PLT full aligned24 中也出现同方向正信号，但效应明显小于 CLT。
Qwen2.5-VL-PLT 的 approximate source-control zeroing 与 restoration 也为正。
```

可以写：

```text
Qwen2.5-VL 的 source/control 差异已经桥接到目标答案首 token logit/rank；
CLT 版本最强，PLT 版本较弱但仍有正向 first-token/rank 支持。
```

可以写：

```text
Qwen2.5-VL 的 decoded generation smoke 没有稳定 source_restore_to_clean；
因此行为桥接当前应停在 first-token/rank 层。
```

可以写：

```text
Qwen2.5-VL-CLT 的 multi-feature source restore 可以比 matched controls
更稳定提高 target answer sequence logprob；
但 clean->masked corruption 没有同步成立，所以这是 restore-only sequence support，
不是 supported sequence-level causal bridge。
```

可以写：

```text
Qwen2.5-VL-PLT 只有 top1 的 very weak partial sequence bridge hint；
CI 跨 0，不能作为稳定行为桥接主证据。
```

可以写：

```text
LLaVA-CLT 目前不能支持稳定 restoration-level feature/source route replication；
它有小的 zeroing 正信号，但更适合作为 hidden-state bridge 支持、feature localization 弱或异质的辅助证据。
```

可以写：

```text
Qwen35-PLT 暂不能进入 VLM evidence-region 主线，因为资产格式、缺 L1、当前 transformers/Qwen3.5 loader 都有阻塞风险；但不能说它不是 VLM，因为 base repo 暴露了 Qwen3VLProcessor。
```

## 当前不能下的结论

不能写：

```text
Gemma3-PLT 已在 Stage3 aligned24 上完成 24 样本全量重跑。
```

不能写：

```text
Qwen2.5-VL 已完成 generation-level causal bridge。
```

不能写：

```text
Qwen2.5-VL 已完成 supported bidirectional sequence-level causal bridge。
```

不能写：

```text
Qwen/LLaVA 已完整复现 Gemma-style source-control route。
```

不能写：

```text
LLaVA 没有跨模态特征。
```

不能写：

```text
PLT 一定比 CLT 更适合，或 CLT 结果无效。
```

## 下一步判据

论文级下一步不再继续扩大旧 aligned24，而是先完成：

```text
paperpack72 双人标注 + 仲裁
Gemma3-PLT paperpack rerun
Qwen2.5-VL-PLT paperpack run
Qwen35-PLT feasibility verdict
PLT-only verdict
```

只有 `022_plt_only_verdict.md` 完成后，才进入 Qwen-CLT / LLaVA-CLT paperpack 辅助线。

基于当前 full aligned24，可以升级为：

```text
Gemma and Qwen2.5-VL show PLT-aligned evidence-region-sensitive feature/source-control support.
```

但必须加限定：

```text
This is approximate source-control support, not full Gemma-style source tracing replication.
```

最终推荐口径：

```text
Gemma 主线已经有完整 source tracing/intervention/control 证据；Stage3 overlap calibration 与历史方向一致，但不是全量重跑。
Qwen2.5-VL 在 PLT-aligned Stage3 中出现同方向 feature/source-control 支持，CLT 辅助线更强；
Qwen2.5-VL 的 source/control 差异还能桥接到目标答案首 token logit/rank；
CLT 上有 restore-only answer sequence score support，PLT 上有很弱 top1 partial sequence hint，但 decoded generation restore 尚未成立；
因此该现象不是 Gemma-only，也不是 Qwen CLT-only。
LLaVA-CLT 仅支持弱 feature/zeroing 与 hidden-state 辅助，不支持稳定 feature-level route replication。
在没有真正 Qwen/LLaVA ReplacementModel source tracing adapter 前，不能写完整 Gemma-style route 跨模型复现。
```
