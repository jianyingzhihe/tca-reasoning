# Stage4-020 Qwen Evidence-First Final Verdict

## 目的

在 Stage4-014/016 已证明 Qwen causal-screened cutter nodes 的基础上，汇总 Stage4-020 的 evidence-first 与 Adapter V3 结果，判断是否可以把 Qwen 从 `cutter-only` 推进到 `evidence-linked route support` 或 `adapter-v3 route support`。

## 输入

- `stage4_qwen_evidence_first_route_summary.csv`
- `stage4_qwen_evidence_first_route_specificity.csv`
- `stage4_qwen_evidence_first_route_decision.json`
- primary 与 strict sensitivity raw artifacts

## 输出

本文件给最终文字口径，但不覆盖 Gemma 主线结论。

## 方法

按四个 gate 判断：

- source/control specificity：source clean zeroing 是否强于 matched controls。
- evidence-mask specificity：answer/union mask 是否强于 shifted/shuffled。
- target specificity：correct target 是否强于 wrong target。
- behavior bridge：first-token/rank 或 sequence score 是否有方向性。

## 结果

已完成 primary 6 prompt-run smoke、primary full 与 strict full。smoke artifact 非空，管线通过；primary/strict full 都给出负向结果。

Primary full:

- Evidence-first candidates 足够多：`384` total / `192` main。
- Evidence-first clean zeroing 不强于 controls，CI 跨 0。
- Evidence-first restore 不稳定强于 shifted/shuffled，rank bridge 未成立。
- Adapter V3 candidates 较少但可运行：`58` total / `32` main。
- Adapter V3 没有通过 source/control、correct-vs-wrong、rank gates。

Strict full:

- Evidence-first candidates：`404` total / `202` main，覆盖 `101` prompt-runs。
- Evidence-first clean zeroing source-minus-controls：mean `0.0035`，CI `[-0.0025, 0.0121]`。
- Evidence-first restore gates 未通过 shifted/shuffled、correct-vs-wrong 与 rank criteria。
- Adapter V3 candidates：`62` total / `34` main，覆盖 `17` prompt-runs。
- Adapter V3 primary/strict 都未通过 source/control、correct-vs-wrong、rank gates。

## 预期与实际偏差

primary 与 strict 一致显示：当前 evidence-first route 和 Adapter V3 route 都没有把 Qwen 推到 Gemma-style mainline support。不是因为 candidates 太少，evidence-first candidates 很多；问题是 evidence-sensitive features 与 answer-causal cutter nodes 在当前设置下没有稳定重合。

## 结论

当前 verdict：`qwen_not_gemma_style_under_adapter` for Stage4-020 primary + strict full；总体 Qwen 仍保留 Stage4-016 的 `causal-screened PLT cutter nodes exist`。

可以下的直白结论：

- Qwen2.5-VL-PLT 确实有一些“剪掉会影响答案”的 PLT 节点，这是 Stage4-014/016 的强结论。
- 但是当我们先从“证据区域 mask 真正影响的 feature”出发，再问它们是不是答案支撑节点，结果没有闭合。
- Adapter V3 把自动路线偏向 evidence-sensitive nodes 后，也没有闭合成 Gemma 那种主线。
- 因此当前不能写 Qwen 复现了 Gemma-style source tracing；可以写当前 Qwen adapter 下没有证成 Gemma-style route replication。

当前不可写：

- `Qwen fully replicates Gemma-style source tracing`
- `Qwen has no evidence-sensitive mechanisms`
- `Qwen feature ids are semantic object nodes`
