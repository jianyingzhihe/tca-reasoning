# Stage4-054 Qwen Evidence-Specific Targeted Intervention

## 目的

验证强 evidence-specific nodes 是否不只是“被证据区域调节”，还会因果支撑正确答案。

## 方法

- `clean_zeroing`: 在 clean run 中剪 source feature，观察 target logit/rank 是否受损。
- `mask_restore`: 在 answer/union/shifted/shuffled mask run 中恢复 source feature contribution。
- controls: same-position matched feature、same-feature random position、random-active feature。
- target control: correct target 与 wrong target 同时打分。

## 判据

- causal answer-support: clean source > controls 且 correct > wrong。
- evidence-linked route: answer/union restore > shifted/shuffled restore。
- rank/logit bridge: target logit 或 rank 至少一个方向稳定。

## 结果记录

报告 per-layer summary，同时保留 top raw case table，避免只看平均值丢掉强局部信号。

## Primary Smoke

2026-05-28 primary smoke 已完成，跨 L10-L17 共 12 个 main candidates，所有 layer 均产出 manifest/raw/run artifacts。

Analyzer decision:

```json
{
  "status": "primary_targeted_near_pass_pending_strict",
  "primary_source_gate": true,
  "primary_mask_gate": true,
  "layers_seen": [10, 11, 12, 13, 14, 15, 16, 17]
}
```

解释：smoke 支持 targeted validation 方向，不是最终结论；需要 primary full 与 strict confirmation。

## Main Full + Strict

2026-05-28 main full 已完成：primary 与 strict 各覆盖 L10-L17，每层 20 candidates，共 160 candidates / pack，raw rows 为 6400 / pack。

修正后的 analyzer 只使用 `mode=full`，不再把 smoke 小样本混入 gate。Full-only 结果：

```json
{
  "status": "qwen_evidence_sensitive_but_not_causal",
  "primary_source_gate": false,
  "strict_source_gate": false,
  "primary_mask_gate": false,
  "strict_mask_gate": false
}
```

Across-layer full aggregate 显示 answer-mask real-vs-shifted 有正向信号，但 source/control 与 correct/wrong 没站稳，因此不能写 causal/evidence-linked route。

完整 pool primary validation 已启动，tag=`all`，用于检查 main 子集是否漏掉稳定因果候选。
