# Stage4-016 Qwen Toward Gemma-Mainline Replication Overnight Run Plan

## 目的

当前 Qwen2.5-VL-PLT 已经支持 `causal-screened PLT cutter nodes exist`，但还不能写成 Gemma-style 主线复现。差距不是只有样本数，而是 Gemma 主线的五个 gate 还没有同时闭合：

- automatic source tracing route
- source/control specificity
- evidence-mask specificity
- wrong-target specificity
- first-token/rank or sequence behavior bridge

本轮目标是用 `paperpack72_primary/strict` 最大化推进 Qwen，不新增标注。

## 输入

- Model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Asset: `KokosDev/qwen2p5vl-7b-plt`
- Main layer: `26`
- Sensitivity layers: `22`, `24`
- Main pack: `paperpack72_primary`
- Confirmation pack: `paperpack72_strict_sensitivity`

## 方法

1. Expanded automatic source tracing V2:
   - candidate pool `8192`
   - graph selected feature nodes `256`
   - compare topk per node `8`
   - intervention top features per sample `32`
   - position filter `visual_answer`

2. Expanded cutter discovery:
   - discovery only from primary V2 rows
   - strict rows only for confirmation
   - `clean rank <= 10`
   - `delta logit <= -0.15` or `rank hurt >= 1`
   - each sample/prompt keeps top2 main candidates and top16 pool candidates

3. Evidence-link V2:
   - activation-drop mediation under `answer/union/shifted/shuffled`
   - grouped restore `top1/top4/top8/top16`
   - source group compared to same-position matched feature, same-feature other-position, and random-active controls

4. Behavior bridge:
   - only after evidence-link rows pass
   - first-token/rank is primary
   - answer sequence score is secondary
   - decoded greedy is exploratory only

## 输出

- `cross_model/stage4_qwen_source_tracing_*_expanded_v2_*`
- `cross_model/stage4_qwen_expanded_cutter_candidate_manifest.csv`
- `cross_model/stage4_qwen_evidence_linked_cutter_v2_raw.csv`
- `cross_model/stage4_qwen_mainline_v2_summary.csv`
- `cross_model/stage4_qwen_mainline_v2_specificity.csv`
- `cross_model/stage4_qwen_mainline_v2_decision.json`

## 判据

- `qwen_gemma_style_supported`: automatic source-traced nodes pass primary/strict source-control, evidence-mask, wrong-target, and behavior gates.
- `qwen_native_evidence_linked_supported`: automatic tracing is incomplete, but Qwen-native candidates pass evidence-link controls.
- `qwen_cutter_only_supported`: expanded clean zeroing remains positive but evidence link fails.
- `qwen_not_gemma_style_under_adapter`: expanded automatic source tracing runs but route/evidence specificity fails.
- `blocked`: engineering/resource/schema failure.

## 当前结论边界

即使本轮成功，也要区分：

- Gemma-style automatic source tracing replication
- Qwen-native evidence-linked cutter support
- Qwen-native causal cutter-only support

不能只因为候选数增加就升级 claim。

## 执行状态

Completed.

Final label after Stage4-017/018:

`qwen_cutter_only_supported`

This means the expanded run strengthened Qwen-native cutter evidence, but did not close Gemma-style automatic tracing or evidence-mask-linked route gates.
