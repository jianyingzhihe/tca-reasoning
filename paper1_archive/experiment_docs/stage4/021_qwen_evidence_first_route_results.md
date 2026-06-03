# Stage4-020 Qwen Evidence-First Route Results

## 目的

记录 Qwen2.5-VL-PLT evidence-first discovery 与 intervention 的实际结果，判断 Qwen 的 causal PLT cutter nodes 是否能进一步闭合到 evidence-region-linked support。

## 输入

- `paperpack72_primary_prompt_runs.csv`
- `paperpack72_strict_sensitivity_prompt_runs.csv`
- `answer_mask / union_mask / shifted_mask / shuffled_mask`
- Qwen2.5-VL-PLT layer 26

## 输出

待生成：

- `stage4_qwen_evidence_first_primary_smoke_candidates.csv`
- `stage4_qwen_evidence_first_primary_smoke_zeroing_raw.csv`
- `stage4_qwen_evidence_first_primary_smoke_group_raw.csv`
- full/strict 同名 artifact
- `stage4_qwen_evidence_first_route_summary.csv`
- `stage4_qwen_evidence_first_route_specificity.csv`
- `stage4_qwen_evidence_first_route_decision.json`

## 方法

先按 evidence mask sensitivity 找 feature/position，再用 source/control、wrong-target、shifted/shuffled restore 验证。主 endpoint 不是 decoded generation，而是 target logit/rank 与 evidence-mask specificity。

## 结果

2026-05-24 smoke 已完成，范围为 primary pack 的 6 个 prompt-runs。

- discovery 产生 `24` 个 evidence-first candidates，其中 main candidates `12` 个，覆盖 `6` 个 prompt-runs。
- clean/source-control smoke 产生 `240` 行 raw intervention/control rows。
- grouped restore smoke 产生非空 artifact。
- analyzer 状态：`smoke_completed_not_decisive`。

初步 gate 只作为调试信号，不作为科学结论：

- evidence-first clean zeroing source-minus-controls：`n=6`，mean `0.0`。
- evidence-first restore real-vs-shifted/shuffled 在 answer_mask top1 上有很小正向均值 `0.0208`，但 CI 触 0，positive fraction `0.1667`。
- rank effect 未出现稳定变化。

因此 smoke 证明的是：管线、schema、artifact fetch 可用；不证明也不否定 Qwen evidence-linked route。

2026-05-24 primary full 已启动：

- 命令：`python scripts/local/run_stage4_qwen_evidence_first_remote.py --mode full --packs primary --timeout-seconds 86400`
- 远端状态检查：discovery 进程运行中，GPU 使用约 `16GB`，磁盘剩余约 `56GB`。
- 当前未观察到资源不足；日志回传偏慢/缓冲，但远端进程处于运行状态。

2026-05-24 primary full 已完成：

- discovery：`384` candidates，覆盖 `96` 个 prompt-runs；main candidates `192`。
- clean/source-control validation：`7680` raw rows。
- grouped restore：`12160+` raw rows。
- analyzer 当前 full verdict：`qwen_not_gemma_style_under_adapter`。

关键 primary 指标：

- evidence-first clean zeroing source-minus-controls：`n=192`，mean `0.0031`，CI `[-0.0029, 0.0109]`，positive fraction `0.1198`。
- evidence-first clean zeroing correct-minus-wrong：mean `-0.0020`，CI `[-0.0088, 0.0055]`。
- best restore gate 是 `answer_mask/top1`，但 source-minus-controls、real-vs-shifted、real-vs-shuffled、correct-minus-wrong 的 CI 都跨 0。
- restore rank effect mean `-0.3125`，没有形成 rank bridge。

解释：evidence-first 确实能找到被真实 mask 更强影响的 feature/position，但这些 feature/position 在 primary full 中没有表现为稳定的 answer-support causal nodes。

## 预期与实际偏差

primary full 显示，至少在当前 Qwen2.5-VL-PLT layer 26、visual_answer positions 和当前 scoring 下，evidence-sensitive feature/position 与 answer-causal cutter nodes 没有稳定重合。这是对 `evidence-linked route support` 的负向证据，但不是对 Qwen 机制本身的负向结论。

## 结论

当前结论仍是：Qwen2.5-VL-PLT 已有 robust causal-screened PLT cutter nodes，但 Stage4-020 primary full 没有证成 evidence-first route。strict sensitivity 正在补跑。

2026-05-25 strict full 已完成：

- discovery：`404` candidates，覆盖 `101` 个 prompt-runs；main candidates `202`。
- clean/source-control validation：`8080` raw rows。
- grouped restore：`12800+` raw rows。
- strict analyzer 与 primary 同方向，未形成反转。

关键 strict 指标：

- evidence-first clean zeroing source-minus-controls：`n=202`，mean `0.0035`，CI `[-0.0025, 0.0121]`，positive fraction `0.1089`。
- evidence-first clean zeroing correct-minus-wrong：mean `-0.0015`，CI `[-0.0080, 0.0056]`。
- best restore gate 仍是 `answer_mask/top1`，但 source-minus-controls、real-vs-shifted、real-vs-shuffled、correct-minus-wrong 的 CI 都跨 0。
- restore rank effect mean `-0.2673`，没有 rank bridge。

最终解释：primary 与 strict 都显示，Qwen PLT 中确实能找到 evidence-sensitive features，但这些 features 在当前 layer 26 / visual_answer position / scoring 下没有稳定承担 answer-support causal role。因此 Stage4-020 不支持 `qwen_native_evidence_linked_supported`。
