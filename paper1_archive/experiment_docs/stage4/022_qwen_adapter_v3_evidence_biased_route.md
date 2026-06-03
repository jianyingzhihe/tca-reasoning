# Stage4-020 Qwen Adapter V3 Evidence-Biased Route Probe

## 目的

测试 Qwen automatic route extraction 是否可以通过 evidence-biased scoring 更接近闭合。Adapter V3 不复用 Gemma 的地图，而是对 Qwen 自己的 automatic source-tracing candidates 重新打分：

`adapter_v3_score = path_mass * evidence_specificity * target_attribution`

## 输入

- Qwen expanded source tracing V2 intervention CSV
- Stage4-020 evidence-first discovery manifest
- Qwen2.5-VL-PLT layer 26

## 输出

待生成：

- `stage4_qwen_adapter_v3_primary_*_manifest.csv`
- `stage4_qwen_adapter_v3_primary_*_zeroing_raw.csv`
- `stage4_qwen_adapter_v3_primary_*_group_raw.csv`
- strict sensitivity 对应 artifact

## 方法

1. 从 automatic source tracing rows 中取 Qwen 自己的 feature/position。
2. 用 evidence-first discovery 结果为每个 node 补 evidence specificity 与 target attribution。
3. 选择 top evidence-biased route candidates。
4. 跑 clean zeroing、source/control、wrong-target、real-vs-shifted/shuffled restore。

## 结果

2026-05-24 primary smoke 已完成。

- Adapter V3 从 6 个 prompt-run smoke 的 evidence discovery 中只形成 `4` 个 candidates，覆盖 `1` 个 prompt-run。
- clean zeroing source-minus-controls：`n=2`，mean `0.0208`，CI 触 0。
- restore real-vs-shifted/shuffled 在单个 row 上为正，但样本数不足，且 source-minus-controls 与 rank gate 未通过。
- analyzer 已改为 `smoke_completed_not_decisive`，避免把 smoke 误读为 final negative verdict。

## 预期与实际偏差

smoke 的 Adapter V3 candidates 偏少，说明 automatic source-tracing nodes 与 evidence-first exact/fallback match 可能较稀疏。full primary 需要检查这是 smoke 小样本问题，还是当前 adapter scoring 的系统性限制。

## 结论

当前不下 Adapter V3 科学结论。下一步跑 primary full；若 full 仍 candidates 稀疏且 gates 不通过，才写 `qwen_not_gemma_style_under_adapter`。

2026-05-24 primary full 已完成。

- Adapter V3 manifest：`58` candidates，覆盖 `16` 个 prompt-runs；main candidates `32`。
- clean zeroing source-minus-controls：`n=32`，mean `-0.0234`，CI `[-0.0892, 0.0286]`。
- clean zeroing correct-minus-wrong：mean `-0.0098`，CI `[-0.0684, 0.0508]`。
- best restore gate 是 `union_mask/top1`，其中 real-vs-shifted mean `0.0215` 且 CI 下界 `0.0039`，但 source-minus-controls、correct-minus-wrong、rank gate 均未通过。

Primary full 结论：Adapter V3 没有把 automatic route extraction 闭合成 Gemma-style route support。这个结论仍限定为“当前 Qwen Adapter V3 + layer 26 + paperpack primary”。

2026-05-25 strict full 已完成。

- Adapter V3 strict manifest：`62` candidates，覆盖 `17` 个 prompt-runs；main candidates `34`。
- clean zeroing source-minus-controls：`n=34`，mean `-0.0196`，CI `[-0.0833, 0.0319]`。
- clean zeroing correct-minus-wrong：mean `-0.0092`，CI `[-0.0662, 0.0478]`。
- best restore gate 仍是 `union_mask/top1`，其中 restore source-minus-controls 和 real-vs-shifted 有小的正向均值，但 positive fraction 低，correct-vs-wrong 与 rank gate 未通过。

Primary 与 strict 共同结论：Adapter V3 evidence-biased scoring 不能把 Qwen automatic source-tracing nodes 推成 Gemma-style route support。允许写 `qwen_not_gemma_style_under_adapter`，但必须限定为当前 adapter 与当前测试设置。
