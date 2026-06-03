# 目的

验证 all-layer candidate screen 找到的 Qwen PLT 候选是否真的因果支撑答案，并且是否特异于 evidence mask 和 correct target。

# 输入

- `stage4_qwen_all_layer_bounded_exhaustive_*_candidates.csv`
- image/mask assets
- Qwen2.5-VL-PLT

# 输出

- zeroing raw/summary
- grouped restore raw/summary
- per-layer specificity metrics
- failed-gate case table

# 方法

对每层 main candidates 运行：

- clean zeroing：剪掉 source feature 后 target logit/rank 是否受损。
- grouped restore：topK `1,4,8,16,32,64,128` 是否恢复 masked run 中的 target。
- controls：activation-matched、drop-matched、mask-insensitive、random-active。
- mask specificity：real answer/union mask 是否强于 shifted/shuffled。
- wrong-target specificity：correct target 是否强于 wrong target。

# 结果

Smoke validation 已完成并可分析。初步结果没有达到正式 positive gate：虽然个别 smoke 指标如 L7 `zeroing_correct_minus_wrong`、L0/L21 `restore_real_minus_shifted` 在 `n=6` 下为正，但样本数不足且没有同时闭合 source/control、mask specificity、wrong-target 与 behavior gates，因此只记为 smoke signal，不升级结论。

Primary full 已 detached 运行，完成后将重新分析 per-layer source/control、real-vs-shifted/shuffled、correct-vs-wrong 和 topK restore。

# 预期与实际偏差

如果 clean zeroing 成立但 restore/mask specificity 不成立，只能写 cutter-like answer-support node，不能写 evidence-linked route。

如果 grouped topK 成立但 single-node 不成立，结论应写 distributed PLT route，而不是 Gemma-style sparse route。

# 结论

待 `047` 汇总。
