# Stage4-005 Qwen Source-Tracing Strict72 Sensitivity

## 目的

在 strict72 sensitivity pack 上验证 Qwen source-tracing primary 结果是否依赖 moderate/manual-review 样本。

## 输入

- paperpack72 strict sensitivity 72 samples × 2 prompts。
- Qwen2.5-VL base + Qwen2.5-VL-PLT layer 26。

## 输出

Strict 输出放在 `doc/experiments/stage4/cross_model/`，文件名包含 `stage4_qwen_source_tracing_strict_full_*`。

## 方法

Primary full 完成后运行 `run_stage4_qwen_source_tracing_remote.py --pack strict --mode full`，再分析。

## 结果

Strict72 sensitivity full 已完成。

- meta rows: A=72, B=72。
- graph ok: A=70, B=70。
- prompt-runs ok: 140/144。
- graph success rate: 97.22%。
- valid matched samples: 70。
- compare rows: 70。
- node rows: 970。
- feature node rows: 420。
- edge rows: 840。
- intervention rows: 560。
- best zeroing mode: `subtract`。
- mean `delta_target_logit`: `-0.00650`。
- overall negative `delta_target_logit` fraction: `8.75%`。
- best-mode negative `delta_target_logit` fraction: `11.07%`。
- rank hurt fraction: `5.71%`。

## 结论

Strict verdict: `qwen_source_tracing_not_supported` under the current Qwen answer-aligned PLT graph adapter.

Primary 和 strict 方向一致：adapter 能生成 graph/compare，但 direct source-traced feature intervention 不能稳定损伤 target。因此 Qwen full Gemma-style source tracing replication 在当前 adapter 下不成立。

## 2026-05-24 Audit Note

Stage4-011 found that this result should be treated as provisional: the original adapter selected features from `outputs.hidden_states[args.layer]` while the intervention hook was applied to `language_model.layers[args.layer]`. The code has been fixed to capture the exact layer output with a forward hook. Strict72 should be rerun before using this file as a final Qwen negative source-tracing result.
