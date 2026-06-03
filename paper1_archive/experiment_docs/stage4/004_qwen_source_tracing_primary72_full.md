# Stage4-004 Qwen Source-Tracing Primary72 Full

## 目的

在 paperpack72 primary 上检验 Qwen2.5-VL-PLT 是否能达到 Gemma-style source tracing replication 的可判定门槛。

## 输入

- paperpack72 primary 72 samples × 2 prompts。
- Qwen2.5-VL base + Qwen2.5-VL-PLT layer 26。

## 输出

Full 输出放在 `doc/experiments/stage4/cross_model/`，文件名包含 `stage4_qwen_source_tracing_primary_full_*`。

## 方法

Smoke 通过后运行 `run_stage4_qwen_source_tracing_remote.py --pack primary --mode full`，再运行 `analyze_stage4_qwen_source_tracing.py --pack primary --mode full`。

## 结果

Primary72 full 已完成。

- meta rows: A=72, B=72。
- graph ok: A=70, B=70。
- prompt-runs ok: 140/144。
- graph success rate: 97.22%。
- valid matched samples: 70。
- compare rows: 70。
- node rows: 968。
- feature node rows: 420。
- edge rows: 840。
- intervention rows: 560。
- best zeroing mode: `subtract`。
- mean `delta_target_logit`: `-0.01755`。
- overall negative `delta_target_logit` fraction: `10.18%`。
- best-mode negative `delta_target_logit` fraction: `12.14%`。
- rank hurt fraction: `6.25%`。

两个超长序列样本因 `n_pos > 512` 被安全阈值跳过；这不是资源不足，且不影响 120/144 prompt-run 门槛。

## 结论

Primary full verdict: `qwen_source_tracing_not_supported` under the current Qwen answer-aligned PLT graph adapter.

解释边界：

- 可以写：Qwen2.5-VL-PLT 在 paperpack primary 上有 Stage3 approximate feature/source-control support，但当前 true source-tracing adapter 没有复现 Gemma-style source tracing。
- 不能写：Qwen 没有 evidence-region-sensitive mechanism。
- 不能写：Qwen 完整复现 Gemma source tracing。
- 不能用这个结果推翻 Stage3 的 hidden / approximate source-control evidence。

## 2026-05-24 Audit Note

Stage4-011 found that this result should be treated as provisional: the original adapter selected features from `outputs.hidden_states[args.layer]` while the intervention hook was applied to `language_model.layers[args.layer]`. The code has been fixed to capture the exact layer output with a forward hook. Primary72 should be rerun before using this file as a final Qwen negative source-tracing result.
