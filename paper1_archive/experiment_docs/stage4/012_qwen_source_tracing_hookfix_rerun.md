# Stage4-012 Qwen Hook-Aligned Source-Tracing Rerun

## 目的

复查 Stage4-011 指出的关键漏洞：旧 Qwen adapter 用 `outputs.hidden_states[layer]` 选 feature，但 intervention hook 挂在 `language_model.layers[layer]` 输出上，可能存在 layer-output mismatch。本实验把 Qwen feature selection 改为直接 forward hook 捕获同一个 layer 输出，然后重跑 smoke、primary72、strict72，并加入 topK 与层位 sensitivity。

## 输入

- 模型：`Qwen/Qwen2.5-VL-7B-Instruct`
- PLT：`KokosDev/qwen2p5vl-7b-plt`
- 数据：paperpack72 primary 与 strict sensitivity
- 主层：layer 26
- 额外层位 smoke：layer 22、24、27
- 主干预：top2 selected feature nodes
- topK sensitivity：top8 selected feature nodes

## 输出

主要 artifact 位于 `doc/experiments/stage4/cross_model/`：

- `stage4_qwen_source_tracing_primary_smoke_hookfix_*`
- `stage4_qwen_source_tracing_primary_full_hookfix_*`
- `stage4_qwen_source_tracing_strict_full_hookfix_*`
- `stage4_qwen_source_tracing_primary_full_hookfix_top8_*`
- `stage4_qwen_source_tracing_strict_full_hookfix_top8_*`
- `stage4_qwen_source_tracing_primary_smoke_hookfix_L22_*`
- `stage4_qwen_source_tracing_primary_smoke_hookfix_L24_*`
- `stage4_qwen_source_tracing_primary_smoke_hookfix_L27_*`
- `stage4_qwen_source_tracing_hookfix_summary.csv`

## 方法

代码修复：

- `run_qwen_answer_aligned_attribute.py` 现在通过 forward hook 捕获 `language_model.layers[layer]` 的实际输出。
- metadata 新增 `hidden_capture_source` 与 `hidden_states_tuple_len`。
- `run_stage4_qwen_source_tracing_remote.py` 新增 `--tag`，避免覆盖旧结果。
- runner 新增 `--layer`、`--compare-topk-per-node`、`--top-features-per-sample`、`--max-feature-nodes`，支持 sensitivity。

执行顺序：

1. `primary smoke --tag hookfix --layer 26`
2. `primary full --tag hookfix --layer 26`
3. `strict full --tag hookfix --layer 26`
4. `primary smoke --tag hookfix_L22 --layer 22`
5. `primary smoke --tag hookfix_L24 --layer 24`
6. `primary smoke --tag hookfix_L27 --layer 27`
7. `primary full --tag hookfix_top8 --layer 26 --top-features-per-sample 8 --compare-topk-per-node 8`
8. `strict full --tag hookfix_top8 --layer 26 --top-features-per-sample 8 --compare-topk-per-node 8`

## 结果

主 full 结果：

| run | graph success | prompt-runs | intervention rows | best negative target-logit fraction | verdict |
|---|---:|---:|---:|---:|---|
| primary top2 hookfix | 97.22% | 140/144 | 560 | 11.79% | `qwen_source_tracing_not_supported` |
| strict top2 hookfix | 97.22% | 140/144 | 560 | 11.07% | `qwen_source_tracing_not_supported` |
| primary top8 hookfix | 97.22% | 140/144 | 2240 | 9.64% | `qwen_source_tracing_not_supported` |
| strict top8 hookfix | 97.22% | 140/144 | 2240 | 8.48% | `qwen_source_tracing_not_supported` |

Layer smoke：

| layer | graph status | best negative target-logit fraction | verdict |
|---:|---|---:|---|
| 22 | 6/6 prompt-runs ok | 16.67% | direction not supported |
| 24 | 6/6 prompt-runs ok | 8.33% | direction not supported |
| 26 | 6/6 prompt-runs ok | 0.00% | direction not supported |
| 27 | blocked | n/a | `index 27 is out of range` for current PLT asset/loader |

Metadata check：

- primary/strict layer26 successful rows all used `language_model.layers[26]_forward_hook_output`。
- L22/L24 successful smoke rows used their corresponding forward-hook outputs。
- Two long-sequence samples still exceeded `--max-n-pos 512`; primary/strict retained 70/72 valid matched samples, above the pre-registered threshold。

## 预期与实际偏差

预期：如果旧 negative 是由 layer-output mismatch 造成，hook-aligned rerun 应显著提高 target-damaging zeroing fraction。

实际：hook-aligned rerun 没有提高；primary/strict top2 与 top8 都保持低比例 target damage。top8 扩大 node set 后也没有出现更强 source-tracing signal。

## 结论

可以写：

`Under the hook-aligned Qwen2.5-VL-PLT source-tracing adapter, primary72 and strict72 do not support Gemma-style source tracing: selected source-traced feature node zeroing rarely damages the target answer logit/rank, and top8 sensitivity does not rescue the result.`

仍然不能写：

- `Qwen has no evidence-region-sensitive mechanism.`
- `Qwen has no cross-modal features.`
- `Qwen fully replicates Gemma-style source tracing.`

当前最稳妥口径：

`Qwen2.5-VL-PLT retains approximate feature/source-control and hidden/first-token bridge evidence from Stage3, but the stricter hook-aligned Gemma-style source-tracing adapter is not supported on paperpack72.`

