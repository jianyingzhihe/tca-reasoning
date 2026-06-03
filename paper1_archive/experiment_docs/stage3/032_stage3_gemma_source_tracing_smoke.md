# 032 Stage3 Gemma Source-Tracing Smoke

## 目的

在不直接启动 full 72 的前提下，验证 Gemma3-PLT paperpack 输入能否进入完整 source-tracing 链路：

```text
eval -> gold-answer-aligned attribution -> A/B graph compare -> answer-aligned intervention smoke
```

这个实验是工程 smoke，不是 Gemma paperpack full result。

## 输入

```text
Gemma3-PLT:
  tianhux2/gemma3-4b-it-plt

base:
  google/gemma-3-4b-it

smoke manifest:
  doc/experiments/stage3/cross_model/stage3_gemma_eval_smoke_B_direct.csv
  doc/experiments/stage3/cross_model/stage3_gemma_eval_smoke_D_visual_only.csv
  doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_smoke_manifest.csv

answer source:
  gold answer from paperpack answer_text
```

Smoke 初始样本为 3 个 strong original visual_readout 样本：

```text
okvqa_val_00528
okvqa_val_00912
okvqa_val_01162
```

## 输出

```text
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_smoke_decision.json
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_smoke_eval_A_D_visual_only.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_smoke_eval_B_B_direct.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_smoke_meta_a.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_smoke_meta_b.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_smoke_valid_samples.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_smoke_sample_compare_controlled.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_smoke_bucket_summary_controlled.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_smoke_nodes_detailed_controlled.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_smoke_edges_detailed_controlled.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_smoke_intervention.csv
```

## 方法

Prompt 设定：

```text
A = D_visual_only
B = B_direct
```

因为 `D_visual_only` 在一个 smoke 样本上生成空文本，runner 做了保守过滤：只保留两个 prompt 下 `generated_text` 都非空且无 `error_message` 的样本进入 attribution/compare/intervention。

保留样本：

```text
okvqa_val_00528
okvqa_val_00912
```

执行脚本：

```text
run_batch_eval.py
run_batch_answer_aligned_attribute.py --answer-source gold
trace_compare_ab_controlled.py
run_answer_aligned_intervention_smoke.py
```

## 结果

```text
status: pass
eval_a_rows: 3
eval_b_rows: 3
valid_smoke_sample_rows: 2
meta_a_rows: 2
meta_b_rows: 2
sample_compare_rows: 2
nodes_detailed_rows: 113
intervention_rows: 4
```

Graph 产物：

```text
A/D_visual_only graph files: 2
B/B_direct graph files: 2
```

Compare 摘要：

```text
node_overlap_jaccard mean: 0.5965
edge_overlap_jaccard mean: 0.4983
a_traced_nodes mean: 27.5
b_traced_nodes mean: 29.0
a_traced_edges mean: 43.0
b_traced_edges mean: 49.5
```

Intervention smoke 产生 4 行，说明 selected feature zeroing 链路可执行。该 smoke 没有解释为机制支持或反对，只证明 paperpack Gemma source-tracing pipeline 可以继续推进到 full run。

## 预期与实际偏差

预期是 3/3 样本进入 attribution；实际 `okvqa_val_01162` 在 `D_visual_only` eval 中 `generated_text` 为空，因此被过滤。这个问题属于 eval/template 行为诊断，不是机制负结果。

另一个工程细节是 `okvqa_val_00528` 的 A 侧 attribution 在 `max_feature_nodes=64` 上触发 fallback，最终在 `48` 成功。这说明 full run 应继续保留 `retry-feature-nodes`。

## 结论

Gemma3-PLT paperpack source-tracing smoke 通过。下一步可以跑 `paperpack72_primary` full Gemma source tracing；如果 full 72 成本过高，最低应先跑 confirmatory 48。

当前仍不能写 Gemma paperpack full result，也不能据此完成 PLT-only verdict。

