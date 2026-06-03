# 033 Stage3 Gemma3-PLT Primary72 Source-Tracing Full

## 目的

在 `paperpack72_primary` 上补齐 Gemma3-PLT 侧的 full source-tracing rerun，替代此前只有少量历史 overlap 与 3-case smoke 的弱校准结果。

本实验用于 PLT-first 主线的 Gemma 侧证据：确认同一个 paperpack72 上，Gemma3-PLT 的 eval、gold-answer aligned attribution graph、A/B controlled compare 与 node intervention smoke 可以完整产出。它不是 Qwen adapter 结果，也不把 `D_visual_only` 解释为更优 prompt。

## 输入

```text
pack: paperpack72_primary
prompt A: D_visual_only
prompt B: B_direct
rows per prompt: 72
answer alignment: gold answer / paperpack answer_text
base model: google/gemma-3-4b-it
transcoder: tianhux2/gemma3-4b-it-plt
remote run root:
  /root/autodl-tmp/tca-reasoning/stage3_gemma_paperpack/source_tracing_primary_full
```

关键本地 artifact：

```text
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_primary_full_decision.json
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_primary_full_analysis.json
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_primary_full_eval_A_D_visual_only.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_primary_full_eval_B_B_direct.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_primary_full_valid_samples.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_primary_full_sample_compare_controlled.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_primary_full_nodes_detailed_controlled.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_primary_full_edges_detailed_controlled.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_primary_full_intervention.csv
```

## 方法

执行顺序：

```text
1. run_batch_eval.py for D_visual_only and B_direct
2. filter valid samples with non-empty generated_text in both prompts
3. run_batch_answer_aligned_attribute.py with --answer-source gold
4. trace_compare_ab_controlled.py with A=D_visual_only and B=B_direct
5. run_answer_aligned_intervention_smoke.py on selected source/control features
6. analyze_stage3_gemma_source_tracing_full.py
```

实现修复：在 `run_batch_answer_aligned_attribute.py` 的 subprocess `circuit_tracer attribute` 调用中加入 `--lazy-encoder --lazy-decoder`。修复前 Gemma attribution 会以非 lazy encoder 路径加载，显存峰值约 35-37GiB 并触发 OOM；修复后主要稳定在约 8-11GiB，primary full 顺利完成。

## 结果

总体状态：

```text
remote decision: pass
local analyzer status: primary_full_passed
eval A rows: 72
eval B rows: 72
valid samples: 71 / 72
graph A files: 71
graph B files: 71
graph success rate vs valid: 1.0
sample compare rows: 71
nodes detailed rows: 4126
edges detailed rows: 7071
intervention rows: 127
```

唯一 eval 过滤样本：

```text
sample_id: okvqa_val_01162
answer_text: tony hawk
reasoning_operation: visual_readout
image_dependence_tier: strong
failure: empty_generated_a under D_visual_only
B_direct generated: The answer is Tony Hawk.
```

Controlled compare 摘要：

```text
node_overlap_jaccard_mean: 0.3092
edge_overlap_jaccard_mean: 0.2094
delta_target_total_in_abs_mean: -7.6160
A traced nodes mean: 28.32
B traced nodes mean: 29.79
A traced edges mean: 48.01
B traced edges mean: 51.58
```

Node rows 摘要：

```text
A feature rows: 629
B feature rows: 686
A token rows: 194
B token rows: 217
A error rows: 1117
B error rows: 1141
```

Intervention smoke 摘要：

```text
A intervention rows: 57
A mean delta_target_logit: +0.1094
A positive-rate delta_target_logit: 0.4912
B intervention rows: 70
B mean delta_target_logit: +0.2812
B positive-rate delta_target_logit: 0.6286
```

## 预期与实际偏差

预期 full primary 能覆盖 72 个样本。实际保留 71 个 valid samples，原因是 `okvqa_val_01162` 在 `D_visual_only` 下空生成；这属于 eval/template 或行为诊断，不写成机制负结果。

Intervention smoke 中出现较多 `position_buffer_exceeded` 与少量 `missing target_token_id`。这说明 selected feature zeroing 的末端位置对齐还有工程限制，但 full source-tracing graph 与 controlled compare 已经完整通过。后续如果要把 Gemma paperpack intervention 也升级到论文级主 endpoint，需要单独修复 position buffer / target-token metadata，而不是把这些 skip 解读为机制失败。

## 结论

Gemma3-PLT 在 `paperpack72_primary` 上已经完成 full source-tracing rerun：eval、gold-answer aligned attribution graph、controlled compare、node/edge details 与 intervention smoke 均产出，并达到进入 strict72 sensitivity 的门槛。

当前可以写：`Gemma3-PLT primary72 source-tracing pipeline passed on the independently annotated paperpack, with 71/72 valid samples and complete A/B graphs.`

当前仍不能写：`PLT-only verdict 已完成`。还需要按 run plan 跑 `paperpack72_strict_sensitivity` full，并将 Gemma primary/strict 与已完成的 Qwen2.5-VL-PLT paperpack 结果合并后再写 verdict。
