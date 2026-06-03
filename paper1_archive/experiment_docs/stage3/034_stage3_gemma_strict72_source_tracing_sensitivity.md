# 034 Stage3 Gemma3-PLT Strict72 Source-Tracing Sensitivity

## 目的

在 `paperpack72_strict_sensitivity` 上复跑 Gemma3-PLT source-tracing，用于检查 primary72 结果是否依赖 5 个 `manual_review_image_dependence` 样本。

本实验是 PLT-only verdict 的 Gemma 侧 sensitivity 输入：它验证 eval、gold-answer aligned attribution graph、A/B controlled compare 是否能在 strict pack 上完整产生。它不比较 `D_visual_only` 是否更好，也不把 strict intervention smoke 的工程失败写成机制负结果。

## 输入

```text
pack: paperpack72_strict_sensitivity
prompt A: D_visual_only
prompt B: B_direct
rows per prompt: 72
answer alignment: gold answer / paperpack answer_text
base model: google/gemma-3-4b-it
transcoder: tianhux2/gemma3-4b-it-plt
remote run root:
  /root/autodl-tmp/tca-reasoning/stage3_gemma_paperpack/source_tracing_strict_full
```

关键本地 artifact：

```text
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_strict_full_decision.json
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_strict_full_analysis.json
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_strict_full_eval_A_D_visual_only.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_strict_full_eval_B_B_direct.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_strict_full_valid_samples.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_strict_full_sample_compare_controlled.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_strict_full_nodes_detailed_controlled.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_strict_full_edges_detailed_controlled.csv
```

## 方法

执行顺序：

```text
1. run_batch_eval.py for D_visual_only and B_direct
2. filter valid samples with non-empty generated_text in both prompts
3. run_batch_answer_aligned_attribute.py with --answer-source gold
4. trace_compare_ab_controlled.py with A=D_visual_only and B=B_direct
5. attempt run_answer_aligned_intervention_smoke.py
6. analyze_stage3_gemma_source_tracing_full.py --pack strict --mode full
```

本轮继续使用 primary full 已验证的 `--lazy-encoder --lazy-decoder` attribution path。strict 续跑过程中 `/root/autodl-tmp` 一度满盘，导致 B-side graph 写入失败；释放已经拉回本地并文档化的远端 primary graph 中间目录后，使用 `--resume` 补齐 strict B-side graph。

## 结果

总体状态：

```text
remote decision: blocked
local analyzer status: strict_full_graph_compare_passed_intervention_blocked
eval A rows: 72
eval B rows: 72
valid samples: 71 / 72
graph A files: 71
graph B files: 71
graph success rate vs valid: 1.0
sample compare rows: 71
nodes detailed rows: 4095
edges detailed rows: 7003
intervention rows: 0
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
node_overlap_jaccard_mean: 0.2999
edge_overlap_jaccard_mean: 0.1996
delta_target_total_in_abs_mean: -7.6097
A traced nodes mean: 28.17
B traced nodes mean: 29.51
A traced edges mean: 47.42
B traced edges mean: 51.21
```

与 primary72 对比：

```text
primary valid samples: 71 / 72
primary graph success rate: 1.0
primary sample compare rows: 71
primary node rows: 4126
primary edge rows: 7071
primary intervention rows: 127

strict valid samples: 71 / 72
strict graph success rate: 1.0
strict sample compare rows: 71
strict node rows: 4095
strict edge rows: 7003
strict intervention rows: 0
```

## 预期与实际偏差

预期 strict full 能复现 primary full 的完整 source-tracing pipeline。实际结果是 strict 的 eval、A/B attribution graph、controlled compare、node/edge details 全部通过，但 intervention smoke 在最后阶段被远端系统以 `SIGKILL(137)` 打断。

诊断过程：

```text
1. strict B-side attribution 最初因 /root/autodl-tmp 满盘在 torch.save 阶段失败。
2. 清理远端 primary graph 中间目录后，strict B-side 通过 --resume 补齐到 71/71。
3. 修复 run_batch_answer_aligned_attribute.py 的 resume metadata bug，使已有 graph 的 target_token_id 能被重新写入 metadata。
4. intervention smoke 随后在 ReplacementModel / HookedVLTransformer 构造阶段被 137 kill。
5. 单独 circuit_tracer attribute load probe 也在同一阶段被 137 kill，说明这是当前远端模型加载/资源状态问题，而不是 strict graph 或 sample compare 的机制负结果。
```

因此 strict sensitivity 的正确口径是：

```text
strict graph/compare sensitivity passed; strict intervention smoke is engineering-blocked in the current remote state.
```

## 结论

Gemma3-PLT 在 `paperpack72_strict_sensitivity` 上完成了 strict eval、gold-answer aligned attribution graph、A/B controlled compare 和 node/edge detail 复跑，且 graph coverage 为 `71/71`。这支持 primary72 source-tracing 结果不是由 manual-review 样本驱动。

当前可以写：

```text
Gemma3-PLT strict72 reproduces the source-tracing graph/compare pipeline on 71 valid samples.
```

当前不能写：

```text
Gemma3-PLT strict72 intervention smoke fully passed.
Qwen2.5-VL fully replicates Gemma-style source tracing.
The strict intervention block is evidence against the mechanism.
```

PLT-only verdict 可以基于 Gemma primary full、Gemma strict graph/compare sensitivity、以及已完成的 Qwen2.5-VL-PLT primary/strict approximate source-control support来写，但必须保留上述边界。
<!-- 2026-05-23 update: strict lightweight intervention repair passed with B-only max_samples=4, top_features_per_sample=1; full strict intervention remains incomplete, but earlier SIGKILL(137) should be treated as remote model-load/resource-policy instability rather than mechanism evidence. Artifact: cross_model/stage3_gemma_source_tracing_strict_full_intervention_repair_B4.csv. -->
