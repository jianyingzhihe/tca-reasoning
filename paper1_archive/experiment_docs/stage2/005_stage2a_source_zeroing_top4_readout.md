# 实验 005：Stage 2A-2 source zeroing top4 初筛

## 目的

本实验是 Stage 2A targeted replication 的第二步。

在实验 004 中，我们已经为 top24 候选样本完成了：

```text
B_direct / D_visual_only eval
B_direct / D_visual_only answer-aligned trace
B/D trace compare
```

本实验继续对 traced feature nodes 做 clean-image source zeroing intervention。

核心目的：

```text
1. 给 traced source feature node 判定方向：
   - support：清零后目标答案 logit 下降；
   - suppressor：清零后目标答案 logit 上升；
   - neutral：清零后目标答案 logit 基本不变。

2. 判断 top24 pretrace pool 中是否有足够 support source nodes 进入 region-mask replication。

3. 为下一步 nearest matched non-source control 构造 source node 清单。
```

## 输入

远端 run root：

```text
/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm/outputs/phase_ab/ab_answer_aligned/stage2a_pretrace_top24_20260519_202951
```

依赖文件：

```text
compare/nodes_detailed_controlled.csv
answer_aligned_meta_a.csv
answer_aligned_meta_b.csv
doc/experiments/stage2/stage2a_candidate_selection/stage2a_trace_selected_ids_top24.csv
```

## 方法

远端运行脚本：

```text
scripts/research/run_answer_aligned_intervention_smoke.py
```

主要参数：

```text
run_root = stage2a_pretrace_top24_20260519_202951
compare_dir = run_root/compare
run = both
top_features_per_sample = 4
transcoder_set = tianhux2/gemma3-4b-it-plt
dtype = bfloat16
```

输出：

```text
remote:
/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm/outputs/phase_ab/ab_answer_aligned/stage2a_pretrace_top24_20260519_202951/intervention/stage2a_source_zeroing_top4.csv

local:
E:\Bridging\remote_sync\2026-05-19_stage2a_pretrace_top24\stage2a_source_zeroing_top4.csv
E:\Bridging\remote_sync\2026-05-19_stage2a_pretrace_top24\stage2a_source_zeroing_top4_enriched.csv
E:\Bridging\remote_sync\2026-05-19_stage2a_pretrace_top24\stage2a_source_zeroing_top4_sample_summary.csv
```

## 方向判定规则

本实验使用 target logit delta 判定 node role：

```text
delta_target_logit = intervened_target_logit - original_target_logit
```

解释：

```text
delta_target_logit < 0:
  清零该 node 后目标答案 logit 下降；
  该 node 原本支持目标答案；
  记为 support。

delta_target_logit > 0:
  清零该 node 后目标答案 logit 上升；
  该 node 原本压制目标答案，或支持竞争答案；
  记为 suppressor。

delta_target_logit = 0:
  当前干预下没有可见方向；
  记为 neutral。
```

## 运行状态

远端运行完成，日志统计：

```text
successful_feature_rows = 87
skipped_missing_target = 0
skipped_out_of_range = 0
skipped_pos_buffer = 0
sample_runs_without_candidates = 13
exhausted_sample_runs = 22
```

解释：

```text
1. 没有 target metadata 缺失；
2. 没有 position out-of-range；
3. 没有 token-position buffer 问题；
4. top4 是上限，不是每个 sample x prompt 都有 4 个可干预 feature；
5. 因此最终成功 intervention row 是 87，不是理论上限 24 x 2 x 4 = 192。
```

注意：`sample_runs_without_candidates = 13` 和 `exhausted_sample_runs = 22` 主要反映 compare graph 中可用 feature nodes 数量有限，不是 intervention 失败。

## 总体结果

source zeroing role counts：

| role | count |
|---|---:|
| support | 46 |
| suppressor | 35 |
| neutral | 6 |
| total | 87 |

按 prompt/run 切分：

| run | role | count |
|---|---|---:|
| A / D_visual_only | support | 19 |
| A / D_visual_only | suppressor | 15 |
| A / D_visual_only | neutral | 3 |
| B / B_direct | support | 27 |
| B / B_direct | suppressor | 20 |
| B / B_direct | neutral | 3 |

样本级覆盖：

| 条件 | sample count |
|---|---:|
| any support source | 19 |
| any suppressor source | 16 |
| existing mask + any support source | 17 |
| clean_prompt + any support source | 8 |
| clean_prompt + existing mask + any support source | 6 |

## 关键样本池

### 最严格池：clean_prompt + existing mask + support

数量：

```text
6 samples
```

样本：

```text
okvqa_val_1994425
okvqa_val_1996815
okvqa_val_2373185
okvqa_val_2496585
okvqa_val_3794755
okvqa_val_602025
```

解释：

```text
这个池最干净，因为它同时满足：
1. B/D target alignment 和 prompt-clean 检查；
2. 已有 answer/relate mask；
3. 至少一个 support source node。
```

但它只有 6 个样本，低于 run plan 里理想的 8 个门槛。

### 主 replication 候选池：existing mask + support

数量：

```text
17 samples
```

这个池不强制 B/D 目标 token 相同，也不强制 D_visual_only 格式完全 canonical。

解释：

```text
如果主分析口径是 prompt-specific route，
即每个 prompt 追踪并干预它自己的 predicted-answer route，
那么这个池可以继续用于 region-mask replication。

如果主分析口径是严格 B/D direct comparison，
则只能使用上面的 6 个 clean_prompt + mask support 样本，统计会偏弱。
```

这与我们当前主 claim 是一致的：

```text
主目标不是证明 D_visual_only 比 B_direct 更好；
主目标是验证 prompt-specific answer-adjacent support routes 是否 evidence-region-sensitive。
```

## 每个已有 mask support 样本的 best support source

| sample_id | clean_prompt | n_support | best support run | best support delta | best support feature | n_suppressor |
|---|---:|---:|---|---:|---|---:|
| okvqa_val_1083155 | False | 1 | B | -3.75 | L11:P196:F151858 | 0 |
| okvqa_val_1740705 | False | 4 | A | -1.0625 | L26:P299:F59821 | 2 |
| okvqa_val_1927165 | False | 3 | B | -1.875 | L25:P286:F118453 | 2 |
| okvqa_val_1994425 | True | 1 | A | -0.21875 | L11:P297:F148968 | 1 |
| okvqa_val_1996815 | True | 2 | B | -0.875 | L26:P290:F109552 | 4 |
| okvqa_val_2131565 | False | 3 | B | -1.0 | L24:P288:F47397 | 2 |
| okvqa_val_2373185 | True | 2 | B | -0.5 | L11:P290:F148968 | 1 |
| okvqa_val_2496585 | True | 2 | A | -1.4004 | L11:P196:F151858 | 0 |
| okvqa_val_2683965 | False | 3 | B | -2.25 | L26:P287:F39687 | 1 |
| okvqa_val_3265105 | False | 1 | A | -1.75 | L11:P196:F151858 | 0 |
| okvqa_val_3326275 | False | 1 | A | -0.4453 | L11:P196:F151858 | 0 |
| okvqa_val_343215 | False | 3 | B | -1.125 | L26:P286:F59821 | 1 |
| okvqa_val_3794755 | True | 2 | B | -0.25 | L26:P288:F84060 | 4 |
| okvqa_val_5735275 | False | 4 | B | -0.875 | L25:P287:F120202 | 3 |
| okvqa_val_602025 | True | 4 | A | -1.5 | L11:P293:F148968 | 4 |
| okvqa_val_667695 | False | 1 | A | -0.5 | L11:P196:F151858 | 1 |
| okvqa_val_80655 | False | 3 | B | -1.5 | L11:P196:F151858 | 3 |

## 结果解释

### 支持主线的地方

source zeroing 明确复现了 signed route：

```text
support rows = 46
suppressor rows = 35
neutral rows = 6
```

这说明新 top24 pool 中的 traced feature nodes 不是统一支持答案，而是 mixed-sign causal nodes。

这与前期 pack10 / strong12 / strong18 的结果一致。

### 对 Stage 2A replication 的影响

最重要的读数是：

```text
existing mask + support source samples = 17
```

这说明：

```text
如果采用 prompt-specific source route 口径，
Stage 2A 有足够样本继续推进 region-mask replication。
```

但最严格的：

```text
clean_prompt + existing mask + support samples = 6
```

略低于理想门槛 8。

因此后续写法必须谨慎：

```text
主 replication 可以使用 prompt-specific route pool；
严格 B/D target-aligned prompt comparison 只能作为 secondary / underpowered slice。
```

### 对 D_visual_only 的影响

这一步继续不支持：

```text
D_visual_only is better than B_direct
```

因为 role counts 里 B_direct 反而有更多 support rows：

```text
B support = 27
A/D support = 19
```

但这不能简单解释成 B 更好。

更稳的解释仍然是：

```text
不同 prompt 暴露或改变了不同 answer-adjacent routes；
prompt 是 route modulation/probing variable，而不是行为优劣主角。
```

## 预期与实际偏差

预期：

```text
top24 中至少 8 个样本出现 support source；
已有 mask 样本中保留足够 source nodes；
后续能构造 nearest control。
```

实际：

```text
any support sample = 19
existing mask + support sample = 17
clean_prompt + existing mask + support sample = 6
```

偏差：

```text
总体 source support yield 高于预期；
严格 B/D clean-prompt support pool 略低于理想门槛；
这提示 Stage 2A 应以 prompt-specific source route replication 为主，
不要把 B/D strict comparison 写成 primary。
```

## 当前结论

本实验支持继续推进 Stage 2A。

可以写成：

```text
Stage 2A source zeroing identified a substantial set of signed source nodes in the new top24 pretrace pool. Among 87 successful node interventions, 46 were support effects and 35 were suppressor effects. Seventeen already-annotated samples contain at least one support source node, providing a viable pool for evidence-region replication under a prompt-specific route framing.
```

中文：

```text
Stage 2A source zeroing 在新的 top24 pretrace pool 中找到了足够多的 signed source nodes。
87 条成功 node intervention 中，46 条是 support，35 条是 suppressor。
已有 answer/relate mask 的样本中有 17 个包含 support source node，因此在 prompt-specific route 口径下足够继续推进 evidence-region replication。
```

## 后续动作

下一步已经启动：

```text
nearest matched non-source control clean screen
```

远端输入：

```text
stage2a_source_zeroing_top4_pilot_clean_nonzero.csv
```

该 pilot CSV 包含：

```text
81 non-neutral source rows
condition = clean
node_role = support / suppressor
```

远端输出目标：

```text
stage2a_nearest_control_clean_top4_nonzero.csv
```

nearest control 完成后要检查：

```text
1. 有多少 support source rows 可以找到 nearest control；
2. source abs effect 是否大于 nearest control abs effect；
3. clean_prompt + existing mask + support + nearest 是否达到 6-8；
4. existing mask + support + nearest 是否达到 8+；
5. 是否需要补标 symbol_text_reading 三个 clean_prompt 样本，或补一个 focused mini queue。
```
