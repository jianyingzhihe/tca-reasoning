# 实验 004：Stage 2A-1 top24 pretrace 完整 trace / compare 读数

## 目的

本实验记录 Stage 2A targeted replication pack 的 pretrace 阶段完整结果。

这一阶段的目标是：

```text
1. 确认 top24 replication candidates 是否能完成 B_direct / D_visual_only 的 answer-aligned trace；
2. 检查是否有足够样本进入后续 source zeroing、nearest control 和 region-mask replication；
3. 对 B/D trace 做基础 compare，得到可排序的 traced feature nodes；
4. 判断当前 top24 是否已经足够支持 Stage 2A 继续推进，还是需要补一个新 pretrace queue。
```

本实验仍然不是最终机制结论。它是 Stage 2A 的筛选与中间表构建步骤。

## 输入

远端 run root：

```text
/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm/outputs/phase_ab/ab_answer_aligned/stage2a_pretrace_top24_20260519_202951
```

本地轻量同步目录：

```text
E:\Bridging\remote_sync\2026-05-19_stage2a_pretrace_top24
```

关键输入 manifest：

```text
E:\Bridging\doc\experiments\stage2\stage2a_candidate_selection\manifest_B_direct_stage2a_pretrace_top24.csv
E:\Bridging\doc\experiments\stage2\stage2a_candidate_selection\manifest_D_visual_only_stage2a_pretrace_top24.csv
E:\Bridging\doc\experiments\stage2\stage2a_candidate_selection\stage2a_trace_selected_ids_top24.csv
E:\Bridging\doc\experiments\stage2\stage2a_candidate_selection\stage2a_trace_queue_top24.csv
```

## 远端输出

远端完整 `.pt` 目录较大：

```text
run root total size: about 20G
```

因此本轮没有把全部 `.pt` graph 拉回本地，而是保留在服务器 run root 上继续跑重分析。同步到本地的是轻量结果：

```text
eval_B_direct.csv
eval_D_visual_only.csv
answer_aligned_meta_b.csv
answer_aligned_meta_a.csv
logs_eval_B_direct.log
logs_eval_D_visual_only.log
logs_trace_B_direct.log
logs_trace_D_visual_only.log
compare_selected_samples.csv
compare_run_A_summary.csv
compare_run_B_summary.csv
compare_sample_compare_controlled.csv
compare_nodes_detailed_controlled.csv
compare_edges_detailed_controlled.csv
compare_bucket_summary_controlled.csv
clean_subset_alignment_clean_stage2a_pretrace_top24.csv
clean_subset_alignment_clean_stage2a_pretrace_top24_summary.csv
stage2a_pretrace_trace_inventory.csv
stage2a_pretrace_clean_compare_readout.csv
```

## 方法

### 1. Eval

对 top24 分别跑：

```text
B_direct
D_visual_only
```

eval 脚本：

```text
scripts/research/run_batch_eval.py
```

### 2. Answer-Aligned Trace

对 B/D 的 predicted answer token 做 answer-aligned attribution：

```text
scripts/research/run_batch_answer_aligned_attribute.py
```

主要参数：

```text
transcoder_set = tianhux2/gemma3-4b-it-plt
dtype = bfloat16
max_feature_nodes = 64
topk = 16
offload = disk
exec_mode = subprocess
retry_feature_nodes = 48,32
answer_source = predicted
```

### 3. Trace Compare

trace 完成后，在远端继续跑：

```text
scripts/research/trace_compare_ab_controlled.py
```

参数：

```text
pt_dir_a = pt_slotA_D_visual_only
pt_dir_b = pt_slotB_B_direct
bucket = stage2a_pretrace_top24
target_logit_rank = 0
topk_per_node = 3
beam_per_depth = 96
coverage = 0.98
max_depth = 40
min_abs_weight = 0.0
```

这里的 A 对应 `D_visual_only`，B 对应 `B_direct`。

### 4. Alignment Clean Subset

trace compare 后运行：

```text
scripts/research/build_alignment_clean_subset.py
```

注意：这里的 `clean_core` 主要检查 B/D 是否具有相同目标 token、trace 状态是否 ok、问题 suffix 是否可 canonical strip、base question 是否匹配。

它不是最终 “source intervention clean-core” 判定，因为 source zeroing intervention 此时还没有完成。

## 结果一：trace 完成度

top24 trace 完成度：

| 项目 | 数量 |
|---|---:|
| B_direct eval 有效行 | 24 / 24 |
| D_visual_only eval 有效行 | 24 / 24 |
| B_direct `.pt` graph | 24 / 24 |
| D_visual_only `.pt` graph | 24 / 24 |
| B meta unique sample | 24 / 24 |
| D meta unique sample | 24 / 24 |
| both trace ok | 24 / 24 |

结论：

```text
Stage 2A-1 pretrace 在工程上完整成功。
没有 missing pt。
没有 trace-level OOM / fatal error。
```

## 结果二：样本构成

top24 构成：

| 维度 | 计数 |
|---|---:|
| total | 24 |
| pretrace_existing_mask | 21 |
| pretrace_discovery_pool | 3 |
| visual_readout | 21 |
| symbol_text_reading | 3 |
| existing answer/relate mask | 21 |
| existing mask + both trace ok | 21 |

这个结果很好，因为 Stage 2A 的核心需求是：

```text
从已经有 answer / relate mask 的 localized 样本里，筛出新的 support source + nearest control candidates。
```

目前至少在 trace 层面，已有 `21` 个带旧标注的候选可以继续推进。

## 结果三：eval 行为侧

行为侧已在实验 003 中单独记录，这里复述关键结论：

| prompt | n | correct | strict_gold | format_ok | empty |
|---|---:|---:|---:|---:|---:|
| B_direct | 24 | 8 | 7 | 24 | 0 |
| D_visual_only | 24 | 5 | 3 | 18 | 0 |

B/D 预测答案发生变化：

```text
13 / 24
```

D_visual_only 的非 canonical assistant prefix / format risk 样本：

```text
okvqa_val_667695
okvqa_val_2131565
okvqa_val_5735275
okvqa_val_1927165
okvqa_val_343215
okvqa_val_4033335
```

解释：

```text
D_visual_only 会改变输出倾向，但仍然不能作为“行为更优 prompt”来写。
它继续更适合作为 route modulation / probing variable。
```

## 结果四：trace compare

`trace_compare_ab_controlled.py` 结果：

```text
selected_samples = 24
dropped_missing_pt = 0
```

bucket summary：

| metric | value |
|---|---:|
| node_overlap_jaccard | 0.2394 |
| edge_overlap_jaccard | 0.1200 |
| a_target_top3_concentration | 0.0818 |
| b_target_top3_concentration | 0.0889 |
| delta_target_top3_concentration | -0.0071 |
| a_target_error_ratio | 0.6048 |
| b_target_error_ratio | 0.5891 |
| delta_target_error_ratio | +0.0157 |
| a_traced_max_depth | 2.5417 |
| b_traced_max_depth | 2.8333 |
| a_traced_nodes | 8.1667 |
| b_traced_nodes | 9.4167 |
| a_traced_edges | 12.4167 |
| b_traced_edges | 15.0417 |

节点表规模：

| 项目 | 数量 |
|---|---:|
| all nodes | 422 |
| feature nodes | 115 |
| A/D_visual_only feature nodes | 48 |
| B/B_direct feature nodes | 67 |

解释：

```text
1. B/D trace overlap 不高，说明 prompt 条件下 answer-adjacent route 确实有差异；
2. 但这仍是 observational compare，不作为主机制结论；
3. 真正用于机制筛选的是后续 source zeroing delta；
4. 115 个 feature nodes 足够做 top-k source intervention screen。
```

## 结果五：alignment-clean 初筛

alignment clean summary：

| metric | count |
|---|---:|
| total_samples | 24 |
| clean_core | 12 |
| clean_prompt | 12 |
| assistant_prefix_noncanonical | 6 |
| hint_contaminated | 0 |
| clean_core_any_run | 0 |
| clean_core_ab_pair | 0 |

注意：

```text
clean_core_any_run = 0 不是失败。
原因是 build_alignment_clean_subset.py 默认寻找 run_root 下名为 intervention_smoke_{bucket}.csv 的 intervention 文件。
本阶段还没有完成 source zeroing，所以 intervention_success_count 还没有写入该脚本预期的默认文件名。
```

因此当前真正可读的是：

```text
clean_core = 12
clean_prompt = 12
assistant_prefix_noncanonical = 6
```

clean_prompt 为 True 的 12 个样本：

| sample_id | answer | type | existing mask | B answer | D answer |
|---|---|---|---:|---|---|
| okvqa_val_3313665 | penny farthing | visual_readout | True | a penny-farthing | a penny-farthing bicycle |
| okvqa_val_1994425 | wetsuit | visual_readout | True | wetsuit | wetsuit |
| okvqa_val_2802115 | chainlink | visual_readout | True | chain-link fence | chain-link fence |
| okvqa_val_2496585 | paint | visual_readout | True | Dulux paint | Dulux paint |
| okvqa_val_2373185 | 8 | visual_readout | True | three | three |
| okvqa_val_1083925 | donuts | visual_readout | True | donuts | donuts |
| okvqa_val_3794755 | laptop | visual_readout | True | a laptop, monitor, and printer | a laptop, a desktop computer, and a tablet |
| okvqa_val_1996815 | microwave | visual_readout | True | microwave | a microwave |
| okvqa_val_602025 | sheep dog | visual_readout | True | Border Collie | Border Collie |
| okvqa_val_5291225 | chinese | symbol_text_reading | False | Japanese | Japanese |
| okvqa_val_1985905 | chevy | symbol_text_reading | False | Chevrolet | Chevrolet |
| okvqa_val_4549785 | honda | symbol_text_reading | False | Yamaha | Yamaha |

其中最直接服务 region-mask replication 的是：

```text
clean_prompt=True AND existing_mask=True
```

数量：

```text
9 samples
```

这已经接近 run plan 的最低门槛，但还需要 source zeroing 和 nearest control 进一步筛。

## 预期与实际偏差

预期：

```text
top24 中至少 10-15 个 trace 成功；
至少 8 个 clean-core 可用；
至少 8 个 support source；
至少 6 个 nearest control；
已有 mask 样本占多数。
```

实际目前已确认：

```text
trace 成功：24 / 24，强于预期；
existing mask + trace 成功：21 / 24，强于预期；
clean_prompt：12 / 24，达到基础筛选要求；
clean_prompt + existing mask：9 / 24，接近/达到 region replication 最低门槛；
support source 与 nearest control：尚需等待 source zeroing 和 matched control screen。
```

偏差解释：

```text
工程 trace yield 很好；
行为正确率不高，但 Stage 2A 的目标不是行为 benchmark；
D_visual_only 格式不稳定仍然存在；
symbol_text_reading 的三个样本 clean_prompt 成立，但缺少现成 mask，需要后续决定是否补标。
```

## 当前结论

Stage 2A-1 pretrace 阶段成功。

可以保守写成：

```text
The Stage 2A pretrace queue produced complete B/D answer-aligned graphs for all 24 selected localized candidates. Among them, 21 already have answer/relate region annotations, and 12 satisfy the stricter B/D target-alignment and prompt-clean checks. This is sufficient to proceed to source-node intervention screening.
```

中文：

```text
Stage 2A top24 pretrace 已经为全部 24 个候选样本产出 B/D answer-aligned graph。
其中 21 个已有 answer/relate 区域标注，12 个满足更严格的 B/D 目标 token 与 prompt 对齐检查。
这足够继续推进到 source zeroing 和 nearest control 筛选。
```

## 对主 claim 的影响

本实验本身不证明 evidence-region-sensitive support routes。

它对主 claim 的贡献是：

```text
1. 提供一个独立于 core24 的新 pretrace pool；
2. 证明该 pool 的 trace yield 足够高；
3. 为 Stage 2A targeted replication 提供候选 source feature nodes；
4. 避免直接从旧 core24 结果中挑 positive case，降低 selection bias。
```

主 claim 是否复现，仍取决于接下来的三步：

```text
source zeroing 是否找到足够 support source nodes；
nearest control 是否可构造；
answer/union region mask 是否比 random region 更强地削弱 support route。
```

## 后续动作

下一步已启动：

```text
Stage 2A-2 source zeroing top4 screen
```

远端输出目标：

```text
/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm/outputs/phase_ab/ab_answer_aligned/stage2a_pretrace_top24_20260519_202951/intervention/stage2a_source_zeroing_top4.csv
```

source zeroing 完成后要做：

```text
1. 给每个 feature node 判定 support / suppressor；
2. 统计 clean_prompt + existing_mask 样本中有多少 support source；
3. 构建 nearest controls；
4. 若 support+nearest+mask 样本数 >= 8，则进入 region-mask replication；
5. 若不足，则补一个 symbol_text_reading 或 visual_readout-heavy queue / 或让用户补标少量 high-yield 样本。
```
