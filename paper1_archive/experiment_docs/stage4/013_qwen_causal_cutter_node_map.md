# Stage4-013 Qwen Causal Cutter Node Map

## 目的

Stage4-012 说明：Qwen2.5-VL-PLT 的 hook-aligned Gemma-style source-tracing adapter 不成立，原因是 direct/source-traced top nodes 被剪掉后很少稳定损伤 target。

但这不等于 Qwen 没有可剪 causal nodes。本实验转向 **causal cutter search**：

先从已经跑完的 top8 intervention 中寻找“剪掉后真的伤 target”的节点，再判断这些节点是否集中、是否 rank-aware、是否值得进入 evidence/matched-control validation。

## 输入

- `stage4_qwen_source_tracing_primary_full_hookfix_top8_intervention.csv`
- `stage4_qwen_source_tracing_strict_full_hookfix_top8_intervention.csv`
- 过滤条件：
  - `original_target_rank <= 10`
  - `delta_target_logit < 0` 或 `delta_target_rank > 0`

## 输出

- `cross_model/stage4_qwen_cutter_positive_nodes_from_top8.csv`
- `cross_model/stage4_qwen_cutter_positive_nodes_summary.csv`
- `cross_model/stage4_qwen_cutter_rankaware_primary_top8.csv`
- `cross_model/stage4_qwen_cutter_rankaware_strict_top8.csv`

## 方法

对 primary/strict top8 intervention rows 做两层筛选：

1. 宽筛：任意 target logit 下降或 rank 变差。
2. rank-aware 筛：只保留原始 target rank ≤ 10 的 rows，避免把本来 target 很弱的样本误当机制证据。

然后统计：

- positive rows 数量；
- 涉及 sample 数；
- feature_id 集中度；
- top damaging rows；
- primary/strict 是否重复出现同类 feature。

## 结果

宽筛结果：

| pack | mode | rows | negative rows | negative frac | rank-hurt rows | affected samples | top negative features |
|---|---|---:|---:|---:|---:|---:|---|
| primary top8 | subtract | 1120 | 108 | 9.64% | 81 | 39 | `1091`, `1215`, `2508` |
| primary top8 | add | 1120 | 98 | 8.75% | 78 | 36 | `1091`, `1215`, `3353` |
| strict top8 | subtract | 1120 | 95 | 8.48% | 71 | 37 | `1091`, `1215`, `2508` |
| strict top8 | add | 1120 | 85 | 7.59% | 64 | 33 | `1091`, `1215`, `3353` |

Rank-aware 筛选：

- primary top8: 157 rows, 35 samples
- strict top8: 142 rows, 35 samples
- top features:
  - primary: `1091`, `1215`, `3353`, `2508`, `4383`
  - strict: `1091`, `3353`, `1215`, `2508`, `4383`

最强 rank-aware examples 主要集中在数字答案：

- `okvqa_val_5252115`, target token `2`, feature `1215`, logit drop up to `-2.75`
- `okvqa_val_5241085`, target token `1`, feature `1215/2508`, logit drop around `-0.875` to `-1.0`
- `okvqa_val_5686755`, target token `3`, feature `1215`, logit drop around `-1.0`
- `okvqa_val_2621615`, target token `1`, feature `1215/2508`, logit drop around `-0.75` to `-0.875`

## 解释

这说明两件事可以同时成立：

1. **Gemma-style direct source tracing 在 Qwen 上没有闭合。** 自动 trace 出来的 top nodes，整体剪切损伤比例很低。
2. **Qwen 里确实存在少数可剪 causal cutter nodes。** 这些节点不是自动 source-tracing 全局成功，但可以作为机制摸索入口。

换句话说：

`Qwen has cuttable PLT nodes, but current automatic Gemma-style source tracing does not reliably select a causal route.`

## 下一步实验

Stage4-014 应做 targeted validation，而不是继续盲目扩大 topK：

1. 选 rank-aware cutter candidates：
   - primary/strict 都出现；
   - clean target rank ≤ 10；
   - logit drop ≥ 0.25 或 rank hurt ≥ 1；
   - 优先 feature `1215/2508/1091/3353` 的高损伤 rows。
2. 对每个 candidate 做：
   - clean source zeroing；
   - wrong-target zeroing；
   - answer_mask / union_mask restore；
   - shifted / shuffled mask controls；
   - same-position matched feature control；
   - same-feature random-position control。
3. 若 source cutter 明显强于 controls，写：
   - `Qwen causal-screened PLT cutter nodes exist.`
4. 仍然不能写：
   - `Qwen fully replicates Gemma-style source tracing.`

## 当前结论

可以写：

`A rank-aware post-hoc cutter map finds sparse Qwen2.5-VL-PLT nodes whose zeroing can damage target logits/ranks, especially for numeric answers, but these remain causal-screened candidates requiring matched and evidence-mask controls.`

不能写：

- `Qwen source tracing is rescued.`
- `These cutter nodes prove full Gemma-style source-route replication.`
- `Feature IDs such as 1215 are object/number semantic nodes without top-activation or heatmap evidence.`

