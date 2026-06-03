# 实验 009：Stage 2A-6 behavior / generation 读数与 route-behavior linkage

## 目的

本实验补 Stage 2A nearest8 的行为侧证据。

实验 008 已经说明：

```text
support source route 在 answer / union evidence-region mask 下整体变弱；
但 source > nearest 和 strict random4 specificity 只是 partial / diagnostic。
```

本实验继续问：

```text
这些 evidence-region masks 是否也会伤害 target rank / target margin？
它们是否会改变最终 decoded answer？
route weakening 与行为损伤之间是否有方向一致关系？
```

这一步不是 node-to-generation causal bridge。它证明的是：

```text
区域遮挡同时影响内部 route 和行为输出。
```

还不能证明：

```text
清零某个 node 会直接改变自然生成答案。
```

## 输入

行为 eval 远端输出：

```text
/root/autodl-tmp/tca-reasoning/annotation/stage2a_region_replication_top24_nearest8/region_mask_stage2a_nearest8_behavior.csv
```

生成 eval 远端输出：

```text
/root/autodl-tmp/tca-reasoning/annotation/stage2a_region_replication_top24_nearest8/region_mask_stage2a_nearest8_generation.csv
```

本地同步文件：

```text
E:\Bridging\annotation\stage2a_region_replication_top24_nearest8\region_mask_stage2a_nearest8_behavior.csv
E:\Bridging\annotation\stage2a_region_replication_top24_nearest8\region_mask_stage2a_nearest8_generation.csv
```

route 输入：

```text
E:\Bridging\annotation\stage2a_region_replication_top24_nearest8\analysis_route\route_weakening_iou1.csv
```

分析脚本：

```text
E:\Bridging\scripts\local\summarize_stage2a_behavior_generation.py
```

## 输出

分析目录：

```text
E:\Bridging\annotation\stage2a_region_replication_top24_nearest8\analysis_behavior
```

关键文件：

```text
behavior_wide_iou0p05.csv
generation_cases_iou0p05.csv
generation_summary_iou0p05.csv
behavior_metric_summary_iou0p05.csv
route_behavior_linkage_iou0p05.csv
route_behavior_correlations_iou0p05.csv
STAGE2A_BEHAVIOR_GENERATION_READOUT_iou0p05.md
STAGE2A_BEHAVIOR_GENERATION_READOUT_iou1.md
```

## 方法

### 1. Behavior eval

远端脚本：

```text
scripts/research/run_region_mask_behavior_eval.py
```

它对每个 `sample_id x run` 去重后运行：

```text
clean
answer_mask
relate_mask
union_mask
random_control_1..4
```

输出：

```text
target_rank
target_vs_competitor_margin
target_logit
target_prob
top5 tokens
```

行为损伤定义：

```text
rank_damage = masked_target_rank - clean_target_rank
margin_drop = clean_margin - masked_margin
```

解释：

```text
rank_damage > 0 表示目标答案 token 排名变差。
margin_drop > 0 表示目标答案相对竞争 token 的 margin 变差。
```

### 2. Generation eval

远端脚本：

```text
scripts/research/run_region_mask_generation_eval.py
```

同样对每个 `sample_id x run` 去重，生成短答案。

输出：

```text
generated_text
predicted_answer
error_message
```

本实验使用 normalized exact change：

```text
answer_changed_from_clean = normalized(predicted_answer_condition) != normalized(predicted_answer_clean)
```

注意：

```text
这是严格字符串变化，不等价于语义错误率。
例如 “a rectangle” 和 “rectangle” 会被视为相同；
但 “obey the law” 和 “a signal to halt” 会被视为不同，即使它们可能与 stop sign 语义相关。
```

### 3. Route-behavior linkage

从实验 008 的 route table 中取：

```text
support source answer_mask weakening
support source union_mask weakening
```

按 `sample_id x run` 聚合，然后与：

```text
answer_mask_rank_damage
union_mask_rank_damage
answer_mask_margin_drop
union_mask_margin_drop
answer_mask_changed
union_mask_changed
```

合并。

相关性只做 exploratory：

```text
Pearson / Spearman
```

不做复杂显著性叙事。

## 运行完成性

behavior eval：

```text
sample-runs = 9
conditions = 8
done rows = 72
output csv = complete
```

generation eval：

```text
sample-runs = 9
conditions = 8
done rows = 72
error rows = 0
empty rows = 0
```

这说明 behavior/generation 侧没有格式崩坏、空答案或运行失败。

## Behavior results

### 1. 固定区域 mask 对 target rank / margin 的影响

| metric | n | mean | median | positive_rate | bootstrap 95% CI |
|---|---:|---:|---:|---:|---|
| answer-mask rank damage | 9 | +34.56 | +2 | 0.667 | [-0.11, +89.89] |
| union-mask rank damage | 9 | +20.89 | +13 | 0.889 | [+6.56, +35.33] |
| answer-mask margin drop | 9 | +2.69 | +1.75 | 0.778 | [-0.62, +6.53] |
| union-mask margin drop | 9 | +3.92 | +3.13 | 0.778 | [+0.92, +7.00] |

解释：

```text
union_mask 的行为损伤最稳定：
target rank 变差，margin 也变差，sample-run 口径的 bootstrap CI 不跨 0。

answer_mask 行为损伤方向为正，但受少数强样本和反向样本影响，CI 跨 0。
```

### 2. 与 random controls 的比较

strict `IoU<=0.05` 下只有 `2` 个 sample-runs 有 valid random controls，因此只能 diagnostic。

strict random4 下：

| metric | n | mean | median | positive_rate |
|---|---:|---:|---:|---:|
| answer rank over random4 | 2 | +116.38 | +116.38 | 0.5 |
| union rank over random4 | 2 | +16.88 | +16.88 | 1.0 |
| answer margin drop over random4 | 2 | +6.25 | +6.25 | 1.0 |
| union margin drop over random4 | 2 | +4.38 | +4.38 | 1.0 |

all-random diagnostic 下：

| metric | n | mean | median | positive_rate | bootstrap 95% CI |
|---|---:|---:|---:|---:|---|
| answer rank over all-random | 9 | +35.17 | +2.00 | 0.667 | [+1.75, +88.31] |
| union rank over all-random | 9 | +21.50 | +22.50 | 0.889 | [+8.25, +35.22] |
| answer margin drop over all-random | 9 | +2.13 | +2.30 | 0.778 | [-0.44, +4.73] |
| union margin drop over all-random | 9 | +3.35 | +2.55 | 0.778 | [+0.97, +5.63] |

解释：

```text
behavior 侧的 answer/union masks 确实比 random controls 更伤目标排名和 margin。
但 all-random 不是预注册 strict random4，因此只能作为 diagnostic / appendix。
```

## Generation results

### 1. 答案变化率

| condition | rows | changed_rows | changed_rate | empty_rows | error_rows |
|---|---:|---:|---:|---:|---:|
| answer_mask | 9 | 5 | 0.556 | 0 | 0 |
| relate_mask | 9 | 4 | 0.444 | 0 | 0 |
| union_mask | 9 | 6 | 0.667 | 0 | 0 |
| strict random valid rows | 8 | 2 | 0.250 | 0 | 0 |
| all random rows | 36 | 14 | 0.389 | 0 | 0 |

解释：

```text
answer/union evidence-region masks 经常改变 decoded answer；
union_mask 变化率最高；
没有 empty answer，也没有 error。
```

### 2. 代表性 case

| sample_id | run | clean answer | answer_mask answer | union_mask answer |
|---|---|---|---|---|
| okvqa_val_1927165 | B | obey the law | a signal to halt | a warning signal |
| okvqa_val_2131565 | B | ceramic | ceramic | asphalt |
| okvqa_val_2683965 | B | a rectangle | a square | rectangle |
| okvqa_val_3794755 | A | a laptop, a desktop computer, and a tablet | a collection of posters, a record player, and a laptop | a television |
| okvqa_val_3794755 | B | a laptop, monitor, and printer | a laptop | a television |
| okvqa_val_80655 | A | a baseball player swinging a bat | the baseball player is catching a ball | swinging the bat |

注意：

```text
这些是 decoded answer change，不直接等于 correctness change。
但它们说明区域遮挡不是只改变内部数值或 target token rank，而是会进入自然生成行为。
```

## Route-behavior linkage

### 1. 相关性

| route metric | behavior metric | n | Pearson | Spearman |
|---|---|---:|---:|---:|
| answer weakening | answer rank damage | 9 | +0.481 | +0.828 |
| union weakening | union rank damage | 9 | -0.354 | -0.167 |
| answer weakening | answer margin drop | 9 | +0.453 | +0.117 |
| union weakening | union margin drop | 9 | -0.496 | -0.577 |

解释：

```text
answer-mask route weakening 与 answer-mask rank damage 的方向关系较强，尤其 Spearman 为 +0.828。
但 union-mask route weakening 与 union behavior damage 不呈现稳定正相关。
```

这说明：

```text
路径变化与行为变化之间已经有局部桥接信号；
但它不是一个统一、线性的全样本解释。
```

### 2. case-level linkage 表

| sample_id | run | answer weakening | union weakening | answer rank damage | union rank damage | answer changed | union changed |
|---|---|---:|---:|---:|---:|---|---|
| okvqa_val_1740705 | A | +0.500 | +0.500 | +27 | +27 | False | False |
| okvqa_val_1927165 | B | +1.542 | +1.271 | +10 | +12 | True | True |
| okvqa_val_2131565 | B | -0.188 | -0.250 | 0 | +5 | False | True |
| okvqa_val_2683965 | B | +1.042 | +1.021 | +241 | +41 | True | False |
| okvqa_val_343215 | B | -1.063 | +1.813 | -17 | -14 | False | True |
| okvqa_val_3794755 | A | 0 | 0 | +47 | +57 | True | True |
| okvqa_val_3794755 | B | -0.125 | -0.125 | +1 | +46 | True | True |
| okvqa_val_5735275 | A | -0.250 | -0.125 | 0 | +1 | False | False |
| okvqa_val_80655 | A | +0.375 | +0.125 | +2 | +13 | True | True |

关键读法：

```text
okvqa_val_1927165 和 okvqa_val_2683965 是最清楚的 positive linkage cases。
okvqa_val_343215 是重要反例：union route weakening 很强，但 target rank 反而改善，decoded answer 变成 grizzly。
okvqa_val_3794755 行为损伤强，但 route weakening 接近 0，说明行为变化未必都由当前 traced support route 解释。
```

## 预期与实际偏差

预期：

```text
answer/union masks 导致 target rank 下降、margin 下降、decoded answer 更常变化；
route weakening 越大，behavior damage 越大。
```

实际：

```text
behavior damage 成立，尤其 union mask。
decoded answer change 成立，且无 empty/error。
answer-route weakening 与 answer-rank damage 有方向一致关系。
union-route weakening 与 union behavior damage 不稳定。
```

因此，behavior side 是 Stage 2A 的强补充，但 route-behavior linkage 只能写成 partial。

## 结论

本实验支持：

```text
evidence-region masks 不只影响内部 route；
它们也会伤害 target rank / margin，并改变 decoded answer。
```

但需要保守写：

```text
route weakening explains some behavior changes, especially under answer_mask；
it does not yet fully explain all union-mask behavior changes.
```

不能写：

```text
当前 traced nodes 已完整解释最终生成答案；
node intervention 已直接改变 decoded answer；
union-mask route weakening 与行为损伤在线性上完全对应。
```

## 对主 claim 的影响

这一步加强了主 claim 的行为关联部分：

```text
support routes are evidence-region-sensitive,
and evidence-region masks also produce rank/margin damage and decoded answer changes.
```

但 node-to-generation bridge 仍未完成。要升级为更强因果闭环，还需要 Stage 2B：

```text
clean generation
support source zeroing generation
nearest-control zeroing generation
answer-mask generation
```

## 后续动作

建议下一步分两条：

1. Stage 2A 收束：写一个 targeted replication verdict，总结 route-only 与 behavior/generation 结果，明确 `main partial replication + behavior support`。
2. Stage 2B 开始：选 3-5 个 positive linkage cases，做 node-to-generation bridge smoke。

优先 case：

```text
okvqa_val_1927165
okvqa_val_2683965
okvqa_val_80655
okvqa_val_3794755
```
