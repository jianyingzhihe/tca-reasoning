# Stage4-067 Qwen Feature Route Manifest

## 目的

本文记录 feature route manifest 的构造规则。核心目标是防止两个常见错误：

```text
错误 1：先看 strict 再挑 route。
错误 2：把少数强节点重新包装成 route，但没有 frozen exact confirmation。
```

所以 manifest 必须只从 primary route-first result 构造，再生成 strict exact confirmation manifest。

## 数据来源

输入文件：

```text
doc/experiments/stage4/cross_model/stage4_qwen_route_first_routefirst_v1_route_candidates.csv
doc/experiments/stage4/cross_model/stage4_qwen_route_first_primary_manifest.csv
doc/experiments/stage4/cross_model/stage4_qwen_route_first_strict_manifest.csv
```

候选过滤：

```text
pack = primary
mode = full
route_first_234 = 1
```

这意味着 route 的候选节点不是从 evidence-sensitive 条件出发，而是先满足：

```text
2 = clean source > controls
3 = restore source > controls
4 = real restore > shifted/shuffled
```

然后 route-level validation 再检查：

```text
1 = evidence-sensitive
5 = correct > wrong
```

## Route 构造

Route base key：

```text
sample_id + prompt_name
```

同一 base key 内，按 frozen route score 排序：

```text
route_node_score
  = clean_source_minus_controls
  + restore_source_minus_controls
  + real_minus_shifted
  + real_minus_shuffled
  + evidence_specificity
  + max(clean_correct_minus_wrong, restore_correct_minus_wrong)
```

Route topK：

```text
4,8,16,32,64
```

每条 route manifest row 保存：

```text
route_id
route_base_id
sample_id
prompt_name
topk
route_node_count
node_candidate_ids
node_layers
node_source_positions
node_feature_ids
node_zeroing_modes
node_scores
target_token_id
target_token
wrong_token_id
wrong_token
image/mask/question/answer fields
```

## Strict 构造

Strict manifest 不重新挑 route。它只把 primary route 中的 candidate_id 映射到 strict frozen candidate。

Strict missing 只记录，不 fallback：

```text
strict_missing_count
strict_missing_fraction
missing_candidate_ids
```

如果 strict 缺失太多，结论只能写：

```text
feature route case-clustered / strict unresolved
```

不能升级成 paperpack-level route support。

## 当前 dry-run 预期

基于 Stage4-060 已完成结果，dry-run 应接近：

```text
primary nodes: 2029
primary base routes: 91
primary route rows: 455
strict exact nodes: 1930
strict route rows: 435
primary unique samples: about 50
strict unique samples: about 48
strict missing fraction mean: 0.0 for materialized strict routes
```

这说明 route-level validation 的样本规模足够先跑 smoke，然后再跑 primary full / strict full。

