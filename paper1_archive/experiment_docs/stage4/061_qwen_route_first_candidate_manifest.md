# Stage4-061 Qwen Route-First Candidate Manifest

## 目的

记录 route-first manifest 的构造规则，避免把 evidence-first 阈值重新带回本轮。

## 构造规则

保留：

```text
sample_id / prompt_name / layer / source_pos / source_feature_id
target_token_id
image_filename
clean_target_rank <= 10
image/mask paths exist
```

不要求：

```text
real_drop_best >= 20
evidence_specificity >= 20
correct_minus_wrong_contribution > 0
target_contribution > 0
```

## 输出

```text
stage4_qwen_route_first_primary_manifest.csv
stage4_qwen_route_first_strict_manifest.csv
stage4_qwen_route_first_manifest_summary.json
```

Strict frozen manifest 只由 primary analysis 的 `2+3+4` candidates 生成。

## 当前 manifest 结果

`2026-05-28 19:56 CST` 已生成 base manifest：

```text
primary rows = 23040
strict rows = 22080
primary unique samples = 50
strict unique samples = 48
```

层分布：

```text
L10 = 12288
L11-L17 = each 1536
```

L10 候选更多是预期结果，因为它同时包含 all-layer broad screen 和 middle-dense L10 三个 position group。Stage4-060 不做 sample/layer cap；集中性只在 `concentration.csv` 中报告，不作为剔除理由。

所有候选来自 Qwen artifacts：

```text
stage4_qwen_all_layer_bounded_exhaustive_primary_full_L10-L17_candidates.csv
stage4_qwen_middle_dense_primary_full_L10_*_candidates.csv
```

未读取 Gemma node id 或 Gemma route map。
