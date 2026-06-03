# Stage4-053 Qwen Evidence-Specific Candidate Manifest

## 目的

把 L10-L17 粗扫和 L10 dense 中的强 evidence-specific PLT candidates 固化成 primary/strict 可复验 manifest，避免继续被全体平均稀释。

## 候选规则

- `real_drop_best >= 20`
- `evidence_specificity >= 20`
- `target_contribution > 0`
- `correct_minus_wrong_contribution > 0`
- `clean_target_rank <= 10`
- 所有通过阈值的候选都保留为 `include_pool=1`
- 第一轮验证子集标记为 `include_main=1`
- `include_main=1` 每个 sample/prompt/layer 最多 top2，每层默认 top20，最多 top24

## Manifest 字段

Manifest 保留 source node identity、mask drop 原始数值、target/wrong contribution、rank bucket、image/mask paths、question/answer text。Strict manifest 不重新发现节点，只替换成 strict paperpack 的 prompt-run metadata。第一轮跑 `include_main=1`，必要时跑完整 `include_pool=1`。

## 验收

- 候选只来自 Qwen artifacts。
- L10-L17 均单独计数。
- image 与 answer/union/shifted/shuffled mask path 全存在。
- 输出 sample/prompt/layer cap 后的分布 summary。

## 当前 manifest 结果

生成时间：2026-05-28 10:39 CST。

```text
pool candidates after filter: 2147
primary pool rows: 2147
strict pool rows: 2145
primary main rows: 160
strict main rows: 160
unique primary samples: 44
main unique samples: 25
```

Layer pool counts:

```text
L10: 146
L11: 219
L12: 260
L13: 269
L14: 185
L15: 417
L16: 257
L17: 394
```

Main validation subset keeps 20 candidates per layer. This is not a discovery cap: the full pool remains available for `--selection all`.
