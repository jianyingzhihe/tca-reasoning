# Stage4-069 Qwen Feature Route Replication Verdict

## 当前状态

本文件用于记录 Stage4-066 feature route-level replication 的最终结论。

运行前的已知基础：

```text
Stage4-060:
  qwen_route_first_full_supported

含义:
  Qwen 中层 PLT 中已经有一批单节点满足 1+2+3+4+5。
```

Stage4-066 要补的是：

```text
这些节点按 sample/prompt 成组后，是否能作为一条 feature route 被整体干预并复现同样 gates。
```

## 待填结果

Smoke：

```text
status: ok
usable routes: 9
raw rows: 360
skipped: 0
```

Primary full：

```text
status: ok
usable routes: 455
raw rows: 18200
unique samples: 50
best-supported topK pattern:
  topK 32/64 pass clean zeroing source > controls
  topK 8 passes real > shifted and real > shuffled
  all topK pass evidence sensitivity and correct > wrong
failed route-level gates:
  no single pre-fixed topK passes all 1+2+3+4+5 aggregate gates
  restore source > controls remains weak
  real > shuffled often has positive mean but positive fraction below 0.5
```

Strict full：

```text
status: ok
usable routes: 435
raw rows: 17400
unique samples: 48
strict missing fraction mean: 0.0
best-supported topK pattern:
  topK 32/64 pass clean zeroing source > controls
  topK 8 passes real > shifted and real > shuffled
  all topK pass evidence sensitivity and correct > wrong
failed route-level gates:
  no single pre-fixed topK passes all 1+2+3+4+5 aggregate gates
  restore source > controls remains weak
  route rank effect has positive mean but positive fraction below 0.5
```

Final decision JSON:

```text
qwen_route_first_nodes_supported_route_unresolved
```

## 关键数字

Primary full examples:

```text
topK 8:
  route_evidence_specificity mean = 107.34, CI low = 94.54, positive_frac = 1.00
  route_real_minus_shifted mean = 0.0439, CI low = 0.0251
  route_real_minus_shuffled mean = 0.0257, CI low = 0.0054
  route_correct_minus_wrong mean = 0.0540, CI low = 0.0388
  route_restore_source_minus_controls mean = 0.0138, CI low = -0.0008

topK 64:
  route_zeroing_source_minus_controls mean = 0.0687, CI low = 0.0398
  route_evidence_specificity mean = 187.01, CI low = 154.81
  route_correct_minus_wrong mean = 0.0869, CI low = 0.0659
  route_restore_source_minus_controls mean = 0.0062, CI low = -0.0070
```

Strict full reproduces the same pattern:

```text
topK 8:
  route_evidence_specificity mean = 110.08, CI low = 97.46
  route_real_minus_shifted mean = 0.0430, CI low = 0.0223
  route_real_minus_shuffled mean = 0.0268, CI low = 0.0084
  route_correct_minus_wrong mean = 0.0550, CI low = 0.0393
  route_restore_source_minus_controls mean = 0.0125, CI low = -0.0023

topK 64:
  route_zeroing_source_minus_controls mean = 0.0670, CI low = 0.0366
  route_evidence_specificity mean = 191.23, CI low = 157.42
  route_correct_minus_wrong mean = 0.0866, CI low = 0.0654
  route_restore_source_minus_controls mean = 0.0060, CI low = -0.0075
```

## 结论模板

如果 primary/strict 都通过：

```text
Qwen supports a Qwen-native feature-level evidence-to-answer route under route-first discovery.
```

如果 grouped topK 通过但小 topK 弱：

```text
Qwen feature route is distributed: it becomes causal/evidence-linked only when multiple PLT features are patched together.
```

如果方向成立但样本覆盖不足：

```text
Qwen has case-clustered feature-route support, but not enough coverage for a paperpack-level route claim.
```

本轮实际属于：

```text
Qwen route-first nodes are supported, but grouped feature route remains unresolved under the current public Qwen-PLT basis and intervention operator.
```

更白话地说：

```text
Qwen 的单个 feature 节点已经能严格证明“像答案传输点”，而且很多也受证据区域影响；
但是把这些点简单打包成一条 feature route 一起补/剪，还没有稳定闭合成 Gemma 那种 route-level causal circuit。
```

## Claim 边界

无论本轮是否成功，都不写：

```text
Qwen fully replicates Gemma-style automatic source tracing.
```

成功时最多写：

```text
Qwen-native feature-level evidence-to-answer route support.
```

失败时也不能写：

```text
Qwen has no evidence-to-answer mechanism.
```

因为 hidden residual route 和 Stage4-060 route-first node-level evidence 已经成立。
