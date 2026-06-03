# Stage4-068 Qwen Feature Route Grouped Validation

## 目的

本实验验证 grouped feature route 是否真的有联合因果效果。关键点是：

```text
必须在同一次 forward 里同时干预 route 内多个 nodes。
```

不能把单节点效果相加后当作 route effect。

## 干预定义

### clean_route_zeroing

输入 clean image，在 route 中每个 node 的 layer/position 上，取该 feature 当前 activation，并沿该 feature 的 PLT decoder direction 做 subtract：

```text
hidden[layer, pos] -= activation(feature) * decoder_vector(feature)
```

如果剪 source route 比剪 control routes 更伤正确答案，就支持 gate 2。

### mask_route_restore

输入 masked image，先在 clean 和 masked 两次 forward 中取每个 node 的 feature activation difference：

```text
drop = clean_activation - masked_activation
```

然后在 masked forward 中同时补回：

```text
hidden[layer, pos] += drop * decoder_vector(feature)
```

如果补 source route 比补 control routes 更能恢复答案，就支持 gate 3。

如果 answer/union mask 下的 restore 效果强于 shifted/shuffled mask，就支持 gate 4。

### route_corrupt

方向性补充。在 clean image 下把 route 推向 masked state：

```text
hidden[layer, pos] -= (clean_activation - masked_activation) * decoder_vector(feature)
```

它只帮助解释方向，不作为主 decision gate。

## Controls

每个 source node 都构造三个同尺寸 route control：

```text
same_size_matched_feature_route_control：
  同一 position 上，activation/direct-effect 尽量匹配的其他 feature。

same_feature_random_position_route_control：
  同一 feature，换到其他 visual/answer-adjacent position。

random_active_route_control：
  同一 position 上随机 active feature。
```

这些 control routes 和 source route node 数相同，避免“剪得更多所以效果更大”的假阳性。

## 主指标

```text
route_zeroing_source_minus_controls
route_restore_source_minus_controls
route_real_minus_shifted
route_real_minus_shuffled
route_evidence_specificity
route_correct_minus_wrong
route_rank_effect
strict_missing_fraction
```

计算方向：

```text
zeroing effect:
  before_logit - after_logit

restore effect:
  after_logit - before_logit

rank effect:
  zeroing: after_rank - before_rank
  restore: before_rank - after_rank
```

正数表示 route 干预朝预期方向工作。

## 通过门槛

Primary 和 strict 同一个 topK 都需要：

```text
usable_routes >= 20
CI low > 0
positive_frac >= 0.5
source > controls
real restore > shifted/shuffled
correct target > wrong target
logit/rank 至少一个方向成立
```

如果 strict route 数不足 20，只能写 case-clustered，不升级成 paperpack-level claim。

## 与 Gemma 的关系

本实验和 Gemma 的统一点是判据：

```text
source/control
mask specificity
wrong-target specificity
behavior bridge
```

不同点是发现算法：

```text
Gemma: automatic source-tracing route first.
Qwen: route-first causal node discovery, then grouped feature route validation.
```

所以成功后写 Qwen-native feature-level route support，而不是 fully Gemma-style source tracing replication。

