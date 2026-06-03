# Stage6-005 Type-Sliced Route Visibility

## 目的

这个分析把 Stage6 结果按问题类型切开，形成一个探索性二级结论：

```text
route 清晰度可能依赖 VQA 问题类型。
```

## 类型

基于 paperpack72 当前字段：

```text
visual_readout:
  44 samples / 88 primary prompt-runs

symbol_text_reading:
  4 samples / 8 primary prompt-runs

compact_scene_inference:
  23 samples / 46 primary prompt-runs

mixed_localized:
  1 sample / 2 prompt-runs, diagnostic only
```

## 分析指标

按类型报告：

```text
route availability
clean route source/control
answer/union mask drop
real-vs-shifted/shuffled
correct-vs-wrong
question rewrite stability
CoT modulation magnitude
format failure rate
decoded answer change rate
```

## 预期解释

允许写：

```text
Compact visual-readout and symbol/text-reading cases show clearer route visibility in this exploratory pack.
```

如果 symbol_text_reading 数量太少：

```text
Symbol/text-reading is case-level diagnostic due to small n.
```

如果 scene inference 很弱：

```text
Broad scene-inference cases appear more heterogeneous and less localized under current route metrics.
```

禁止写：

```text
All visual_readout questions have clear routes.
Symbol reading is universally easier.
Scene inference has no evidence route.
```

