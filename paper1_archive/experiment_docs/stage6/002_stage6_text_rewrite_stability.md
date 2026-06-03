# Stage6-002 Text Rewrite Stability

## 目的

这个实验回答一个审稿人很可能会问的问题：

```text
你们看到的 evidence-to-answer route，会不会只是某个固定问题措辞触发出来的 prompt artifact？
```

我们不试图证明所有文本改写都稳定，只探索：

```text
在视觉证据明确、答案区域紧凑的 VQA 样本中，等价问题改写是否比视觉证据遮挡更少破坏 route effect。
```

## 实验设计

每个样本使用 3 个 question variants：

```text
original_question
paraphrase_1
paraphrase_2
```

改写要求：

```text
语义不变
答案不变
不加入新视觉线索
不加入常识暗示
不改变答案粒度
```

每个 question variant 跑：

```text
clean
answer_mask
union_mask
shifted_mask
shuffled_mask
```

optional：

```text
wrong_image
```

## 主指标

```text
route_effect_clean_corr:
  不同 question rewrites 下 clean route effect 的相关性。

route_effect_masked_corr:
  不同 question rewrites 下 answer/union mask route effect 的相关性。

real_minus_shifted / real_minus_shuffled:
  真实证据区域遮挡是否比控制 mask 更影响 route。

rewrite_delta_vs_mask_delta:
  文本等价改写造成的变化，是否小于视觉证据遮挡造成的变化。

decoded_answer_stability:
  clean 条件下答案是否仍可读且格式稳定。
```

## 预期可写结论

如果成立：

```text
In compact visually grounded VQA cases, route effects are relatively stable under meaning-preserving question rewrites but sensitive to visual evidence perturbation.
```

中文：

```text
在证据区域明确的问题里，换一种问法不会像遮掉图像证据那样破坏 route。
```

如果不成立：

```text
Question wording substantially modulates route visibility, suggesting prompt-sensitive route access rather than wording-invariant visual routing.
```

这也不是失败，它说明：

```text
route 可能同时受图像证据和文本入口调节。
```

## 失败/混淆项

必须单独报告：

```text
format failure
empty answer
answer tokenization shift
paraphrase changes answer granularity
clean decoded answer already wrong
route unavailable for rewritten question
```

