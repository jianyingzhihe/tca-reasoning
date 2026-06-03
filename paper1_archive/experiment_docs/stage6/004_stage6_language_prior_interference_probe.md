# Stage6-004 Language Prior Interference Probe

## 目的

这是 optional exploratory probe，不作为 Stage6 主二级结论。

它回答：

```text
如果给模型一个语言先验或常见错误答案，视觉证据 route 会不会变弱？
错误答案竞争 route 会不会变强？
```

这个方向更接近后续 Paper 2 的 hallucination / arbitration 主题，所以本轮只做小规模诊断。

## Prompt 设计

对每个样本构造一个 distractor answer。

Prompt 模板：

```text
A common guess might be <distractor>, but answer based only on the image.
The answer is
```

对照 prompt：

```text
Answer based only on the image.
The answer is
```

## 主指标

```text
target_route_effect_change:
  语言先验干扰后，正确答案 route 是否变弱。

wrong_target_route_effect_change:
  distractor / wrong answer route 是否变强。

evidence_specificity_change:
  answer/union mask specificity 是否下降。

decoded_answer_flip:
  clean 条件下是否更容易输出 distractor。
```

## 解释边界

如果看到干扰：

```text
Language priors can modulate or compete with visual evidence routes in VQA.
```

不能写：

```text
The model hallucinates because of this route.
```

如果没看到干扰：

```text
This small probe did not find strong language-prior interference in selected compact visual cases.
```

不能写：

```text
Language priors do not affect VQA routes.
```

