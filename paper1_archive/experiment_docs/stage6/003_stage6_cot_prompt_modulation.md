# Stage6-003 CoT / Prompt Modulation

## 目的

这个实验不问：

```text
CoT 是否让 VQA 更准？
```

而问：

```text
CoT / visual-evidence instruction 是否改变 evidence-to-answer route 的强度、证据特异性或答案竞争结构？
```

这是 prompt 机制实验，不是 benchmark accuracy 实验。

## Prompt Families

四个 prompt family：

```text
B_direct:
  Direct short answer.

D_visual_only:
  Answer based on visual evidence, no step-by-step.

C_step_only:
  Think step by step, no explicit visual-evidence instruction.

A_step_visual:
  Think step by step and use visual evidence.
```

所有 prompt 必须统一最终答案格式：

```text
The answer is <short answer>.
```

主 endpoint 只看 final answer token 附近。

## 实验条件

每个 sample × question_variant × prompt_family 跑：

```text
clean
answer_mask
union_mask
shifted_mask
shuffled_mask
```

如果 compute 允许，再加：

```text
wrong_image
```

## 主指标

```text
evidence_specificity_by_prompt:
  answer/union mask effect 是否强于 shifted/shuffled。

source_control_by_prompt:
  source route/node 是否强于 controls。

correct_wrong_by_prompt:
  correct target 是否强于 wrong target。

competition_shift:
  wrong target / suppressor / competing answer 是否增强。

format_failure:
  CoT 是否造成 final answer 不可解析。
```

## 可能结论

### A. CoT enhances evidence route

```text
Step-by-step visual prompting increases evidence-specific route effects without format collapse.
```

这是最漂亮但不强求。

### B. CoT modulates but does not strengthen grounding

```text
CoT changes route strength or answer competition, but does not consistently increase evidence specificity.
```

这是最可能、也最安全的结论。

### C. CoT is format-confounded

```text
CoT mainly changes output format / answer token position, limiting route comparison.
```

这也有价值，因为它解释为什么不能把早期 A/B prompt 差异当作 grounding 结论。

## 禁止结论

```text
CoT improves VQA accuracy.
D_visual_only is better than B_direct.
A_step_visual is more grounded in general.
```

Stage6 只讨论 route modulation。

