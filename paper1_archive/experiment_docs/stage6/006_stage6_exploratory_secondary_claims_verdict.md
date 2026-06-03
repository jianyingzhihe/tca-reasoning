# Stage6-006 Exploratory Secondary Claims Verdict

## 当前状态

Stage6 尚未运行。本文用于记录后续 exploratory secondary results。

当前预注册二级结论候选：

```text
1. text_rewrite_stability_supported
2. cot_visual_modulation_supported
3. type_route_visibility_gradient_supported
4. language_prior_interference_diagnostic
```

## 运行后填写

### Text Rewrite Stability

```text
status: pending
sample count:
question variants:
route stability:
visual perturbation contrast:
format failures:
interpretation:
```

### CoT / Prompt Modulation

```text
status: pending
prompt families:
evidence specificity change:
source/control change:
correct/wrong change:
competition/suppressor change:
format failures:
interpretation:
```

### Type-Sliced Visibility

```text
status: pending
visual_readout:
symbol_text_reading:
compact_scene_inference:
small-n warnings:
interpretation:
```

### Language Prior Probe

```text
status: optional_pending
target route effect:
wrong/distractor route effect:
decoded answer flip:
interpretation:
```

## 最终写法模板

如果 Stage6 支持 text stability：

```text
In compact VQA cases, evidence-to-answer route effects are more robust to meaning-preserving question rewrites than to visual evidence perturbation.
```

如果 Stage6 支持 prompt modulation：

```text
CoT and visual-evidence prompts modulate route strength and answer competition, but do not necessarily strengthen visual grounding.
```

如果 Stage6 支持 type gradient：

```text
Route visibility is strongest in compact visual-readout / text-reading cases and weaker or more heterogeneous in scene-inference cases.
```

总边界：

```text
Stage6 exploratory results supplement the main mechanism claim.
They do not replace the Gemma/Qwen route evidence and should not be stated as universal prompt or CoT claims.
```

