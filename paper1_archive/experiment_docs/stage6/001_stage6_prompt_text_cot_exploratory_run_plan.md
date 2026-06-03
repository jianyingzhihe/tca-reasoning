# Stage6 Prompt/Text/CoT Exploratory Mechanism Run Plan

## Summary

Stage5 已经用于 CLT heterogeneity mapping，因此本轮探索性 prompt/text/CoT 机制线编号为 Stage6。

Stage6 不承担新的主 claim，也不要求证明一个必须成立的结论。它的定位是围绕已成立主线补二级结果：

```text
主线:
  多模态模型中存在 evidence-to-answer 内部因果流。
  Gemma 是完整 sparse source-tracing route。
  Qwen 是 hidden route 稳定、feature node-level route-first 成立，但 grouped feature route 尚未闭合。

Stage6 目标:
  探索这条 evidence-to-answer 机制是否主要锚定视觉证据，
  以及它如何被问题文本改写、visual-evidence prompt、CoT prompt 调节。
```

最重要的边界：

```text
Stage6 不证明 CoT 更好。
Stage6 不证明某个 prompt 更好。
Stage6 不试图替代 Gemma/Qwen 主机制结论。
Stage6 只回答 prompt/text 是否调节已经发现的 evidence-to-answer route。
```

## Secondary Claims To Test

### Secondary Claim A: Text rewrite stability

目标问题：

```text
我们的 route 是不是固定问题措辞触发的 text-template artifact？
```

探索性结论模板：

```text
For visually grounded VQA cases, evidence-to-answer route effects are more stable under meaning-preserving question rewrites than under visual evidence perturbation.
```

中文说法：

```text
换一种等价问法，route 应该大体还在；
遮真实证据区域或换错图，route 应该明显受损。
```

### Secondary Claim B: CoT / visual prompt modulation

目标问题：

```text
CoT 是增强视觉证据路径，还是引入语言侧推理/竞争路径？
```

探索性结论模板：

```text
CoT and visual-evidence instructions modulate route strength and competition, but do not necessarily strengthen visual grounding.
```

中文说法：

```text
CoT 可能改变路径分配，不一定让模型更依赖视觉证据。
```

### Secondary Claim C: Route visibility by question type

目标问题：

```text
什么样的 VQA 问题最容易出现清楚的 evidence route？
```

探索性结论模板：

```text
Compact visual-readout and symbol/text-reading cases show clearer evidence-route signatures than broad scene-inference cases.
```

中文说法：

```text
答案集中在图像某个明确区域时，route 更清楚；场景推断更分布、更异质。
```

## Current Assets And Constraints

已有 paperpack72：

```text
primary prompt-runs = 144
samples = 72
prompts = B_direct / D_visual_only
```

样本类型分布：

```text
visual_readout = 44 samples / 88 prompt-runs
compact_scene_inference = 23 samples / 46 prompt-runs
symbol_text_reading = 4 samples / 8 prompt-runs
mixed_localized = 1 sample / 2 prompt-runs
```

因此 Stage6 的分层默认不是均衡大样本，而是 exploratory stratified small pack：

```text
visual_readout: 6-8 samples
symbol_text_reading: all available high-quality samples, target 4
compact_scene_inference: 4-6 samples
mixed_localized: optional diagnostic only
```

Qwen 可用强候选：

```text
Stage4-060 route-first evidence-gold nodes:
  primary route_first_evidence_gold = 1334
  strict route_first_evidence_gold = 1295

Stage4-066 feature route:
  grouped route not fully closed
  but route evidence sensitivity and correct > wrong are stable
```

Gemma 可用主线：

```text
Gemma3-PLT source-tracing route is the sparse baseline.
Stage6 可以优先从 Gemma 做 text/Cot route robustness。
Qwen 只做 node-level / hidden-level robustness，不强行做 grouped feature route。
```

## Experiment Flow

### Step 0: Build Stage6 exploratory sample pack

选择 12-16 个样本：

```text
优先:
  strong image_dependence_tier
  compact answer/union mask
  decoded answer format stable
  Stage4 route-first evidence_gold 或 Gemma source route 可用

避免:
  answer 本身需要大量外部常识
  mask 太大或太散
  原始 clean answer 已不稳定
  prompt 输出格式容易崩
```

每个样本生成：

```text
original_question
paraphrase_1
paraphrase_2
```

改写规则：

```text
语义不变
不加入新线索
不暗示答案
不改变答案粒度
尽量保持短问句
```

### Step 1: Text rewrite stability

对每个 sample × question_variant 跑：

```text
clean
answer_mask
union_mask
shifted_mask
shuffled_mask
optional wrong_image
```

核心指标：

```text
route_effect_correlation_across_rewrites
answer_mask_drop
union_mask_drop
real_minus_shifted
real_minus_shuffled
source_minus_controls
correct_minus_wrong
decoded_answer_change
format_failure
```

### Step 2: Prompt / CoT modulation

Prompt families 固定为四个：

```text
B_direct:
  direct short-answer prompt.

D_visual_only:
  answer from visual evidence, but no step-by-step.

C_step_only:
  think step by step, but no explicit visual-evidence instruction.

A_step_visual:
  think step by step and answer from visual evidence.
```

所有 prompt 必须强制最终答案格式：

```text
The answer is <short answer>.
```

分析时只评价最终答案 token 附近，不把中间 CoT 文本作为主 endpoint。

### Step 3: Language-prior interference probe

只做 optional 小实验，不作为主二级 claim。

目的：

```text
测试语言先验/常识提示会不会削弱视觉证据路径，或增强 wrong-answer competition route。
```

Prompt 例子：

```text
A common guess might be <distractor>, but answer based only on the image. The answer is
```

该实验只在 Stage6 text/Cot smoke 结果干净后运行。

### Step 4: Type-sliced analysis

按 question type 分析：

```text
visual_readout
symbol_text_reading
compact_scene_inference
```

输出不是“哪类一定更好”，而是：

```text
which type has clearer route visibility
which type has more prompt sensitivity
which type has more language-prior vulnerability
```

## Decision Criteria

Stage6 是探索性，不设置 paper-level 必须通过的 hard claim。但每个二级结论需要可判定状态：

```text
text_rewrite_stability_supported:
  equivalent rewrites preserve route effects more than visual perturbations do.

text_rewrite_route_unstable:
  equivalent rewrites substantially change route effects, suggesting prompt wording sensitivity.

cot_visual_modulation_supported:
  prompt family significantly changes route strength / evidence specificity / competition without format collapse.

cot_format_confounded:
  CoT primarily changes output format or answer token position, making route comparison unreliable.

type_route_visibility_gradient_supported:
  compact visual/text-reading cases show clearer route metrics than scene-inference cases.

exploratory_inconclusive:
  signals are mixed, small-n, or dominated by a few samples.
```

Stage6 最强允许写法：

```text
Evidence-to-answer routes are modulated by prompt/text conditions and appear more visually anchored than wording-anchored in compact VQA cases.
```

禁止写法：

```text
CoT improves visual grounding.
Prompt D is better than prompt B.
Question rewrite invariance is proven generally.
All VQA routes are text-robust.
```

## Implementation Changes

新增 Stage6 文档：

```text
001_stage6_prompt_text_cot_exploratory_run_plan.md
002_stage6_text_rewrite_stability.md
003_stage6_cot_prompt_modulation.md
004_stage6_language_prior_interference_probe.md
005_stage6_type_sliced_route_visibility.md
006_stage6_exploratory_secondary_claims_verdict.md
```

建议新增脚本：

```text
build_stage6_prompt_text_pack.py
run_stage6_prompt_text_cot_remote.py
analyze_stage6_prompt_text_cot.py
```

Artifact prefix：

```text
stage6_prompt_text_cot_*
```

输出位置：

```text
doc/experiments/stage6/cross_model/
```

## Test Plan

Local preflight：

```text
py_compile new scripts
sample pack rows > 0
all image/mask paths exist
question rewrites non-empty
prompt families preserve final-answer format
no Gemma node ids used for Qwen candidate selection
strict/follow-up not used to tune exploratory thresholds
```

Smoke：

```text
4 samples
2 question variants
2 prompt families: B_direct, A_step_visual
conditions: clean, answer_mask, shuffled_mask
```

Full exploratory：

```text
12-16 samples
3 question variants
4 prompt families
conditions: clean, answer_mask, union_mask, shifted_mask, shuffled_mask
optional wrong_image
```

Acceptance for analysis：

```text
format failure reported
decoded answer stability reported
route metrics reported by question variant and prompt family
type-sliced summary reported
small-n / case-clustered warnings explicit
```

## Assumptions

```text
Stage6 is exploratory, not a new mainline proof.
Stage5 CLT docs remain untouched.
Primary target is paper secondary results, not new strongest mechanism claim.
Gemma can be used as sparse-route baseline.
Qwen should be analyzed at hidden-level and feature node-level; grouped route-level is not forced.
No new human evidence masks are added.
Question rewrites can be manually curated or generated then manually audited before remote run.
```

