# Stage 2L：Cross-Model 进一步验证计划

## 1. 这一步要回答什么

Stage 2I、2J、2K 之后，我们现在的跨模型结论已经比最初更强，但仍然要保持边界清楚。

目前最稳的说法是：

```text
Qwen2.5-VL 和 LLaVA 中也能观察到 evidence-mask-sensitive 的 hidden-state bridge。
其中 Qwen 的 matched-control specificity 更强，LLaVA 有 bridge 信号但更容易被 delta-matched control 吸收。
这些结果支持“类似现象不是 Gemma-only”，但还没有升级成 Gemma-style source-control causal route replication。
```

Stage 2L 的目的不是立刻宣称跨模型完整复现，而是继续验证下面这个更精确的问题：

```text
Qwen/LLaVA 中被关键证据区域影响的 hidden-state bridge，是否具有稳定的样本外复现、类型稳定性、位置特异性、控制特异性，并且能进一步连接到 target rank / decoded answer 的行为变化？
```

换句话说，Stage 2L 要判断：

```text
这是一个真实的跨模型机制线索，还是只是一组由 hidden delta、样本选择、answer-adjacent 汇聚位置共同造成的弱现象？
```

## 2. 当前证据分层

### 2.1 Gemma 主线

Gemma 仍然是唯一具备完整主链证据的模型：

```text
source tracing
node intervention
nearest/random controls
wrong-image sensitivity
region-mask sensitivity
rank/generation linkage
```

因此 Gemma 可以支撑主论文里的机制主结论。

### 2.2 Qwen 辅助跨模型线

Qwen 已经具备：

```text
readout-level evidence-region sensitivity
hidden-position restore/corrupt bridge
decoded bridge partial support
delta/activation matched controls 下大体仍然成立
typed analysis 中 symbol_text_reading 最强
```

Qwen 目前可以写成：

```text
stronger auxiliary cross-model evidence for hidden-state-level evidence-to-answer bridge
```

但还不能写成：

```text
Qwen 已经复现 Gemma source-control causal route
```

### 2.3 LLaVA 辅助跨模型线

LLaVA 已经具备：

```text
readout-level evidence-mask sensitivity
layer 15 hidden-position bridge
decoded bridge partial support
activation-matched control 下仍有优势
```

但 LLaVA 的弱点是：

```text
delta-matched controls 能解释相当一部分效果
matched-control specificity 只能写成 partial / heterogeneous
```

LLaVA 目前可以写成：

```text
LLaVA supports cross-model hidden bridge evidence, but specificity is weaker and partly explained by clean-vs-mask delta magnitude.
```

## 3. Stage 2L 总体策略

Stage 2L 分三层推进。

第一层是“样本外复现”：

```text
把 Qwen/LLaVA 的 hidden bridge 从当前 12 个样本扩到更大、类型更均衡的 localized candidate pack。
```

第二层是“更强控制”：

```text
继续使用 delta-matched、activation-matched、answer-adjacent-fixed、mask-shuffled、wrong-target controls，判断 best bridge 是否真的具有 specificity。
```

第三层是“行为桥接”：

```text
只对通过 hidden-position specificity 的 case 跑 decoded answer 或 first-token/rank bridge，避免在弱 case 上过度解释 generation。
```

## 4. 实验 2L-1：Case Panel 与失败模式面板

### 目的

先不新跑模型，直接利用 Stage 2I/2J/2K 已有结果，选出 figure-ready 和 failure-diagnostic case。

这一步要回答：

```text
哪些 case 最适合作为 Qwen 跨模型辅助证据？
哪些 case 暴露了 LLaVA 的 delta-control 弱点？
哪些 case 适合放进正文图，哪些只适合放附录？
```

### 输入

```text
doc/experiments/stage2/cross_model/stage2k_matched_control_explanation_case.csv
doc/experiments/stage2/cross_model/stage2i_selected_12_manifest.csv
doc/experiments/stage2/cross_model/stage2j_matched_control_case_table.csv
```

### 方法

对每个 sample / prompt / model 计算：

```text
restore_combo_minus_delta_combo
restore_combo_minus_activation_combo
corrupt_combo_minus_delta_combo
corrupt_combo_minus_activation_combo
decoded_bridge_changed_away_from_union
rank_restore
logit_restore
reasoning_operation / visual_structure / image_dependence
```

然后分成四类：

```text
Qwen strong-positive cases
Qwen weak/failure cases
LLaVA activation-positive but delta-absorbed cases
LLaVA strong-positive rare cases
```

### 成功标准

```text
至少选出 3 个 Qwen strong-positive cases。
至少选出 2 个 LLaVA diagnostic cases。
每个 case 都能说明它是正文候选、附录候选，还是失败分析候选。
```

### 预期产物

```text
doc/experiments/stage2/044_stage2l_cross_model_case_panel.md
doc/experiments/stage2/cross_model/stage2l_case_panel.csv
doc/experiments/stage2/cross_model/stage2l_case_panel_decision.json
```

## 5. 实验 2L-2：扩大样本复现

### 目的

验证 Stage 2I/2J 的结论是否只来自 12 个样本，还是能在更大 localized sample pack 中保持方向。

核心问题：

```text
Qwen 的 matched-control specificity 是否在更多样本上仍然成立？
LLaVA 的 delta-control weakness 是否仍然存在？
```

### 样本设计

优先从已有 candidate manifest 中扩展，不立刻要求新标注。

推荐规模：

```text
第一轮扩到 24 个 localized samples。
如果 24 个中有效样本不足，再扩到 36 个候选。
```

类型配比尽量固定：

```text
symbol_text_reading: 12
visual_readout: 8
scene_inference: 4
```

如果 scene_inference 的 evidence region 过于弥漫，可以保守降级为 appendix diagnostic，不进入 primary typed read。

### 输入

```text
existing localized masks
stage2i candidate manifest
Qwen layer 26
LLaVA layer 15
B_direct / D_visual_only prompts
```

### 条件

```text
clean
union_mask
answer_mask, if mask quality is compact enough
random / low-delta / delta-matched / activation-matched controls
```

### 主指标

```text
restore source-like bridge minus delta-matched-plus-answer-adjacent
restore source-like bridge minus activation-matched-plus-answer-adjacent
corrupt source-like bridge minus delta-matched-plus-answer-adjacent
corrupt source-like bridge minus activation-matched-plus-answer-adjacent
```

### 成功标准

Qwen：

```text
restore 和 corrupt 至少有 3/4 matched-control comparisons 为正。
restore 的 bootstrap CI 不跨 0。
symbol_text_reading 切片复现 stable_positive。
```

LLaVA：

```text
activation-matched comparison 稳定为正。
delta-matched comparison 若继续 weak，则写成 confirmed partial specificity。
如果 delta-matched 也转正，才升级 LLaVA claim。
```

### 预期产物

```text
doc/experiments/stage2/045_stage2l_cross_model_expanded_replication.md
doc/experiments/stage2/cross_model/stage2l_expanded_manifest.csv
doc/experiments/stage2/cross_model/stage2l_qwen_expanded_patch.csv
doc/experiments/stage2/cross_model/stage2l_llava_expanded_patch.csv
doc/experiments/stage2/cross_model/stage2l_expanded_summary.csv
doc/experiments/stage2/cross_model/stage2l_expanded_bootstrap.csv
doc/experiments/stage2/cross_model/stage2l_expanded_decision.json
```

## 6. 实验 2L-3：Mask-shuffled 与 wrong-target negative controls

### 目的

Stage 2J 的 matched controls 控制了 hidden delta 和 activation norm，但还没有完全排除：

```text
任何强遮挡都会造成类似 bridge。
任何 target token 都能被 patch 恢复。
```

因此需要负控制。

### 负控制 1：mask-shuffled

方法：

```text
对同一张图使用其他 sample 的 answer/union mask，或者在同图内把 mask 随机平移到非证据区域。
保持 mask 面积接近。
```

预期：

```text
真实 evidence mask 的 bridge 应强于 mask-shuffled bridge。
```

### 负控制 2：wrong-target token

方法：

```text
把 target answer token 换成同样长度或同样频率附近的错误答案 token。
比较 hidden bridge 对 correct target 和 wrong target 的 restore/corrupt。
```

预期：

```text
如果 bridge 真的连接 evidence-to-answer，correct target 的 restore 应强于 wrong target。
```

### 成功标准

```text
Qwen correct-target bridge > wrong-target bridge。
Qwen evidence-mask bridge > mask-shuffled bridge。
LLaVA 若只在 activation-control 下成立，则继续写成 partial。
```

### 预期产物

```text
doc/experiments/stage2/046_stage2l_negative_controls.md
doc/experiments/stage2/cross_model/stage2l_negative_control_summary.csv
doc/experiments/stage2/cross_model/stage2l_negative_control_decision.json
```

## 7. 实验 2L-4：Evidence-to-answer 汇聚位置分析

### 目的

Stage 2I/2J 显示 best bridge 通常是：

```text
top_hidden_delta visual positions + answer-adjacent text positions
```

这说明跨模型桥可能不是“纯视觉 token 直接决定答案”，而是：

```text
视觉证据信号先在 image positions 变化，然后在 answer-adjacent / assistant-prefix 附近汇聚，最终影响 target answer。
```

这一步要验证这个解释。

### 位置组

```text
image_source_like_only
answer_adjacent_only
image_source_like_plus_answer_adjacent
question_text_only
assistant_prefix_only
last_prompt_token_only
```

### 指标

```text
logit_restore_vs_union
rank_restore_vs_union
logit_corrupt_vs_clean
rank_corrupt_vs_clean
gap_closure
```

### 成功标准

```text
image + answer-adjacent 明显强于 image-only 和 answer-adjacent-only。
如果 answer-adjacent-only 已经接近 full combo，则说明跨模型 bridge 更多是 answer-local aggregation，而不是纯 visual-position specificity。
```

### 预期产物

```text
doc/experiments/stage2/047_stage2l_evidence_to_answer_bridge.md
doc/experiments/stage2/cross_model/stage2l_bridge_position_decomposition.csv
doc/experiments/stage2/cross_model/stage2l_bridge_position_decision.json
```

## 8. 实验 2L-5：Decoded generation only-on-passing-cases

### 目的

避免在弱 case 上强行解释 generation。只对 2L-1/2L-2 中通过 specificity 的 case 做短生成。

### 输入 case

```text
Qwen strong-positive cases: 3-5
LLaVA diagnostic-positive cases: 2-3
```

### 条件

```text
clean generation
union_mask generation
hidden patch restore generation
matched-control patch generation
```

### 指标

```text
answer_changed_from_union
answer_returns_to_clean
format_prefix_ok
empty_or_error
target first-token rank
target first-token logit
```

### 成功标准

```text
hidden patch restore 比 matched-control patch 更常把答案从 union_mask 方向拉回 clean。
如果 decoded answer 不变，只写 first-token / rank bridge，不写 generation-level causal bridge。
```

### 预期产物

```text
doc/experiments/stage2/048_stage2l_decoded_generation_bridge.md
doc/experiments/stage2/cross_model/stage2l_decoded_generation_bridge.csv
doc/experiments/stage2/cross_model/stage2l_decoded_generation_decision.json
```

## 9. 实验 2L-6：Qwen feature/source tracing 工程预研

### 目的

如果想把 Qwen 从 hidden-state bridge 推到更接近 Gemma 主线，最终需要 feature/source tracing。

这一步不是立刻复现 Gemma，而是检查工程上能否做：

```text
Qwen CLT feature activation
feature-level ablation
feature-level patch
feature matched control
target-rank linkage
```

### 风险

```text
Qwen ReplacementModel adapter 仍未完成。
CLT feature readout 已经可行，但 feature-level causal intervention 之前不稳定。
Qwen token/image position mapping 比 LLaVA 更复杂。
```

### 最小成功标准

```text
对一个 Qwen strong case，能稳定在同一 layer/position 读取 top evidence-sensitive CLT features。
能对这些 features 做 scale-down / zeroing，并读 target logit/rank。
control feature 的影响弱于 evidence-sensitive feature。
```

### 预期产物

```text
doc/experiments/stage2/049_stage2l_qwen_feature_source_tracing_prestudy.md
doc/experiments/stage2/cross_model/stage2l_qwen_feature_intervention_smoke.csv
doc/experiments/stage2/cross_model/stage2l_qwen_feature_intervention_decision.json
```

## 10. 是否需要新标注

短期不需要。

Stage 2L-1、2L-3、2L-4 可以直接用现有 12 个样本和已有 mask。

Stage 2L-2 如果扩到 24 / 36 个样本，优先使用已有 localized masks。只有当下面情况出现时才需要新标注：

```text
symbol_text_reading + visual_readout 的 compact evidence samples 不足。
已有 mask 和当前 candidate backbone overlap 不够。
answer/union mask 质量不能支持 shuffled/random control。
```

如果需要补标，建议只补：

```text
8-12 个 compact answer-region 样本
优先 symbol_text_reading
其次 visual_readout
暂缓 diffuse scene_inference
```

## 11. Claim 更新规则

如果 Stage 2L 成功，可以升级到：

```text
Across Gemma, Qwen, and LLaVA, evidence-region perturbations reveal answer-relevant hidden-state bridges, with Qwen showing the strongest auxiliary matched-control specificity and LLaVA showing partial but weaker specificity.
```

仍然不能升级到：

```text
Cross-model source-control causal routes are fully replicated.
```

除非后续完成：

```text
Qwen/LLaVA source tracing
feature/node intervention
matched non-source controls
region-mask sensitivity
rank/generation linkage
```

## 12. 推荐执行顺序

```text
1. Stage 2L-1：case panel 与失败模式面板。
2. Stage 2L-3：negative controls，因为成本低、信息量高。
3. Stage 2L-4：evidence-to-answer 汇聚位置分析。
4. Stage 2L-2：扩大到 24 / 36 个样本。
5. Stage 2L-5：只在通过 specificity 的 case 上跑 decoded generation。
6. Stage 2L-6：Qwen feature/source tracing 工程预研。
```

这个顺序的理由是：

```text
先用已有数据把“什么样的 case 成立、什么样的 case 失败”说清楚；
再用负控制排除最常见的替代解释；
再扩大样本；
最后才做最贵的 feature/source tracing 工程。
```

