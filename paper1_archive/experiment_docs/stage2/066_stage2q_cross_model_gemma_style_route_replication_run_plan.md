# Stage 2Q：Qwen / LLaVA Gemma-Style 完整主线复现实验计划

## 1. Summary

Stage 2Q 的目标是回答一个更硬的问题：

```text
Qwen / LLaVA 是否能形成接近 Gemma 主线的完整机制链？
```

这里的“完整主线”不是指单个 hidden patch 或 feature patch 成功，而是尽可能复现 Gemma 的证据结构：

```text
source discovery / source-like route selection
→ source vs matched control intervention
→ evidence region mask sensitivity
→ real mask > shifted / shuffled mask
→ correct target > wrong target
→ target rank / first-token / decoded answer linkage
```

当前状态：

```text
Gemma:
  完整主线已成立，是主模型。

Qwen:
  hidden-state bridge 已成立。
  feature-level bridge 已在 heldout 上成立。
  approximate source-control probe 已在 heldout 上成立。
  还没有真正完成 Gemma-style source tracing adapter。

LLaVA:
  hidden-state bridge 已成立。
  feature-level bridge 未证成。
  source-control probe 只有 partial。
  下一步应作为 smaller-effect / diagnostic replication，而不是强行追正结论。
```

## 2. Claim Boundary

Stage 2Q 即使成功，也要分层写：

```text
Qwen approximate full-route support:
  如果 Qwen 在 source discovery、source-control、mask sensitivity、rank/generation linkage 都成立，
  可以写成 Qwen provides approximate Gemma-style route support。

LLaVA partial full-route support:
  如果 LLaVA 只有 hidden/rank 成立，但 feature/source-control 弱，
  写成 LLaVA supports hidden-state-level route bridge but not feature-level source-control route。

Full Gemma-style route replication:
  只有真正完成 source tracing adapter，并在 source/control/region/generation 全链条成立，
  才能写 Qwen/LLaVA fully replicate Gemma-style source-control routes。
```

不能写：

```text
Qwen/LLaVA 已完整复现 Gemma route。
LLaVA 没有跨模态机制。
D_visual_only 是更好的 prompt。
Qwen/LLaVA feature 是对象级语义节点。
```

## 2.1 Transcoder / Loader 可比性风险

Stage 2Q 必须显式承认 Qwen / LLaVA 的 transcoder 资产和 Gemma 主线不同。

```text
Gemma mainline:
  transcoder = tianhux2/gemma3-4b-it-plt
  type = PLT / current Gemma3-compatible transcoder set
  pipeline = ReplacementModel + Gemma3ForConditionalGeneration path
  strength = full attribution graph + source tracing + node intervention

Qwen:
  transcoder = KokosDev/qwen2p5vl-7b-clt
  observed type = config.yaml + layer_*.safetensors
  config_model_kind = transcoder_set
  hooks = blocks.{layer}.hook_resid_pre -> blocks.{layer}.hook_resid_post
  current pipeline = native Qwen forward + CLT feature readout/patch/probe
  limitation = no full ReplacementModel source-tracing adapter yet

LLaVA:
  transcoder = KokosDev/llava15-7b-clt
  observed type = custom transcoder_L*.pt + mapping_L*.pt
  config = no standard config.yaml
  current pipeline = native LLaVA forward + custom CLT/mapping readout
  limitation = stronger format mismatch; feature route evidence must be treated as diagnostic
```

对 Stage 2Q 的影响：

```text
1. Qwen / LLaVA 的 feature_id 不应被写成和 Gemma node 完全等价。
2. Qwen / LLaVA 的 source-control 实验应写成 source-like / approximate source-control。
3. 除非后续实现真正的 Qwen/LLaVA ReplacementModel + attribution graph adapter，
   否则不能写 full Gemma-style source tracing replication。
4. 如果 Qwen/LLaVA 在 native patch/probe 上成功，只能升级为 approximate Gemma-style route support。
```

## 3. 实验对象

### 3.1 模型

```text
Qwen:
  model = Qwen2.5-VL-7B-Instruct
  primary layer = 26
  CLT = KokosDev/qwen2p5vl-7b-clt
  role = main cross-model replication candidate

LLaVA:
  model = LLaVA-1.5-7B
  primary hidden layer = 15
  diagnostic feature layers = 15, 18
  CLT = KokosDev/llava15-7b-clt
  role = smaller-effect / heterogeneity diagnostic candidate
```

### 3.2 样本

优先复用 Stage 2N / 2P 的 localized masks，不新增标注。

```text
Qwen:
  使用 Stage 2P heldout positive rows。
  默认 24 prompt-runs。
  优先 answer_mask 与 union_mask 都可用的样本。

LLaVA:
  使用 Stage 2N hidden-positive rows。
  默认 16-24 prompt-runs。
  如果 feature-level 仍弱，保留 hidden/rank diagnostic，不强行写 feature route。
```

样本条件：

```text
image_dependence = strong
evidence region = localized
mask_condition = answer_mask / union_mask
prompt = B_direct / D_visual_only
```

## 4. Stage 2Q-1：Cross-Model Route Candidate Discovery

目的：

```text
构造比 Stage 2O 更接近 Gemma source tracing 的 source candidate 列表。
```

方法：

```text
1. 对 clean、answer_mask、union_mask 跑 forward。
2. 在 target answer first-token / target token 方向计算 feature attribution。
3. 在 visual evidence positions 与 answer-adjacent positions 上筛 source-like candidates。
4. source candidate 必须同时满足：
   - clean_activation - masked_activation > 0
   - decoder_vector · target_logit_direction > 0
   - zeroing 后 target logit / rank 受损
   - real mask sensitivity > shifted / shuffled sensitivity
5. matched control 在同 layer / bucket 中选择：
   - activation 接近
   - feature norm 接近
   - drop 或 attribution 接近
   - 但 target specificity 或 mask specificity 弱
```

输出：

```text
stage2q_route_candidates_qwen.csv/json
stage2q_route_candidates_llava.csv/json
stage2q_route_candidate_summary.csv
```

成功标准：

```text
Qwen:
  至少 12 个 prompt-runs 有 usable source-control pairs。

LLaVA:
  至少 8 个 prompt-runs 有 usable source-control pairs；
  若不足，标记为 feature-route candidate discovery blocked/weak。
```

## 5. Stage 2Q-2：Source-Control Intervention Mainline

目的：

```text
复现 Gemma 主线中最核心的 source > matched control 证据。
```

条件：

```text
clean
answer_mask
union_mask
shifted_mask
mask_shuffled
wrong_target
```

干预：

```text
source_zeroing
matched_control_zeroing
source_restore
matched_control_restore
```

指标：

```text
source_damage_vs_clean
control_damage_vs_clean
source_minus_control
source_restore_vs_mask
control_restore_vs_mask
real_minus_shifted
real_minus_shuffled
correct_minus_wrong
target_rank_effect
gap_closure
```

成功标准：

```text
Qwen:
  source_minus_control 在 zeroing 与 restore 至少一个方向 CI 不跨 0。
  real_minus_shifted / real_minus_shuffled 稳定为正。
  correct_minus_wrong 稳定为正。

LLaVA:
  若 source_minus_control 方向稳定但 effect 小，可写 partial。
  若 matched controls 吸收效果，只保留 hidden-state bridge，不写 feature/source route。
```

## 6. Stage 2Q-3：Evidence Region Sensitivity Mainline

目的：

```text
确认 source-like route 对人工 evidence region 的敏感性强于 controls。
```

对照：

```text
answer_mask
union_mask
random4
shifted_mask
mask_shuffled
matched feature control
wrong target
```

指标：

```text
answer_minus_random4
union_minus_random4
answer_minus_shifted
union_minus_shifted
source_sensitivity_minus_control_sensitivity
```

成功标准：

```text
Qwen:
  至少 answer_mask 或 union_mask 中一个稳定强于 random/shifted/shuffled controls。
  source sensitivity > matched control sensitivity。

LLaVA:
  若 hidden-state region sensitivity 成立但 feature source-control 不成立，
  写成 hidden-state-level evidence-region sensitivity。
```

## 7. Stage 2Q-4：Rank / First-Token / Decoded Answer Linkage

目的：

```text
补上“机制变化是否联系到行为输出”的桥。
```

优先顺序：

```text
1. target logit / rank
2. first answer token probability
3. short greedy decoded answer
```

条件：

```text
clean
answer_mask
union_mask
source_zeroing
matched_control_zeroing
source_restore
matched_control_restore
shifted_mask
wrong_target
```

成功标准：

```text
如果 decoded answer 改变：
  可以写 generation-level bridge support。

如果 decoded answer 不变，但 first-token / rank 稳定：
  只能写 first-token/rank bridge support。

如果 rank/logit 也不稳定：
  不能写 route explains behavior。
```

## 8. Stage 2Q-5：Verdict

最终结论分四档：

```text
supported:
  source-control、evidence-region sensitivity、negative controls、rank/generation bridge 全部成立。

partial:
  source-control 或 evidence-region sensitivity 成立，但 generation bridge 不成立。

hidden-only:
  hidden-state patch 成立，但 feature/source route 不成立。

not_supported / blocked:
  source candidate 不可构造，或 controls 吸收效果，或接口无法完成。
```

推荐输出文件：

```text
067_stage2q_qwen_gemma_style_route_replication.md
068_stage2q_llava_gemma_style_route_replication.md
069_stage2q_cross_model_full_route_verdict.md

cross_model/stage2q_qwen_route_candidates.csv
cross_model/stage2q_qwen_route_intervention.csv
cross_model/stage2q_qwen_region_sensitivity.csv
cross_model/stage2q_qwen_behavior_linkage.csv
cross_model/stage2q_qwen_decision.json

cross_model/stage2q_llava_route_candidates.csv
cross_model/stage2q_llava_route_intervention.csv
cross_model/stage2q_llava_region_sensitivity.csv
cross_model/stage2q_llava_behavior_linkage.csv
cross_model/stage2q_llava_decision.json
```

## 9. 当前预期

Qwen 预期：

```text
Qwen 很可能能通过 approximate Gemma-style route support。
因为 Stage 2P 已经显示：
  feature bridge bidirectional supported
  source-control restore supported
  source-control zeroing supported
  real > shuffled
  correct > wrong
```

LLaVA 预期：

```text
LLaVA 可能只能到 hidden-only 或 partial。
原因是 Stage 2P layer sweep 没有找到稳定强于 matched controls 的 feature bridge。
但仍然值得跑，因为可以证明：
  要么 LLaVA 也有小效应 route；
  要么它的跨模型证据停留在 hidden-state 层，而不是 feature/source 层。
两种结果都能帮助论文收束 claim。
```

## 10. 执行优先级

```text
第一优先级：
  Qwen Stage 2Q-1 / 2Q-2 / 2Q-3。
  目标是确认 Qwen 是否能形成 approximate full route。

第二优先级：
  Qwen Stage 2Q-4 decoded / first-token linkage。
  目标是增强行为桥。

第三优先级：
  LLaVA Stage 2Q diagnostic route replication。
  目标是明确 LLaVA 是 hidden-only、partial，还是 feature/source 也能成立。
```

## 11. 与主论文 claim 的关系

Stage 2Q 成功前，论文主 claim 应保持：

```text
Gemma provides the full causal route evidence.
Qwen provides strong heldout-supported feature/source-control auxiliary evidence.
LLaVA provides hidden-state replication and weak feature diagnostics.
```

Stage 2Q 若 Qwen 成功，可升级为：

```text
Qwen provides approximate Gemma-style route support,
although full source-tracing adapter replication remains future work.
```

Stage 2Q 若 LLaVA 成功，可升级为：

```text
LLaVA shows smaller-effect partial route support.
```

Stage 2Q 若 LLaVA 失败，仍然写：

```text
LLaVA hidden-state bridge is replicated, but feature/source route localization remains unproven.
```
