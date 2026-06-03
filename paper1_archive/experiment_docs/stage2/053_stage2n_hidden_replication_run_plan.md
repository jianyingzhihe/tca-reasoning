# Stage 2N：Cross-Model Hidden-State Replication 加固计划

## 1. 目标

Stage 2N 的目标是强化跨模型 hidden-state 证据，而不是继续扩大 feature smoke。

核心问题：

```text
Qwen/LLaVA 的 evidence-region-sensitive hidden bridge 是否能在 Stage 2M 未使用过的 localized masks 上独立复现？
```

结论边界：

```text
只证明 hidden-state bridge。
不证明 feature-level causal bridge。
不证明 Gemma-style source-control route replication。
不写 D_visual_only 比 B_direct 更好。
```

## 2. 样本设计

Manifest 构建脚本：

```text
scripts/local/build_stage2n_heldout_cross_model_manifest.py
```

输入：

```text
doc/experiments/stage2/cross_model/stage2i_cross_model_candidate_manifest.csv
doc/experiments/stage2/cross_model/stage2m_selected_24_manifest.csv
```

输出：

```text
doc/experiments/stage2/cross_model/stage2n_heldout_manifest.csv
doc/experiments/stage2/cross_model/stage2n_all52_manifest.csv
doc/experiments/stage2/cross_model/stage2n_annotation_supplement_needed.csv
doc/experiments/stage2/cross_model/stage2n_manifest_summary.json
```

当前 manifest：

```text
eligible_count = 52
Stage 2M excluded = 24
Stage 2N heldout = 28
usable_status = pass_min20
```

类型分布：

```text
heldout:
  visual_readout = 9
  untyped_localized = 19

all52:
  symbol_text_reading = 11
  visual_readout = 20
  scene_inference = 2
  untyped_localized = 19
```

注意：

```text
untyped_localized 是已有标注中缺少 reasoning_operation 元数据的 localized 样本。
它们可以用于 heldout hidden replication，但不能用于强类型化 claim。
```

## 3. 实验设计

固定模型：

```text
Qwen: Qwen2.5-VL-7B-Instruct, layer 26
LLaVA: LLaVA-1.5-7B, layer 15
prompts: B_direct, D_visual_only
directions: restore, corrupt
primary group: top_hidden_delta_plus_answer_adjacent
```

Mask 条件：

```text
answer_mask
union_mask
```

每个 mask condition 独立计算：

```text
clean hidden
masked hidden
top_hidden_delta positions
matched controls
random controls
answer-adjacent positions
```

主对照：

```text
source-like group:
  top_hidden_delta_plus_answer_adjacent

matched controls:
  delta_matched_plus_answer_adjacent
  activation_matched_plus_answer_adjacent

other controls:
  low_delta_control
  random_control_1..4

decomposition:
  top_hidden_delta
  answer_adjacent_text
```

负控制：

```text
wrong_target: union_mask 条件下比较 correct target > wrong target
mask_shuffled: union_mask 条件下比较 real evidence mask > shifted mask
```

## 4. 脚本

远端 runner：

```text
scripts/local/run_stage2n_hidden_replication_remote.py
```

新增研究脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_cross_model_hidden_position_patch_mask_condition_smoke.py
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/analyze_stage2n_hidden_replication.py
```

复用研究脚本：

```text
run_cross_model_wrong_target_negative_control_smoke.py
run_cross_model_mask_shuffled_negative_control_smoke.py
analyze_stage2l_negative_controls.py
analyze_stage2l_mask_shuffled_control.py
```

## 5. 成功标准

Heldout primary：

```text
Qwen 和 LLaVA 在 heldout pack 上 source-like > random controls 方向稳定。
wrong_target control 支持 correct > wrong。
mask_shuffled control 支持 real > shuffled。
```

Stronger support：

```text
answer_mask 和 union_mask 至少一个成立。
如果两个都成立，写成 stronger evidence-region sensitivity。
visual_readout 至少复现一条完整链条。
```

保守失败处理：

```text
如果只有 all52 pooled 成立，但 heldout 不成立，只写 pooled auxiliary evidence。
如果 LLaVA 更小但方向稳定，写 smaller-effect replication。
如果 matched controls 吸收效果，保留 hidden bridge，但降低 specificity claim。
```

## 6. 输出

Hidden 输出：

```text
stage2n_qwen_hidden_position_patch.csv/json
stage2n_llava_hidden_position_patch.csv/json
stage2n_heldout_hidden_summary.csv
stage2n_all52_hidden_summary.csv
stage2n_hidden_specificity_case.csv
stage2n_hidden_specificity_summary.csv
stage2n_hidden_typed_summary.csv
stage2n_hidden_mask_condition_summary.csv
stage2n_hidden_prompt_summary.csv
stage2n_hidden_replication_decision.json
```

Strict controls 输出：

```text
stage2n_wrong_target_summary.csv
stage2n_wrong_target_decision.json
stage2n_mask_shuffled_summary.csv
stage2n_mask_shuffled_decision.json
```

文档输出：

```text
054_stage2n_heldout_hidden_bridge_replication.md
055_stage2n_stricter_controls_and_mask_condition.md
056_stage2n_hidden_replication_verdict.md
```
