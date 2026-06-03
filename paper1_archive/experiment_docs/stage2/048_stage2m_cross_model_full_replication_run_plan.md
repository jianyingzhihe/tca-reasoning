# Stage 2M：Qwen / LLaVA Cross-Model 完整复现推进计划

## 1. 目标

Stage 2M 的目标是把 Qwen/LLaVA 从 Stage 2L 的 hidden-state auxiliary evidence 推进到更接近 Gemma 主线的跨模型复现。

这里把“完整复现”拆成三档：

```text
Tier 1: Hidden-state full replication
Tier 2: Feature-level causal bridge
Tier 3: Source-control route replication
```

本轮优先执行 Tier 1。

## 2. Tier 1 判据

Qwen 和 LLaVA 都要在 expanded samples 上通过：

```text
1. matched controls:
   top_hidden_delta_plus_answer_adjacent > delta_matched_plus_answer_adjacent
   top_hidden_delta_plus_answer_adjacent > activation_matched_plus_answer_adjacent

2. wrong-target control:
   correct target effect > wrong target effect

3. mask-shuffled control:
   real evidence mask effect > shifted mask effect

4. first-token target logit/rank:
   restore/corrupt 方向一致
```

如果 Qwen 全部稳定、LLaVA effect size 更小但 wrong-target 与 mask-shuffled 稳定成立，则写成：

```text
Qwen strong auxiliary replication;
LLaVA smaller-effect hidden-state replication with partial matched-control specificity.
```

## 3. Manifest

使用脚本：

```text
scripts/local/build_stage2m_cross_model_manifest.py
```

输出：

```text
doc/experiments/stage2/cross_model/stage2m_selected_24_manifest.csv
doc/experiments/stage2/cross_model/stage2m_annotation_supplement_needed.csv
doc/experiments/stage2/cross_model/stage2m_manifest_summary.json
```

当前 manifest 判定：

```text
selected_count = 24
available_eligible_count = 52
usable_status = pass_min20

selected_type_counts:
  symbol_text_reading = 11
  visual_readout = 11
  scene_inference = 2

quota_deficits:
  symbol_text_reading = 1
  scene_inference = 2
```

严格 `12 / 8 / 4` 类型配比无法完全满足，因此不硬凑弥漫样本；缺口由高分 localized visual_readout 样本补齐。

## 4. Tier 1 执行

使用脚本：

```text
scripts/local/run_stage2m_hidden_bridge_remote.py
```

远端执行内容：

```text
Qwen layer 26 hidden-position matched controls
LLaVA layer 15 hidden-position matched controls
Qwen wrong-target negative control
LLaVA wrong-target negative control
Qwen mask-shuffled negative control
LLaVA mask-shuffled negative control
```

主要输出：

```text
stage2m_qwen_hidden_position_patch.csv/json
stage2m_llava_hidden_position_patch.csv/json
stage2m_qwen_wrong_target_negative_control.csv/json
stage2m_llava_wrong_target_negative_control.csv/json
stage2m_qwen_mask_shuffled_negative_control.csv/json
stage2m_llava_mask_shuffled_negative_control.csv/json
stage2m_matched_control_model_summary.csv
stage2m_wrong_target_summary.csv
stage2m_mask_shuffled_summary.csv
stage2m_full_replication_tier1_decision.json
049_stage2m_expanded_hidden_bridge_replication.md
```

## 5. 后续 Tier 2 / Tier 3

Tier 1 通过后才继续：

```text
Tier 2: feature-level clean->masked corruption and masked->clean restoration
Tier 3: source/control route replication
```

如果 Tier 2 失败，则最终写成：

```text
Cross-model hidden-state bridge replication succeeds, but feature-level causal route replication remains unproven.
```

如果 Tier 2 至少 Qwen 成功，LLaVA partial，则写成：

```text
The phenomenon is not Gemma-only; Qwen provides strong auxiliary replication and LLaVA shows smaller but target/evidence-specific hidden-state replication.
```

## 6. 明确不可写

除非后续额外完成 source tracing、feature/node intervention、matched non-source controls、region-mask sensitivity、rank/generation linkage 全链条，否则不能写：

```text
Qwen/LLaVA fully replicate Gemma source-control causal routes.
```

