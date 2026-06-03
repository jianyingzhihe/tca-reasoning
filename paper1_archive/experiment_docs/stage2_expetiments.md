# Stage 2 Experiments：持续实验日志

日期：2026-05-19  
用途：记录 Stage 2 每一个实验 part 的目的、输入、输出、方法、结果、预期与实际偏差、结论和后续动作。

本文件是 Stage 2 的总日志。更长的单实验记录放在：

```text
doc/experiments/stage2/
```

后续规则：

1. 每完成一个实验 part，就先在本文增加一条总记录；
2. 如果该实验有较多表格、图片、日志或中间判断，就在 `doc/experiments/stage2/` 下单独新建一个完整 md；
3. 每条记录都必须写清楚目的、输入、输出、方法、结果、预期与实际偏差、结论；
4. 不允许只写“跑了”“成功了”“失败了”，必须解释它对主 claim 的影响；
5. 工程失败要和机制失败分开写，不能混为一谈；
6. 所有 strongest-evidence 表述必须继续排除 `D_visual_only + no_image`。

---

## 0. 固定主线

Stage 2 不再追求证明：

> `D_visual_only` 比 `B_direct` 更好。

Stage 2 要补强的是：

> 在 localized、strong image-dependence 的 VQA 样本中，VLM 答案生成附近存在可因果识别的 support routes；这些 support routes 对 wrong image 和 answer / union evidence-region mask 敏感，强于 nearest matched non-source controls 和 same-area random controls，并且这种 route weakening 与 target rank / decoded answer change 有行为关联。

Stage 2 的主要补强顺序：

```text
Stage 2A: targeted replication pack
Stage 2B: node-to-generation bridge
Stage 2C: semantic feature / region cases
Stage 2D: suppressor deep case
Stage 2F: cross-model feasibility and mini replication
```

优先级判断：

```text
必须优先：Stage 2A targeted replication
可以并行轻量推进：Stage 2F-0 asset survey
Stage 2A 后推进：Stage 2B / 2C / 2F loader smoke
不抢主线：restoration、broader type extension、manual transcoder training
```

---

## 1. 术语解释

`VLM`：Vision-Language Model，视觉语言模型，输入可以同时包含图像和文本。

`VQA`：Visual Question Answering，视觉问答任务。

`answer-adjacent route`：答案生成附近的内部路径。这里不是指完整神经网络所有计算，而是指通过 attribution / tracing 在目标答案 token 附近追踪到的 feature nodes 和它们对目标 logit 的影响路径。

`support route`：支持路径。清零该节点或路径后，目标答案 logit / probability / rank 变差，说明它原本在支持目标答案。

`suppressor route`：抑制或竞争路径。清零该节点后，目标答案 logit 反而上升，说明它可能在压制目标答案，或者支持竞争答案。

`source node`：tracing 中找到的源节点，是当前主实验认为可能参与答案路径的 feature node。

`nearest control`：节点对照。选择与 source node 在层、位置或激活属性上匹配，但不是 traced source 的 non-source feature，用来排除“附近任意 feature 都有效”的解释。

`random region control`：区域对照。遮挡与 answer mask 面积相近、但不与 answer / relate 区域重叠的随机区域，用来排除“遮挡任意区域都会造成同样效果”的解释。

`answer mask`：人工标注的最小答案证据区域。

`relate mask`：与答案相关的辅助区域、上下文区域或第二证据区域。

`union mask`：answer mask 和 relate mask 的并集。若一个样本有多个 answer 标注区域，则多个 answer shapes 也必须合并覆盖。

`evidence-region sensitivity`：证据区域敏感性。遮挡 answer / union 区域后，support route effect 被削弱，并且这种削弱大于 random region control 和 nearest node control。

`node-to-generation bridge`：节点到生成行为的桥。目标是证明 node intervention 不只影响 target logit，也会影响 first answer token distribution 或 decoded answer。

`cross-model smoke`：跨模型烟测。不是正式复现实验，而是验证另一个模型和 transcoder 资产是否能加载、hook、读出 feature activation、施加 intervention。

---

## 2. 总进度表

| Part | 名称 | 状态 | 主目的 | 当前判断 |
|---|---|---:|---|---|
| 2A | Targeted replication pack | 已完成首轮判定 | 独立复现 core24 evidence-region-sensitive support route | nearest8 给出 partial but useful replication：support route evidence-region weakening 复现，union 最稳；source>nearest 与 strict random4 仍偏弱 |
| 2B | Node-to-generation bridge | 已完成 first-token expansion + decoded-loop smoke | 补 node intervention 到生成行为的直接桥 | first-token bridge 已从 4-case 扩到 nearest8 support pairs；手写 greedy decoded loop 已跑通 2 个 case，其中 1927165 出现 source-specific decoded answer change |
| 2C | Semantic feature / region cases | 已完成首版 case package | 做 2-3 个高质量解释性 case | `1927165` 是主正例，`2683965` 是 first-token 与 decoded dissociation 边界例；不能升级为 object-level semantic node 证明 |
| 2D | Suppressor deep case | 已完成 first-token；decoded 最小 probe 已修复重跑 | 判断 suppressor 是否与竞争答案有关 | `3794755` 支持 suppressor first-token secondary evidence；decoded 目前没有 source-specific positive |
| 2F-0 | Cross-model asset survey | 已完成初查 | 查是否有现成 VLM circuit/transcoder 资产 | 发现 Qwen2.5-VL / LLaVA 候选 |
| 2F-1 | Cross-model loader smoke | 已完成 asset-level light smoke | 验证 Qwen2.5-VL CLT 是否能进入方法链 | partial：config/27层权重列表/hook metadata 可读；完整 CLT lazy load 因未下载 safetensors 被跳过；ReplacementModel 仍 Gemma3-only |
| 2F-2 | Qwen download / hook-forward / feature readout | 已完成 Qwen asset + native forward + Q1 readout | 验证 Qwen 是否能进入 hidden-state-to-CLT readout 层 | Qwen CLT 全量、base snapshot、native hook-forward、layer 0/13/26 CLT readout 均已跑通；仍不是 attribution/intervention |
| 2F-3 | Qwen position/mask readout + LLaVA base smoke | 已完成本轮判定 | 验证 Qwen position 对齐和 evidence-mask readout；推进 LLaVA processor/config + hook 可行性 | Qwen Q2 `pass_position_mapping`，Qwen Q3 `partial_mask_readout` 且 layer 26 image span 对 answer/union mask 最敏感；LLaVA L1 `partial_base_asset_pass`，L2 被慢速下载阻塞 |

---

## 3. 实验记录索引

| 编号 | 实验 | 详细文档 | 状态 | 一句话结论 |
|---|---|---|---:|---|
| 000 | Cross-model asset survey | [000_cross_model_asset_survey.md](stage2/000_cross_model_asset_survey.md) | 已完成初查 | 有现成 VLM 候选，优先 Qwen2.5-VL-7B CLT，但需要 loader smoke |
| 001 | Stage 2A candidate selection | [001_stage2a_candidate_selection.md](stage2/001_stage2a_candidate_selection.md) | 已完成 selection | immediate-ready 只有 2 个且 diffuse；已生成 top24 pretrace queue 和 B/D manifests |
| 002 | Stage 2A-1 top24 pretrace remote start | [002_stage2a_pretrace_remote_start.md](stage2/002_stage2a_pretrace_remote_start.md) | 远端运行中 | 首次 CSV BOM 问题已修复；B/D eval 已 24/24 跑通，正在跑 answer-aligned trace |
| 003 | Stage 2A-1 top24 pretrace eval partial readout | [003_stage2a_pretrace_eval_partial_readout.md](stage2/003_stage2a_pretrace_eval_partial_readout.md) | 已完成 eval 初读 | top24 无空答案；B 格式 24/24，D 格式 18/24；D 不支持“行为更优”叙事 |
| 004 | Stage 2A-1 top24 pretrace trace/compare readout | [004_stage2a_pretrace_trace_compare_readout.md](stage2/004_stage2a_pretrace_trace_compare_readout.md) | 已完成 trace/compare 读数 | B/D trace 均 24/24；21 个已有 mask；12 个 clean_prompt；已启动 source zeroing top4 |
| 005 | Stage 2A-2 source zeroing top4 readout | [005_stage2a_source_zeroing_top4_readout.md](stage2/005_stage2a_source_zeroing_top4_readout.md) | 已完成 source zeroing | 87 条成功干预；support=46、suppressor=35；existing mask + support samples=17 |
| 006 | Stage 2A-3 nearest-control clean-screen readout | [006_stage2a_nearest_control_clean_readout.md](stage2/006_stage2a_nearest_control_clean_readout.md) | 已完成 nearest clean-screen | nearest 匹配 27/81 non-neutral rows；support+mask+nearest samples=8，足够启动小而严格的 region-mask replication |
| 007 | Stage 2A-4 region-mask replication pack and remote start | [007_stage2a_region_replication_pack_and_remote_start.md](stage2/007_stage2a_region_replication_pack_and_remote_start.md) | 已完成启动 | 已构建 nearest8 pack：8 samples、20 source-control pairs、40 manifest rows；远端 random4 region-mask job 已启动并完成 |
| 008 | Stage 2A-5 nearest8 region-route readout | [008_stage2a_region_route_readout.md](stage2/008_stage2a_region_route_readout.md) | 已完成 route 读数 | support source 的 answer/union weakening 方向复现，union 更稳；source>nearest 方向为正但 CI 跨 0；strict random4 覆盖不足，只能 diagnostic |
| 009 | Stage 2A-6 behavior / generation readout | [009_stage2a_behavior_generation_readout.md](stage2/009_stage2a_behavior_generation_readout.md) | 已完成行为读数 | union mask 稳定伤害 target rank / margin，并更常改变 decoded answer；answer-mask route weakening 与 answer-mask rank damage 有正相关信号 |
| 010 | Stage 2A targeted replication verdict | [010_stage2a_targeted_replication_verdict.md](stage2/010_stage2a_targeted_replication_verdict.md) | 已完成判定 | Stage 2A 判为 main partial replication + behavior support；不能升级为 full independent strong replication |
| 011 | Stage 2B first-token node bridge smoke | [011_stage2b_first_token_node_bridge_smoke.md](stage2/011_stage2b_first_token_node_bridge_smoke.md) | 已完成 smoke | 工程上已能做 `feature_intervention -> first answer token distribution`；`2683965` 是强正向 case，但总体仍异质、非强统计结论 |
| 012 | Stage 2B nearest8 first-token bridge expansion | [012_stage2b_first_token_nearest8_expansion.md](stage2/012_stage2b_first_token_nearest8_expansion.md) | 已完成扩展 | 扩到 Stage 2A nearest8 全部 support pairs；`union_mask` 的 source-minus-nearest rank/logit gap 最稳定，answer_mask 为正但受强 case 驱动 |
| 013 | Stage 2B greedy decoded loop smoke | [013_stage2b_greedy_decoded_loop_smoke.md](stage2/013_stage2b_greedy_decoded_loop_smoke.md) | 已完成 2-case smoke | 手写 decoded loop 可行；`1927165` 有 source-specific decoded change，`2683965` 则显示 first-token rank damage 不必然改变 decoded answer |
| 014 | Stage 2C semantic / region case package | [014_stage2c_case_package.md](stage2/014_stage2c_case_package.md) | 已完成首版 case package | `1927165` 是最强 figure-ready 正例；`2683965` 是 first-token 与 decoded-answer dissociation 的边界 case；不升级为 object-level semantic node 证明 |
| 015 | Stage 2D suppressor deep case | [015_stage2d_suppressor_deep_case.md](stage2/015_stage2d_suppressor_deep_case.md) | 已完成 first-token；decoded 最小 probe 已修复重跑 | `3794755` 的 suppressor source 清零在 first-token 层面方向一致地改善目标 token；修复后 `016/union4` decoded probe 跑通，但 source 未改变答案，nearest 仅出现 `television -> a television` 的格式性变化 |
| 016 | Stage 2F cross-model loader smoke | [016_stage2f_cross_model_loader_smoke.md](stage2/016_stage2f_cross_model_loader_smoke.md) | 已完成 asset-level light smoke | Qwen2.5-VL CLT config、27 层 layer 文件列表和 hook metadata 可读；由于没有完整 safetensors cache 且 ReplacementModel 仍 Gemma3-only，判定为 partial 而非跨模型复现 |
| 017 | Stage 2F-2 Qwen download and lazy load | [017_stage2f_qwen_download_and_lazy_load.md](stage2/017_stage2f_qwen_download_and_lazy_load.md) | 已完成 asset download + base loader smoke | Qwen CLT 27/27 层全量下载并 lazy load 通过；Qwen base 经 ModelScope fallback 补齐 5 个 shard，本地 processor/config/meta loader 判定为 pass |
| 018 | Stage 2F-2 Qwen hook-forward smoke | [018_stage2f_qwen_hook_forward_smoke.md](stage2/018_stage2f_qwen_hook_forward_smoke.md) | 已完成 native hook-forward smoke | Qwen native 图文 forward 跑通，hidden states 和 `model.language_model.layers.0` hook shape 可读；因缺 Qwen ReplacementModel adapter，判定为 partial 而非跨模型机制复现 |
| 019 | Stage 2F Qwen CLT feature readout smoke | [019_stage2f_qwen_clt_feature_readout_smoke.md](stage2/019_stage2f_qwen_clt_feature_readout_smoke.md) | 已完成 Q1 readout | Qwen hidden states 可编码到 CLT features `[1, seq, 8192]`，但全局 top activation 曾集中在 `position 2`，需要 Q2 position mapping 排除模板位置假信号 |
| 020 | Stage 2F LLaVA asset format smoke | [020_stage2f_llava_asset_format_smoke.md](stage2/020_stage2f_llava_asset_format_smoke.md) | 已完成 L0 asset format | `KokosDev/llava15-7b-clt` 的 `.pt` / mapping 资产可读，hidden_dim=4096、feature_dim=8192；需要 LLaVA adapter |
| 021 | Stage 2F Qwen token/position mapping smoke | [021_stage2f_qwen_token_position_mapping_smoke.md](stage2/021_stage2f_qwen_token_position_mapping_smoke.md) | 已完成 Q2 position mapping | `position 2` 被定位为 system/template newline，不是视觉或答案附近位置；image/question/assistant/last prompt token 都能定位 |
| 022 | Stage 2F Qwen clean vs evidence-mask feature readout | [022_stage2f_qwen_clean_vs_mask_feature_readout.md](stage2/022_stage2f_qwen_clean_vs_mask_feature_readout.md) | 已完成 Q3 partial readout | 2/3 主 case 有 mask；Qwen layer 26 image span 在 answer/union mask 下出现最明显 top feature drop 与 top-k 改组，但仍只是 readout-level 证据 |
| 023 | Stage 2F LLaVA base/hook-forward smoke | [023_stage2f_llava_base_hook_forward_smoke.md](stage2/023_stage2f_llava_base_hook_forward_smoke.md) | 已完成 L1 partial；L2 blocked | LLaVA processor/config 可读且 text hidden size=4096；base 权重下载过慢，未完成 native hook-forward |

---

## 4. 实验 000：Cross-model Asset Survey

### 4.1 目的

检查是否已经有别人训练好的 VLM circuit / transcoder / CLT 资产，避免过早进入手动训练。

核心问题：

```text
有没有现成小模型或中等模型可以做跨模型验证？
如果没有，是否需要手动训练？
如果有，哪个最适合作为 Stage 2F 的第一候选？
```

### 4.2 输入

本地输入：

```text
vlm-circuit-tracing/README.md
vlm-circuit-tracing/circuit_tracer_vlm/README.md
vlm-circuit-tracing/circuit_tracer_vlm/circuit_tracer/replacement_model.py
```

远端输入：

```text
Hugging Face model cards
Hugging Face repo file lists
candidate config.yaml
```

### 4.3 输出

已输出：

```text
doc/experiments/runplan_stage2.md
doc/experiments/stage2_expetiments.md
doc/experiments/stage2/000_cross_model_asset_survey.md
doc/experiments/stage2/cross_model/cross_model_candidate_table.csv
```

计划后续输出：

```text
doc/experiments/stage2/cross_model/cross_model_loader_smoke.md
```

### 4.4 方法

方法分三步：

1. 检查本地 VLM circuit-tracing README，确认当前主 pipeline 是否只围绕 Gemma3-4B-IT；
2. 搜索 Hugging Face 上与 VLM、transcoder、CLT、circuit-tracer 相关的公开资产；
3. 拉取候选 repo 的 `config.yaml` 或 file list，判断是否有 base model、layer weights、hook point、格式说明。

### 4.5 结果

当前发现：

| 候选资产 | 对应模型 | 类型 | 当前判断 |
|---|---|---|---|
| `tianhux2/gemma3-4b-it-plt` | `google/gemma-3-4b-it` | PLT/transcoder set | 当前主实验已使用 |
| `KokosDev/qwen2p5vl-7b-clt` | `Qwen/Qwen2.5-VL-7B-Instruct` | CLT | 跨模型最高优先级候选 |
| `KokosDev/qwen2p5vl-7b-plt` | Qwen2.5-VL-7B 相关 | PLT | 备用候选，需要核验 card/config 一致性 |
| `KokosDev/llava15-7b-clt` | `llava-hf/llava-1.5-7b-hf` | CLT | 有价值，但格式与当前 loader 差异更大 |
| Gemma-2 2B / Llama-3.2 1B / Qwen-3 系列 | language-only LM | PLT/CLT | 只能做方法 sanity check，不能证明 VLM 视觉路径 |

本地代码约束：

```text
ReplacementModel 当前主要绑定 Gemma3ForConditionalGeneration。
Qwen2.5-VL / LLaVA 不能假定一键替换。
必须先做 loader / hook / processor / transcoder format smoke。
```

### 4.6 预期与实际偏差

预期：

```text
可能没有可用的 VLM transcoder 资产；
如果没有，就需要考虑手动训练。
```

实际：

```text
找到了 Qwen2.5-VL-7B 和 LLaVA-1.5-7B 的公开候选资产。
这比预期更好。
但它们并不等于当前 pipeline 可直接运行。
最大风险从“没有资产”转成了“资产和本地 loader / hook 不兼容”。
```

### 4.7 结论

当前不建议立刻手动训练 transcoder。

Stage 2F 的正确下一步是：

```text
优先做 KokosDev/qwen2p5vl-7b-clt 的 loader / hook smoke。
如果 smoke 成功，再做 1-case attribution/intervention smoke。
如果 one-case smoke 成功，再做 3-5 case mini region replication。
```

跨模型进入 Stage 2，但不应该替代 Stage 2A targeted replication。

### 4.8 后续动作

下一步建议：

1. 先推进 Stage 2A candidate selection；
2. 并行轻量维护 `cross_model_candidate_table.csv`；
3. 在不占用主实验资源的情况下，做 Qwen2.5-VL loader smoke；
4. 若 smoke 卡住，记录工程失败，不把它解释为机制失败。

---

## 5. 实验 005：Stage 2A-2 source zeroing top4 读数

### 目的

给 top24 pretrace pool 中的 answer-adjacent traced source feature nodes 做 clean-image zeroing，判定 support / suppressor / neutral，并检查是否有足够样本进入 Stage 2A targeted replication。

### 输入

```text
remote run root:
/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm/outputs/phase_ab/ab_answer_aligned/stage2a_pretrace_top24_20260519_202951

compare dir:
.../stage2a_pretrace_top24_20260519_202951/compare

selected ids:
doc/experiments/stage2/stage2a_candidate_selection/stage2a_trace_selected_ids_top24.csv
```

### 输出

```text
remote_sync/2026-05-19_stage2a_pretrace_top24/stage2a_source_zeroing_top4.csv
remote_sync/2026-05-19_stage2a_pretrace_top24/stage2a_source_zeroing_top4_enriched.csv
remote_sync/2026-05-19_stage2a_pretrace_top24/stage2a_source_zeroing_top4_sample_summary.csv
doc/experiments/stage2/005_stage2a_source_zeroing_top4_readout.md
```

### 方法

对每个 sample x prompt，从 compare graph 中按 path mass 取 top non-generic feature nodes，运行 feature zeroing。用 `delta_target_logit = intervened_target_logit - original_target_logit` 判定方向。

```text
delta < 0: support
delta > 0: suppressor
delta = 0: neutral
```

### 实际结果

```text
successful intervention rows = 87
support rows = 46
suppressor rows = 35
neutral rows = 6
samples with support = 19
samples with suppressor = 16
existing mask + support samples = 17
clean_prompt + existing mask + support samples = 6
```

### 预期与实际偏差

预期是希望 strict clean_prompt + existing mask + support 至少达到 8 个样本。实际只有 6 个，低于理想门槛。

但 prompt-specific route 口径下，existing mask + support 有 17 个样本，足够继续推进 region-mask replication。

### 结论

Stage 2A 新 pool 复现了 signed route：traced nodes 中既有 support，也有 suppressor。下一步不应强行做 B/D target-aligned 主比较，而应采用 prompt-specific route replication。

### 对主 claim 的影响

这一步支持“答案附近存在 signed causal routes”的基础 claim，并为 evidence-region sensitivity 复现实验提供候选 source nodes。

### 后续动作

继续做 nearest matched-control clean-screen，检查 source nodes 是否强于 nearest non-source controls。

---

## 6. 实验 006：Stage 2A-3 nearest-control clean-screen 读数

### 目的

为实验 005 得到的 non-neutral source rows 寻找 nearest matched non-source controls，并在 clean condition 下做同样 zeroing。目标是判断 strict control 覆盖率，以及 source node 是否强于 nearest control。

### 输入

```text
stage2a_source_zeroing_top4_pilot_clean_nonzero.csv
run root = stage2a_pretrace_top24_20260519_202951
match_mode = nearest
position_alignment = strict
```

### 输出

```text
remote_sync/2026-05-19_stage2a_pretrace_top24/stage2a_nearest_control_clean_top4_nonzero.csv
remote_sync/2026-05-19_stage2a_pretrace_top24/stage2a_nearest_control_clean_top4_nonzero_enriched.csv
remote_sync/2026-05-19_stage2a_pretrace_top24/stage2a_nearest_control_clean_top4_nonzero_per_source_summary.csv
doc/experiments/stage2/006_stage2a_nearest_control_clean_readout.md
```

### 方法

用 `run_modality_counterfactual_matched_control.py` 的 nearest mode 为每条 source row 寻找 strict matched non-source feature。对 matched control 做 clean zeroing，然后将 control delta 与 source delta 合并，按 role 转成可比较的 effect。

```text
support effect = -delta_target_logit
suppressor effect = +delta_target_logit
source-control gap = source_effect - control_effect
```

### 实际结果

```text
source non-neutral rows = 81
nearest matched rows = 27
matched samples = 10
support nearest rows = 21
suppressor nearest rows = 6
existing mask + support + nearest samples = 8
clean_prompt + existing mask + support + nearest samples = 1
```

support source-control gap：

| pool | rows | samples | mean gap | median gap | positive rate | bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---|
| support all, per-source | 21 | 10 | +0.583 | +0.500 | 0.810 | [+0.262, +0.923] |
| support existing-mask, per-source | 16 | 8 | +0.516 | +0.438 | 0.812 | [+0.121, +0.941] |
| support all, sample mean | 10 | 10 | +0.431 | +0.594 | 0.800 | [+0.008, +0.827] |
| support existing-mask, sample mean | 8 | 8 | +0.344 | +0.438 | 0.750 | [-0.149, +0.828] |

suppressor source-control gap：

| pool | rows | samples | mean gap | median gap | positive rate | bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---|
| suppressor all, per-source | 6 | 2 | +1.104 | +1.125 | 1.000 | [+0.875, +1.354] |
| suppressor existing-mask, per-source | 4 | 1 | +1.156 | +1.125 | 1.000 | [+0.875, +1.469] |

### 预期与实际偏差

预期是 nearest-control 至少保住 8 个 support + mask 样本。实际刚好达到 8 个，没有余量。

strict nearest matching 只保住 `27/81` 条 non-neutral rows，说明 matching 很保守；这降低样本量，但也让 surviving controls 更干净。

### 结论

Stage 2A 可以继续启动一个小而严格的 region-mask replication：

```text
primary pool = existing mask + support + nearest-control samples
sample count = 8
analysis framing = prompt-specific route replication
primary role = support
secondary role = suppressor
```

不能把 B/D clean_prompt target-aligned prompt contrast 作为主分析，因为该池只剩 1 个 support + nearest + existing-mask 样本。

### 对主 claim 的影响

这一步支持 specificity claim：traced support source nodes 在 clean zeroing 下整体强于 nearest non-source controls。

但它也提示我们要保守：existing-mask pool 按 sample 聚合后 CI 跨 0，因此 nearest clean-screen 不是最终强证据，只是为 region-mask replication 建立严格对照池。

### 后续动作

进入 Stage 2A-4：

```text
构建 8-sample source+nearest region-mask replication pack
导出 answer / relate / union masks
运行 clean / answer_mask / relate_mask / union_mask / random_control_1..4
分析 support answer/union weakening 是否大于 random4 且 source > nearest
```

---

## 7. 实验 007：Stage 2A-4 region-mask replication pack 构建与远端启动

### 目的

把实验 006 中得到的 `existing mask + support + nearest-control` 样本整理成独立 region-mask replication pack，并启动 evidence-region mask counterfactual。

### 输入

```text
stage2a_nearest_control_clean_top4_nonzero_enriched.csv
answer_aligned_meta_a.csv
answer_aligned_meta_b.csv
旧 LabelMe answer/relate 标注资产
```

### 输出

```text
annotation/stage2a_region_replication_top24_nearest8
annotation/stage2a_region_replication_top24_nearest8/region_experiment_manifest.csv
annotation/stage2a_region_replication_top24_nearest8/region_experiment_manifest_remote.csv
annotation/stage2a_region_replication_top24_nearest8/exported_masks/mask_export_summary.csv
doc/experiments/stage2/007_stage2a_region_replication_pack_and_remote_start.md
```

### 方法

新增并运行：

```text
scripts/local/build_stage2a_region_replication_pack.py
```

该脚本为每条 nearest source-control pair 同时生成 source row 和 nearest_control row，并复制对应图片与 LabelMe JSON。之后用 `export_labelme_regions_to_masks.py` 导出 `answer.png` 和 `relate.png`。

远端使用 `run_region_mask_counterfactual_mainline.py`，固定运行：

```text
clean
answer_mask
relate_mask
union_mask
random_control_1..4
```

### 实际结果

pack 构建完成：

```text
selected source-control pairs = 20
manifest rows = 40
sample count = 8
support sample count = 8
support pairs = 16
suppressor pairs = 4
mask json files = 8
```

远端任务已启动：

```text
remote pack dir = /root/autodl-tmp/tca-reasoning/annotation/stage2a_region_replication_top24_nearest8
remote pid = 838006
remote log = run_stage2a_region_random4_remote.log
remote output = region_mask_stage2a_nearest8_random4.csv
```

初查显示模型加载成功。

### 预期与实际偏差

预期是构建 `8` 个样本的 strict nearest-supported region pack。实际达到预期。

需要注意的是，部分样本的 answer mask 面积很大，说明它们虽然不是 diffuse 全图问题，但也不能解释成“小物体级局部证据”。

### 结论

Stage 2A targeted replication 已进入正式 region-mask counterfactual 阶段。它将直接检验 support route 是否同时满足：

```text
answer/union evidence-region sensitive
source > nearest-control
answer/union > random region control
```

### 对主 claim 的影响

实验 007 本身是启动与数据准备，不直接给出机制结论。真正的机制判定要等实验 008 的 region-mask 读数。

### 后续动作

继续监控远端 job，完成后同步 CSV/log，并写入实验 008。

---

## 8. 实验 008：Stage 2A-5 nearest8 region-route 读数

### 目的

读取 Stage 2A nearest8 region-mask counterfactual 结果，判断 support source route 是否在 answer / union mask 下变弱，以及这种削弱是否强于 nearest-control 和 random-region controls。

### 输入

```text
annotation/stage2a_region_replication_top24_nearest8/region_mask_stage2a_nearest8_random4.csv
annotation/stage2a_region_replication_top24_nearest8/region_experiment_manifest.csv
```

### 输出

```text
annotation/stage2a_region_replication_top24_nearest8/analysis_route/
doc/experiments/stage2/008_stage2a_region_route_readout.md
```

### 方法

用 `summarize_stage2a_region_route_replication.py` 合并 region CSV 和 manifest 的 `pair_id`，计算：

```text
support weakening = masked_delta - clean_delta
suppressor weakening = clean_delta - masked_delta
source-minus-nearest = source_weakening - nearest_weakening
answer/union-minus-random4 = answer/union_weakening - random4_weakening
```

主随机对照仍使用预注册 `random_control_actual_iou <= 0.05`。

### 实际结果

运行完整性：

```text
expected rows = 40 manifest rows x 8 conditions = 320
actual done rows = 320
skip rows = 0
clean calibration exact match rate = 1.0
```

核心读数：

| metric | unit | n | mean | bootstrap 95% CI | 读法 |
|---|---|---:|---:|---|---|
| support source answer-mask weakening | pair | 16 | +0.480 | [+0.109, +0.852] | positive |
| support source answer-mask weakening | sample_mean | 8 | +0.237 | [-0.281, +0.760] | heterogeneous |
| support source union-mask weakening | pair | 16 | +0.574 | [+0.254, +0.902] | positive |
| support source union-mask weakening | sample_mean | 8 | +0.536 | [+0.078, +1.049] | strongest in this pack |
| source-minus-nearest answer weakening | sample_mean | 8 | +0.184 | [-0.254, +0.579] | directional only |
| source-minus-nearest union weakening | sample_mean | 8 | +0.064 | [-0.313, +0.409] | weak / heterogeneous |

strict random4 覆盖：

```text
support source rows with valid random4 = 4/16
support source samples with valid random4 = 2/8
```

### 预期与实际偏差

预期是 `answer/union > random4` 和 `source > nearest` 都能稳定成立。实际是：

```text
answer/union weakening 本身成立，union 更稳；
source > nearest 方向为正，但 CI 跨 0；
strict random4 因有效覆盖不足，不能作为 strongest evidence。
```

### 结论

Stage 2A nearest8 是一个 partial but useful replication：

```text
它方向性复现了 evidence-region-sensitive support route；
但没有强复现 random-region specificity；
node specificity over nearest-control 也只能写成 directional/partial。
```

### 对主 claim 的影响

core24 仍是 strongest evidence。Stage 2A nearest8 应写成 targeted directional replication，尤其支持 union evidence-region mask 会削弱 support source route。

### 后续动作

继续补 behavior eval：检查 answer/union mask 是否造成 target rank / margin damage，并把 route weakening 与行为损伤对应起来。

---

## 9. 实验 009：Stage 2A-6 behavior / generation 读数

### 目的

补齐 Stage 2A nearest8 的行为侧证据：在 `answer_mask` / `union_mask` 证据区域遮挡下，目标答案 token 的 rank、margin 和最终 decoded answer 是否同步受损，并检查 route weakening 与行为损伤之间是否有方向一致的联系。

这一步不声称“节点清零会直接改变自然生成”，而是检验“区域遮挡是否同时影响内部 route 和最终行为”。

### 输入

```text
annotation/stage2a_region_replication_top24_nearest8/region_mask_stage2a_nearest8_behavior.csv
annotation/stage2a_region_replication_top24_nearest8/region_mask_stage2a_nearest8_generation.csv
annotation/stage2a_region_replication_top24_nearest8/analysis_route/route_weakening_iou1.csv
doc/experiments/stage2/009_stage2a_behavior_generation_readout.md
```

### 输出

```text
annotation/stage2a_region_replication_top24_nearest8/analysis_behavior/
```

关键 artifact 包括 `behavior_metric_summary_iou0p05.csv`、`generation_summary_iou0p05.csv`、`generation_cases_iou0p05.csv`、`route_behavior_linkage_iou0p05.csv`、`route_behavior_correlations_iou0p05.csv`。

### 方法

对每个去重后的 `sample_id x run` 运行 `clean`、`answer_mask`、`relate_mask`、`union_mask`、`random_control_1..4`。行为损伤定义为：

```text
rank_damage = masked_target_rank - clean_target_rank
margin_drop = clean_margin - masked_margin
answer_changed_from_clean = normalized(masked_answer) != normalized(clean_answer)
```

route-behavior linkage 只做 exploratory Pearson / Spearman，不引入复杂统计模型。

### 实际结果

运行完整性：

```text
behavior rows = 72
generation rows = 72
empty rows = 0
error rows = 0
sample-runs = 9
conditions = 8
```

核心行为读数：

| metric | n | mean | median | positive_rate | bootstrap 95% CI |
|---|---:|---:|---:|---:|---|
| answer-mask rank damage | 9 | +34.56 | +2 | 0.667 | [-0.11, +89.89] |
| union-mask rank damage | 9 | +20.89 | +13 | 0.889 | [+6.56, +35.33] |
| answer-mask margin drop | 9 | +2.69 | +1.75 | 0.778 | [-0.62, +6.53] |
| union-mask margin drop | 9 | +3.92 | +3.13 | 0.778 | [+0.92, +7.00] |

decoded answer change：

| condition | rows | changed_rows | changed_rate | empty_rows | error_rows |
|---|---:|---:|---:|---:|---:|
| answer_mask | 9 | 5 | 0.556 | 0 | 0 |
| relate_mask | 9 | 4 | 0.444 | 0 | 0 |
| union_mask | 9 | 6 | 0.667 | 0 | 0 |
| strict random valid rows | 8 | 2 | 0.250 | 0 | 0 |
| all random rows | 36 | 14 | 0.389 | 0 | 0 |

route-behavior linkage：

| route metric | behavior metric | n | Pearson | Spearman |
|---|---|---:|---:|---:|
| answer weakening | answer rank damage | 9 | +0.481 | +0.828 |
| union weakening | union rank damage | 9 | -0.354 | -0.167 |
| answer weakening | answer margin drop | 9 | +0.453 | +0.117 |
| union weakening | union margin drop | 9 | -0.496 | -0.577 |

### 预期与实际偏差

预期是 answer / union mask 都会造成 rank damage、margin drop 和 decoded answer change。实际结果更细：`union_mask` 的行为伤害最稳定；`answer_mask` 的方向为正，但 CI 跨 0；answer-mask route weakening 与 answer-mask rank damage 的方向关系较强；union-mask 行为伤害很强，但不被当前 traced support route 的线性 weakening 简单解释。

### 结论

Stage 2A 的行为侧结果支持“证据区域遮挡不只是改变内部 route 数值，也会进入 target rank / margin 和 decoded answer”。但它仍然不能替代 Stage 2B，因为目前还没有证明 `node intervention -> decoded generation` 的直接因果桥。

### 对主 claim 的影响

这一步加强了“route 变化与行为变化有关”的论证。最稳读法是：Stage 2A nearest8 方向性复现 evidence-region-sensitive support route，并补上了区域遮挡造成答案排名和自然生成变化的行为证据。

### 后续动作

写 Stage 2A verdict 文档，并把下一步推进到 Stage 2B node-to-generation bridge smoke。

---

## 10. 实验 010：Stage 2A targeted replication verdict

### 目的

把 Stage 2A 的 pretrace、source zeroing、nearest-control、region route、behavior/generation 结果统一判定，避免继续依赖分散文档临时解释。

### 输入

```text
doc/experiments/stage2/001_stage2a_candidate_selection.md
doc/experiments/stage2/002_stage2a_pretrace_remote_start.md
doc/experiments/stage2/003_stage2a_pretrace_eval_partial_readout.md
doc/experiments/stage2/004_stage2a_pretrace_trace_compare_readout.md
doc/experiments/stage2/005_stage2a_source_zeroing_top4_readout.md
doc/experiments/stage2/006_stage2a_nearest_control_clean_readout.md
doc/experiments/stage2/007_stage2a_region_replication_pack_and_remote_start.md
doc/experiments/stage2/008_stage2a_region_route_readout.md
doc/experiments/stage2/009_stage2a_behavior_generation_readout.md
```

### 输出

```text
doc/experiments/stage2/010_stage2a_targeted_replication_verdict.md
```

### 方法

按 run plan success criteria 做逐条判定：

```text
support source answer/union weakening
source > nearest-control
answer/union > strict random4
rank / margin damage
decoded answer change
route-behavior linkage
```

### 实际结果

最终判定：

```text
Stage 2A = main partial replication + behavior support
```

关键读法：

```text
support source union-mask weakening 是本轮最稳 route 结果；
source > nearest-control 方向为正但 CI 跨 0；
strict random4 在 nearest8 中覆盖不足，不能作为 strongest evidence；
union mask 稳定伤害 target rank / margin；
answer/union masks 经常改变 decoded answer，且 empty/error = 0。
```

### 结论

Stage 2A 支持主 claim 的收窄版本，但不能把主结论升级成 full independent strong replication。core24 仍是 strongest evidence；Stage 2A 的价值是新 targeted pack 中方向复现 route weakening，并补充行为侧支持。

### 后续动作

推进 Stage 2B node-to-generation bridge。

---

## 11. 实验 011：Stage 2B first-token node bridge smoke

### 目的

补主链最缺的一环：从 `node intervention -> target logit` 进一步推进到 `node intervention -> generation-side distribution`。

本轮只做 smoke，不声称完整 decoded generation causal bridge。目标是回答：

```text
在固定 assistant_prefix = "The answer is " 的条件下，
清零 support source node 是否会改变下一答案 token 的 logit、rank 或 top1 token？
这种影响是否比 nearest-control node 更强？
```

### 输入

4-case smoke manifest：

```text
annotation/stage2b_node_generation_smoke/stage2b_node_generation_smoke_manifest_remote.csv
```

选择的 source-control pairs：

```text
003_okvqa_val_1927165_B_support_L11_P196_F151858
010_okvqa_val_2683965_B_support_L26_P287_F39687
019_okvqa_val_80655_A_support_L11_P196_F151858
012_okvqa_val_3794755_A_support_L26_P293_F84060
```

新增脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_region_mask_node_first_token_smoke.py
scripts/local/summarize_stage2b_first_token_smoke.py
```

### 输出

远端与本地同步结果：

```text
annotation/stage2b_node_generation_smoke/stage2b_first_token_source_nearest_4case.csv
annotation/stage2b_node_generation_smoke/stage2b_first_token_source_nearest_4case.log
annotation/stage2b_node_generation_smoke/analysis_first_token/
```

详细记录：

```text
doc/experiments/stage2/011_stage2b_first_token_node_bridge_smoke.md
```

### 方法

先尝试旧 `feature_intervention_generate` 路径，发现它不兼容当前多模态 batch：

```text
HookedVLTransformer.generate asserts input must be torch.Tensor or str;
当前 VLM batch 是包含 input_ids / image / attention_mask 的 dict。
```

因此本轮改为 first-token bridge：

```text
baseline_logits = model.forward_from_batch(batch)
intervention_logits = model.feature_intervention(batch, [(layer, pos, feature_id, 0.0)])
```

然后比较下一答案 token：

```text
delta_target_logit
delta_target_prob
baseline_target_rank
intervention_target_rank
rank_damage_by_intervention
baseline_top1_token
intervention_top1_token
top1_changed_by_intervention
```

条件：

```text
clean
answer_mask
union_mask
```

节点：

```text
source
nearest_control
```

总行数：

```text
4 cases x 2 node_source x 3 conditions = 24 rows
```

### 实际结果

运行完整：

```text
rows = 24
error rows = 0
```

node-source summary：

| node_source | condition | n | mean_rank_damage | mean_delta_target_logit | top1_changed_count |
|---|---:|---:|---:|---:|---:|
| source | clean | 4 | +0.25 | -1.0625 | 1 |
| source | answer_mask | 4 | +20.25 | -0.171875 | 1 |
| source | union_mask | 4 | +6.00 | -0.390625 | 1 |
| nearest_control | clean | 4 | 0.00 | -0.125 | 1 |
| nearest_control | answer_mask | 4 | -28.75 | +0.734375 | 1 |
| nearest_control | union_mask | 4 | -2.75 | +0.515625 | 1 |

source-minus-nearest gap：

| condition | n | mean_rank_damage_gap | positive_rank_damage_gap_rate | mean_delta_logit_gap | negative_delta_logit_gap_rate |
|---|---:|---:|---:|---:|---:|
| clean | 4 | +0.25 | 0.25 | -0.9375 | 1.00 |
| answer_mask | 4 | +49.00 | 0.50 | -0.90625 | 0.75 |
| union_mask | 4 | +8.75 | 1.00 | -0.90625 | 0.75 |

最强正向 case：

```text
okvqa_val_2683965, answer_mask
source rank_damage = +96
nearest rank_damage = -107
rank_damage_gap = +203
source delta_logit = -0.875
nearest delta_logit = +2.8125
```

异质 case：

```text
okvqa_val_1927165: answer_mask 下 source top1 从 "a" 变成 "stop"，但 target rank 反而改善；
okvqa_val_80655: union_mask 下 source 和 nearest 都改变 top1，不是干净 source-specific；
okvqa_val_3794755: clean 下 source 和 nearest 都改变 top1，更像非特异或格式/竞争答案敏感。
```

### 预期与实际偏差

预期：source 清零应比 nearest 更常造成 target rank damage 和 target logit drop。

实际：方向上有支持，尤其 `union_mask` 和 `2683965`，但异质性明显：

```text
source delta logit 平均比 nearest 更负；
union_mask 下 rank_damage_gap 4/4 为正；
answer_mask 下被 2683965 强烈驱动，只有 2/4 pair 为正；
top1 changed 不是 source-specific，因为 nearest 也会在部分 case 改变 top1。
```

### 结论

Stage 2B first-token smoke 的判定是：

```text
工程可行；
机制上 mixed but informative；
有一个强正向 bridge case；
还不能写成强 node-to-generation 统计结论。
```

这一步真正证明的是：

```text
我们现在可以在多模态 batch 上直接测 source/nearest node zeroing 对下一答案 token 分布的影响。
```

还不能证明：

```text
node zeroing 已经稳定改变完整 decoded answer。
```

### 对主 claim 的影响

它补上了主 claim 的一个关键工程入口，但只提供初步机制证据。最稳写法是：

```text
As an initial node-to-generation bridge, source-node zeroing can alter first-answer-token rank/logit in selected cases, with stronger source-specific damage in the best visual-readout case, but the effect remains heterogeneous.
```

中文：

```text
作为节点到生成行为的初步桥接，source node 清零可以在部分 case 中改变下一答案 token 的 rank/logit；最强视觉读出 case 呈现明确 source-specific damage，但整体仍异质。
```

### 后续动作

下一步建议两条并行：

```text
1. 扩大 first-token smoke 到 core24 / Stage2A 中 clean support+nearest 可用的更多 positive cases；
2. 单独修 decoded generation loop，让第一步可施加 node intervention，后续 token 用手写 greedy loop 生成。
```

---

## 12. 实验 012：Stage 2B nearest8 first-token bridge expansion

### 目的

把实验 011 的 4-case first-token bridge 扩大到 Stage 2A nearest8 region pack 的全部 support source/nearest pairs，判断 `node intervention -> first answer token distribution` 的桥接信号是否具有更稳定方向。

核心问题：

```text
support source node 清零是否比 nearest-control node 更伤下一答案 token？
这种 source-minus-nearest 差异在 clean / answer_mask / union_mask 哪个条件下最稳定？
```

### 输入

```text
remote manifest:
/root/autodl-tmp/tca-reasoning/annotation/stage2a_region_replication_top24_nearest8/region_experiment_manifest_remote.csv

remote mask root:
/root/autodl-tmp/tca-reasoning/annotation/stage2a_region_replication_top24_nearest8/exported_masks

local synced csv:
annotation/stage2b_node_generation_smoke/stage2b_first_token_support_nearest8.csv
```

### 输出

```text
annotation/stage2b_node_generation_smoke/stage2b_first_token_support_nearest8.csv
annotation/stage2b_node_generation_smoke/stage2b_first_token_support_nearest8.log
annotation/stage2b_node_generation_smoke/analysis_first_token_nearest8/
doc/experiments/stage2/012_stage2b_first_token_nearest8_expansion.md
```

### 方法

继续使用实验 011 新增的 first-token 脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_region_mask_node_first_token_smoke.py
```

运行设置：

```text
node_role = support
node_source = source + nearest_control
conditions = clean, answer_mask, union_mask
```

规模：

```text
support source-control pairs = 16
node sources = 2
conditions = 3
expected rows = 16 x 2 x 3 = 96
```

主要指标：

```text
rank_damage_by_intervention = intervention_target_rank - baseline_target_rank
delta_target_logit = intervention_target_logit - baseline_target_logit
rank_damage_gap = source_rank_damage - nearest_rank_damage
delta_logit_gap = source_delta_logit - nearest_delta_logit
```

解释：

```text
rank_damage_gap > 0:
    source 清零比 nearest-control 更伤目标 token rank。

delta_logit_gap < 0:
    source 清零比 nearest-control 更降低目标 token logit。
```

### 实际结果

运行完整性：

```text
done rows = 96
csv rows including header = 97
error rows = 0
```

node-source summary：

| node_source | condition | n | mean_rank_damage | median_rank_damage | positive_rank_damage_rate | mean_delta_target_logit |
|---|---:|---:|---:|---:|---:|---:|
| source | clean | 16 | +3.25 | 0.00 | 0.4375 | -1.0273 |
| source | answer_mask | 16 | +16.06 | +1.00 | 0.5000 | -0.5469 |
| source | union_mask | 16 | +6.00 | +4.00 | 0.8750 | -0.4531 |
| nearest_control | clean | 16 | +0.625 | 0.00 | 0.3750 | -0.5117 |
| nearest_control | answer_mask | 16 | -2.94 | 0.00 | 0.3125 | -0.2969 |
| nearest_control | union_mask | 16 | +1.06 | 0.00 | 0.3125 | -0.1016 |

source-minus-nearest pair-level bootstrap：

| condition | metric | n | mean | bootstrap 95% CI | direction |
|---|---|---:|---:|---|---|
| clean | rank_damage_gap | 16 | +2.625 | [+0.375, +5.564] | source more damaging |
| clean | delta_logit_gap | 16 | -0.516 | [-0.930, -0.117] | source lowers target logit more |
| answer_mask | rank_damage_gap | 16 | +19.000 | [+1.563, +47.377] | positive, strong-case driven |
| answer_mask | delta_logit_gap | 16 | -0.250 | [-0.770, +0.102] | weak / CI crosses 0 |
| union_mask | rank_damage_gap | 16 | +4.938 | [+2.000, +8.563] | stable |
| union_mask | delta_logit_gap | 16 | -0.352 | [-0.766, -0.043] | stable |

source-minus-nearest sample-run bootstrap：

| condition | metric | n | mean | bootstrap 95% CI | direction |
|---|---|---:|---:|---|---|
| clean | rank_damage_gap | 9 | +2.611 | [+0.111, +6.167] | positive |
| clean | delta_logit_gap | 9 | -0.326 | [-0.762, +0.125] | weak |
| answer_mask | rank_damage_gap | 9 | +11.870 | [+0.185, +32.704] | positive, heterogeneous |
| answer_mask | delta_logit_gap | 9 | -0.170 | [-0.431, +0.038] | weak |
| union_mask | rank_damage_gap | 9 | +4.167 | [+1.833, +6.667] | stable |
| union_mask | delta_logit_gap | 9 | -0.277 | [-0.468, -0.097] | stable |

关键 case：

```text
okvqa_val_2683965:
answer_mask sample-run rank_damage_gap = +93.67
answer_mask delta_logit_gap = -1.0625
union_mask rank_damage_gap = +9.00
union_mask delta_logit_gap = -0.875
```

```text
okvqa_val_1740705:
answer/union rank_damage_gap = +10.00
clean rank_damage_gap = +16.00
```

```text
okvqa_val_80655:
union rank_damage_gap = +7.00
answer rank_damage_gap = -1.50
```

### 预期与实际偏差

预期：

```text
source 清零整体应比 nearest-control 更伤下一答案 token；
answer_mask 和 union_mask 应比 clean 更相关。
```

实际：

```text
union_mask 是最稳条件：rank_damage_gap 和 delta_logit_gap 在 pair-level 与 sample-run level 都方向稳定，CI 不跨 0。
answer_mask 均值很强，但主要受 okvqa_val_2683965 等强 case 驱动，delta_logit_gap CI 跨 0。
clean 也有 source-minus-nearest rank/logit gap，说明 source nodes 本身在 clean generation distribution 中已有作用，不是只在遮挡条件下才出现。
```

### 结论

实验 012 比实验 011 更稳地支持：

```text
support source node zeroing can damage first-answer-token distribution more than nearest-control zeroing, especially under union evidence-region masking.
```

中文：

```text
support source node 清零确实能比 nearest-control 更明显伤害下一答案 token 分布，尤其在 union evidence-region mask 下最稳定。
```

但仍然要保守：

```text
这仍是 first-token bridge，不是完整 decoded answer bridge。
answer_mask 结果存在强 case 驱动。
clean 条件也有 gap，说明不能简单解释成“只有证据遮挡时 source 才重要”。
```

### 对主 claim 的影响

这一步明显增强 Stage 2B：

```text
之前 4-case smoke 只能说工程可行、有一个强 case；
现在 nearest8 expansion 可以说 first-token bridge 有稳定方向，尤其 union_mask 下 source > nearest。
```

它支持更保守的主结论补充：

```text
The same traced support nodes that are evidence-region sensitive also show generation-side relevance: zeroing them damages first-answer-token rank/logit more than matched nearest controls in the Stage 2A nearest8 pack, most consistently under union-mask conditions.
```

但不能升级为：

```text
node zeroing directly changes natural decoded answers.
```

### 后续动作

下一步建议：

```text
1. 把 first-token bridge 写入 Stage 2 主报告，作为 node-to-generation 的初步桥；
2. 选择 okvqa_val_2683965 做 decoded generation loop 单 case；
3. 如果 decoded loop 能跑通，再扩到 3 个 case，而不是直接大跑。
```

---

## 13. 实验 013：Stage 2B greedy decoded loop smoke

### 目的

在实验 012 的 first-token bridge 之后，进一步测试 node intervention 是否能进入完整短答案生成。由于旧的 `feature_intervention_generate` 不兼容当前多模态 batch，本实验新增手写 greedy loop。

核心问题：

```text
source node zeroing 是否能改变 decoded short answer？
这种改变是否强于 nearest-control？
first-token rank/logit damage 是否一定会转化为 decoded answer change？
```

### 输入

新增脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_region_mask_node_greedy_decode_smoke.py
```

运行 case：

```text
okvqa_val_2683965 / pair010
okvqa_val_1927165 / pair003
```

### 输出

```text
annotation/stage2b_node_generation_smoke/stage2b_greedy_decode_2683965_pair010.csv
annotation/stage2b_node_generation_smoke/stage2b_greedy_decode_2683965_pair010.log
annotation/stage2b_node_generation_smoke/stage2b_greedy_decode_1927165_pair003.csv
annotation/stage2b_node_generation_smoke/stage2b_greedy_decode_1927165_pair003.log
doc/experiments/stage2/013_stage2b_greedy_decoded_loop_smoke.md
```

### 方法

手写 greedy loop：

```text
1. 从 multimodal batch 开始，保留 image；
2. 每一步用 forward_from_batch 或 feature_intervention 得到 next-token logits；
3. 取 argmax token；
4. 把 token append 到 input_ids 和 attention_mask；
5. 重复 max_new_tokens = 6；
6. 比较 baseline short answer 与 intervention short answer。
```

intervention 方式：

```text
source / nearest-control feature 在原始 feature_pos 上每一步清零。
```

条件：

```text
clean
answer_mask
union_mask
```

### 实际结果

#### `okvqa_val_2683965 / pair010`

| node_source | condition | baseline answer | intervention answer | changed |
|---|---|---|---|---|
| source | clean | oval | oval | false |
| source | answer_mask | square | square | false |
| source | union_mask | rectangle | rectangle | false |
| nearest_control | clean | oval | oval | false |
| nearest_control | answer_mask | square | rectangle | true |
| nearest_control | union_mask | rectangle | rectangle | false |

读法：

```text
这个 case 的 first-token source rank damage 很强；
但 source zeroing 没有改变 greedy decoded answer；
nearest-control 反而在 answer_mask 下把 decoded answer 从 square 改成 rectangle。
```

因此：

```text
first-token rank damage 不必然转化为 decoded answer change。
```

#### `okvqa_val_1927165 / pair003`

| node_source | condition | baseline answer | intervention answer | changed |
|---|---|---|---|---|
| source | clean | to halt movement | to halt or cease movement | true |
| source | answer_mask | a sign | stop | true |
| source | union_mask | stop | stop | false |
| nearest_control | clean | to halt movement | to halt movement | false |
| nearest_control | answer_mask | a sign | a sign | false |
| nearest_control | union_mask | stop | stop | false |

读法：

```text
这个 case 给出 source-specific decoded answer change；
nearest-control 三个条件下均不改变 decoded answer。
```

但要注意：

```text
clean 下 source 改变是语义近似表达变化；
answer_mask 下 source 从 a sign 变成 stop，方向更有机制意义；
union_mask baseline 已经是 stop，因此 source 不再改变答案。
```

### 预期与实际偏差

预期：

```text
okvqa_val_2683965 是 first-token 最强 case，可能也会是 decoded-loop 最强 case。
```

实际：

```text
2683965 没有 source decoded change；
1927165 反而出现 source-specific decoded change。
```

这说明：

```text
first-token rank/logit bridge 和 decoded-answer bridge 有关，但不是一一对应。
```

### 结论

实验 013 的正式判定：

```text
decoded-loop node intervention is technically feasible;
source-specific decoded answer change exists in at least one case;
but first-token damage does not guarantee decoded-answer damage.
```

中文：

```text
手写 decoded loop 在工程上跑通；至少一个 case 中 source node 清零能特异性改变短答案生成；但 first-token 损伤不保证最终 decoded answer 改变。
```

### 对主 claim 的影响

这一步补强了主 claim 的行为闭环，但仍然是 case-level：

```text
可以说：node intervention can enter decoded short-answer generation in selected cases.
不能说：source node zeroing reliably changes decoded answers across the pack.
```

### 后续动作

下一步建议：

```text
1. 不把 decoded-loop 写成主统计结论；
2. 把 1927165 作为 node-to-decoded-answer positive case；
3. 把 2683965 作为 first-token-vs-decoded dissociation case；
4. 若继续扩展，只补 2-3 个 case：1740705、80655、3794755，不要大跑。
```

---

## 实验 014：Stage 2C semantic / region case package

### 目的

把 Stage 2A 和 Stage 2B 已经跑出的关键结果整理成可进入主报告 / 论文图的 case package。

核心问题：

```text
能否找到 1-2 个高质量 case，把 evidence-region mask、route weakening、behavior change 和 node-to-generation bridge 放进同一个可解释展示里？
```

这一步的定位是：

```text
figure-ready functional case package
```

不是：

```text
object-level semantic node proof
```

### 输入

复用已有数据，不新增大规模远端实验：

```text
annotation/stage2a_region_replication_top24_nearest8/region_experiment_manifest.csv
annotation/stage2a_region_replication_top24_nearest8/analysis_route/route_weakening_iou1.csv
annotation/stage2a_region_replication_top24_nearest8/analysis_behavior/behavior_wide_iou0p05.csv
annotation/stage2a_region_replication_top24_nearest8/analysis_behavior/generation_cases_iou0p05.csv
annotation/stage2b_node_generation_smoke/analysis_first_token_nearest8/first_token_pair_gap_table.csv
annotation/stage2b_node_generation_smoke/stage2b_greedy_decode_1927165_pair003.csv
annotation/stage2b_node_generation_smoke/stage2b_greedy_decode_2683965_pair010.csv
```

新增轻量脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/map_feature_positions_to_image_tokens.py
```

### 输出

文档：

```text
doc/experiments/stage2/014_stage2c_case_package.md
```

数据与图像包：

```text
annotation/stage2c_case_package/stage2c_main_case_summary.csv
annotation/stage2c_case_package/stage2c_support_pair_shortlist.csv
annotation/stage2c_case_package/stage2c_case_package_manifest.json
annotation/stage2c_case_package/images/okvqa_val_1927165/
annotation/stage2c_case_package/images/okvqa_val_2683965/
```

远端 token-position mapping 同步结果：

```text
annotation/stage2b_node_generation_smoke/stage2c_position_map_main_cases.csv
annotation/stage2b_node_generation_smoke/stage2c_position_map_main_cases.log
annotation/stage2b_node_generation_smoke/run_stage2c_position_map_main_cases.sh
```

### 方法

1. 对两个关键 pair 做 token-position mapping：

```text
003_okvqa_val_1927165_B_support_L11_P196_F151858
010_okvqa_val_2683965_B_support_L26_P287_F39687
```

2. 合并以下信息：

```text
route weakening
source-minus-nearest gap
answer / union region-mask behavior metrics
first-token intervention gap
greedy decoded-loop intervention result
token-position mapping
answer / relate / union overlay images
```

3. 输出 case summary 和 overlay images。

### 预期

预期 `1927165` 会成为最干净的正例：

```text
source route evidence-region-sensitive
source > nearest
region mask changes decoded answer
source node intervention changes decoded answer
nearest control does not
```

预期 `2683965` 会是 strong first-token case，但 decoded loop 可能不完全一致。

### 实际结果

#### `okvqa_val_1927165`

核心读数：

```text
source node = L11/P196/F151858
nearest control = L23/P286/F130599

source token position = <image_soft_token>
source is image token pos = True
source image token index = 192

source answer-mask weakening = +1.8125
source union-mask weakening = +1.5000
source answer minus random4 = +0.9375
source union minus random4 = +0.6250

source minus nearest answer weakening = +0.8125
source minus nearest union weakening = +0.5000

answer_mask decoded generation changed = True
union_mask decoded generation changed = True

greedy decoded loop:
source clean changed = True
source answer_mask changed = True
nearest clean changed = False
nearest answer_mask changed = False
```

判定：

```text
figure-ready positive case
```

#### `okvqa_val_2683965`

核心读数：

```text
source node = L26/P287/F39687
nearest control = L11/P196/F151858

source token position = <end_of_turn>
source is image token pos = False
nearest is image token pos = True

source answer-mask weakening = +1.3750
source union-mask weakening = +1.3125
source answer minus random4 = +0.7500
source union minus random4 = +0.6875

source minus nearest answer weakening = -1.0625
source minus nearest union weakening = -0.3750

answer_mask rank damage = +241
union_mask rank damage = +41

first-token answer_mask source-minus-nearest rank gap = +203
first-token union_mask source-minus-nearest rank gap = +20

greedy decoded loop:
source clean changed = False
source answer_mask changed = False
source union_mask changed = False
nearest answer_mask changed = True
```

判定：

```text
boundary / dissociation case
```

### 预期与实际偏差

`1927165` 符合预期，是当前最适合主图展示的正例。

`2683965` 的偏差很关键：它有很强的 first-token damage 和 region behavior damage，但没有 source-specific decoded answer change。这说明：

```text
first-token distribution damage 不必然转化为最终 decoded answer change。
```

这能帮助我们把 node-to-generation bridge 写得更严谨。

### 结论

实验 014 的正式判定：

```text
Stage 2C produced one strong figure-ready positive case and one useful boundary case.
```

中文：

```text
Stage 2C 已经产出一个强正例和一个有价值的边界例子。1927165 可作为主图候选，2683965 可作为 first-token 与 decoded answer 不完全一致的保守性说明。
```

### 对主 claim 的影响

加强的 claim：

```text
在 localized strong-image-dependence cases 中，support routes 可以表现出 evidence-region sensitivity，并且在个别高质量 case 中能连接到 generation-side 变化。
```

不能升级的 claim：

```text
所有 source nodes 都位于 image tokens。
所有 source nodes 都强于 nearest controls。
first-token damage 足以稳定预测 decoded-answer change。
source nodes 是 object-level semantic nodes。
```

### 后续动作

建议下一步：

```text
1. 把 1927165 做成主图草稿；
2. 把 2683965 写成 appendix / boundary case；
3. 如果继续 Stage 2C，只补 1740705 和 80655 的 token-position mapping；
4. 然后转入 Stage 2D suppressor deep case 或 Stage 2F cross-model loader smoke。
```

---

## 实验 015：Stage 2D suppressor deep case

### 目的

把 suppressor route 从“mixed-sign 现象”推进到一个更具体的 secondary case：

```text
如果一个 node 被定义为 suppressor，那么清零它是否会在 first-token 分布上改善目标答案 token？
```

这一步只服务 secondary claim，不改变 support-route 主线。

### 输入

```text
annotation/stage2a_region_replication_top24_nearest8/region_experiment_manifest.csv
annotation/stage2a_region_replication_top24_nearest8/analysis_route/route_weakening_iou1.csv
annotation/stage2a_region_replication_top24_nearest8/analysis_behavior/behavior_wide_iou0p05.csv
annotation/stage2d_suppressor_deep_case/stage2d_first_token_suppressor_3794755.csv
```

### 输出

```text
doc/experiments/stage2/015_stage2d_suppressor_deep_case.md
annotation/stage2d_suppressor_deep_case/stage2d_suppressor_case_summary.csv
annotation/stage2d_suppressor_deep_case/analysis_first_token_suppressor/
annotation/stage2d_suppressor_deep_case/images/okvqa_val_3794755/
```

### 方法

1. 从 nearest8 pack 中筛查 suppressor pairs；
2. 发现 4 个 suppressor pairs 全部来自 `okvqa_val_3794755`；
3. 对这 4 个 pairs 跑 first-token node intervention；
4. 同时保留 source node 与 nearest-control node；
5. 尝试 decoded-loop suppressor smoke，但因超时/OOM 只作为工程边界记录。

### 预期

若 suppressor 定义成立，则清零 source suppressor 应该：

```text
target token logit 上升；
target token rank 变好或不变；
相对于 nearest control 呈现相反方向。
```

### 实际结果

First-token source-minus-nearest gap summary：

```text
condition    mean_rank_damage_gap    mean_delta_logit_gap
clean        -2.00                   +1.15625
answer_mask  -2.50                   +0.671875
union_mask   -4.75                   +0.87500
```

Suppressor 解释口径：

```text
rank_damage_gap < 0 表示 source 清零比 nearest 清零更有利于目标 token rank。
delta_logit_gap > 0 表示 source 清零比 nearest 清零更能提高目标 token logit。
```

最清楚的 pair：

```text
013 / A / union_mask:
source rank damage = -3
nearest rank damage = +5
rank gap = -8
source delta logit = +0.1875
nearest delta logit = -1.1875
delta logit gap = +1.375

016 / B / union_mask:
source rank damage = -3
nearest rank damage = +3
rank gap = -6
source delta logit = +0.125
nearest delta logit = -1.750
delta logit gap = +1.875
```

Route 层面最强的 pair：

```text
014 / A:
source answer weakening = +0.875
source union weakening = +0.875
source-minus-nearest answer = +0.6875
source-minus-nearest union = +0.6875

017 / B:
source answer weakening = +1.000
source union weakening = +1.000
source-minus-nearest answer = +0.875
source-minus-nearest union = +0.8125
```

Decoded-loop 尝试：

```text
013+016 full decoded smoke: 900s timeout, partial log 7 rows, completed rows all unchanged。
016 union4 decoded smoke: returned CSV but both rows CUDA OOM。
```

错误原因与修复：

```text
OOM 直接原因是 GPU 上存在约 33GB 的并发/残留 Python 进程；
结果丢失原因是 decoded 脚本原本只在全部 rows 结束后统一写 CSV。

已修复:
run_region_mask_node_greedy_decode_smoke.py 新增 --stream-output、--node-source、--max-rows；
每个 condition 后调用 torch.cuda.empty_cache()；
本地 py_compile 通过。
```

修复后重跑最小 split probe：

```text
pair = 016
condition = union_mask
max_new_tokens = 4
source / nearest_control 分开跑

source:
television -> television
changed = False

nearest:
television -> a television
changed = True
```

读法：

```text
decoded 工程最小 probe 已经跑通；
但当前没有 source-specific decoded-answer positive evidence。
nearest 的变化更像 article / format variation。
```

### 预期与实际偏差

First-token 结果符合预期，说明 suppressor 的符号读法有意义。

Decoded-loop 没有达到预期，但这是工程限制：

```text
full run timeout;
single-pair short run OOM;
不能把它解释为 suppressor decoded effect 不存在。
```

### 结论

实验 015 的正式判定：

```text
suppressor first-token bridge positive;
decoded-loop suppressor bridge minimally runnable but not positive.
```

中文：

```text
`3794755` 的 suppressor source 清零在 first-token 层面方向一致地改善目标 token，而且强于 nearest control；decoded 最小 probe 修复后已跑通，但 source 没有改变答案，所以 suppressor 继续作为 secondary first-token evidence。
```

### 对主 claim 的影响

加强：

```text
signed route 的 support / suppressor 区分有 first-token 侧的方向性支撑。
```

不加强：

```text
suppressor 的自然生成作用；
suppressor 的统一机制解释；
suppressor 作为主证据。
```

### 后续动作

建议暂时不扩大 suppressor 样本。若要继续，只做工程优化：

```text
1. 写 single-row decoded probe；
2. 每次只跑一个 pair / 一个 condition；
3. max_new_tokens 降到 1-2；
4. 如果仍 OOM，就把 suppressor 固定为 first-token secondary。
```

---

---

## 实验 016：Stage 2F cross-model loader smoke

### 目的

验证 `KokosDev/qwen2p5vl-7b-clt` 这套 Qwen2.5-VL CLT 资产是否能进入我们的方法链。这个实验只做 asset-level / loader-level smoke，不做跨模型机制复现，不加载完整 Qwen base model。

### 输入

```text
candidate repo:
KokosDev/qwen2p5vl-7b-clt

script:
E:\Bridging\vlm-circuit-tracing\circuit_tracer_vlm\scripts\research\run_cross_model_asset_loader_smoke.py

remote env:
/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm
```

### 输出

```text
E:\Bridging\doc\experiments\stage2\016_stage2f_cross_model_loader_smoke.md
E:\Bridging\doc\experiments\stage2\cross_model\stage2f_qwen2p5vl_clt_loader_smoke.json
E:\Bridging\doc\experiments\stage2\cross_model\stage2f_qwen2p5vl_clt_layer_shapes.csv
```

### 方法

新增只读脚本，检查：

```text
config.yaml;
model_name / n_layers / hidden_dim / feature_dim;
feature_input_hook / feature_output_hook;
layer_*.safetensors file list;
observed layer index coverage;
remote file size;
local safetensors cache completeness;
ReplacementModel backend;
optional AutoProcessor local smoke.
```

为了避免无意占用空间，脚本后来被收紧：

```text
--max-shape-downloads default = 0
shape inspection 只对已有本地完整 safetensors 执行
不自动下载完整 Qwen base model
不自动下载完整 CLT safetensors
```

首次远端 shape 尝试中断后留下的 10,485,760 bytes `.incomplete` HF cache 文件已确认路径并删除。

### 实际结果

远端正式 smoke 结果：

```text
decision.status = partial
config.status = ok
repo_files.status = ok
safetensors_count = 27
layer_file_count = 27
expected_layers = 27
observed layers = 0..26
missing_layers = []
extra_layers = []
```

config 关键信息：

```text
model_kind = transcoder_set
architecture = qwen2.5-vl
model_name = Qwen/Qwen2.5-VL-7B-Instruct
n_layers = 27
hidden_dim = 3584
feature_dim = 8192
feature_input_hook = blocks.{layer}.hook_resid_pre
feature_output_hook = blocks.{layer}.hook_resid_post
file_pattern = layer_{layer}.safetensors
```

权重文件：

```text
27 个 layer_*.safetensors 均可见；
每层 remote_size_bytes = 117,464,384；
总大小约 3.17GB decimal / 2.95GiB。
```

本地 cache：

```text
local_safetensors_count = 0
safetensors_snapshot_complete = false
lazy_clt_load.status = skipped_not_cached
```

当前 pipeline 风险：

```text
ReplacementModel.from_pretrained_and_transcoders 仍 Gemma3-oriented;
uses_gemma3_for_conditional_generation = true;
qwen_adapter_present = false.
```

### 预期与实际偏差

预期里我们把该 repo 概括成 `cross_layer_transcoder`；实际 config 里是：

```text
model_kind = transcoder_set
architecture = qwen2.5-vl
```

这不是资产失败，而是 metadata 口径不同。后续不能只用 `model_kind == cross_layer_transcoder` 做判断。

另一个偏差是服务器没有完整 CLT safetensors cache，因此本轮不尝试 lazy CLT load。这个是安全策略导致的 partial，不是机制失败。

### 结论

实验 016 的正式判定：

```text
partial asset-level pass;
full pipeline blocked by adapter/cache/base-model issues.
```

中文：

```text
Qwen2.5-VL CLT 资产是真的，config、27 层 layer 文件列表和 hook metadata 都可读；但现在没有完整 safetensors cache，AutoProcessor local check 不完整，当前 ReplacementModel 仍是 Gemma3-only，所以还不能跑 Qwen attribution / intervention / region-mask 主实验。
```

### 对主 claim 的影响

不加强主机制 claim。它只说明跨模型方向可以继续做工程适配。

不能写：

```text
跨模型复现已经完成；
Qwen2.5-VL 上也存在 evidence-region-sensitive support routes；
Qwen adapter 已完成。
```

可以写：

```text
Qwen2.5-VL CLT 是可继续推进的候选资产；
当前 blocked 点是完整权重缓存、processor/base model 适配和 ReplacementModel/Qwen hook adapter。
```

### 后续动作

下一步按成本从低到高：

```text
1. 修正 cross_model_candidate_table 里的 metadata：model_kind=transcoder_set, architecture=qwen2.5-vl；
2. 如果需要 shape，单独批准下载一个 117MB layer 做 tensor header smoke；
3. 如果要完整 CLT lazy load，需要明确接受约 3GB 权重下载；
4. 真正进入机制实验前，必须先写 Qwen adapter / hook-forward smoke；
5. 若 Qwen CLT adapter 成本过高，再转向 KokosDev/qwen2p5vl-7b-plt 做同类 smoke。
```

---

## 实验 017：Stage 2F-2 Qwen 下载、Lazy Load 与 Base Loader Smoke

### 目的

验证 `KokosDev/qwen2p5vl-7b-clt` 与 `Qwen/Qwen2.5-VL-7B-Instruct` 是否具备进入跨模型方法链的最低资产条件。这里的目标不是证明 Qwen 上也存在 evidence-region-sensitive support routes，而是确认 CLT 权重、base 权重、processor、config 和 meta-model loader 是否可用。

### 输入

```text
CLT repo: KokosDev/qwen2p5vl-7b-clt
Base model: Qwen/Qwen2.5-VL-7B-Instruct
Server HF cache: /root/autodl-tmp/tca-reasoning/data/hf_cache
```

### 输出

```text
stage2/017_stage2f_qwen_download_and_lazy_load.md
stage2/cross_model/stage2f_qwen_download_manifest.json
stage2/cross_model/stage2f_qwen_clt_layer_shapes.csv
stage2/cross_model/stage2f_qwen_base_loader_smoke.json
```

### 方法

先下载 Qwen CLT 全量 27 层 safetensors，抽查 `layer_0 / layer_13 / layer_26` 的 tensor names、shape 和 dtype，并尝试 `load_transcoder_from_hub(..., lazy_encoder=True, lazy_decoder=True)`。

随后下载 Qwen base model。HF/Xet 路线中 `model-00003 / model-00004 / model-00005` 能完成，但 `model-00001 / model-00002` 在 `cas-bridge.xethub.hf.co` 上出现 503 或速度降到几百 KB/s。为避免无限等待，改用 ModelScope fallback 补齐两个缺失 shard，再把完整 shard 放入本地 HF snapshot 目录，用本地 path 做 loader smoke。

同时修复两个工程依赖：

```text
torchvision 缺失：安装并锁定 torchvision==0.26.0+cu130
hf_transfer 缺失：安装 hf_transfer，用于 HF Hub 并行下载尝试
```

### 预期

```text
CLT 27 层完整可下载并 lazy load。
Base processor/config/meta loader 可用。
Base 5 个 safetensors shard 最终能在服务器本地形成完整 snapshot。
```

### 实际结果

```text
CLT decision.status = pass
CLT layer count = 27 / 27
CLT lazy load = ok
Qwen base local asset decision.status = pass
processor_ok = true
full_base_assets_ok = true
meta_model_ok = true
```

关键数字：

```text
CLT single layer size = 117,464,384 bytes
CLT tensor shapes: W_enc/W_dec = [8192, 3584], b_enc = [8192], b_dec = [3584]
Base weight_file_count = 5
Base weight_total_bytes = 16,584,414,560
Base parameter_count_meta = 8,292,166,656
```

### 预期与实际偏差

HF/Xet 没有稳定完成全部 base shard 下载，最终靠 ModelScope fallback 补齐。这个偏差属于网络与镜像工程问题，不是模型资产不可用。另一个偏差是 `AutoProcessor` 需要 `torchvision`，而初始环境没有安装；安装最新版 `torchvision 0.27.0` 又与当前 `torch 2.11.0+cu130` 不匹配，最终回退到 `torchvision 0.26.0+cu130` 后解决。

### 结论

```text
Stage 2F-2 Qwen asset download + lazy/base loader smoke = pass.
```

可以说：

```text
Qwen2.5-VL CLT 和 Qwen2.5-VL base model 已经具备继续做 native hook-forward smoke 的资产条件。
```

不能说：

```text
Qwen 已经接入当前 ReplacementModel。
Qwen 已经能跑 attribution / intervention。
Qwen 已经复现主机制结论。
```

### 对主 claim 的影响

不改变当前 Gemma3 主 claim，只加强 Stage 2F 的可行性：跨模型方向不再停留在 repo file list 层面，而是已经完成 Qwen CLT 全量下载、CLT lazy load、Qwen base 本地资产与 meta loader 验证。

### 后续动作

继续做 native Qwen hook-forward smoke。若 forward 和 hook shape 可读，再进入 Qwen adapter / hook mapping 的最小工程验证。

---

## 实验 018：Stage 2F-2 Qwen Native Hook-Forward Smoke

### 目的

验证完整 Qwen 本地 snapshot 是否可以执行最小图文 forward，并读到 hidden states 与普通 PyTorch layer hook activation shape。这一步仍然不是跨模型机制复现，而是 adapter 前置 smoke。

### 输入

```text
Local Qwen snapshot: /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5
Image: /root/autodl-tmp/tca-reasoning/data/okvqa/images/val2014/COCO_val2014_000000192716.jpg
Question: What does stop mean?
Layer index: 0
GPU: NVIDIA vGPU-48GB
```

### 输出

```text
stage2/018_stage2f_qwen_hook_forward_smoke.md
stage2/cross_model/stage2f_qwen_hook_forward_smoke.json
```

### 方法

使用 `AutoProcessor` 处理一张 OK-VQA 图片和一个英文问题；用 `Qwen2_5_VLForConditionalGeneration.from_pretrained(local_snapshot, torch_dtype=bfloat16, device_map="auto")` 加载模型；在 `model.language_model.layers.0` 注册普通 PyTorch forward hook；执行 `output_hidden_states=True` 的前向传播，并记录输入 shape、logits shape、hidden states count、选中层 hidden shape 与 hook input/output shape。

### 预期

```text
processor 可加载。
5 个 checkpoint shards 可加载。
forward 可跑通。
hidden states 可读。
至少一个 native language layer module 可 hook。
```

### 实际结果

```text
processor.status = ok
model_load.status = ok
forward.status = ok
decision.status = partial
reason = native_qwen_forward_ok_but_no_replacement_model_adapter
```

关键 shape：

```text
input_ids = [1, 440]
pixel_values = [1656, 1176]
logits = [1, 440, 152064]
hidden_states_count = 29
selected_hidden_shape = [1, 440, 3584]
hook module = model.language_model.layers.0
hook input_shape = [[1, 440, 3584]]
hook output_shape = [[1, 440, 3584]]
```

GPU：

```text
free_gb before = 46.986
free_gb after = 31.148
```

### 预期与实际偏差

native Qwen forward 完全跑通，符合预期。主要偏差在 hook 命名和 pipeline 兼容性：Qwen native module 名是 `model.language_model.layers.0`，而 CLT metadata 使用的是 `blocks.{layer}.hook_resid_pre/post`。这说明后续需要 Qwen-specific adapter，把 native hidden states / module hooks 映射到 CLT 期望 hook point。

### 结论

```text
Stage 2F-2 Qwen native hook-forward smoke = partial success.
```

`partial` 的原因不是 native forward 失败，而是当前主 `ReplacementModel` 仍是 Gemma3-oriented。Qwen 现在已经通过资产、base loader、native forward、hidden-state readout 和普通 hook shape 检查，但还不能直接进入 attribution / intervention 主实验。

### 对主 claim 的影响

不加强当前 Gemma3 evidence-region-sensitive route 主结论，但显著降低跨模型支线的不确定性。现在可以把 Stage 2F 的下一步写成 Qwen adapter / CLT feature readout smoke，而不是继续停在“有没有现成资产”。

### 后续动作

```text
1. 建立 Qwen native module 到 CLT hook metadata 的映射。
2. 验证某一层 hidden state 能否送入对应 CLT layer encoder。
3. 做 single-layer feature activation readout smoke。
4. adapter 成功后，再考虑极小规模 Qwen attribution / region-mask mini replication。
```

---

## 19. 后续实验记录模板

每个新实验必须按下面结构追加：

```markdown
## 实验 XXX：实验名

### 目的

### 输入

### 输出

### 方法

### 预期

### 实际结果

### 预期与实际偏差

### 结论

### 对主 claim 的影响

### 后续动作
```
---

## 实验 019：Stage 2F Qwen CLT Feature Readout Smoke

### 目的

本实验推进 `runplan_cross_model_stage2f_qwen_llava.md` 中的 Qwen Q1。它要验证：

```text
Qwen2.5-VL native forward 产生的 hidden state，
是否能进入 KokosDev/qwen2p5vl-7b-clt 的 encoder，
并得到可读的 feature activations。
```

这一步不是 attribution，不是 intervention，也不是跨模型机制复现。它只是确认 Qwen public CLT 资产已经能进入我们方法链的 feature-readout 层。

### 输入

```text
Base model snapshot:
/root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5

Transcoder:
KokosDev/qwen2p5vl-7b-clt

Image:
/root/autodl-tmp/tca-reasoning/data/okvqa/images/val2014/COCO_val2014_000000192716.jpg

Question:
What does stop mean?

Layers:
0, 13, 26
```

### 输出

```text
stage2/019_stage2f_qwen_clt_feature_readout_smoke.md
stage2/cross_model/stage2f_qwen_clt_feature_readout_smoke.json
```

### 方法

使用新增脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_qwen_clt_feature_readout_smoke.py
```

执行步骤：

```text
1. 加载 Qwen processor。
2. 加载 Qwen2.5-VL-7B base model。
3. lazy 加载 Qwen CLT / TranscoderSet。
4. 对一个 OK-VQA image-question 跑 native forward。
5. 读取 hidden_states。
6. 对 layer 0 / 13 / 26 调用 transcoders.encode_layer(hidden, layer)。
7. 记录 feature shape、active count、max activation、top-k feature。
```

### 预期

```text
hidden state shape 应该是 [1, seq_len, 3584]。
CLT feature shape 应该是 [1, seq_len, 8192]。
encode_layer 不应报错。
```

### 实际结果

最终状态：

```text
decision.status = pass_adapter_readout
reason = native_qwen_hidden_states_encoded_by_qwen_clt
```

核心 shape：

```text
input_ids = [1, 440]
logits = [1, 440, 152064]
hidden_states_count = 29
```

Feature readout：

```text
layer 0:
  hidden = [1, 440, 3584]
  features = [1, 440, 8192]
  active_positive_count = 795736
  max_activation = 11.0625

layer 13:
  hidden = [1, 440, 3584]
  features = [1, 440, 8192]
  active_positive_count = 2419292
  max_activation = 1712.0

layer 26:
  hidden = [1, 440, 3584]
  features = [1, 440, 8192]
  active_positive_count = 1881607
  max_activation = 880.0
```

### 预期与实际偏差

shape 与 encode 全部符合预期。主要偏差是 layer 13 和 layer 26 的最高激活集中在 `position 2`，这提示当前还没有完成 token / image / prompt position 对齐。也就是说，这些 top features 现在还不能解释为 answer-adjacent route，只能解释为“Qwen hidden state 可以进入 CLT feature space”。

### 结论

```text
Stage 2F Qwen Q1 = pass_adapter_readout。
```

可以说：

```text
Qwen2.5-VL 已通过 asset loading、native hook-forward、CLT feature-readout feasibility。
```

不能说：

```text
Qwen 已经复现 Gemma3 evidence-region-sensitive support routes。
```

### 对主 claim 的影响

不改变当前 Gemma3 主结论，但显著推进 cross-model feasibility。Qwen 现在不再只是“资产可用”，而是已经能做 hidden-state-to-feature readout。下一步应进入 Qwen Q2：token / position mapping，再进入 clean vs answer-mask feature readout。

### 后续动作

```text
1. 做 Qwen token / position mapping smoke。
2. 排除特殊 token / prefix position 主导的假信号。
3. 用已有 localized mask case 做 clean vs answer-mask feature readout。
4. 若方向稳定，再考虑 minimal intervention adapter。
```

---

## 实验 020：Stage 2F LLaVA Asset Format Smoke

### 目的

本实验推进 LLaVA / Llama-family VLM 路线的 L0。它要验证：

```text
KokosDev/llava15-7b-clt 是否是真实可读的 LLaVA CLT-like 资产；
它的 transcoder_L*.pt 和 mapping_L*.pt 是否能下载和 torch.load；
是否有足够 tensor shape 信息支持后续 adapter。
```

这里的“Llama 上实验”指 LLaVA-1.5 这类 Llama-family VLM，不是纯 Llama language-only model。

### 输入

```text
Transcoder repo:
KokosDev/llava15-7b-clt

Base model:
llava-hf/llava-1.5-7b-hf

Layer:
0

Files:
transcoder_L0.pt
mapping_L0.pt
```

### 输出

```text
stage2/020_stage2f_llava_asset_format_smoke.md
stage2/cross_model/stage2f_llava_asset_format_smoke.json
```

### 方法

使用新增脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_llava_asset_format_smoke.py
```

执行步骤：

```text
1. 用 Hugging Face API 读取 repo file list。
2. 选择 layer 0 的 transcoder 与 mapping 文件。
3. 下载 transcoder_L0.pt 和 mapping_L0.pt。
4. 用 torch.load(map_location="cpu") 读取。
5. 记录 dict keys、tensor shape、dtype、metadata。
```

### 预期

```text
LLaVA 资产可能不是当前 loader 的 config.yaml + layer_*.safetensors 格式。
如果 .pt 文件可读，就进入 format adapter 设计；
如果 .pt 不可读，则 LLaVA 暂时降级。
```

### 实际结果

最终状态：

```text
decision.status = pass_format_readable
reason = sample_pt_files_downloaded_and_torch_loaded
```

Repo：

```text
file_count = 65
pt_file_count = 62
has_config_yaml = false
has_readme = true
```

文件覆盖：

```text
mapping_L0.pt ... mapping_L30.pt
transcoder_L0.pt ... transcoder_L30.pt
```

样本文件：

```text
transcoder_L0.pt size = 268,480,223 bytes
mapping_L0.pt size = 201,328,637 bytes
```

`transcoder_L0.pt`：

```text
type = dict
layer = 0
hidden_dim = 4096
feature_dim = 8192

_orig_mod.enc.1.weight = [8192, 4096], bfloat16
_orig_mod.enc.1.bias = [8192], bfloat16
_orig_mod.dec.weight = [4096, 8192], bfloat16
_orig_mod.dec.bias = [4096], bfloat16
mlp_to_clt_mapping = [4096, 8192], float32
```

`mapping_L0.pt`：

```text
mlp_to_clt_mapping = [4096, 8192], float32
decoder_weights = [4096, 8192], bfloat16
description = MLP neuron -> CLT feature correlations from training data
```

### 预期与实际偏差

资产比预想中更可读：`.pt` 和 mapping 都能成功 `torch.load`，而且 tensor shape 很清楚。偏差在于它不是当前 Qwen/Gemma 风格的 `config.yaml + layer_*.safetensors + W_enc/W_dec`，所以不能直接调用现有 `load_transcoder_from_hub`。LLaVA 需要单独 adapter。

### 结论

```text
Stage 2F LLaVA L0 = pass_format_readable。
```

可以说：

```text
LLaVA-1.5 是可继续推进的 Llama-family VLM cross-model 候选。
```

不能说：

```text
LLaVA 已经能跑 feature readout / intervention / region-mask replication。
```

### 对主 claim 的影响

不改变主 claim。这个结果只说明 LLaVA 不是“空路线”，但还没有进入 base model loader、native hook-forward 或 feature readout。

### 后续动作

```text
1. 做 LLaVA base processor/config smoke。
2. 若空间允许，下载 base weights 或至少做 meta/config loader。
3. 做 LLaVA native hook-forward smoke。
4. 写 .pt/mapping adapter，把 hidden state [batch, seq, 4096] 映射到 feature activation [batch, seq, 8192]。
```

---

## 实验 021：Stage 2F Qwen Token / Position Mapping Smoke

### 目的

本实验对应 Stage 2F-3 的 Qwen Q2。实验 019 已证明 Qwen native hidden state 能进入 Qwen CLT encoder，但当时 layer 13 / layer 26 的全局 top activation 大量集中在 `position 2`。因此，本实验专门回答：

```text
Qwen2.5-VL 的 input sequence 中，image span、question span、assistant prefix、last prompt token 和 position 2 分别在哪里？
position 2 是否属于视觉证据/答案附近位置，还是系统模板位置？
```

这一步只做 token / position alignment 和 feature readout 诊断，不做 attribution、不做 intervention，也不写成跨模型机制复现。

### 输入

```text
Base model snapshot:
/root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5

Transcoder:
KokosDev/qwen2p5vl-7b-clt

Smoke image-question:
COCO_val2014_000000192716.jpg / What does stop mean?

Main case:
okvqa_val_2847255 / COCO_val2014_000000284725.jpg

Layers:
0, 13, 26
```

### 输出

```text
doc/experiments/stage2/021_stage2f_qwen_token_position_mapping_smoke.md
doc/experiments/stage2/cross_model/stage2f_qwen_position_mapping_smoke_192716.json
doc/experiments/stage2/cross_model/stage2f_qwen_position_mapping_smoke_192716_tokens.csv
doc/experiments/stage2/cross_model/stage2f_qwen_position_mapping_smoke_192716_buckets.csv
doc/experiments/stage2/cross_model/stage2f_qwen_position_mapping_2847255.json
doc/experiments/stage2/cross_model/stage2f_qwen_position_mapping_2847255_tokens.csv
doc/experiments/stage2/cross_model/stage2f_qwen_position_mapping_2847255_buckets.csv
```

新增脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_qwen_token_position_mapping_smoke.py
```

### 方法

脚本用 Qwen processor 构造图文 prompt，读取 `input_ids` 和 token text，然后按以下规则打 bucket：

```text
special/template: 系统 prompt、模板换行、chat template 等非问题主体位置
image_marker_or_span: <|vision_start|> 到 <|vision_end|> 之间的视觉 token span
question: tokenizer 子序列匹配得到的问题文本 span
assistant_prefix: question span 之后的 assistant 起始 token，不匹配 system prompt 中的 "assistant"
last_prompt_token: 生成前最后一个 prompt token
position_2_diagnostic: 专门诊断之前异常高激活的 position 2
```

随后在 layer `0 / 13 / 26` 上对每个 bucket 读取 top CLT features，检查高激活是否来自视觉/问题位置，还是来自模板位置。

### 预期

```text
至少能稳定定位 question span 和 last prompt token。
如果 image span 无法精确定位，Q2 也可以判 partial。
position 2 若是 special/template，则后续 Q3 不再把全局 top activation 当成视觉 readout。
```

### 实际结果

两个样本均通过：

```text
decision.status = pass_position_mapping
```

Smoke sample `COCO_val2014_000000192716`：

```text
sequence_length = 440
image_span = [14, 430]
question_span = [430, 435]
assistant_start = 438
last_prompt_token = 439
position_2 token_id = 198
position_2 token_text = Ċ
```

Main case `okvqa_val_2847255`：

```text
sequence_length = 402
image_span = [14, 361]
question_span = [361, 397]
assistant_start = 400
last_prompt_token = 401
position_2 token_id = 198
position_2 token_text = Ċ
```

关键诊断：

```text
position 2 是 system/template newline token，不是 image token、question token、assistant prefix 或 answer-adjacent token。
```

### 预期与实际偏差

实际结果比最低预期更好：image span、question span、assistant prefix 和 last prompt token 都能定位。过程中发现一个重要 bug：初版脚本曾把 system prompt 中的 “helpful assistant” 误标为 assistant prefix。修复后改为只在 question span 之后寻找 assistant 起始位置，并重新运行 Q2/Q3。

### 结论

```text
Stage 2F Qwen Q2 = pass_position_mapping。
```

可以写：

```text
Qwen token/position mapping 可行；之前的 position 2 高激活是模板位置信号，不能解释成视觉证据或答案附近路径。
```

不能写：

```text
Qwen 已经发现 answer route；
Qwen 已经复现 Gemma3 evidence-region-sensitive support routes。
```

### 对主 claim 的影响

不改变 Gemma3 主 claim。它为 Qwen Q3 提供必要前提：后续 Qwen readout 必须按 `image_marker_or_span / question / assistant_prefix / last_prompt_token` 分 bucket 分析，不能用全局 top activation 直接讲机制。

### 后续动作

继续 Qwen Q3：在已有人工 evidence masks 上做 clean vs answer/union mask feature readout，并只写成 readout-level evidence-region sensitivity。

---

## 实验 022：Stage 2F Qwen Clean vs Evidence-Mask Feature Readout

### 目的

本实验对应 Stage 2F-3 的 Qwen Q3。在 Q2 已经完成 token/position mapping 后，本实验检查：

```text
Qwen2.5-VL 的 CLT feature readout 是否会被人工 answer / union evidence mask 改变？
这种改变是否集中在 image span，而不是模板位置？
```

这一步仍然不是因果实验。它不做 source tracing、不做 feature zeroing、不做 nearest node control，也不写成 cross-model mechanism replication。

### 输入

```text
Samples:
okvqa_val_2847255
okvqa_val_4157235
okvqa_val_3605295

Prompts:
B_direct
D_visual_only

Conditions:
clean
answer_mask
union_mask

Layers:
0, 13, 26

Mask sources:
annotation/okvqa_evidence_labelme_round4_core24_easy
annotation/okvqa_evidence_labelme_round4_mainline16
stage2f_cross_model/qwen_q3_assets/exported_masks
```

### 输出

```text
doc/experiments/stage2/022_stage2f_qwen_clean_vs_mask_feature_readout.md
doc/experiments/stage2/cross_model/stage2f_qwen_clean_vs_mask_feature_readout.json
doc/experiments/stage2/cross_model/stage2f_qwen_clean_vs_mask_feature_readout.csv
doc/experiments/stage2/cross_model/stage2f_qwen_clean_vs_mask_feature_readout_summary.csv
```

新增脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_qwen_clean_vs_mask_feature_readout.py
```

### 方法

对每个可用样本、prompt、condition、layer 和 token bucket：

```text
1. 运行 Qwen native forward。
2. 读取指定层 hidden states。
3. 用 Qwen CLT encoder 得到 feature activations。
4. 在 clean 条件中为每个 bucket 固定 top-k feature set。
5. 对 answer_mask / union_mask 计算 clean_activation - masked_activation。
```

主读数：

```text
mean_topk_drop = clean top-k features 在 mask 后的平均下降
topk_jaccard_change = 1 - Jaccard(clean_topk, masked_topk)
bucket_mean_shift = clean bucket feature mean - masked bucket feature mean
```

bucket：

```text
image_marker_or_span
question
assistant_prefix
last_prompt_token
position_2_diagnostic
```

### 预期

```text
3 个主 case 都能跑 clean / answer_mask / union_mask。
若 Qwen 有 readout-level evidence sensitivity，应优先出现在 image span，而不是 position 2。
```

### 实际结果

最终判定：

```text
decision.status = partial_mask_readout
usable_samples = 2 / 3
skipped_samples = [okvqa_val_3605295]
summary_rows = 120
detail_rows = 2400
```

mask 可用性：

```text
okvqa_val_2847255 = ok
okvqa_val_4157235 = ok
okvqa_val_3605295 = missing mask / missing LabelMe JSON
```

关键 aggregate 结果，按 2 个可用样本 × 2 个 prompt 平均：

```text
Layer 0 image bucket, answer_mask:
mean_topk_drop = +0.6805
bucket_mean_shift = +0.0127
topk_jaccard_change = 0.5926

Layer 0 image bucket, union_mask:
mean_topk_drop = +0.5355
bucket_mean_shift = +0.0191
topk_jaccard_change = 0.5165

Layer 13 image bucket, answer_mask:
mean_topk_drop = -0.0563
bucket_mean_shift = +0.1597
topk_jaccard_change = 0.0476

Layer 13 image bucket, union_mask:
mean_topk_drop = +1.3750
bucket_mean_shift = +0.3115
topk_jaccard_change = 0.0476

Layer 26 image bucket, answer_mask:
mean_topk_drop = +23.0203
bucket_mean_shift = +0.0968
topk_jaccard_change = 0.7879

Layer 26 image bucket, union_mask:
mean_topk_drop = +23.9453
bucket_mean_shift = +0.0360
topk_jaccard_change = 0.8225
```

对照诊断：

```text
position_2_diagnostic 在 answer/union mask 下基本不变。
question / last_prompt bucket 的变化弱且不稳定。
```

### 预期与实际偏差

偏差一：计划中 3 个样本只有 2 个有可用 mask，`okvqa_val_3605295` 缺少 answer/relate/exported mask 或 LabelMe JSON。本轮没有临时制造假 mask，也没有临时换样本，因此判为 partial。

偏差二：Qwen readout-level 变化明显集中在 layer 26 的 image span，而不是 question 或 last prompt token。这让结果更像视觉 readout 诊断，但仍不能升级成 causal route。

### 结论

```text
Stage 2F Qwen Q3 = partial_mask_readout with readout-level positive signal。
```

可以写：

```text
Qwen2.5-VL 的 CLT feature readout 对人工 evidence mask 有可观察反应；该反应主要集中在 image span，尤其 layer 26。
```

不能写：

```text
Qwen 已经复现 causal support route；
Qwen source node 强于 nearest control；
Qwen 已经完成跨模型机制复现。
```

### 对主 claim 的影响

不加强 Gemma3 主机制 claim。它支持 Stage 2F 的 feasibility 读法：Qwen 不只是能加载和 forward，也能在 image span feature readout 上看到 evidence-mask sensitivity。最强主证据仍来自 Gemma3 localized evidence-region-sensitive support route。

### 后续动作

```text
1. 补齐 `okvqa_val_3605295` mask，或换成已有 mask 的第三个主 case，把 Q3 从 2/3 补到 3/3。
2. 基于 layer 26 image-span sensitive features，设计 Qwen minimal intervention adapter。
3. 在 adapter 完成前，不做 attribution / intervention / source-control claim。
```

---

## 实验 023：Stage 2F LLaVA Base / Hook-Forward Smoke

### 目的

本实验对应 Stage 2F-3 的 LLaVA L1/L2。实验 020 已证明 `KokosDev/llava15-7b-clt` 的 `.pt` / mapping 资产可读。本轮继续检查：

```text
llava-hf/llava-1.5-7b-hf 的 processor/config 是否可读？
如果下载和显存允许，base model 是否可以跑 native hook-forward？
```

本实验不是 LLaVA feature readout，也不是跨模型机制复现。

### 输入

```text
Base model:
llava-hf/llava-1.5-7b-hf

Test image:
COCO_val2014_000000192716.jpg

Question:
What does stop mean?
```

### 输出

```text
doc/experiments/stage2/023_stage2f_llava_base_hook_forward_smoke.md
doc/experiments/stage2/cross_model/stage2f_llava_base_hook_forward_smoke.json
```

新增脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_llava_base_hook_forward_smoke.py
```

### 方法

脚本先读取 `AutoProcessor` 和 `AutoConfig`。若磁盘和 GPU 允许，则下载 `llava-hf/llava-1.5-7b-hf` base weights，加载模型，跑一个图文 forward，并记录 hidden states 与 language layer hook shape。

实际执行中，processor/config 成功后开始下载 base shards，但 HF/Xet 长期只有约 `160-300KB/s`，13 分钟后 shard 仍只有约 `3-5%`。为了不让 LLaVA 支线卡住 Qwen 主线，本轮中止 full-base 下载，并用 `--skip-base-forward` 写入 partial JSON。

### 预期

```text
processor/config 至少可读。
如果网络正常，base model 可下载并完成 native hook-forward smoke。
```

### 实际结果

最终判定：

```text
decision.status = partial_base_asset_pass
reason = processor_config_ok_but_base_forward_skipped_after_slow_hf_xet_download
```

Processor：

```text
processor.status = ok
processor_class = LlavaProcessor
has_tokenizer = true
```

Config：

```text
config.status = ok
model_type = llava
architectures = [LlavaForConditionalGeneration]
text_hidden_size = 4096
text_num_hidden_layers = 32
vision_config_type = CLIPVisionConfig
```

Base / hook：

```text
base weights not fully downloaded
model not loaded
native forward not run
hook shape not available
```

### 预期与实际偏差

LLaVA processor/config 达到预期，但 base weights 下载被 HF/Xet 慢速阻塞，未完成 L2 hook-forward。这个是网络/下载工程阻塞，不是 LLaVA asset format 失败，也不是 LLaVA 机制失败。

### 结论

```text
Stage 2F LLaVA L1 = partial_base_asset_pass。
Stage 2F LLaVA L2 = blocked by slow HF/Xet base-weight download。
```

可以写：

```text
LLaVA processor/config 可读，text hidden size = 4096，与 public CLT-like asset 的 hidden_dim = 4096 对齐。
```

不能写：

```text
LLaVA native hook-forward 已经成功；
LLaVA hidden states 已经能进入 CLT adapter；
LLaVA 已经复现 evidence-region-sensitive route。
```

### 对主 claim 的影响

不改变 Gemma3 主 claim。LLaVA 现在只是 Llama-family VLM 备选线，已经通过 asset format 和 processor/config 检查，但还没到 feature readout 或 hook-forward。

### 后续动作

```text
1. 查找 ModelScope 或其他更快镜像，补齐 `llava-hf/llava-1.5-7b-hf` base weights。
2. base 权重完整后重跑 native hook-forward。
3. hook-forward 成功后，再写 `.pt` / mapping adapter。
4. 在 LLaVA adapter 完成前，不做 LLaVA evidence-region claim。
```
---

## 追加记录：2026-05-20 Stage 2F Cross-Model 收束推进

### 实验 024：Qwen Q3 三 case clean-vs-mask readout

详细文档：`doc/experiments/stage2/024_stage2f_qwen_q3_threecase_mask_readout.md`

目的：

```text
把 Qwen2.5-VL 支线从 2/3 usable 的 partial mask readout 推进到 3/3 usable；
验证 Qwen CLT readout 是否能在多个 current-mainline case 上复用 answer/union evidence mask；
继续保持 readout-only 口径，不写成 causal replication。
```

输入：

```text
model = Qwen/Qwen2.5-VL-7B-Instruct local snapshot
CLT = KokosDev/qwen2p5vl-7b-clt
samples = okvqa_val_2847255, okvqa_val_4157235, okvqa_val_3658865
prompts = B_direct, D_visual_only
conditions = clean, answer_mask, union_mask
layers = 0,13,26
```

输出：

```text
doc/experiments/stage2/cross_model/stage2f_qwen_position_mapping_3658865.json
doc/experiments/stage2/cross_model/stage2f_qwen_position_mapping_3658865_tokens.csv
doc/experiments/stage2/cross_model/stage2f_qwen_position_mapping_3658865_buckets.csv
doc/experiments/stage2/cross_model/stage2f_qwen_clean_vs_mask_feature_readout_3case.json
doc/experiments/stage2/cross_model/stage2f_qwen_clean_vs_mask_feature_readout_3case.csv
doc/experiments/stage2/cross_model/stage2f_qwen_clean_vs_mask_feature_readout_3case_summary.csv
```

结果：

```text
position mapping for okvqa_val_3658865 = pass_position_mapping
Qwen Q3 mask readout = pass_mask_readout
usable_samples = 3/3
layer 26 image bucket:
  answer_mask mean_topk_drop ≈ +15.06
  union_mask mean_topk_drop ≈ +18.44
```

结论：

```text
Qwen 支线已经不仅是 asset/load/hook feasibility；
它在 3 个 evidence-mask case 上有清楚的 readout-level evidence-region sensitivity；
但仍不能写成 attribution、intervention、source route 或跨模型 causal replication。
```

### 实验 025：LLaVA ModelScope base、hook-forward 与 layer 0 CLT readout

详细文档：`doc/experiments/stage2/025_stage2f_llava_modelscope_hook_and_feature_readout.md`

目的：

```text
绕开 HF/Xet 慢下载，用 ModelScope 下载 llava-1.5-7b-hf；
验证 LLaVA native image-text forward 和 language-layer hook；
把 LLaVA hidden state 送入 public CLT-like encoder，完成 layer 0 feature readout smoke。
```

输入：

```text
base model = swift/llava-1.5-7b-hf
local path = /root/autodl-tmp/tca-reasoning/data/modelscope_cache/swift/llava-1___5-7b-hf
CLT-like asset = KokosDev/llava15-7b-clt
test image = COCO_val2014_000000192716.jpg
question = What does stop mean?
```

结果：

```text
ModelScope download = pass
native hook-forward = pass_hook_forward
input_ids_shape = [1, 593]
pixel_values_shape = [1, 3, 336, 336]
hidden_states_count = 33
selected_hidden_shape = [1, 593, 4096]
hook module = model.language_model.layers.0
CLT layer 0 feature readout = pass_feature_readout
feature_dim = 8192
```

结论：

```text
LLaVA 支线已经通过 base/hook/feature-readout smoke；
可以继续做 token-position mapping 和 evidence-mask readout；
但本阶段仍不涉及 source node、matched control 或 causal route。
```

### 实验 026：LLaVA token/position mapping smoke

详细文档：`doc/experiments/stage2/026_stage2f_llava_token_position_mapping_smoke.md`

目的：

```text
确认 LLaVA 输入序列中的 image token span、question、assistant prefix、last prompt token 能稳定定位；
为 LLaVA clean-vs-mask readout 确定 bucket 口径。
```

输入：

```text
sample = okvqa_val_2847255
image = COCO_val2014_000000284725.jpg
model = ModelScope LLaVA local path
CLT-like asset = KokosDev/llava15-7b-clt
layer = 0
```

结果：

```text
decision.status = pass_position_mapping
sequence_length = 624
image_span = [5, 581]
image_token_count = 576
question_span = [581, 619]
assistant_span = [620, 624]
last_prompt_token = 623
```

预期与实际偏差：

```text
原计划尝试 layer 0,15,30；
实际发现 transcoder_L15.pt 首次下载速度只有约 110-170KB/s；
为避免实验卡死，本轮改为 layer 0-only smoke。
```

结论：

```text
LLaVA token/position mapping 已经可用；
layer 0 的 bucket feature readout 可用；
高层 readout 需要先单独下载 layer 15/30 transcoder。
```

### 实验 027：LLaVA 3-case clean-vs-mask feature readout

详细文档：`doc/experiments/stage2/027_stage2f_llava_clean_vs_mask_feature_readout.md`

目的：

```text
检查 LLaVA layer 0 image token span 是否对人工 answer/union evidence mask 有可观察 feature-readout 反应。
```

输入：

```text
samples = okvqa_val_2847255, okvqa_val_4157235, okvqa_val_3658865
prompts = B_direct, D_visual_only
conditions = clean, answer_mask, union_mask
layer = 0
```

输出：

```text
doc/experiments/stage2/cross_model/stage2f_llava_clean_vs_mask_feature_readout_3case.json
doc/experiments/stage2/cross_model/stage2f_llava_clean_vs_mask_feature_readout_3case.csv
doc/experiments/stage2/cross_model/stage2f_llava_clean_vs_mask_feature_readout_3case_summary.csv
```

结果：

```text
decision.status = pass_mask_readout
usable_samples = 3/3
summary rows = 60
detail rows = 1200

image_token_span:
  answer_mask mean_topk_drop ≈ +0.0747
  union_mask mean_topk_drop ≈ +0.0990

question:
  answer_mask mean_topk_drop ≈ +0.0138
  union_mask mean_topk_drop ≈ +0.0497

assistant_prefix / last_prompt_token:
  mean_topk_drop 为轻微负值
```

结论：

```text
LLaVA layer 0 readout 出现弱但一致的 evidence-mask sensitivity；
反应主要在 image_token_span，union_mask 比 answer_mask 更稳定；
这说明 LLaVA 支线已经从 load/hook feasibility 推进到 readout-level evidence-mask sensitivity。
```

保守边界：

```text
不能写 LLaVA 已经复现 Gemma3 causal support route；
不能写 source node 强于 matched control；
不能写 prompt D_visual_only 更好；
不能把 layer 0 readout 当成 answer-adjacent causal route。
```

### 后台任务：LLaVA layer 15/30 transcoder 下载

目的：

```text
为后续 LLaVA high-layer readout 准备 transcoder_L15.pt 与 transcoder_L30.pt；
避免当前交互阻塞在慢速 HF 下载上。
```

状态：

```text
server pid = 861832
remote script = /root/autodl-tmp/tca-reasoning/stage2f_cross_model/download_llava_high_layers.py
remote manifest = /root/autodl-tmp/tca-reasoning/stage2f_cross_model/stage2f_llava_high_layer_download.json
remote logs = /root/autodl-tmp/tca-reasoning/stage2f_cross_model/logs/llava_high_layer_download_*.log
```

注意：

```text
这是资产下载任务，不是机制实验；
下载完成后再补 LLaVA layer 15/30 clean-vs-mask readout；
如果高层 readout 比 layer 0 更强，才考虑 LLaVA minimal intervention adapter。
```

### 实验 028：LLaVA layer 15/30 high-layer clean-vs-mask readout

详细文档：`doc/experiments/stage2/028_stage2f_llava_high_layer_clean_vs_mask_readout.md`

目的：

```text
在 LLaVA layer 15 和 layer 30 上复用 3-case evidence-mask readout；
判断高层是否比 layer 0 更接近 evidence-sensitive image readout。
```

输入：

```text
samples = okvqa_val_2847255, okvqa_val_4157235, okvqa_val_3658865
prompts = B_direct, D_visual_only
conditions = clean, answer_mask, union_mask
layers = 15,30
```

输出：

```text
doc/experiments/stage2/cross_model/stage2f_llava_high_layer_download.json
doc/experiments/stage2/cross_model/stage2f_llava_clean_vs_mask_feature_readout_3case_layers15_30.json
doc/experiments/stage2/cross_model/stage2f_llava_clean_vs_mask_feature_readout_3case_layers15_30.csv
doc/experiments/stage2/cross_model/stage2f_llava_clean_vs_mask_feature_readout_3case_layers15_30_summary.csv
```

结果：

```text
download:
  transcoder_L15.pt = ok, 256.043MB
  transcoder_L30.pt = ok, 256.043MB

readout:
  decision.status = pass_mask_readout
  usable_samples = 3/3

image_token_span:
  layer 15 answer_mask mean_topk_drop ≈ +1.1018
  layer 15 union_mask mean_topk_drop ≈ +1.9326
  layer 30 answer_mask mean_topk_drop ≈ +0.7435
  layer 30 union_mask mean_topk_drop ≈ -2.6451
```

结论：

```text
LLaVA layer 15 明显强于 layer 0，尤其 union_mask 在 3 个 case、2 个 prompt 下都为正；
layer 30 结果异质，不适合作为主候选；
下一步若推进 LLaVA intervention，应优先选择 layer 15，而不是 layer 30。
```

保守边界：

```text
这仍然是 readout-only；
不能写 LLaVA causal route replication；
不能写 source > nearest control；
不能写 prompt modulation。
```

### 实验 029：Stage 2G cross-model 复盘表

详细文档：`doc/experiments/stage2/029_stage2g_cross_model_recap.md`

目的：

```text
把 Gemma3 / Qwen2.5-VL / LLaVA 放到同一条证据阶梯中，
明确区分 readout replication、intervention replication 和 source-control causal replication，
避免把 Qwen/LLaVA 的 feature readout 误写成 causal route 复现。
```

输入：

```text
Qwen:
  stage2f_qwen_clean_vs_mask_feature_readout_3case_summary.csv

LLaVA:
  stage2f_llava_clean_vs_mask_feature_readout_3case_summary.csv
  stage2f_llava_clean_vs_mask_feature_readout_3case_layers15_30_summary.csv

Gemma:
  已有 source tracing / node zeroing / nearest control / random control / wrong-image / region-mask 主线结果
```

输出：

```text
doc/experiments/stage2/029_stage2g_cross_model_recap.md
doc/experiments/stage2/cross_model/stage2g_cross_model_recap.csv
```

方法：

```text
统一按三档证据整理：
1. readout replication：feature activation 对 evidence mask 有反应；
2. intervention replication：feature 干预能损伤 target answer logit/rank；
3. source-control causal replication：source route 干预强于 matched controls。
```

结果：

```text
Gemma3:
  仍是唯一完成完整 source/control causal route chain 的模型。

Qwen layer 26 image bucket:
  answer_mask mean_topk_drop ≈ +15.0573
  union_mask mean_topk_drop ≈ +18.4448
  union_mask 6/6 run 为正

LLaVA layer 15 image bucket:
  answer_mask mean_topk_drop ≈ +1.1018
  union_mask mean_topk_drop ≈ +1.9326
  union_mask 6/6 run 为正

LLaVA layer 30:
  heterogeneous，union_mask 平均为负，不作为主候选。
```

结论：

```text
当前结果不说明主结论只能在 Gemma 上有效；
更准确的说法是：Qwen 和 LLaVA 也出现 evidence-region-sensitive feature readout，
但跨模型 causal route replication 尚未证明。
```

保守边界：

```text
不能写 Qwen/LLaVA 已经复现 Gemma causal support route；
不能写 source > nearest control；
不能写 D_visual_only 更好；
不能写 feature 是对象级语义节点。
```

### 实验 030：Stage 2G feature intervention dose/sign probe

详细文档：`doc/experiments/stage2/030_stage2g_feature_intervention_dose_probe.md`

目的：

```text
把 Qwen layer 26 和 LLaVA layer 15 从 readout-level sensitivity 往 intervention replication 推一步；
检查 evidence-sensitive top features 的 decoder-direction 干预是否会稳定损伤 target answer logit/rank，
并且是否强于 activation-matched but mask-insensitive control features。
```

输入：

```text
models:
  Qwen2.5-VL-7B-Instruct + KokosDev/qwen2p5vl-7b-clt, layer 26
  LLaVA-1.5-7B + KokosDev/llava15-7b-clt, layer 15

samples:
  okvqa_val_2847255
  okvqa_val_4157235
  okvqa_val_3658865

prompts:
  B_direct
  D_visual_only
```

输出：

```text
raw:
  stage2g_qwen_feature_intervention_dose_probe.json/csv
  stage2g_llava_feature_intervention_dose_probe.json/csv

analysis:
  stage2g_feature_intervention_dose_summary.csv
  stage2g_feature_intervention_dose_specificity.csv
  stage2g_feature_intervention_dose_case_table.csv
  stage2g_feature_intervention_dose_decision.json
```

方法：

```text
1. 用 clean - union_mask feature drop 选择 evidence-sensitive top-k features；
2. 选择 clean activation 接近、但 mask drop 较小的 top-k control features；
3. 对 clean input 的目标层 hidden state 做 feature decoder-direction patch；
4. 同时测试 subtract 和 add 两种方向；
5. dose 固定为 scale 1, 3, 5；
6. 只看 first answer token / target token 的 logit、rank、top1 change。
```

结果：

```text
Qwen primary ablation:
  evidence_topk_subtract_s5 mean rank damage = 0.0000
  rank damage > 0 = 0/6
  mean logit damage = +0.1146
  logit damage > 0 = 2/6
  subtract specificity logit = -0.4375，说明 control damage 更强

Qwen signed high-dose:
  evidence_topk_add_s5 mean logit damage = +0.9063
  logit damage > 0 = 6/6
  但 rank 不变，因此只能算 partial signed response

LLaVA primary ablation:
  evidence_topk_subtract_s5 mean rank damage = -0.3333
  rank damage > 0 = 0/6
  mean logit damage = -0.0645
  subtract specificity rank/logit 都为负

LLaVA signed high-dose:
  evidence_topk_add_s5 mean rank damage = +0.8333
  rank damage > 0 = 3/6
  mean logit damage = +0.5938
  evidence-control logit specificity = +0.5690
  但这是 add direction，不是 primary ablation
```

预期与实际偏差：

```text
预期：如果 evidence-sensitive features 是 support features，
subtract/zeroing 应该损伤 target rank/logit，且强于 control。

实际：primary subtract ablation 没过；
Qwen 和 LLaVA 都只出现 partial signed high-dose response，
尤其 LLaVA add_s5 有较明显方向效应，但不能升级为 causal route replication。
```

结论：

```text
Qwen/LLaVA 的 readout-level evidence-region sensitivity 仍然成立；
但 feature-direction ablation smoke 没有建立跨模型 intervention replication。
因此当前 cross-model 结论仍应写成：

Evidence-region-sensitive feature readouts appear in Qwen and LLaVA as well,
but cross-model causal route replication remains unproven.
```

下一步：

```text
不建议继续盲目扩大 readout；
优先做 Stage 2G-5 mask-to-clean feature restoration smoke：
在 union_mask forward 中，把 clean - union 的 evidence feature contribution patch 回去，
比较 evidence_restore 是否比 control_restore 更能恢复 target logit/rank。
```

### Stage 2G-5 下一步 Run Plan：mask-to-clean feature restoration

详细文档：`doc/experiments/stage2/031_stage2g_mask_to_clean_feature_restoration_run_plan.md`

提出原因：

```text
Stage 2G dose/sign probe 显示：
readout-sensitive feature 不能直接等同于 support causal route；
clean hidden 上的 subtract/add decoder direction 也可能因为 sign、残差几何或 OOD hidden state 而不稳定。
```

核心设计：

```text
不再任意 subtract/add clean feature direction；
而是在 union_mask forward 中，
把 clean image 相对 union_mask 丢失的 evidence feature contribution patch 回去，
检查 target logit/rank 是否恢复，并与 control_restore 比较。
```

预期判定：

```text
如果 evidence_restore 比 control_restore 更能恢复 target logit/rank，
则 Qwen/LLaVA 可以从 readout replication 升级到 feature-level causal bridge；
如果不成立，则 cross-model 仍停在 readout-level evidence，不能写 causal replication。
```

### 实验 032：Stage 2G mask-to-clean feature restoration smoke

详细文档：`doc/experiments/stage2/032_stage2g_mask_to_clean_feature_restoration_smoke.md`

目的：

```text
执行 Stage 2G-5：
在 union_mask forward 中补回 clean - union_mask 丢失的 evidence feature contribution，
测试 Qwen/LLaVA 是否能从 readout replication 推进到 feature-level causal bridge。
```

输入：

```text
Qwen2.5-VL-7B-Instruct:
  CLT = KokosDev/qwen2p5vl-7b-clt
  layer = 26
  bucket = image_marker_or_span

LLaVA-1.5-7B:
  CLT-like asset = KokosDev/llava15-7b-clt
  layer = 15
  bucket = image_token_span

samples:
  okvqa_val_2847255
  okvqa_val_4157235
  okvqa_val_3658865

prompts:
  B_direct
  D_visual_only
```

输出：

```text
raw:
  stage2g_qwen_feature_restoration_smoke.json/csv
  stage2g_llava_feature_restoration_smoke.json/csv

analysis:
  stage2g_feature_restoration_baseline_gap.csv
  stage2g_feature_restoration_summary.csv
  stage2g_feature_restoration_specificity.csv
  stage2g_feature_restoration_case_table.csv
  stage2g_feature_restoration_decision.json
```

方法：

```text
1. 先比较 clean vs union_mask 的 target logit/rank；
2. 用 clean - union_mask feature drop 选择 evidence-sensitive features；
3. 选择 activation 相近但 mask drop 小的 control features；
4. 在 union_mask forward 中 patch 回 feature_drop × decoder_vector；
5. 比较 evidence_restore 和 control_restore 的 target logit/rank 恢复。
```

结果：

```text
union_mask 本身有效：
  Qwen mean clean-union logit gap = +3.9844，positive logit gap = 6/6
  Qwen mean clean-union rank gap = +90.1667，positive rank gap = 4/6
  LLaVA mean clean-union logit gap = +2.9492，positive logit gap = 6/6
  LLaVA mean clean-union rank gap = +54.0000，positive rank gap = 6/6

Qwen restoration:
  evidence_topk_restore_s1.0 mean logit restore = +0.0365
  positive logit restore = 3/6
  mean rank restore = +0.3333
  positive rank restore = 1/6
  evidence-control logit restore = +0.0052
  evidence-control rank restore = -1.0000

LLaVA restoration:
  evidence_topk_restore_s1.0 mean logit restore = -0.0313
  positive logit restore = 2/6
  mean rank restore = +1.1667
  positive rank restore = 2/6
  evidence-control logit restore = -0.0749
  evidence-control rank restore = -3.1667
```

预期与实际偏差：

```text
预期：
  evidence_restore 应该恢复 target logit/rank，并且强于 control_restore。

实际：
  Qwen 只有非常弱的 partial logit restore，rank specificity 不成立；
  LLaVA evidence_restore 不如 control_restore，logit restore 平均为负。
```

结论：

```text
Stage 2G-5 没有达到 feature-level causal bridge 成功标准。
Qwen/LLaVA 的 readout-level evidence-region sensitivity 仍然成立；
union_mask 对目标答案行为也确实有强损伤；
但当前 CLT feature-level restoration 不能证明跨模型 causal route replication。
```

当前 cross-model 口径更新为：

```text
Evidence-region-sensitive readouts and behavior damage under evidence masks are visible in Qwen and LLaVA,
but feature-level interventions/restorations do not yet establish cross-model causal route replication.
```

下一步建议：

```text
不要继续盲目扩大 feature restoration 样本；
优先做 hidden-state clean patch upper bound：
直接把 union_mask 的目标层/bucket hidden state 替换成 clean hidden state，
判断“完整 hidden state restoration”是否能恢复 target answer。
如果完整 hidden patch 都不能恢复，说明 layer/bucket 不是足够行为因果的位置；
如果完整 hidden patch 可以恢复，而 feature patch 不行，说明问题在 CLT feature selection / decoder patch。
```

### 实验 033：Stage 2G hidden-state clean patch upper bound

详细文档：`doc/experiments/stage2/033_stage2g_hidden_state_clean_patch_upper_bound.md`

目的：

```text
诊断实验 032 的 feature-level restoration 失败原因：
如果直接 patch 整个目标层/bucket hidden state 可以恢复 target answer，
说明跨模型行为桥存在，但当前 CLT feature-level patch 没有隔离出足够的因果子空间。
```

输入：

```text
Qwen2.5-VL-7B-Instruct:
  layer = 26
  bucket = image_marker_or_span

LLaVA-1.5-7B:
  layer = 15
  bucket = image_token_span

samples:
  okvqa_val_2847255
  okvqa_val_4157235
  okvqa_val_3658865

prompts:
  B_direct
  D_visual_only
```

输出：

```text
raw:
  stage2g_qwen_hidden_patch_smoke.json/csv
  stage2g_llava_hidden_patch_smoke.json/csv

analysis:
  stage2g_hidden_patch_baseline_gap.csv
  stage2g_hidden_patch_summary.csv
  stage2g_hidden_patch_case_table.csv
  stage2g_hidden_patch_decision.json
```

方法：

```text
在 union_mask forward 中：
  hidden[:, bucket_positions, :] += scale × (clean_hidden[:, bucket_positions, :] - union_hidden[:, bucket_positions, :])

scale:
  0.5, 1.0, 1.5

主读数:
  scale = 1.0，即完整替换目标 bucket hidden state。
```

结果：

```text
Qwen hidden_bucket_restore_s1.0:
  mean logit restore = +0.7656
  positive logit restore = 5/6
  mean rank restore = +58.5000
  positive rank restore = 4/6
  mean logit gap closure = +0.2004

LLaVA hidden_bucket_restore_s1.0:
  mean logit restore = +2.4128
  positive logit restore = 6/6
  mean rank restore = +50.5000
  positive rank restore = 4/6
  mean logit gap closure = +0.6669
```

预期与实际偏差：

```text
预期：
  如果 layer/bucket 本身有行为相关信息，hidden patch 应该恢复 target logit/rank。

实际：
  Qwen 和 LLaVA 都支持 hidden-state upper-bound；
  尤其 LLaVA s1.0 的 logit gap closure 很高。
```

结论：

```text
跨模型不是没有行为因果桥。
Qwen layer 26 image bucket 和 LLaVA layer 15 image bucket 的 hidden state，
都携带可恢复 target answer signal。

但这仍然不是 source-control causal route replication：
它说明 hidden-state-level upper bound 成立，
不说明 CLT feature-level specificity 或 source node specificity 成立。
```

更新后的 cross-model 口径：

```text
Qwen/LLaVA:
  readout-level evidence-region sensitivity: supported
  evidence-mask behavior damage: supported
  hidden-state patch causal bridge upper bound: supported
  CLT feature-level intervention/restoration: not supported / partial only
  source-control causal route replication: not established

Gemma3:
  full source/control causal route chain remains the strongest mainline evidence.
```

### Stage 2H Run Plan：cross-model causal localization

详细文档：`doc/experiments/stage2/034_stage2h_cross_model_causal_localization_run_plan.md`

目的：

```text
把 Stage 2G 的 whole image-bucket hidden patch 上界继续拆开，
定位 Qwen/LLaVA 的恢复效果来自哪些 image positions，
这些 positions 是否强于 random / low-delta controls，
以及是否能同时 restore masked run 和 corrupt clean run。
```

输入：

```text
samples:
  okvqa_val_2847255
  okvqa_val_4157235
  okvqa_val_3658865

models:
  Qwen layer 26 image_marker_or_span
  LLaVA layer 15 image_token_span
```

主要实验：

```text
Stage 2H-1:
  image-bucket position localization

Stage 2H-2:
  bidirectional hidden patch, restore + corrupt

Stage 2H-3:
  answer-adjacent bridge comparison

Stage 2H-4:
  decoded answer smoke, only if first-token/rank bridge passes
```

判定边界：

```text
即使 Stage 2H 成功，也只能写 hidden-state-level causal localization；
不能写 Qwen/LLaVA 已经复现 Gemma source-control causal route。
```

### Stage 2H Hidden-Position Patch Smoke：cross-model hidden-state causal localization

详细文档：`doc/experiments/stage2/035_stage2h_hidden_position_patch_smoke.md`

目的：

```text
把 Stage 2G 的 whole image-bucket hidden patch 上界拆成更小的位置组，
检查 Qwen/LLaVA 中哪些 hidden positions 能 restore union_mask run，
哪些 positions 能 corrupt clean run，
以及视觉位置是否需要 answer-adjacent text positions 共同形成答案桥接。
```

输入：

```text
models:
  Qwen2.5-VL-7B-Instruct, layer 26, image_marker_or_span
  LLaVA-1.5-7B, layer 15, image_token_span

samples:
  okvqa_val_2847255
  okvqa_val_4157235
  okvqa_val_3658865

prompts:
  B_direct
  D_visual_only
```

输出：

```text
raw:
  stage2h_qwen_hidden_position_patch.json/csv
  stage2h_llava_hidden_position_patch.json/csv

analysis:
  stage2h_hidden_position_patch_baseline_gap.csv
  stage2h_hidden_position_patch_summary.csv
  stage2h_hidden_position_patch_specificity.csv
  stage2h_hidden_position_patch_case_table.csv
  stage2h_hidden_position_patch_decision.json
```

方法：

```text
restore:
  在 union_mask forward 中，把目标 positions 的 hidden state patch 回 clean hidden state。

corrupt:
  在 clean forward 中，把目标 positions 的 hidden state patch 向 union_mask hidden state。

位置组:
  whole_bucket
  top_hidden_delta
  answer_adjacent_text
  top_hidden_delta_plus_answer_adjacent
  random_control_1..4
  low_delta_control
  LLaVA additionally: evidence_region, evidence_region_plus_answer_adjacent
```

结果：

```text
Qwen best group = top_hidden_delta_plus_answer_adjacent
  restore source-minus-random logit = +4.3255
  restore source-minus-random rank = +78.9167
  restore positive logit = 6/6
  corrupt source-minus-random logit = +3.1667
  corrupt source-minus-random rank = +17.0000
  corrupt positive logit = 6/6

LLaVA best group = top_hidden_delta_plus_answer_adjacent
  restore source-minus-random logit = +0.7137
  restore source-minus-random rank = +3.7083
  restore positive logit = 6/6
  corrupt source-minus-random logit = +0.4961
  corrupt source-minus-random rank = +3.3750
  corrupt positive logit = 6/6
```

预期与实际偏差：

```text
预期：
  如果 Stage 2G whole-bucket upper bound 不是偶然结果，
  更小的位置组应能恢复一部分 target answer signal。

实际：
  两个模型都支持 hidden-position bridge；
  但最强组不是 pure visual positions，
  而是 top hidden-delta visual positions + answer-adjacent text positions。
```

重要降级：

```text
Qwen:
  image_grid_thw 存在，但 grid 与 visual span 不可靠匹配；
  因此只写 hidden-delta source-like localization，
  不写 evidence-region token localization。

LLaVA:
  576 image tokens -> 24×24 grid 映射可用；
  但 evidence_region alone 没有强于 random controls；
  最强结果仍是 top_hidden_delta_plus_answer_adjacent。
```

结论：

```text
Stage 2H 支持 Qwen/LLaVA 的 hidden-state-level causal localization。
这加强了“跨模型不是完全没有相似视觉证据桥”的判断。

但它仍然不是 Gemma 式 source-control causal route replication；
也不是 CLT feature-level causal bridge；
更不是对象级语义节点解释。
```

下一步：

```text
优先做 Stage 2H-4 decoded answer smoke。
只对通过的 best bridge group 跑短 greedy generation。
若 decoded answer 不变，仍只写 first-token/rank bridge。
若 decoded answer 改变，才写 generation-level bridge smoke。
```

### Stage 2H-4 Decoded Answer Smoke：hidden-position bridge 到短生成答案

详细文档：`doc/experiments/stage2/036_stage2h_decoded_answer_smoke.md`

目的：

```text
检查 Stage 2H-1/2 通过的 hidden-position bridge 是否能传导到短 greedy generation。
具体问法是：
如果 union_mask 改变了 decoded answer，
把 top_hidden_delta_plus_answer_adjacent positions patch 回 clean hidden state，
是否能让生成答案回到 clean/target answer，或至少离开 union answer。
```

输入：

```text
models:
  Qwen2.5-VL-7B-Instruct, layer 26
  LLaVA-1.5-7B, layer 15

samples:
  okvqa_val_2847255
  okvqa_val_4157235
  okvqa_val_3658865

prompts:
  B_direct
  D_visual_only

generation:
  greedy
  max_new_tokens = 3
  direction = restore only
```

输出：

```text
raw:
  stage2h_qwen_decoded_answer_smoke.json/csv
  stage2h_llava_decoded_answer_smoke.json/csv

analysis:
  stage2h_decoded_answer_smoke_summary.csv
  stage2h_decoded_answer_smoke_case_table.csv
  stage2h_decoded_answer_smoke_decision.json
```

结果：

```text
Qwen:
  clean target hit = 6/6
  union target hit = 2/6
  best bridge target hit = 4/6
  informative clean-vs-union rows = 4
  best bridge changed away from union = 4/4 informative rows
  best bridge returned to clean/target = 2/4 informative rows
  low_delta_control changed away from union = 0/6
  random_control_1 changed away from union = 0/6

LLaVA:
  clean vs union decoded answer gap = 0/6
  best bridge still restores first-token rank/logit,
  but decoded answer cannot show recovery because clean and union answers are identical.
```

预期与实际偏差：

```text
预期：
  如果 hidden-position bridge 强到生成层，
  best bridge 应在 decoded answer 中恢复 clean/target answer。

实际：
  Qwen 只 partial。
  samsung case 成功从 unknown 恢复到 samsung；
  dog case 从 polo 离开，但变成 a horse，没有恢复 dog。
  LLaVA 因 clean/union 生成本来一致，本轮生成桥 underpowered。
```

结论：

```text
Qwen:
  partial decoded-answer bridge smoke。

LLaVA:
  first-token/rank bridge remains stronger evidence；
  decoded generation bridge not established in this smoke。

总体：
  Stage 2H-4 加强了 Qwen 的跨模型行为桥，
  但仍不足以把 Qwen/LLaVA 升级成 source-control causal route replication。
```

### Stage 2I-0 Cross-Model 扩样本候选 manifest

详细计划文档：`doc/experiments/stage2/037_stage2i_cross_model_expansion_run_plan.md`

目的：
```text
把 Stage 2H 的 3-case cross-model smoke 扩展成 8-12 个样本的 targeted expansion。
这一阶段不是证明 Qwen/LLaVA 已经复现 Gemma 的 source-control causal route，
而是检验 evidence-mask sensitivity、hidden-position bridge 和 decoded bridge 是否能在更多 localized cases 上复现。
```

方法：
```text
复用已有 answer/relate mask 包，统一构建候选样本表。
对每个样本自动计算 answer_area_frac / union_area_frac，
把 answer 区域接近整图的样本降级为 diffuse_or_fullscreen，
避免把“几乎全屏答案区域”的样本混入主证据。
```

输入：
```text
annotation/okvqa_evidence_labelme_round4_core24_easy/exported_masks
annotation/okvqa_evidence_labelme_round4_core16_extra/exported_masks
annotation/okvqa_evidence_labelme_round4_ultraeasy16_fresh/exported_masks
annotation/okvqa_evidence_labelme_round5_expanded20_route7/exported_masks
annotation/stage2a_region_replication_top24_nearest8/exported_masks
```

输出：
```text
doc/experiments/stage2/cross_model/stage2i_cross_model_candidate_manifest.csv
doc/experiments/stage2/cross_model/stage2i_selected_12_manifest.csv
doc/experiments/stage2/cross_model/stage2i_manifest_summary.json
```

当前结果：
```text
total_mask_rows = 68
unique_samples = 59
primary_eligible = 56
secondary_large_localized = 4
excluded_diffuse_or_fullscreen = 8
selected_count = 12
primary_answer_area_frac_mean = 0.209879
```

Stage 2I 当前选出的 12 个候选：
```text
okvqa_val_1593205  answer=tokyo          type=symbol_text_reading
okvqa_val_4739195  answer=spanish        type=symbol_text_reading
okvqa_val_4502065  answer=tortoise       type=symbol_text_reading
okvqa_val_2954205  answer=move           type=symbol_text_reading
okvqa_val_1729795  answer=arrow          type=symbol_text_reading
okvqa_val_3959785  answer=kuwait airway  type=symbol_text_reading
okvqa_val_1058855  answer=fire hydrant   type=visual_readout
okvqa_val_340155   answer=racquet        type=visual_readout
okvqa_val_01514    answer=50 pounds      type=visual_readout
okvqa_val_4043385  answer=german         type=symbol_text_reading
okvqa_val_4938465  answer=schwinn        type=symbol_text_reading
okvqa_val_2708155  answer=left           type=visual_readout
```

结论：
```text
现有标注资产足够启动 Stage 2I，不需要立刻让用户补标注。
下一步先跑 Qwen selected-8 decoded bridge expansion，
同时跑 LLaVA selected-12 generation-gap screen。
如果 LLaVA 能找到足够 clean-vs-union decoded answer gap，再跑 bridge 条件。
```

### Stage 2I-1/2 Cross-Model decoded bridge 扩样本结果

详细结果文档：`doc/experiments/stage2/039_stage2i_cross_model_expansion_results.md`

目的：
```text
把 Stage 2H 的 3-case cross-model decoded/hidden bridge 扩展到更多 localized samples。
重点检验：
1. Qwen partial decoded bridge 是否能扩样本复现；
2. LLaVA decoded generation 是否只是前一轮样本 underpowered；
3. best hidden-position bridge 是否强于 low-delta/random controls。
```

输入：
```text
Qwen:
  selected top 8 samples
  B_direct, D_visual_only
  layer 26

LLaVA:
  selected top 12 samples
  B_direct, D_visual_only
  layer 15
```

输出：
```text
doc/experiments/stage2/cross_model/stage2i_qwen_decoded_bridge_expansion.csv/json
doc/experiments/stage2/cross_model/stage2i_llava_generation_gap_screen.csv/json
doc/experiments/stage2/cross_model/stage2i_llava_decoded_bridge_expansion.csv/json
doc/experiments/stage2/cross_model/stage2i_cross_model_bridge_summary.csv
doc/experiments/stage2/cross_model/stage2i_cross_model_bridge_decision.json
doc/experiments/stage2/cross_model/stage2i_bridge_bootstrap.csv/json
```

结果：
```text
Qwen:
  status = partial_decoded_bridge_expanded
  total runs = 16
  informative clean-vs-union rows = 15
  best changed away from union = 11/15
  best same as clean = 3/15
  best target hit = 2/16
  best mean logit restore vs union = +5.109375
  best mean rank restore vs union = +1473.6875
  low_delta/random changed away from union = 1/15

LLaVA:
  status = partial_decoded_bridge_expanded
  total runs = 24
  informative clean-vs-union rows = 19
  best changed away from union = 8/19
  best same as clean = 3/19
  best target hit = 5/24
  best mean logit restore vs union = +1.636556
  best mean rank restore vs union = +42.083333
  low_delta/random changed away from union = 0/19
```

paired bootstrap：
```text
Qwen best - random:
  changed_away_from_union mean = +0.666667, 95% CI = [+0.333333, +0.933333]
  logit_restore mean = +5.247917, 95% CI = [+3.958333, +6.400000]

LLaVA best - random:
  changed_away_from_union mean = +0.421053, 95% CI = [+0.210526, +0.631579]
  logit_restore mean = +1.613487, 95% CI = [+0.903783, +2.333676]
```

预期与实际偏差：
```text
预期：Qwen 可能复现 partial bridge；LLaVA 可能继续 decoded-generation underpowered。
实际：Qwen 复现；LLaVA 扩样本后有 19/24 informative rows，并且 best bridge 强于 controls。
偏差：best bridge 不明显强于 answer_adjacent_text，说明信号可能含有较强 answer-adjacent 汇聚成分，不能写成纯 image-token route。
```

结论：
```text
Stage 2I 把跨模型证据从 readout / 3-case hidden bridge 推进到 multi-case partial decoded bridge。
Qwen 和 LLaVA 都显示 best hidden-position bridge 对 decoded answer 的影响强于 low-delta/random controls。
但这仍然不是 Gemma 式 source-control causal route replication，也不是 object-level semantic node 证明。
```

### Stage 2I-3 Cross-Model bidirectional hidden-position expansion

详细结果文档：`doc/experiments/stage2/040_stage2i_bidirectional_hidden_position_expansion.md`

目的：
```text
在 Stage 2I decoded bridge 扩样本之后，继续检查同一类 hidden-position bridge 是否也有双向因果定位：
restore: union_mask -> clean hidden patch 是否恢复 target logit/rank；
corrupt: clean -> union hidden patch 是否损伤 target logit/rank。
```

输入：
```text
samples = stage2i_selected_12_manifest.csv
models = Qwen2.5-VL-7B-Instruct layer 26, LLaVA-1.5-7B layer 15
prompts = B_direct, D_visual_only
directions = restore, corrupt
main group = top_hidden_delta_plus_answer_adjacent
controls = random_control_1..4, low_delta_control
```

输出：
```text
doc/experiments/stage2/cross_model/stage2i_qwen_hidden_position_patch.csv/json
doc/experiments/stage2/cross_model/stage2i_llava_hidden_position_patch.csv/json
doc/experiments/stage2/cross_model/stage2i_hidden_position_patch_summary.csv
doc/experiments/stage2/cross_model/stage2i_hidden_position_patch_specificity.csv
doc/experiments/stage2/cross_model/stage2i_hidden_position_patch_decision.json
doc/experiments/stage2/cross_model/stage2i_hidden_position_patch_bootstrap.csv/json
```

结果：
```text
Qwen:
  decision_status = supported_bidirectional_hidden_position_localization
  restore source_minus_random_logit = +3.333659
  restore source_minus_random_rank = +875.364584
  restore positive_logit = 20/24
  corrupt source_minus_random_logit = +2.956380
  corrupt source_minus_random_rank = +410.885417
  corrupt positive_logit = 20/24

LLaVA:
  decision_status = supported_bidirectional_hidden_position_localization
  restore source_minus_random_logit = +1.220296
  restore source_minus_random_rank = +13.604167
  restore positive_logit = 21/24
  corrupt source_minus_random_logit = +0.577840
  corrupt source_minus_random_rank = +5.468749
  corrupt positive_logit = 18/24
```

paired bootstrap:
```text
Qwen best - random_mean:
  restore logit mean = +3.333659, 95% CI = [+2.143880, +4.560547]
  corrupt logit mean = +2.956380, 95% CI = [+1.823568, +4.099609]

LLaVA best - random_mean:
  restore logit mean = +1.220296, 95% CI = [+0.696615, +1.789998]
  corrupt logit mean = +0.577840, 95% CI = [+0.325724, +0.867676]
```

预期与实际偏差：
```text
预期：restore 应该强于 controls，corrupt 可能较弱。
实际：两个模型在 restore 和 corrupt 两个方向都稳定强于 controls。
但 best group 是 top_hidden_delta_plus_answer_adjacent，不是 pure visual group。
Qwen answer-adjacent 成分很强；LLaVA visual/evidence-region positions 也有明显信号。
```

结论：
```text
Stage 2I-3 把跨模型证据升级为 multi-case bidirectional hidden-state bridge localization。
这明显加强 Qwen/LLaVA 作为 auxiliary cross-model support 的价值。
但它仍不是 CLT feature-level source tracing，也不是 Gemma-style source-control causal route replication。
```

### Stage 2J Cross-Model matched-control hidden-position specificity

详细结果文档：`doc/experiments/stage2/041_stage2j_matched_control_specificity.md`

目的：
```text
在 Stage 2I 的 random/low-delta controls 之后，加入更强的 delta-matched / activation-matched hidden-position controls。
核心问题是：
best bridge 是否只是因为 source-like positions 的 hidden delta 更大，或者 activation norm 更大？
```

方法：
```text
delta_matched_control:
  选择与 top_hidden_delta source positions 的 clean-vs-union delta norm 最接近的非 source visual positions。

activation_matched_control:
  选择与 top_hidden_delta source positions 的 clean activation norm 最接近的非 source visual positions。

delta_matched_plus_answer_adjacent / activation_matched_plus_answer_adjacent:
  在 matched visual positions 上加同一组 answer-adjacent text positions，
  用于控制 answer-adjacent 成分不变，只替换 visual/source-like 部分。
```

实现校正：
```text
第一次 smoke 发现 matched-control 分数表键值写反，导致 matched visual positions 为空。
该结果没有进入结论。
修复后确认：
Qwen matched visual positions = 32，plus-answer-adjacent = 36；
LLaVA matched visual positions = 64，plus-answer-adjacent = 68。
```

结果：
```text
Qwen:
  status = matched_control_mostly_supported
  restore combo_vs_delta_matched_plus:
    mean = +0.406250, 95% CI = [+0.096354, +0.789062]
  restore combo_vs_activation_matched_plus:
    mean = +0.338542, 95% CI = [+0.044271, +0.710938]
  corrupt combo_vs_delta_matched_plus:
    mean = +0.149089, 95% CI = [+0.042969, +0.276693]
  corrupt combo_vs_activation_matched_plus:
    mean = +0.113281, 95% CI = [-0.006510, +0.259766]

LLaVA:
  status = matched_control_partial
  restore combo_vs_delta_matched_plus:
    mean = +0.146484, 95% CI = [-0.269043, +0.549805]
  restore combo_vs_activation_matched_plus:
    mean = +1.096029, 95% CI = [+0.503906, +1.720052]
  corrupt combo_vs_delta_matched_plus:
    mean = +0.193522, 95% CI = [-0.046712, +0.493978]
  corrupt combo_vs_activation_matched_plus:
    mean = +0.225911, 95% CI = [+0.007975, +0.503581]
```

预期与实际偏差：
```text
预期：best bridge 若真有 source-like specificity，应强于 activation/delta matched controls。
实际：
Qwen 大多数 matched-control 比较仍稳定为正；
LLaVA 强于 activation-matched controls，但 delta-matched controls 吸收了相当一部分效果。
```

结论：
```text
Stage 2J 让跨模型结论更精确：
Qwen 是更强的 auxiliary cross-model replication；
LLaVA 有 hidden-state bridge 和 partial decoded bridge，但 matched-control specificity 只能写 partial / heterogeneous。
这仍不是 Gemma-style source-control causal route replication。
```

## Stage 2K Matched-Control 解释性复盘

详细结果文档：`doc/experiments/stage2/042_stage2k_matched_control_explanation_analysis.md`

目的：
```text
Stage 2J 已经显示：Qwen 的 matched-control specificity 大体成立；LLaVA 在 activation-matched control 下仍有优势，但 delta-matched control 会吸收相当一部分效果。
Stage 2K 不重新跑模型，而是对 Stage 2J 原始结果做解释性拆解，判断这个结论是否稳定、由哪些题型支撑、以及 best bridge 的优势到底来自 visual source-like positions 还是 answer-adjacent text positions。
```

输入：
```text
doc/experiments/stage2/cross_model/stage2j_qwen_matched_control_patch.csv
doc/experiments/stage2/cross_model/stage2j_llava_matched_control_patch.csv
doc/experiments/stage2/cross_model/stage2i_selected_12_manifest.csv
```

输出：
```text
doc/experiments/stage2/cross_model/stage2k_matched_control_explanation_case.csv
doc/experiments/stage2/cross_model/stage2k_matched_control_explanation_model_summary.csv
doc/experiments/stage2/cross_model/stage2k_matched_control_explanation_typed_summary.csv
doc/experiments/stage2/cross_model/stage2k_matched_control_explanation_decision.json
```

方法：
```text
固定 answer-adjacent text positions，只比较 source-like visual positions 与 delta-matched / activation-matched visual controls 的差异。
核心指标包括 combo_minus_delta_combo、combo_minus_activation_combo，以及 visual-only increment over answer-adjacent。
这样可以区分三种可能：真正的 source-like specificity、hidden delta magnitude 解释、answer-adjacent 汇聚位置解释。
```

结果：
```text
Qwen:
  status = specificity_explanation_supported
  restore combo_minus_delta = stable_positive
  restore combo_minus_activation = stable_positive
  corrupt combo_minus_delta = stable_positive
  corrupt combo_minus_activation = weak_or_heterogeneous_positive

LLaVA:
  status = delta_explanation_partially_supported
  restore combo_minus_delta = weak_or_heterogeneous_positive
  restore combo_minus_activation = stable_positive
  corrupt combo_minus_delta = weak_or_heterogeneous_positive
  corrupt combo_minus_activation = stable_positive
```

类型化结果：
```text
Qwen 的 specificity 主要由 symbol_text_reading 支撑。
Qwen visual_readout 切片较弱，不能把 Qwen 强结论泛化到所有 localized visual cases。
LLaVA 在 activation-matched control 下有稳定优势，但 delta-matched weakness 跨类型存在。
```

预期与实际偏差：
```text
预期：如果 Qwen/LLaVA 都是真正强 specificity，二者都应稳定强于 delta-matched 和 activation-matched controls。
实际：Qwen 基本符合预期；LLaVA 只稳定强于 activation-matched，delta-matched 下变弱。
这说明 LLaVA 的 bridge 有真实信号，但其中一部分可能由 clean-vs-union hidden delta magnitude 解释。
```

结论：
```text
Stage 2K 支持“Qwen 是更强的 cross-model auxiliary replication；LLaVA 是 partial / heterogeneous bridge evidence”。
这不会削弱 Gemma 主线，因为 Gemma 主线仍然拥有完整 source tracing + intervention + controls。
它对 cross-model claim 的影响是：可以写 Qwen/LLaVA 有 hidden-state-level evidence-to-answer bridge 线索，但不能写已经完成 Gemma-style source-control causal route replication。
```

## Stage 2L Cross-Model 进一步验证计划

详细计划文档：`doc/experiments/stage2/043_stage2l_cross_model_verification_run_plan.md`

目的：
```text
继续验证 Qwen/LLaVA 中的 evidence-mask-sensitive hidden-state bridge 是否具有样本外复现、类型稳定性、位置特异性、控制特异性，并且能进一步连接到 target rank / decoded answer 行为变化。
```

计划分层：
```text
1. Case panel 与失败模式面板：从现有 Stage 2I/2J/2K 结果里挑出正文候选、附录候选、失败诊断候选。
2. 扩大样本复现：从 12 个样本扩到 24 / 36 个 localized samples，重点看 Qwen 是否继续稳定、LLaVA 是否仍被 delta controls 吸收。
3. Negative controls：加入 mask-shuffled 和 wrong-target token controls，排除“任意遮挡/任意 target 都能恢复”的替代解释。
4. Evidence-to-answer 汇聚位置分析：拆分 image-only、answer-adjacent-only、image + answer-adjacent，判断视觉信号是否在答案附近汇聚。
5. Decoded generation only-on-passing-cases：只对通过 specificity 的 case 跑短生成，避免在弱 case 上过度解释。
6. Qwen feature/source tracing 工程预研：如果要升级到更接近 Gemma 主线，必须继续做 feature-level intervention 和 matched feature controls。
```

当前结论边界：
```text
如果 Stage 2L 成功，可以升级为：
Across Gemma, Qwen, and LLaVA, evidence-region perturbations reveal answer-relevant hidden-state bridges, with Qwen showing the strongest auxiliary matched-control specificity and LLaVA showing partial but weaker specificity.

但仍不能写成：
Cross-model source-control causal routes are fully replicated.
```

## Stage 2L-1 Cross-Model Case Panel 与失败模式面板

详细结果文档：`doc/experiments/stage2/044_stage2l_cross_model_case_panel.md`

目的：
```text
复用 Stage 2I/2J/2K 结果，挑出最适合进入正文或附录的跨模型 case，并把失败/诊断 case 单独列出来。
这一步回答：Qwen/LLaVA 哪些 case 真正支持 hidden-state-level evidence-to-answer bridge，哪些 case 只说明 partial / heterogeneous。
```

输入：
```text
doc/experiments/stage2/cross_model/stage2k_matched_control_explanation_case.csv
doc/experiments/stage2/cross_model/stage2i_selected_12_manifest.csv
doc/experiments/stage2/cross_model/stage2i_qwen_decoded_bridge_expansion.csv
doc/experiments/stage2/cross_model/stage2i_llava_decoded_bridge_expansion.csv
```

输出：
```text
doc/experiments/stage2/cross_model/stage2l_case_panel.csv
doc/experiments/stage2/cross_model/stage2l_case_panel_decision.json
doc/experiments/stage2/044_stage2l_cross_model_case_panel.md
```

方法：
```text
对每个 model x sample x prompt 汇总 restore/corrupt 两个方向。
核心指标是 source-like combo 是否强于 delta_matched_plus_answer_adjacent 和 activation_matched_plus_answer_adjacent。
同时记录 decoded bridge 是否改变 union_mask 下的答案，以及题型、问题、答案、answer mask 面积比例。
```

结果：
```text
总行数 = 48

category_counts:
  llava_strong_positive = 12
  llava_activation_supported = 6
  llava_delta_diagnostic = 2
  llava_weak_or_failure = 4
  qwen_strong_positive = 5
  qwen_moderate_positive = 4
  qwen_weak_or_failure = 15

recommended_counts:
  main_or_appendix_positive_case = 17
  appendix_positive_case = 10
  diagnostic_or_failure_case = 21
```

代表性 Qwen 正证据候选：
```text
okvqa_val_3959785 / kuwait airway / B_direct, D_visual_only
okvqa_val_4502065 / tortoise / B_direct, D_visual_only
okvqa_val_1058855 / fire hydrant / B_direct
```

代表性 LLaVA 正证据或诊断候选：
```text
正证据候选:
  okvqa_val_01514 / 50 pounds
  okvqa_val_1058855 / fire hydrant
  okvqa_val_1593205 / tokyo

诊断候选:
  okvqa_val_3959785 / kuwait airway
  okvqa_val_2954205 / move
  okvqa_val_1729795 / arrow
```

预期与实际偏差：
```text
预期：Qwen 会更适合作为强 auxiliary replication，LLaVA 更可能是 partial。
实际：Qwen 的强 case 较集中，尤其 symbol_text_reading；LLaVA 在 case-level 分类里出现不少 strong-positive 行，但结合 Stage 2K 的 delta-matched 结果，仍然不能直接升级为强 specificity。
```

结论：
```text
Stage 2L-1 支持继续推进实验验证。
下一步最值得做的是 Stage 2L-3 negative controls 和 Stage 2L-4 evidence-to-answer 汇聚位置分析，而不是直接把跨模型 claim 升级。
```

## Stage 2L-3 Wrong-Target Negative Control

详细结果文档：`doc/experiments/stage2/045_stage2l_wrong_target_negative_control.md`

目的：
```text
检验 hidden-position bridge 是否只是普遍推高/拉低任意答案 token，还是更偏向当前样本的正确答案 target。
如果 correct target 的 restore/corrupt 效果稳定强于 wrong target，则支持 answer-specific hidden bridge。
```

输入：
```text
doc/experiments/stage2/cross_model/stage2i_selected_12_manifest.csv
现有 12 个 localized samples 的 image / answer mask / relate mask
Qwen layer 26
LLaVA layer 15
B_direct / D_visual_only
```

输出：
```text
doc/experiments/stage2/cross_model/stage2l_qwen_wrong_target_negative_control.csv
doc/experiments/stage2/cross_model/stage2l_qwen_wrong_target_negative_control.json
doc/experiments/stage2/cross_model/stage2l_llava_wrong_target_negative_control.csv
doc/experiments/stage2/cross_model/stage2l_llava_wrong_target_negative_control.json
doc/experiments/stage2/cross_model/stage2l_wrong_target_case_table.csv
doc/experiments/stage2/cross_model/stage2l_wrong_target_summary.csv
doc/experiments/stage2/cross_model/stage2l_wrong_target_decision.json
doc/experiments/stage2/045_stage2l_wrong_target_negative_control.md
```

方法：
```text
每个样本保留 correct target answer。
wrong target 从 manifest 的下一个不同样本中选取。
同一组 hidden-position patch 同时计算 correct_effect_logit 与 wrong_effect_logit。
主组固定为 top_hidden_delta_plus_answer_adjacent。
判据是 correct_minus_wrong_logit 是否稳定大于 0。
```

执行修复：
```text
第一次 LLaVA 运行 blocked，因为 wrong-target 选择规则过严。
LLaVA tokenizer 常让不同答案共享前导空格 token；旧脚本只要任意候选 token overlap 就丢弃整个 wrong answer。
修复后改为过滤掉 overlap token，保留非重叠 wrong-target candidates。
修复后 LLaVA usable_runs = 24 / 24。
```

主结果：
```text
Qwen top_hidden_delta_plus_answer_adjacent:
  corrupt correct_minus_wrong = +3.485067, 95% CI = [+2.478394, +4.456096]
  restore correct_minus_wrong = +3.950033, 95% CI = [+2.830078, +5.071940]
  verdict = wrong_target_control_supported

LLaVA top_hidden_delta_plus_answer_adjacent:
  corrupt correct_minus_wrong = +0.529970, 95% CI = [+0.171885, +0.987908]
  restore correct_minus_wrong = +1.485235, 95% CI = [+0.661184, +2.312505]
  verdict = wrong_target_control_supported
```

预期与实际偏差：
```text
预期：Qwen 应该更强；LLaVA 可能因为 Stage 2J/2K 的 delta-control weakness 而较弱。
实际：Qwen 明显更强；LLaVA 虽然 effect size 较小，但 correct target 仍稳定强于 wrong target。
这说明 LLaVA 的 bridge 不完全是 general logit movement，但仍不能抹掉 delta-matched control 的部分解释。
```

结论：
```text
Stage 2L-3 加强了 cross-model auxiliary evidence：
Qwen 与 LLaVA 的 hidden bridge 都具有 answer-specific 成分。
但它仍不是 Gemma-style source-control causal route replication。
Qwen 继续是更强的 cross-model auxiliary line；LLaVA 可以写成 wrong-target specificity supported, matched-control specificity partial。
```

## Stage 2L-4 Evidence-to-Answer Bridge 位置分解

详细结果文档：`doc/experiments/stage2/046_stage2l_evidence_to_answer_bridge_decomposition.md`

目的：
```text
拆解 Stage 2I/2J 中最强的 top_hidden_delta_plus_answer_adjacent bridge。
判断它到底来自 image-only hidden positions、answer-adjacent text positions，还是二者组合产生的 evidence-to-answer convergence。
```

输入：
```text
doc/experiments/stage2/cross_model/stage2j_qwen_matched_control_patch.csv
doc/experiments/stage2/cross_model/stage2j_llava_matched_control_patch.csv
```

输出：
```text
doc/experiments/stage2/cross_model/stage2l_bridge_decomposition_case_table.csv
doc/experiments/stage2/cross_model/stage2l_bridge_decomposition_summary.csv
doc/experiments/stage2/cross_model/stage2l_bridge_decomposition_decision.json
doc/experiments/stage2/046_stage2l_evidence_to_answer_bridge_decomposition.md
```

方法：
```text
对每个 model x sample x prompt x direction 比较三组 hidden patch：
image-only = top_hidden_delta
answer-adjacent-only = answer_adjacent_text
combo = top_hidden_delta_plus_answer_adjacent

核心指标：
combo_minus_image
combo_minus_answer
combo_minus_max_single
```

结果：
```text
LLaVA:
  corrupt:
    combo_minus_image = +0.359863, stable_positive
    combo_minus_answer = +0.225912, stable_positive
    combo_minus_max_single = +0.044596, weak_positive
  restore:
    combo_minus_image = +0.590657, stable_positive
    combo_minus_answer = +1.197917, stable_positive
    combo_minus_max_single = +0.419759, stable_positive
  verdict = visual_plus_answer_bridge_supported

Qwen:
  corrupt:
    combo_minus_image = +2.758464, stable_positive
    combo_minus_answer = +0.147786, stable_positive
    combo_minus_max_single = -0.013672, heterogeneous
  restore:
    combo_minus_image = +3.075521, stable_positive
    combo_minus_answer = +0.819010, stable_positive
    combo_minus_max_single = +0.460938, weak_positive
  verdict = visual_plus_answer_bridge_supported
```

预期与实际偏差：
```text
预期：combo 应该强于至少一个单独组；如果强于两个单独组，则 evidence-to-answer convergence 更可信。
实际：两个模型的 combo_minus_image 和 combo_minus_answer 都为 stable_positive，说明 visual positions 与 answer-adjacent positions 都贡献了额外信息。
但 Qwen corrupt 的 combo_minus_max_single 为 heterogeneous，说明有些 case 中 answer-adjacent-only 已经很强，combo 不一定稳定强于最强单独组。
```

结论：
```text
Stage 2L-4 支持把跨模型 hidden bridge 写成 evidence-to-answer bridge / convergence，而不是纯 image-token route。
它加强了 Qwen/LLaVA 的 hidden-state-level auxiliary evidence，但仍然不构成 feature-level source-control causal route replication。
```

## Stage 2L-3b Mask-Shuffled Negative Control

详细结果文档：`doc/experiments/stage2/047_stage2l_mask_shuffled_negative_control.md`

目的：
```text
检验 hidden bridge 是否只是由任意遮挡造成，而不是由真实证据区域遮挡造成。
如果真实 evidence mask 的 patch effect 强于同图平移后的 mask_shuffled control，就支持 evidence-location specificity。
```

输入：
```text
doc/experiments/stage2/cross_model/stage2i_selected_12_manifest.csv
12 个 localized samples 的 image / answer mask / relate mask
Qwen layer 26
LLaVA layer 15
B_direct / D_visual_only
```

输出：
```text
doc/experiments/stage2/cross_model/stage2l_qwen_mask_shuffled_negative_control.csv
doc/experiments/stage2/cross_model/stage2l_qwen_mask_shuffled_negative_control.json
doc/experiments/stage2/cross_model/stage2l_llava_mask_shuffled_negative_control.csv
doc/experiments/stage2/cross_model/stage2l_llava_mask_shuffled_negative_control.json
doc/experiments/stage2/cross_model/stage2l_mask_shuffled_case_table.csv
doc/experiments/stage2/cross_model/stage2l_mask_shuffled_summary.csv
doc/experiments/stage2/cross_model/stage2l_mask_shuffled_decision.json
doc/experiments/stage2/047_stage2l_mask_shuffled_negative_control.md
```

方法：
```text
对每个样本构造两个遮挡条件：
1. evidence_mask：真实 answer/relate union mask。
2. mask_shuffled：把 union mask 在同图内平移到非原始位置。

source-like positions 固定由真实 evidence_mask 的 clean-vs-mask hidden delta 选出。
然后比较同一组 positions 在 evidence_mask 与 mask_shuffled 条件下的 patch effect。
```

主结果：
```text
Qwen top_hidden_delta_plus_answer_adjacent:
  corrupt real_minus_shuffled = +2.719401, 95% CI = [+1.637370, +3.860026]
  restore real_minus_shuffled = +3.156250, 95% CI = [+1.980469, +4.381510]
  verdict = mask_shuffled_control_supported

LLaVA top_hidden_delta_plus_answer_adjacent:
  corrupt real_minus_shuffled = +0.551432, 95% CI = [+0.305013, +0.839681]
  restore real_minus_shuffled = +1.383952, 95% CI = [+0.848958, +1.963704]
  verdict = mask_shuffled_control_supported
```

预期与实际偏差：
```text
预期：真实 evidence mask 应该强于 shifted mask，但 LLaVA 可能更弱。
实际：Qwen 和 LLaVA 在 restore/corrupt 两个方向都稳定 real > shuffled。
这比 wrong-target control 更直接地支持 evidence-location specificity。
```

结论：
```text
Stage 2L-3b 明确加强跨模型辅助证据：
Qwen/LLaVA 的 hidden bridge 不只是由任意遮挡造成，而是对真实证据区域遮挡更敏感。
这仍然是 hidden-state-level bridge specificity，不是 feature-level source route replication。
```

## Stage 2M Run Plan：Cross-Model 完整复现拆档推进

详细计划文档：`doc/experiments/stage2/048_stage2m_cross_model_full_replication_run_plan.md`

目的：
```text
把 Qwen/LLaVA 的跨模型证据从 Stage 2L 的 hidden-state bridge 辅助证据，推进到更接近 Gemma 主链的复现检查。
但为了避免口径漂移，本阶段把“完整复现”拆成三档：
Tier 1: hidden-state full replication
Tier 2: feature-level causal bridge
Tier 3: source-control route replication
```

核心设计：
```text
Qwen: Qwen2.5-VL-7B-Instruct, layer 26
LLaVA: LLaVA-1.5-7B, layer 15
prompts: B_direct, D_visual_only
main group: top_hidden_delta_plus_answer_adjacent
样本目标: 24 个 localized samples
主控制: matched hidden controls, wrong-target control, mask-shuffled control
```

结论边界：
```text
Stage 2M 即使 Tier 1 成功，也只能说明 Qwen/LLaVA 有 hidden-state-level 的同类现象。
只有 Tier 2 feature restore/corrupt 成功后，才可以写 feature-level causal bridge。
只有额外完成 source tracing、source/control route intervention、region-mask sensitivity 与 rank/generation linkage，才可写 Gemma-style source-control route replication。
```

## Stage 2M-0：24-Sample Manifest 构建

输入：
```text
已有 localized mask 资产
doc/experiments/stage2/cross_model/stage2m_selected_24_manifest.csv
```

输出：
```text
doc/experiments/stage2/cross_model/stage2m_selected_24_manifest.csv
doc/experiments/stage2/cross_model/stage2m_annotation_supplement_needed.csv
doc/experiments/stage2/cross_model/stage2m_manifest_summary.json
```

结果：
```text
target_count = 24
selected_count = 24
available_eligible_count = 52
symbol_text_reading = 11
visual_readout = 11
scene_inference = 2
usable_status = pass_min20
```

预期与实际偏差：
```text
原计划类型配比是 symbol_text_reading=12, visual_readout=8, scene_inference=4。
实际可用 localized masks 中 scene_inference 不足，只能得到 2 个；没有用 diffuse / 弥漫样本硬凑。
因此 Stage 2M 的类型化结论仍以 symbol_text_reading 和 visual_readout 为主，scene_inference 只作 underpowered diagnostic。
```

结论：
```text
Stage 2M 的样本规模从 12 扩到 24 成功。
但类型配比不是理想的 12/8/4，因此不能把跨模型结论推广到所有 scene inference。
```

## Stage 2M-1：Expanded Hidden-State Cross-Model Replication

详细结果文档：`doc/experiments/stage2/049_stage2m_expanded_hidden_bridge_replication.md`

目的：
```text
检验 Qwen 和 LLaVA 是否都存在 evidence-sensitive、target-specific、evidence-location-specific hidden-state bridge。
这一步对应 Stage 2M Tier 1，不判断 feature-level causal bridge，也不判断 Gemma-style source-control route replication。
```

输入：
```text
24 个 localized samples
Qwen layer 26
LLaVA layer 15
B_direct / D_visual_only
clean / union_mask hidden states
matched controls: delta_matched_plus_answer_adjacent, activation_matched_plus_answer_adjacent
negative controls: wrong-target, mask-shuffled
```

输出：
```text
doc/experiments/stage2/cross_model/stage2m_qwen_hidden_position_patch.csv/json
doc/experiments/stage2/cross_model/stage2m_llava_hidden_position_patch.csv/json
doc/experiments/stage2/cross_model/stage2m_qwen_wrong_target_negative_control.csv/json
doc/experiments/stage2/cross_model/stage2m_llava_wrong_target_negative_control.csv/json
doc/experiments/stage2/cross_model/stage2m_qwen_mask_shuffled_negative_control.csv/json
doc/experiments/stage2/cross_model/stage2m_llava_mask_shuffled_negative_control.csv/json
doc/experiments/stage2/cross_model/stage2m_hidden_position_patch_summary.csv
doc/experiments/stage2/cross_model/stage2m_matched_control_model_summary.csv
doc/experiments/stage2/cross_model/stage2m_wrong_target_summary.csv
doc/experiments/stage2/cross_model/stage2m_mask_shuffled_summary.csv
doc/experiments/stage2/cross_model/stage2m_full_replication_tier1_decision.json
```

方法：
```text
对每个 model x sample x prompt 比较 clean 与 union_mask。
source-like hidden group 固定为 top_hidden_delta_plus_answer_adjacent。
restore: 在 masked run 中 patch 回 clean hidden state。
corrupt: 在 clean run 中 patch 成 masked hidden state。
matched controls: 用 delta-matched / activation-matched visual positions 加 answer-adjacent positions 作对照。
wrong-target control: 检查 patch 是否更恢复正确 target，而不是任意 target。
mask-shuffled control: 检查真实证据区域 mask 是否强于同图平移的 shuffled mask。
```

可用性：
```text
Qwen hidden matched controls: 48/48 prompt-runs usable
LLaVA hidden matched controls: 48/48 prompt-runs usable
Qwen wrong-target control: 48/48 prompt-runs usable
LLaVA wrong-target control: 48/48 prompt-runs usable
Qwen mask-shuffled control: 48/48 prompt-runs usable
LLaVA mask-shuffled control: 48/48 prompt-runs usable
```

主结果：
```text
overall_status = cross_model_hidden_replication_supported_qwen_stronger_llava_smaller

Qwen:
  tier1_status = tier1_hidden_mostly_supported
  matched_control_status = matched_specificity_partial
  wrong_target_status = wrong_target_control_supported
  mask_shuffled_status = mask_shuffled_control_supported

LLaVA:
  tier1_status = tier1_hidden_full_supported
  matched_control_status = matched_specificity_supported
  wrong_target_status = wrong_target_control_supported
  mask_shuffled_status = mask_shuffled_control_supported
```

Hidden-position patch 结果：
```text
Qwen:
  best_restore_source_minus_random_logit = +2.809571
  best_restore_source_minus_random_rank = +468.588542
  best_restore_positive_logit_n = 40/48
  matched_corrupt_source_minus_random_logit = +2.346354
  matched_corrupt_source_minus_random_rank = +211.416666
  matched_corrupt_positive_logit_n = 40/48

LLaVA:
  best_restore_source_minus_random_logit = +1.194641
  best_restore_source_minus_random_rank = +28.317709
  best_restore_positive_logit_n = 41/48
  matched_corrupt_source_minus_random_logit = +0.662903
  matched_corrupt_source_minus_random_rank = +7.026041
  matched_corrupt_positive_logit_n = 37/48
```

Matched-control 结果：
```text
Qwen:
  restore combo_minus_delta = +0.186198, stable_positive
  restore combo_minus_activation = +0.216146, stable_positive
  corrupt combo_minus_delta = +0.051758, weak_or_heterogeneous_positive
  corrupt combo_minus_activation = +0.037760, weak_or_heterogeneous_positive

LLaVA:
  restore combo_minus_delta = +0.187012, weak_or_heterogeneous_positive
  restore combo_minus_activation = +0.925293, stable_positive
  corrupt combo_minus_delta = +0.263753, stable_positive
  corrupt combo_minus_activation = +0.283447, stable_positive
```

Wrong-target 结果：
```text
Qwen:
  corrupt correct_minus_wrong = +2.689514, CI [+1.899679, +3.462199]
  restore correct_minus_wrong = +3.229248, CI [+2.322876, +4.131470]
  status = stable_correct_gt_wrong

LLaVA:
  corrupt correct_minus_wrong = +0.688402, CI [+0.370091, +1.009811]
  restore correct_minus_wrong = +1.594312, CI [+1.053792, +2.113640]
  status = stable_correct_gt_wrong
```

Mask-shuffled 结果：
```text
Qwen:
  corrupt real_minus_shuffled = +2.228841, CI [+1.371094, +3.073242]
  restore real_minus_shuffled = +2.698568, CI [+1.786458, +3.668620]
  status = stable_real_gt_shuffled

LLaVA:
  corrupt real_minus_shuffled = +0.681885, CI [+0.380046, +0.982015]
  restore real_minus_shuffled = +1.338013, CI [+0.874959, +1.808472]
  status = stable_real_gt_shuffled
```

预期与实际偏差：
```text
预期：Qwen 应该更强，LLaVA 可能更小或异质。
实际：Qwen 的绝对 effect size 明显更大；LLaVA 的 effect size 更小但 target/location controls 稳定。
Qwen 的 matched-control corrupt 方向 weaker，因此 Qwen 写成 mostly supported 而不是 fully supported。
LLaVA 虽然 effect 小，但 matched-control 相对 specificity 更稳定。
```

结论：
```text
Stage 2M Tier 1 支持：Qwen 和 LLaVA 都存在 expanded hidden-state bridge replication。
这个现象不是 Gemma-only；但当前只到 hidden-state-level。
不能写 Qwen/LLaVA 已经完整复现 Gemma source-control causal routes。
```

## Stage 2M-2a：Decoded Bridge Passing-Case 候选选择

脚本：
```text
scripts/local/build_stage2m_decoded_candidate_manifest.py
```

目的：
```text
只从 Tier 1 passing / high-score cases 中挑 decoded bridge 样本，避免把弱样本直接塞进 generation 表造成噪声。
```

输入：
```text
doc/experiments/stage2/cross_model/stage2m_matched_control_case.csv
doc/experiments/stage2/cross_model/stage2m_hidden_position_patch_case_table.csv
doc/experiments/stage2/cross_model/stage2m_selected_24_manifest.csv
```

输出：
```text
doc/experiments/stage2/cross_model/stage2m_decoded_candidate_prompt_rows.csv
doc/experiments/stage2/cross_model/stage2m_decoded_candidate_manifest.csv
doc/experiments/stage2/cross_model/stage2m_decoded_candidate_decision.json
```

方法：
```text
对每个 model x sample x prompt 计算候选分数：
1. hidden restore/corrupt effect_logit
2. restore/corrupt gap closure
3. source combo logit
4. source > delta_matched / activation_matched controls
5. 类型 bonus: Qwen 偏 symbol_text_reading，LLaVA 偏 visual_readout / symbol_text_reading

为了让 decoded smoke 覆盖更多图像，最终按 unique sample 而不是单纯 prompt-row 选择：
Qwen 选 8 个 unique samples
LLaVA 选 6 个 unique samples
```

结果：
```text
status = pass
prompt_rows = 14
unique_samples = 10
Qwen unique samples = 8
LLaVA unique samples = 6
```

入选样本：
```text
Qwen:
  okvqa_val_4502065, okvqa_val_3608785, okvqa_val_2954205, okvqa_val_3959785,
  okvqa_val_1058855, okvqa_val_2847255, okvqa_val_1593205, okvqa_val_3658865

LLaVA:
  okvqa_val_136595, okvqa_val_3608785, okvqa_val_1593205,
  okvqa_val_4739195, okvqa_val_4502065, okvqa_val_1058855
```

结论：
```text
这些候选只说明“适合进入 decoded bridge smoke”，不说明 generation-level bridge 已成立。
下一步将只在这些 passing cases 上跑 short greedy generation。
```

## Stage 2M-2：Decoded Bridge on Passing Cases

详细结果文档：`doc/experiments/stage2/050_stage2m_decoded_bridge_on_passing_cases.md`

目的：
```text
在 Tier 1 hidden bridge 通过的高分样本上跑短生成，检查 hidden patch 是否能把 masked generation 往 clean/target answer 方向拉回。
```

输入：
```text
doc/experiments/stage2/cross_model/stage2m_decoded_candidate_manifest.csv
Qwen layer 26
LLaVA layer 15
B_direct / D_visual_only
greedy generation, max_new_tokens = 3
```

输出：
```text
doc/experiments/stage2/cross_model/stage2m_qwen_decoded_bridge.csv/json
doc/experiments/stage2/cross_model/stage2m_llava_decoded_bridge.csv/json
doc/experiments/stage2/cross_model/stage2m_decoded_bridge_summary.csv
doc/experiments/stage2/cross_model/stage2m_decoded_bridge_case_table.csv
doc/experiments/stage2/cross_model/stage2m_decoded_bridge_decision.json
```

方法：
```text
对每个 sample x prompt 比较：
clean generation
union_mask generation
restore::top_hidden_delta_plus_answer_adjacent
restore::delta_matched_plus_answer_adjacent
restore::activation_matched_plus_answer_adjacent
restore::low_delta_control
restore::random_control_1

核心判断不是“所有答案是否恢复”，而是：
1. source-like hidden patch 是否提高 target_hit；
2. 是否让 predicted answer 更接近 clean answer；
3. first-token target logit / rank 是否恢复；
4. matched/random controls 是否同样恢复。
```

可用性：
```text
Qwen usable_runs = 16/16, rows = 128
LLaVA usable_runs = 12/12, rows = 108
```

结果：
```text
Qwen:
  status = partial_generation_bridge_smoke
  informative_clean_vs_union_rows = 13
  source restore target_hit = 4/16
  union_mask target_hit = 2/16
  low_delta target_hit = 3/16
  random_control target_hit = 3/16
  source same_as_clean on informative rows = 3
  source mean_logit_restore_vs_union = +5.255859
  source mean_rank_restore_vs_union = +1247.875

LLaVA:
  status = partial_generation_bridge_smoke
  informative_clean_vs_union_rows = 11
  source restore target_hit = 2/12
  union_mask target_hit = 0/12
  low_delta target_hit = 0/12
  random_control target_hit = 0/12
  source same_as_clean on informative rows = 2
  source mean_logit_restore_vs_union = +4.100260
  source mean_rank_restore_vs_union = +45.25
```

预期与实际偏差：
```text
预期：hidden patch 可能恢复部分 decoded answer，但不一定稳定恢复完整自然答案。
实际：两个模型都只有 partial generation bridge。
Qwen first-token/rank 恢复很强，但 matched controls 也恢复不少，source-specific generation claim 不够强。
LLaVA target_hit 较少，但 source restore 相比 low/random controls 更干净。
```

结论：
```text
Stage 2M-2 支持 behavior-side smoke：hidden patch 可以在部分 passing cases 上影响 decoded answer。
但它不支持“稳定生成答案恢复”，因此主结论仍应落在 first-token/rank bridge，而不是 full generation restoration。
```

## Stage 2M-3：Feature-Level Causal Bridge Smoke

详细结果文档：`doc/experiments/stage2/051_stage2m_feature_level_causal_bridge.md`

目的：
```text
尝试把 Qwen/LLaVA 的 hidden bridge 下沉到 CLT feature 层。
如果 evidence-sensitive features 承载该机制，则 evidence_topk features 的 restore/corrupt 应强于 activation/drop/mask-insensitive/random feature controls。
```

输入：
```text
Stage 2M passing cases
Qwen: 8 unique samples, 16 prompt-runs
LLaVA: 6 unique samples, 12 prompt-runs
position group = top_hidden_delta_plus_answer_adjacent
top_k_features = 4
scale = 1.0
```

输出：
```text
doc/experiments/stage2/cross_model/stage2m_qwen_feature_bridge.csv/json
doc/experiments/stage2/cross_model/stage2m_llava_feature_bridge.csv/json
doc/experiments/stage2/cross_model/stage2m_feature_bridge_summary.csv
doc/experiments/stage2/cross_model/stage2m_feature_bridge_specificity.csv
doc/experiments/stage2/cross_model/stage2m_feature_bridge_decision.json
```

方法：
```text
在 source-like positions 上编码 clean / union_mask hidden states。
选择 clean-minus-mask drop 最大的 active CLT features 作为 evidence_topk。
构造四类 feature controls：
activation_matched_topk
drop_matched_topk
mask_insensitive_topk
random_topk

干预：
masked→clean restoration: 在 union_mask run 中加回 clean-minus-mask feature decoder contribution。
clean→masked corruption: 在 clean run 中减去 clean-minus-mask feature decoder contribution。
```

可用性：
```text
Qwen usable_runs = 16/16, rows = 192
LLaVA usable_runs = 12/12, rows = 144
```

结果：
```text
overall_status = feature_bridge_not_established
Qwen status = feature_bridge_not_established
LLaVA status = feature_bridge_not_established
```

Qwen：
```text
evidence_topk restore:
  positive_logit_n = 7/16
  positive_rank_n = 8/16
  mean_logit_effect = +0.011719
  mean_rank_effect = +140.5625

evidence_topk corrupt:
  positive_logit_n = 6/16
  positive_rank_n = 4/16
  mean_logit_effect = -0.011719
  mean_rank_effect = +0.3125
```

LLaVA：
```text
evidence_topk restore:
  positive_logit_n = 2/12
  positive_rank_n = 1/12
  mean_logit_effect = -0.010091
  mean_rank_effect = 0.0

evidence_topk corrupt:
  positive_logit_n = 2/12
  positive_rank_n = 0/12
  mean_logit_effect = -0.016276
  mean_rank_effect = -0.083333
```

预期与实际偏差：
```text
预期：如果 feature bridge 成立，evidence_topk 应在 restore 和 corrupt 至少一个方向稳定强于 matched controls。
实际：
Qwen restore rank 有弱正信号，但 logit effect 很小，且不稳定强于 mask-insensitive/random controls；corrupt 方向不成立。
LLaVA feature-level restore/corrupt 基本不成立。
```

结论：
```text
Stage 2M-3 没有证明 feature-level causal bridge。
跨模型复现当前最强层级仍是 hidden-state-level，而不是 CLT feature-level。
```

## Stage 2M Final Verdict

详细结论文档：`doc/experiments/stage2/052_stage2m_full_replication_verdict.md`

最终判断：
```text
Tier 1 hidden-state full replication: supported
Tier 1.5 decoded bridge: partial smoke support
Tier 2 feature-level causal bridge: not established
Tier 3 source-control route replication: not attempted / not established
```

可以写：
```text
Qwen 和 LLaVA 都显示 evidence-region-sensitive hidden-state bridge，因此跨模型辅助证据说明该现象不是 Gemma-only。
Qwen effect size 更大；LLaVA effect 更小但 target/location specificity 稳定。
```

不能写：
```text
Qwen/LLaVA 复现了 Gemma 的 source-control causal routes。
Qwen/LLaVA CLT features 已经形成 feature-level causal bridge。
Qwen/LLaVA 的 decoded answer 能稳定被 hidden patch 恢复。
```

下一步建议：
```text
不要盲目扩大 feature smoke。
先分析为什么 feature effects 被 controls 吸收，尤其 Qwen restore-rank 正信号是否由少数 case 驱动。
若继续 Tier 2，应尝试 attribution-weighted multi-feature group 或真正的 answer-adjacent source tracing adapter。
```

## Stage 2N Run Plan：Cross-Model Hidden-State Replication 加固

详细计划文档：`doc/experiments/stage2/053_stage2n_hidden_replication_run_plan.md`

目的：
```text
Stage 2M 已经说明 Qwen2.5-VL 和 LLaVA 在 hidden-state 层存在 evidence-region-sensitive bridge。
但是 Stage 2M 使用的是第一批 24 个 localized samples，仍可能被质疑为少数样本偶然现象。

Stage 2N 的目标是用 Stage 2M 没有使用过的 heldout localized masks 做独立复现：
1. 验证 Qwen/LLaVA 的 hidden-state bridge 是否能在未用样本上继续成立；
2. 同时加入 answer_mask 与 union_mask 两种 mask condition；
3. 加入 wrong-target 与 mask-shuffled negative controls；
4. 保持结论边界：只证明 hidden-state-level cross-model replication，不升级到 feature-level causal bridge 或 Gemma-style source-control route replication。
```

专有名词解释：
```text
hidden-state bridge：
  在模型某一层的 hidden states 上，把 clean 图像运行中的部分位置状态 patch 到 masked 图像运行里，或反过来把 masked 状态 patch 到 clean 运行里，观察目标答案 logit/rank 是否恢复或受损。

restore：
  masked→clean restoration。在 evidence mask 后的运行中补回 clean hidden states，若目标答案 logit/rank 恢复，说明这些 hidden positions 携带了被遮挡破坏的答案相关信号。

corrupt：
  clean→masked corruption。在 clean 运行中替换为 masked hidden states，若目标答案 logit/rank 下降，说明这些位置对 clean 答案支持有因果作用。

source-like group：
  这里不是 Gemma tracing 得到的 source node，而是跨模型临时构造的 hidden-state 位置组：top_hidden_delta_plus_answer_adjacent。
  它由 clean-vs-mask hidden delta 最大的视觉位置，加上 answer-adjacent text positions 构成。

random controls：
  同数量的随机视觉位置 control，用来排除“patch 任意视觉位置都有效”的解释。

matched controls：
  包括 delta_matched_plus_answer_adjacent 与 activation_matched_plus_answer_adjacent，用来排除“只是 hidden delta 大”或“只是 activation 大”的解释。

wrong-target control：
  用错误答案 token 作为目标，检查 patch 效果是否 target-specific。

mask-shuffled control：
  使用打乱位置的 mask，检查效果是否来自真实证据区域，而不是 mask 面积或遮挡扰动本身。
```

输入：
```text
候选样本来源：
  doc/experiments/stage2/cross_model/stage2i_cross_model_candidate_manifest.csv

eligible localized samples = 52
Stage 2M 已用样本 = 24
Stage 2N heldout samples = 28

heldout 类型分布：
  visual_readout = 9
  untyped_localized = 19

all52 类型分布：
  symbol_text_reading = 11
  visual_readout = 20
  scene_inference = 2
  untyped_localized = 19

模型：
  Qwen2.5-VL-7B-Instruct, layer 26
  LLaVA-1.5-7B, layer 15

prompts：
  B_direct
  D_visual_only

mask conditions：
  answer_mask
  union_mask

directions：
  restore
  corrupt

primary group：
  top_hidden_delta_plus_answer_adjacent
```

输出：
```text
计划和结果文档：
  doc/experiments/stage2/053_stage2n_hidden_replication_run_plan.md
  doc/experiments/stage2/054_stage2n_heldout_hidden_bridge_replication.md
  doc/experiments/stage2/055_stage2n_stricter_controls_and_mask_condition.md
  doc/experiments/stage2/056_stage2n_hidden_replication_verdict.md

manifest：
  doc/experiments/stage2/cross_model/stage2n_heldout_manifest.csv
  doc/experiments/stage2/cross_model/stage2n_all52_manifest.csv
  doc/experiments/stage2/cross_model/stage2n_annotation_supplement_needed.csv
  doc/experiments/stage2/cross_model/stage2n_manifest_summary.json

主实验 raw artifacts：
  doc/experiments/stage2/cross_model/stage2n_qwen_hidden_position_patch.csv/json
  doc/experiments/stage2/cross_model/stage2n_llava_hidden_position_patch.csv/json

negative controls：
  doc/experiments/stage2/cross_model/stage2n_qwen_wrong_target_negative_control.csv/json
  doc/experiments/stage2/cross_model/stage2n_llava_wrong_target_negative_control.csv/json
  doc/experiments/stage2/cross_model/stage2n_qwen_mask_shuffled_negative_control.csv/json
  doc/experiments/stage2/cross_model/stage2n_llava_mask_shuffled_negative_control.csv/json

summary / decision：
  doc/experiments/stage2/cross_model/stage2n_heldout_hidden_summary.csv
  doc/experiments/stage2/cross_model/stage2n_all52_hidden_summary.csv
  doc/experiments/stage2/cross_model/stage2n_hidden_specificity_case.csv
  doc/experiments/stage2/cross_model/stage2n_hidden_specificity_summary.csv
  doc/experiments/stage2/cross_model/stage2n_hidden_typed_summary.csv
  doc/experiments/stage2/cross_model/stage2n_hidden_mask_condition_summary.csv
  doc/experiments/stage2/cross_model/stage2n_hidden_prompt_summary.csv
  doc/experiments/stage2/cross_model/stage2n_hidden_replication_decision.json
  doc/experiments/stage2/cross_model/stage2n_wrong_target_summary.csv
  doc/experiments/stage2/cross_model/stage2n_wrong_target_decision.json
  doc/experiments/stage2/cross_model/stage2n_mask_shuffled_summary.csv
  doc/experiments/stage2/cross_model/stage2n_mask_shuffled_decision.json
```

方法：
```text
1. 构造 heldout manifest：
   从 52 个 eligible localized samples 中排除 Stage 2M 已用 24 个，得到 28 个 heldout samples。
   若 heldout usable 少于 20，则停止扩大并生成补标清单；本次 heldout_count = 28，因此通过最小数量要求。

2. 对 Qwen 和 LLaVA 分别运行 hidden-state patch：
   Qwen 固定 layer 26。
   LLaVA 固定 layer 15。
   每个 sample × prompt × mask_condition × direction 都计算 primary group 与 controls 的 target token logit/rank effect。

3. 对每个 mask condition 独立计算 clean-vs-mask hidden delta：
   answer_mask 和 union_mask 分开计算 source-like positions，避免把 union_mask 的位置选择错误套到 answer_mask 上。

4. 加入 controls：
   random_control_1..4：同数量随机视觉位置。
   low_delta_control：低 hidden delta 位置。
   delta_matched_plus_answer_adjacent：hidden delta 匹配的 control。
   activation_matched_plus_answer_adjacent：activation 匹配的 control。

5. 加严 negative controls：
   wrong_target：检查 correct target effect 是否大于 wrong target effect。
   mask_shuffled：检查真实 evidence mask 是否强于 shuffled mask。

6. Bootstrap / status 判断：
   stable_positive 表示均值为正且 95% CI 不跨 0。
   weak_or_heterogeneous_positive 表示均值为正但 CI 或 case consistency 不够稳定。
   not_positive 表示不支持正向结论。
```

## Stage 2N-1：Heldout Hidden Bridge Replication

详细结果文档：`doc/experiments/stage2/054_stage2n_heldout_hidden_bridge_replication.md`

运行可用性：
```text
Qwen answer/union hidden-position matched controls:
  usable_runs = 112
  rows = 3136

LLaVA answer/union hidden-position matched controls:
  usable_runs = 112
  rows = 3584
```

主决策：
```text
overall_status = cross_model_heldout_hidden_replication_supported

Qwen:
  status = heldout_hidden_replication_supported
  source_random_stable_rows = 4/4
  matched_positive_rows = 2/4

LLaVA:
  status = heldout_hidden_replication_supported
  source_random_stable_rows = 4/4
  matched_positive_rows = 3/4
```

Qwen heldout primary results：
```text
answer_mask corrupt:
  source_minus_random_logit_mean = +3.602121
  status = stable_positive

answer_mask restore:
  source_minus_random_logit_mean = +4.067732
  status = stable_positive

union_mask corrupt:
  source_minus_random_logit_mean = +3.226981
  status = stable_positive

union_mask restore:
  source_minus_random_logit_mean = +3.830357
  status = stable_positive
```

LLaVA heldout primary results：
```text
answer_mask corrupt:
  source_minus_random_logit_mean = +0.707947
  status = stable_positive

answer_mask restore:
  source_minus_random_logit_mean = +1.132621
  status = stable_positive

union_mask corrupt:
  source_minus_random_logit_mean = +0.781546
  status = stable_positive

union_mask restore:
  source_minus_random_logit_mean = +0.834726
  status = stable_positive
```

mask condition 汇总：
```text
Qwen answer_mask:
  source_minus_random_logit_mean = +3.834926
  status = stable_positive

Qwen union_mask:
  source_minus_random_logit_mean = +3.528669
  status = stable_positive

LLaVA answer_mask:
  source_minus_random_logit_mean = +0.920284
  status = stable_positive

LLaVA union_mask:
  source_minus_random_logit_mean = +0.808136
  status = stable_positive
```

all52 pooled source effect：
```text
Qwen answer_mask corrupt / restore:
  mean_effect_logit = +3.606585 / +4.149833
  status = stable_positive / stable_positive

Qwen union_mask corrupt / restore:
  mean_effect_logit = +2.826472 / +3.423528
  status = stable_positive / stable_positive

LLaVA answer_mask corrupt / restore:
  mean_effect_logit = +0.700474 / +1.505301
  status = stable_positive / stable_positive

LLaVA union_mask corrupt / restore:
  mean_effect_logit = +0.730281 / +1.322660
  status = stable_positive / stable_positive
```

类型化结果：
```text
Stage 2N heldout:
  visual_readout 在 Qwen 和 LLaVA 上 answer_mask / union_mask / corrupt / restore 全部为 stable_positive。
  untyped_localized 在 Qwen 和 LLaVA 上 answer_mask / union_mask / corrupt / restore 全部为 stable_positive。

Stage 2M + Stage 2N all evidence:
  symbol_text_reading 和 visual_readout 是当前最稳定的类型。
  scene_inference 样本太少，且 Stage 2M 中方向不稳定，不能作为稳定类型 claim。

注意：
  Stage 2N heldout 中有 19 个 untyped_localized，说明它们有 localized mask，但缺少 reasoning_operation 标签。
  因此 Stage 2N 可以支持 heldout replication，但类型化 claim 仍应主要依赖 typed rows。
```

Prompt interaction：
```text
Qwen B_direct answer_mask:
  source_minus_random_logit_mean = +4.038993
  status = stable_positive

Qwen D_visual_only answer_mask:
  source_minus_random_logit_mean = +3.630859
  status = stable_positive

LLaVA B_direct answer_mask:
  source_minus_random_logit_mean = +0.941354
  status = stable_positive

LLaVA D_visual_only answer_mask:
  source_minus_random_logit_mean = +0.899214
  status = stable_positive
```

结论：
```text
Stage 2N-1 支持独立 heldout hidden-state replication。
Qwen effect size 明显大于 LLaVA。
LLaVA effect size 更小，但 answer_mask 与 union_mask、restore 与 corrupt 都稳定为正。
两个 prompt 都成立，因此不能写 D_visual_only 更好，只能写 prompt 作为 modulation factor。
```

## Stage 2N-2：Stricter Controls And Mask Condition

详细结果文档：`doc/experiments/stage2/055_stage2n_stricter_controls_and_mask_condition.md`

目的：
```text
Stage 2N-1 证明 source-like hidden positions 强于 random controls。
Stage 2N-2 进一步检查两个更严格问题：
1. 效果是否 target-specific，也就是 correct target 是否强于 wrong target；
2. 效果是否 evidence-location-specific，也就是真实 evidence mask 是否强于 shuffled mask。
```

wrong-target 运行：
```text
Qwen:
  usable_runs = 56

LLaVA:
  usable_runs = 56
```

wrong-target 结果：
```text
Qwen corrupt:
  correct_minus_wrong_logit_mean = +3.221331
  status = stable_correct_gt_wrong

Qwen restore:
  correct_minus_wrong_logit_mean = +3.855661
  status = stable_correct_gt_wrong

LLaVA corrupt:
  correct_minus_wrong_logit_mean = +0.688545
  status = stable_correct_gt_wrong

LLaVA restore:
  correct_minus_wrong_logit_mean = +1.013022
  status = stable_correct_gt_wrong
```

mask-shuffled 运行：
```text
Qwen:
  usable_runs = 56

LLaVA:
  usable_runs = 56
```

mask-shuffled 结果：
```text
Qwen corrupt:
  real_minus_shuffled_logit_mean = +2.717076
  status = stable_real_gt_shuffled

Qwen restore:
  real_minus_shuffled_logit_mean = +3.136998
  status = stable_real_gt_shuffled

LLaVA corrupt:
  real_minus_shuffled_logit_mean = +0.690186
  status = stable_real_gt_shuffled

LLaVA restore:
  real_minus_shuffled_logit_mean = +1.114118
  status = stable_real_gt_shuffled
```

matched controls 的限制：
```text
Qwen:
  source-like group 强于 random / low_delta controls 很稳定；
  但对 delta_matched 和 activation_matched controls，4 个主项里只有 2 个为 positive/stable 或弱正。

LLaVA:
  source-like group 强于 random / low_delta controls 很稳定；
  对 matched controls 4 个主项里有 3 个 positive/stable 或弱正。

解释：
  matched controls 能吸收一部分效果，说明当前 hidden-state bridge 有明确 evidence/target/location specificity，
  但还不能等同于 Gemma 主线中的 traced source node > matched non-source control。
```

结论：
```text
Stage 2N-2 支持两个更强约束：
1. correct target effect > wrong target effect；
2. real evidence mask effect > shuffled mask effect。

这使跨模型结果从“readout/patch 有效”推进到更严格的 hidden-state causal specificity。
但它仍然是 hidden-state-level specificity，不是 feature-level causal bridge，也不是 Gemma-style source-control route replication。
```

## Stage 2N Final Verdict

详细结论文档：`doc/experiments/stage2/056_stage2n_hidden_replication_verdict.md`

最终判断：
```text
Stage 2N heldout hidden-state replication: supported
wrong-target negative control: supported
mask-shuffled negative control: supported
answer_mask condition: supported
union_mask condition: supported
prompt pooled conclusion: supported
feature-level causal bridge: not established in Stage 2N
Gemma-style source-control route replication: not established in Stage 2N
```

现在可以写：
```text
Evidence-region-sensitive hidden-state bridges replicate on Stage 2N heldout localized samples in both Qwen2.5-VL and LLaVA-1.5.
Qwen shows larger effects; LLaVA shows smaller but stable target- and location-specific effects.
This supports that the cross-model phenomenon is not Gemma-only at the hidden-state bridge level.
```

仍然不能写：
```text
Qwen/LLaVA 完整复现了 Gemma 的 source-control causal routes。
Qwen/LLaVA CLT features 已经形成 feature-level causal bridge。
hidden patch 能稳定恢复 decoded answer。
D_visual_only 比 B_direct 更好。
这些 hidden positions 是明确 object-level semantic nodes。
```

预期与实际偏差：
```text
预期：
  如果 Stage 2N 成功，heldout pack 应该在 Qwen 和 LLaVA 上都显示 source-like > random controls，
  并且 wrong-target / mask-shuffled controls 应支持 target/location specificity。

实际：
  以上主项均成立。
  Qwen 效果量较大，LLaVA 效果量较小但稳定。
  但 matched controls 部分吸收效果，说明 specificity 已强于 random controls，
  但还没有达到 Gemma traced source-control route 那种严格节点级证据。
```

对主线 claim 的影响：
```text
Gemma 主线：
  不变。Gemma 仍然是唯一完成 source tracing、node intervention、nearest/random controls、
  wrong-image/region-mask sensitivity、rank/generation linkage 的完整主链模型。

跨模型辅助证据：
  明显增强。Stage 2N 说明 Qwen/LLaVA 至少在 hidden-state bridge 层也有 evidence-region-sensitive、target-specific、
  location-specific 的现象，因此不能简单说“这个现象只在 Gemma 上有效”。

论文口径：
  主结论仍写 Gemma causal route。
  跨模型部分写 auxiliary replication / hidden-state bridge replication。
  不把 Qwen/LLaVA 写成完整 causal route replication。
```

下一步建议：
```text
1. 不要继续盲目扩大 feature smoke；Stage 2M 已经显示 feature-level causal bridge 暂不成立。
2. 如果继续冲更强跨模型 claim，优先做 Qwen 的 attribution-weighted multi-feature patch，而不是单 feature top-k。
3. 对 LLaVA 保持 smaller-effect replication 口径，重点呈现 target/location specificity，而不是强行追求 Gemma 级别 route。
4. 若要升级到真正 source-control route replication，需要为 Qwen/LLaVA 建立 answer-adjacent tracing adapter 和 matched non-source controls。
```

## Stage 2O Run Plan：Feature-Level Bridge 与 Source-Control Route Probe

详细计划文档：`doc/experiments/stage2/057_stage2o_feature_bridge_v2_run_plan.md`

目的：
```text
Stage 2N 已经证明 Qwen/LLaVA 的 evidence-region-sensitive hidden-state bridge 在 heldout localized samples 上复现。
Stage 2O 不继续扩大 hidden-state 样本，而是尝试把跨模型证据下沉到更硬的机制层：

1. Attribution-weighted multi-feature bridge：
   检查 Qwen/LLaVA 是否存在 feature-level causal bridge。

2. Approximate source-control route probe：
   检查是否能构造接近 Gemma source/control 的 feature pair。
```

输入：
```text
Stage 2N hidden bridge strongest rows:
  Qwen = 12 prompt-runs
  LLaVA = 8 prompt-runs

sample_count = 10
sample_ids:
  okvqa_val_136595
  okvqa_val_1593205
  okvqa_val_2496585
  okvqa_val_2683965
  okvqa_val_343215
  okvqa_val_3608785
  okvqa_val_3918255
  okvqa_val_4157235
  okvqa_val_4502065
  okvqa_val_4739195

Qwen:
  model = Qwen2.5-VL-7B-Instruct
  layer = 26
  CLT = KokosDev/qwen2p5vl-7b-clt

LLaVA:
  model = LLaVA-1.5-7B
  layer = 15
  CLT = KokosDev/llava15-7b-clt
```

输出：
```text
doc/experiments/stage2/058_stage2o_attribution_weighted_feature_bridge.md
doc/experiments/stage2/059_stage2o_source_control_route_probe.md
doc/experiments/stage2/060_stage2o_feature_and_route_verdict.md

doc/experiments/stage2/cross_model/stage2o_qwen_attribution_feature_bridge.csv/json
doc/experiments/stage2/cross_model/stage2o_llava_attribution_feature_bridge.csv/json
doc/experiments/stage2/cross_model/stage2o_feature_bridge_summary.csv
doc/experiments/stage2/cross_model/stage2o_feature_bridge_specificity.csv
doc/experiments/stage2/cross_model/stage2o_feature_bridge_decision.json

doc/experiments/stage2/cross_model/stage2o_qwen_source_control_probe.csv/json
doc/experiments/stage2/cross_model/stage2o_llava_source_control_probe.csv/json
doc/experiments/stage2/cross_model/stage2o_source_control_summary.csv
doc/experiments/stage2/cross_model/stage2o_source_control_specificity.csv
doc/experiments/stage2/cross_model/stage2o_source_control_mask_specificity.csv
doc/experiments/stage2/cross_model/stage2o_source_control_decision.json
```

方法：
```text
Stage 2O-1:
  Stage 2M 只按 clean-mask drop 选择 feature。
  Stage 2O 改为 attribution-weighted feature selection：

  feature_score =
    ReLU(weighted clean-minus-mask feature drop)
    * ReLU(decoder_vector · target unembedding vector)

  这会优先选择“既被 evidence mask 破坏，又对 target answer logit 有正贡献”的 feature。

Stage 2O-2:
  从 evidence_attribution_topk 中筛 source feature。
  要求 source feature zeroing 对 target logit 有正 damage。
  再从 activation/drop/attribution-matched controls 中选最接近的 matched control。

  对 source/control 同时做：
    zeroing
    restoration
    answer_mask
    union_mask
    wrong-target check
    shifted-mask check
```

## Stage 2O-1：Attribution-Weighted Multi-Feature Bridge

详细结果文档：`doc/experiments/stage2/058_stage2o_attribution_weighted_feature_bridge.md`

运行可用性：
```text
Qwen:
  usable_runs = 12/12
  rows = 360

LLaVA:
  usable_runs = 8/8
  rows = 240
```

总体判断：
```text
overall_status = partial_or_model_specific_feature_bridge_support
```

Qwen：
```text
status = feature_bridge_one_direction_supported

restore:
  n_rows = 12
  positive_logit_n = 11/12
  mean_logit_effect = +1.057292
  95% CI = [+0.557292, +1.572917]
  effect_status = stable_positive
  above_all_controls_n = 10/12

corrupt:
  n_rows = 12
  positive_logit_n = 10/12
  mean_logit_effect = +0.447917
  95% CI = [+0.223958, +0.708333]
  effect_status = stable_positive
  above_all_controls_n = 6/12
```

LLaVA：
```text
status = feature_bridge_partial_or_weak

restore:
  n_rows = 8
  positive_logit_n = 3/8
  mean_logit_effect = -0.002930
  95% CI = [-0.012695, +0.005859]
  effect_status = not_positive
  above_all_controls_n = 0/8

corrupt:
  n_rows = 8
  positive_logit_n = 4/8
  mean_logit_effect = +0.003906
  95% CI = [-0.018555, +0.028320]
  effect_status = weak_or_heterogeneous_positive
  above_all_controls_n = 2/8
```

结论：
```text
Qwen 的 feature-level bridge 在 restoration 方向成立。
Qwen corrupt 方向自身为 stable_positive，但 control specificity 不够强，因此只写 one-direction support。
LLaVA 没有建立 feature-level bridge。
```

## Stage 2O-2：Cross-Model Source-Control Route Probe

详细结果文档：`doc/experiments/stage2/059_stage2o_source_control_route_probe.md`

运行可用性：
```text
Qwen:
  usable_pairs = 24
  rows = 144

LLaVA:
  usable_pairs = 16
  rows = 96
```

总体判断：
```text
overall_status = model_specific_approx_source_control_support
```

Qwen：
```text
status = approximate_source_control_route_supported

source_control_restore:
  n = 24
  positive_n = 19/24
  mean source_minus_control_logit = +0.395833
  95% CI = [+0.158854, +0.677083]
  status = stable_positive

source_control_zeroing:
  n = 24
  positive_n = 24/24
  mean source_minus_control_logit = +0.489583
  95% CI = [+0.333333, +0.661458]
  status = stable_positive

real_minus_shuffled:
  n = 24
  positive_n = 17/24
  mean = +0.377604
  95% CI = [+0.135417, +0.664062]
  status = stable_positive

correct_minus_wrong:
  n = 4
  positive_n = 4/4
  mean = +0.406169
  95% CI = [+0.382812, +0.435384]
  status = stable_positive
```

LLaVA：
```text
status = source_control_probe_partial

source_control_restore:
  n = 16
  positive_n = 4/16
  mean source_minus_control_logit = -0.000732
  95% CI = [-0.002930, +0.001465]
  status = not_positive

source_control_zeroing:
  n = 16
  positive_n = 12/16
  mean source_minus_control_logit = +0.012207
  95% CI = [+0.005371, +0.019043]
  status = stable_positive

real_minus_shuffled:
  n = 16
  positive_n = 7/16
  mean = +0.003174
  95% CI = [+0.000488, +0.005859]
  status = stable_positive

correct_minus_wrong:
  n = 4
  positive_n = 4/4
  mean = +0.015472
  95% CI = [+0.003295, +0.027649]
  status = stable_positive
```

结论：
```text
Qwen 支持 approximate source-control route probe：
source feature 比 matched control 更能 restore / zero target，并且 real mask > shifted mask、correct > wrong。

LLaVA 是 partial：
zeroing、target specificity、location specificity 有方向；
但 restoration specificity 不成立。
```

## Stage 2O Final Verdict

详细结论文档：`doc/experiments/stage2/060_stage2o_feature_and_route_verdict.md`

最终判断：
```text
Qwen:
  hidden-state bridge: supported by Stage 2N
  feature-level bridge: one-direction supported by Stage 2O
  approximate source-control probe: supported by Stage 2O
  full Gemma-style source-control route replication: not established

LLaVA:
  hidden-state bridge: supported by Stage 2N
  feature-level bridge: not established
  source-control probe: partial
  full Gemma-style source-control route replication: not established
```

现在可以写：
```text
Cross-model evidence is strongest for Qwen:
beyond heldout hidden-state bridge replication, Qwen shows attribution-weighted feature restoration and approximate source-control probe support.
LLaVA replicates the hidden-state bridge and shows partial source-like zeroing evidence, but feature-level route replication remains unproven.
```

仍然不能写：
```text
Qwen/LLaVA 完整复现 Gemma-style source-control route。
LLaVA 复现 feature-level causal bridge。
Qwen/LLaVA feature 是对象级语义节点。
hidden patch 或 feature patch 能稳定恢复完整 decoded answer。
```

---

# Stage 2P：Cross-Modal Feature Evidence 加固与反例诊断

详细文档：

```text
doc/experiments/stage2/061_stage2p_cross_modal_feature_run_plan.md
doc/experiments/stage2/062_stage2p_qwen_heldout_feature_route_replication.md
doc/experiments/stage2/063_stage2p_llava_layer_feature_diagnostic.md
doc/experiments/stage2/064_stage2p_cross_modal_feature_verdict.md
```

## Stage 2P 总目的

Stage 2P 的目的不是继续扩大 hidden-state bridge，而是回答：

```text
1. Qwen 的 Stage 2O feature/source-control 正结果是否能在 heldout prompt-runs 上复现？
2. LLaVA 的 feature-level 失败是否只是 layer/top-k 选择问题？
3. 当前能不能写“跨模型/跨模态 feature-level evidence”？
```

结论边界：

```text
可以加固 Qwen 的 feature/source-control 辅助证据。
不把 LLaVA 失败写成“没有跨模态 feature”。
不写 Qwen/LLaVA 完整复现 Gemma-style source-control route。
```

## Stage 2P-1：Qwen Heldout Feature/Route Replication

输入：

```text
Qwen heldout prompt-runs = 24
来源 = Stage 2N all52
排除 = Stage 2O 已使用的 10 个 sample_id
模型 = Qwen2.5-VL-7B-Instruct
layer = 26
CLT = KokosDev/qwen2p5vl-7b-clt
mask conditions = answer_mask, union_mask
```

输出：

```text
doc/experiments/stage2/cross_model/stage2p_qwen_feature_answer.csv/json
doc/experiments/stage2/cross_model/stage2p_qwen_feature_union.csv/json
doc/experiments/stage2/cross_model/stage2p_qwen_feature_bridge_summary.csv
doc/experiments/stage2/cross_model/stage2p_qwen_feature_bridge_specificity.csv
doc/experiments/stage2/cross_model/stage2p_qwen_feature_bridge_decision.json

doc/experiments/stage2/cross_model/stage2p_qwen_source_control_probe.csv/json
doc/experiments/stage2/cross_model/stage2p_qwen_source_control_summary.csv
doc/experiments/stage2/cross_model/stage2p_qwen_source_control_specificity.csv
doc/experiments/stage2/cross_model/stage2p_qwen_source_control_mask_specificity.csv
doc/experiments/stage2/cross_model/stage2p_qwen_source_control_decision.json
```

运行可用性：

```text
feature bridge answer_mask:
  usable_runs = 24
  rows = 720

feature bridge union_mask:
  usable_runs = 24
  rows = 720

source-control probe:
  usable_pairs = 48
  rows = 288
```

Feature bridge 结果：

```text
Qwen status = feature_bridge_bidirectional_supported
```

Restore：

```text
answer_mask:
  n_rows = 24
  positive_logit_n = 24/24
  mean_logit_effect = +1.279297
  95% CI = [+0.867839, +1.729818]
  effect_status = stable_positive

union_mask:
  n_rows = 24
  positive_logit_n = 23/24
  mean_logit_effect = +1.089844
  95% CI = [+0.630208, +1.617188]
  effect_status = stable_positive

specificity:
  restore pooled n_rows = 48
  above_all_controls_n = 40/48
  mean_positive_control_count = 3.6875 / 4
```

Corrupt：

```text
answer_mask:
  n_rows = 24
  positive_logit_n = 23/24
  mean_logit_effect = +0.747396
  95% CI = [+0.434896, +1.132812]
  effect_status = stable_positive

union_mask:
  n_rows = 24
  positive_logit_n = 20/24
  mean_logit_effect = +0.721354
  95% CI = [+0.348958, +1.161458]
  effect_status = stable_positive

specificity:
  corrupt pooled n_rows = 48
  above_all_controls_n = 32/48
  mean_positive_control_count = 3.270833 / 4
```

Source-control 结果：

```text
Qwen status = approximate_source_control_route_supported
```

核心指标：

```text
source_control_restore:
  n = 48
  positive_n = 41/48
  mean source_minus_control_logit = +0.476888
  95% CI = [+0.305664, +0.683268]
  status = stable_positive

source_control_zeroing:
  n = 48
  positive_n = 48/48
  mean source_minus_control_logit = +0.863281
  95% CI = [+0.713542, +1.022135]
  status = stable_positive

real_minus_shuffled:
  n = 48
  positive_n = 43/48
  mean = +0.455404
  95% CI = [+0.283203, +0.659180]
  status = stable_positive

correct_minus_wrong:
  n = 4
  positive_n = 4/4
  mean = +0.669886
  95% CI = [+0.452488, +0.887284]
  status = stable_positive
```

Stage 2P-1 结论：

```text
Qwen heldout feature bridge: bidirectional supported
Qwen heldout approximate source-control probe: supported
```

这说明 Stage 2O 的 Qwen 正结果不是少数 prompt-runs 的偶然结果。

## Stage 2P-2：LLaVA Layer/Feature Diagnostic

输入：

```text
LLaVA diagnostic prompt-runs = 8
来源 = Stage 2O LLaVA diagnostic rows
模型 = LLaVA-1.5-7B
CLT = KokosDev/llava15-7b-clt
layers = 12, 15, 18, 21
top_k = 1, 8, 32
mask_condition = union_mask
```

运行可用性：

```text
12 configurations completed
每个配置 usable_runs = 8
每个配置 rows = 240
```

总体判断：

```text
status = llava_layer_sweep_weak_or_partial
strong_configs = []
observed_layers = 12, 15, 18, 21
missing_layers = []
```

最接近正结果的配置：

```text
layer 18, top_k 32, corrupt:
  n_rows = 8
  positive_logit_n = 7/8
  mean_logit_effect = +0.060547
  95% CI = [+0.004883, +0.103516]
  effect_status = stable_positive
  above_all_controls_n = 1/8
  mean_positive_control_count = 2.625 / 4
```

Stage 2P-2 结论：

```text
LLaVA 有 weak layer-dependent feature signals。
但 source-like feature 没有稳定强于 matched controls。
因此不能写 LLaVA feature-level bridge supported。
也不能写 LLaVA 没有跨模态 feature。
```

## Stage 2P Final Verdict

当前可写：

```text
Cross-model evidence is strongest for Qwen:
Qwen shows heldout-supported feature-level restoration/corruption and approximate source-control probe effects.
LLaVA replicates hidden-state bridge and shows weak layer-dependent feature signals,
but feature-level route localization remains unproven.
```

中文版本：

```text
跨模型证据最强的是 Qwen：
Qwen 已经有 heldout-supported feature/source-control 辅助证据。
LLaVA 有 hidden-state bridge 复现和弱 feature 诊断信号，
但 feature-level route 尚未证成。
```

仍不能写：

```text
Qwen/LLaVA 完整复现 Gemma-style source-control route。
LLaVA 没有跨模态 feature。
Qwen feature 是对象级语义节点。
D_visual_only 比 B_direct 更好。
```

---

# Stage 2P 到 Stage 2Q：当前结论与完整主线复现计划

详细文档：

```text
doc/experiments/stage2/065_stage2p_to_stage2q_current_verdict.md
doc/experiments/stage2/066_stage2q_cross_model_gemma_style_route_replication_run_plan.md
```

当前项目级判断：

```text
局部机制 claim 已经成立。
跨模型 readout / hidden-state claim 已经成立。
Qwen feature-level and approximate source-control auxiliary evidence 已经成立。
LLaVA feature-level route 尚未成立。
full cross-model Gemma-style route replication 尚未完成。
```

Transcoder 可比性 caveat：

```text
Gemma 主线使用 tianhux2/gemma3-4b-it-plt，
这是当前 Gemma3 ReplacementModel pipeline 原生兼容的 PLT / transcoder set。

Qwen 使用 KokosDev/qwen2p5vl-7b-clt，
有 config.yaml 和 layer_*.safetensors，hook 为 blocks.{layer}.hook_resid_pre/post；
但当前仍是 native Qwen forward + CLT feature patch/probe，
不是完整 ReplacementModel attribution graph。

LLaVA 使用 KokosDev/llava15-7b-clt，
是 custom transcoder_L*.pt + mapping_L*.pt 格式，没有标准 config.yaml；
因此 LLaVA feature-level 结果更应作为 diagnostic，不应和 Gemma source node 完全等价。
```

如果只写收窄后的论文主结论，可以下定论：

```text
在 localized、strong image-dependence、证据区域可标注的 VQA 样本中，
Gemma 存在完整 evidence-region-sensitive answer support route；
Qwen 提供 heldout-supported feature/source-control 辅助证据；
LLaVA 提供 hidden-state 层面的跨模型辅助证据。
```

如果要写更强的跨模型完整机制结论，还不能下定论。

还缺 Stage 2Q：

```text
Qwen / LLaVA Gemma-style 完整主线复现实验：

source discovery / source-like route selection
→ source vs matched control intervention
→ evidence region mask sensitivity
→ real mask > shifted / shuffled mask
→ correct target > wrong target
→ target rank / first-token / decoded answer linkage
```

Stage 2Q 成功前不能写：

```text
Qwen/LLaVA fully replicate Gemma-style source-control routes。
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
LLaVA hidden-state bridge is replicated,
but feature/source route localization remains unproven.
```
