# Stage4-001 Qwen Full Source-Tracing Run Plan

## 目的

Stage4 的核心目标是把 `Qwen2.5-VL-PLT` 从 Stage3 的 `approximate feature/source-control support` 推进到真正可判定的 `Gemma-style source tracing replication`。这里的关键不是继续扩大近似 feature patch，而是实现一个 Qwen 版 answer-aligned attribution graph adapter，使 Qwen 的 graph `.pt` 能被现有 `trace_compare_ab_controlled.py`、node/control compare 和 intervention smoke 读取。

## 输入

- paperpack72 primary / strict manifest 与 prompt-runs。
- base model: `Qwen/Qwen2.5-VL-7B-Instruct`。
- PLT asset: `KokosDev/qwen2p5vl-7b-plt`。
- prompts: `B_direct` 与 `D_visual_only`。
- target alignment: gold answer first-token target，与 Stage3 Qwen-PLT paperpack 和 Gemma paperpack 口径保持一致。

## 输出

- `cross_model/stage4_qwen_source_tracing_{pack}_{mode}_meta_a.csv`
- `cross_model/stage4_qwen_source_tracing_{pack}_{mode}_meta_b.csv`
- `cross_model/stage4_qwen_source_tracing_{pack}_{mode}_valid_samples.csv`
- `cross_model/stage4_qwen_source_tracing_{pack}_{mode}_sample_compare_controlled.csv`
- `cross_model/stage4_qwen_source_tracing_{pack}_{mode}_nodes_detailed_controlled.csv`
- `cross_model/stage4_qwen_source_tracing_{pack}_{mode}_edges_detailed_controlled.csv`
- `cross_model/stage4_qwen_source_tracing_{pack}_{mode}_intervention.csv`
- `cross_model/stage4_qwen_source_tracing_{pack}_{mode}_decision.json`

## 方法

1. Qwen adapter 对每个 sample/prompt 做真实图文 forward，提取 language layer 26 hidden state。
2. 使用 Qwen PLT 对该层 hidden state encode，选取对 gold answer first token 贡献最高的 active feature-position nodes。
3. 保存 schema-compatible graph `.pt`：`cfg/input_tokens/active_features/selected_features/logit_tokens/logit_probabilities/activation_values/adjacency_matrix`。
4. 用现有 `trace_compare_ab_controlled.py` 对 `D_visual_only` 和 `B_direct` graph 做 controlled backtrace compare。
5. 用 Qwen intervention smoke 对 compare 中的 feature nodes 做 zeroing，判断 target logit/rank 是否按 source-control 方向变化。

## 判据

- `smoke`：3 cases × 2 prompts 产生 graph、meta、compare、nodes、edges、intervention rows。
- `primary72 full`：valid prompt-runs ≥ 120/144，A/B graph coverage ≥ 80%，compare/node/edge 非空，intervention 主项方向成立。
- `strict72 sensitivity`：graph/compare 与 primary 同方向；若 intervention 被资源工程阻塞，只写 strict graph/compare sensitivity。
- 只有 adapter 跑通且 controls/source route 成立，才写 `qwen_full_source_tracing_supported`。

## 结论边界

- Adapter blocked 不等于 Qwen 负结果。
- Adapter 跑通但 source/control/negative controls 不成立时，才允许写 `Qwen does not support Gemma-style source tracing under this adapter and paperpack`。
- 即使 Qwen 成功，也不写所有 VLM 都复现。

## 当前执行状态

Primary smoke、primary72 full、strict72 full 均已完成。工程 schema、graph compare、intervention runner 都跑通；primary/strict 都满足 graph success 门槛。但 source-traced feature node zeroing 方向不成立，因此当前 verdict 是：

`Qwen2.5-VL-PLT approximate source-control support remains, but Gemma-style source tracing replication is not supported under the current Qwen answer-aligned PLT adapter.`

下一步进入 CLT finalization，不再用 Qwen full source-tracing adapter 继续扩大样本。
