# Stage4-031 Qwen Prior Experiment Critical Audit

## 目的

用审稿人视角复查 Stage3/Stage4 Qwen2.5-VL-PLT 实验，回答三个问题：

1. 是否已经遍历所有节点？
2. 是否能保证找到所有 correct answer 节点？
3. 是否只测试了 layer 26，是否遗漏更早/更晚层？

## 简短结论

不能写“已经遍历所有节点”，也不能写“已保证找到所有 correct 节点”。

当前已经完成的是多轮 bounded search：

- hidden 层：已扫 `12,16,20,22,24,26,27`，primary/strict 都显示 layer 26 hidden evidence action 最强，但 wrong-target specificity 未闭合。
- automatic Qwen source-tracing adapter：主要围绕 layer 26，后续加了 layer 22/24 smoke 和 layer 26 topK/position sensitivity；仍不是全层全节点遍历。
- PLT layer sweep：已在 `26,12,24,22` 做 evidence-first discovery、zeroing 和 grouped restore；每层 `96` prompt-runs、`384` candidate rows、`192` main candidates。但这不是所有 features/positions。
- Gemma3-PLT source tracing 不等于单层 layer 26；Gemma paperpack nodes 覆盖多层，primary full `nodes_detailed_controlled.csv` 中有 `36` 个 layer 值（含 `-1` 和模型层），说明 Gemma 主线是 graph/route 级多层选择。

## 已发现或已修复的逻辑问题

### 1. Hidden-state off-by-one

Stage4 早期 Qwen adapter/hidden patch 存在潜在 off-by-one：`outputs.hidden_states[layer]` 与 `language_model.layers[layer]` 的 hook 输出不完全对齐。HF decoder hidden states 通常为 `[embedding, layer0_out, layer1_out, ...]`。

已修复：

- Hidden lattice patch `language_model.layers[L]` 时使用 `hidden_states[L+1]`。
- Raw CSV 新增 `hidden_state_index` 便于审计。

影响：

- 早期 layer 26 direct adapter negative 不能作为最终 negative。
- 后续 hook-aligned rerun 和 Stage4-024 hidden lattice 更可信。

### 2. Layer out-of-range

原计划含 `28,30`，但当前 Qwen2.5-VL-7B language layer index 为 `0..27`。已修复为自动跳过越界层，并默认扫 `12,16,20,22,24,26,27`。

影响：

- 不能写“测过 28/30”。
- 可以写“在当前可 hook 的 Qwen language layers 中，Stage4-024 覆盖了高层 27，但尚未做 full all-layer exhaustive hidden sweep”。

### 3. Qwen automatic source tracing 不是 Gemma ReplacementModel 等价实现

Qwen adapter 是 schema-compatible / Qwen-native approximation，不是 Gemma ReplacementModel 的内部等价替代。

影响：

- Adapter 失败不能直接推出 Qwen 没有机制。
- 只能写“当前 Qwen adapter 下未复现 Gemma-style automatic route selection”。

### 4. PLT feature search 不是全节点遍历

Stage4-024 PLT sweep 每层 discovery 设置：

- `candidate_pool_size=8192`
- `top_per_prompt_run=4`
- `main_per_prompt_run=2`
- `position_group=visual_answer`
- `top_ks=1,4,8,16,32,64`

实际 primary full:

- L12: `384` candidates / `192` main / `96` prompt-runs。
- L22: `384` candidates / `192` main / `96` prompt-runs。
- L24: `384` candidates / `192` main / `96` prompt-runs。
- L26: `384` candidates / `192` main / `96` prompt-runs。

这证明 search 是严格、可复现的 candidate search，但不是 all feature × all position × all layer exhaustive enumeration。

## 当前证据边界

### 可以写

Qwen2.5-VL-PLT 在 paperpack 上有稳定 hidden-level evidence-mask-sensitive causal action。Primary 与 strict 都显示 layer 26 `visual+answer / answer_mask / restore` 对 target logit/rank 强，且 real mask 强于 shifted/shuffled。

Qwen-PLT evidence-first feature discovery 能找到强 evidence-sensitive features。PLT activation-drop real-vs-shifted/shuffled 很强。

### 不能写

Qwen 已完整复现 Gemma-style source tracing。

Qwen 已经证明存在 target-specific PLT source-control route。

Qwen 没有跨模态机制。

已遍历所有节点或已保证找出所有 correct 节点。

## 推荐下一步

下一步应做 Stage4-032 bounded exhaustive retest，而不是继续只扩大当前 top candidate。

核心变化：

- 层：覆盖所有可用 PLT 层 `0..26` 或全部 asset-supported layers，而不是只看 `26,12,24,22`。
- hidden：覆盖所有可 hook language layers `0..27` 的 coarse sweep，再对强层做 dense sweep。
- 位置：同时测试 `visual_only`、`answer_adjacent_only`、`visual_answer`、`top_hidden_delta`，不要只用 `visual_answer`。
- feature：每层每 prompt-run 输出 coverage manifest，记录候选占总 active feature 的比例、topK 之外的 residual risk。
- intervention：从 single-layer 扩展到 multi-layer grouped patch，测试 `distributed_route`。

