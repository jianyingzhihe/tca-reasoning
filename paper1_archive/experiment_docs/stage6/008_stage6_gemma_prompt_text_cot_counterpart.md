# Stage6-008 Gemma Prompt/Text/CoT Counterpart

## 目的

补齐 Stage6 的 Gemma 对照。前一轮 Stage6 `prompt_text_cot` 已完成的是 Qwen route-first feature-node 探索，不是 Gemma+Qwen 双模型探索。Gemma 之前已经在 Stage3 证明了 PLT sparse source-tracing 主链路，但还没有跑过与 Qwen Stage6 同构的“问题改写 / visual prompt / CoT prompt”探索。

本实验不新增主 claim，只回答一个二级问题：

```text
Gemma 已经发现的 evidence-to-answer sparse source-tracing route，
在等价问题改写和 CoT/visual prompt 改变下是否仍然稳定？
```

## 输入

使用与 Qwen Stage6 相同的 12 个样本：

```text
visual_readout: 5
symbol_text_reading: 2
compact_scene_inference: 5
```

每个样本使用 3 个问题文本版本：

```text
original
paraphrase_1
paraphrase_2
```

每个问题文本使用 4 个 prompt family：

```text
B_direct
D_visual_only
C_step_only
A_step_visual
```

## 方法

Gemma 不复用 Qwen 的 route-first node manifest。Gemma 的强证据层级是 PLT source-tracing graph，所以本轮对 Gemma 采用 graph-level counterpart：

```text
A side = 当前 condition prompt
B side = 同一样本的 B_direct / original baseline
```

也就是说，每个 sample / prompt / rewrite condition 都生成一张 Gemma answer-aligned attribution graph，并与同一样本的 baseline graph 比较。

核心比较指标：

```text
node_overlap_jaccard
edge_overlap_jaccard
delta_target_total_in_abs
delta_target_feature_ratio
delta_target_error_ratio
delta_traced_nodes
delta_traced_edges
```

## 输出

artifact prefix：

```text
stage6_gemma_prompt_text_cot_*
```

本地输出目录：

```text
doc/experiments/stage6/cross_model/
```

脚本：

```text
scripts/local/build_stage6_gemma_prompt_text_pack.py
scripts/local/run_stage6_gemma_prompt_text_cot_remote.py
scripts/local/analyze_stage6_gemma_prompt_text_cot.py
```

## 判读边界

如果 Gemma 在 paraphrase 下 route overlap 仍高、path mass / route size 改变不大，可以写：

```text
Gemma's sparse evidence-to-answer route is robust to meaning-preserving question rewrites.
```

如果 CoT/visual prompt 改变 route overlap 或 composition，可以写：

```text
CoT/visual instructions modulate route allocation, but this is a prompt-level modulation of an already established sparse route.
```

如果 CoT 导致格式失败或 answer-aligned graph 缺失，只写工程/格式敏感性，不写机制失败。

## 与 Qwen 的关系

Qwen Stage6 使用的是 route-first feature nodes，因为 Qwen 当前最强 feature-level 证据在 node-level，而 grouped feature route 尚未完全闭合。

Gemma Stage6 使用 source-tracing graph，因为 Gemma 已经有完整 sparse source-tracing route。

二者不是完全相同的操作层级，但回答同一个探索性问题：

```text
已发现的 evidence-to-answer 内部路径是否主要锚定视觉证据，
以及它是否会被问题文本改写 / CoT prompt 明显调节？
```

