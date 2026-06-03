# Stage4-002 Qwen Source-Tracing Adapter Design

## 目的

Gemma 主线使用 `ReplacementModel/HookedVLTransformer` 生成 attribution graph。该路径目前是 Gemma3-oriented，不能直接加载 Qwen2.5-VL。因此 Stage4 先实现一个 Qwen-native adapter：使用 Qwen 原生 forward、Qwen PLT encode/decode，并输出现有 compare 脚本可读的 graph schema。

## Adapter 范围

- 支持 Qwen2.5-VL 图文输入和 paperpack prompt。
- 支持 layer 26 PLT encode/decode。
- 支持 gold answer first-token target。
- 支持 graph `.pt` schema：`cfg/input_tokens/active_features/selected_features/logit_tokens/logit_probabilities/activation_values/adjacency_matrix`。
- 支持 A/B prompt graph compare 与 Qwen feature node intervention smoke。

## Graph Layout

节点顺序与 Gemma graph 对齐：

`[feature nodes, error nodes, token nodes, logit nodes]`

当前 Qwen adapter 的第一版 graph 是 direct answer-aligned graph：

- feature node: `(layer, position, feature_id)`。
- feature → target logit edge: `feature_activation * decoder_vector dot target_logit_direction`。
- token → feature edge: same-position activation bridge，用于 controlled backtrace 形成 feature/token route。
- logit node: gold answer first token。

这不是 Gemma ReplacementModel 的完整内部 attribution 等价实现；它是一个 schema-compatible Qwen source-tracing adapter，用于检验 Qwen PLT feature nodes 是否能形成可比较、可干预的 source/control route。

## 风险与防线

- 如果 graph 只能生成但 intervention 不成立，只能写 `adapter graph evidence insufficient`。
- 如果 graph 生成失败或模型加载失败，写 `qwen_adapter_blocked`。
- 如果 graph/intervention/controls 全部成立，才升级为 `qwen_full_source_tracing_supported`。

