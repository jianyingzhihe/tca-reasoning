# 目的

重新生成中层 Qwen-native PLT candidates，避免 broad screen 因候选预算、position group 混合、或 main cap 太小漏掉更合适的 source nodes。

# 输入

- paperpack72 primary prompt-runs。
- layers `10..17`。
- position groups `visual_only`, `answer_adjacent_only`, `visual_answer`。

# 输出

每个 layer/position group 的 dense candidates manifest，保留 feature id、position、candidate rank、best mask condition、target attribution、wrong-target contribution、numeric/non-numeric、mask paths。

# 方法

每个 layer/position group 单独 discovery：

- `candidate_pool_size=32768`
- `top_per_prompt_run=64`
- `main_per_prompt_run=32`
- main cap `1024`

primary 只用于发现；strict 不参与挑规则。

# 结果

待运行。

# 预期与实际偏差

若 exact/visual candidates 稀疏而 answer-adjacent candidates 较强，说明当前 Qwen-PLT source route 更像 answer-support，而非直接 evidence-region-linked route。

# 结论

待运行后更新。
