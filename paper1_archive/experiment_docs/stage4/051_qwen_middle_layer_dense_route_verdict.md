# 目的

汇总 Qwen 中层 dense PLT route scan。

# 输入

- Stage4-048/049/050 artifacts。
- Stage4-038 hidden-to-PLT mediation results。
- Stage4-044 all-layer partial results。

# 输出

最终状态限定为：

- `qwen_middle_sparse_plt_route_supported`
- `qwen_middle_distributed_plt_route_supported`
- `qwen_answer_support_not_evidence_linked`
- `qwen_plt_localization_failed_hidden_supported`
- `blocked`

# 方法

只把 primary + strict 均通过的 gates 写成支持。primary-only near-pass 只写 diagnostic。

# 结果

待运行。

# 预期与实际偏差

如果中层 dense scan 仍不能闭合 real-vs-shifted/shuffled，则进一步支持 Qwen route 主要在 hidden residual / PLT reconstruction error，而不是 public sparse PLT feature basis。

# 结论

待运行后更新。
