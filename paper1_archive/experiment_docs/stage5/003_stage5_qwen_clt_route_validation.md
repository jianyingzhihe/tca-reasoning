# 目的

对 Qwen-CLT screen 中的 near-pass 层/topK 做 full validation，判断它是稳定 CLT route、分布式 CLT route，还是只是不稳定边缘信号。

# 输入

- Qwen-CLT primary screen near-pass table。
- paperpack72 primary/strict。

# 输出

- full validation raw CSV/JSON。
- strict confirmation raw CSV/JSON。
- Qwen-CLT decision JSON。

# 方法

Full validation 包括 feature/source-control、real-vs-shifted/shuffled、wrong-target specificity、rank/first-token或 sequence-score bridge。strict 只复验 primary-selected configs，不重新挑层/topK。

# 结果

待运行。

# 预期与实际偏差

若 sparse topK 失败但 grouped/multi-layer topK 成立，写 `qwen_clt_distributed_route_supported`，不写 Gemma-style full replication。

# 结论

待运行后更新。
