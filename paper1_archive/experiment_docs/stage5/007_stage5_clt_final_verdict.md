# 目的

汇总 Stage5 CLT 机制异质性图谱结论。

# 输入

- Stage5 Qwen-CLT screen/validation/strict artifacts。
- Stage5 LLaVA-CLT screen/validation/strict artifacts。
- Stage4 PLT/hidden comparison artifacts。

# 输出

最终状态限定为：

- `qwen_clt_route_supported`
- `qwen_clt_distributed_route_supported`
- `llava_clt_layer_dependent_weak_route`
- `clt_localization_failed_hidden_supported`
- `clt_not_established_under_tested_assets`
- `blocked`

# 方法

只把 primary + strict 均满足的 gates 写成 final support。只通过 primary 或 smoke 的结果写成 diagnostic。

# 结果

待运行。

# 预期与实际偏差

CLT success strengthen cross-representation robustness；CLT failure does not weaken PLT/hidden evidence。

# 结论

待运行后更新。
