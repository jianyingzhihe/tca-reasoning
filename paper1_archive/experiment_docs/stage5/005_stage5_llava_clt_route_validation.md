# 目的

对 LLaVA-CLT near-pass configs 做 validation，判断其是否为 layer-dependent weak route、distributed route，或 CLT localization failure。

# 输入

- LLaVA-CLT screen near-pass table。
- paperpack72 primary/strict。

# 输出

- validation raw CSV/JSON。
- strict confirmation raw CSV/JSON。
- LLaVA-CLT decision JSON。

# 方法

验证 source/control、real-vs-shuffled、wrong-target、rank/first-token gates。若 LLaVA 只有 partial gates，通过 verdict 写为 weak/diagnostic，不写没有机制。

# 结果

待运行。

# 预期与实际偏差

若所有 tested layers/topKs 失败，结论限定为 `clt_not_established_under_tested_assets`。

# 结论

待运行后更新。
