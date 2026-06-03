# 目的

比较 CLT 与 PLT/hidden results 的表示差异，避免把某一种转码器可见性当成模型机制本身。

# 输入

- Stage4 Qwen PLT all-layer/hidden/error results。
- Stage3/5 Qwen-CLT results。
- Stage3/5 LLaVA-CLT results。

# 输出

- representation comparison table。
- model/asset-dependent visibility summary。

# 方法

对同一 paperpack 口径比较：hidden route、PLT sparse feature、PLT reconstruction error、CLT feature/source route、CLT distributed route。

# 结果

待运行。

# 预期与实际偏差

若 Qwen hidden route 与 CLT route 成立但 PLT sparse route 失败，写 representation-dependent localization。若 LLaVA hidden upper-bound 成立但 CLT feature route 失败，写 CLT localization weak/failed。

# 结论

待运行后更新。
