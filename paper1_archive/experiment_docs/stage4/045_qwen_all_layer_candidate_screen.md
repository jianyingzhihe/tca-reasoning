# 目的

记录 Stage4-044 的全层候选筛选策略，避免把“没找到”误写成“客观不存在”。

# 输入

- primary/strict prompt-runs。
- 每张图的 clean、answer_mask、union_mask、shifted_mask、shuffled_mask 条件。
- Qwen PLT layers `0..27`。

# 输出

每层候选 manifest，字段包括 layer、source feature、source position、evidence specificity、target attribution、wrong-target attribution、clean activation、candidate rank、exact-position 标记和 sample/prompt 信息。

# 方法

候选分数综合：

- evidence specificity：`clean - real_mask` 是否强于 shifted/shuffled。
- target attribution：feature decoder direction 是否支持 gold target token。
- wrong-target penalty：避免同样支持 wrong token。
- clean activation：避免候选本身几乎不激活。
- hidden mediation weight：可选参考 Stage4-033/038 的 hidden layer evidence route 强度。

每层保留 top 候选，再按 prompt-run 限制，防止少数样本或单一数字答案支配全局。

# 结果

Smoke candidate screen 已完成：

- L0: `96` candidates / `6` prompt-runs / `48` main rows。
- L7: `96` candidates / `6` prompt-runs / `48` main rows。
- L14: `96` candidates / `6` prompt-runs / `48` main rows。
- L21: `96` candidates / `6` prompt-runs / `48` main rows。
- L27: `0` candidates。

Primary full 已启动。full 模式在 discovery 后会把每层 main intervention candidates cap 到 top `256`，防止少数层或 prompt-runs 造成无界干预成本。

# 预期与实际偏差

如果 exact-position candidates 极少，说明当前 PLT feature basis 对 evidence-position route 的定位能力有限，但还不能推出 Qwen hidden route 不存在。

# 结论

候选筛选只用于 discovery。真正机制结论必须依赖后续 causal validation。
