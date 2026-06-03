# 目的

验证 dense candidates 是否真实因果支撑 target，并判断这种支撑是否特异于 evidence mask。

# 输入

- Stage4-048 dense candidates。
- image/mask assets。
- Qwen2.5-VL-PLT。

# 输出

- clean zeroing / restore raw rows。
- source-control specificity。
- real-vs-shifted/shuffled specificity。
- wrong-target specificity。
- near-pass case table。

# 方法

每个 candidate 运行 clean zeroing 与 mask restore，control 包括 matched feature、same-feature-random-position、random active feature。grouped restore topK 为 `1,2,4,8,16,32,64,128,256`。

near-pass rows 可加跑 scale `0.5,1,2,4` 的 decoder-add diagnostic，用于判断 operator 是否太弱。

# 结果

待运行。

# 预期与实际偏差

若 grouped restore 成立但 top1 不成立，写 distributed PLT route。若 clean zeroing 成立但 mask specificity 不成立，写 answer-support not evidence-linked。

# 结论

待运行后更新。
