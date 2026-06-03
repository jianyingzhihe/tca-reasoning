# 目的

避免继续默认 Qwen-CLT 只看 L26。对 Qwen-CLT 全可用层做 bounded screen，找出 CLT 表示中可能的 evidence-to-answer route 层与 topK。

# 输入

- Qwen2.5-VL base。
- `KokosDev/qwen2p5vl-7b-clt` layers `0..26`。
- paperpack72 primary/strict。
- answer/union/shifted/shuffled masks。

# 输出

- `stage5_clt_heterogeneity_qwen2p5vl_clt_*_feature_union.csv/json`
- `stage5_clt_heterogeneity_qwen2p5vl_clt_*_source_control.csv/json`
- per-layer/topK summary 与 near-pass table。

# 方法

Smoke: layers `0,7,14,21,26`，topK `1,8,32`，6 prompt-runs。

Screen: layers `0..26`，topK `1,4,8,16,32,64`，primary pack。使用 Qwen 自己的 CLT activations、mask sensitivity、target attribution 和 hidden-delta positions 生成候选，不读 Gemma node ids。

# 结果

待运行。

# 预期与实际偏差

Stage4 Qwen-PLT 显示 hidden route 在 L14，但 sparse PLT route 未闭合。因此 Qwen-CLT 若在 L14 附近或其他中层出现信号，将支持 representation-dependent visibility。

# 结论

待运行后更新。
