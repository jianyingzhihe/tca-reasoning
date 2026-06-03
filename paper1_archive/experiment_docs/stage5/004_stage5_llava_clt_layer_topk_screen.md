# 目的

重新评估 LLaVA-CLT 的 layer/topK 可见性，避免从 L15 或少量 Stage3 结果直接下负结论。

# 输入

- LLaVA-1.5 base。
- `KokosDev/llava15-7b-clt`。
- paperpack72 primary/strict。

# 输出

- LLaVA layer/topK feature/source-control artifacts。
- usable/missing layer report。
- LLaVA near-pass table。

# 方法

Smoke layers: `0,12,15,18,21,30`，topK `1,8,32`。若某层 asset 或 loader 不可用，记录为 missing/blocked，不把它当机制负结果。

Screen 只使用 smoke 可用层，topK `1,4,8,16,32,64`。

# 结果

待运行。

# 预期与实际偏差

Stage2/3 LLaVA-CLT 曾出现 hidden-level upper-bound 与弱 feature/source diagnostic。Stage5 只判断 public LLaVA-CLT 的可定位性，不否定 LLaVA hidden mechanism。

# 结论

待运行后更新。
