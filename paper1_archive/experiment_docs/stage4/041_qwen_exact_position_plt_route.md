# Stage4-041 Qwen Exact-Position PLT Route

## 目的

检查 Adapter V4 失败是否来自大量 `same_feature` fallback。只允许同一 sample、prompt、position、feature 同时满足 evidence sensitivity 与 answer support。

## 输入

- V4 primary/strict manifest
- V4 zeroing/group artifacts

## 输出

- `stage4_qwen_native_route_exact_position_*`

## 方法

过滤 `v4_match_level == exact_pos_feature`，重新统计：

- source > controls
- correct > wrong
- real mask restore > shifted/shuffled
- answer/union mask 分开

## 结果

待运行。

## 预期与实际偏差

如果 exact candidates 太少，写 `exact PLT route candidate sparse`，不写机制负结论。

## 结论

待运行。
