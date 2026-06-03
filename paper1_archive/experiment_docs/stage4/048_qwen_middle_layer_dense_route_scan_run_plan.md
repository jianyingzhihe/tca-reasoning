# 目的

在 Stage4-044 broad all-layer screen 之后，集中扫描 Qwen2.5-VL-PLT 中层 `10..17`，判断是否存在更精细的 sparse/distributed PLT evidence-linked route。

# 输入

- paperpack72 primary/strict prompt-runs。
- Qwen2.5-VL base 与 `KokosDev/qwen2p5vl-7b-plt`。
- answer/union/shifted/shuffled masks。
- Stage4-044 partial results：L10/L13/L15 有 source-control 近通过；L14 hidden route 先验强但 sparse PLT 弱；L17-L20 转弱。

# 输出

- `stage4_qwen_middle_dense_*_candidates.csv/json`
- `stage4_qwen_middle_dense_*_zeroing_raw.csv/json`
- `stage4_qwen_middle_dense_*_group_raw.csv/json`
- `stage4_qwen_middle_dense_summary.csv`
- `stage4_qwen_middle_dense_specificity.csv`
- `stage4_qwen_middle_dense_decision.json`

# 方法

当前 all-layer full 等 L21 grouped restore 完成后停止，不继续 L22-L27。Stage4-048 只跑中层 dense scan：

- layers: `10,11,12,13,14,15,16,17`
- position groups: `visual_only`, `answer_adjacent_only`, `visual_answer`
- candidate pool: `32768`
- top-per-prompt-run: `64`
- main-per-prompt-run: `32`
- per-layer/position-group main cap: `1024`
- grouped restore topK: `1,2,4,8,16,32,64,128,256`

# 结果

待运行。

# 预期与实际偏差

若 source-control 成立但 real-vs-shifted/shuffled 不成立，结论写 `answer-support but not evidence-linked`。若 dense scan 仍失败，结论写 `public Qwen-PLT sparse localization failed under middle-layer dense search`，不否定 hidden-level mechanism。

# 结论

待 `051_qwen_middle_layer_dense_route_verdict.md` 汇总。
