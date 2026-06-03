# 目的

汇总 Stage4-044 全层 bounded exhaustive search 对 Qwen PLT route 的结论。

# 输入

- `stage4_qwen_all_layer_bounded_exhaustive_summary.csv`
- `stage4_qwen_all_layer_bounded_exhaustive_specificity.csv`
- `stage4_qwen_all_layer_bounded_exhaustive_decision.json`
- Stage4-038 hidden-to-PLT mediation results

# 输出

最终状态限定为：

- `qwen_sparse_plt_route_supported`
- `qwen_distributed_plt_route_supported`
- `qwen_route_may_live_in_plt_error_confirmed`
- `qwen_sparse_plt_route_not_supported_under_bounded_exhaustive`
- `blocked`

# 方法

primary 用于发现层/候选/operator。strict 只在 primary near-pass 后确认，不参与调参。所有结论分开报告 source/control、real-vs-shifted/shuffled、correct-vs-wrong、rank/logit gate。

# 结果

Smoke decision: `qwen_route_may_live_in_plt_error_confirmed`，但该状态仅表示 smoke 没有推翻 Stage4-038 的 hidden/residual-error 结论，不是 final verdict。

Primary full 正在运行。当前远端已完成 L0-L10 的 candidates/zeroing/grouped restore，并正在 L11 discovery。本地已完整拉回并分析 L0-L9。

L0-L4 partial analysis:

- 每层 `1536` candidates，`768` main candidates，覆盖 `96` prompt-runs。
- 每层 zeroing raw rows: `30720`。
- 每层 grouped restore raw rows: `21504`。
- 没有任何层同时满足 sparse/source-control、real-vs-shifted/shuffled、correct-vs-wrong gates。
- 仅观察到很弱的 rank/correct-target 边缘信号：L0/L1/L2/L3/L4 的 `zeroing_rank_effect` CI low 为正，但 positive fraction 只有约 `0.057-0.069`；L3 的 `zeroing_correct_minus_wrong` CI low 为正，但 positive fraction 约 `0.34`。这些不足以支持 sparse PLT route。
- L0-L4 的 grouped restore/source-control 或 real-mask specificity 未形成稳定 positive gate。

Partial metric artifact: `cross_model/stage4_qwen_all_layer_bounded_exhaustive_L0_L4_partial_metrics.csv`。

临时判断：前 5 层没有显示 Qwen sparse/topK PLT route；这与 Stage4-038 的 `route_may_live_in_plt_error` 方向一致。但这不是最终结论，因为 L5-L27 还未完整本地分析。

L5-L9 partial analysis:

- 每层 `1536` candidates，`768` main candidates，覆盖 `96` prompt-runs。
- 每层 zeroing raw rows: `30720`。
- 每层 grouped restore raw rows: `21504`。
- L5/L9 的 `zeroing_rank_effect` CI low 为正，但 positive fraction 仍只有约 `0.056-0.063`，不足以支持稳定 answer-support route。
- L6 `restore_source_minus_controls / top1 / union_mask` 为正：mean `0.0157`，CI low `0.0033`，positive fraction `0.50`。这是一个可跟踪的边缘信号，但它没有同时伴随稳定 real-vs-shifted/shuffled specificity。
- L5/L7 的部分 topK answer-mask restore source/control mean 为正，但 CI 跨 0。
- L7/L9 的 real-mask 指标 mean 有正向趋势，但 CI 跨 0，positive fraction 低于 confirmatory 门槛。

Partial metric artifact: `cross_model/stage4_qwen_all_layer_bounded_exhaustive_L5_L9_partial_metrics.csv`。

临时判断：L5-L9 比 L0-L4 有更明显的弱 source/control 边缘信号，但仍未闭合 `source/control + evidence-mask specificity + correct/wrong` 三个 gate。因此截至 L0-L9，仍不支持 sparse/topK PLT route，只能作为后续关注 L6/L7/L9 的 diagnostic signal。

L10-L16 middle-layer partial analysis:

- 每层 `1536` candidates，`768` main candidates，覆盖 `96` prompt-runs。
- 每层 zeroing raw rows: `30720`。
- 每层 grouped restore raw rows: `21504`。
- 近通过信号主要出现在 L10/L11/L13/L15，而不是 L14：
  - L10 `restore_source_minus_controls / answer_mask / topK 8-128`: mean `0.0143`，CI low `0.0010`，positive fraction `0.552`。
  - L11 `restore_source_minus_controls / union_mask / topK 8-128`: mean `0.0130`，CI low `0.0007`，positive fraction `0.438`。
  - L12 `zeroing_source_minus_controls`: mean `0.0054`，CI low `0.00014`，positive fraction `0.411`。
  - L13 `restore_source_minus_controls / union_mask / topK1`: mean `0.0178`，CI low `0.0051`，positive fraction `0.521`。
  - L15 `restore_source_minus_controls / answer_mask / topK4`: mean `0.0159`，CI low `0.0056`，positive fraction `0.521`。
- real-mask specificity 没有稳定通过：L10 answer_mask real-vs-shifted mean 最高约 `0.0173`，但 CI low 仍跨 0。
- L14 专项结果偏弱：
  - `zeroing_source_minus_controls` mean `0.00345`，CI low `< 0`。
  - restore/source-control、real-vs-shifted/shuffled 多数为 0 附近或负向。
  - 因此 L14 hidden route 成立不等于 L14 sparse PLT feature route 成立。

Partial metric artifact: `cross_model/stage4_qwen_all_layer_bounded_exhaustive_L10_L16_partial_metrics.csv`。
Partial decision artifact: `cross_model/stage4_qwen_all_layer_bounded_exhaustive_L10_L16_partial_decision.json`。

临时判断：中层确实比早层更有 source-control 信号，但仍未闭合 evidence-mask specificity。因此当前更支持 `Qwen hidden route exists, sparse/topK PLT route remains unestablished; route may live in PLT error/non-feature residual or require different operator/position grouping`。L10/L13/L15 是后续 near-pass diagnostic 候选，L14 不是当前 sparse PLT near-pass 层。

L17-L20 later-layer partial analysis:

- 每层 `1536` candidates，`768` main candidates，覆盖 `96` prompt-runs。
- 每层 zeroing raw rows: `30720`。
- 每层 grouped restore raw rows: `21504`。
- 只有 L17 `zeroing_source_minus_controls` 出现极弱 near-pass：mean `0.00418`，CI low `0.000027`，positive fraction `0.411`。
- grouped restore source/control 均未稳定通过；最好的 L17/L18/L19 restore source-control mean 约 `0.006-0.007`，CI low 均跨 0。
- real-mask specificity 仍未通过；最好的 L18 union topK1 real-vs-shifted mean `0.0168`，但 CI low `< 0`。

Partial metric artifact: `cross_model/stage4_qwen_all_layer_bounded_exhaustive_L17_L20_partial_metrics.csv`。
Partial decision artifact: `cross_model/stage4_qwen_all_layer_bounded_exhaustive_L17_L20_partial_decision.json`。

临时判断：L17-L20 没有超过 L10/L13/L15 的中层 near-pass。后层目前更支持 `no sparse/topK PLT route support under tested operators`，仍不影响 hidden-level route 结论。

# 预期与实际偏差

若全层 sparse/topK 失败，但 hidden residual 与 PLT reconstruction error 仍成立，结论是：Qwen 的 evidence-to-answer route 主要在 hidden/residual 或 PLT reconstruction error/non-feature residual 中，而不是当前 public PLT 的 sparse feature basis。

# 结论

待 primary full 后更新。任何负结论都只限定在当前 public Qwen-PLT、当前 paperpack、当前 bounded exhaustive 搜索和 inference-time intervention 设置下。
