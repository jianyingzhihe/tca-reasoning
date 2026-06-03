# Stage4-033 Qwen All-Layer Hidden Retest

## 目的

本实验用于严谨检查 `layer_miss`：此前 Qwen hidden/PLT 实验大量集中在 layer 26，Stage4-024 虽然扩到 `12,16,20,22,24,26,27`，但仍不是全层覆盖。

这里覆盖 Qwen2.5-VL language layers `0..27`，只判断 hidden residual 层面是否存在 evidence-region-sensitive answer action，不直接声称 PLT feature route 或 Gemma-style source tracing 成立。

## 输入

- `paperpack72_primary_prompt_runs.csv`
- `paperpack72_strict_sensitivity_prompt_runs.csv`
- base model: `Qwen/Qwen2.5-VL-7B-Instruct`
- mask conditions: `answer_mask`, `union_mask`, `shifted_mask`, `shuffled_mask`

## 输出

Artifact tag: `alllayers`

- `cross_model/stage4_qwen_decisive_route_hidden_primary_smoke_alllayers_raw.csv`
- `cross_model/stage4_qwen_decisive_route_hidden_primary_full_alllayers_raw.csv`
- `cross_model/stage4_qwen_decisive_route_hidden_strict_full_alllayers_raw.csv`
- `cross_model/stage4_qwen_decisive_route_summary.csv`
- `cross_model/stage4_qwen_decisive_route_specificity.csv`
- `cross_model/stage4_qwen_decisive_route_decision.json`

## 方法

层：

`0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27`

位置组：

- `visual_span`
- `answer_adjacent`
- `top_hidden_delta`
- `visual+answer`

干预：

- masked -> clean restore
- clean -> masked corrupt

判据：

- target logit/rank effect
- real mask > shifted/shuffled
- correct target > wrong target
- primary discovery 与 strict confirmation 分开；strict 只确认 primary 选出的 gate，不在 strict 上重新挑结论 gate。

## 结果

### Smoke

Smoke 通过，raw rows 为 `10752`，证明 all-layer hidden runner 可执行。Smoke 只用于工程验收，不作为机制结论。

### Primary Full

Primary full 完成：

- usable prompt-runs: `96`
- raw rows: `172032`
- primary-selected gate:
  - layer: `14`
  - direction: `restore`
  - position group: `top_hidden_delta`
  - mask condition: `answer_mask`

Primary 指标：

- target logit effect mean: `1.7544`, CI low: `1.2872`
- real minus shifted mean: `1.1148`, CI low: `0.6851`
- real minus shuffled mean: `1.2932`, CI low: `0.8464`
- correct minus wrong mean: `0.3148`, CI low: `0.0424`
- target rank effect mean: `803.74`, CI low: `171.58`

### Strict Full

Strict full 已完成：

- usable prompt-runs: `101`
- raw rows: `180992`
- run status: `ok`
- final checkpoint: `2026-05-26 12:45:11`

Strict confirmation 对 primary-selected gate 进行同 gate 验证：

- layer: `14`
- direction: `restore`
- position group: `top_hidden_delta`
- mask condition: `answer_mask`

Strict 指标：

- target logit effect mean: `1.7242`, CI low: `1.2684`
- real minus shifted mean: `1.1144`, CI low: `0.7042`
- real minus shuffled mean: `1.2183`, CI low: `0.8010`
- correct minus wrong mean: `0.3311`, CI low: `0.0578`
- target rank effect mean: `766.53`, CI low: `151.23`

Decision status:

`qwen_hidden_route_supported`

## 预期与实际偏差

之前 layer 26-heavy 的测试确实存在 layer miss 风险。全层 sweep 显示 strongest confirmed hidden gate 位于 layer 14，而不是 layer 26。

另一个分析口径问题也已修复：strict set 不能重新挑一个分数最高的新 gate 来替代 primary discovery gate。更新后的 analyzer 明确使用 primary-selected gate 做 strict confirmation；strict 自己的 best gate 只保留为 diagnostic。

## 结论

Qwen2.5-VL 在 hidden residual 层面存在稳定的 evidence-region-sensitive answer-support action：primary 和 strict 都确认 layer 14 / restore / top_hidden_delta / answer_mask gate，并且 target effect、real-vs-shifted、real-vs-shuffled、correct-vs-wrong、rank effect 都为正且 CI 不跨 0。

这支持 `Qwen hidden-level evidence route`，也说明此前 layer 26 聚焦实验可能漏掉关键层。

但本实验仍不能推出 PLT feature/source route 成立，也不能推出 Qwen fully replicates Gemma-style source tracing。下一步应围绕 layer 14 及邻近层做 PLT dense sweep、multi-layer route patch 和 Adapter V4 route probe。
