# Stage4-042 Qwen Hidden-to-PLT Mediation

## 目的

判断已经成立的 Qwen hidden layer 14 route 是否能被 PLT feature basis 重构。

## 输入

- paperpack72 primary/strict prompt-runs
- answer/union/shifted/shuffled masks
- Qwen2.5-VL-PLT

## 输出

- `stage4_qwen_native_route_hidden_to_plt_*_raw.csv`
- `stage4_qwen_native_route_hidden_to_plt_*_run.json`
- `stage4_qwen_native_route_hidden_to_plt_summary.csv`
- `stage4_qwen_native_route_hidden_to_plt_decision.json`

## 方法

同一 layer 14、top_hidden_delta positions 下比较：

- full hidden residual restore
- PLT topK feature reconstruction restore: K=`8,16,32,64,128`
- PLT reconstruction error restore

主判据：

- target logit/rank restore
- real mask > shifted/shuffled
- correct target > wrong target

## 结果

Primary smoke 已完成。

Smoke 输入：

- pack: `primary`
- prompt-runs: `6`
- layer: `14`
- position group: `top_hidden_delta`
- topK: `8,16,32,64,128`

Smoke 输出：

- `cross_model/stage4_qwen_native_route_hidden_to_plt_primary_smoke_stage4_038_raw.csv`
- `cross_model/stage4_qwen_native_route_hidden_to_plt_primary_smoke_stage4_038_run.json`
- `cross_model/stage4_qwen_native_route_hidden_to_plt_metrics.csv`
- `cross_model/stage4_qwen_native_route_hidden_to_plt_decision.json`

Smoke decision:

`qwen_plt_mediated_distributed_route_supported_smoke`

关键 smoke 信号：

- `plt_topk_reconstruction` 在 `answer_mask/top16` 上 target effect 为正：mean `0.0729`, CI low `0.03125`
- `plt_topk_reconstruction` 在 `answer_mask/top16` 上 real > shifted/shuffled 方向为正。
- `plt_reconstruction_error` 也出现 answer_mask correct-minus-wrong 正向 smoke 信号。

注意：smoke 只证明 runner 和算子方向值得 full，不作为论文结论。

Primary full 与 strict full 已完成。

覆盖：

- primary: `144/144` prompt-runs, raw rows `12672`
- strict: `144/144` prompt-runs, raw rows `12672`

Primary full decision：

`qwen_route_may_live_in_plt_error`

Strict full decision：

`qwen_route_may_live_in_plt_error`

关键 strict 指标：

- `hidden_residual / answer_mask / target_effect`: mean `0.8205`, CI low `0.5860`
- `hidden_residual / answer_mask / real > shifted`: mean `0.5204`, CI low `0.3007`
- `hidden_residual / answer_mask / real > shuffled`: mean `0.4943`, CI low `0.2643`
- `plt_reconstruction_error / top8 / answer_mask / target_effect`: mean `0.8105`, CI low `0.5794`
- `plt_reconstruction_error / top8 / answer_mask / real > shifted`: mean `0.5094`, CI low `0.2898`
- `plt_reconstruction_error / top8 / answer_mask / real > shuffled`: mean `0.4856`, CI low `0.2603`

Interpretation:

`plt_reconstruction_error` almost reproduces the full hidden residual effect, while sparse `plt_topk_reconstruction` does not pass the analyzer gate.

## 预期与实际偏差

如果 hidden residual restore 成立但 PLT topK 不成立，写 `hidden route exists but tested PLT feature basis did not mediate it`。

## 结论

Qwen 的 evidence-to-answer route 在 hidden residual 层成立，并且该因果效果在 primary/strict 上主要保留于 PLT reconstruction error / non-feature residual，而不是当前 tested sparse topK PLT feature basis。该结论支持 Qwen-native route heterogeneity，不支持 Qwen fully replicates Gemma-style sparse source tracing。
