# Stage4-037 Qwen Adapter V4 Results

## 目的

记录 Adapter V4 对 Qwen layer 14 route 的自动选择与因果验证结果。

## 输入

- L14 source-tracing intervention rows
- L14 PLT evidence-first candidates
- Adapter V4 manifest

## 输出

- `stage4_qwen_adapter_v4_*_manifest.csv/json`
- `stage4_qwen_adapter_v4_*_zeroing_raw.csv/json`
- `stage4_qwen_adapter_v4_*_group_raw.csv/json`
- `stage4_qwen_adapter_v4_summary.csv`
- `stage4_qwen_adapter_v4_specificity.csv`
- `stage4_qwen_adapter_v4_decision.json`

## 方法

主 gates：

- source clean zeroing > matched/random controls
- correct target > wrong target
- real mask restore > shifted/shuffled restore
- rank or sequence bridge direction positive

decoded greedy 只作为 secondary，不作为主判据。

## 结果

### Primary Smoke

Adapter V4 smoke 已完成。

输入：

- source-tracing prefix: `stage4_qwen_source_tracing_primary_smoke_adapter_v4_L14`
- PLT evidence prefix: `stage4_qwen_decisive_route_plt_primary_smoke_L14`
- V4 prefix: `stage4_qwen_adapter_v4_primary_smoke_L14_layer14`

候选：

- V4 rows: `5`
- main rows: `4`
- prompt-runs: `1`
- exact_pos_feature rows: `2`
- same_feature rows: `3`

初步 smoke 指标：

- clean source-minus-controls mean: `0.0729`, n=`4`
- clean correct-minus-wrong mean: `0.1563`, n=`4`
- restore gate has positive smoke direction, but n=`1`

Decision:

`smoke_completed_not_decisive`

## 预期与实际偏差

如果 Adapter V4 exact candidates 少于阈值，写 `plt_candidate_sparse` 或 `qwen_hidden_route_supported_plt_unresolved`。

如果 primary 成立但 strict 不成立，只写 exploratory/partial，不升级 mainline claim。

## 结论

Smoke 证明 Adapter V4 manifest build、zeroing controls 和 grouped restore 都可运行。由于 smoke 候选只覆盖 1 个 prompt-run，不能作机制结论；full primary/strict 仍需继续。
