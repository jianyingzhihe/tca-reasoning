# Stage4-039 Qwen V4 Failure Decomposition

## 目的

解释 Adapter V4 为什么能找到 answer-support PLT nodes，但没有闭合成 evidence-linked source route。

## 输入

- `stage4_qwen_adapter_v4_primary_full_L14_layer14_manifest.csv`
- `stage4_qwen_adapter_v4_primary_full_L14_layer14_zeroing_raw.csv`
- `stage4_qwen_adapter_v4_primary_full_L14_layer14_group_raw.csv`
- strict 对应 artifacts

## 输出

- `stage4_qwen_native_route_v4_failure_decomposition_summary.csv`
- `stage4_qwen_native_route_v4_failure_decomposition_decision.json`

## 方法

拆分四类证据：

- activation specificity：real mask 是否比 shifted/shuffled 更改变候选 feature activation。
- clean zeroing specificity：source zeroing 是否强于 controls。
- restore failure：feature delta restore 是否恢复 target logit/rank。
- match quality：exact-pos-feature 与 same-feature fallback 的占比和效果差异。

## 结果

已完成本地 decomposition。

关键结果：

- failure type: `operator_or_feature_not_sufficient`
- activation specificity 成立：
  - primary union real > shifted: mean `5.7174`, CI low `2.1953`
  - strict union real > shifted: mean `4.8112`, CI low `1.5212`
- clean zeroing source > controls 成立：
  - primary mean `0.0574`, CI low `0.0473`
  - strict mean `0.0565`, CI low `0.0471`
- restore specificity 不成立：
  - primary answer real > shifted: mean `0.0048`, CI low `-0.0285`
  - strict answer real > shifted: mean `0.0065`, CI low `-0.0234`
- exact-position candidates 偏少：
  - primary exact main rows: `14`
  - strict exact main rows: `10`

输出：

- `cross_model/stage4_qwen_native_route_v4_failure_decomposition_summary.csv`
- `cross_model/stage4_qwen_native_route_v4_failure_decomposition_metrics.csv`
- `cross_model/stage4_qwen_native_route_v4_failure_decomposition_decision.json`

## 预期与实际偏差

如果 activation specificity 成立但 restore 失败，优先解释为 `operator_failure` 或 `feature_not_sufficient`，不是“Qwen 没有 route”。

## 结论

Adapter V4 失败不是因为完全没有 evidence-sensitive feature，也不是因为 clean zeroing 没有 answer-support signal；更像当前 feature restore/operator 或 sparse feature sufficiency 不够。Exact-position route 目前候选稀疏，不能单独作为强负结论。
