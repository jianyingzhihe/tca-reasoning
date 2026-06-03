# Stage4-052 Qwen Evidence-Specific Node Targeted Validation Run Plan

## 目的

当前 all-layer 粗扫与 L10 dense scan 已经显示：Qwen2.5-VL-PLT 在 L10-L17 存在大量遮真实 evidence region 后 activation 明显下降、而 shifted/shuffled control 不下降的候选节点。下一步不再继续全量 dense 平均，而是验证这些强 evidence-specific nodes 是否稳定、是否因果支撑答案、是否能 strict 复现。

## 输入

- all-layer bounded exhaustive primary candidates: `stage4_qwen_all_layer_bounded_exhaustive_primary_full_L10..L17_candidates.csv`
- middle dense L10 candidates: `stage4_qwen_middle_dense_primary_full_L10_*_candidates.csv`
- paperpack72 primary / strict prompt-runs and masks

## 方法

- 只使用 Qwen 自己生成的候选，不读取 Gemma node id。
- 主筛选规则固定为 `real_drop_best >= 20`、`evidence_specificity >= 20`、`target_contribution > 0`、`correct_minus_wrong_contribution > 0`、`clean_target_rank <= 10`。
- Manifest 保留所有通过阈值的 pool candidates；`main` 只是在第一轮因果验证中使用的分层子集。
- `main` 子集每个 `sample_id + prompt_name + layer` 最多保留 top2；每层默认最多 20 个候选，硬上限 24，避免少数样本支配第一轮结论。
- 若 `main` 通过或接近通过，再用 `--selection all` 对完整 pool 做 exhaustive targeted validation。
- primary 用于发现并冻结候选；strict 只复验同一套 layer/position/feature，不重新调参。

## 预期输出

- `stage4_qwen_evidence_specific_nodes_primary_manifest.csv`
- `stage4_qwen_evidence_specific_nodes_strict_manifest.csv`
- per-layer targeted validation raw/run artifacts
- summary、specificity、case table、decision JSON

## 结论边界

成功时写 Qwen-native evidence-specific causal support；失败时只能写当前 public Qwen-PLT 下 targeted evidence-specific candidates 未闭合成 causal route，不否定 hidden-level mechanism。
