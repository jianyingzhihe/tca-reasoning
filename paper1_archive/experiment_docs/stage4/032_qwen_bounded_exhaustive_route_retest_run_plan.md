# Stage4-032 Qwen Bounded Exhaustive Route Retest Run Plan

## 目的

把 Qwen2.5-VL-PLT 的下一轮实验从“继续调参找正例”改成“有边界的穷举式排查”。目标不是数学上遍历全模型所有可能节点，而是在公开 PLT asset 和 inference-time intervention 可承受范围内，明确覆盖：

- 所有可用层；
- 多种位置定义；
- 多种 feature selection score；
- single-layer 与 multi-layer route；
- hidden route 与 PLT route 的对应关系。

最终要能区分：

- `layer_miss`
- `position_miss`
- `feature_selection_miss`
- `distributed_route`
- `plt_localization_failure`
- `route_absent_under_bounded_tests`

## 前置审计结论

详见 `031_qwen_prior_experiment_critical_audit.md`。

关键点：

- 之前没有遍历所有节点，不能保证找到所有 correct 节点。
- 之前 Qwen source-tracing adapter 主要集中在 layer 26。
- Stage4-024 已扩展到 hidden `12,16,20,22,24,26,27` 和 PLT `26,12,24,22`，但仍不是全层。
- Gemma source tracing 是多层 graph/route，不应要求 Qwen 与 Gemma layer 对齐；Qwen 可以有不同层、不同地图、不同 route topology。

## 实验 1：All-Layer Hidden Coarse Sweep

### 输入

- `paperpack72_primary_prompt_runs.csv`
- `paperpack72_strict_sensitivity_prompt_runs.csv`
- Qwen2.5-VL base

### 层

- 全部 language layers：`0..27`

### 位置组

- `visual_span`
- `answer_adjacent`
- `visual+answer`
- `top_hidden_delta`

### 条件

- `answer_mask`
- `union_mask`
- `shifted_mask`
- `shuffled_mask`

### 干预

- masked -> clean restore
- clean -> masked corrupt

### 输出

- all-layer hidden raw CSV
- per-layer/per-position summary
- gate matrix

### 通过标准

一层或多层满足：

- target effect CI 不跨 0；
- real > shifted/shuffled；
- rank effect 方向成立；
- correct > wrong 若不成立，需要记录为 `non_target_specific_hidden_action`，不能升级为 route。

## 实验 2：All-PLT-Layer Evidence-First Discovery

### 输入

- all-layer hidden sweep 的 strong/near-strong 层；
- 以及全 PLT 层 `0..26` coarse coverage。

### 层策略

两阶段：

1. coarse：所有 PLT layers `0..26`，每层 `top_per_prompt_run=2`，只跑 discovery + activation-drop summary。
2. dense：hidden strong / activation-drop strong 的前 `6-8` 层，跑 full zeroing + group restore。

### 位置组

- `visual_only`
- `answer_adjacent_only`
- `visual_answer`

### feature scores

至少三套并列：

- `evidence_sensitivity * target_attribution`
- `evidence_sensitivity * target_attribution * hidden_mediation`
- `zeroing_screened_damage` 只作为 confirmation，不参与 strict discovery。

### 输出

- coverage manifest：每层候选数、active features 估计、topK 截断比例。
- discovery CSV。
- activation-drop specificity CSV。

### 通过标准

如果 discovery 有强 activation-drop 但干预失败，支持 `PLT localization/intervention mismatch`，不支持 answer route。

## 实验 3：Dense Multi-Layer Route Patch

### 输入

- 实验 1 的 hidden strong layers；
- 实验 2 的 PLT evidence-sensitive layers。

### layer groups

预注册：

- `12+22+24+26`
- `20+22+24+26`
- `22+24+26`
- `12+24+26`
- `all_hidden_strong_top4`

### 干预

- multi-layer hidden restore/corrupt；
- multi-layer PLT group restore/corrupt；
- matched controls；
- shifted/shuffled；
- wrong target。

### 通过标准

- single-layer 失败但 multi-layer 成立：`distributed_route`
- hidden multi-layer 成立但 PLT multi-layer 失败：`hidden_route_exists_but_plt_localization_failed`
- multi-layer 也失败：进入 Adapter V4 / bounded negative。

## 实验 4：Adapter V4 Bounded Route Probe

### score

`route_score = hidden_mediation * evidence_sensitivity * target_attribution * zeroing_damage`

### discovery/confirmation 隔离

- primary 只用于发现；
- strict 只用于确认；
- strict 不能参与节点选择。

### 输出

- Qwen-native route map；
- node table；
- control table；
- intervention table；
- decision JSON。

### 通过标准

若 primary/strict 都满足 source/control、real-vs-shuffled、correct-vs-wrong、rank/sequence bridge，则写：

`Qwen-native source-tracing-like route support`

仍不写：

`Qwen fully replicates Gemma ReplacementModel source tracing`

## 实验 5：Bounded Negative Verdict

只有以下全部失败时，才写限定负结论：

- all-layer hidden target-specific route 不成立；
- all-PLT-layer single-layer route 不成立；
- multi-layer route patch 不成立；
- Adapter V4 route 不成立。

允许写：

`Gemma-style sparse PLT route is not supported under broad Qwen-native bounded tests.`

不允许写：

`Qwen has no cross-modal mechanism.`

## 本地检查

- `py_compile` 所有新增/修改脚本。
- manifest 检查 primary/strict prompt-runs、mask、image、answer。
- coverage manifest 必须写明没有真正 all-node exhaustive。

## 远端执行顺序

1. all-layer hidden coarse smoke：6 prompt-runs。
2. all-layer hidden primary full。
3. all-layer hidden strict confirmation。
4. all-PLT-layer discovery smoke。
5. all-PLT-layer discovery primary coarse。
6. dense PLT full on selected layers。
7. multi-layer route patch。
8. Adapter V4 route probe。

## 当前推荐

先运行实验 1 的 all-layer hidden coarse/full。原因：hidden 是更接近因果路径的上界。如果 all-layer hidden 都没有 target-specific gate，PLT route 成立概率很低；如果 hidden 某些低层/高层 target-specific gate 成立，就能直接指导 PLT dense sweep。

