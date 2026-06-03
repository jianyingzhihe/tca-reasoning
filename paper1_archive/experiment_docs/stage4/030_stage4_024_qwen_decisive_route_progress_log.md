# Stage4-024 Qwen Decisive Route Progress Log

## 目的

记录 Stage4-024 当前推进状态，避免把长跑中的中间结果误写成最终 verdict。本轮目标是系统排除 `layer_miss`、`distributed_route`、`plt_localization_failure` 和 `route_absent_under_test`，而不是继续只追一个正例。

## 已修复的实验问题

- `layer_out_of_range`：原计划包含 layer `28,30`，但当前 `Qwen2.5-VL-7B-Instruct` language model 可 hook 层为 `0..27`。已将默认 sweep 改为 `12,16,20,22,24,26,27`，并在脚本中自动记录越界层。
- `hidden_state_off_by_one`：HF `hidden_states` 包含 embedding 输出，因此 patch `language_model.layers[L]` 的输出时应使用 `hidden_states[L+1]`。已修复，并在 raw CSV 中新增 `hidden_state_index` 便于审计。
- `resume/checkpoint`：hidden lattice 已新增 `--resume` 与 `--checkpoint-every`，strict run 已用 checkpoint 成功执行。
- `premature_negative_status`：hidden 阶段失败不再直接写 `qwen_gemma_style_sparse_route_not_supported`，最终负结论必须等 PLT layer sweep、multi-layer patch 和 Adapter V4 都完成。

## Hidden Causal Lattice 当前结果

- Primary full：`96` usable prompt-runs，`43008` raw rows。
- Strict full：`101` usable prompt-runs，`45248` raw rows。
- Primary/strict 最强形状一致：layer `26`，`restore`，`visual+answer`，`answer_mask`。
- 通过/接近通过的证据：
  - target logit effect 强，primary mean 约 `4.14`，strict mean 约 `4.24`。
  - real mask 明显强于 shifted/shuffled。
  - target rank effect 明显为正，primary mean 约 `904.6`，strict mean 约 `863.4`。
- 未通过 gate：
  - `correct_minus_wrong` CI 跨 0，primary mean 约 `-0.04`，strict mean 约 `0.19`。

## 当前解释

Hidden 层面可以写：Qwen 在 paperpack 上存在强的 evidence-mask-sensitive hidden causal action，且 primary/strict 形状稳定。

不能写：Qwen 已经有 target-specific evidence-to-answer route。原因是 wrong-target specificity 没闭合。

也不能写：Qwen 没有 Gemma-style sparse PLT route。原因是 PLT layer sweep、multi-layer patch 和 Adapter V4 还没有完成。

## PLT Layer Sweep 当前状态

- Smoke 已完成：layers `26,12,24,22` 的 candidate discovery、source/control zeroing、group restore 都可产出 artifact。
- Primary full 已完成：layers `26,12,24,22`，每层均有 `384` candidate rows、`192` main rows、`96` prompt-runs with candidates。
- 每层 zeroing 与 group restore 均完成；primary full artifacts 已拉回并进入 `stage4_qwen_decisive_route_summary.csv` / `specificity.csv` / `decision.json`。

### PLT primary full 结果

- Activation-drop gate 很强：例如 L26 `union_mask top4` 的 `activation_drop_real_minus_shifted` mean 约 `101.6`，CI low 约 `93.3`，positive fraction `1.0`；`activation_drop_real_minus_shuffled` 同样强。
- 这说明 answer/union mask 确实强烈影响了 Qwen-PLT features，且不是 shifted/shuffled control 都能解释。
- 但 causal intervention gate 没闭合：
  - L26 clean zeroing `source_minus_controls` mean 约 `0.003`，CI 跨 0，positive fraction 约 `0.12`。
  - L26 clean zeroing `correct_minus_wrong` mean 约 `-0.002`，CI 跨 0。
  - L26 best restore `answer_mask top1` 的 `source_minus_controls` CI low 约 `0.001`，但 positive fraction 只有约 `0.16`，且 `real_minus_shifted` / `real_minus_shuffled` / `correct_minus_wrong` / rank gate 均未通过。
  - L12/L24/L22 也没有层通过 clean zeroing + restore + specificity gate。

## 当前综合分析

Stage4-024 到目前为止更支持 `hidden-level evidence action exists, but current PLT feature localization / intervention route failed`，而不是 `route_absent_under_test`。

原因是：
- Hidden patch 的 target logit/rank 与 real-vs-shifted/shuffled 在 primary 与 strict 都稳定强。
- PLT evidence-sensitive features 的 activation-drop real-vs-control 极强。
- 但把这些 PLT feature 当作 source/control 节点去 zero/restore，答案 logit/rank 几乎不动，wrong-target specificity 也不成立。

因此当前不能写：
- `Qwen fully replicates Gemma-style source tracing`
- `Qwen has target-specific PLT route`
- `Qwen has no cross-modal mechanism`

当前可以写：
- `Qwen shows robust hidden-level evidence-mask-sensitive causal action.`
- `Qwen PLT features are strongly evidence-mask-sensitive, but current single-layer PLT feature interventions do not establish answer-mediating source-control routes.`
- `This points to PLT localization/intervention mismatch or distributed route as the next hypotheses, not a broad absence of mechanism.`

## 下一步

1. 不建议立刻跑 PLT strict confirmation，因为 primary full 的 PLT intervention gate 明确没过；strict 只会重复一个未通过 gate。
2. 下一步应进入 multi-layer route patch：用 hidden 强层 `26,24,22,12` 组合，同时 patch/restore 多层 hidden 或多层 PLT groups。
3. 若 multi-layer hidden 通过但 multi-layer PLT 仍失败，结论升级为 `hidden route exists, current PLT localization failed`。
4. 若 multi-layer PLT 通过，结论升级为 `distributed / multi-layer Qwen-native route support`。
5. Adapter V4 应在 multi-layer 结果后跑，用 `hidden_mediation * evidence_sensitivity * target_attribution * zeroing_damage` 生成 Qwen-native route map。
6. 只有 hidden、single-layer PLT、multi-layer patch、Adapter V4 全部失败后，才能写限定负结论：`Gemma-style sparse PLT route not supported under broad Qwen-native tests`。
