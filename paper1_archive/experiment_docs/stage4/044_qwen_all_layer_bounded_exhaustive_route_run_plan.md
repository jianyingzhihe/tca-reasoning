# 目的

本实验用于尽可能排除 Qwen2.5-VL-PLT 中“我们漏了正确层或正确节点”的问题。它不是字面逐个 feature/position 做完整 forward 干预，而是 bounded exhaustive：先对 `0..27` 全语言层的 active PLT features × allowed positions 做向量化候选筛选，再只对每层 top 候选做真实 causal intervention。

# 输入

- `paperpack72_primary_prompt_runs.csv`，主实验 `144` prompt-runs。
- `paperpack72_strict_sensitivity_prompt_runs.csv`，只在 primary 出现 near-pass 后做 confirmation。
- Qwen2.5-VL-7B-Instruct 与 `KokosDev/qwen2p5vl-7b-plt`。
- answer/union/shifted/shuffled masks。

# 输出

- `cross_model/stage4_qwen_all_layer_bounded_exhaustive_*_candidates.csv/json`
- `cross_model/stage4_qwen_all_layer_bounded_exhaustive_*_zeroing_raw.csv/json`
- `cross_model/stage4_qwen_all_layer_bounded_exhaustive_*_group_raw.csv/json`
- `cross_model/stage4_qwen_all_layer_bounded_exhaustive_summary.csv`
- `cross_model/stage4_qwen_all_layer_bounded_exhaustive_specificity.csv`
- `cross_model/stage4_qwen_all_layer_bounded_exhaustive_decision.json`

# 方法

先跑 smoke：层 `0,7,14,21,27`，每层 `6` prompt-runs，检查候选筛选、zeroing、grouped restore、fetch/analyze 全链路。

smoke 通过后跑 primary full：层 `0..27`，全部 `144` prompt-runs。每层向量化候选筛选默认使用 broad pre-candidate pool，再保留每 prompt-run top `16`，main intervention top `8`。真实干预包括 clean zeroing、source restore/grouped restore，并对 activation/drop/mask-insensitive/random-active controls、shifted/shuffled masks、wrong target 做对照。

如果 primary 出现 near-pass，再用 strict pack 复验同一层/规则，不用 strict 重新挑超参。

# 结果

Smoke 已完成：`primary / layers 0,7,14,21,27 / 6 prompt-runs` 全链路通过并已拉回 artifacts。候选数：L0/L7/L14/L21 各 `96`，L27 为 `0`。smoke 没有达到 sparse/topK route 支持门槛，当前只作为链路检查和初步诊断。

Primary full 已后台启动：`primary / layers 0..27 / 144 prompt-runs`，远端 detached PID `788845`，log 位于 `/root/autodl-tmp/tca-reasoning/stage4_qwen_all_layer_bounded_exhaustive/logs/plt_layer_sweep_full_primary_L0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_20260526_231424.log`。

# 预期与实际偏差

预期可能有三种结果：找到某层 sparse PLT route；只找到 distributed/topK route；或者全层 sparse/topK 仍失败，同时 Stage4-038 的 hidden residual / PLT reconstruction error 结论保留。

本实验不能声称“遍历了所有可能节点的全部 forward 干预”。它只能声称在当前 public Qwen-PLT、allowed positions、active features、top candidate budget 下做了 bounded exhaustive search。

# 结论

待 `047_qwen_all_layer_route_verdict.md` 汇总。若全层仍找不到 sparse/topK route，负结论只限定为当前公开 Qwen-PLT 和本搜索范围，不否定 Qwen 有 evidence-to-answer mechanism。
