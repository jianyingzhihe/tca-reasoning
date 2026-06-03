# 028 Qwen2.5-VL-PLT Primary72 Full Results

## 目的

把 Qwen2.5-VL-PLT 从 3-case smoke 推进到 `paperpack72_primary` full run，检查新标注包上是否仍有 feature/source-control 层面的 evidence-region-sensitive 信号。

## 输入

```text
paperpack: paperpack72_primary
samples: 72
prompt-runs: 144
base model: Qwen/Qwen2.5-VL-7B-Instruct
transcoder: KokosDev/qwen2p5vl-7b-plt
layer: 26
primary position group: top_hidden_delta_plus_answer_adjacent
feature top-k: 8
controls: activation_matched, drop_matched, attribution_matched_mask_insensitive, random_active
```

## 输出

```text
stage3_qwen2p5vl_plt_feature_union_primary_full.csv/json
stage3_qwen2p5vl_plt_source_control_primary_full.csv/json
stage3_qwen2p5vl_plt_primary_full_*summary.csv
stage3_qwen2p5vl_plt_primary_full_*specificity.csv
stage3_qwen2p5vl_plt_primary_full_case_table.csv
stage3_qwen2p5vl_plt_primary_full_decision.json
```

## 方法

feature bridge 选择既被 evidence mask 削弱、又对 target answer 有正向 attribution 的 feature group，并与 matched controls 比较 restore/corrupt 的 target logit/rank effect。

source-control probe 以 approximate source/control pair 为单位，比较 source zeroing / source restore 与 matched-control zeroing / restore；同时检查 `correct > wrong` 和 legacy spatial control。注意：该 full runner 的旧 `mask_shuffled` 字段实际是 shifted control，true shifted/shuffled controls 已在 `030` 中补跑。

## 结果

自动 decision：

```text
qwen_plt_primary_support
```

这个状态只表示 Qwen2.5-VL-PLT 在 primary72 上通过 approximate feature/source-control 支持标准，不表示 Gemma-style source tracing 已复现。

核心数值：

| endpoint | result |
|---|---:|
| feature rows | 4320 |
| feature prompt-runs | 144 |
| source-control rows | 1452 |
| source-control prompt-runs | 128 |
| usable source-control pairs | 242 |
| feature primary positive comparisons | 8 / 8 |
| source-control real-mask positive comparisons | 4 / 4 |
| legacy real-vs-shifted positive comparisons | 4 / 8 |

最稳的正结果：

```text
feature evidence > activation-matched:
  restore mean +0.1297, CI [+0.0831, +0.1785]
  corrupt mean +0.1311, CI [+0.0790, +0.1845]

source zeroing > matched-control zeroing:
  answer_mask mean +0.1857, CI [+0.1588, +0.2152]
  union_mask mean +0.2048, CI [+0.1616, +0.2613]

legacy source real restore > shifted-control restore:
  answer_mask mean +0.0516, CI [+0.0248, +0.0809]
  union_mask mean +0.0501, CI [+0.0246, +0.0775]
```

较弱或未闭合的部分：

```text
source restore > matched-control restore:
  answer_mask positive and CI > 0
  union_mask positive but CI slightly crosses 0

feature corrupt against drop/random/mask-insensitive controls:
  mean positive but several CI cross 0

shifted/shuffled controls:
  completed later in 030; strongest support is answer-mask real > shifted

decoded / sequence bridge:
  not part of this run
```

## 预期与实际偏差

预期是 full run 能明显强于 smoke，并给出更稳的 heldout 证据。实际结果支持这一点，尤其 source zeroing 和 activation-matched feature specificity 很稳。

预期也希望 restore 同样强。实际 restore 有正信号但弱于 zeroing，说明当前 intervention 更像“移除 source 会损伤答案支持”，而不是“补回 source 总能恢复答案”。这和之前 decoded bridge 不稳定是一致的。

## 结论

这一步显著加强了 Qwen2.5-VL-PLT 的跨模型 PLT 主线证据：

```text
Qwen2.5-VL-PLT has paperpack72-primary approximate feature/source-control support for evidence-region-sensitive answer mechanisms.
```

但 PLT-only paper-grade 结论还不能落定，因为还缺：

```text
strict72_sensitivity
Gemma3-PLT paperpack rerun/calibration
Gemma3-PLT paperpack rerun/calibration
sequence/decoded behavior bridge if we want behavior-level claim
```
