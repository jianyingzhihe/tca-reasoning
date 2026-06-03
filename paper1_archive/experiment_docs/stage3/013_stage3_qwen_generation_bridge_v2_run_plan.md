# 013 Stage3 Qwen Generation Bridge V2 Run Plan

## 目的

Stage3-13 的目标是加固 Qwen2.5-VL 的 behavior bridge。此前 Stage3-011 已经显示 Qwen 的 feature/source-control 差异可以影响目标答案首 token 的 logit/rank；Stage3-012 的短 greedy decoded smoke 则显示 `source_restore` 还不能稳定把最终 decoded answer 拉回 clean answer。

因此本轮不再只看单 feature 和 greedy answer，而是加入更细的行为指标：

```text
first-token/rank bridge:
  目标答案第一个 token 的 logit/rank 发生方向性变化。

sequence-level bridge:
  目标答案完整 token 序列的 logprob 被 source restore/corrupt 按预期改变，
  且 source 强于 matched controls。

decoded generation bridge:
  短 greedy generation 的最终答案被 source restore 拉回 clean/target，
  且强于 controls。
```

本轮成功前不写 `generation-level causal bridge`。失败也不写 Qwen 没有相关机制，只说明当前多 feature 干预仍不足以稳定改变 decoded answer。

## 输入

固定模型与资产：

| 名称 | repo |
|---|---|
| Qwen2.5-VL base | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Qwen2.5-VL-PLT | `KokosDev/qwen2p5vl-7b-plt` |
| Qwen2.5-VL-CLT | `KokosDev/qwen2p5vl-7b-clt` |

manifest 从 Stage3 已有 Qwen behavior/source-control 结果中筛选，规则固定为：

```text
clean_target_rank <= 5
mask_target_rank > clean_target_rank
clean decoded answer 命中或接近 target answer
mask decoded answer 不同于 clean，优先不命中 target
source-control restore 或 zeroing 至少一项为正
每个资产最多 12 个 prompt-runs；不足则全选并记录不足原因
```

实际严格筛选得到 7 个 prompt-runs：

| 资产 | rows |
|---|---:|
| Qwen2.5-VL-CLT | 4 |
| Qwen2.5-VL-PLT | 3 |

数量不足 12 的原因不是运行失败，而是 behavior-aware 条件较严格：必须同时满足 clean 命中、mask 破坏答案、rank 变差、且已有 source/control 正信号。

## 输出

本轮固定产物：

```text
doc/experiments/stage3/013_stage3_qwen_generation_bridge_v2_run_plan.md
doc/experiments/stage3/014_stage3_qwen_multifeature_sequence_bridge.md
doc/experiments/stage3/015_stage3_qwen_generation_bridge_v2_verdict.md
doc/experiments/stage3/cross_model/stage3_qwen_generation_bridge_v2_manifest.csv
doc/experiments/stage3/cross_model/stage3_qwen_generation_bridge_v2_manifest_report.json
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_plt_generation_bridge_v2.csv
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_clt_generation_bridge_v2.csv
doc/experiments/stage3/cross_model/stage3_qwen_generation_bridge_v2_summary.csv
doc/experiments/stage3/cross_model/stage3_qwen_generation_bridge_v2_case_table.csv
doc/experiments/stage3/cross_model/stage3_qwen_generation_bridge_v2_decision.json
```

## 方法

每个 selected prompt-run 同时跑两类方向：

```text
masked -> clean restore:
  在 masked run 中补回 evidence-attribution feature group 的 clean-mask contribution。

clean -> masked corruption:
  在 clean run 中移除 evidence-attribution feature group 的 contribution。
```

feature group 从单 feature 扩展为多 feature：

```text
top1, top4, top8, top16, top32 evidence_attribution features
```

matched controls 固定为：

```text
activation_matched_topk
drop_matched_topk
attribution_matched_mask_insensitive_topk
random_active_topk
```

主指标：

```text
answer_sequence_logprob:
  目标答案 token 序列的 log probability。越高表示模型越愿意生成目标答案序列。

first_token_logit/rank:
  目标答案第一个 token 的 logit/rank。

short_greedy_decoded_answer:
  最终短 greedy generation answer。它是最强行为层证据，但也最难被局部 feature patch 改动。
```

## 成功判据

`supported_sequence_bridge` 要求：

```text
source restore 的 target answer sequence logprob 恢复强于 matched controls；
source corrupt 的 target answer sequence logprob 损伤也强于 controls；
restore 与 corrupt 两个方向的 bootstrap CI 主项均不跨 0。
```

`partial_sequence_bridge` 要求：

```text
restore 与 corrupt 两个方向的 source-control mean 均为正，
但 CI 仍跨 0 或样本数太小。
```

`first_token_only` 表示：

```text
sequence-level 双向桥接没有成立；
但 first-token/rank 仍有正向 source-control 信号。
```

`decoded generation bridge` 只有在 source restore 稳定把 decoded answer 拉回 clean/target 且强于 controls 时才成立。

## 预期与实际偏差

预期 Qwen2.5-VL-CLT 可能最强，因为它在 Stage3-011 首 token/rank 桥接中最稳；Qwen2.5-VL-PLT 可能较弱但更接近 Gemma PLT 主线资产类型。

实际 manifest 只有 7 个严格可用 prompt-runs，说明 generation-level 候选比 first-token/rank 候选少得多。这会让 sequence score 统计更保守，也限制 decoded answer 恢复比例的解释力度。

## 当前边界

本轮不是 Qwen ReplacementModel/source tracing adapter。Qwen 的 source/control 仍是 approximate source-control probe，不等同于 Gemma 主线完整 source tracing。

本轮也不比较 `D_visual_only` 是否更好。prompt 只作为 modulation factor。
