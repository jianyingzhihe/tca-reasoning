# 006 Stage3 Behavior Bridge

## 目的

把内部 feature/source-control 结果和外部行为连接起来。这里的行为桥接分两层：

```text
first-token/rank bridge:
  干预后目标答案 token 的 logit/rank 发生方向性变化。

decoded generation bridge:
  干预后短 greedy generation 的最终答案发生方向性变化。
```

## 输入

只选择内部证据通过的 rows：

```text
Qwen2.5-VL-PLT source/control positive rows
Qwen2.5-VL-CLT source/control positive rows
必要时加入 LLaVA-CLT hidden/source-like positive rows
```

## 输出

```text
stage3_behavior_bridge_first_token.csv/json
stage3_behavior_bridge_decoded.csv/json
006_stage3_behavior_bridge.md 更新结果
```

## 方法

条件固定为：

```text
clean generation
answer_mask generation
union_mask generation
source_zeroing generation
control_zeroing generation
source_restore generation
control_restore generation
```

判定逻辑：

```text
decoded answer 有方向变化:
  可写 generation bridge support。

decoded answer 不变，但 first-token/rank 稳定:
  只写 first-token/rank bridge。

source/control 差异不成立:
  不写 causal specificity。
```

## 当前结果

已完成 first-token/rank bridge 的本地分析，也完成了 Qwen2.5-VL-PLT / Qwen2.5-VL-CLT 的 short decoded generation smoke。
随后 Stage3-13 又完成了 Qwen generation bridge v2：把干预从 single-feature 扩展到 attribution-weighted multi-feature group，并加入 target answer sequence logprob。

Primary first-token/rank 结果：

| 资产 | 判定 | 核心读数 |
|---|---|---|
| Qwen2.5-VL-CLT | supported_first_token_rank_bridge | answer/union restore 与 zeroing 的 source-control logit CI 均大于 0 |
| Qwen2.5-VL-PLT | partial_first_token_rank_bridge | logit 方向稳定，union restore 与 zeroing rank 也为正；answer restore rank 较弱 |
| LLaVA-CLT | partial_first_token_rank_bridge | zeroing 小而稳定，restoration 不稳定 |

关键 artifact：

```text
stage3_behavior_first_token_comparisons.csv
stage3_behavior_first_token_summary.csv
stage3_behavior_first_token_decision.json
stage3_decoded_bridge_summary.csv
stage3_decoded_bridge_case_table.csv
stage3_decoded_bridge_decision.json
```

Decoded smoke 结果：

| 资产 | selection | pairs | source_restore_to_clean | 结论 |
|---|---|---:|---:|---|
| Qwen2.5-VL-PLT | effect_gap | 4 | 0/4 | first-token/rank only |
| Qwen2.5-VL-PLT | rankaware | 6 | 0/6 | first-token/rank only |
| Qwen2.5-VL-CLT | effect_gap | 4 | 0/4 | zeroing 有变化，但 restore 不回 clean |
| Qwen2.5-VL-CLT | rankaware | 6 | 0/6 | first-token/rank only |

Sequence-score v2 结果：

| 资产 | 判定 | 核心读数 | decoded restore |
|---|---|---|---:|
| Qwen2.5-VL-CLT | first_token_only + restore-only sequence support | restore top1/top4/top32 的 source-control sequence CI 大于 0；corrupt 方向为负或跨 0 | 0/4 |
| Qwen2.5-VL-PLT | partial_sequence_bridge | top1 restore +0.004、corrupt +0.007，但 CI 均跨 0；其他 topK 不成立 | 0/3 |

## 预期与实际偏差

预期 Qwen2.5-VL-CLT 最可能先通过 behavior bridge，实际 first-token/rank 层成立。Qwen2.5-VL-PLT 有 partial bridge，说明 PLT-aligned Qwen 的内部 source-control 信号确实能触达答案首 token，但强度弱于 CLT。LLaVA-CLT 只保留 partial/diagnostic。

预期 decoded generation 可能在 rank-aware rows 上出现少量恢复；实际 source_restore 没有稳定把 decoded answer 恢复到 clean answer。偏差说明 greedy 文本生成比 first-token/rank 更难被单 feature restore 改动。

Stage3-13 预期 multi-feature restore 和 answer sequence score 会比 single-feature greedy smoke 更敏感。实际 CLT 的 restore sequence score 确实出现更强信号，但 clean->masked corruption 没有同步成立；PLT 只有 top1 极弱 partial 信号。偏差说明 Qwen 的 evidence-attribution features 可以补回部分目标答案倾向，但还没有证明它们足以双向控制完整答案序列，更没有稳定控制 greedy decoded answer。

## 当前结论

Stage3 当前可以写 first-token/rank-level behavior bridge，尤其 Qwen2.5-VL-CLT 最稳，Qwen2.5-VL-PLT 有较弱但正向的 PLT-aligned 行为桥接。

Stage3 当前仍不能写 decoded generation-level cross-model causal bridge。Stage3-13 允许对 Qwen2.5-VL-CLT 增加一个更细限定：存在 restore-only sequence-score support；但由于 corrupt 不成立，不能升级为 supported sequence-level causal bridge。
