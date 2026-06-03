# 012 Stage3 Decoded Behavior Bridge Smoke

## 目的

检查 Qwen2.5-VL-PLT / Qwen2.5-VL-CLT 中已经通过 source-control 与 first-token/rank bridge 的 rows，是否进一步表现为短 greedy generation 的答案变化。

本实验回答的是更强一层的问题：

```text
source/control feature 干预是否能改变最终 decoded answer？
```

它比 first-token/rank bridge 更难，因为短文本生成会受到 top-1 token、解码惯性、答案格式和多 token 答案共同影响。

## 输入

第一轮 effect-gap manifest：

```text
doc/experiments/stage3/cross_model/stage3_qwen_decoded_bridge_manifest.csv
```

第二轮 rank-aware manifest：

```text
doc/experiments/stage3/cross_model/stage3_qwen_decoded_bridge_rankaware_manifest.csv
```

模型与资产：

```text
Qwen2.5-VL-PLT = KokosDev/qwen2p5vl-7b-plt
Qwen2.5-VL-CLT = KokosDev/qwen2p5vl-7b-clt
base = Qwen/Qwen2.5-VL-7B-Instruct
```

## 输出

```text
stage3_qwen2p5vl_plt_decoded_bridge.csv/json
stage3_qwen2p5vl_clt_decoded_bridge.csv/json
stage3_qwen2p5vl_plt_decoded_bridge_rankaware.csv/json
stage3_qwen2p5vl_clt_decoded_bridge_rankaware.csv/json
stage3_decoded_bridge_summary.csv
stage3_decoded_bridge_case_table.csv
stage3_decoded_bridge_decision.json
```

脚本：

```text
scripts/local/build_stage3_decoded_bridge_manifest.py
scripts/local/build_stage3_decoded_bridge_rankaware_manifest.py
scripts/local/run_stage3_qwen_decoded_bridge_remote.py
scripts/local/analyze_stage3_decoded_bridge.py
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_stage3_qwen_feature_decode_bridge.py
```

## 方法

每个 selected pair 跑 6 个条件：

```text
baseline_clean
baseline_mask
source_restore
control_restore
source_zeroing
control_zeroing
```

每个条件生成 3 个 greedy tokens，并记录：

```text
predicted_answer
target_hit
first_step_target_logit
first_step_target_rank
changed_vs_clean
changed_vs_mask
```

两轮选择策略不同：

```text
effect_gap:
  按 source-control restore + zeroing 的内部 logit gap 选 top rows。

rankaware:
  优先选择 clean target rank <= 5 且 mask 后 target rank 变差的 rows。
  这样更适合 decoded generation 诊断。
```

专有名词解释：

```text
decoded generation:
  模型实际生成的文本答案，而不是单个 token 的 logit 或 rank。

greedy generation:
  每一步都选择概率最高的下一个 token，不采样。

source_restore_to_clean:
  source restore 后生成答案回到 clean 答案；这是 generation bridge 的强信号。
```

## 结果

运行状态：

| 资产 | selection | pairs | rows | 运行状态 |
|---|---|---:|---:|---|
| Qwen2.5-VL-PLT | effect_gap | 4 | 24 | completed |
| Qwen2.5-VL-CLT | effect_gap | 4 | 24 | completed |
| Qwen2.5-VL-PLT | rankaware | 6 | 36 | completed |
| Qwen2.5-VL-CLT | rankaware | 6 | 36 | completed |

核心 summary：

| 资产 | selection | clean/mask 答案不同 | clean target hit | source_restore_to_clean | source_zeroing 改变 clean | 平均 source restore logit vs mask | 平均 source restore rank vs mask |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-PLT | effect_gap | 3/4 | 1/4 | 0/4 | 0/4 | +0.156 | +0.250 |
| Qwen2.5-VL-PLT | rankaware | 6/6 | 3/6 | 0/6 | 1/6 | +0.057 | +25.667 |
| Qwen2.5-VL-CLT | effect_gap | 4/4 | 0/4 | 0/4 | 2/4 | +0.828 | +18.250 |
| Qwen2.5-VL-CLT | rankaware | 6/6 | 4/6 | 0/6 | 0/6 | +0.177 | +41.167 |

## 预期与实际偏差

预期 rank-aware selection 会比 effect-gap selection 更适合 generation bridge，实际确实选出了更多 clean target hit 且 clean/mask 答案不同的 case。

但实际结果显示：

```text
source_restore 虽然能提高 first-step target logit/rank，
但没有稳定把 decoded answer 从 mask 答案恢复到 clean 答案。
```

这说明当前 Qwen feature/source-control 证据已经能触达 first-token/rank 层，但还不足以稳定改写短文本生成。

## 结论

当前可以写：

```text
Qwen2.5-VL 的 source/control feature 干预有 first-token/rank 行为桥接；
rank-aware decoded smoke 也看到 first-step logit/rank 恢复，
但 decoded answer restoration 尚未成立。
```

当前不能写：

```text
Qwen2.5-VL 已经完成 generation-level causal bridge。
Qwen feature source/control 干预可以稳定恢复自然语言答案。
```

下一步如果继续推进 generation bridge，需要更强的干预方式，例如：

```text
1. 多 feature 组合 restore，而不是单 source feature。
2. 更长 answer span 的 constrained decoding / target-token sequence score。
3. 只选 clean 生成正确、mask 生成错误且 first-token gap 适中的 case。
4. 做 full answer logprob / sequence likelihood，而不是只看 greedy decoded text。
```
