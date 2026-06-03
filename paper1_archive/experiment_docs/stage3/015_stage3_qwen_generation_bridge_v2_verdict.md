# 015 Stage3 Qwen Generation Bridge V2 Verdict

## 目的

本文件给出 Stage3-13 的结论口径：Qwen2.5-VL 当前是否能从 first-token/rank bridge 升级到 sequence-level bridge 或 decoded generation bridge。

## 输入

主要依据：

```text
013_stage3_qwen_generation_bridge_v2_run_plan.md
014_stage3_qwen_multifeature_sequence_bridge.md
stage3_qwen_generation_bridge_v2_summary.csv
stage3_qwen_generation_bridge_v2_case_table.csv
stage3_qwen_generation_bridge_v2_decision.json
```

## 判定矩阵

| 资产 | first-token/rank | sequence-level | decoded generation | 当前判定 |
|---|---|---|---|---|
| Qwen2.5-VL-CLT | 正向，延续 Stage3-011 | restore 单向较强；corrupt 不成立 | restore 0/4 回 clean | first_token_only，附带单向 sequence restore support |
| Qwen2.5-VL-PLT | 正向但弱 | top1 极弱 partial；CI 跨 0 | restore 0/3 回 clean | partial_sequence_bridge，但不能升级为 supported |

## 能下的结论

可以写：

```text
Qwen2.5-VL 的 source/control 差异已经能稳定影响目标答案首 token logit/rank。
```

可以写：

```text
在 Qwen2.5-VL-CLT 上，多 feature evidence-attribution restore
可以比 matched controls 更稳定提高 target answer sequence logprob；
但该信号是 restore-only，缺少 clean->masked corruption 的对称损伤。
```

可以写：

```text
Qwen2.5-VL-PLT 在 top1 feature group 上有 very weak partial sequence bridge hint，
但效应很小且 CI 跨 0。
```

可以写：

```text
目前 decoded generation bridge 仍未成立：
source_restore 没有稳定把 greedy decoded answer 拉回 clean/target。
```

## 不能下的结论

不能写：

```text
Qwen2.5-VL 已经完成 generation-level causal bridge。
```

不能写：

```text
Qwen2.5-VL 的 feature/source-control route 已经完整复现 Gemma-style source tracing。
```

不能写：

```text
CLT/PLT 结果失败说明 Qwen 没有这类机制。
```

不能写：

```text
D_visual_only prompt 更好。
```

## 与主线 claim 的关系

Stage3-13 不削弱 Gemma 主线。Gemma 主线的核心证据仍是：

```text
source tracing + node intervention + nearest/random controls
+ wrong-image/region-mask sensitivity + behavior/rank linkage
```

Stage3-13 对跨模型 claim 的影响是加细边界：

```text
Qwen2.5-VL 不是只在内部 feature/source-control 数值上有信号；
这些信号能触达 first-token/rank，并在 CLT restore 方向上触达 answer sequence logprob。
但从 sequence score 到 decoded generation 的最后一步仍未闭合。
```

因此最稳口径是：

```text
Gemma has the full mainline evidence.
Qwen2.5-VL provides PLT-aligned and CLT-auxiliary cross-model support
at feature/source-control and first-token/rank levels,
with limited sequence-score restore evidence but no stable decoded generation bridge.
```

## 后续如果继续推进

下一步如果还要冲 decoded generation bridge，应避免重复当前局部 feature patch，而是考虑更接近生成过程的设计：

```text
1. constrained decoding / forced answer scoring:
   继续以 answer sequence score 为主，避免 greedy 解码阈值掩盖小效应。

2. multi-position multi-layer restore:
   同时 patch visual source-like positions 与 answer-adjacent positions，
   并允许 layer 24/25/26 联合，而不是只用单层。

3. logit-lens / answer-token trajectory:
   跟踪目标答案每个 token 在 generation steps 中的 logprob/rank，
   判断恢复失败卡在哪一个 token。

4. 真正 Qwen ReplacementModel/source tracing adapter:
   只有这个完成后，才能尝试最接近 Gemma 主线的 source route 复现。
```

当前不建议再把 decoded greedy restore 作为唯一成功标准；它太硬，容易把已经存在的 first-token/rank 和 sequence-score 证据全部压扁。
