# Stage 2M-2：Decoded Bridge on Passing Cases

## 1. 目的

本实验只在 Stage 2M Tier 1 通过的高分样本上跑短生成，检验：

```text
hidden-state patch 能否不只恢复 target logit / rank，也让 decoded answer 往 clean answer 或 target answer 方向移动。
```

这一步是 generation-level smoke，不是完整 generation restoration，也不是 Gemma-style source-control route replication。

## 2. 输入

候选选择：

```text
doc/experiments/stage2/cross_model/stage2m_decoded_candidate_prompt_rows.csv
doc/experiments/stage2/cross_model/stage2m_decoded_candidate_manifest.csv
doc/experiments/stage2/cross_model/stage2m_decoded_candidate_decision.json
```

模型与层：

```text
Qwen: Qwen2.5-VL-7B-Instruct, layer 26
LLaVA: LLaVA-1.5-7B, layer 15
prompts: B_direct, D_visual_only
generation: greedy, max_new_tokens = 3
```

样本规模：

```text
Qwen: 8 unique samples, 16 prompt-runs
LLaVA: 6 unique samples, 12 prompt-runs
```

## 3. 方法

对每个 sample x prompt 比较：

```text
clean generation
union_mask generation
restore::top_hidden_delta_plus_answer_adjacent
restore::delta_matched_plus_answer_adjacent
restore::activation_matched_plus_answer_adjacent
restore::low_delta_control
restore::random_control_1
```

关键术语：

```text
decoded bridge：隐藏状态 patch 后，最终生成文本是否从 masked answer 往 clean/target answer 方向移动。
informative row：clean generation 与 union_mask generation 不同的样本；只有这类样本才能判断 generation 是否被拉回。
target_hit：生成答案是否命中目标答案。
same_as_clean：patch 后的生成答案是否等于 clean generation。
```

## 4. 输出

```text
doc/experiments/stage2/cross_model/stage2m_qwen_decoded_bridge.csv/json
doc/experiments/stage2/cross_model/stage2m_llava_decoded_bridge.csv/json
doc/experiments/stage2/cross_model/stage2m_decoded_bridge_summary.csv
doc/experiments/stage2/cross_model/stage2m_decoded_bridge_case_table.csv
doc/experiments/stage2/cross_model/stage2m_decoded_bridge_decision.json
```

## 5. 结果

可用性：

```text
Qwen usable_runs = 16/16, rows = 128
LLaVA usable_runs = 12/12, rows = 108
```

Decision：

```text
Qwen status = partial_generation_bridge_smoke
LLaVA status = partial_generation_bridge_smoke
```

Qwen：

```text
informative_clean_vs_union_rows = 13
source restore target_hit = 4/16
union_mask target_hit = 2/16
low_delta target_hit = 3/16
random_control target_hit = 3/16
source same_as_clean on informative rows = 3
source target hits on informative rows = 2
source mean_logit_restore_vs_union = +5.255859
source mean_rank_restore_vs_union = +1247.875
```

LLaVA：

```text
informative_clean_vs_union_rows = 11
source restore target_hit = 2/12
union_mask target_hit = 0/12
low_delta target_hit = 0/12
random_control target_hit = 0/12
source same_as_clean on informative rows = 2
source target hits on informative rows = 2
source mean_logit_restore_vs_union = +4.100260
source mean_rank_restore_vs_union = +45.25
```

## 6. 读法

Qwen 的 first-token / rank bridge 很强，但 decoded answer 的 source-specificity 只是部分成立：

```text
source restore target_hit 高于 union_mask，但只略高于 low_delta/random controls。
source restore 的 mean logit/rank 很大，但 delta/activation matched controls 也能恢复相当多 first-token signal。
```

LLaVA 的 decoded bridge 更干净但规模更小：

```text
source restore target_hit = 2/12，而 low_delta/random controls = 0/12。
但总体命中数仍少，只能写 partial generation bridge smoke。
```

## 7. 结论边界

可写：

```text
On passing cases, Qwen and LLaVA show partial decoded-answer bridge support: hidden patch can move some masked generations toward the clean/target answer, with stronger first-token/rank recovery than full natural-answer restoration.
```

不可写：

```text
Qwen/LLaVA hidden patch reliably restores decoded answers.
Qwen/LLaVA have completed generation-level causal route replication.
```
