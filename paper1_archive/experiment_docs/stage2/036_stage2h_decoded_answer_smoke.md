# Stage 2H-4 实验记录：Qwen/LLaVA hidden-position patch decoded answer smoke

## 0. 一句话结论

Stage 2H-4 已完成短 greedy generation smoke。结果是：

```text
Qwen: 出现 partial generation bridge。best hidden-position bridge 能让 4/6 rows 的 decoded answer 离开 union_mask answer，其中 2/4 informative rows 回到 clean/target answer。
LLaVA: first-token/rank bridge 仍然存在，但 decoded generation underpowered，因为 clean 与 union_mask 的短生成答案在 6/6 rows 中完全相同。
```

因此本轮可以写：

```text
Qwen provides partial decoded-answer bridge smoke.
LLaVA remains first-token/rank bridge only at this stage.
```

不能写：

```text
Qwen/LLaVA 已经完整证明 generation-level causal explanation；
Qwen/LLaVA 已经复现 Gemma source-control causal route；
hidden-position patch 已经等价于自然 source node intervention。
```

## 1. 实验目的

Stage 2H-1/2 已经证明，在 Qwen 和 LLaVA 中：

```text
top_hidden_delta_plus_answer_adjacent hidden positions
可以 restore union_mask run 的 target logit/rank，
也可以 corrupt clean run 的 target logit/rank。
```

但这仍然只是 first-token/rank bridge。Stage 2H-4 的目的，是检查这个 bridge 是否会传导到短 decoded answer：

```text
如果 union_mask 让模型生成错误答案，
那么把 best hidden-position group patch 回 clean state，
是否能让短 greedy generation 更接近 clean answer 或 target answer？
```

本轮仍是 smoke，不是大规模生成评估。

## 2. 输入

样本：

```text
okvqa_val_2847255
okvqa_val_4157235
okvqa_val_3658865
```

prompts：

```text
B_direct
D_visual_only
```

模型：

| model | layer | bucket |
| --- | ---: | --- |
| Qwen2.5-VL-7B-Instruct | 26 | `image_marker_or_span` |
| LLaVA-1.5-7B | 15 | `image_token_span` |

生成设置：

```text
max_new_tokens = 3
decoding = greedy
direction = restore only
```

对比条件：

```text
baseline::clean
baseline::union_mask
restore::top_hidden_delta_plus_answer_adjacent
restore::answer_adjacent_text
restore::low_delta_control
restore::random_control_1
LLaVA additional:
  restore::evidence_region_plus_answer_adjacent
```

## 3. 输出

Raw：

```text
doc/experiments/stage2/cross_model/stage2h_qwen_decoded_answer_smoke.json
doc/experiments/stage2/cross_model/stage2h_qwen_decoded_answer_smoke.csv
doc/experiments/stage2/cross_model/stage2h_llava_decoded_answer_smoke.json
doc/experiments/stage2/cross_model/stage2h_llava_decoded_answer_smoke.csv
```

Analysis：

```text
doc/experiments/stage2/cross_model/stage2h_decoded_answer_smoke_summary.csv
doc/experiments/stage2/cross_model/stage2h_decoded_answer_smoke_case_table.csv
doc/experiments/stage2/cross_model/stage2h_decoded_answer_smoke_decision.json
```

脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_cross_model_hidden_position_decode_smoke.py
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/analyze_stage2h_decoded_answer_smoke.py
scripts/local/run_stage2h_decoded_answer_smoke_remote.py
```

## 4. 方法

### 4.1 Hidden patch

本轮只做 restore：

```text
在 union_mask generation 中，把选定 positions 的 hidden state patch 回 clean hidden state。
```

形式上：

```text
patched_hidden[position] = union_hidden[position] + (clean_hidden[position] - union_hidden[position])
```

patch 位置来自 Stage 2H-1/2 已验证的组，主组固定为：

```text
top_hidden_delta_plus_answer_adjacent
```

也就是：

```text
clean 与 union_mask hidden 差异最大的 visual positions
+
last prompt / assistant prefix 附近的 answer-adjacent text positions
```

### 4.2 自定义 greedy generation

本轮没有直接调用普通 `model.generate`，而是写了一个短 greedy loop：

```text
每一步重新 forward 当前 input_ids；
在目标 language layer 上注册 forward hook；
对 prompt 内指定 positions 执行 hidden patch；
取最后位置 logits 的 argmax 作为下一个 token；
最多生成 3 个新 token。
```

这样做的原因：

```text
需要在 generation 每一步保持 hidden-position patch；
同时避免和不同模型的 generate/cache 接口耦合过深。
```

### 4.3 判读指标

主要记录：

| 指标 | 含义 |
| --- | --- |
| `predicted_answer` | 从短 continuation 中抽取出的答案 |
| `target_hit` | 英文字符串层面的 target answer 命中 |
| `semantic_target_hit` | 加入少量同义/翻译映射后的 target 命中，例如 `china` 和 `中国` |
| `answer_changed_vs_union` | patch 后答案是否不同于 union_mask baseline |
| `same_as_clean_n` | patch 后答案是否等于 clean baseline |
| `mean_logit_restore_vs_union` | prompt first-token target logit 相对 union_mask 的恢复 |
| `mean_rank_restore_vs_union` | prompt first-token target rank 相对 union_mask 的恢复 |

注意：

```text
decoded answer smoke 的主读数是 predicted_answer；
mean_logit_restore_vs_union 和 mean_rank_restore_vs_union 只是和 Stage 2H-1/2 对齐的辅助读数。
```

## 5. Qwen 结果

### 5.1 汇总表

| condition | n | target hit | changed vs union | same as clean | same as union | mean logit restore | mean rank restore | decoded answers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline::clean` | 6 | 6/6 | 4/6 | 6/6 | 2/6 | +3.9844 | +90.1667 | `a dog`, `china`, `samsung` |
| `baseline::union_mask` | 6 | 2/6 | 0/6 | 2/6 | 6/6 | +0.0000 | +0.0000 | `china`, `polo`, `unknown` |
| `restore::top_hidden_delta_plus_answer_adjacent` | 6 | 4/6 | 4/6 | 4/6 | 2/6 | +4.4427 | +90.0000 | `a horse`, `china`, `samsung` |
| `restore::answer_adjacent_text` | 6 | 2/6 | 4/6 | 2/6 | 2/6 | +3.6510 | +89.1667 | `a horse`, `china`, `nokia` |
| `restore::low_delta_control` | 6 | 2/6 | 0/6 | 2/6 | 6/6 | +0.0052 | -2.1667 | `china`, `polo`, `unknown` |
| `restore::random_control_1` | 6 | 2/6 | 0/6 | 2/6 | 6/6 | +0.1458 | +20.5000 | `china`, `polo`, `unknown` |

### 5.2 关键 case

| sample | prompt | clean | union_mask | best bridge restore | 读法 |
| --- | --- | --- | --- | --- | --- |
| `okvqa_val_2847255` | `B_direct` | `china` | `china` | `china` | clean/union 本来一致，不是生成桥证据 |
| `okvqa_val_2847255` | `D_visual_only` | `china` | `china` | `china` | clean/union 本来一致，不是生成桥证据 |
| `okvqa_val_4157235` | `B_direct` | `a dog` | `polo` | `a horse` | best bridge 离开 union，但未回到 target |
| `okvqa_val_4157235` | `D_visual_only` | `a dog` | `polo` | `a horse` | best bridge 离开 union，但未回到 target |
| `okvqa_val_3658865` | `B_direct` | `samsung` | `unknown` | `samsung` | 成功回到 clean/target |
| `okvqa_val_3658865` | `D_visual_only` | `samsung` | `unknown` | `samsung` | 成功回到 clean/target |

### 5.3 Qwen 判定

Decision：

```text
partial_generation_bridge_smoke
```

理由：

```text
clean vs union_mask decoded answer 不同的 informative rows = 4；
best bridge 在 4/4 informative rows 中都改变了 union answer；
其中 2/4 informative rows 回到 clean/target answer；
low_delta_control 和 random_control_1 都没有改变 union answer。
```

保守读法：

```text
Qwen 的 hidden-position bridge 可以影响 decoded answer，但不是稳定“救回正确答案”。
它更准确地说明：best bridge 能把生成从 masked answer basin 中推出，有时回到 target，有时进入另一个视觉/语言竞争答案。
```

这和主线很一致：内部路径能影响答案竞争，但生成层面仍有多 token、竞争答案和语言先验因素。

## 6. LLaVA 结果

### 6.1 汇总表

| condition | n | semantic target hit | changed vs union | same as clean | same as union | mean logit restore | mean rank restore | decoded answers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline::clean` | 6 | 2/6 | 0/6 | 6/6 | 6/6 | +2.9492 | +54.0000 | `中国`, `1`, `0` |
| `baseline::union_mask` | 6 | 2/6 | 0/6 | 6/6 | 6/6 | +0.0000 | +0.0000 | `中国`, `1`, `0` |
| `restore::top_hidden_delta_plus_answer_adjacent` | 6 | 2/6 | 0/6 | 6/6 | 6/6 | +2.1393 | +49.3333 | `中国`, `1`, `0` |
| `restore::evidence_region_plus_answer_adjacent` | 6 | 2/6 | 0/6 | 6/6 | 6/6 | +1.1882 | +41.6667 | `中国`, `1`, `0` |
| `restore::answer_adjacent_text` | 6 | 2/6 | 0/6 | 6/6 | 6/6 | +0.5072 | +9.1667 | `中国`, `1`, `0` |
| `restore::low_delta_control` | 6 | 2/6 | 0/6 | 6/6 | 6/6 | +0.0729 | +6.3333 | `中国`, `1`, `0` |
| `restore::random_control_1` | 6 | 2/6 | 0/6 | 6/6 | 6/6 | +1.3906 | +47.1667 | `中国`, `1`, `0` |

### 6.2 LLaVA 判定

Decision：

```text
decoded_generation_underpowered_no_clean_union_answer_gap
```

原因：

```text
clean 与 union_mask 的短 decoded answers 在 6/6 rows 中完全相同；
因此即使 hidden patch 恢复了 first-token target rank/logit，
也没有可观察的 decoded answer gap 可供恢复。
```

这不是说 LLaVA 没有跨模型机制信号。更准确的读法是：

```text
LLaVA Stage 2H 仍支持 first-token/rank bridge；
但本轮 decoded generation 设计没有形成 clean/union answer-level 差异，
所以不能升级为 generation-level bridge。
```

## 7. 和 Stage 2H-1/2 的关系

Stage 2H-1/2 的结论仍然成立：

```text
Qwen/LLaVA 的 best hidden-position bridge 可以恢复 target first-token logit/rank；
同一类位置也可以在反向 patch 中损伤 clean run；
这是 hidden-state-level causal localization。
```

Stage 2H-4 的新增结论是：

```text
Qwen 有部分 decoded answer 传导；
LLaVA 暂时没有 decoded answer 传导证据，因为生成答案没有 clean/union 差异。
```

因此跨模型证据阶梯更新为：

| 证据层级 | Qwen | LLaVA |
| --- | --- | --- |
| readout evidence-region sensitivity | supported | supported |
| evidence-mask target logit/rank damage | supported | supported |
| whole-bucket hidden patch upper bound | supported | supported |
| hidden-position causal localization | supported | supported |
| decoded answer bridge smoke | partial | underpowered / not established |
| CLT feature-level causal bridge | not established | not established |
| source-control route replication | not established | not established |

## 8. 最终读法

可以写：

```text
Qwen/LLaVA provide cross-model support below the full route-replication level.
Qwen further shows a partial decoded-answer bridge: the best hidden-position patch can move generation away from the masked answer and sometimes recover the clean/target answer.
LLaVA remains a first-token/rank bridge in this smoke because decoded answers did not differ between clean and union_mask.
```

必须保守写：

```text
Stage 2H-4 does not close the full causal loop for Qwen/LLaVA.
It is a smoke-level generation bridge, strongest for Qwen and not established for LLaVA.
Gemma remains the only model with full source/control causal route evidence.
```

## 9. 下一步

下一步不需要新标注，可以先做一个更合理的跨模型生成筛选：

| 优先级 | 实验 | 目的 |
| --- | --- | --- |
| 1 | Qwen 8-12 case decoded bridge expansion | 看 partial generation bridge 是否可复现 |
| 2 | LLaVA clean/union generation-gap screen | 先筛出 clean 与 union decoded answer 不同的 LLaVA 样本，否则 generation bridge 没有可恢复对象 |
| 3 | Qwen/LLaVA corrupt-direction decoded smoke | 检查 clean run 被 best bridge 反向 patch 后是否生成受损 |
| 4 | source adapter / tracing | 只有这一步之后，才可能讨论 source-control route replication |
