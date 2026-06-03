# Stage 2H 实验记录：Qwen/LLaVA hidden-position patch 因果定位 smoke

## 0. 一句话结论

Stage 2H 在 3 个已有主 case、2 个 prompt、2 个跨模型候选上完成了 hidden-state 级别的位置拆分实验。结果显示：

```text
Qwen2.5-VL layer 26 和 LLaVA-1.5 layer 15 中，部分 hidden-state 位置组可以从 union_mask 状态恢复 target answer logit/rank，也可以从 clean 状态损伤 target answer logit/rank。
最强位置组不是纯 image bucket，而是 top hidden-delta visual positions + answer-adjacent text positions。
```

因此，本轮支持：

```text
cross-model hidden-state-level causal localization / evidence-to-answer bridge
```

但仍然不支持直接写：

```text
Qwen/LLaVA 已经复现 Gemma 的 source-control causal route；
Qwen/LLaVA 的 CLT feature-level causal bridge 已经成立；
这些 features 或 hidden positions 已经是对象级语义节点。
```

## 1. 实验目的

Stage 2G 已经证明了一件重要但仍然粗粒度的事情：

```text
在 Qwen/LLaVA 中，把 union_mask run 的整段 image-bucket hidden state patch 回 clean hidden state，
可以恢复 target answer 的 first-token logit 或 rank。
```

这个结果说明跨模型不是完全停在 readout 相关性上，而是存在 hidden-state 级别的行为因果桥上界。但 Stage 2G 仍然有一个问题：

```text
整段 image bucket 太大。
如果 whole-bucket patch 有效，我们不知道恢复来自哪些位置；
也不知道这些位置是否比随机 visual positions 更特殊；
更不知道视觉位置是否需要和 answer-adjacent text positions 共同作用。
```

所以 Stage 2H 的目标是把 Stage 2G 的 whole-bucket patch 拆开，回答四个更细的问题：

| 问题 | 本轮对应实验 |
| --- | --- |
| 恢复效果来自哪些 hidden positions？ | Stage 2H-1 image-bucket position localization |
| source-like positions 是否强于 random / low-delta controls？ | Stage 2H-1 specificity comparison |
| 同一组 positions 是否既能 restore 又能 corrupt？ | Stage 2H-2 bidirectional hidden patch |
| 视觉位置是否需要 answer-adjacent text positions 配合？ | Stage 2H-3 answer-adjacent bridge |

## 2. 输入与实验范围

### 2.1 样本

本轮不新增标注，复用已有 3 个主 case 的 `answer/relate/union mask`：

| sample_id | 使用原因 |
| --- | --- |
| `okvqa_val_2847255` | 主图优先 case，已有区域 mask，跨模型 readout/hidden patch 可跑 |
| `okvqa_val_4157235` | 主图优先 case，已有区域 mask，跨模型 readout/hidden patch 可跑 |
| `okvqa_val_3658865` | 备用/扩展主 case，已有区域 mask，跨模型 readout/hidden patch 可跑 |

每个样本跑两个 prompt：

| prompt | 角色 |
| --- | --- |
| `B_direct` | 直接回答 prompt |
| `D_visual_only` | 视觉证据 prompt，但只作为 route modulation，不写成“更好” |

总有效读数：

```text
3 samples × 2 prompts = 6 rows / model / direction / group
```

### 2.2 模型与层

| model | layer | bucket | 选择依据 |
| --- | ---: | --- | --- |
| Qwen2.5-VL-7B-Instruct | 26 | `image_marker_or_span` | Stage 2F/2G 中 readout 和 whole-bucket hidden patch 最强 |
| LLaVA-1.5-7B | 15 | `image_token_span` | Stage 2F/2G 中最稳的 LLaVA readout 层 |

### 2.3 输入 artifact

主要输入来自前序 Stage 2F/2G 资产：

```text
doc/experiments/stage2/cross_model/stage2f_qwen_clean_vs_mask_feature_readout_3case.csv
doc/experiments/stage2/cross_model/stage2f_llava_clean_vs_mask_feature_readout_3case_layers15_30.csv
doc/experiments/stage2/cross_model/stage2g_hidden_patch_decision.json
annotation/okvqa_evidence_labelme_round4_core24_easy/exported_masks
```

本轮新增脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_cross_model_hidden_position_patch_smoke.py
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/analyze_stage2h_hidden_position_patch.py
scripts/local/run_stage2h_hidden_position_patch_remote.py
```

## 3. 输出文件

### 3.1 Raw 输出

```text
doc/experiments/stage2/cross_model/stage2h_qwen_hidden_position_patch.json
doc/experiments/stage2/cross_model/stage2h_qwen_hidden_position_patch.csv
doc/experiments/stage2/cross_model/stage2h_llava_hidden_position_patch.json
doc/experiments/stage2/cross_model/stage2h_llava_hidden_position_patch.csv
```

### 3.2 汇总输出

```text
doc/experiments/stage2/cross_model/stage2h_hidden_position_patch_baseline_gap.csv
doc/experiments/stage2/cross_model/stage2h_hidden_position_patch_summary.csv
doc/experiments/stage2/cross_model/stage2h_hidden_position_patch_specificity.csv
doc/experiments/stage2/cross_model/stage2h_hidden_position_patch_case_table.csv
doc/experiments/stage2/cross_model/stage2h_hidden_position_patch_decision.json
```

### 3.3 文档输出

```text
doc/experiments/stage2/034_stage2h_cross_model_causal_localization_run_plan.md
doc/experiments/stage2/035_stage2h_hidden_position_patch_smoke.md
doc/experiments/stage2_expetiments.md
doc/experiments/stage2/cross_model/cross_model_candidate_table.csv
```

## 4. 方法

### 4.1 条件

本轮主要比较两个 forward 状态：

| condition | 含义 |
| --- | --- |
| `clean` | 原图输入 |
| `union_mask` | 遮挡 `answer ∪ relate` 区域后的输入 |

这里不用 `answer_mask` 单独做主读数，是因为 Stage 2G hidden patch 上界已经以 `union_mask` 为主要 corruption anchor；Stage 2H 目标是定位这个上界，而不是重新比较不同 mask 条件。

### 4.2 patch 方向

本轮做双向 hidden patch。

`restore` 方向：

```text
在 union_mask forward 中，把目标 positions 的 hidden state 替换或推回 clean hidden state。
正值表示 target answer signal 被恢复。
```

形式上：

```text
patched_hidden[position] = union_hidden[position] + scale × (clean_hidden[position] - union_hidden[position])
```

`corrupt` 方向：

```text
在 clean forward 中，把目标 positions 的 hidden state 推向 union_mask hidden state。
正值表示 target answer signal 被损伤。
```

形式上：

```text
patched_hidden[position] = clean_hidden[position] + scale × (union_hidden[position] - clean_hidden[position])
```

本轮主读数固定：

```text
scale = 1.0
```

也就是完整替换目标 positions 的 hidden state。

### 4.3 位置组

| group_name | group_kind | 含义 |
| --- | --- | --- |
| `whole_bucket` | `upper_bound` | Stage 2G 的整段 bucket patch，上界参考 |
| `evidence_region` | `source_like` | LLaVA 中由 24×24 image grid 和 evidence mask overlap 得到的视觉证据位置；Qwen 不使用这个严格 claim |
| `top_hidden_delta` | `source_like` | clean 与 union_mask hidden-state 差异最大的 visual positions |
| `answer_adjacent_text` | `bridge_text` | last prompt / assistant prefix 附近的少量文本位置 |
| `evidence_region_plus_answer_adjacent` | `bridge_combo` | evidence-region visual positions 加 answer-adjacent text positions |
| `top_hidden_delta_plus_answer_adjacent` | `bridge_combo` | top hidden-delta visual positions 加 answer-adjacent text positions |
| `random_control_1..4` | `random_control` | 同数量随机 visual positions |
| `low_delta_control` | `control` | hidden delta 较低的 visual positions |

### 4.4 模型特定定位细节

LLaVA：

```text
image token span = 576 tokens
grid = 24×24
evidence-region positions = 64 positions
grid status = ok
```

因此 LLaVA 可以写成：

```text
evidence-region token localization was technically available.
```

Qwen：

```text
processor 提供 image_grid_thw；
但 visual span 长度和 grid 映射无法可靠对齐；
raw diagnostics 显示 qwen_grid_unavailable_or_mismatch。
```

因此 Qwen 不能写严格 evidence-region token localization，只能写：

```text
hidden-delta source-like position localization
```

这是一个重要降级，不影响 Qwen hidden-state bridge 结果，但影响“区域 token 对齐”的 claim 强度。

## 5. 基线：clean vs union_mask 的损伤

在做 patch 前，先确认 `union_mask` 确实损伤 target answer：

| model | bucket | n | mean clean-union logit gap | positive logit gap | mean clean-union rank gap | positive rank gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen | `image_marker_or_span` | 6 | +3.9844 | 6/6 | +90.1667 | 4/6 |
| LLaVA | `image_token_span` | 6 | +2.9492 | 6/6 | +54.0000 | 6/6 |

解释：

```text
positive logit gap 表示 clean target logit 高于 union_mask；
positive rank gap 表示 union_mask 后 target rank 变差。
```

这个基线说明，本轮 3-case smoke 中，`union_mask` 对两个模型都确实造成了 target-answer 行为侧损伤，patch 实验有可恢复空间。

## 6. Qwen 结果

### 6.1 Restore：从 union_mask 恢复 target signal

| group | n | mean logit restore | positive logit | mean rank restore | positive rank | mean gap closure | top1 changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `whole_bucket` | 6 | +0.7656 | 5/6 | +58.5000 | 4/6 | +0.2004 | 1 |
| `top_hidden_delta` | 6 | +0.3490 | 5/6 | +37.5000 | 4/6 | +0.1010 | 1 |
| `answer_adjacent_text` | 6 | +3.6510 | 4/6 | +89.1667 | 3/6 | +2.0271 | 4 |
| `top_hidden_delta_plus_answer_adjacent` | 6 | +4.4427 | 6/6 | +90.0000 | 4/6 | +2.3317 | 4 |
| `low_delta_control` | 6 | +0.0052 | 1/6 | -2.1667 | 0/6 | +0.0060 | 0 |

最关键的 specificity 读数：

| comparison | logit | rank | gap closure |
| --- | ---: | ---: | ---: |
| `top_hidden_delta_plus_answer_adjacent - random4_mean` | +4.3255 | +78.9167 | +2.2827 |
| `top_hidden_delta_plus_answer_adjacent - low_delta_control` | +4.4375 | +92.1667 | +2.3256 |
| `top_hidden_delta - random4_mean` | +0.2318 | +26.4167 | +0.0520 |
| `top_hidden_delta - low_delta_control` | +0.3438 | +39.6667 | +0.0950 |

读法：

```text
Qwen 的纯 top_hidden_delta visual positions 已有正向恢复，但最强恢复来自 top_hidden_delta visual positions + answer-adjacent text positions。
这一组合明显强于 random visual controls 和 low-delta controls。
```

### 6.2 Corrupt：从 clean 损伤 target signal

| group | n | mean logit damage | positive logit | mean rank damage | positive rank | mean gap closure | top1 changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `whole_bucket` | 6 | +0.7292 | 6/6 | +0.3333 | 2/6 | +0.3890 | 2 |
| `top_hidden_delta` | 6 | +0.1458 | 4/6 | +0.0000 | 0/6 | +0.0597 | 0 |
| `answer_adjacent_text` | 6 | +3.1563 | 6/6 | +15.1667 | 2/6 | +1.5332 | 2 |
| `top_hidden_delta_plus_answer_adjacent` | 6 | +3.2188 | 6/6 | +17.0000 | 2/6 | +1.5398 | 2 |
| `low_delta_control` | 6 | +0.0208 | 1/6 | +0.0000 | 0/6 | +0.0020 | 0 |

最关键的 specificity 读数：

| comparison | logit | rank | gap closure |
| --- | ---: | ---: | ---: |
| `top_hidden_delta_plus_answer_adjacent - random4_mean` | +3.1667 | +17.0000 | +1.5192 |
| `top_hidden_delta_plus_answer_adjacent - low_delta_control` | +3.1979 | +17.0000 | +1.5379 |
| `top_hidden_delta - random4_mean` | +0.0938 | +0.0000 | +0.0391 |
| `top_hidden_delta - low_delta_control` | +0.1250 | +0.0000 | +0.0577 |

读法：

```text
同一最佳组 top_hidden_delta_plus_answer_adjacent 在 restore 和 corrupt 两个方向都成立。
这满足 Stage 2H-2 的 bidirectional hidden-position localization 判据。
```

### 6.3 Qwen 判定

Qwen 的 decision JSON 判定为：

```text
supported_bidirectional_hidden_position_localization
```

但 claim 必须限定为：

```text
hidden-delta source-like positions + answer-adjacent positions form a hidden-state causal bridge.
```

不能写成：

```text
Qwen evidence-region token localization 已严格成立。
```

原因是 Qwen 的 image grid 映射仍为：

```text
qwen_grid_unavailable_or_mismatch
```

## 7. LLaVA 结果

### 7.1 Restore：从 union_mask 恢复 target signal

| group | n | mean logit restore | positive logit | mean rank restore | positive rank | mean gap closure | top1 changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `whole_bucket` | 6 | +2.4128 | 6/6 | +50.5000 | 4/6 | +0.6669 | 0 |
| `evidence_region` | 6 | +0.6660 | 4/6 | +36.3333 | 4/6 | +0.1220 | 0 |
| `top_hidden_delta` | 6 | +1.5755 | 6/6 | +47.3333 | 3/6 | +0.2525 | 0 |
| `answer_adjacent_text` | 6 | +0.5072 | 6/6 | +9.1667 | 6/6 | +0.4520 | 0 |
| `evidence_region_plus_answer_adjacent` | 6 | +1.1882 | 6/6 | +41.6667 | 6/6 | +0.5660 | 0 |
| `top_hidden_delta_plus_answer_adjacent` | 6 | +2.1393 | 6/6 | +49.3333 | 6/6 | +0.7144 | 0 |
| `low_delta_control` | 6 | +0.0729 | 6/6 | +6.3333 | 2/6 | +0.0270 | 0 |

最关键的 specificity 读数：

| comparison | logit | rank | gap closure |
| --- | ---: | ---: | ---: |
| `top_hidden_delta_plus_answer_adjacent - random4_mean` | +0.7137 | +3.7083 | +0.4643 |
| `top_hidden_delta_plus_answer_adjacent - low_delta_control` | +2.0664 | +43.0000 | +0.6874 |
| `evidence_region_plus_answer_adjacent - random4_mean` | -0.2375 | -3.9583 | +0.3160 |
| `evidence_region_plus_answer_adjacent - low_delta_control` | +1.1152 | +35.3333 | +0.5390 |
| `evidence_region - random4_mean` | -0.7596 | -9.2917 | -0.1281 |
| `evidence_region - low_delta_control` | +0.5931 | +30.0000 | +0.0950 |

读法：

```text
LLaVA 中 evidence_region 技术上可定位，但单独 evidence_region 没有强于 random controls；
top_hidden_delta_plus_answer_adjacent 是最稳的 restore 组；
evidence_region_plus_answer_adjacent 对 low-delta control 强，但对 random4_mean 不稳定。
```

这说明 LLaVA 的区域定位结果要比 Qwen 更可映射，但机制上并不是“人工证据区域 token 单独就足够”。更稳的说法是：

```text
visual evidence positions need to be combined with answer-adjacent positions, and top hidden-delta positions are more predictive than raw evidence-overlap positions in this smoke.
```

### 7.2 Corrupt：从 clean 损伤 target signal

| group | n | mean logit damage | positive logit | mean rank damage | positive rank | mean gap closure | top1 changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `whole_bucket` | 6 | +1.5417 | 4/6 | +10.6667 | 3/6 | +0.2338 | 1 |
| `evidence_region` | 6 | -0.0911 | 4/6 | +0.3333 | 1/6 | -0.1528 | 0 |
| `top_hidden_delta` | 6 | +0.0104 | 4/6 | +0.3333 | 1/6 | +0.0056 | 0 |
| `answer_adjacent_text` | 6 | +0.4609 | 6/6 | +3.3333 | 4/6 | +0.4155 | 0 |
| `evidence_region_plus_answer_adjacent` | 6 | +0.4544 | 6/6 | +3.5000 | 5/6 | +0.3660 | 0 |
| `top_hidden_delta_plus_answer_adjacent` | 6 | +0.4870 | 6/6 | +3.6667 | 5/6 | +0.4283 | 0 |
| `low_delta_control` | 6 | +0.0156 | 4/6 | +0.0000 | 0/6 | +0.0154 | 0 |

最关键的 specificity 读数：

| comparison | logit | rank | gap closure |
| --- | ---: | ---: | ---: |
| `top_hidden_delta_plus_answer_adjacent - random4_mean` | +0.4961 | +3.3750 | +0.4564 |
| `top_hidden_delta_plus_answer_adjacent - low_delta_control` | +0.4714 | +3.6667 | +0.4129 |
| `evidence_region_plus_answer_adjacent - random4_mean` | +0.4635 | +3.2083 | +0.3941 |
| `evidence_region_plus_answer_adjacent - low_delta_control` | +0.4388 | +3.5000 | +0.3506 |
| `evidence_region - random4_mean` | -0.0820 | +0.0417 | -0.1247 |
| `evidence_region - low_delta_control` | -0.1068 | +0.3333 | -0.1682 |

读法：

```text
LLaVA 中单独 evidence_region 不稳定；
但 evidence_region_plus_answer_adjacent 和 top_hidden_delta_plus_answer_adjacent 在 corrupt 方向都强于 controls。
其中 top_hidden_delta_plus_answer_adjacent 同时满足 restore 和 corrupt 两个方向的主判据。
```

### 7.3 LLaVA 判定

LLaVA 的 decision JSON 判定为：

```text
supported_bidirectional_hidden_position_localization
```

但要带两个保守限定：

```text
第一，最强组仍是 top_hidden_delta_plus_answer_adjacent，而不是 evidence_region alone；
第二，evidence_region 的 grid 映射可用，但 evidence_region alone 没有足够强的 random-control specificity。
```

## 8. Prompt 读法

本轮没有把 prompt 作为“谁更好”的主问题，也没有按 `B_direct` vs `D_visual_only` 写行为优劣结论。所有主读数都是 pooled across prompt：

```text
3 samples × 2 prompts = 6 rows
```

这和当前主线一致：

```text
prompt 是 route modulation / exposure variable，
不是行为优劣主 claim。
```

## 9. 预期与实际偏差

### 9.1 符合预期的部分

预期：

```text
如果 Stage 2G 的 whole-bucket hidden patch 上界不是偶然现象，
那么更小的位置组应该能恢复一部分 target answer signal。
```

实际：

```text
Qwen 和 LLaVA 都出现了可定位的 hidden-position bridge；
最佳组都在 6/6 rows 上产生正向 logit restore；
最佳组也在 6/6 rows 上产生正向 logit corruption/damage；
最佳组强于 random controls 和 low-delta controls。
```

### 9.2 超出预期的部分

Qwen 的 `top_hidden_delta_plus_answer_adjacent` restore 很强：

```text
mean logit restore = +4.4427
mean gap closure = +2.3317
```

gap closure 超过 1，说明 patch 后 target logit 不只是回到 clean/union 差距以内，而是出现了 over-restoration。这是有信号的，但也提示：

```text
hidden patch 不是自然因果路径的精确还原；
它是一个强干预，可以超过原始 clean-run effect。
```

### 9.3 低于预期或需要降级的部分

Qwen：

```text
image_grid_thw 存在，但无法可靠映射到 visual span；
所以 Qwen 不能写 evidence-region token localization。
```

LLaVA：

```text
evidence_region alone 没有强于 random controls；
说明人工证据区域与 hidden-state 行为因果位置并不一一对应。
```

Stage 2H-4：

```text
后续已补跑 decoded answer smoke，详见：
doc/experiments/stage2/036_stage2h_decoded_answer_smoke.md

结果是：
Qwen = partial generation bridge smoke；
LLaVA = decoded generation underpowered，仍主要保留 first-token/rank bridge。
```

## 10. 接受标准对照

| 接受标准 | 结果 | 状态 |
| --- | --- | --- |
| Stage 2H-1：source-like positions mean logit restore > random controls | Qwen `top_hidden_delta_plus_answer_adjacent` +4.3255；LLaVA +0.7137 | success |
| Stage 2H-1：positive logit restore ≥ 4/6 | Qwen 6/6；LLaVA 6/6 | success |
| Stage 2H-1：rank restore positive ≥ 3/6 | Qwen 4/6；LLaVA 6/6 | success |
| Stage 2H-2：同一组 restore/corrupt 均强于 controls | Qwen 与 LLaVA 的最佳组均满足 | success |
| Stage 2H-3：image + answer-adjacent 强于单独 image | Qwen 与 LLaVA 均满足，尤其 Qwen 很明显 | success |
| Stage 2H-4：decoded answer bridge | Qwen partial；LLaVA underpowered；详见 `036_stage2h_decoded_answer_smoke.md` | partial |

## 11. 最终判定

### 11.1 可以写入报告的结论

```text
In Qwen2.5-VL and LLaVA-1.5, evidence-mask damage is not only visible in readout statistics.
At selected layers, clean hidden-state patches over source-like visual positions plus answer-adjacent text positions can restore target-answer first-token logit/rank from union-mask runs, and the reverse patch can damage clean runs.
This provides cross-model hidden-state-level causal localization support.
```

中文口径：

```text
Qwen/LLaVA 中也存在 hidden-state 级别的视觉证据到答案信号桥接。
这种桥接在本轮 3-case smoke 中表现为：
遮挡证据区域后，特定 hidden positions 的 clean-state patch 可以恢复 target answer logit/rank；
反向把 clean hidden state 推向 masked hidden state 会损伤 target answer logit/rank；
最强位置组包含 top hidden-delta visual positions 与 answer-adjacent text positions。
```

### 11.2 不能写的结论

```text
不能写 Qwen/LLaVA 已经复现 Gemma source-control causal route。
不能写 Qwen/LLaVA 已经找到 CLT feature-level causal source nodes。
不能写 hidden positions 是对象级语义节点。
不能写 D_visual_only 比 B_direct 更好。
不能写完整 decoded generation bridge 已成立；Qwen 只能写 partial decoded bridge smoke，LLaVA 只能写 first-token/rank bridge。
```

### 11.3 与 Gemma 主线的关系

Gemma 主线仍然是完整机制链：

```text
source tracing
node intervention
nearest/random controls
wrong-image sensitivity
region-mask sensitivity
rank/generation linkage
```

Qwen/LLaVA 当前升级到：

```text
readout sensitivity
evidence-mask behavior damage
whole-bucket hidden patch upper bound
hidden-position bridge localization
```

但还未到：

```text
feature-level source/control causal route replication
```

所以 Stage 2H 对主 claim 的影响是：

```text
它加强了“该现象不是 Gemma-only”的外部有效性；
但不改变主 claim 的证据等级排序。
Gemma 仍是 full-chain main evidence；
Qwen/LLaVA 是 cross-model support at hidden-state bridge level。
```

## 12. 下一步

Stage 2H-4 已经补跑。现在最自然的下一步有三条：

| 优先级 | 下一步 | 目的 |
| --- | --- | --- |
| 1 | Qwen 8-12 case decoded bridge expansion | 检查 partial generation bridge 是否可复现 |
| 2 | LLaVA clean/union generation-gap screen | 先找 clean 与 union decoded answer 不同的 LLaVA 样本 |
| 3 | Qwen/LLaVA source adapter / route tracing | 才能向 source-control causal route replication 推进 |

如果只做一件事，建议先做：

```text
Qwen 8-12 case decoded bridge expansion。
```
