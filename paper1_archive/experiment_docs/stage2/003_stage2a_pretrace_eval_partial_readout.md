# 实验 003：Stage 2A-1 top24 pretrace eval 部分读数

## 目的

本文件记录 Stage 2A-1 top24 pretrace 在 answer-aligned trace 完成前已经可以读取的行为侧结果。

这一步的目的不是最终筛选 replication samples，而是先判断：

```text
1. B_direct / D_visual_only 是否都能在 top24 样本上生成非空答案；
2. 是否存在明显 format collapse 或 empty-answer 问题；
3. 哪些样本在 clean generation 上已经比较可疑，后续需要谨慎进入机制复现实验；
4. D_visual_only 是否仍然只能作为 route modulation / probing variable，而不能被写成行为优势 prompt。
```

## 输入

远端 run root：

```text
/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm/outputs/phase_ab/ab_answer_aligned/stage2a_pretrace_top24_20260519_202951
```

已同步到本地的文件：

```text
E:\Bridging\remote_sync\2026-05-19_stage2a_pretrace_top24\eval_B_direct.csv
E:\Bridging\remote_sync\2026-05-19_stage2a_pretrace_top24\eval_D_visual_only.csv
E:\Bridging\remote_sync\2026-05-19_stage2a_pretrace_top24\logs_eval_B_direct.log
E:\Bridging\remote_sync\2026-05-19_stage2a_pretrace_top24\logs_eval_D_visual_only.log
```

生成的本地合并表：

```text
E:\Bridging\remote_sync\2026-05-19_stage2a_pretrace_top24\stage2a_pretrace_eval_merged_summary.csv
```

## 方法

对 `eval_B_direct.csv` 和 `eval_D_visual_only.csv` 做 per-sample 合并。

记录字段包括：

```text
sample_id
question
gold_answer
B_predicted_answer
D_predicted_answer
B_correct
D_correct
B_vqa_score
D_vqa_score
answer_changed_B_vs_D
B_format_ok
D_format_ok
```

这里的 `format_ok` 是严格检查 `generated_text` 是否以：

```text
The answer is
```

开头。

这个指标不等同于答案对错，而是用来判断 prompt 是否导致格式坍塌、解释性输出或 prefix 不稳定。

## 汇总结果

总体行为读数：

| prompt | n | correct | strict_gold | format_ok | empty |
|---|---:|---:|---:|---:|---:|
| B_direct | 24 | 8 | 7 | 24 | 0 |
| D_visual_only | 24 | 5 | 3 | 18 | 0 |

B/D 预测答案发生变化：

```text
13 / 24
```

直接解释：

```text
1. top24 没有 empty-answer 问题；
2. B_direct 的格式非常稳定；
3. D_visual_only 出现 6/24 的 format prefix violation；
4. D_visual_only 没有在行为准确率上优于 B_direct；
5. B/D 之间确实会改变输出倾向，但这种改变不能直接解释为 D 更 grounded。
```

## D_visual_only format violation 样本

以下 6 个样本在 `D_visual_only` 下没有严格以 `The answer is` 开头：

| sample_id | gold | D predicted answer |
|---|---|---|
| okvqa_val_1927165 | stop | `**Visual Evidence:**` |
| okvqa_val_2131565 | ceramic | `The plate is red and appears to be ceramic or porcelain` |
| okvqa_val_343215 | teddy | `a plush toy` |
| okvqa_val_4033335 | dessert | `Image of a dark, dense cake covered in a shiny dark sauce,` |
| okvqa_val_5735275 | 1 way | `Left` |
| okvqa_val_667695 | bitten | `The pretzel is broken and has a large chunk of cheese missing` |

这些样本并不一定全部排除，但后续做 prompt comparison 时必须标记。

尤其是：

```text
okvqa_val_1927165
```

它在 D 下直接输出 `**Visual Evidence:**`，属于明显 prompt-format failure，不适合作为 D prompt 行为优势的证据。

## B_direct correct 样本

按当前 `correct` 字段，B_direct 正确的样本是：

```text
okvqa_val_1083925
okvqa_val_1740705
okvqa_val_1985905
okvqa_val_1994425
okvqa_val_1996815
okvqa_val_2131565
okvqa_val_3313665
okvqa_val_3326275
```

## D_visual_only correct 样本

按当前 `correct` 字段，D_visual_only 正确的样本是：

```text
okvqa_val_1083925
okvqa_val_1740705
okvqa_val_1985905
okvqa_val_1994425
okvqa_val_1996815
```

## per-sample 行为表

| sample_id | gold | B answer | B correct | D answer | D correct | B/D changed |
|---|---|---|---:|---|---:|---:|
| okvqa_val_1083155 | dalmation | a German Dachshund | 0 | a spotted dachshund | 0 | True |
| okvqa_val_1083925 | donuts | donuts | 1 | donuts | 1 | False |
| okvqa_val_1740705 | teddy | Teddy | 1 | teddy bear | 1 | True |
| okvqa_val_1927165 | stop | to halt or cease movement | 0 | `**Visual Evidence:**` | 0 | True |
| okvqa_val_1985905 | chevy | Chevrolet | 1 | Chevrolet | 1 | False |
| okvqa_val_1994425 | wetsuit | wetsuit | 1 | wetsuit | 1 | False |
| okvqa_val_1996815 | microwave | microwave | 1 | a microwave | 1 | False |
| okvqa_val_2131565 | ceramic | ceramic | 1 | The plate is red and appears to be ceramic or porcelain | 0 | True |
| okvqa_val_2373185 | 8 | three | 0 | three | 0 | False |
| okvqa_val_2496585 | paint | Dulux paint | 0 | Dulux paint | 0 | False |
| okvqa_val_2683965 | octagon | oval | 0 | a shield | 0 | True |
| okvqa_val_2802115 | chainlink | chain-link fence | 0 | chain-link fence | 0 | False |
| okvqa_val_3265105 | toyota | Mitsubishi | 0 | Mitsubishi Outlander | 0 | True |
| okvqa_val_3313665 | penny farthing | a penny-farthing | 1 | a penny-farthing bicycle | 0 | True |
| okvqa_val_3326275 | wilson | Wilson | 1 | Yonex | 0 | True |
| okvqa_val_343215 | teddy | teddy bear | 0 | a plush toy | 0 | True |
| okvqa_val_3794755 | laptop | a laptop, monitor, and printer | 0 | a laptop, a desktop computer, and a tablet | 0 | True |
| okvqa_val_4033335 | dessert | Sticky toffee pudding | 0 | Image of a dark, dense cake covered in a shiny dark sauce, | 0 | True |
| okvqa_val_4549785 | honda | Yamaha | 0 | Yamaha | 0 | False |
| okvqa_val_5291225 | chinese | Japanese | 0 | Japanese | 0 | False |
| okvqa_val_5735275 | 1 way | left | 0 | Left | 0 | False |
| okvqa_val_602025 | sheep dog | Border Collie | 0 | Border Collie | 0 | False |
| okvqa_val_667695 | bitten | it's been bitten into | 0 | The pretzel is broken and has a large chunk of cheese missing | 0 | True |
| okvqa_val_80655 | bat | Swinging his bat | 0 | The baseball player is swinging a bat at a baseball | 0 | True |

## 预期与实际偏差

预期：

```text
top24 是 localized / strong image-dependence 候选，clean behavior 可能比随机样本好一些；
D_visual_only 可能会略微提高视觉证据使用，但不应作为主 claim。
```

实际：

```text
B_direct correct = 8/24
D_visual_only correct = 5/24
D_visual_only format_ok = 18/24
B/D changed = 13/24
```

偏差解释：

```text
1. top24 候选是为了寻找可复现实验样本，不是行为准确率 benchmark；
2. 一些样本虽然 gold correctness 为 0，但仍可能对 predicted-answer route 有机制价值；
3. D_visual_only 继续暴露 format / explanation-style 风险，因此不能作为“更好 prompt”来写；
4. 后续真正进入 replication pack 的样本必须同时看 clean-core trace、support source、nearest control 和 mask 可用性，而不能只凭 clean correctness。
```

## 当前结论

eval 侧支持以下保守判断：

```text
top24 可以继续进入 trace，因为没有 empty generation；
B_direct 行为格式更稳定；
D_visual_only 会改变输出倾向，但也更容易引入格式/解释性输出；
prompt 仍应被写成 route modulation / probing variable，而不是行为优劣比较对象。
```

这与 Stage 2 run plan 的口径一致：

```text
主目标是复现 evidence-region-sensitive support routes；
不是证明 D_visual_only 比 B_direct 更好。
```

## 对后续筛选的影响

后续构建 Stage 2A region replication candidates 时，建议优先级如下：

```text
第一优先级：
  trace 成功 + support source + nearest control + existing mask + clean answer/format 稳定

第二优先级：
  trace 成功 + support source + nearest control + existing mask，但 clean answer 不完全正确

谨慎使用：
  D_visual_only format violation 样本

默认排除或只作反例：
  D 输出明显崩成 explanation prefix 的样本，例如 okvqa_val_1927165
```

## 下一步

等待远端 answer-aligned trace 完成后，继续生成完整 pretrace readout：

```text
1. 同步 answer_aligned_meta_a/b 与 .pt graphs；
2. 汇总 B/D trace success count；
3. 标记 clean-core usable samples；
4. 运行 source zeroing intervention screen；
5. 构建 nearest node controls；
6. 选出真正进入 region-mask replication 的 10-15 个样本。
```
