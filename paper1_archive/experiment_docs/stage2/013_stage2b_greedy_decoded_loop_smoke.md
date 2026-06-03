# 实验 013：Stage 2B greedy decoded loop smoke

## 目的

Stage 2B 的目标是把机制链从：

```text
node intervention -> target logit / first-token distribution
```

进一步推进到：

```text
node intervention -> decoded short answer
```

实验 011 和 012 已经证明 first-token bridge 可行，而且在 Stage 2A nearest8 support pairs 上有方向性，尤其 `union_mask` 下 source-minus-nearest gap 最稳定。但 first-token 分布变化是否真的会进入完整短答案生成，仍然没有证明。

本实验做一个小而深的 decoded-loop smoke：

```text
不用旧 feature_intervention_generate；
改用手写 greedy loop；
先只跑两个 case；
判断 source zeroing 是否能改变 decoded answer，并与 nearest-control 对照。
```

---

## 为什么需要手写 decoded loop

旧的生成路径：

```text
model.feature_intervention_generate(batch, ...)
```

失败原因：

```text
HookedVLTransformer.generate 只接受 torch.Tensor 或 str；
当前 VLM 输入是 batch dict，包含 input_ids / attention_mask / image。
```

具体错误：

```text
AssertionError:
File ".../HookedVLTransformer.py", line 2370, in generate
    assert isinstance(tokens, torch.Tensor)
```

所以本实验绕开 `generate` helper，手写一个极简 greedy loop：

```text
每一步 forward_from_batch / feature_intervention
取 next-token argmax
append 到 input_ids / attention_mask
重复若干步
```

这不是最高效的生成方式，但最适合当前 VLM hook pipeline。

---

## 输入

远端 manifest：

```text
/root/autodl-tmp/tca-reasoning/annotation/stage2a_region_replication_top24_nearest8/region_experiment_manifest_remote.csv
```

mask root：

```text
/root/autodl-tmp/tca-reasoning/annotation/stage2a_region_replication_top24_nearest8/exported_masks
```

选择的 case：

| sample_id | pair_id | 选择原因 |
|---|---|---|
| `okvqa_val_2683965` | `010_okvqa_val_2683965_B_support_L26_P287_F39687` | first-token source-specific rank damage 最强，是 decoded-loop 的第一候选 |
| `okvqa_val_1927165` | `003_okvqa_val_1927165_B_support_L11_P196_F151858` | first-token 中 source top1 会改变而 nearest 不变，适合测试 decoded answer change |

---

## 输出

本地同步结果：

```text
E:\Bridging\annotation\stage2b_node_generation_smoke\stage2b_greedy_decode_2683965_pair010.csv
E:\Bridging\annotation\stage2b_node_generation_smoke\stage2b_greedy_decode_2683965_pair010.log
E:\Bridging\annotation\stage2b_node_generation_smoke\stage2b_greedy_decode_1927165_pair003.csv
E:\Bridging\annotation\stage2b_node_generation_smoke\stage2b_greedy_decode_1927165_pair003.log
```

新增脚本：

```text
E:\Bridging\vlm-circuit-tracing\circuit_tracer_vlm\scripts\research\run_region_mask_node_greedy_decode_smoke.py
```

---

## 方法

### 1. 生成条件

每个 case 跑：

```text
clean
answer_mask
union_mask
```

每个条件跑：

```text
source zeroing
nearest-control zeroing
```

每行同时记录：

```text
baseline greedy answer
intervention greedy answer
answer_changed_by_intervention
```

### 2. Greedy loop

伪代码：

```python
cur = batch
generated = []
for step in range(max_new_tokens):
    if intervention is None:
        logits = model.forward_from_batch(cur)
    else:
        logits, _ = model.feature_intervention(
            cur,
            [(layer, pos, feature_id, 0.0)],
            freeze_attention=True,
            apply_activation_function=True,
        )
    next_id = argmax(logits[0, -1, :])
    generated.append(next_id)
    cur.input_ids = concat(cur.input_ids, next_id)
    cur.attention_mask = concat(cur.attention_mask, 1)
```

当前设置：

```text
max_new_tokens = 6
decoding = greedy argmax
intervention = 每一步都在原始 feature_pos 上清零同一个 feature
```

注意：

```text
这是 smoke，不是最终 generation implementation。
每一步都重新跑 full multimodal forward，因此慢但稳定。
```

---

## 运行完整性

两个 case 都完整跑完：

```text
okvqa_val_2683965 rows = 6
okvqa_val_1927165 rows = 6
error rows = 0
```

说明：

```text
手写 decoded loop 可以在当前 ReplacementModel / VLM batch / feature_intervention 路径上工作。
```

---

## 结果一：`okvqa_val_2683965 / pair010`

### 输出表

| node_source | condition | baseline answer | intervention answer | changed |
|---|---|---|---|---|
| source | clean | `oval` | `oval` | false |
| source | answer_mask | `square` | `square` | false |
| source | union_mask | `rectangle` | `rectangle` | false |
| nearest_control | clean | `oval` | `oval` | false |
| nearest_control | answer_mask | `square` | `rectangle` | true |
| nearest_control | union_mask | `rectangle` | `rectangle` | false |

### 读法

这个结果有点反直觉，但很重要。

在实验 012 的 first-token 结果中，`okvqa_val_2683965 / pair010` 是最强 source-specific rank-damage case：

```text
answer_mask:
source rank_damage = +96
nearest rank_damage = -107
```

但 decoded loop 中：

```text
source zeroing 没有改变最终短答案；
nearest-control zeroing 反而在 answer_mask 下把答案从 square 改成 rectangle。
```

这说明：

```text
first-token target-rank damage 不必然等于 decoded-answer change。
```

可能原因包括：

```text
1. target token 是 oval，但 answer_mask 下 greedy baseline 已经是 square；
2. source zeroing 伤害 target token rank，但没有改变当前 top1 token；
3. nearest-control 虽然不是 traced source，但可能改变竞争 token structure；
4. decoded answer 是 argmax 序列，只有当 top1 或后续 token 改变时才表现为字符串变化。
```

### 结论

`okvqa_val_2683965` 不适合作为 source-to-decoded-answer positive case。

它更适合作为：

```text
first-token bridge 与 decoded-answer bridge 不完全等价的 dissociation case。
```

---

## 结果二：`okvqa_val_1927165 / pair003`

### 输出表

| node_source | condition | baseline answer | intervention answer | changed |
|---|---|---|---|---|
| source | clean | `to halt movement` | `to halt or cease movement` | true |
| source | answer_mask | `a sign` | `stop` | true |
| source | union_mask | `stop` | `stop` | false |
| nearest_control | clean | `to halt movement` | `to halt movement` | false |
| nearest_control | answer_mask | `a sign` | `a sign` | false |
| nearest_control | union_mask | `stop` | `stop` | false |

### 读法

这个 case 是当前最好的 decoded-loop positive case。

source zeroing：

```text
clean:
to halt movement -> to halt or cease movement

answer_mask:
a sign -> stop

union_mask:
stop -> stop
```

nearest-control zeroing：

```text
三个条件都不改变答案。
```

这说明：

```text
source node intervention 可以进入 decoded short-answer generation；
而且这个改变不是 nearest-control 同样能造成的。
```

但要保守：

```text
clean 下变化是语义近似表达变化；
answer_mask 下从 a sign 到 stop 更有机制意义；
union_mask baseline 已经是 stop，所以没有变化空间。
```

### 结论

`okvqa_val_1927165` 可以作为：

```text
node-to-decoded-answer positive case
```

但不是强统计结论，只是 case-level bridge evidence。

---

## 预期与实际偏差

### 原预期

根据 first-token 结果，最期待的是：

```text
okvqa_val_2683965 会成为最强 decoded-loop positive case。
```

### 实际

实际相反：

```text
2683965 source 不改变 decoded answer；
1927165 source 改变 decoded answer，而且 nearest-control 不改变。
```

这说明：

```text
first-token rank/logit damage 和 decoded answer change 有联系，但不是一一对应。
```

---

## 结论

实验 013 的正式判定：

```text
decoded-loop node intervention is technically feasible;
source-specific decoded answer change exists in at least one case;
but decoded-answer effects are heterogeneous and do not follow first-token rank damage monotonically.
```

中文：

```text
手写 decoded loop 工程上可行；至少一个 case 中 source node 清零能特异性改变短答案生成；但 decoded-answer 效应异质，不能从 first-token rank damage 单调推出。
```

---

## 对主 claim 的影响

这一步补强了主 claim 中“路径变化与行为有关”的部分。

现在可以更有底气地说：

```text
In selected cases, source-node interventions can propagate into decoded short-answer generation.
```

中文：

```text
在特定 case 中，source node 干预可以传递到短答案自然生成。
```

但仍不能写：

```text
source-node zeroing reliably changes decoded answers across the pack.
```

也不能写：

```text
first-token damage is sufficient to predict decoded answer changes.
```

---

## 最适合进入报告的表述

建议写成：

```text
As a case-level bridge to generation, we implemented a greedy decoded loop that applies feature zeroing during short-answer generation. In okvqa_val_1927165, source-node zeroing changed the decoded answer under clean and answer-mask conditions, while the matched nearest-control node did not. However, in okvqa_val_2683965, strong first-token rank damage did not translate into a source-specific decoded-answer change. This indicates that node-to-generation effects are real but heterogeneous, and that first-token distributional damage is not sufficient to predict full decoded-answer changes.
```

中文：

```text
作为生成侧桥接个案，我们实现了一个手写 greedy decoded loop，在短答案生成过程中施加 feature zeroing。对于 okvqa_val_1927165，source node 清零在 clean 和 answer_mask 条件下改变了 decoded answer，而 nearest-control 不改变；但对于 okvqa_val_2683965，强 first-token rank damage 并没有转化为 source-specific decoded-answer change。这说明 node-to-generation 效应是真实存在的，但具有异质性，first-token 分布损伤不足以单独预测完整 decoded answer 变化。
```

---

## 后续动作

建议下一步：

```text
1. 不再大规模跑 decoded loop，避免工程成本吞掉主线；
2. 把 1927165 放入 case-study 候选；
3. 把 2683965 放入 dissociation / boundary case；
4. 如果还要补，只跑 2 个 case：
   - 1740705：first-token source gap 稳，可能有 decoded change；
   - 80655：union source gap 稳，但 answer_mask 反向，可测试异质性。
```

---

## 一句话总结

实验 013 证明 decoded-loop node intervention 在工程上可行，并发现一个 source-specific decoded-answer positive case：`okvqa_val_1927165`。但 `okvqa_val_2683965` 显示 first-token rank damage 不一定改变 decoded answer，因此 node-to-generation 目前只能作为 case-level bridge，而不是主统计结论。
