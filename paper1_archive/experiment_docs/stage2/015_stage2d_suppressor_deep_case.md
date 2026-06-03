# Stage 2D：Suppressor Deep Case

日期：2026-05-20

## 1. 实验目的

Stage 2D 的目的，是把 suppressor route 从“有 mixed-sign 现象”推进到更可解释的 case-level 读法。

当前主结论仍然以 support route 为核心。Suppressor route 只作为 secondary，因为它可能对应多种机制：

1. 压制正确答案；
2. 支持竞争答案；
3. 与格式、语言先验或不确定性有关；
4. 被区域遮挡间接改变。

本实验要回答一个更窄的问题：

```text
在一个 suppressor-rich case 中，清零 suppressor source node 是否会让目标答案 token 在 first-token distribution 中变好？
```

如果是，这支持：

```text
suppressor source node 具有和 support source node 相反的符号方向。
```

但本实验不试图证明：

```text
suppressor 的统一机制已经解释清楚。
```

## 2. 术语解释

`suppressor node`：清零该 node 后，目标答案 logit 上升，说明该 node 原本可能在压制目标答案，或者支持竞争答案。

`rank damage`：干预后目标 token rank 变差的程度。对 support node 来说，正值通常表示清零伤害目标答案。对 suppressor node 来说，如果清零让目标答案变好，则 rank damage 会是负值或 0。

`delta target logit`：干预后目标答案 logit 减去 baseline 目标答案 logit。对 suppressor node 来说，正值表示清零 suppressor 后目标答案 logit 上升。

`source-minus-nearest rank gap`：`source_rank_damage - nearest_rank_damage`。在 suppressor 实验里，负值通常是好信号，因为它表示 source 清零比 nearest 清零更少伤害、甚至更能改善目标答案 rank。

## 3. 输入

主输入：

```text
E:\Bridging\annotation\stage2a_region_replication_top24_nearest8\region_experiment_manifest.csv
E:\Bridging\annotation\stage2a_region_replication_top24_nearest8\analysis_route\route_weakening_iou1.csv
E:\Bridging\annotation\stage2a_region_replication_top24_nearest8\analysis_behavior\behavior_wide_iou0p05.csv
```

远端 first-token suppressor 输出：

```text
E:\Bridging\annotation\stage2d_suppressor_deep_case\stage2d_first_token_suppressor_3794755.csv
E:\Bridging\annotation\stage2d_suppressor_deep_case\stage2d_first_token_suppressor_3794755.log
```

本地汇总输出：

```text
E:\Bridging\annotation\stage2d_suppressor_deep_case\stage2d_suppressor_case_summary.csv
E:\Bridging\annotation\stage2d_suppressor_deep_case\analysis_first_token_suppressor\
```

图像包：

```text
E:\Bridging\annotation\stage2d_suppressor_deep_case\images\okvqa_val_3794755\
```

## 4. 样本与候选

Stage 2A nearest8 里一共有 4 个 suppressor source-control pairs，全部来自同一个样本：

```text
sample_id: okvqa_val_3794755
question: What electronic devices are pictured here?
target token: laptop
reasoning_operation: visual_readout
image_dependence: strong
```

4 个 suppressor pair：

```text
013_okvqa_val_3794755_A_suppressor_L30_P293_F39206
014_okvqa_val_3794755_A_suppressor_L31_P293_F149640
016_okvqa_val_3794755_B_suppressor_L30_P288_F39206
017_okvqa_val_3794755_B_suppressor_L31_P288_F149640
```

其中：

```text
A = D_visual_only
B = B_direct
```

## 5. 方法

### 5.1 Region route readout

直接读取 Stage 2A 的 suppressor route weakening：

```text
support weakening 定义:
masked_delta - clean_delta

suppressor weakening 定义:
clean_delta - masked_delta
```

对 suppressor 来说，正的 weakening 表示：

```text
区域遮挡后，该 suppressor route 的“压制目标答案”作用变弱。
```

### 5.2 First-token suppressor bridge

远端运行：

```text
run_region_mask_node_first_token_smoke.py
--node-role suppressor
--conditions clean,answer_mask,union_mask
```

每个 pair 同时跑：

```text
source node
nearest_control node
```

输出目标：

```text
baseline target rank
intervention target rank
rank_damage_by_intervention
delta_target_logit
top1 token before/after
```

### 5.3 Greedy decoded-loop smoke

尝试对 suppressor 做 decoded-loop 小跑：

1. 先跑 `013 + 016` 两个 pair、三个条件；
2. 该轮超过 900 秒保护性超时；
3. 拉回 partial log 后发现已完成 7 行，但 CSV 因脚本结束前才写出，所以没有完整 CSV；
4. 再缩小到单 pair `016`、单条件 `union_mask`、`max_new_tokens = 4`；
5. 该轮返回 CSV，但出现 CUDA OOM。

因此 decoded-loop suppressor 目前只作为工程边界记录，不作为机制结论。

## 6. Route 结果

### 6.1 Pair 013

```text
pair: 013
prompt: D_visual_only
source node: L30/P293/F39206
nearest node: L11/P293/F148968

source clean delta = +0.625
source answer weakening = +0.5000
source union weakening = +0.4375
source answer minus random4 = +0.46875
source union minus random4 = +0.40625

source-minus-nearest answer weakening = +0.125
source-minus-nearest union weakening = -0.125
```

### 6.2 Pair 014

```text
pair: 014
prompt: D_visual_only
source node: L31/P293/F149640
nearest node: L27/P293/F25360

source clean delta = +0.875
source answer weakening = +0.8750
source union weakening = +0.8750
source answer minus random4 = +0.78125
source union minus random4 = +0.78125

source-minus-nearest answer weakening = +0.6875
source-minus-nearest union weakening = +0.6875
```

### 6.3 Pair 016

```text
pair: 016
prompt: B_direct
source node: L30/P288/F39206
nearest node: L11/P288/F148968

source clean delta = +0.625
source answer weakening = +0.6250
source union weakening = +0.5000
source answer minus random4 = +0.5625
source union minus random4 = +0.4375

source-minus-nearest answer weakening = +0.2500
source-minus-nearest union weakening = -0.2500
```

### 6.4 Pair 017

```text
pair: 017
prompt: B_direct
source node: L31/P288/F149640
nearest node: L27/P288/F25360

source clean delta = +1.000
source answer weakening = +1.0000
source union weakening = +1.0000
source answer minus random4 = +0.84375
source union minus random4 = +0.84375

source-minus-nearest answer weakening = +0.8750
source-minus-nearest union weakening = +0.8125
```

### 6.5 Route 小结

Route 层面，suppressor case 有清楚信号：

```text
F149640 pairs, especially 014 and 017, show strong evidence-region weakening and source > nearest.
F39206 pairs, 013 and 016, show source evidence sensitivity but source > nearest is weaker / condition-dependent.
```

这说明 suppressor route 不是完全噪声，但它比 support route 更适合写成 secondary。

## 7. First-token 结果

### 7.1 汇总

source-minus-nearest gap summary：

```text
condition    mean_rank_damage_gap    mean_delta_logit_gap
clean        -2.00                   +1.15625
answer_mask  -2.50                   +0.671875
union_mask   -4.75                   +0.87500
```

解释：

对 suppressor 来说：

```text
rank_damage_gap < 0
```

表示 source 清零比 nearest 清零更有利于目标 token rank。

同时：

```text
delta_logit_gap > 0
```

表示 source 清零比 nearest 清零更能提高目标 token logit。

所以 first-token 结果方向非常一致。

### 7.2 Pair 013

```text
clean:
source rank damage = -1
nearest rank damage = +2
rank gap = -3
source delta logit = +0.625
nearest delta logit = -0.625
delta logit gap = +1.25

answer_mask:
source rank damage = 0
nearest rank damage = +6
rank gap = -6
source delta logit = +0.125
nearest delta logit = -1.000
delta logit gap = +1.125

union_mask:
source rank damage = -3
nearest rank damage = +5
rank gap = -8
source delta logit = +0.1875
nearest delta logit = -1.1875
delta logit gap = +1.375
```

### 7.3 Pair 016

```text
clean:
source rank damage = 0
nearest rank damage = +3
rank gap = -3
source delta logit = +0.625
nearest delta logit = -1.000
delta logit gap = +1.625

answer_mask:
source rank damage = 0
nearest rank damage = +4
rank gap = -4
source delta logit = 0
nearest delta logit = -1.375
delta logit gap = +1.375

union_mask:
source rank damage = -3
nearest rank damage = +3
rank gap = -6
source delta logit = +0.125
nearest delta logit = -1.750
delta logit gap = +1.875
```

### 7.4 Top-1 token

大多数 source suppressor 清零没有改变 top-1 token。

但 nearest control 在 B 条件下有更明显的不稳定：

```text
pair 016 clean nearest:
laptop -> a

pair 016 union nearest:
television -> a
```

这说明 first-token suppressor bridge 更应该写成 target-token rank/logit signal，而不是 top-1 answer switching signal。

## 8. Decoded-loop 尝试

### 8.1 两 pair decoded smoke

尝试：

```text
pair_ids = 013,016
conditions = clean,answer_mask,union_mask
max_new_tokens = 8
```

第一次结果：

```text
超过 900 秒保护性超时。
```

partial log 显示已完成 7 行：

```text
013 A source clean: unchanged
013 A source answer_mask: unchanged
013 A source union_mask: unchanged
013 A nearest clean: unchanged
013 A nearest answer_mask: unchanged
013 A nearest union_mask: unchanged
016 B source clean: unchanged
```

因为脚本只在全部结束后写 CSV，所以没有完整结构化 CSV。该结果只能作为 diagnostic，不能作为正式 decoded 结论。

### 8.2 单 pair union4 decoded smoke

尝试：

```text
pair_id = 016
condition = union_mask
max_new_tokens = 4
```

结果：

```text
返回 CSV，但 source 和 nearest 两行均 CUDA OOM。
```

错误类型：

```text
OutOfMemoryError: CUDA out of memory
```

解释：

这是工程资源限制，不是机制失败。first-token suppressor bridge 已经能跑通；decoded-loop suppressor 需要进一步优化内存，例如：

1. 每个 node_source 单独启动进程；
2. 生成长度降到 1-2 token；
3. 在 decoded loop 中缓存 baseline；
4. 清理模型后再跑；
5. 或专门写一个 suppressor-only minimal decoded probe。

### 8.3 错误原因与脚本修复

复查后，错误分成两类：

```text
1. OOM 直接原因：
   OOM 日志显示 GPU 上已有一个占用约 33GB 的 Python 进程，同时当前 decoded probe 又加载了约 13GB，因此剩余显存不足。

2. 结果丢失原因：
   decoded 脚本原本只在全部 rows 结束后统一写 CSV。两 pair decoded smoke 虽然 partial log 已经显示完成 7 行，但因为 900s 超时发生在脚本写 CSV 之前，所以没有结构化 CSV。
```

对应修复：

```text
script:
E:\Bridging\vlm-circuit-tracing\circuit_tracer_vlm\scripts\research\run_region_mask_node_greedy_decode_smoke.py

新增:
--stream-output
--node-source source|nearest_control
--max-rows

同时在每个 condition 后调用 torch.cuda.empty_cache()
```

修复后本地语法检查通过：

```text
python -m py_compile E:\Bridging\vlm-circuit-tracing\circuit_tracer_vlm\scripts\research\run_region_mask_node_greedy_decode_smoke.py
```

### 8.4 修复后重跑结果

修复后重新跑最小 split decoded probe：

```text
pair_id = 016_okvqa_val_3794755_B_suppressor_L30_P288_F39206
condition = union_mask
max_new_tokens = 4
node_source = source, nearest_control 分开跑
stream_output = true
PYTORCH_CUDA_ALLOC_CONF = expandable_segments:True
```

输出：

```text
E:\Bridging\annotation\stage2d_suppressor_deep_case\stage2d_greedy_decode_suppressor_pair016_union4_source_stream.csv
E:\Bridging\annotation\stage2d_suppressor_deep_case\stage2d_greedy_decode_suppressor_pair016_union4_nearest_control_stream.csv
```

结果：

```text
source suppressor:
baseline answer = television
intervention answer = television
answer_changed_by_intervention = False

nearest control:
baseline answer = television
intervention answer = a television
answer_changed_by_intervention = True
```

解释：

```text
修复后 decoded probe 已经可以完成并写出 CSV；
但在这个最小 decoded case 中，source suppressor 清零没有改变 decoded short answer；
nearest control 的变化只是 television -> a television，更像格式 / article variation，不是实质答案改变。
```

因此 decoded-loop suppressor bridge 的判定从：

```text
工程没跑通
```

更新为：

```text
工程最小 probe 已跑通；但目前没有 source-specific decoded-answer positive evidence。
```

## 9. 预期与实际偏差

### 9.1 符合预期

First-token 方向符合 suppressor 定义：

```text
source suppressor zeroing tends to improve target token rank/logit;
nearest-control zeroing often worsens target token rank/logit.
```

这和 support node 的方向相反，说明 signed route 的符号读法是有意义的。

### 9.2 偏离预期

Decoded-loop 没有达到预期。最初是工程限制，修复后最小 probe 已能跑通，但结果仍然不支持 source-specific decoded-answer change：

```text
full two-pair run timed out;
single-pair union run initially hit CUDA OOM;
after stream-output + split node_source fix, pair016/union4 completed;
source: television -> television;
nearest: television -> a television.
```

因此不能说 suppressor source node 已经被证明会改变 decoded answer。

## 10. 结论

Stage 2D 的正式判定：

```text
suppressor first-token bridge positive;
decoded-loop suppressor bridge minimally runnable but not positive.
```

中文：

```text
在 okvqa_val_3794755 这个 suppressor-rich case 中，source suppressor 清零在 first-token 分布上表现出方向一致的目标 token 改善，且强于 nearest control；修复后 decoded 最小 probe 已经跑通，但 source 清零没有改变 decoded answer。因此 suppressor 当前仍只能作为 secondary first-token evidence。
```

## 11. 对主 claim 的影响

加强：

```text
signed route 的 support / suppressor 区分不是任意标签；suppressor 清零在 first-token 层面有相反方向的 target-token effect。
```

不升级：

```text
suppressor 机制已经解释清楚。
suppressor source node 能稳定改变自然生成答案。
suppressor 是主证据。
```

建议主文案：

```text
Suppressor routes also recur and show signed first-token effects, but their decoded-answer role remains more heterogeneous and is treated as secondary.
```

## 12. 后续动作

如果继续 Stage 2D，建议不要扩大样本，只做极小验证：

1. 写一个 single-row decoded probe，避免一次加载后连续跑多行；
2. 对 `013` 和 `016` 分开跑，继续使用 `--stream-output` 避免超时后丢 CSV；
3. 只生成 1-2 token，先验证 decoded first step；
4. 如果仍无 source-specific decoded change，就停止 decoded suppressor，把 Stage 2D 固定为 first-token secondary evidence。
