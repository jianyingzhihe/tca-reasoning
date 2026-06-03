# 实验 033：Stage 2G Hidden-State Clean Patch Upper Bound

## 目的

实验 032 的结果说明：

```text
union_mask 会强烈损伤 Qwen/LLaVA 的 target answer logit/rank；
但 CLT feature-level evidence restore 没有稳定恢复 target signal。
```

这留下一个关键问题：

```text
失败是因为当前 layer/bucket 本身不包含可恢复的答案行为信号，
还是因为 CLT feature selection / decoder patch 太弱？
```

因此本实验做一个更强的上界诊断：

```text
直接在 union_mask forward 中，
把目标层、目标 bucket 的 hidden state 从 union hidden 拉回 clean hidden。
```

如果这个 hidden-state patch 能恢复 target answer，那么说明：

```text
该 layer/bucket 确实包含可恢复的行为相关信息；
之前 feature-level restoration 失败更可能是 feature 子空间、decoder direction 或 source selection 没打通。
```

如果 hidden-state patch 也失败，则说明：

```text
当前 layer/bucket 不是足够的行为因果位置。
```

## 输入

模型：

```text
Qwen2.5-VL-7B-Instruct:
  layer = 26
  bucket = image_marker_or_span

LLaVA-1.5-7B:
  layer = 15
  bucket = image_token_span
```

样本：

```text
okvqa_val_2847255
okvqa_val_4157235
okvqa_val_3658865
```

prompt：

```text
B_direct
D_visual_only
```

每个模型：

```text
3 samples × 2 prompts = 6 usable runs
```

## 输出

原始输出：

```text
doc/experiments/stage2/cross_model/stage2g_qwen_hidden_patch_smoke.json
doc/experiments/stage2/cross_model/stage2g_qwen_hidden_patch_smoke.csv
doc/experiments/stage2/cross_model/stage2g_llava_hidden_patch_smoke.json
doc/experiments/stage2/cross_model/stage2g_llava_hidden_patch_smoke.csv
```

分析输出：

```text
doc/experiments/stage2/cross_model/stage2g_hidden_patch_baseline_gap.csv
doc/experiments/stage2/cross_model/stage2g_hidden_patch_summary.csv
doc/experiments/stage2/cross_model/stage2g_hidden_patch_case_table.csv
doc/experiments/stage2/cross_model/stage2g_hidden_patch_decision.json
```

脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_cross_model_hidden_patch_smoke.py
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/analyze_stage2g_hidden_patch_smoke.py
scripts/local/run_stage2g_hidden_patch_remote.py
```

## 方法

### 1. 计算 clean 与 union hidden

对每个 `sample_id × prompt`：

```text
clean_outputs = model(clean_image)
union_outputs = model(union_mask_image)

clean_hidden = clean_outputs.hidden_states[target_layer]
union_hidden = union_outputs.hidden_states[target_layer]
```

### 2. 目标 bucket

Qwen：

```text
image_marker_or_span
```

LLaVA：

```text
image_token_span
```

### 3. hidden patch

在 union_mask forward 的目标层输出上 patch：

```text
patch = clean_hidden[:, bucket_positions, :] - union_hidden[:, bucket_positions, :]
union_hidden_patched[:, bucket_positions, :] = union_hidden[:, bucket_positions, :] + scale × patch
```

scale：

```text
0.5
1.0
1.5
```

`scale = 1.0` 表示：

```text
把 union_mask 的该 bucket hidden state 完整替换为 clean hidden state。
```

### 4. 指标

```text
target_logit
target_rank
logit_restore_vs_union
rank_restore_vs_union
logit_gap_closure
top1_changed_vs_union
```

其中：

```text
logit_restore_vs_union = target_logit(patched) - target_logit(union_mask)
rank_restore_vs_union = target_rank(union_mask) - target_rank(patched)
```

正值表示恢复。

## 结果一：union_mask baseline 损伤

这部分与实验 032 一致：

| 模型 | n runs | mean clean-union logit gap | positive logit gap | mean clean-union rank gap | positive rank gap | clean→union top1 changed |
|---|---:|---:|---:|---:|---:|---:|
| Qwen layer 26 | `6` | `+3.9844` | `6/6` | `+90.1667` | `4/6` | `4/6` |
| LLaVA layer 15 | `6` | `+2.9492` | `6/6` | `+54.0000` | `6/6` | `0/6` |

说明：

```text
evidence mask 对两个模型都有明确行为损伤。
```

## 结果二：Qwen hidden-state patch

| 条件 | mean logit restore | positive logit restore | mean rank restore | positive rank restore | mean logit gap closure | top1 changed |
|---|---:|---:|---:|---:|---:|---:|
| hidden_bucket_restore_s0.5 | `+0.0781` | `4/6` | `+2.8333` | `2/6` | `+0.0898` | `0/6` |
| hidden_bucket_restore_s1.0 | `+0.7656` | `5/6` | `+58.5000` | `4/6` | `+0.2004` | `1/6` |
| hidden_bucket_restore_s1.5 | `+1.6510` | `4/6` | `+81.0000` | `4/6` | `+0.2311` | `2/6` |

关键 case：

```text
okvqa_val_4157235 / B_direct:
  union rank = 272
  hidden s1.0 patched rank = 92
  rank restore = +180
  logit restore = +1.8125

okvqa_val_4157235 / D_visual_only:
  union rank = 271
  hidden s1.0 patched rank = 103
  rank restore = +168
  logit restore = +1.4688

okvqa_val_3658865 / D_visual_only:
  union rank = 2
  hidden s1.0 patched rank = 1
  rank restore = +1
  top1 changes back to Samsung
```

Qwen 判定：

```text
supported_hidden_state_upper_bound
```

解释：

```text
Qwen layer 26 image bucket 的 hidden state 中确实包含可恢复 target answer signal；
但是 CLT feature-level restore 没能稳定提取出足够的因果子空间。
```

## 结果三：LLaVA hidden-state patch

| 条件 | mean logit restore | positive logit restore | mean rank restore | positive rank restore | mean logit gap closure | top1 changed |
|---|---:|---:|---:|---:|---:|---:|
| hidden_bucket_restore_s0.5 | `+0.3216` | `4/6` | `+23.5000` | `3/6` | `+0.0375` | `0/6` |
| hidden_bucket_restore_s1.0 | `+2.4128` | `6/6` | `+50.5000` | `4/6` | `+0.6669` | `0/6` |
| hidden_bucket_restore_s1.5 | `NaN` | `0/6` | `+56.3333` | `6/6` | `NaN` | `6/6` |

注意：

```text
LLaVA s1.5 出现 NaN / top1 collapse，因此不能作为 strongest evidence。
主读数应使用 s1.0。
```

关键 case：

```text
okvqa_val_4157235 / B_direct:
  union rank = 100
  hidden s1.0 patched rank = 8
  rank restore = +92
  logit restore = +5.875
  gap closure ≈ 0.9412

okvqa_val_4157235 / D_visual_only:
  union rank = 208
  hidden s1.0 patched rank = 8
  rank restore = +200
  logit restore = +6.8281
  gap closure ≈ 0.9348

okvqa_val_3658865 / D_visual_only:
  union rank = 15
  hidden s1.0 patched rank = 8
  rank restore = +7
  logit restore = +0.7422
  gap closure ≈ 1.0215
```

LLaVA 判定：

```text
supported_hidden_state_upper_bound
```

解释：

```text
LLaVA layer 15 image bucket 的 hidden state 也包含可恢复 target answer signal；
而且 s1.0 的平均 logit gap closure 很高，约 0.6669。
```

## 与实验 032 的关系

实验 032：

```text
CLT feature-level restoration 不成功。
```

实验 033：

```text
hidden-state bucket-level restoration 成功。
```

这两者合起来说明：

```text
跨模型不是完全没有行为因果桥；
Qwen/LLaVA 的目标层 image bucket hidden state 确实携带可恢复的答案信号；
但当前 CLT top-drop feature selection + decoder-vector patch 没能把这个信号分解成稳定的 feature-level causal route。
```

这是一条很重要的诊断结论。

## 对 cross-model claim 的更新

现在可以比实验 032 稍微更进一步，但仍不能写强 causal route replication。

可以写：

```text
In Qwen and LLaVA, evidence masks damage target-answer behavior,
and patching the corresponding clean image-bucket hidden states can partially restore target logits/ranks.
However, current CLT feature-level interventions/restorations do not yet isolate source-control causal routes.
```

中文：

```text
在 Qwen 和 LLaVA 中，关键证据区域遮挡会损伤目标答案行为；
把对应层的 clean image-bucket hidden state patch 回 masked run，可以部分恢复 target logit/rank。
但当前 CLT feature-level 干预/恢复还没有隔离出 source-control causal routes。
```

## 保守边界

不能写：

```text
Qwen/LLaVA 已经复现 Gemma source route。
Qwen/LLaVA source nodes beat matched controls。
CLT evidence features 就是 causal support features。
hidden patch 证明了对象级语义节点。
```

可以写：

```text
Qwen/LLaVA 支持 cross-model hidden-state-level causal bridge upper bound。
feature-level causal specificity 仍未证明。
Gemma3 仍是唯一拥有完整 source/control causal route chain 的主模型。
```

## 下一步

如果继续跨模型，下一步不应继续扩大 feature restore，而应做更精细的分解：

```text
1. bucket 内 position ablation / patch：
   找出 image bucket 中哪些 positions 对恢复贡献最大。

2. patch answer-adjacent text positions：
   image bucket patch 有上界效果，但 final answer 也可能经过 assistant prefix / last prompt token 汇聚。

3. Qwen/LLaVA source tracing adapter：
   真正从 answer-adjacent target token 回溯 source nodes，
   而不是只按 clean-union feature drop 选择 features。
```

当前最重要的写法是：

```text
cross-model readout + hidden-state patch bridge supported;
feature-level / source-control causal route replication not yet supported.
```
