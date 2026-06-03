# Stage 2G-5 Run Plan：Mask-to-Clean Feature Restoration Smoke

## Summary

Stage 2G dose/sign probe 的结论是：

```text
Qwen/LLaVA 的 evidence-region-sensitive readout 成立；
但直接在 clean hidden 上 subtract/add evidence feature decoder direction，
没有证明 primary ablation causal replication。
```

因此下一步不应该继续盲目扩大 readout，也不应该直接进入 decoded answer bridge。

更合理的下一步是：

```text
在 union_mask input 上，把 clean image 相比 union_mask 丢失的 evidence feature contribution patch 回去；
检查 target answer logit/rank 是否恢复，并与 control feature restoration 比较。
```

这个实验叫：

```text
mask-to-clean feature restoration smoke
```

它比上一轮 subtract/add 更贴近现有发现，因为它直接利用了：

```text
区域遮挡实际造成的 feature drop。
```

## 为什么需要这个实验

上一轮 `feature-direction dose/sign probe` 暴露出一个问题：

```text
readout feature 对 evidence mask 敏感，
不代表直接 subtract clean feature direction 就一定会损伤答案。
```

可能原因包括：

```text
1. 选出的 feature 是 readout-sensitive feature，不一定是 answer-support source route；
2. CLT decoder direction patch 是近似干预，不等同于 Gemma ReplacementModel 的 source node zeroing；
3. feature direction 的符号和层内残差几何可能不适合直接用 subtract/add 解读；
4. clean input 上强行加减 feature direction，可能制造 out-of-distribution hidden state。
```

因此 Stage 2G-5 改成更自然的 counterfactual：

```text
既然 union_mask 会让某些 evidence feature activation 下降，
那把这些下降的 contribution 恢复到 union_mask forward 中，
是否能恢复 target answer？
```

## 主问题

```text
Qwen layer 26 和 LLaVA layer 15 中，
被 union_mask 削弱的 evidence-sensitive features，
是否对 target answer rank/logit 具有可恢复的行为影响？
```

## 模型与样本

模型优先级保持 Stage 2G：

```text
1. Qwen2.5-VL-7B-Instruct
   CLT = KokosDev/qwen2p5vl-7b-clt
   layer = 26
   bucket = image_marker_or_span

2. LLaVA-1.5-7B
   CLT-like asset = KokosDev/llava15-7b-clt
   layer = 15
   bucket = image_token_span
```

样本固定复用：

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

不新增标注。

## 条件设计

每个 `sample_id × prompt × model` 跑：

```text
clean
union_mask
union_mask + evidence_restore_top1_s0.5
union_mask + evidence_restore_top1_s1.0
union_mask + evidence_restore_top1_s1.5
union_mask + evidence_restore_topk_s0.5
union_mask + evidence_restore_topk_s1.0
union_mask + evidence_restore_topk_s1.5
union_mask + control_restore_topk_s0.5
union_mask + control_restore_topk_s1.0
union_mask + control_restore_topk_s1.5
```

其中：

```text
evidence_restore:
  使用 clean - union_mask drop 最大的 evidence-sensitive features。

control_restore:
  使用 clean activation 相近、但 mask drop 较小的 control features。
```

## Patch 定义

对于 clean 和 union_mask forward：

```text
clean_features = encoder(clean_hidden)
union_features = encoder(union_hidden)
feature_drop = clean_features - union_features
```

在 union_mask forward 的目标层输出上 patch：

```text
restore_patch = scale × feature_drop[evidence_feature_ids] × decoder_vectors[evidence_feature_ids]
union_hidden_restored = union_hidden + restore_patch
```

control 条件同理，只是换成 control feature ids。

注意：

```text
这个 patch 是 restoration，不是 ablation。
它测试的是“把 evidence mask 损失的 feature contribution 补回去，是否能恢复答案信号”。
```

## 指标

基础指标：

```text
target_logit
target_rank
target_prob
top1_token
top1_changed_vs_union
```

恢复指标：

```text
logit_restore = target_logit(restored) - target_logit(union_mask)
rank_restore = target_rank(union_mask) - target_rank(restored)
```

其中：

```text
logit_restore > 0 表示 target logit 被恢复；
rank_restore > 0 表示 target rank 变好。
```

gap closure：

```text
clean_to_union_logit_gap = target_logit(clean) - target_logit(union_mask)
logit_gap_closure = logit_restore / clean_to_union_logit_gap
```

如果 `clean_to_union_logit_gap` 很小或方向为负，则该 run 只做描述，不参与 strongest evidence。

specificity：

```text
evidence_restore - control_restore
```

分别看：

```text
rank_restore difference
logit_restore difference
gap_closure difference
```

## 成功标准

primary success：

```text
1. evidence_restore_topk_s1.0 或 s1.5 在至少 2/3 case 上提升 target logit 或 target rank；
2. evidence_restore 的平均 restore effect 大于 control_restore；
3. 至少一个模型满足上述标准；
4. 没有明显 NaN、<s> top1 collapse 或格式异常主导结果。
```

secondary success：

```text
1. Qwen 或 LLaVA 只有 logit 恢复，没有 rank 恢复；
2. 只在 B_direct 或 D_visual_only 中成立；
3. top1 不变，但 target rank/logit 方向一致改善。
```

失败标准：

```text
1. evidence_restore 不优于 control_restore；
2. restoration 方向和 clean-union gap 无关；
3. 大量 NaN / special token collapse；
4. 只有单个 prompt 或单个样本出现不可复现正结果。
```

## 结果解释规则

如果成功，只能写：

```text
Qwen/LLaVA show a cross-model feature-level causal bridge:
restoring evidence-mask-dropped feature contributions partially restores target answer rank/logit.
```

仍然不能写：

```text
Qwen/LLaVA 已经复现 Gemma source-control causal route；
因为这还不是 source tracing，也不是 nearest matched source-control。
```

如果失败，则写：

```text
Qwen/LLaVA support readout-level cross-model evidence,
but current feature-level intervention/restoration probes do not establish causal answer-route replication.
```

## 预期产物

脚本建议路径：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_cross_model_feature_restoration_smoke.py
scripts/local/run_stage2g_feature_restoration_remote.py
```

输出：

```text
doc/experiments/stage2/cross_model/stage2g_qwen_feature_restoration_smoke.json
doc/experiments/stage2/cross_model/stage2g_qwen_feature_restoration_smoke.csv
doc/experiments/stage2/cross_model/stage2g_llava_feature_restoration_smoke.json
doc/experiments/stage2/cross_model/stage2g_llava_feature_restoration_smoke.csv
doc/experiments/stage2/cross_model/stage2g_feature_restoration_summary.csv
doc/experiments/stage2/032_stage2g_mask_to_clean_feature_restoration_smoke.md
```

## 对主线的意义

Stage 2G-5 是 cross-model causal bridge 的下一块拼图：

```text
Stage 2F:
  Qwen/LLaVA readout 对 evidence mask 敏感。

Stage 2G dose/sign:
  直接 feature-direction ablation 没有建立 causal replication。

Stage 2G-5:
  测试 evidence mask 实际损失的 feature contribution 是否能恢复 target answer signal。
```

如果 Stage 2G-5 成功，cross-model 证据可以从：

```text
readout replication only
```

升级到：

```text
feature-level causal bridge
```

但仍低于 Gemma 的：

```text
source-control causal route replication
```
