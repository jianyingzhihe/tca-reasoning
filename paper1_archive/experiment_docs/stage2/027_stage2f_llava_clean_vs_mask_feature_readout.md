# 实验 027：Stage 2F LLaVA Clean vs Evidence-Mask Feature Readout

日期：2026-05-20

## 1. 实验目的

本实验把 LLaVA 支线从“能做单张图 CLT feature readout”推进到“能复用现有人工 evidence mask 做 clean vs answer/union mask readout”。

核心问题是：

```text
在 LLaVA / Llama-family VLM 中，遮挡人工标注的关键证据区域后，image token span 上的 public CLT-like feature activation 是否发生系统性变化？
```

边界必须说清楚：

```text
这是 readout-level evidence-region sensitivity；
不是 attribution graph；
不是 node intervention；
不是 source route tracing；
没有 matched node control；
不能写成跨模型 causal mechanism replication。
```

## 2. 专有名词解释

```text
answer_mask：
人工标注的最小答案证据区域。例如牌子文字、目标物体、关键局部。

union_mask：
answer_mask 与 relate_mask 的并集，表示答案证据区域加相关上下文。

mean_topk_drop：
在 clean 图像中选 top-k features，再比较 mask 后这些 feature 的 activation。计算 clean_activation - masked_activation。正值表示遮挡后该 bucket 的 clean top feature 被削弱。

topk_jaccard_change：
clean top-k feature 集合和 masked top-k feature 集合的变化程度。越大表示 top feature identity 越不稳定。

bucket_mean_shift：
bucket 内全部 feature activation 的均值变化。它不是主指标，只帮助判断整体分布是否移动。
```

## 3. 输入

模型：

```text
base model = /root/autodl-tmp/tca-reasoning/data/modelscope_cache/swift/llava-1___5-7b-hf
source repo = swift/llava-1.5-7b-hf
```

CLT-like asset：

```text
repo = KokosDev/llava15-7b-clt
layer = 0
file = transcoder_L0.pt
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

条件：

```text
clean
answer_mask
union_mask
```

脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_llava_clean_vs_mask_feature_readout.py
scripts/local/run_stage2f5_llava_mask_remote.py
```

## 4. 输出

```text
doc/experiments/stage2/cross_model/stage2f_llava_clean_vs_mask_feature_readout_3case.json
doc/experiments/stage2/cross_model/stage2f_llava_clean_vs_mask_feature_readout_3case.csv
doc/experiments/stage2/cross_model/stage2f_llava_clean_vs_mask_feature_readout_3case_summary.csv
```

## 5. 方法

对每个 `sample_id x prompt`：

```text
1. 读取 clean image。
2. 读取 answer_mask 与 relate_mask。
3. 构造 union_mask = answer_mask ∪ relate_mask。
4. 生成三种图像条件：clean、answer_mask、union_mask。
5. 用同一个 LLaVA prompt 跑 native image-text forward。
6. 读取 hidden_states[layer + 1]。
7. 用 public LLaVA transcoder_L0.pt encoder 计算 CLT-like feature activation。
8. 按 token bucket 比较 clean top-k feature 在 mask 后的 activation drop。
```

本轮 token bucket 固定为：

```text
image_token_span
question
post_image_text
assistant_prefix
last_prompt_token
```

主读法优先看 `image_token_span`，因为 Stage 2F 的目的只是检查 LLaVA 是否也能在图像 token 区域读到 evidence-mask sensitivity。

## 6. 实际结果

最终判定：

```text
decision.status = pass_mask_readout
usable_samples = 3
requested_samples = 3
skipped_samples = []
```

输出规模：

```text
summary rows = 60
detail rows = 1200
```

### 6.1 聚合结果

`mean_topk_drop` 的聚合均值：

```text
answer_mask / image_token_span: +0.0747
union_mask  / image_token_span: +0.0990

answer_mask / question: +0.0138
union_mask  / question: +0.0497

answer_mask / post_image_text: +0.0134
union_mask  / post_image_text: +0.0499

answer_mask / assistant_prefix: -0.0043
union_mask  / assistant_prefix: -0.0080

answer_mask / last_prompt_token: -0.0209
union_mask  / last_prompt_token: -0.0310
```

读法：

```text
image_token_span 上 answer/union mask 都产生正向 drop；
union_mask 的 image-token drop 更大、更稳定；
question/post-image text 也有较小正向移动；
assistant/last-prompt token 没有正向削弱，反而略为负值；
这支持“LLaVA layer 0 readout 对 evidence mask 有可观察反应”，但不支持 causal route claim。
```

### 6.2 分 case 的 image-token 结果

```text
okvqa_val_2847255 / B_direct / answer_mask: +0.1174
okvqa_val_2847255 / B_direct / union_mask:  +0.0688
okvqa_val_2847255 / D_visual_only / answer_mask: +0.1174
okvqa_val_2847255 / D_visual_only / union_mask:  +0.0688

okvqa_val_4157235 / B_direct / answer_mask: +0.0276
okvqa_val_4157235 / B_direct / union_mask:  +0.1172
okvqa_val_4157235 / D_visual_only / answer_mask: +0.0276
okvqa_val_4157235 / D_visual_only / union_mask:  +0.1172

okvqa_val_3658865 / B_direct / answer_mask: +0.0790
okvqa_val_3658865 / B_direct / union_mask:  +0.1109
okvqa_val_3658865 / D_visual_only / answer_mask: +0.0790
okvqa_val_3658865 / D_visual_only / union_mask:  +0.1109
```

注意：layer 0 的 B/D image-token 结果几乎相同。这不意外，因为 LLaVA 的 image token span 位于文本问题和 prompt instruction 之前，且 layer 0 是非常早期的 readout。这个结果不能用来讨论 prompt modulation。

## 7. 预期与实际偏差

原计划优先尝试：

```text
layer = 0,15,30
```

实际偏差：

```text
transcoder_L15.pt 首次下载速度约 110-170KB/s；
文件大小约 268MB；
三层版本会把本轮 smoke 卡在高层下载上。
```

处理方式：

```text
终止三层 job；
改为 layer 0-only；
先完成最小可解释闭环；
将 layer 15/30 留作后续单独下载和补跑。
```

这意味着本实验只证明 LLaVA layer 0 readout feasibility 和弱 evidence-mask sensitivity，不证明更深层的 answer-adjacent route。

## 8. 结论

可以写：

```text
LLaVA / Llama-family VLM 支线已经从 processor/config、base hook-forward 推进到 layer 0 CLT-like clean-vs-mask readout；
在 3 个已标注 OK-VQA case 上，answer/union evidence mask 会在 image_token_span 上产生正向 top-k feature drop；
union mask 的 drop 更稳定，说明 evidence-region masking 对 LLaVA image-token feature readout 有可观察影响。
```

不能写：

```text
LLaVA 复现了 Gemma3 的 causal support route；
LLaVA source node 强于 nearest control；
LLaVA 的 feature drop 已经解释了答案变化；
prompt D_visual_only 在 LLaVA 上更好或更 visual grounded。
```

## 9. 对 Stage 2 的意义

这轮结果让 cross-model 支线从“资产能加载”升级为：

```text
Qwen：3-case evidence-mask readout pass，layer 26 image bucket 反应强；
LLaVA：3-case layer 0 evidence-mask readout pass，image token span 有弱但一致的正向 drop。
```

因此，下一步跨模型路线可以分成两条：

```text
Qwen：优先设计 minimal intervention / hook adapter，因为 Qwen 的高层 image-bucket mask response 更强。
LLaVA：先后台下载 layer 15/30 transcoder，再补 high-layer readout；若高层也出现更强 evidence-mask sensitivity，再考虑 intervention adapter。
```
