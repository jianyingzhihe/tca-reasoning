# 实验 028：Stage 2F LLaVA High-Layer Clean vs Evidence-Mask Readout

日期：2026-05-20

## 1. 实验目的

上一轮实验 027 已经证明 LLaVA layer 0 可以完成 3-case clean-vs-mask feature readout，并且 image token span 上有弱但一致的 evidence-mask sensitivity。本实验继续检查：

```text
LLaVA 的更深 language layers，尤其 layer 15 与 layer 30，是否比 layer 0 显示更强的 evidence-region sensitivity。
```

这一步仍然是 cross-model feasibility / readout 实验，不是机制复现。它不包含：

```text
source node tracing
matched node control
node intervention
attribution graph
decoded generation under node intervention
```

因此不能把结果写成 LLaVA 已经复现 Gemma3 的 causal support route。

## 2. 专有名词解释

```text
High-layer readout：
读取更深层 hidden state，并送入对应 layer 的 public CLT-like encoder。本实验读取 layer 15 和 layer 30。

Evidence-mask sensitivity：
遮挡人工标注的 answer / union evidence region 后，clean top-k feature activation 下降。这里用 mean_topk_drop 表示。

mean_topk_drop：
clean_activation - masked_activation。正值表示遮挡后 clean top feature 被削弱；负值表示遮挡后该 feature 反而增强。

Heterogeneous：
不同样本或不同 mask 条件方向不一致，因此只能描述现象，不能作为稳定机制结论。
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
layers = 15, 30
files = transcoder_L15.pt, transcoder_L30.pt
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

## 4. 输出

下载 manifest：

```text
doc/experiments/stage2/cross_model/stage2f_llava_high_layer_download.json
```

readout artifact：

```text
doc/experiments/stage2/cross_model/stage2f_llava_clean_vs_mask_feature_readout_3case_layers15_30.json
doc/experiments/stage2/cross_model/stage2f_llava_clean_vs_mask_feature_readout_3case_layers15_30.csv
doc/experiments/stage2/cross_model/stage2f_llava_clean_vs_mask_feature_readout_3case_layers15_30_summary.csv
```

## 5. 方法

先在服务器后台下载：

```text
transcoder_L15.pt
transcoder_L30.pt
```

下载完成后，复用实验 027 的同一套脚本和同一批 3 个样本，只把 `--layers` 从 `0` 改成：

```text
15,30
```

对每个 `sample_id x prompt x condition x layer`：

```text
1. 跑 LLaVA native image-text forward。
2. 读取 hidden_states[layer + 1]。
3. 通过对应 layer 的 LLaVA CLT-like encoder 得到 feature activation。
4. 在 image_token_span、question、post_image_text、assistant_prefix、last_prompt_token bucket 上比较 clean top-k feature 的 masked drop。
```

主读法仍然只看 image_token_span，因为这是 cross-model readout feasibility，不是 answer-adjacent route tracing。

## 6. 下载结果

后台下载完成：

```text
finished_at = 2026-05-20 20:36:28
transcoder_L15.pt status = ok, size_mb = 256.043
transcoder_L30.pt status = ok, size_mb = 256.043
```

服务器路径：

```text
/root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--KokosDev--llava15-7b-clt/snapshots/2ab7f0bb160ba7cfbd4ce2e0fa018ffa9a87b98f/transcoder_L15.pt
/root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--KokosDev--llava15-7b-clt/snapshots/2ab7f0bb160ba7cfbd4ce2e0fa018ffa9a87b98f/transcoder_L30.pt
```

## 7. Readout 结果

最终判定：

```text
decision.status = pass_mask_readout
usable_samples = 3
requested_samples = 3
skipped_samples = []
```

encoder load：

```text
layer 15: status = ok, hidden_dim = 4096, feature_dim = 8192
layer 30: status = ok, hidden_dim = 4096, feature_dim = 8192
```

### 7.1 Image-token span 聚合结果

```text
layer 15 / answer_mask / image_token_span:
  n = 6
  mean_topk_drop = +1.1018
  min = -0.1840
  max = +2.1988

layer 15 / union_mask / image_token_span:
  n = 6
  mean_topk_drop = +1.9326
  min = +1.4676
  max = +2.1885

layer 30 / answer_mask / image_token_span:
  n = 6
  mean_topk_drop = +0.7435
  min = -0.7344
  max = +3.1344

layer 30 / union_mask / image_token_span:
  n = 6
  mean_topk_drop = -2.6451
  min = -7.4648
  max = +0.7570
```

### 7.2 其他 bucket 聚合结果

Layer 15：

```text
answer_mask / question: +0.0699
answer_mask / post_image_text: +0.0740
answer_mask / assistant_prefix: -0.0381
answer_mask / last_prompt_token: +0.0165

union_mask / question: +0.0525
union_mask / post_image_text: +0.0566
union_mask / assistant_prefix: -0.0031
union_mask / last_prompt_token: +0.0811
```

Layer 30：

```text
answer_mask / question: -0.0622
answer_mask / post_image_text: -0.0622
answer_mask / assistant_prefix: -0.7494
answer_mask / last_prompt_token: +0.1621

union_mask / question: -0.1460
union_mask / post_image_text: -0.1460
union_mask / assistant_prefix: +0.2554
union_mask / last_prompt_token: -0.0816
```

## 8. 分 case 观察

Layer 15 的 union mask 最稳定：

```text
okvqa_val_2847255 / union_mask: +2.1418
okvqa_val_4157235 / union_mask: +2.1885
okvqa_val_3658865 / union_mask: +1.4676
```

Layer 15 的 answer mask 也总体为正，但 `okvqa_val_3658865` 稍微为负：

```text
okvqa_val_2847255 / answer_mask: +2.1988
okvqa_val_4157235 / answer_mask: +1.2906
okvqa_val_3658865 / answer_mask: -0.1840
```

Layer 30 非常异质：

```text
okvqa_val_2847255 / union_mask: -1.2273
okvqa_val_4157235 / union_mask: +0.7570
okvqa_val_3658865 / union_mask: -7.4648
```

## 9. 预期与实际偏差

预期：

```text
更深层可能比 layer 0 更接近 answer-adjacent computation，因此 evidence-mask response 可能更强。
```

实际：

```text
layer 15 明显强于 layer 0，尤其 union_mask 稳定为正；
layer 30 并不稳定，union_mask 甚至整体为负；
因此 LLaVA 不是简单“越深越 evidence-sensitive”，更像中层存在较清楚的 image evidence readout。
```

与 layer 0 对比：

```text
layer 0 / answer_mask / image_token_span ≈ +0.0747
layer 0 / union_mask / image_token_span ≈ +0.0990

layer 15 / answer_mask / image_token_span ≈ +1.1018
layer 15 / union_mask / image_token_span ≈ +1.9326
```

这说明 layer 15 的 readout response 明显强于 layer 0。

## 10. 结论

可以写：

```text
LLaVA 支线已经从 layer 0 readout 推进到 high-layer readout；
layer 15 的 image_token_span 对 answer/union evidence mask 有明显更强的 readout response；
union_mask 在 layer 15 上最稳定，3 个 case、2 个 prompt 都为正；
这为 LLaVA 进入下一步 minimal intervention / hook adapter 提供了更合适的候选层。
```

需要保守写：

```text
layer 30 结果异质，不能作为稳定 evidence-region-sensitive layer；
prompt B/D 在 image token span 上仍几乎相同，不能写 prompt modulation；
这仍是 readout-only，不是 causal support route。
```

不能写：

```text
LLaVA 已经复现 Gemma3 的 causal mechanism；
LLaVA source routes 强于 controls；
LLaVA high-layer feature 已经解释最终答案变化；
layer 30 也稳定支持 evidence-mask sensitivity。
```

## 11. 下一步

优先级：

```text
1. 把 LLaVA layer 15 作为下一步 minimal intervention adapter 的候选层；
2. 先做 1 个 case 的 hook-level feature ablation / activation patch smoke；
3. 若 layer 15 intervention 能改变 target token 或 first-token distribution，再扩到 3-case；
4. layer 30 暂时不作为主候选，只作为异质性附录。
```
