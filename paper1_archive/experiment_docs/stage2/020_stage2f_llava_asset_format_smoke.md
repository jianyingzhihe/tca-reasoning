# 实验 020：Stage 2F LLaVA-1.5 CLT Asset Format Smoke

日期：2026-05-20

## 1. 实验目的

本实验用于验证 LLaVA / Llama-family VLM 路线是否有继续推进价值。

用户提到“Llama 上能不能做”，这里必须先拆清楚：

```text
纯 Llama 是 language-only model，不能输入图像，也不能做 image-region mask。
真正能作为视觉跨模型候选的是 LLaVA-1.5 这种 Llama-family VLM。
```

因此，本实验不测试纯 Llama，而是测试：

```text
KokosDev/llava15-7b-clt 是否是可读取的 LLaVA-1.5 CLT-like 资产？
它的 .pt / mapping 文件能否下载并 torch.load？
它是否有足够清楚的 tensor shape，支持后续写 adapter？
```

## 2. 术语解释

`LLaVA`：一种 Llama-family VLM，通常由视觉编码器、视觉到语言投影器、Llama-family language model 组成。

`Llama-family VLM`：底层语言模型来自 Llama-family，但模型整体可以输入图像。LLaVA 属于这个类别。

`asset format smoke`：只检查公开资产文件列表、文件格式、tensor keys 和 shape。它不加载 base model，不跑图文 forward，也不做机制实验。

`mapping_L*.pt`：本资产中每层对应的 mapping 文件。当前读到的描述是 `MLP neuron -> CLT feature correlations from training data`，说明它可能记录 MLP neuron 与 CLT feature 的关联映射。

## 3. 输入

Transcoder repo：

```text
KokosDev/llava15-7b-clt
```

默认 base model：

```text
llava-hf/llava-1.5-7b-hf
```

测试层：

```text
layer 0
```

下载文件：

```text
transcoder_L0.pt
mapping_L0.pt
```

## 4. 输出

本地 artifact：

```text
E:\Bridging\doc\experiments\stage2\020_stage2f_llava_asset_format_smoke.md
E:\Bridging\doc\experiments\stage2\cross_model\stage2f_llava_asset_format_smoke.json
```

远端 artifact：

```text
/root/autodl-tmp/tca-reasoning/stage2f_cross_model/stage2f_llava_asset_format_smoke.json
```

新增脚本：

```text
E:\Bridging\vlm-circuit-tracing\circuit_tracer_vlm\scripts\research\run_llava_asset_format_smoke.py
```

## 5. 方法

远端运行逻辑：

```bash
cd /root/autodl-tmp/tca-reasoning/circuit_tracer_vlm
source scripts/server/dev.sh
source /etc/network_turbo
export PYTHONPATH=/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm:${PYTHONPATH:-}

.venv/bin/python -u scripts/research/run_llava_asset_format_smoke.py \
  --repo-id KokosDev/llava15-7b-clt \
  --layer-index 0 \
  --download-sample \
  --out-json /root/autodl-tmp/tca-reasoning/stage2f_cross_model/stage2f_llava_asset_format_smoke.json
```

脚本步骤：

1. 用 Hugging Face API 读取 repo file list。
2. 选择 `transcoder_L0.pt` 和 `mapping_L0.pt`。
3. 下载两个样本文件到服务器 HF cache。
4. 用 `torch.load(..., map_location="cpu")` 读取文件。
5. 递归摘要 dict / tensor keys、shape、dtype。
6. 根据可读性判断是否值得进入 LLaVA base loader / hook-forward。

## 6. 结果

最终判定：

```text
decision.status = pass_format_readable
reason = sample_pt_files_downloaded_and_torch_loaded
```

Repo file list：

```text
file_count = 65
pt_file_count = 62
has_config_yaml = false
has_readme = true
```

文件结构：

```text
mapping_L0.pt ... mapping_L30.pt
transcoder_L0.pt ... transcoder_L30.pt
training_summary.json
README.md
```

这说明该资产覆盖 31 个 layer，即 `0-30`。

下载样本：

```text
transcoder_L0.pt size = 268,480,223 bytes
mapping_L0.pt size = 201,328,637 bytes
```

`transcoder_L0.pt` 读取结果：

```text
type = dict
len = 6
layer = 0
hidden_dim = 4096
feature_dim = 8192
```

关键 tensor：

```text
_orig_mod.enc.0.weight shape = [4096], dtype = bfloat16
_orig_mod.enc.0.bias shape = [4096], dtype = bfloat16
_orig_mod.enc.1.weight shape = [8192, 4096], dtype = bfloat16
_orig_mod.enc.1.bias shape = [8192], dtype = bfloat16
_orig_mod.dec.weight shape = [4096, 8192], dtype = bfloat16
_orig_mod.dec.bias shape = [4096], dtype = bfloat16
mlp_to_clt_mapping shape = [4096, 8192], dtype = float32
```

训练 metadata：

```text
steps = 5000
final_l0_pct = 0.2002030611038208
dead_features = 0
dead_pct = 0.0
final_rec_loss = 0.00026525105931796134
```

`mapping_L0.pt` 读取结果：

```text
type = dict
len = 6
layer = 0
hidden_dim = 4096
feature_dim = 8192
```

关键 tensor：

```text
mlp_to_clt_mapping shape = [4096, 8192], dtype = float32
decoder_weights shape = [4096, 8192], dtype = bfloat16
```

描述字段：

```text
MLP neuron -> CLT feature correlations from training data
```

## 7. 预期与实际偏差

预期：

```text
LLaVA 资产可能比 Qwen 更难读，因为它不是 config.yaml + layer_*.safetensors 格式。
```

实际：

```text
文件列表可读。
layer 0 的 transcoder 和 mapping 都可下载。
两个 .pt 文件都能 torch.load。
tensor shape 明确，hidden_dim = 4096，feature_dim = 8192。
```

偏差：

```text
该 repo 没有 current loader 直接需要的 config.yaml。
文件名和 key 不是当前 TranscoderSet loader 的 layer_*.safetensors / W_enc / W_dec 风格。
encoder 看起来是两段结构：enc.0 与 enc.1。
mapping 文件携带 MLP neuron 到 CLT feature 的相关矩阵，不能直接等价为当前 Gemma3/Qwen 的 hook_resid_pre feature encoder。
```

因此，LLaVA 虽然资产可读，但不能直接进入当前 pipeline。

## 8. 结论

本实验支持：

```text
KokosDev/llava15-7b-clt 是真实可读的 LLaVA-family VLM circuit/transcoder 资产。
它至少覆盖 layer 0-30。
layer 0 hidden_dim = 4096，feature_dim = 8192。
sample transcoder 与 mapping 文件都能下载并 torch.load。
LLaVA 可以作为 Llama-family VLM cross-model 备用路线继续推进。
```

本实验不支持：

```text
LLaVA base model 已经可加载。
LLaVA native hook-forward 已经可行。
LLaVA hidden state 已经能送入 transcoder。
LLaVA 已经复现 Gemma3/Qwen 的 evidence-region-sensitive route。
纯 Llama 已经支持视觉证据结论。
```

最准确判定：

```text
Stage 2F LLaVA L0 = pass_format_readable。
```

## 9. 下一步

LLaVA 下一步不是直接机制实验，而是：

```text
LLaVA L1：base processor/config smoke。
LLaVA L2：native hook-forward smoke。
LLaVA L3：custom .pt/mapping adapter + feature readout smoke。
```

如果 L1/L2 失败，则 LLaVA 只保留为“有资产但工程风险较高”的备用路线。

如果 L1/L2 通过，则可以写一个小 adapter：

```text
读取 LLaVA hidden state [batch, seq, 4096]
匹配 transcoder_L*.pt 中 enc/dec 权重
输出 feature activation [batch, seq, 8192]
```

保守写法：

```text
LLaVA-1.5 provides a plausible Llama-family VLM route. Its public CLT-like assets are readable, but the format is custom and requires a dedicated adapter before feature readout, intervention, or evidence-region replication can be attempted.
```
