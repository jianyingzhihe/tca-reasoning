# 实验 025：Stage 2F LLaVA ModelScope 下载、Hook-Forward 与 CLT Feature Readout

日期：2026-05-20

## 1. 实验目的

本实验继续推进 LLaVA / Llama-family VLM cross-model 支线。上一轮 `023_stage2f_llava_base_hook_forward_smoke.md` 已经证明：

```text
LLaVA processor/config 可读；
text_hidden_size = 4096；
public LLaVA CLT-like assets 也是 hidden_dim = 4096；
但 HF/Xet 下载 base weights 太慢，未完成 native hook-forward。
```

本轮目标是：

```text
1. 使用 ModelScope 镜像下载 LLaVA base weights；
2. 修复 LLaVA processor metadata 缺失导致的 image-token mismatch；
3. 跑通 native image-text forward 和 language layer hook；
4. 将 LLaVA hidden state 送入 public CLT asset 的 encoder，做 layer 0 feature-readout smoke。
```

这仍然不是跨模型机制复现，不做 attribution、不做 intervention、不做 evidence-mask causal route。

## 2. 输入

Base model：

```text
ModelScope repo = swift/llava-1.5-7b-hf
local path = /root/autodl-tmp/tca-reasoning/data/modelscope_cache/swift/llava-1___5-7b-hf
```

CLT-like asset：

```text
KokosDev/llava15-7b-clt
layer = 0
file = transcoder_L0.pt
```

测试样本：

```text
image = COCO_val2014_000000192716.jpg
question = What does stop mean?
```

## 3. 输出

```text
doc/experiments/stage2/cross_model/stage2f_llava_modelscope_download.json
doc/experiments/stage2/cross_model/stage2f_llava_base_hook_forward_smoke_modelscope.json
doc/experiments/stage2/cross_model/stage2f_llava_base_hook_forward_smoke_modelscope_fixed.json
doc/experiments/stage2/cross_model/stage2f_llava_base_hook_forward_smoke_modelscope_fixed2.json
doc/experiments/stage2/cross_model/stage2f_llava_clt_feature_readout_layer0.json
doc/experiments/stage2/cross_model/stage2f_llava_clt_feature_readout_layer0_fixed.json
doc/experiments/stage2/cross_model/stage2f_llava_clt_feature_readout_layer0_fixed.csv
```

涉及脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_llava_base_hook_forward_smoke.py
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_llava_clt_feature_readout_smoke.py
scripts/local/run_stage2f4_cross_model_remote.py
```

## 4. 方法

### 4.1 下载

使用 `modelscope.hub.snapshot_download` 下载：

```text
swift/llava-1.5-7b-hf
```

写入服务器：

```text
/root/autodl-tmp/tca-reasoning/data/modelscope_cache
```

### 4.2 Hook-forward

加载：

```text
LlavaForConditionalGeneration
LlavaProcessor
```

输入 prompt：

```text
USER: <image>
What does stop mean?
ASSISTANT:
```

注册 hook：

```text
model.language_model.layers.0
```

记录：

```text
input_ids_shape
pixel_values_shape
logits_shape
hidden_states_count
selected_hidden_shape
hook input / output shape
```

### 4.3 Processor metadata 修复

第一次 hook-forward 失败：

```text
unsupported operand type(s) for //: 'int' and 'NoneType'
```

原因：

```text
processor.patch_size 缺失。
```

修复：

```text
processor.patch_size = config.vision_config.patch_size = 14
processor.vision_feature_select_strategy = config.vision_feature_select_strategy = default
```

第二次 hook-forward 失败：

```text
Image features and image tokens do not match: tokens: 575, features 2359296
```

诊断：

```text
processor 生成 575 个 <image> token；
模型视觉特征实际对应 576 个 image patches；
需要 num_additional_image_tokens = 1。
```

最终修复：

```text
processor.num_additional_image_tokens = 1
```

### 4.4 Feature readout

读取 `transcoder_L0.pt`，其结构是：

```text
top-level:
layer
hidden_dim
feature_dim
state_dict
training_metadata
mlp_to_clt_mapping

state_dict:
_orig_mod.enc.0.weight
_orig_mod.enc.0.bias
_orig_mod.enc.1.weight
_orig_mod.enc.1.bias
_orig_mod.dec.weight
_orig_mod.dec.bias
```

readout 计算：

```text
hidden = hidden_states[layer + 1]
hidden_norm = LayerNorm(hidden, enc.0.weight, enc.0.bias)
features = ReLU(Linear(hidden_norm, enc.1.weight, enc.1.bias))
```

bucket：

```text
image_token_span
post_image_text
last_prompt_token
```

## 5. 实际结果

### 5.1 ModelScope 下载

判定：

```text
status = pass
model_dir = /root/autodl-tmp/tca-reasoning/data/modelscope_cache/swift/llava-1___5-7b-hf
```

磁盘：

```text
disk_before.free_gb = 132.483
disk_after.free_gb = 119.221
```

解释：

```text
ModelScope 下载约占 13GB，比 HF/Xet 路线稳定很多。
```

### 5.2 Hook-forward

最终判定：

```text
decision.status = pass_hook_forward
reason = llava_native_forward_and_hook_ok
```

processor：

```text
processor_class = LlavaProcessor
patch_size = 14
vision_feature_select_strategy = default
num_additional_image_tokens = 1
```

forward：

```text
input_ids_shape = [1, 593]
attention_mask_shape = [1, 593]
pixel_values_shape = [1, 3, 336, 336]
logits_shape = [1, 593, 32064]
hidden_states_count = 33
selected_hidden_shape = [1, 593, 4096]
```

hook：

```text
registered_module = model.language_model.layers.0
hook input_shape = [[1, 593, 4096]]
hook output_shape = [1, 593, 4096]
```

### 5.3 CLT feature readout

最终判定：

```text
decision.status = pass_feature_readout
reason = llava_hidden_states_encoded_by_public_clt_assets
```

token buckets：

```text
image_token_span = 576
post_image_text = 11
last_prompt_token = 1
```

layer 0：

```text
hidden_shape = [1, 593, 4096]
hidden_dim = 4096
feature_dim = 8192
```

image_token_span readout：

```text
position_count = 576
max_activation = 3.6699
active_positive_count = 3,159,371
top features:
feature 5865 activation 3.6699
feature 1361 activation 3.4219
feature 7314 activation 3.4121
```

last_prompt_token readout：

```text
position_count = 1
max_activation = 4.5469
active_positive_count = 1,758
top features:
feature 5438 activation 4.5469
feature 7932 activation 1.6582
feature 3923 activation 1.0752
```

post_image_text readout：

```text
position_count = 11
max_activation = 5.6836
active_positive_count = 18,977
top features:
feature 5438 activation 5.6836
feature 7974 activation 2.8242
feature 1966 activation 2.4102
```

## 6. 预期与实际偏差

预期：

```text
ModelScope 可能能解决 HF/Xet 慢速下载；
processor/config 应可读；
base hook-forward 可能还会遇到 LLaVA processor / image token 兼容问题。
```

实际：

```text
ModelScope 下载成功；
base load 成功；
processor metadata 需要手动回填；
修复后 native hook-forward 成功；
public CLT layer 0 feature readout 成功。
```

主要偏差：

```text
LLaVA processor 原始 metadata 不完整，必须补 patch_size 和 num_additional_image_tokens；
public CLT asset 的 encoder 权重不在顶层，而在 state_dict 内；
需要使用 enc.0 LayerNorm + enc.1 Linear，而不是直接读取顶层 W_enc。
```

## 7. 结论

本实验支持：

```text
LLaVA-1.5 base model 已经可在服务器本地加载；
LLaVA native image-text forward 可跑通；
language layer hook 可读；
LLaVA hidden state [1, 593, 4096] 能进入 public CLT-like asset，并得到 feature_dim=8192 的 feature readout；
LLaVA 已从 asset-format / processor-config feasibility 推进到 hook-forward + feature-readout feasibility。
```

本实验不支持：

```text
LLaVA 已经完成 attribution；
LLaVA 已经完成 source node tracing；
LLaVA 已经完成 feature intervention；
LLaVA 已经复现 Gemma3 evidence-region-sensitive support routes；
LLaVA 已经做了 evidence-mask readout 或 causal region-mask replication。
```

最准确判定：

```text
Stage 2F LLaVA L1/L2/L3 = pass base download + pass hook-forward + pass layer-0 CLT feature readout。
```

## 8. 对主 claim 的影响

不改变当前 Gemma3 主 claim。这个实验的价值是把 LLaVA 从“可能可用的 Llama-family VLM 资产”推进成“确实可以加载、hook、读出 CLT feature”的跨模型候选。

现在可以更有把握地把 cross-model 后续计划写成：

```text
Qwen: readout-level evidence-mask sensitivity -> minimal intervention adapter
LLaVA: hook/readout feasibility -> evidence-mask readout smoke
```

而不是停留在“有没有模型/资产”的阶段。

## 9. 后续动作

LLaVA 下一步建议：

```text
1. 给 LLaVA 加 token / position mapping 脚本；
2. 复用已有 evidence masks 做 clean vs answer/union feature readout；
3. 优先只跑 layer 0，再扩到 layer 15 / layer 30；
4. 若 readout-level evidence-mask sensitivity 成立，再考虑 LLaVA feature intervention adapter；
5. 在 intervention 之前，不写 LLaVA causal route claim。
```

