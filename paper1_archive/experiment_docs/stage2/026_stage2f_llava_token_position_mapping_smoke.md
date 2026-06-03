# 实验 026：Stage 2F LLaVA Token/Position Mapping Smoke

日期：2026-05-20

## 1. 实验目的

本实验继续推进 Stage 2F 的 LLaVA / Llama-family VLM 支线。上一轮已经证明：

```text
LLaVA base model 可以通过 ModelScope 下载并本地加载；
native image-text forward 可以跑通；
language layer hook 可以读到 hidden state；
layer 0 hidden state 可以送入 public LLaVA CLT-like encoder 得到 feature readout。
```

本轮目标是补齐位置读法：

```text
确认 LLaVA prompt 中 image token span、question、assistant prefix、last prompt token 的位置；
确认这些 token bucket 能进入 public CLT-like asset 做 feature readout；
为后续 clean vs evidence-mask readout 提供 bucket 对齐依据。
```

边界：这仍然不是 attribution、不是 intervention、不是 source route tracing，也不是跨模型机制复现。它只证明 LLaVA 的 token/position readout 入口已经可用。

## 2. 专有名词解释

```text
Token/position mapping：
把模型输入序列中的每个 token 位置标成 image token、question token、assistant prefix 等区域，避免误把模板 token 当成答案附近位置。

Image token span：
LLaVA 把图像编码后展开到语言序列中的一段连续 <image> token。本次样本中为 576 个位置。

CLT-like encoder：
public LLaVA asset 中的 transcoder encoder。这里仅用 encoder 把 hidden state 映射到 feature activation，不使用 decoder，不做 causal intervention。

Bucket readout：
对某个 token bucket 内的 feature activation 做 top-k 读取，例如 image_token_span 中哪些 feature 激活最高。
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
sample_id = okvqa_val_2847255
image = COCO_val2014_000000284725.jpg
question = What country might this be based on the writing on the bus?
prompt = question + visual-evidence instruction + fixed short-answer format
```

脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_llava_token_position_mapping_smoke.py
scripts/local/run_stage2f5_llava_mask_remote.py
```

## 4. 输出

```text
doc/experiments/stage2/cross_model/stage2f_llava_position_mapping_2847255.json
doc/experiments/stage2/cross_model/stage2f_llava_position_mapping_2847255_tokens.csv
doc/experiments/stage2/cross_model/stage2f_llava_position_mapping_2847255_buckets.csv
```

## 5. 方法

首先加载 `LlavaProcessor`、`AutoConfig` 和 `LlavaForConditionalGeneration`。由于该 converted checkpoint 的 processor metadata 不完整，沿用上一轮修复：

```text
processor.patch_size = config.vision_config.patch_size = 14
processor.vision_feature_select_strategy = default
processor.num_additional_image_tokens = 1
```

然后用固定 LLaVA prompt：

```text
USER: <image>
{question}
ASSISTANT:
```

脚本记录：

```text
input_ids
image token id
question span
assistant span
last prompt token
每个 token 的 bucket label
```

最后读取 `hidden_states[layer + 1]`，用 `transcoder_L0.pt` 中的：

```text
_orig_mod.enc.0 = LayerNorm
_orig_mod.enc.1 = Linear + ReLU
```

计算 layer 0 的 CLT-like feature activation，并按 bucket 输出 top-k feature。

## 6. 实际结果

最终判定：

```text
decision.status = pass_position_mapping
reason = image_question_and_last_prompt_positions_mapped
```

输入序列：

```text
sequence_length = 624
image_span = [5, 581]
image_token_count = 576
question_span = [581, 619]
assistant_span = [620, 624]
last_prompt_token = 623
```

bucket 计数：

```text
image_token_span = 576
post_image_text = 39
question = 38
assistant_prefix = 4
last_prompt_token = 1
special/template = 579
other_text_or_template = 4
```

Layer 0 bucket readout：

```text
image_token_span max_activation = 3.546875
question max_activation = 5.5546875
post_image_text max_activation = 5.5546875
assistant_prefix max_activation = 4.28125
last_prompt_token max_activation = 4.28125
```

image bucket top features：

```text
feature 4388: activation 3.546875
feature 2004: activation 3.341796875
feature 628: activation 3.330078125
```

## 7. 预期与实际偏差

原计划尝试 `layer = 0,15,30`。实际运行中，LLaVA base model 很快加载完成，但 `transcoder_L15.pt` 首次从 HuggingFace 下载速度只有约 `110-170KB/s`，文件大小约 `268MB`，预计需要半小时级别。为了不让 Stage 2F 卡在高层权重下载上，本轮主动降级为 layer 0-only smoke。

这个偏差属于下载工程问题，不是 LLaVA forward 失败，也不是 CLT readout 失败。高层 readout 可以后续单独后台下载后再补。

## 8. 结论

可以写：

```text
LLaVA 的 image token span、question、assistant prefix 和 last prompt token 可以稳定定位；
layer 0 hidden state 可以通过 public LLaVA CLT-like encoder 得到 feature readout；
LLaVA 已经具备进入 evidence-mask readout smoke 的最低条件。
```

不能写：

```text
LLaVA 已经复现 Gemma3 的 causal support route；
LLaVA source nodes 强于 matched controls；
LLaVA 已经完成跨模型机制复现。
```
