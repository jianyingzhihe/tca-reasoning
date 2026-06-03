# Stage 2F-2：Qwen2.5-VL Native Hook-Forward Smoke

日期：2026-05-20

## 1. 实验目的

本实验接在 `017_stage2f_qwen_download_and_lazy_load.md` 之后，目标是验证：

```text
在 Qwen2.5-VL-7B-Instruct base 权重已经本地可用的情况下，是否可以完成一个最小图文 forward，并读到 hidden states 与普通 PyTorch layer hook activation shape。
```

这一步仍然不是跨模型机制复现。它只回答：

```text
native transformers Qwen 是否能被加载、处理一张 OK-VQA 图片和一个问题，并暴露可 hook 的语言模型层 activation？
```

本实验不包含：

```text
Qwen attribution
Qwen CLT feature activation injection
Qwen source node tracing
Qwen node zeroing intervention
Qwen region-mask experiment
Gemma3 ReplacementModel 适配
```

## 2. 术语解释

`native Qwen forward`：直接使用 `transformers.Qwen2_5_VLForConditionalGeneration` 跑前向传播，不经过当前 Gemma3-oriented `ReplacementModel`。

`hidden states`：transformers 模型在各层输出的隐藏状态张量。这里用于确认每一层 token representation 可读。

`forward hook`：PyTorch 在某个 module 前向传播时注册的回调函数，用来捕获该 module 输入或输出 shape。

`hook-forward smoke`：只验证 hook 和 forward 是否可行的小测试。它不代表后续 attribution / intervention pipeline 已经适配。

`partial`：本实验中的 `partial` 是预期状态，意思是 native Qwen forward 成功，但当前主 pipeline 仍缺 Qwen ReplacementModel adapter，所以不能升级为 full cross-model pipeline pass。

## 3. 输入

本地 Qwen snapshot：

```text
/root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5
```

测试图片：

```text
/root/autodl-tmp/tca-reasoning/data/okvqa/images/val2014/COCO_val2014_000000192716.jpg
```

测试问题：

```text
What does stop mean?
```

测试层：

```text
layer_index = 0
```

脚本：

```text
E:\Bridging\vlm-circuit-tracing\circuit_tracer_vlm\scripts\research\run_qwen_hook_forward_smoke.py
```

## 4. 输出

本地 artifact：

```text
E:\Bridging\doc\experiments\stage2\018_stage2f_qwen_hook_forward_smoke.md
E:\Bridging\doc\experiments\stage2\cross_model\stage2f_qwen_hook_forward_smoke.json
```

远端 artifact：

```text
/root/autodl-tmp/tca-reasoning/stage2f_cross_model/stage2f_qwen_hook_forward_smoke.json
```

## 5. 方法

运行前检查：

```text
GPU = NVIDIA vGPU-48GB
free_gb before = 46.986
test image exists = true
```

运行命令逻辑：

```bash
.venv/bin/python -u scripts/research/run_qwen_hook_forward_smoke.py \
  --model-name /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --image-path /root/autodl-tmp/tca-reasoning/data/okvqa/images/val2014/COCO_val2014_000000192716.jpg \
  --question "What does stop mean?" \
  --layer-index 0 \
  --min-gpu-free-gb 18 \
  --out-json /root/autodl-tmp/tca-reasoning/stage2f_cross_model/stage2f_qwen_hook_forward_smoke.json
```

脚本步骤：

1. 用 `AutoProcessor.from_pretrained(local_snapshot, local_files_only=True)` 加载 processor。
2. 用 `Qwen2_5_VLForConditionalGeneration.from_pretrained(local_snapshot, torch_dtype=bfloat16, device_map="auto")` 加载 base model。
3. 用 processor 构造图文输入。
4. 在 `model.language_model.layers.0` 注册普通 PyTorch forward hook。
5. 执行 `model(..., output_hidden_states=True, use_cache=False)`。
6. 记录 logits shape、hidden states count、选中层 hidden shape、hook input/output shape。
7. 检查当前 `ReplacementModel` 兼容边界。

## 6. 结果

最终判定：

```text
decision.status = partial
reason = native_qwen_forward_ok_but_no_replacement_model_adapter
```

processor：

```text
status = ok
processor_class = Qwen2_5_VLProcessor
has_tokenizer = true
```

model load：

```text
status = ok
class = Qwen2_5_VLForConditionalGeneration
device = cuda:0
checkpoint shards loaded = 5 / 5
```

input summary：

```text
input_ids shape = [1, 440], dtype int64
attention_mask shape = [1, 440], dtype int64
pixel_values shape = [1656, 1176], dtype float32
image_grid_thw shape = [1, 3], dtype int64
```

forward：

```text
status = ok
logits_shape = [1, 440, 152064]
hidden_states_count = 29
selected_hidden_shape = [1, 440, 3584]
```

native hook：

```text
candidate_layer_modules = ["model.language_model.layers.0"]
registered_module = model.language_model.layers.0
hook input_shape = [[1, 440, 3584]]
hook output_shape = [[1, 440, 3584]]
```

GPU after：

```text
free_gb after = 31.148
```

解释：

```text
Qwen native 图文 forward 已经跑通。
Qwen 语言模型第 0 层可以被普通 PyTorch hook 捕获输入/输出 activation。
hidden state 的 hidden_dim = 3584，与 CLT config 的 hidden_dim = 3584 对齐。
```

## 7. 预期与实际偏差

预期：

```text
如果 base 权重完整，native Qwen forward 应该可以跑通。
如果 hook module 名可发现，应能记录 activation shape。
```

实际：

```text
native forward 跑通。
hook module 名为 model.language_model.layers.0。
hidden states 与 hook shape 可读。
```

主要偏差：

```text
当前 hook 名不是 circuit-tracer / TransformerLens 风格的 blocks.{layer}.hook_resid_pre/post。
CLT config 中的 hook metadata 写的是 blocks.{layer}.hook_resid_pre/post。
因此需要后续 Qwen adapter 把 native module / hidden state 映射到 CLT 期望 hook point。
```

## 8. 结论

本实验可以支持：

```text
Qwen2.5-VL-7B-Instruct native forward 可行。
Qwen 图文 processor 可行。
Qwen hidden states 可读。
Qwen language layer forward hook activation shape 可读。
Qwen hidden_dim 与 CLT hidden_dim 对齐。
```

本实验不能支持：

```text
Qwen 已经接入当前 ReplacementModel。
Qwen 已经能跑 CLT attribution。
Qwen 已经能跑 feature intervention。
Qwen 已经复现 Gemma3 evidence-region-sensitive support routes。
```

当前最准确的判定是：

```text
Stage 2F-2 native Qwen hook-forward smoke = partial success.
```

原因：

```text
native Qwen forward / hook 成功，但主 pipeline 仍缺 Qwen adapter。
```

## 9. 后续动作

如果继续推进 Stage 2F，下一步不应该直接声称跨模型复现，而应该进入更小的 adapter smoke：

```text
1. 建立 Qwen native module name 到 CLT hook metadata 的映射。
2. 设计最小 Qwen wrapper，只读 hidden states，不做 intervention。
3. 验证某一层 hidden state 是否能送入对应 CLT layer encoder。
4. 再尝试 single layer feature activation readout。
5. 最后才考虑 attribution / source zeroing / region-mask mini replication。
```

保守文案：

```text
Qwen2.5-VL now passes asset, lazy-load, and native hook-forward feasibility checks, but it remains outside the main Gemma3 circuit-tracing pipeline until a Qwen-specific ReplacementModel adapter is implemented.
```
