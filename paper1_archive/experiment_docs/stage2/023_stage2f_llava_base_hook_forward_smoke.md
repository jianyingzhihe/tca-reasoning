# 实验 023：Stage 2F LLaVA Base / Hook-Forward Smoke

日期：2026-05-20

## 1. 实验目的

本实验对应 Stage 2F-3 的 LLaVA L1/L2。上一轮 `020_stage2f_llava_asset_format_smoke.md` 已经确认 `KokosDev/llava15-7b-clt` 的 `transcoder_L0.pt` 和 `mapping_L0.pt` 可读。本轮要继续检查：

```text
llava-hf/llava-1.5-7b-hf 的 processor/config 是否可读？
如果下载和显存允许，base model 是否能跑 native hook-forward？
```

本实验仍然不是 LLaVA feature readout，也不是跨模型机制复现。

## 2. 输入

Base model：

```text
llava-hf/llava-1.5-7b-hf
```

测试图像：

```text
/root/autodl-tmp/tca-reasoning/data/okvqa/images/val2014/COCO_val2014_000000192716.jpg
```

测试问题：

```text
What does stop mean?
```

## 3. 输出

```text
doc/experiments/stage2/023_stage2f_llava_base_hook_forward_smoke.md
doc/experiments/stage2/cross_model/stage2f_llava_base_hook_forward_smoke.json
```

新增脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_llava_base_hook_forward_smoke.py
```

## 4. 方法

脚本原始计划：

1. 读取 `AutoProcessor`。
2. 读取 `AutoConfig`。
3. 如果 `/root/autodl-tmp` 可用空间大于 40GB 且 GPU 空闲大于 18GB，则下载并加载 LLaVA base。
4. 跑一个图文 forward。
5. 捕获 language layer hook shape。

实际执行中，processor/config 成功后开始下载 base weights：

```text
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors
```

但 HF/Xet 下载速度长期降到约 `160-300KB/s`，13 分钟后 shard 仍只有约 `3-5%`，预计总耗时会到数小时。因此本轮主动中止 full-base 下载，并用 `--skip-base-forward` 写入 partial JSON，避免阻塞 Qwen 主线分析。

## 5. 结果

最终判定：

```text
decision.status = partial_base_asset_pass
reason = processor_config_ok_but_base_forward_skipped_after_slow_hf_xet_download
```

Processor：

```text
processor.status = ok
processor_class = LlavaProcessor
has_tokenizer = true
```

Config：

```text
config.status = ok
model_type = llava
architectures = [LlavaForConditionalGeneration]
text_hidden_size = 4096
text_num_hidden_layers = 32
vision_config_type = CLIPVisionConfig
```

Base model：

```text
not fully downloaded
not loaded
no native forward
no hook shape
```

## 6. 预期与实际偏差

预期：

```text
如果网络正常，LLaVA base 应可下载并做 native hook-forward smoke。
```

实际：

```text
processor/config 成功；
base weights 下载受到 HF/Xet 慢速限制；
未完成 base load；
未跑 hook-forward。
```

这属于工程/网络阻塞，不是 LLaVA 机制失败，也不是 LLaVA asset format 失败。

## 7. 结论

本实验支持：

```text
LLaVA-1.5 processor/config 可读。
LLaVA language hidden size = 4096，与 LLaVA CLT asset 的 hidden_dim = 4096 对齐。
LLaVA 仍是可继续推进的 Llama-family VLM 候选。
```

本实验不支持：

```text
LLaVA base model 已经本地完整可用。
LLaVA native hook-forward 已经可行。
LLaVA hidden state 已经能进入 CLT-like adapter。
LLaVA 已经复现任何 evidence-region-sensitive route。
```

最准确判定：

```text
Stage 2F LLaVA L1 = partial_base_asset_pass。
Stage 2F LLaVA L2 = blocked by slow HF/Xet base-weight download。
```

## 8. 后续动作

如果继续推 LLaVA，应先解决下载路径，而不是直接写 adapter：

```text
1. 查找 ModelScope 或其他镜像是否有 llava-hf/llava-1.5-7b-hf 等价权重。
2. 或改用已缓存/更小的 LLaVA-family VLM。
3. base 权重完整后再做 native hook-forward。
4. hook-forward 成功后才做 .pt/mapping adapter。
```

保守写法：

```text
LLaVA currently passes asset-format and processor/config feasibility checks, and its hidden size matches the public CLT-like assets. Full hook-forward remains blocked by slow base-weight download rather than by a demonstrated model incompatibility.
```
