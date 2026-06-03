# Stage 2F-2：Qwen2.5-VL CLT 下载、Base 资产补齐与 Lazy Load 判定

日期：2026-05-20

## 1. 实验目的

本实验属于 Stage 2F 的跨模型可行性支线，目标不是证明跨模型机制复现，而是回答一个更靠前的工程问题：

```text
Qwen2.5-VL-7B-Instruct 及其公开 CLT 资产，是否已经具备进入我们方法链的最低资产条件？
```

这里的“最低资产条件”包括：

1. Qwen2.5-VL CLT 的 27 层 transcoder 权重可以下载、读取 tensor shape，并可被现有 `load_transcoder_from_hub(..., lazy_encoder=True, lazy_decoder=True)` lazy load。
2. Qwen2.5-VL base model 的 processor / tokenizer / config / 5 个 safetensors shard 可以在服务器落盘。
3. 在不改 Gemma3 主 pipeline 的前提下，Qwen base model 可以作为本地 `transformers` 模型资产通过 processor 和 meta-model loader smoke。
4. 明确记录当前仍不能直接运行 attribution / intervention 主实验的原因。

本实验不包含：

```text
Qwen attribution
Qwen source node tracing
Qwen feature intervention
Qwen region-mask replication
跨模型机制复现
```

因此，即使本轮判定为 `pass`，它也只表示“资产和 native loader 可行”，不表示“Qwen 已经复现 Gemma3 上的 evidence-sensitive routes”。

## 2. 术语解释

`CLT`：Cross-Layer Transcoder，跨层转码器。这里指 KokosDev 发布的 Qwen2.5-VL-7B CLT，每层一个 `layer_*.safetensors` 文件，用来把模型 hidden activations 映射到稀疏 feature 空间。

`lazy load`：懒加载。只验证 transcoder set 的结构、metadata 和必要对象是否能被构造，不把全部 encoder / decoder 权重立即加载进显存。

`base model asset smoke`：base 模型资产烟测。只检查 processor、tokenizer、config、权重文件和 meta-model 构造，不做任务推理或机制干预。

`HF cache`：Hugging Face 本地缓存目录，本轮服务器路径为 `/root/autodl-tmp/tca-reasoning/data/hf_cache`。

`ModelScope fallback`：当 Hugging Face / Xet 下载卡住时，从 ModelScope 国内镜像补齐同名 Qwen 权重文件。

`ReplacementModel`：当前主 pipeline 中连接 base VLM 与 transcoder 的包装模型。当前实现仍主要绑定 Gemma3 路径，所以 Qwen native load 成功不等于可以直接跑 circuit-tracer 主实验。

## 3. 输入

CLT 资产：

```text
KokosDev/qwen2p5vl-7b-clt
```

Base model：

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

服务器主要路径：

```text
/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm
/root/autodl-tmp/tca-reasoning/data/hf_cache
/root/autodl-tmp/tca-reasoning/stage2f_cross_model
```

本地脚本：

```text
E:\Bridging\vlm-circuit-tracing\circuit_tracer_vlm\scripts\research\run_cross_model_asset_download_smoke.py
E:\Bridging\vlm-circuit-tracing\circuit_tracer_vlm\scripts\research\run_qwen_base_asset_smoke.py
```

## 4. 输出

本地 artifact：

```text
E:\Bridging\doc\experiments\stage2\017_stage2f_qwen_download_and_lazy_load.md
E:\Bridging\doc\experiments\stage2\cross_model\stage2f_qwen_download_manifest.json
E:\Bridging\doc\experiments\stage2\cross_model\stage2f_qwen_clt_layer_shapes.csv
E:\Bridging\doc\experiments\stage2\cross_model\stage2f_qwen_base_loader_smoke.json
```

远端中间 artifact：

```text
/root/autodl-tmp/tca-reasoning/stage2f_cross_model/stage2f_qwen_download_manifest.json
/root/autodl-tmp/tca-reasoning/stage2f_cross_model/stage2f_qwen_clt_layer_shapes.csv
/root/autodl-tmp/tca-reasoning/stage2f_cross_model/stage2f_qwen_base_loader_smoke.json
```

## 5. 方法

### 5.1 CLT 全量下载与 shape smoke

运行目标：

```text
下载 KokosDev/qwen2p5vl-7b-clt 全量 27 层 safetensors
检查 layer_0 / layer_13 / layer_26 tensor names、shape、dtype
尝试 lazy CLT load
```

关键命令逻辑：

```bash
.venv/bin/python -u scripts/research/run_cross_model_asset_download_smoke.py \
  --repo-id KokosDev/qwen2p5vl-7b-clt \
  --inspect-layers 0,13,26 \
  --min-free-gb 5 \
  --out-json /root/autodl-tmp/tca-reasoning/stage2f_cross_model/stage2f_qwen_download_manifest.json \
  --out-csv /root/autodl-tmp/tca-reasoning/stage2f_cross_model/stage2f_qwen_clt_layer_shapes.csv
```

结果：

```text
decision.status = pass
27 / 27 layer files downloaded
single layer size = 117,464,384 bytes
total CLT layer bytes ≈ 3.17GB decimal
load_transcoder_from_hub(..., lazy_encoder=True, lazy_decoder=True) = ok
loaded class = TranscoderSet
n_layers = 27
d_transcoder = 8192
```

抽查 tensor shape：

```text
layer_0 / layer_13 / layer_26:
W_dec: [8192, 3584], dtype bfloat16
W_enc: [8192, 3584], dtype bfloat16
b_dec: [3584], dtype bfloat16
b_enc: [8192], dtype bfloat16
```

解释：

```text
Qwen2.5-VL CLT 资产本身可下载、可读 shape、可 lazy load。
这比 Stage 2F-1 的 asset-level partial smoke 更进一步，因为这次不是只读 repo file list，而是已经完成全量 CLT 权重下载和 lazy loader 验证。
```

### 5.2 Base model 直接 HF 下载尝试

最初计划直接用 Hugging Face 下载：

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

下载前检查：

```text
/root/autodl-tmp 可用空间充足，约 149GB
所需 base 权重约 16GB
```

首次 HF 下载现象：

```text
processor/tokenizer/config 可下载
开始下载 5 个 safetensors shard
model-00003 / model-00004 / model-00005 最终可落入 HF cache
model-00001 / model-00002 在 Xet CAS 下载路径上速度极慢或遇到 503
```

观察到的典型问题：

```text
HTTP 503 from cas-bridge.xethub.hf.co
部分 shard 下载速度下降到几百 KB/s
长时间等待不经济
```

这一步的判断：

```text
HF/Xet 路线不是资产不可用，而是服务器到 Xet CAS 的下载稳定性不足。
```

### 5.3 依赖修复

base loader smoke 暴露出两个依赖问题。

第一，`AutoProcessor` 需要 `torchvision`：

```text
错误类型：ImportError
原因：AutoVideoProcessor requires the Torchvision library
```

服务器 torch 版本：

```text
torch 2.11.0+cu130
cuda 13.0
```

首次安装 `torchvision 0.27.0` 后导入失败：

```text
RuntimeError: operator torchvision::nms does not exist
```

原因判断：

```text
torchvision wheel 与当前 torch ABI 不匹配。
```

修复：

```bash
.venv/bin/python -m pip install --force-reinstall --no-deps torchvision==0.26.0
```

验证结果：

```text
torchvision 0.26.0+cu130
torchvision_import_ok = True
```

第二，`hf_transfer` 未安装：

```text
错误类型：ValueError
原因：HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer package is not available
```

修复：

```bash
.venv/bin/python -m pip install hf_transfer
```

验证结果：

```text
hf_transfer_present = True
```

此外，`run_qwen_base_asset_smoke.py` 里 meta-model 构造也做了兼容修复：

```text
旧写法：Qwen2_5_VLForConditionalGeneration.from_config(config)
问题：当前 transformers 类没有 from_config 方法
新写法：Qwen2_5_VLForConditionalGeneration(config)
```

该修复只影响 smoke 脚本，不改变主 Gemma3 pipeline。

### 5.4 ModelScope fallback 补齐缺失 shard

由于 HF/Xet 路线卡在 `model-00001` 和 `model-00002`，改用 ModelScope 下载缺失权重。

安装：

```bash
.venv/bin/python -m pip install modelscope
```

版本：

```text
modelscope 1.37.0
```

下载命令逻辑：

```bash
.venv/bin/python -m modelscope.cli.cli download \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --local_dir /root/autodl-tmp/tca-reasoning/data/modelscope_qwen2p5vl_7b_instruct \
  --max-workers 4 \
  model-00001-of-00005.safetensors model-00002-of-00005.safetensors
```

结果：

```text
model-00001-of-00005.safetensors = 3,900,233,256 bytes
model-00002-of-00005.safetensors = 3,864,726,320 bytes
download status = success
typical speed ≈ 15-18 MB/s per shard
```

随后将两个完整 shard 覆盖到 HF snapshot 目录：

```text
/root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/
```

最终 snapshot 内 5 个权重文件：

```text
model-00001-of-00005.safetensors
model-00002-of-00005.safetensors
model-00003-of-00005.safetensors
model-00004-of-00005.safetensors
model-00005-of-00005.safetensors
```

## 6. 最终结果

最终 base loader smoke 使用本地 snapshot path，而不是再次请求 HF：

```text
model_name = /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5
input_is_local_path = true
```

最终判定：

```text
decision.status = pass
processor_ok = true
full_base_assets_ok = true
meta_model_ok = true
```

processor：

```text
processor_class = Qwen2_5_VLProcessor
tokenizer_class = Qwen2TokenizerFast
```

config：

```text
model_type = qwen2_5_vl
architecture = Qwen2_5_VLForConditionalGeneration
hidden_size = 3584
num_hidden_layers = 28
vocab_size = 152064
```

meta-model：

```text
class = Qwen2_5_VLForConditionalGeneration
parameter_count_meta = 8,292,166,656
```

本地 snapshot 文件统计：

```text
file_count = 15
weight_file_count = 5
weight_total_bytes = 16,584,414,560
total_bytes = 16,595,979,762
```

磁盘状态：

```text
最终 HF cache 所在分区剩余约 124GB
```

## 7. 预期与实际偏差

预期：

```text
HF cache 直接下载 Qwen base model
AutoProcessor local_files_only 成功
base config/meta smoke 成功
```

实际：

```text
HF/Xet 对 model-00001 / model-00002 不稳定，出现 503 和极低速
torchvision 缺失，需要补依赖
torchvision 0.27.0 与 torch 2.11.0+cu130 不匹配，需要回退到 0.26.0
hf_transfer 未安装，需要补依赖
最终通过 ModelScope fallback 补齐缺失 shard
```

偏差解释：

```text
这些偏差都是下载/依赖/loader 层面的工程问题，不是 Qwen CLT 或 Qwen base model 的机制结果失败。
```

## 8. 结论

本轮 Stage 2F-2 下载与 lazy load 判定为：

```text
CLT asset download + lazy load: pass
Qwen base local asset smoke: pass
```

可以保守说明：

```text
KokosDev/qwen2p5vl-7b-clt 与 Qwen/Qwen2.5-VL-7B-Instruct 的资产层面已经具备进入下一步 hook-forward smoke 的条件。
```

仍不能说明：

```text
Qwen 已经支持当前 circuit-tracer ReplacementModel pipeline
Qwen 能直接跑 attribution / feature intervention
Qwen 复现了 Gemma3 上的 evidence-region-sensitive support routes
跨模型机制复现已经成立
```

下一步：

```text
运行 native Qwen hook-forward smoke，确认图文输入、hidden states 和普通 PyTorch hook activation shape 可读。
若该 smoke 通过，后续才设计 Qwen adapter / hook mapping / minimal intervention smoke。
```
