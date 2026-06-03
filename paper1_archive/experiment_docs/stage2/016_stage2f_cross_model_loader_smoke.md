# Stage 2F-1：Cross-Model Loader Smoke

日期：2026-05-20

## 1. 实验目的

Stage 2F-1 的目的不是证明跨模型机制复现，而是做一个很窄的兼容性烟测：

```text
现成 Qwen2.5-VL CLT / transcoder 资产是否能进入我们的后续方法链？
```

本轮只检查资产层面和 loader 风险，不加载完整 Qwen2.5-VL-7B base model，不跑 attribution，不跑 intervention，不宣称任何跨模型机制结论。

固定候选：

```text
KokosDev/qwen2p5vl-7b-clt
base model: Qwen/Qwen2.5-VL-7B-Instruct
```

为什么选它：

1. Hugging Face repo 有 `config.yaml`；
2. 有按层保存的 `layer_*.safetensors`；
3. config 里写明 base model、层数、hidden dim、feature dim、hook point；
4. 比 LLaVA 候选更接近当前 cross-layer transcoder 资产格式；
5. 适合作为 Stage 2F 的第一候选，但不能直接替代当前 Gemma3 主 pipeline。

## 2. 术语解释

`cross-model loader smoke`：跨模型加载烟测。意思是只验证另一套模型 / transcoder 资产能不能被发现、读取 config、列出权重文件、识别 hook metadata，并粗略判断当前代码是否可能适配。它不是正式实验复现。

`CLT`：Cross-Layer Transcoder，跨层转码器。通常把模型某些层的 hidden activation 映射到稀疏 feature，再从 feature 重构或影响后续残差流。这里的重点是它是否有明确 layer 文件和 hook 点。

`hook point`：模型内部可以读写 activation 的位置，例如 `blocks.{layer}.hook_resid_pre` 和 `blocks.{layer}.hook_resid_post`。如果 hook 点和当前模型 wrapper 不兼容，就不能直接跑 attribution / intervention。

`ReplacementModel`：我们当前 pipeline 中把 base VLM 与 transcoder 连接起来的包装模型。当前代码仍主要走 Gemma3 路径，所以 Qwen2.5-VL 不能默认直接运行。

`partial`：在本实验里表示 config 和远端权重列表可读，但因为不下载完整 safetensors、且当前没有 Qwen adapter，所以没有完成 lazy CLT load 或完整 pipeline load。

## 3. 输入

本地脚本：

```text
E:\Bridging\vlm-circuit-tracing\circuit_tracer_vlm\scripts\research\run_cross_model_asset_loader_smoke.py
```

远端运行环境：

```text
/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm
/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm/.venv/bin/python
HF_HOME=/root/autodl-tmp/tca-reasoning/data/hf_cache
```

候选资产：

```text
Hugging Face repo: KokosDev/qwen2p5vl-7b-clt
```

本轮不使用：

```text
Qwen/Qwen2.5-VL-7B-Instruct full base model
完整 CLT safetensors 下载
任何 OK-VQA 样本
任何 attribution / feature intervention / region-mask pipeline
```

## 4. 输出

正式输出：

```text
E:\Bridging\doc\experiments\stage2\016_stage2f_cross_model_loader_smoke.md
E:\Bridging\doc\experiments\stage2\cross_model\stage2f_qwen2p5vl_clt_loader_smoke.json
E:\Bridging\doc\experiments\stage2\cross_model\stage2f_qwen2p5vl_clt_layer_shapes.csv
```

远端中间输出：

```text
/root/autodl-tmp/tca-reasoning/stage2f_cross_model/stage2f_qwen2p5vl_clt_loader_smoke.json
/root/autodl-tmp/tca-reasoning/stage2f_cross_model/stage2f_qwen2p5vl_clt_layer_shapes.csv
```

脚本状态：

```text
python -m py_compile 通过
CLI --help 通过
远端 light smoke 通过并返回 status=partial
```

## 5. 方法

### 5.1 脚本新增

新增只读脚本：

```text
run_cross_model_asset_loader_smoke.py
```

核心检查项：

1. 读取 HF `config.yaml`；
2. 记录 `model_kind`、`architecture`、`model_name`、`n_layers`、`hidden_dim`、`feature_dim`；
3. 记录 `feature_input_hook` 和 `feature_output_hook`；
4. 列出 repo 中的 `.safetensors` 文件；
5. 解析 `layer_*.safetensors` 的层号覆盖情况；
6. 记录每个 layer 文件的远端大小；
7. 只检查本地 cache 中是否已有完整 safetensors，不主动下载完整权重；
8. 只在完整 safetensors 已缓存或显式打开下载开关时尝试 lazy CLT load；
9. 检查当前 `ReplacementModel` 是否仍绑定 Gemma3；
10. 可选检查 `AutoProcessor.from_pretrained(..., local_files_only=True)`，但不下载完整 base model。

### 5.2 安全约束

本轮明确禁止自动下载完整 Qwen base model。

初始脚本曾允许最多下载 1 个 safetensors 用于 shape inspection。远端第一次运行时发现：

```text
layer_0.safetensors size = 117,464,384 bytes
```

由于服务器下载速度很慢，且本轮目标只是 light smoke，我主动停止该进程，并修改脚本：

```text
--max-shape-downloads 默认从 1 改为 0
未知文件大小时拒绝 shape download
shape inspection 只对已有本地完整缓存执行
新增阶段日志，避免远端长时间无输出
```

因此最终正式输出不包含 tensor name / tensor shape，只包含远端文件列表、层号、远端大小和 shape inspection 状态。这个取舍是有意的：本轮优先避免占用空间和长时间下载。

中断的首次远端下载留下了一个 10,485,760 bytes 的 `.incomplete` 缓存文件，路径位于 Qwen CLT 的 HF cache blobs 目录中；确认路径安全后已删除。最终服务器没有保留完整或半完整 `layer_0.safetensors`。

### 5.3 本地检查

本地语法检查：

```powershell
python -m py_compile E:\Bridging\vlm-circuit-tracing\circuit_tracer_vlm\scripts\research\run_cross_model_asset_loader_smoke.py
```

结果：

```text
通过
```

本地 CLI 检查：

```powershell
python E:\Bridging\vlm-circuit-tracing\circuit_tracer_vlm\scripts\research\run_cross_model_asset_loader_smoke.py --help
```

结果：

```text
通过，参数包括 --transcoder-set、--out-json、--out-csv、--max-shape-downloads、--try-processor-local 等。
```

### 5.4 本地 HF 访问尝试

本机尝试直接访问 HF：

```text
transcoder_set = KokosDev/qwen2p5vl-7b-clt
```

结果：

```text
blocked
```

原因：

```text
本地 DNS 无法解析 huggingface.co
config.yaml 无法从本机下载
repo file list 无法从本机读取
```

这个结果只说明本机网络不可用，不说明 Qwen CLT 资产不可用。

### 5.5 远端正式 light smoke

远端命令逻辑：

```bash
cd /root/autodl-tmp/tca-reasoning/circuit_tracer_vlm
source scripts/server/dev.sh
export PYTHONPATH=/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm:${PYTHONPATH:-}
export HF_HUB_ETAG_TIMEOUT=20
export HF_HUB_DOWNLOAD_TIMEOUT=20

.venv/bin/python -u scripts/research/run_cross_model_asset_loader_smoke.py \
  --transcoder-set KokosDev/qwen2p5vl-7b-clt \
  --max-shape-downloads 0 \
  --try-processor-local \
  --out-json /root/autodl-tmp/tca-reasoning/stage2f_cross_model/stage2f_qwen2p5vl_clt_loader_smoke.json \
  --out-csv /root/autodl-tmp/tca-reasoning/stage2f_cross_model/stage2f_qwen2p5vl_clt_layer_shapes.csv
```

注意：

```text
--try-processor-local 只做 processor 本地可用性检查，不加载完整 Qwen base model。
--max-shape-downloads 0 保证不主动下载 safetensors 权重。
```

## 6. 结果

### 6.1 config.yaml 结果

正式 JSON 记录：

```text
config status = ok
model_kind = transcoder_set
architecture = qwen2.5-vl
model_name = Qwen/Qwen2.5-VL-7B-Instruct
n_layers = 27
hidden_dim = 3584
feature_dim = 8192
feature_input_hook = blocks.{layer}.hook_resid_pre
feature_output_hook = blocks.{layer}.hook_resid_post
file_pattern = layer_{layer}.safetensors
layers.start = 0
layers.end = 26
layers.total = 27
```

重要偏差：

```text
之前候选表里把它概括成 model_kind = cross_layer_transcoder。
实际 config.yaml 里 model_kind = transcoder_set，architecture = qwen2.5-vl。
```

解释：

```text
这不是资产失败，而是 metadata 口径不同。
后续脚本不能只靠 model_kind == cross_layer_transcoder 判断是否为 CLT-like 资产。
应结合 file_pattern、layers、hook metadata 和具体 loader 格式判断。
```

### 6.2 repo file list 结果

正式 JSON 记录：

```text
repo file status = ok
total_files = 57
safetensors_count = 27
layer_file_count = 27
w_enc_count = 0
w_dec_count = 0
expected_layers = 27
observed_indexed_layers = 0..26
missing_layers = []
extra_layers = []
```

解释：

```text
远端 HF repo 中存在完整 27 层 layer 文件。
从文件存在性和层号覆盖角度看，Qwen2.5-VL CLT 资产不是 blocked。
```

### 6.3 layer file size 结果

CSV 共 27 行，每行一个 `layer_*.safetensors`。

每层大小：

```text
117,464,384 bytes
```

总大小约：

```text
117,464,384 * 27 = 3,171,538,368 bytes
约 3.17 GB decimal
约 2.95 GiB
```

本轮不下载完整权重，因此 CSV 中：

```text
shape_status = not_inspected
tensor_shapes = empty
local_path = empty
```

解释：

```text
这不是 shape inspection 失败，而是 light smoke 的安全策略。
如果后续要检查 tensor names / tensor shapes，可以单独允许下载 1 个 layer 文件，成本约 117MB。
```

### 6.4 本地 cache 结果

正式 JSON 记录：

```text
safetensors_snapshot_status = ok
safetensors_snapshot_path = /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--KokosDev--qwen2p5vl-7b-clt/snapshots/6895a00ad9df06cd02ecbbc6feecce47c442f984
local_safetensors_count = 0
local_safetensors_files = []
safetensors_snapshot_complete = false
```

解释：

```text
服务器 cache 中已有 config snapshot 目录，但没有完整 safetensors 权重文件。
因此不能在不下载权重的前提下尝试真正的 lazy CLT load。
```

### 6.5 lazy CLT load 结果

正式 JSON 记录：

```text
lazy_clt_load.status = skipped_not_cached
reason = Full transcoder safetensors snapshot is not cached and --allow-full-transcoder-download is false.
local_safetensors_count = 0
remote_safetensors_count = 27
```

解释：

```text
lazy CLT load 没有失败；它被安全策略跳过。
如果后续明确允许下载约 3GB CLT 权重，可以再尝试 load_transcoder_from_hub(..., lazy_encoder=True, lazy_decoder=True)。
但本轮不做这个，因为 Stage 2F-1 的目标不是完整 loader。
```

### 6.6 processor local smoke 结果

正式 JSON 记录：

```text
processor_local_smoke.attempted = true
status = failed
error_type = TypeError
error = expected str, bytes or os.PathLike object, not NoneType
```

远端日志同时显示读取了一个很小的 processor metadata 文件：

```text
preprocessor_config.json: 350B
```

解释：

```text
这不是 base model load。
它说明当前服务器没有形成一个完整、可直接 local_files_only 加载的 Qwen2.5-VL processor cache。
后续如果要做 Qwen adapter，需要单独处理 processor/tokenizer/chat-template/image processor 资产。
```

### 6.7 ReplacementModel backend 结果

正式 JSON 记录：

```text
replacement_model_import_ok = true
uses_gemma3_for_conditional_generation = true
uses_auto_model = false
qwen_adapter_present = false
note = ReplacementModel.from_pretrained_and_transcoders is Gemma3-oriented
```

解释：

```text
当前主 pipeline 仍然是 Gemma3-oriented。
即使 Qwen CLT 资产文件齐全，也不能直接用当前 ReplacementModel 跑 Qwen attribution / intervention。
必须先写 Qwen adapter 或最小 hook-forward smoke。
```

## 7. 预期与实际偏差

### 7.1 符合预期的部分

符合：

```text
config.yaml 可读；
base model metadata 可读；
hook metadata 可读；
layer_*.safetensors 文件列表可读；
27 层覆盖完整；
没有自动下载完整 Qwen base model；
ReplacementModel Gemma3-only 风险被明确记录。
```

这些结果说明：

```text
KokosDev/qwen2p5vl-7b-clt 是一个真实可见的跨模型候选资产。
Stage 2F 没有在资产发现阶段 blocked。
```

### 7.2 不符合预期的部分

偏差 1：

```text
候选表预期 model_kind = cross_layer_transcoder。
实际 config.yaml 是 model_kind = transcoder_set，architecture = qwen2.5-vl。
```

影响：

```text
后续 adapter 不能写死 model_kind == cross_layer_transcoder。
```

偏差 2：

```text
服务器没有完整 safetensors cache。
```

影响：

```text
不能在 light smoke 中尝试 lazy CLT load。
如果要继续，需要明确承担约 3GB CLT 权重下载成本。
```

偏差 3：

```text
AutoProcessor local smoke 失败。
```

影响：

```text
即使后续下载 CLT 权重，Qwen base pipeline 仍需要 processor / tokenizer / image processing adapter。
```

偏差 4：

```text
当前 ReplacementModel 仍 Gemma3-only。
```

影响：

```text
Qwen attribution / intervention 不能直接运行。
这是真正的下一阶段工程门槛。
```

## 8. 判定

本轮正式判定：

```text
partial
```

为什么不是 pass：

```text
没有完整 safetensors 本地 cache；
没有执行 lazy CLT load；
没有 Qwen adapter；
没有 base model loader smoke；
没有 hook-forward smoke。
```

为什么不是 blocked：

```text
config 可读；
repo file list 可读；
27 个 layer 文件完整可见；
hook metadata 完整；
远端大小可读；
ReplacementModel 风险已定位清楚。
```

最保守读法：

```text
Qwen2.5-VL CLT 是可继续推进的跨模型候选资产，但当前只通过了资产级 light smoke。
它还不能进入我们的 attribution / intervention / region-mask 主实验链。
```

## 9. 对主 claim 的影响

本轮不支持任何新的机制结论。

不能说：

```text
Qwen2.5-VL 上也有 evidence-region-sensitive support routes；
跨模型复现已经完成；
Qwen adapter 已经可用；
VLM 机制具有跨模型普遍性。
```

可以说：

```text
我们已经确认一个 Qwen2.5-VL-7B CLT 候选资产在 metadata 和远端权重列表层面可读；
它具备继续做跨模型工程适配的基本条件；
当前 blocked 点不是资产不存在，而是完整权重缓存、processor/base model 适配和 ReplacementModel/Qwen hook adapter。
```

对主论文路线的意义：

```text
Stage 2F 可以继续作为“跨模型可行性”支线推进；
但它不能替代 Stage 2A/2B 的 Gemma3 主证据；
下一步若要冲更强论文，cross-model 仍至少需要 loader/hook smoke 后再谈 mini replication。
```

## 10. 后续动作

建议按风险从低到高推进。

### 10.1 低成本下一步：更新候选表和 loader 规则

需要把候选表中 Qwen CLT 的 metadata 修正为：

```text
model_kind = transcoder_set
architecture = qwen2.5-vl
file_pattern = layer_{layer}.safetensors
```

并把 loader 判断从：

```text
model_kind == cross_layer_transcoder
```

改成：

```text
file_pattern + layers + hook metadata + explicit adapter rule
```

### 10.2 可选下一步：单层 safetensors header / shape smoke

如果需要 tensor names / shapes，可以只下载一个 layer：

```text
layer_0.safetensors
size = 117,464,384 bytes
```

成功后记录：

```text
tensor names
tensor shapes
dtype
是否与 hidden_dim=3584 / feature_dim=8192 一致
```

注意：

```text
这会增加约 117MB 下载和缓存，不是必须。
```

### 10.3 中等成本下一步：完整 CLT lazy load

如果明确允许下载完整 CLT：

```text
27 layers * 117,464,384 bytes
约 3.17GB
```

再尝试：

```python
load_transcoder_from_hub(
    "KokosDev/qwen2p5vl-7b-clt",
    lazy_encoder=True,
    lazy_decoder=True,
)
```

这一步只能证明 CLT loader 可以读权重，仍不等于 Qwen base model pipeline 可跑。

### 10.4 高成本下一步：Qwen adapter / hook-forward smoke

真正进入 cross-model pipeline 需要：

1. Qwen2.5-VL base model loader；
2. Qwen processor / tokenizer / image processor；
3. Qwen hook point 到当前 `blocks.{layer}.hook_resid_pre/post` 的映射；
4. `ReplacementModel` 或新 wrapper 支持 Qwen；
5. 最小图文输入 forward；
6. 读取一层 activation；
7. 对一个 feature 做最小 intervention；
8. 只在这些通过后，才考虑 1-case attribution smoke。

### 10.5 备用路线

如果 Qwen CLT adapter 成本过高，备用候选：

```text
KokosDev/qwen2p5vl-7b-plt
```

但它也需要同样的 config/shape/hook compatibility smoke。

LLaVA 候选暂时不优先，因为格式和当前 loader 差异更大。

## 11. 本轮结论

Stage 2F-1 的结论是：

```text
Qwen2.5-VL-7B CLT 候选资产通过了轻量资产层 smoke：
config 可读、27 层 layer 文件完整可见、hook metadata 完整、文件大小可读。
但由于本轮不下载完整 CLT 权重、服务器没有完整 safetensors cache、AutoProcessor local check 未通过、当前 ReplacementModel 仍 Gemma3-only，因此不能直接跑 Qwen attribution / intervention。
```

最终状态：

```text
partial asset-level pass;
full pipeline blocked by adapter/cache/base-model issues.
```

最短中文读法：

```text
资产是真的，结构也清楚；但现在还不能跑 Qwen 机制实验。下一步不是扩大样本，而是先做 Qwen adapter / hook-forward smoke，或者明确批准约 3GB CLT 权重下载后做完整 lazy-loader 检查。
```
