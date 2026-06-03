# Stage 2F Cross-Model Run Plan：Qwen 与 Llama-family 路线

日期：2026-05-20

## 0. 一句话结论

本阶段的 cross-model 目标不是立刻证明“我们的机制结论已经跨模型复现”，而是按风险从低到高验证：

```text
现成 VLM circuit / transcoder 资产
→ 是否能加载
→ 是否能读到目标 hook / hidden state
→ 是否能把 hidden state 送入对应 transcoder
→ 是否能做最小 feature readout / feature intervention
→ 是否有资格进入 3-5 case 的 evidence-region mini replication
```

Qwen2.5-VL 是主路线；LLaVA-1.5 是 Llama-family VLM 备用路线；纯 Llama 语言模型只作为方法 sanity check，不进入视觉证据主结论。

## 1. 为什么要做 cross-model

当前主线结论来自 Gemma3-VLM backbone：

```text
在 localized、strong image-dependence 的 VQA 样本中，
answer-adjacent support routes 对 wrong image 和人工 answer / union evidence mask 敏感，
这种敏感性强于 nearest non-source node control 和 random region control，
并且区域遮挡会更频繁改变最终生成答案。
```

这个结论已经足以支撑一个收窄后的机制 claim，但如果要往更强论文推进，需要回答一个自然质疑：

```text
这是不是 Gemma3 + 当前 transcoder pipeline 的特例？
```

Stage 2F 的作用就是逐步降低这个质疑。它不是替代主线，也不是现在就扩大成完整跨模型论文，而是建立一条可执行的外部有效性路线。

## 2. 术语解释

`VLM`：Vision-Language Model，视觉语言模型，能同时输入图片和文本，并生成文本回答。

`Qwen2.5-VL`：通义千问视觉语言模型系列。本阶段使用 `Qwen/Qwen2.5-VL-7B-Instruct` 作为 Qwen 路线 base model。

`LLaVA`：Large Language and Vision Assistant，一类基于 Llama-family language model 加视觉编码器/投影器的 VLM。本计划把它作为“Llama-family VLM”路线，而不是把纯 Llama 语言模型当成视觉模型。

`CLT`：Cross-Layer Transcoder，跨层转码器。它把模型某层 residual hidden state 映射到稀疏 feature 空间，用于 circuit tracing。

`PLT`：Per-Layer Transcoder，逐层转码器。每层独立映射 hidden state 到 feature 空间。

`hook`：模型前向传播中可捕获 activation 的位置，例如某一层 residual stream 的输入/输出。

`ReplacementModel`：当前 Gemma3 主 pipeline 中，把 base VLM 和 transcoder 连接起来以支持 attribution / intervention 的包装模型。目前主要是 Gemma3-oriented，不能假设 Qwen 或 LLaVA 能直接接入。

`feature readout smoke`：只读取 hidden state，并送入 transcoder encoder 得到 feature activations。它不做 attribution graph，不做节点干预，也不证明机制复现。

`mini replication`：在新模型上用 3-5 个 case 尝试复现主线链条，例如 answer mask weakening、source > control、decoded answer change。只有 adapter 和 readout/intervention smoke 通过后才进入这一步。

## 3. 总体边界

### 3.1 本阶段可以证明什么

如果 Stage 2F 顺利，本阶段最多能逐步证明：

```text
Qwen 或 LLaVA 具有可用的公开 VLM transcoder 资产；
这些资产的 hidden_dim / layer count / hook metadata 与 base VLM activation 大体对齐；
我们能在 native VLM forward 中读到对应层 hidden state；
我们能把 hidden state 送入对应 transcoder，得到稀疏 feature activation；
后续有工程路径把它接入 attribution / intervention / region-mask pipeline。
```

如果继续推进到后半段并成功，才可以说：

```text
新模型上存在与 Gemma3 主线类似的 evidence-region-sensitive answer-support route 初步信号。
```

### 3.2 本阶段不能过早证明什么

本阶段早期 smoke 不能证明：

```text
Qwen / LLaVA 已经复现 Gemma3 的主机制结论；
Qwen / LLaVA 的 feature 与 Gemma3 feature 可一一对应；
视觉证据提示在跨模型上更好；
某个 feature 是对象级语义节点；
纯 Llama 语言模型可以支撑视觉证据区域敏感性 claim。
```

## 4. 模型路线选择

## 4.1 Qwen 主路线

候选资产：

```text
transcoder repo = KokosDev/qwen2p5vl-7b-clt
base model = Qwen/Qwen2.5-VL-7B-Instruct
asset type = CLT / transcoder_set style layer_*.safetensors
n_layers in CLT asset = 27
base language hidden layers = 28
hidden_dim = 3584
feature_dim = 8192
feature_input_hook = blocks.{layer}.hook_resid_pre
feature_output_hook = blocks.{layer}.hook_resid_post
```

已完成状态：

```text
Qwen CLT 27/27 layer files downloaded。
load_transcoder_from_hub(..., lazy_encoder=True, lazy_decoder=True) 成功。
Qwen base model processor/tokenizer/config/base weights 本地可用。
native Qwen 图文 forward 成功。
hidden_states_count = 29。
selected hidden shape = [1, 440, 3584]。
native hook module = model.language_model.layers.0。
当前状态 = native forward partial success，但缺 Qwen ReplacementModel adapter。
```

判断：

```text
Qwen 是 Stage 2F 第一优先级。
原因不是它已经能做机制复现，而是它最接近“现有资产能进入方法链”的状态。
```

## 4.2 LLaVA / Llama-family VLM 备用路线

候选资产：

```text
transcoder repo = KokosDev/llava15-7b-clt
base model = llava-hf/llava-1.5-7b-hf
asset type = CLT-like custom format
observed format = transcoder_L*.pt + mapping_L*.pt
standard config.yaml = absent or not current-loader-compatible
```

判断：

```text
LLaVA 是 Llama-family 的视觉路线。
它可以回答 cross-model VLM feasibility 问题，
但文件格式、base wrapper、vision projector、hook naming 都比 Qwen 风险更高。
因此 LLaVA 不应该抢在 Qwen feature readout 前面。
```

## 4.3 纯 Llama 语言模型路线

可能资产：

```text
Llama-3.2 language-only transcoders
Gemma-2 language-only transcoders
Qwen language-only transcoders
```

判断：

```text
纯 Llama 没有图像输入，也没有 image-region mask。
它不能验证 evidence-region-sensitive visual route。
它只能作为 attribution / intervention 方法 sanity check。
```

因此计划中“Llama 上的实验”默认指：

```text
优先：LLaVA-1.5-7B 这条 Llama-family VLM 路线。
可选：纯 Llama LM 只做语言模型方法 sanity，不进入主视觉 claim。
```

## 5. Qwen 实验计划

### Q0：资产与 native forward smoke（已完成）

目的：

```text
验证 Qwen CLT 和 Qwen base model 是否能在服务器上落盘、加载、前向传播。
```

已产出：

```text
doc/experiments/stage2/017_stage2f_qwen_download_and_lazy_load.md
doc/experiments/stage2/018_stage2f_qwen_hook_forward_smoke.md
doc/experiments/stage2/cross_model/stage2f_qwen_download_manifest.json
doc/experiments/stage2/cross_model/stage2f_qwen_base_loader_smoke.json
doc/experiments/stage2/cross_model/stage2f_qwen_hook_forward_smoke.json
```

当前判定：

```text
Qwen asset / lazy load = pass
Qwen native hook-forward = partial
partial 原因 = native Qwen forward 成功，但缺 Qwen ReplacementModel adapter
```

### Q1：CLT feature readout smoke（当前立即执行）

目的：

```text
验证 Qwen native hidden state 是否能送入 KokosDev/qwen2p5vl-7b-clt 的对应层 encoder。
```

输入：

```text
Qwen local base snapshot
KokosDev/qwen2p5vl-7b-clt
一个 OK-VQA image-question
layers = 0, 13, 26
```

方法：

```text
1. 用 native Qwen processor 构造图文输入。
2. 用 Qwen base model forward 得到 hidden_states。
3. 取 layer+1 的 hidden state 作为近似 residual-post-layer / next-layer input readout。
4. 对每个指定 layer 调用 transcoder.encode_layer(hidden, layer_id)。
5. 记录 feature shape、top-k feature id、activation value、active feature count、dtype、device。
```

成功标准：

```text
hidden_dim = 3584 与 CLT encoder input dim 对齐。
encode_layer 不报错。
features shape = [batch, seq, 8192] 或等价形状。
至少能读出非零 top features。
```

判定边界：

```text
如果通过，只能说明 Qwen hidden state → CLT feature readout 可行。
它仍不是 attribution / intervention，也不是机制复现。
```

### Q2：Qwen token / position mapping smoke

目的：

```text
把 Qwen feature activation 与 text token、image token、assistant answer prefix 的位置关系对齐。
```

方法：

```text
1. 解析 processor 输出的 input_ids、image_grid_thw、chat template。
2. 标记 image-token span、question-token span、generation-prefix span、last prompt token。
3. 分别汇总 image positions、text positions、last token position 的 top active features。
4. 检查 top features 是否主要集中在合理位置，而不是由 padding / special tokens 主导。
```

成功标准：

```text
能稳定定位至少 text/question positions 与 last prompt position。
如果 image-token span 可定位，则额外记录 image positions。
```

### Q3：Qwen clean vs answer-mask feature readout

目的：

```text
在不做 intervention 的前提下，先检查 answer region mask 是否会改变 Qwen CLT feature activations。
```

候选样本：

```text
优先使用 Gemma3 主线中已有 answer/union mask 且 localized 的 2-3 个 case。
```

方法：

```text
1. 对同一 sample 跑 clean image 与 answer_mask image。
2. 读取同一 layers / positions 的 CLT feature activation。
3. 比较 top support-like feature activation 是否下降。
4. 暂时不说 source route，只说 feature readout sensitivity。
```

成功标准：

```text
answer_mask 相比 clean 出现方向一致的 feature activation shift。
```

判定边界：

```text
这一步仍然不是 causal route。
它只是为后续 Qwen attribution / intervention adapter 提供 evidence-region sensitivity 的 readout 依据。
```

### Q4：Qwen adapter / minimal intervention feasibility

目的：

```text
建立 Qwen native module names 与 CLT metadata hook names 的映射，并尝试最小节点干预。
```

需要解决：

```text
blocks.{layer}.hook_resid_pre/post
↔ native Qwen model.language_model.layers.{layer} input/output
```

最低可行版本：

```text
不一开始重写完整 ReplacementModel。
先写一个 QwenNativeActivationAdapter：
读取 hidden state；
在指定 layer/position 上做 residual zeroing 或 feature-direction projection；
比较 target token logit / first-token probability。
```

成功标准：

```text
能在 1 个样本、1 个 layer、1 个 position 上完成 source-like activation intervention。
能记录 target logit/rank/margin 的变化。
```

### Q5：Qwen mini evidence-region replication

进入条件：

```text
Q1 feature readout pass。
Q2 position mapping pass。
Q3 clean vs mask readout 有方向信号。
Q4 minimal intervention 至少 partial。
```

设计：

```text
3-5 个 localized strong-image-dependence samples。
conditions = clean / answer_mask / union_mask / random_control。
nodes = Qwen top active answer-adjacent feature candidates + matched feature/position controls。
metrics = feature activation weakening, target rank damage, answer changed。
```

成功标准：

```text
至少 2-3 个 case 方向与 Gemma3 主线一致。
```

## 6. LLaVA / Llama-family VLM 实验计划

### L0：LLaVA asset format smoke

目的：

```text
确认 KokosDev/llava15-7b-clt 的文件格式、层数、tensor names、mapping 文件含义。
```

输入：

```text
KokosDev/llava15-7b-clt
```

方法：

```text
1. 用 Hugging Face API 读取 repo file list。
2. 下载最小必要文件，优先一个 transcoder_L0.pt 和一个 mapping_L0.pt。
3. 用 torch.load(..., map_location="cpu") 读取 tensor keys / shape / dtype。
4. 判断是否能写 adapter，把该格式转换成当前 TranscoderSet / CLT loader 可读结构。
```

成功标准：

```text
至少能读出 layer 0 的 encoder/decoder 或等价 tensor。
能明确 mapping_L*.pt 的作用。
```

### L1：LLaVA base processor/config smoke

目的：

```text
确认 llava-hf/llava-1.5-7b-hf 的 processor/config 是否可用。
```

方法：

```text
先只下载 processor/tokenizer/config。
如果磁盘和网络都允许，再下载 base weights。
```

成功标准：

```text
AutoProcessor.from_pretrained(...) 成功。
base config 可读，hidden_size / layer_count 可读。
```

### L2：LLaVA native hook-forward smoke

目的：

```text
验证 LLaVA base model 能跑一个图文 forward，并读到 language model layer hidden state 或 hook activation。
```

成功标准：

```text
forward ok。
hidden state shape 可读。
layer hook module 可定位。
hidden_dim 与 LLaVA CLT tensor input dim 对齐。
```

### L3：LLaVA transcoder readout adapter

目的：

```text
如果 L0 确认 .pt/mapping 可读，则尝试把 LLaVA hidden state 送入对应 transcoder。
```

成功标准：

```text
单层 feature readout 可行。
top features 可读。
```

### L4：LLaVA mini replication

进入条件：

```text
L0-L3 通过或 partial 但工程风险可控。
```

设计：

```text
最多 2-3 个 localized samples。
先做 readout-level clean vs answer_mask。
只有 readout 成功后才尝试 intervention。
```

## 7. 纯 Llama 实验计划

纯 Llama 不作为视觉主线，但可以做一个可选 sanity：

```text
问题：当前 attribution / intervention 基础方法是否能在另一个 language-only model 上运行？
```

可做实验：

```text
Llama-3.2-1B language-only transcoder loader smoke。
text-only prompt 的 feature readout / token logit intervention smoke。
```

不做内容：

```text
不做 image mask。
不做 VQA evidence-region claim。
不把结果写进 strongest multimodal evidence。
```

## 8. 优先级排序

本轮执行顺序固定为：

```text
1. Qwen Q1：CLT feature readout smoke。
2. LLaVA L0：asset format smoke。
3. Qwen Q2：token / position mapping smoke。
4. Qwen Q3：clean vs answer-mask feature readout。
5. LLaVA L1-L2：base loader + native hook-forward smoke。
6. Qwen Q4：minimal adapter / intervention feasibility。
7. Qwen Q5 或 LLaVA L4：mini replication。
```

理由：

```text
Qwen 已经下载并 native forward 通过，继续 Q1 是最短路径。
LLaVA 还停留在 asset survey，先 L0 才知道值不值得下载 base。
纯 Llama 不优先，因为它不能回答视觉证据区域敏感性。
```

## 9. 当前立刻执行的实验

本轮先执行两个低风险实验：

### 实验 A：Qwen Q1 CLT feature readout smoke

输出：

```text
doc/experiments/stage2/019_stage2f_qwen_clt_feature_readout_smoke.md
doc/experiments/stage2/cross_model/stage2f_qwen_clt_feature_readout_smoke.json
```

预期判定：

```text
pass_adapter_readout 或 partial。
```

### 实验 B：LLaVA L0 asset format smoke

输出：

```text
doc/experiments/stage2/020_stage2f_llava_asset_format_smoke.md
doc/experiments/stage2/cross_model/stage2f_llava_asset_format_smoke.json
```

预期判定：

```text
pass_format_readable 或 partial_format_readable。
```

## 10. 写作口径

如果 Qwen Q1 通过，可以写：

```text
Qwen2.5-VL now passes asset loading, native hook-forward, and CLT feature-readout feasibility checks. This indicates that public Qwen VLM transcoder assets can enter our analysis chain at the readout level.
```

不能写：

```text
Qwen 复现了 Gemma3 的 evidence-region-sensitive support routes。
```

如果 LLaVA L0 通过，可以写：

```text
LLaVA-1.5 provides a plausible Llama-family VLM route, but requires a custom format adapter before it can enter the same analysis chain.
```

不能写：

```text
Llama 已经跨模型复现视觉路径。
```

## 11. 成功 / 部分成功 / 失败判定

### Qwen

`pass`：

```text
hidden state shape 与 CLT input dim 对齐；
encode_layer 成功；
top feature activations 可读；
输出 artifact 完整。
```

`partial`：

```text
native hidden state 可读，但 CLT encode 因 dtype/device/shape/hook layer off-by-one 等问题失败；
错误类型明确，下一步可修。
```

`blocked`：

```text
base model 无法加载；
CLT 无法加载；
hidden_dim 不匹配；
GPU/磁盘不足导致无法继续。
```

### LLaVA

`pass`：

```text
repo file list 可读；
至少一个 transcoder_L*.pt 和 mapping_L*.pt 可读；
tensor key/shape/dtype 可记录；
能判断 adapter 入口。
```

`partial`：

```text
file list 可读，但权重格式需要额外依赖或下载受阻；
仍能明确下一步。
```

`blocked`：

```text
repo 不可读；
权重不可下载；
torch.load 无法解析且无法判断格式。
```

## 12. 对主论文 claim 的影响

短期：

```text
不会改变主 claim。
主 claim 仍然是 Gemma3 localized strong-image-dependence VQA 上的 evidence-region-sensitive support routes。
```

中期：

```text
如果 Qwen Q1-Q4 通过，可以在论文中加入“cross-model feasibility”小节。
```

长期：

```text
如果 Qwen Q5 或 LLaVA L4 复现，则可以把主 claim 从 Gemma3-focused pilot 升级为 early cross-model evidence。
```

最保守最终表述：

```text
Cross-model work is currently an engineering feasibility branch. It supports the plausibility of extending the method beyond Gemma3, but does not yet constitute independent cross-model replication of the main mechanism claim.
```
