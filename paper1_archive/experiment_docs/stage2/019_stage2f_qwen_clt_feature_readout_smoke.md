# 实验 019：Stage 2F Qwen2.5-VL CLT Feature Readout Smoke

日期：2026-05-20

## 1. 实验目的

本实验接在 `017_stage2f_qwen_download_and_lazy_load.md` 与 `018_stage2f_qwen_hook_forward_smoke.md` 之后，目标是回答一个更靠近方法链的问题：

```text
Qwen2.5-VL native forward 产生的 hidden state，是否能送入公开 Qwen CLT 的 encoder，并得到可读的 feature activations？
```

这一步仍然不是跨模型机制复现。它只验证 `native Qwen hidden state -> Qwen CLT feature readout` 这条桥是否可行。

如果这一步失败，后续 Qwen attribution / intervention / region-mask replication 都没有必要继续。如果这一步成功，说明 Qwen 已经从“资产可加载”进入“feature readout 可行”的阶段。

## 2. 术语解释

`hidden state`：模型每一层输出或输入位置上的隐藏向量。对 Qwen2.5-VL，本次读到的 hidden state 形状为 `[batch, seq, hidden_dim]`。

`CLT`：Cross-Layer Transcoder，跨层转码器。本次使用的 Qwen CLT 实际以 `TranscoderSet` 形式加载，每层有一个 `layer_*.safetensors` 文件。

`feature readout`：把 hidden state 输入 transcoder encoder，读出稀疏 feature activation。它只读特征，不做因果干预。

`adapter readout`：这里指不用完整 `ReplacementModel`，直接用 native Qwen hidden state 和 CLT encoder 做最小连接。

`resid_pre`：Transformer 某层输入的 residual stream。Qwen CLT metadata 写的是 `blocks.{layer}.hook_resid_pre`，所以本轮默认用 `hidden_states[layer]` 近似对应。

## 3. 输入

Base model：

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

本地服务器 snapshot：

```text
/root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5
```

Transcoder asset：

```text
KokosDev/qwen2p5vl-7b-clt
```

测试图像：

```text
/root/autodl-tmp/tca-reasoning/data/okvqa/images/val2014/COCO_val2014_000000192716.jpg
```

测试问题：

```text
What does stop mean?
```

测试层：

```text
0, 13, 26
```

## 4. 输出

本地 artifact：

```text
E:\Bridging\doc\experiments\stage2\019_stage2f_qwen_clt_feature_readout_smoke.md
E:\Bridging\doc\experiments\stage2\cross_model\stage2f_qwen_clt_feature_readout_smoke.json
```

远端 artifact：

```text
/root/autodl-tmp/tca-reasoning/stage2f_cross_model/stage2f_qwen_clt_feature_readout_smoke.json
```

新增脚本：

```text
E:\Bridging\vlm-circuit-tracing\circuit_tracer_vlm\scripts\research\run_qwen_clt_feature_readout_smoke.py
```

## 5. 方法

远端运行逻辑：

```bash
cd /root/autodl-tmp/tca-reasoning/circuit_tracer_vlm
source scripts/server/dev.sh
source /etc/network_turbo
export PYTHONPATH=/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm:${PYTHONPATH:-}

.venv/bin/python -u scripts/research/run_qwen_clt_feature_readout_smoke.py \
  --model-name /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --transcoder-set KokosDev/qwen2p5vl-7b-clt \
  --image-path /root/autodl-tmp/tca-reasoning/data/okvqa/images/val2014/COCO_val2014_000000192716.jpg \
  --question "What does stop mean?" \
  --layers 0,13,26 \
  --hidden-state-offset 0 \
  --top-k 12 \
  --min-gpu-free-gb 18 \
  --out-json /root/autodl-tmp/tca-reasoning/stage2f_cross_model/stage2f_qwen_clt_feature_readout_smoke.json
```

脚本步骤：

1. 加载 Qwen2.5-VL processor。
2. 加载 Qwen2.5-VL base model 到 GPU。
3. lazy 加载 `KokosDev/qwen2p5vl-7b-clt`。
4. 用 processor 构造图文输入。
5. 跑 native Qwen forward，打开 `output_hidden_states=True`。
6. 对 layer `0 / 13 / 26` 取 `hidden_states[layer]`。
7. 调用 `transcoders.encode_layer(hidden, layer)`。
8. 记录 feature shape、active feature count、max activation、top-k feature id 与位置。

## 6. 结果

最终判定：

```text
decision.status = pass_adapter_readout
reason = native_qwen_hidden_states_encoded_by_qwen_clt
```

Processor：

```text
processor_class = Qwen2_5_VLProcessor
status = ok
has_tokenizer = true
```

Base model：

```text
class = Qwen2_5_VLForConditionalGeneration
device = cuda:0
status = ok
```

Transcoder：

```text
class = TranscoderSet
n_layers = 27
d_transcoder = 8192
config_hidden_dim = 3584
config_feature_dim = 8192
feature_input_hook = blocks.{layer}.hook_resid_pre
feature_output_hook = blocks.{layer}.hook_resid_post
```

Forward：

```text
input_ids shape = [1, 440]
pixel_values shape = [1656, 1176]
logits shape = [1, 440, 152064]
hidden_states_count = 29
```

Layer-level feature readout：

```text
layer 0:
  hidden shape = [1, 440, 3584]
  feature shape = [1, 440, 8192]
  active_positive_count = 795736
  max_activation = 11.0625
  mean_positive_activation = 0.497150
  top feature example = position 15, feature 3366, activation 11.0625

layer 13:
  hidden shape = [1, 440, 3584]
  feature shape = [1, 440, 8192]
  active_positive_count = 2419292
  max_activation = 1712.0
  mean_positive_activation = 6.606767
  top feature example = position 2, feature 7439, activation 1712.0

layer 26:
  hidden shape = [1, 440, 3584]
  feature shape = [1, 440, 8192]
  active_positive_count = 1881607
  max_activation = 880.0
  mean_positive_activation = 7.739564
  top feature example = position 2, feature 5834, activation 880.0
```

GPU：

```text
before free_gb = 46.986
after free_gb = 31.094
```

## 7. 预期与实际偏差

预期：

```text
如果 Qwen hidden_dim 与 CLT hidden_dim 对齐，则 encode_layer 应该能成功。
features shape 应为 [batch, seq, 8192]。
```

实际：

```text
完全符合 shape 预期。
三个测试层都成功 encode。
```

值得注意的偏差：

```text
layer 13 和 layer 26 的 top activation 集中在 position 2。
这说明当前 readout 还没有完成 token / position 语义对齐。
position 2 很可能是特殊 token、vision prefix 或模板相关位置，而不是答案附近位置。
```

因此，下一步不能直接把 top feature 当成 answer route。必须先做 Qwen Q2：token / image / question / answer-prefix position mapping。

## 8. 结论

本实验支持：

```text
Qwen2.5-VL native hidden states 可以被 KokosDev/qwen2p5vl-7b-clt encoder 读取。
Qwen public CLT 资产已经可以进入我们的方法链的 feature-readout 层。
Qwen cross-model 路线从 asset/native-forward feasibility 推进到 readout feasibility。
```

本实验不支持：

```text
Qwen 已经接入当前 Gemma3 ReplacementModel。
Qwen 已经可以跑 attribution graph。
Qwen 已经可以做 feature intervention。
Qwen 已经复现 Gemma3 evidence-region-sensitive support routes。
```

最准确判定：

```text
Stage 2F Qwen Q1 = pass_adapter_readout。
```

## 9. 下一步

优先下一步：

```text
Qwen Q2：token / position mapping smoke。
```

具体要做：

1. 标出 Qwen prompt 中 image tokens、question tokens、assistant generation prefix、last prompt token 的位置。
2. 分别读取这些位置上的 top-k features。
3. 排除特殊 token / prefix position 主导的假信号。
4. 再进入 Qwen Q3：clean vs answer-mask feature readout。

保守写法：

```text
Qwen2.5-VL now passes asset loading, native hook-forward, and CLT feature-readout feasibility checks. However, the result remains a readout-level smoke test until token-position alignment and intervention adapters are implemented.
```
