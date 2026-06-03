# 实验 000：Cross-model Asset Survey

日期：2026-05-19  
状态：已完成初查  
对应计划：Stage 2F-0

---

## 1. 目的

本实验不是正式机制实验，而是 Stage 2F 的前置资产调查。

要回答：

```text
1. 有没有别人已经训练好的 VLM circuit / transcoder / CLT 资产？
2. 有没有小模型或中等模型适合做跨模型验证？
3. 如果没有，是否需要手动训练？
4. 如果有，哪个候选最适合先做 smoke？
```

这个实验的主要价值是避免过早走向“自己训练 transcoder”这条重工程路线。

---

## 2. 输入

本地文件：

```text
E:/Bridging/vlm-circuit-tracing/README.md
E:/Bridging/vlm-circuit-tracing/circuit_tracer_vlm/README.md
E:/Bridging/vlm-circuit-tracing/circuit_tracer_vlm/circuit_tracer/replacement_model.py
```

远端来源：

```text
https://huggingface.co/tianhux2/gemma3-4b-it-plt
https://huggingface.co/KokosDev/qwen2p5vl-7b-clt
https://huggingface.co/KokosDev/qwen2p5vl-7b-plt
https://huggingface.co/KokosDev/llava15-7b-clt
https://github.com/safety-research/circuit-tracer
```

---

## 3. 输出

本次输出：

```text
E:/Bridging/doc/experiments/runplan_stage2.md
E:/Bridging/doc/experiments/stage2_expetiments.md
E:/Bridging/doc/experiments/stage2/000_cross_model_asset_survey.md
E:/Bridging/doc/experiments/stage2/cross_model/cross_model_candidate_table.csv
```

后续应补输出：

```text
E:/Bridging/doc/experiments/stage2/cross_model/cross_model_loader_smoke.md
```

---

## 4. 方法

方法：

1. 读取本地 `vlm-circuit-tracing` 文档，确认当前主工作流的模型边界；
2. 搜索公开 Hugging Face 资产，重点查 `VLM + transcoder + CLT + circuit-tracer`；
3. 拉取候选 repo 的 `config.yaml` 或 file list；
4. 检查候选是否包含 base model、每层权重、hook point、维度、训练信息；
5. 对照本地 `ReplacementModel`，判断是否能直接加载或需要 adapter。

---

## 5. 结果

### 5.1 当前主模型资产

```text
repo = tianhux2/gemma3-4b-it-plt
base model = google/gemma-3-4b-it
status = 当前主实验已使用
```

这是目前最稳定的资产，也是 core24 和现有 region-mask 主结果的基础。

### 5.2 Qwen2.5-VL-7B CLT

```text
repo = KokosDev/qwen2p5vl-7b-clt
base model = Qwen/Qwen2.5-VL-7B-Instruct
type = CLT
format = layer_*.safetensors + config.yaml
n_layers = 27
hidden_dim = 3584
feature_dim = 8192
priority = highest
```

判断：

> 这是目前最值得先试的跨模型候选。

原因：

1. 它明确是 VLM；
2. 它有 `config.yaml`；
3. 它有 per-layer safetensors；
4. 它写明 base model 是 `Qwen/Qwen2.5-VL-7B-Instruct`；
5. 它比 LLaVA 候选更接近当前 loader 需要的 transcoder set 结构。

风险：

1. 当前本地 `ReplacementModel` 仍主要绑定 `Gemma3ForConditionalGeneration`；
2. Qwen2.5-VL 的 processor、image token、hook point 可能不同；
3. `blocks.{layer}.hook_resid_pre` / `blocks.{layer}.hook_resid_post` 是否与本地 HookedVLTransformer 路径兼容，需要 smoke；
4. 即使权重可读，也不保证 attribution graph / intervention path 可直接运行。

### 5.3 Qwen2.5-VL-7B PLT

```text
repo = KokosDev/qwen2p5vl-7b-plt
type = PLT
format = layer_*.safetensors + config.yaml
priority = second
```

判断：

> 可作为备用候选，但需要先核验 model card 和 config 对 base model 的描述是否完全一致。

风险：

1. model card / quickstart 命名存在需要核验的地方；
2. hook points 与当前 Gemma3 PLT 不同；
3. 不确定能否直接进入现有 region-mask intervention pipeline。

### 5.4 LLaVA-1.5-7B CLT

```text
repo = KokosDev/llava15-7b-clt
base model = llava-hf/llava-1.5-7b-hf
type = CLT
format = transcoder_L*.pt / mapping_L*.pt
priority = lower
```

判断：

> 有研究价值，但不适合作为第一个 smoke，因为文件格式和当前 loader 差异更大。

风险：

1. 不是当前 `config.yaml + layer_*.safetensors` 风格；
2. 可能需要单独 adapter；
3. LLaVA 的 vision encoder / projector / language model hook 路径与 Gemma3 差异明显；
4. 如果直接攻它，容易变成一条新工程线。

### 5.5 Language-only circuit assets

已有不少语言模型资产：

```text
Gemma-2 2B
Llama-3.2 1B
Qwen-3 0.6B / 1.7B / 4B / 8B / 14B
```

判断：

> 这些不能作为 VLM 视觉证据路径的跨模型复现，只能做方法 sanity check。

原因：

1. 没有图像输入；
2. 没有 image-region mask；
3. 无法验证 evidence-region sensitivity；
4. 只能验证 attribution / intervention 方法在另一个 LM 上能跑。

---

## 6. 预期

原始预期：

```text
可能没有可直接用于 VLM 的公开 circuit/transcoder 资产。
如果没有，就需要考虑手动训练。
```

成功预期：

```text
找到至少一个现成 VLM candidate。
candidate 有明确 base model、权重文件、hook point 和维度信息。
```

失败预期：

```text
只有 language-only assets。
没有 VLM assets。
或者 VLM assets 没有权重 / 没有 config / 无法判断 base model。
```

---

## 7. 实际结果

实际结果好于预期：

```text
发现 Qwen2.5-VL-7B 和 LLaVA-1.5-7B 的 VLM candidate assets。
```

但实际也暴露了新风险：

```text
最大问题不是“没有资产”，而是“资产是否能进入当前 Gemma3-oriented pipeline”。
```

本地代码限制：

```text
ReplacementModel.from_pretrained_and_transcoders 当前直接调用 Gemma3ForConditionalGeneration。
这意味着 Qwen2.5-VL 和 LLaVA 不能假设一键替换。
```

---

## 8. 预期与实际偏差

符合预期的部分：

```text
跨模型会有明显工程成本。
feature space / source-control / annotation 不能直接复用。
```

偏离预期的部分：

```text
原本以为可能没有公开 VLM assets。
实际发现至少有 Qwen2.5-VL 和 LLaVA 候选。
```

新的主要风险：

```text
loader / hook / processor / transcoder format compatibility。
```

这属于工程风险，不是机制风险。

---

## 9. 结论

本实验结论：

```text
Stage 2F 可以进入计划。
优先候选是 KokosDev/qwen2p5vl-7b-clt。
不建议现在手动训练 transcoder。
跨模型不应该抢 Stage 2A targeted replication 的主优先级。
```

对主 claim 的影响：

```text
当前没有新增机制证据。
但明确了跨模型验证的可行路径。
如果后续 Qwen smoke 成功，能增强“不是 Gemma3 特例”的外部效度。
```

---

## 10. 后续动作

下一步：

1. 维护 `cross_model_candidate_table.csv`，后续发现新资产就补行；
2. 先做 `KokosDev/qwen2p5vl-7b-clt` 的 loader / hook smoke；
3. smoke 只验证加载、image prompt、feature activation shape、single feature intervention；
4. smoke 成功后再做 one-case attribution / intervention；
5. one-case 成功后才进入 3-5 case mini region replication；
6. 如果 Qwen smoke 失败，再考虑 LLaVA adapter 或手动训练。

---

## 11. 暂不手动训练的理由

手动训练 transcoder / CLT 理论上可行，但当前不是最优路径。

原因：

1. 已经找到现成 VLM candidate；
2. 手动训练会引入训练质量、数据分布、dead features、reconstruction loss 等额外变量；
3. 即使训练完成，仍需要解决 VLM wrapper / hook / attribution graph 接入；
4. 当前主线最缺的是 Stage 2A independent replication，不是另起一个 training project；
5. 手动训练更适合作为 Stage 3 或跨模型资产不可用时的后备路线。

如果未来必须训练，最低流程是：

```text
choose hook-friendly VLM
collect image-text activation dataset
cache per-layer activations
train PLT or CLT
validate reconstruction / sparsity / dead features
integrate with ReplacementModel or equivalent wrapper
run attribution smoke
run intervention smoke
run region-mask mini replication
```
