# 实验 021：Stage 2F Qwen Token / Position Mapping Smoke

日期：2026-05-20

## 1. 实验目的

本实验对应 Stage 2F-3 的 Qwen Q2。上一轮 `019_stage2f_qwen_clt_feature_readout_smoke.md` 已经证明 Qwen hidden state 可以进入 Qwen CLT encoder，但当时 top activation 大量集中在 `position 2`，还不知道这个位置到底是视觉 token、问题 token、答案前缀，还是模板 token。

本实验要回答：

```text
Qwen2.5-VL 的 input sequence 中，image span、question span、assistant prefix、last prompt token 和 position 2 分别在哪里？
这些位置上的 CLT top features 是否可读？
position 2 是否属于可解释的视觉/答案区域，还是模板相关位置？
```

这一步仍然只是 readout / alignment smoke，不做 attribution、不做 intervention、不做跨模型机制复现。

## 2. 输入

Qwen base model：

```text
/root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5
```

Qwen CLT：

```text
KokosDev/qwen2p5vl-7b-clt
```

两组测试：

```text
Smoke:
image = COCO_val2014_000000192716.jpg
question = What does stop mean?

Main mapping:
sample_id = okvqa_val_2847255
image = COCO_val2014_000000284725.jpg
question = What country might this be based on the writing on the bus? Use visual evidence, then reply with only one short sentence in exactly this format: The answer is <short answer>.
```

测试层：

```text
0, 13, 26
```

## 3. 输出

```text
doc/experiments/stage2/021_stage2f_qwen_token_position_mapping_smoke.md
doc/experiments/stage2/cross_model/stage2f_qwen_position_mapping_smoke_192716.json
doc/experiments/stage2/cross_model/stage2f_qwen_position_mapping_smoke_192716_tokens.csv
doc/experiments/stage2/cross_model/stage2f_qwen_position_mapping_smoke_192716_buckets.csv
doc/experiments/stage2/cross_model/stage2f_qwen_position_mapping_2847255.json
doc/experiments/stage2/cross_model/stage2f_qwen_position_mapping_2847255_tokens.csv
doc/experiments/stage2/cross_model/stage2f_qwen_position_mapping_2847255_buckets.csv
```

新增脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_qwen_token_position_mapping_smoke.py
```

## 4. 方法

脚本执行以下步骤：

1. 用 Qwen processor 构造图文 prompt。
2. 读取 `input_ids` 并转换成 token text。
3. 用 tokenizer 子序列匹配定位 question span。
4. 用 `<|vision_start|> / <|image_pad|> / <|vision_end|>` 定位 image span。
5. 用 question span 之后的 assistant token 定位 assistant prefix，避免把 system prompt 中的 “helpful assistant” 误标成 assistant prefix。
6. 标记 `last_prompt_token` 和 `position_2_diagnostic`。
7. 对每个 bucket 的 positions，读取 layer `0 / 13 / 26` 的 top CLT features。

## 5. 结果

### 5.1 Smoke sample：`COCO_val2014_000000192716`

判定：

```text
decision.status = pass_position_mapping
```

关键位置：

```text
sequence_length = 440
image_span = [14, 430]
question_span = [430, 435]
assistant_start = 438
last_prompt_token = 439
position_2 token_id = 198
position_2 token_text = Ċ
```

bucket count：

```text
image_marker_or_span = 416
question = 5
assistant_prefix = 2
last_prompt_token = 1
position_2_diagnostic = 1
```

解释：

```text
position 2 是 system prompt 早期的 newline token，不是 image token，也不是 answer-adjacent token。
```

### 5.2 Main mapping：`okvqa_val_2847255`

判定：

```text
decision.status = pass_position_mapping
```

关键位置：

```text
sequence_length = 402
image_span = [14, 361]
question_span = [361, 397]
assistant_start = 400
last_prompt_token = 401
position_2 token_id = 198
position_2 token_text = Ċ
```

bucket count：

```text
image_marker_or_span = 347
question = 36
assistant_prefix = 2
last_prompt_token = 1
position_2_diagnostic = 1
```

layer 26 示例 top features：

```text
position_2_diagnostic:
feature 5834 @ position 2 activation 880.0
feature 2929 @ position 2 activation 872.0

image_marker_or_span:
feature 5105 @ position 292 activation 90.0
feature 4872 @ position 355 activation 83.0

question:
feature 5252 @ position 369 activation 74.0
feature 2650 @ position 370 activation 71.0

assistant_prefix / last_prompt_token:
feature 5067 @ position 400 activation 63.5
feature 5834 @ position 401 activation 60.25
```

## 6. 预期与实际偏差

预期：

```text
能够定位 question 和 last prompt token；image span 如果不能精确定位也可以 partial。
```

实际：

```text
question、image span、assistant prefix、last prompt token 全部可定位。
```

重要修正：

```text
第一次脚本版本把 system prompt 里的 "helpful assistant" 误标成 assistant prefix。
修复后 assistant_start 改为 question span 之后的 assistant token。
修复后重新运行 Q2/Q3。
```

## 7. 结论

本实验支持：

```text
Qwen token/position mapping 可行。
position 2 的强激活是模板/system newline 相关，不能解释为视觉证据或答案附近位置。
Qwen 后续 readout 应重点看 image span、question span、assistant prefix、last prompt token，而不是直接使用全局 top activation。
```

本实验不支持：

```text
Qwen 已经发现 answer route。
Qwen 已经复现 Gemma3 evidence-region-sensitive support routes。
```

最准确判定：

```text
Stage 2F Qwen Q2 = pass_position_mapping。
```

## 8. 后续动作

进入 Qwen Q3：

```text
clean vs answer_mask / union_mask feature readout。
只写成 readout-level evidence-region sensitivity。
不写 attribution、intervention 或 causal route。
```
