# Stage 2H Run Plan：Qwen/LLaVA Cross-Model Causal Localization

## Summary

Stage 2G 的关键结果是：

```text
Qwen/LLaVA 的 evidence mask 会损伤 target answer logit/rank；
把 masked run 的整段 image-bucket hidden state patch 回 clean hidden state，可以恢复 target logit/rank；
但 CLT feature-level intervention/restoration 还没有稳定成立。
```

因此 Stage 2H 不再继续证明 readout 会变，而是把 `hidden-state causal bridge upper bound` 进一步拆开：

```text
恢复效果来自哪些 image positions？
这些 positions 是否强于 random / low-delta controls？
同一组 positions 是否能 restore masked run，又能 corrupt clean run？
是否需要 answer-adjacent text positions 共同参与？
```

当前不需要新增标注，先复用 3 个主 case：

```text
okvqa_val_2847255
okvqa_val_4157235
okvqa_val_3658865
```

## Claim 边界

即使 Stage 2H 成功，也只写：

```text
cross-model hidden-state-level causal localization / bridge
```

不能写：

```text
Qwen/LLaVA 已经复现 Gemma source-control causal route；
Qwen/LLaVA 的 CLT features 已经是 causal support features；
Qwen/LLaVA feature 是对象级语义节点。
```

Gemma 仍然是完整 source/control causal route chain 的主证据。

## 实验设计

### Stage 2H-1：image-bucket position localization

目标：

```text
把 whole image-bucket hidden patch 拆成更小的位置组，
判断哪些 visual positions 承担主要恢复效果。
```

位置组：

```text
whole_bucket
evidence_region_positions
top_hidden_delta_positions
low_delta_control_positions
random_control_1..4
```

模型固定：

```text
Qwen:
  layer = 26
  bucket = image_marker_or_span

LLaVA:
  layer = 15
  bucket = image_token_span
```

LLaVA 区域映射：

```text
576 image tokens -> 24×24 grid
用 union mask 或 answer mask 与 grid cell overlap 选 evidence-region positions。
```

Qwen 区域映射：

```text
优先检查 processor image_grid_thw；
如果无法可靠映射到 visual span，则只使用 top_hidden_delta_positions，
不写严格 evidence-region token localization。
```

### Stage 2H-2：bidirectional hidden patch

目标：

```text
检查同一组 source-like positions 是否既能 restore union_mask run，
也能 corrupt clean run。
```

方向：

```text
restore:
  union_hidden += clean_hidden - union_hidden

corrupt:
  clean_hidden += union_hidden - clean_hidden
```

核心判据：

```text
restore 方向提升 target logit/rank；
corrupt 方向损伤 target logit/rank；
两者都强于 random/low-delta controls。
```

### Stage 2H-3：answer-adjacent bridge

目标：

```text
比较视觉位置和 answer-adjacent text positions 的作用，
判断视觉证据信号是否需要在 last prompt / assistant prefix 附近汇聚。
```

位置组：

```text
image source positions only
answer-adjacent text positions only
image source + answer-adjacent positions
```

如果 `image + answer-adjacent` 明显强于单独 image 或 text，则写成：

```text
evidence-to-answer bridge hint
```

不是 source route 复现。

### Stage 2H-4：decoded answer smoke

只有 Stage 2H-1/2 出现通过 case 时才做。

若 decoded answer 不变：

```text
只写 first-token / rank bridge。
```

若 decoded answer 改变：

```text
可以写 generation-level bridge smoke。
```

## 成功标准

Stage 2H-1：

```text
source-like positions 的 mean logit restore > random controls；
positive logit restore ≥ 4/6；
rank restore positive ≥ 3/6 作为强支持。
```

Stage 2H-2：

```text
同一组 positions 同时满足 restore 和 corrupt 两个方向；
restore/corrupt 均强于 controls；
否则只写 partial localization。
```

Stage 2H-3：

```text
image + answer-adjacent 明显强于单独 image 或 text；
否则不写 bridge composition claim。
```

## 预期产物

脚本：

```text
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_cross_model_hidden_position_patch_smoke.py
vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/analyze_stage2h_hidden_position_patch.py
scripts/local/run_stage2h_hidden_position_patch_remote.py
```

输出：

```text
doc/experiments/stage2/cross_model/stage2h_qwen_hidden_position_patch.json
doc/experiments/stage2/cross_model/stage2h_qwen_hidden_position_patch.csv
doc/experiments/stage2/cross_model/stage2h_llava_hidden_position_patch.json
doc/experiments/stage2/cross_model/stage2h_llava_hidden_position_patch.csv
doc/experiments/stage2/cross_model/stage2h_hidden_position_patch_summary.csv
doc/experiments/stage2/cross_model/stage2h_hidden_position_patch_specificity.csv
doc/experiments/stage2/cross_model/stage2h_hidden_position_patch_case_table.csv
doc/experiments/stage2/cross_model/stage2h_hidden_position_patch_decision.json
doc/experiments/stage2/035_stage2h_hidden_position_patch_smoke.md
```

## 最终读法

如果 Stage 2H 成功：

```text
Qwen/LLaVA support cross-model hidden-state-level causal localization:
specific visual/source-like positions can restore masked target-answer signal and damage clean target-answer signal.
```

如果 Stage 2H 不成功：

```text
Qwen/LLaVA support readout sensitivity and whole-bucket hidden-state causal bridge,
but current position-level localization and feature/source-level replication remain unresolved.
```
