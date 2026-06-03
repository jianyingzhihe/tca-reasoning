# 002 Stage3 Asset Preflight

## 目的

检查 Stage3 五个公开资产是否能进入双轨实验，并把不可用原因分类为格式、层缺失、base model、VLM 能力、loader 或工程 adapter 风险。

## 输入

```text
PLT:
  tianhux2/gemma3-4b-it-plt
  KokosDev/qwen2p5vl-7b-plt
  KokosDev/qwen35-4b-plt

CLT:
  KokosDev/qwen2p5vl-7b-clt
  KokosDev/llava15-7b-clt
```

## 输出

```text
cross_model/stage3_asset_preflight_local.json/csv
cross_model/stage3_asset_preflight.json/csv
cross_model/stage3_asset_table.csv
```

## 方法

对每个 transcoder repo 检查：

```text
repo metadata 是否可读
config.yaml 是否存在
model_kind
feature_input_hook / feature_output_hook
layer_*.safetensors 数量
custom .pt 数量
mapping .pt 数量
缺失层
是否适合 VLM evidence-region 主线
```

说明：本轮 preflight 主要是 asset-level 检查。Qwen2.5-VL 与 LLaVA 的 base model forward 已在后续 smoke 中通过；Qwen35 仍需单独 base/VLM 可行性检查。

## 结果

| 资产 | 类型 | 状态 | 关键结果 |
|---|---|---|---|
| `tianhux2/gemma3-4b-it-plt` | PLT | pass | `config.yaml` 可读，`model_kind=transcoder_set`，34 个 layer safetensors，hook 为 `hook_mlp_in -> hook_mlp_out` |
| `KokosDev/qwen2p5vl-7b-plt` | PLT | pass | `config.yaml` 可读，27 个 layer safetensors，hook 为 `hook_resid_mid -> hook_mlp_out` |
| `KokosDev/qwen2p5vl-7b-clt` | CLT | pass | `config.yaml` 可读，27 个 layer safetensors，同时有 custom `.pt`；hook 为 `blocks.{layer}.hook_resid_pre -> blocks.{layer}.hook_resid_post` |
| `KokosDev/llava15-7b-clt` | CLT | partial | 无标准 `config.yaml`，31 个 custom `.pt` + 31 个 mapping `.pt`，只能走 custom loader |
| `KokosDev/qwen35-4b-plt` | PLT | partial/high-risk | 无标准 `config.yaml`，31 个 custom `.pt`，缺 L1；base repo 有 `Qwen3VLProcessor`，但当前 transformers 不认识 `model_type=qwen3_5` |

## 预期与实际偏差

预期 Qwen2.5-VL-PLT 可进入 PLT 主线，实际符合。预期 Qwen35-PLT 高风险，实际风险更明确：它不是标准 safetensors/config 格式，且缺 L1。补充 base feasibility 显示 `Qwen/Qwen3.5-4B` 有 `Qwen3VLProcessor` 与视频/图像预处理配置，因此不能说它不是 VLM；但当前 transformers 版本无法加载 `model_type=qwen3_5` config，需要升级 transformers 或使用对应 remote/custom code 后才能 forward。

## 结论

Stage3 可立即推进的主线是：

```text
Gemma3-PLT baseline
Qwen2.5-VL-PLT main cross-model target
Qwen2.5-VL-CLT auxiliary comparison
LLaVA-CLT auxiliary/diagnostic
```

Qwen35-PLT 暂不作为强主线证据，只做可行性判定；当前更准确状态是 `partial_vlm_candidate_needs_forward`。后续若要推进，需要先解决 transformers/Qwen3.5 loader，再处理 custom `.pt` PLT 与缺 L1 问题。
