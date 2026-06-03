# 004 Stage3 PLT-Aligned Mainline

## 目的

用 PLT 资产对齐 Gemma3 与 Qwen2.5-VL，避免用 Gemma-PLT 主线去直接比较 Qwen/LLaVA-CLT 时产生资产类型混淆。

## 输入

```text
Gemma3-PLT:
  base = google/gemma-3-4b-it
  transcoder = tianhux2/gemma3-4b-it-plt

Qwen2.5-VL-PLT:
  base = Qwen/Qwen2.5-VL-7B-Instruct
  transcoder = KokosDev/qwen2p5vl-7b-plt

Qwen35-PLT:
  base = Qwen/Qwen3.5-4B
  transcoder = KokosDev/qwen35-4b-plt
```

## 输出

```text
Gemma3-PLT Stage3 baseline/calibration artifacts
Qwen2.5-VL-PLT feature bridge artifacts
Qwen2.5-VL-PLT source-control artifacts
Qwen35-PLT feasibility artifacts
```

## 方法

固定 Stage3 aligned24 manifest，对每个样本跑 `B_direct` 与 `D_visual_only`。

主条件：

```text
clean
answer_mask
union_mask
shifted_mask
mask_shuffled
wrong_target
```

核心干预：

```text
source_zeroing
matched_control_zeroing
source_restore
matched_control_restore
```

## 当前结果

Qwen2.5-VL-PLT 已完成首轮 smoke：

```text
usable prompt-runs = 6/6
raw rows = 180
answer_adjacent_text restore evidence-control = +0.294
answer_adjacent_text corrupt evidence-control = +0.188
combined restore evidence-control = +0.085
combined corrupt evidence-control = +0.086
```

Full aligned24 已完成 Qwen2.5-VL-PLT feature bridge 与 approximate source-control probe。Gemma3-PLT 已完成 Stage3 overlap calibration，但不是完整 aligned24 重跑；Qwen35-PLT 因 custom `.pt`、缺 L1、当前 transformers 不支持 `model_type=qwen3_5`，暂不进入完整主线。

Qwen2.5-VL-PLT full feature bridge：

```text
usable prompt-runs = 48/48
raw feature rows = 1440
answer_adjacent_text restore evidence-control = +0.341, CI [+0.244, +0.449]
answer_adjacent_text corrupt evidence-control = +0.207, CI [+0.116, +0.294]
combined restore evidence-control = +0.071, CI [+0.030, +0.113]
combined corrupt evidence-control = +0.028, CI [-0.012, +0.067]
```

Qwen2.5-VL-PLT approximate source-control：

```text
usable source-control pairs = 75
answer_mask restore source-control = +0.048, CI [+0.012, +0.093]
answer_mask zeroing source-control = +0.232, CI [+0.153, +0.345]
union_mask restore source-control = +0.056, CI [+0.035, +0.076]
union_mask zeroing source-control = +0.179, CI [+0.137, +0.225]
```

Gemma3-PLT overlap calibration：

```text
Stage3 aligned24 samples = 24
strict Gemma source+nearest overlap samples = 4
support source-nearest comparison rows = 7
answer_mask source-nearest = +0.973, CI [+0.170, +1.875]
union_mask source-nearest = +1.143, CI [+0.321, +2.071]
answer_mask source-random16 = +0.469, CI [+0.061, +0.890]
union_mask source-random16 = +0.522, CI [+0.122, +0.924]
```

## 预期与实际偏差

预期 Qwen2.5-VL-PLT 至少能跑通 feature bridge smoke，实际 full aligned24 也通过，并且 source-control zeroing/restore 主项均为正。效应比 Qwen2.5-VL-CLT 小，但方向一致，这是一个重要的 representation-dependence 线索。

预期 Gemma3-PLT 可能能直接在 Stage3 aligned24 上复用较多历史 source/control 行；实际只有 4 个严格可用 overlap 样本，因此只能写 calibration，不能写 full aligned24 Gemma baseline。

## 当前结论

PLT-aligned mainline 可以继续推进，优先级为：

```text
1. Gemma3-PLT Stage3 manifest calibration
2. Qwen2.5-VL-PLT behavior bridge
3. Qwen35-PLT loader/forward feasibility only
4. 若要写强 cross-model route claim，再实现真正 Qwen ReplacementModel/source tracing adapter
```

其中第 1 项已完成 overlap calibration；若要完成 full Gemma3-PLT Stage3 baseline，需要对 Stage3 aligned24 剩余 20 个缺 source/control 的样本重新做 Gemma source tracing 与 nearest-control 构造。
