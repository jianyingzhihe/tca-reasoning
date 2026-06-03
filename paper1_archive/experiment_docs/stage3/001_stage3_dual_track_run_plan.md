# 001 Stage3 双轨 Run Plan

> 当前有效论文级执行顺序已更新为 `016_stage3_plt_first_paper_grade_run_plan.md`：先完成 PLT 主线和 PLT-only verdict，再进入 CLT 辅助线。本文件保留为 Stage3 初始双轨历史计划。

## 目的

重新规划跨模型实验，修正 Stage2 中 Qwen/LLaVA 主要使用 CLT、而 Gemma 主线使用 PLT 所带来的资产类型不对齐问题。

## 输入

```text
PLT 主线:
  Gemma3: tianhux2/gemma3-4b-it-plt
  Qwen2.5-VL: KokosDev/qwen2p5vl-7b-plt
  Qwen35: KokosDev/qwen35-4b-plt

CLT 辅助线:
  Qwen2.5-VL: KokosDev/qwen2p5vl-7b-clt
  LLaVA-1.5: KokosDev/llava15-7b-clt

样本:
  Stage2N all52 localized masks 中选 aligned24
  每个样本跑 B_direct 与 D_visual_only
```

## 输出

```text
README.md
stage3_experiments.md
001-008 中文实验文档
cross_model/stage3_aligned24_manifest.csv
cross_model/stage3_aligned48_prompt_runs.csv
cross_model/stage3_asset_table.csv
cross_model/stage3_* raw/summary/decision artifacts
```

## 方法

Stage3 分六步推进：

```text
1. Asset preflight:
   检查 config、层数、hook、tensor shape、base model/VLM 风险。

2. Qwen CLT vs PLT audit:
   同一批样本、prompt、mask 下比较 Qwen2.5-VL-CLT 与 Qwen2.5-VL-PLT。

3. PLT-aligned mainline:
   用 Gemma3-PLT 作为 baseline，Qwen2.5-VL-PLT 作为主跨模型对象，Qwen35-PLT 先做可行性判定。

4. CLT auxiliary completion:
   补齐 Qwen-CLT 与 LLaVA-CLT 的 controls，作为 robustness/heterogeneity 证据。

5. Behavior bridge:
   只对内部证据通过的 rows 做 first-token/rank/short decoded generation。

6. Final verdict:
   按模型轴和资产轴输出 conservative verdict。
```

## 预期

最理想结果是：Gemma3 与 Qwen2.5-VL 都在 PLT 主线中出现 evidence-region-sensitive feature/source-control support；Qwen-CLT 作为稳健性支持；LLaVA-CLT 作为异质性诊断。

## 结果

初始 manifest 和 smoke 已完成：

```text
aligned samples = 24
prompt-runs = 48
available eligible localized samples = 33
symbol_text_reading = 11
visual_readout = 11
scene_inference = 2
```

实际类型分布和预期有偏差：原计划 scene_inference = 4，但可用 localized mask 只有 2 个进入 aligned24。这里不硬凑 diffuse/非局部样本，避免把 claim 扩大到不适合的样本类型。

## 结论

Stage3 双轨设计成立。当前最关键的可检验问题变成：

```text
Qwen2.5-VL 的 evidence-sensitive feature/source-control evidence 是否在 PLT 与 CLT 中同方向？
如果同方向但大小不同，说明结论不是 CLT-only，但存在 representation dependence。
```
