# 016 Stage3 PLT-First Paper-Grade Run Plan

## 目的

本文件是 Stage3 当前有效 run plan，取代 `001_stage3_dual_track_run_plan.md` 的并行推进顺序。新的论文级执行顺序固定为：

```text
先 PLT 主线完整证成 -> 写 PLT-only verdict -> 再做 CLT 辅助补强。
```

这样做是为了降低审稿风险：Gemma 主线使用 PLT，如果 Qwen/LLaVA 主要依赖 CLT，会被质疑资产类型不一致。因此 Stage3 的主 claim 先只基于 PLT 对齐结果，CLT 只作为 robustness、heterogeneity 或 negative diagnostic。

## 输入

新增独立 heldout 标注包：

```text
paperpack72:
  72 个全新 localized / strong-image-dependence OK-VQA 候选样本
  单人 pass1 标注 answer_mask / relate_mask
  延迟 pass2 reliability 复标，用于自一致性审计
  仲裁生成 adjudicated answer / relate / union mask
  自动生成 shifted / shuffled / random16 controls
```

PLT 主线资产：

```text
Gemma3-PLT: tianhux2/gemma3-4b-it-plt
Qwen2.5-VL-PLT: KokosDev/qwen2p5vl-7b-plt
Qwen35-PLT: KokosDev/qwen35-4b-plt
```

CLT 辅助资产：

```text
Qwen2.5-VL-CLT: KokosDev/qwen2p5vl-7b-clt
LLaVA-CLT: KokosDev/llava15-7b-clt
```

## 输出

新增文档链：

```text
016_stage3_plt_first_paper_grade_run_plan.md
017_paperpack72_annotation_protocol.md
018_paperpack72_manifest_report.md
019_gemma3_plt_paperpack_results.md
020_qwen2p5vl_plt_paperpack_results.md
021_qwen35_plt_feasibility_verdict.md
022_plt_only_verdict.md
023_qwen_clt_paperpack_results.md
024_llava_clt_paperpack_results.md
025_stage3_paper_grade_final_verdict.md
```

新增 paperpack72 artifacts：

```text
paperpack72/paperpack72_candidate_pool.csv
paperpack72/paperpack72_manifest_template.csv
paperpack72/paperpack72_exclusion_manifest.csv
paperpack72/paperpack72_quota_shortfall.csv
paperpack72/paperpack72_candidate_pool_report.json
paperpack72/annotator_a_assignment.csv
paperpack72/annotator_b_assignment.csv
```

## 方法

### 1. Paperpack72 标注

从现有未用 OK-VQA 候选池中排除 Stage2/Stage3 已用 sample_id，且只保留历史人工/legacy visual label 明确为 `localized` 的样本，构建 72 个标注种子。

原计划的类型配额为：

| 类型 | 目标数 |
|---|---:|
| symbol_text_reading | 24 |
| visual_readout | 24 |
| compact_scene_inference | 12 |
| mixed_localized / OCR-object hybrid | 12 |

当前优先级已调整为：`localized` 档位优先于类型配额。自动候选类型只用于分配标注任务，不作为最终类型。最终以单人 pass1、延迟 pass2 reliability、自一致性审计、仲裁和 `final_reasoning_operation` 为准。

### 2. PLT 主线

PLT 阶段必须先完成并写 verdict，之后才允许进入 CLT。

Gemma3-PLT：

```text
source tracing
support/suppressor direction
node zeroing
nearest control
random16 control
answer/union/shifted/shuffled mask
wrong-image / wrong-target
first-token/rank
decoded smoke
```

Qwen2.5-VL-PLT：

```text
feature bridge
approximate source-control probe
matched controls
real > shifted / shuffled
correct > wrong
first-token/rank
answer sequence score
decoded smoke
```

Qwen35-PLT：

```text
base/VLM loader
processor
hook/module availability
PLT layer availability
feature encode/decode
```

若 Qwen35 阻塞，只写 `blocked_for_vlm_mainline`，不作为负结果。

### 3. CLT 补强

CLT 只在 `022_plt_only_verdict.md` 完成后启动。

Qwen2.5-VL-CLT 在同一 paperpack72 上补齐与 Qwen-PLT 可比的 controls 和 behavior metrics，用于回答 transcoder 类型影响。

LLaVA-CLT 只作为 auxiliary / heterogeneity diagnostic：如果 hidden bridge 成立但 feature route 弱，就写成异质性，而不是写 LLaVA 没有机制。

## 预注册主指标

Gemma3-PLT primary endpoints：

```text
support source answer/union > random16
source > nearest
node zeroing 损伤 target
region mask 改变 first-token/rank 或 decoded answer
```

Qwen2.5-VL-PLT primary endpoints：

```text
source > matched controls
real mask > shifted/shuffled mask
correct target > wrong target
first-token/rank bridge
```

Qwen sequence score 与 decoded answer 是 secondary endpoints。decoded 不成立时不能升级到 generation-level causal bridge。

## 预期与实际偏差

预期 paperpack72 会提供比 Stage3 aligned24 更强的独立复现基础。实际候选池显示：严格 `localized` 未用候选有 78 个，足够抽取 72 个；但其中自动启发式识别的 `symbol_text_reading` 和 `mixed_localized` 不足，因此类型切片必须以最终人工确认标签为准。

## 当前结论边界

Stage3 论文级主线尚未完成。当前能做的是：

```text
完成 paperpack72 候选池、标注协议和 PLT-first 执行计划。
```

在 paperpack72 完成 pass1 标注、pass2 reliability 复标、仲裁和 mask/control 导出前，不能启动正式 PLT full paperpack experiments。
