# 026 Paperpack72 Pass1 Audit And Replacements

## 目的

记录 `paperpack72` 第一轮单人 LabelMe 标注完成后的质量审计结果，并把用户指出的非强图像依赖样本从正式实验候选中降级为 `exclude_or_replace`。

这一步的目标不是扩大样本，而是保证后续 PLT-first 论文级实验只使用 localized、strong-image-dependence、证据区域可定位的样本。

## 输入

Pass1 标注包：

```text
E:\Bridging\annotation\stage3_paperpack72_labelme\single_pass1_primary72_clean
```

问题与答案表：

```text
E:\Bridging\annotation\stage3_paperpack72_labelme\single_pass1_primary72_clean\QUESTION_SHEET.md
```

用户明确指出的非强图像依赖样本：

| sample_id | question | answer | decision |
|---|---|---|---|
| `okvqa_val_03189` | What is the purpose of the fin on the pink surfboard? | `balance` | exclude_or_replace |
| `okvqa_val_3508555` | Why do people blow out candles on their birthday? | `make wish` | exclude_or_replace |
| `okvqa_val_4536495` | What kind of environments do zebras live in? | `dry` | exclude_or_replace |

## 输出

审计输出：

```text
E:\Bridging\doc\experiments\stage3\paperpack72\paperpack72_pass1_audit.csv
E:\Bridging\doc\experiments\stage3\paperpack72\paperpack72_pass1_review_or_exclude.csv
E:\Bridging\doc\experiments\stage3\paperpack72\paperpack72_pass1_audit_summary.json
```

替换候选包：

```text
E:\Bridging\annotation\stage3_paperpack72_labelme\replacement_pass1_candidates12
E:\Bridging\annotation\stage3_paperpack72_labelme\replacement_pass1_candidates12\QUESTION_SHEET.md
E:\Bridging\doc\experiments\stage3\paperpack72\paperpack72_replacement_candidates12.csv
```

新增脚本：

```text
E:\Bridging\scripts\local\prepare_stage3_paperpack72_replacement_pack.py
```

## 方法

1. 重新统计 pass1 标注目录中的 `.jpg` 和 `.json`，确认主包不是缺标状态。
2. 用 `audit_stage3_paperpack72_pass1.py` 检查每个样本是否有 `answer` 与 `relate` 标注、是否存在非法 label、是否被用户或启发式规则标记为疑似非图像依赖。
3. 将用户明确指出的 3 个样本固定标为 `exclude_or_replace`。
4. 将规则命中的 5 个样本标为 `manual_review_image_dependence`，后续只在人工复审确认图像确实必要时保留。
5. 从 reserve pool 中手动挑选 12 个更强图像依赖的替换候选，生成独立 LabelMe 包，避免覆盖已完成的 72 个 pass1 JSON。

## 结果

Pass1 完成情况：

| item | count |
|---|---:|
| jpg files | 72 |
| json files | 72 |
| `answer` shapes | 83 |
| `relate` shapes | 72 |
| missing json | 0 |

审计建议：

| recommendation | count |
|---|---:|
| keep_pending_export | 64 |
| manual_review_image_dependence | 5 |
| exclude_or_replace | 3 |

需要复审或替换的 8 个样本：

| sample_id | recommendation | reason |
|---|---|---|
| `okvqa_val_03189` | exclude_or_replace | 用户指出问题文本已经给出 surfboard fin，答案更像通用功能常识 |
| `okvqa_val_2078985` | manual_review_image_dependence | calories 问题可能依赖外部常识，需要确认图像是否真正必要 |
| `okvqa_val_4059455` | manual_review_image_dependence | purpose 问题可能是通用功能常识，需要确认是否依赖图中 metal scaffolding |
| `okvqa_val_5252115` | manual_review_image_dependence | calories 问题可能依赖外部常识，需要确认图像是否真正必要 |
| `okvqa_val_02733` | manual_review_image_dependence | purpose 问题需要确认车辆类别是否必须由图像识别 |
| `okvqa_val_3508555` | exclude_or_replace | 用户指出 birthday candle custom 是常识题，图像不提供关键答案证据 |
| `okvqa_val_4536495` | exclude_or_replace | 用户指出 zebra habitat 是常识题，图像不提供关键答案证据 |
| `okvqa_val_802195` | manual_review_image_dependence | purpose 问题需要确认 red rag 的视觉证据是否足够具体 |

替换候选包包含 12 个候选，其中前 4 个来自 `localized` reserve，后 8 个来自 `multi_region` reserve。`multi_region` 候选不自动进入主 claim，只有在证据区域仍可局部标注、且 image-dependence 通过复审时才可补入 final manifest。

## 预期与实际偏差

预期是旧的 `localized` visual tier 能直接给出 72 个强图像依赖样本。实际发现 `localized` 只说明历史标注认为证据区域相对局部，并不保证问题答案一定强依赖图像。

这不会推翻 paperpack72，但说明必须增加一个 image-dependence 质量闸门。最终论文里不能写成“双人标注”，应写成：

```text
single annotator pass1 + user quality audit + delayed self-consistency pass2 + explicit exclusion/replacement manifest
```

## 结论

`paperpack72` pass1 已完成，但还不能直接进入 PLT 正式实验。当前最稳的下一步是：

1. 标注 `replacement_pass1_candidates12` 中的候选。
2. 人工复审 5 个 `manual_review_image_dependence` 样本。
3. 组合出 final include set，目标仍是 72；如果不够，论文级最低门槛为 confirmatory 48。
4. 导出 `answer_mask / relate_mask / union_mask`，并生成 shifted/shuffled/random16 controls。
5. 之后再启动 Gemma3-PLT 与 Qwen2.5-VL-PLT 的 paperpack 主实验。
