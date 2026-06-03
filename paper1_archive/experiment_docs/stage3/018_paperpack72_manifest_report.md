# 018 Paperpack72 Manifest Report

## 目的

记录 paperpack72 候选池构建结果，并明确当前还不能启动正式实验的原因。

## 输入

候选来源：

```text
annotation/okvqa_type_label_round4_400/manifest.csv
annotation/okvqa_type_label_round3_320/manifest.csv
annotation/okvqa_type_label_round2_new32/manifest.csv
annotation/okvqa_type_label_round_80_combined/analysis_stage1_type_labels_80/type_labeled_manifest.csv
```

排除来源：

```text
Stage2 cross_model artifacts
Stage3 cross_model artifacts
已有 okvqa_evidence_labelme_* manifest
stage2a region replication manifest
```

## 输出

已生成：

```text
paperpack72/paperpack72_candidate_pool.csv
paperpack72/paperpack72_manifest_template.csv
paperpack72/paperpack72_exclusion_manifest.csv
paperpack72/paperpack72_quota_shortfall.csv
paperpack72/paperpack72_candidate_pool_report.json
paperpack72/annotator_a_assignment.csv
paperpack72/annotator_b_assignment.csv
```

脚本：

```text
scripts/local/build_stage3_paperpack72_candidate_pool.py
```

## 方法

脚本执行以下步骤：

```text
1. 从已有类型候选池读取 OK-VQA 样本。
2. 收集 Stage2/Stage3 已用 sample_id。
3. 排除已用样本和 diffuse/global 明确负例。
4. 用问题文本和已有 visual labels 生成 proposed_reasoning_operation。
5. 按目标配额选择 72 个 annotation seed。
6. 如果某类严格候选不足，从 reserve localized candidates backfill，并标记需要人工类型确认。
```

## 结果

候选池初版：

| 指标 | 数量 |
|---|---:|
| source candidates | 400 |
| used ids excluded | 97 |
| strict localized eligible candidates after exclusion | 78 |
| selected annotation seed | 72 |

当前最终 `primary72` 采用视觉档位优先，不再硬凑类型配额。72 个 annotation seed 的视觉档位为：

| visual_tier | 数量 |
|---|---:|
| localized | 72 |

但按自动启发式推断的实际候选类型并不均衡：

| inferred type | 数量 |
|---|---:|
| symbol_text_reading | 3 |
| visual_readout | 43 |
| compact_scene_inference | 25 |
| mixed_localized | 1 |

reserve pool：

| visual_tier | 数量 |
|---|---:|
| localized | 6 |
| multi_region | 43 |
| unlabeled | 146 |

短缺项：

| type | selected | 原计划目标 | 处理 |
|---|---:|---:|---|
| symbol_text_reading | 3 | 24 | 不硬凑；如需要文字类扩展，可从 multi_region/unlabeled reserve 另建 extension pack |
| mixed_localized | 1 | 12 | 不硬凑；作为 exploratory slice 报告 |

旧 A 标注复用情况：

| reuse_status | 数量 |
|---|---:|
| copied_to_single_pass1_localized | 4 |
| old_annotator_a_row_without_json | 10 |
| not_found_in_old_annotator_a | 58 |

## 预期与实际偏差

预期自动候选池能直接满足四类配额。实际发现严格 `localized` 候选足够，但 `symbol_text_reading` 与 `mixed_localized` 严格候选不足。论文中应优先报告 localized heldout 主结果，类型切片只按最终人工确认分布报告。`multi_region` 可作为 extension pack；`unlabeled` 可作为 prescreen reserve；`diffuse_global` 原则上只作为边界/负诊断，不进入主 claim。

因此 paperpack72 当前状态是：

```text
annotation seed ready
final manifest not ready
formal experiments blocked by double annotation + adjudication
```

## 结论

paperpack72 已经有可标注的 72-row seed，但还不是论文实验 manifest。下一步必须完成：

```text
single pass1 answer/relate mask 标注
delayed pass2 reliability 复标
pass1/pass2 IoU/Dice 统计
仲裁 answer/relate/union mask
shifted/shuffled/random16 controls
final_include 与 final_reasoning_operation 审核
```

完成后才能启动 Gemma3-PLT 与 Qwen2.5-VL-PLT 的 full paperpack rerun。
