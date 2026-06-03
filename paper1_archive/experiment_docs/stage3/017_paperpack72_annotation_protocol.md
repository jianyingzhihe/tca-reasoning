# 017 Paperpack72 Annotation Protocol

## 目的

为论文级独立 heldout pack 建立可审计标注协议，降低审稿人对 evidence region 主观性的攻击空间。

本协议的核心原则是：

```text
单人完整 pass1 标注 -> 延迟 pass2 reliability 复标 -> 计算自一致性 -> 仲裁生成最终 mask -> 明确排除失败样本。
```

## 输入

标注任务包：

```text
annotation/stage3_paperpack72_labelme/single_pass1_localized/manifest.csv
annotation/stage3_paperpack72_labelme/single_pass2_localized_reliability/manifest.csv
```

每行包含：

```text
sample_id
image_filename / local_image_path / image_url
question_text
answer_text
proposed_reasoning_operation
quota_slot
```

`proposed_reasoning_operation` 只是自动候选分配，不是最终标签。标注者需要根据图像、问题和答案重新判断最终类型。

## 输出

每个样本需要输出：

```text
single_pass1_answer_mask
single_pass1_relate_mask
single_pass2_reliability_answer_mask
single_pass2_reliability_relate_mask
adjudicated_answer_mask
adjudicated_relate_mask
adjudicated_union_mask
shifted_mask
shuffled_mask
random16 masks
```

最终 manifest：

```text
paperpack72/paperpack72_manifest.csv
paperpack72/paperpack72_exclusion_manifest.csv
```

## 标注规则

### answer_mask

只标最直接支持答案的视觉区域。例如：

```text
文字/标牌问题：答案文字所在区域。
对象读出问题：被问对象或属性的最小可见区域。
紧凑推理问题：判断答案所必需的最小核心区域。
```

### relate_mask

标支持理解 answer_mask 的上下文区域。例如：

```text
完整标牌边界、相关物体、关系对象、必要场景线索。
```

### union_mask

仲裁阶段由 `answer_mask ∪ relate_mask` 得到，用于主 region-mask intervention。

## 排除规则

以下样本进入 exclusion manifest，不硬凑：

```text
证据弥散，无法局部标注；
答案需要纯外部知识，图像区域不能定位；
问题或答案歧义严重；
pass1/pass2 answer_mask IoU 极低且复核后无法达成一致；
图像缺失、损坏或与问题不匹配；
clean image 下目标答案不合理；
no-image / wrong-image 下视觉依赖不明显。
```

## 一致性指标

每个样本记录：

```text
answer_iou_pass1_pass2
relate_iou_pass1_pass2
answer_dice_pass1_pass2
relate_dice_pass1_pass2
answer_area_frac
relate_area_frac
union_area_frac
compactness_label
final_reasoning_operation
image_dependence_final
```

建议阈值：

```text
answer_mask pass1/pass2 IoU >= 0.20: 自动进入仲裁候选。
answer_mask pass1/pass2 IoU < 0.20: 必须人工复核；如果确实多解或弥散，则排除。
union_area_frac > 0.50: 标记为可能 diffuse，需要复核。
```

## 预期与实际偏差

预期 72 个标注种子中会有部分样本被排除。为了保持论文级严谨性，不要求强行保留 72 个；正式实验最低门槛为：

```text
confirmatory >= 48 个最终纳入样本；
目标仍是 72 个最终纳入样本。
```

如果最终某个类型切片不足，必须在论文中报告，而不是用非局部样本补齐。

## 结论

paperpack72 只有在完成 pass1 标注、pass2 reliability 复标、仲裁和 mask/control 生成后，才能作为正式 PLT/CLT 实验输入。论文中应如实写为 single-annotator reliability protocol，而不是双人独立标注。
