# 027 Paperpack72 Final Assets And Qwen PLT Smoke

## 目的

把已完成的 `paperpack72` 单人标注转成可执行实验资产，并按 PLT-first 顺序先做 Qwen2.5-VL-PLT 远端 smoke。

本实验只验证新 paperpack 是否能进入 Qwen PLT feature/source-control 链路；它不是 full paperpack 结果，也不是 PLT-only verdict。

## 输入

原始 pass1 标注：

```text
E:\Bridging\annotation\stage3_paperpack72_labelme\single_pass1_primary72_clean
```

替换候选标注：

```text
E:\Bridging\annotation\stage3_paperpack72_labelme\replacement_pass1_candidates12
```

审计文件：

```text
E:\Bridging\doc\experiments\stage3\paperpack72\paperpack72_pass1_audit.csv
```

## 输出

Final manifest 与 mask/control 资产：

```text
E:\Bridging\doc\experiments\stage3\paperpack72\paperpack81_annotated_pool.csv
E:\Bridging\doc\experiments\stage3\paperpack72\paperpack72_primary_manifest.csv
E:\Bridging\doc\experiments\stage3\paperpack72\paperpack72_primary_prompt_runs.csv
E:\Bridging\doc\experiments\stage3\paperpack72\paperpack72_strict_sensitivity_manifest.csv
E:\Bridging\doc\experiments\stage3\paperpack72\paperpack72_strict_sensitivity_prompt_runs.csv
E:\Bridging\doc\experiments\stage3\paperpack72\paperpack72_final_exclusion_manifest.csv
E:\Bridging\doc\experiments\stage3\paperpack72\paperpack81_mask_export_summary.csv
E:\Bridging\doc\experiments\stage3\paperpack72\paperpack81_control_mask_summary.csv
E:\Bridging\doc\experiments\stage3\paperpack72\paperpack81_mask_geometry_warnings.csv
E:\Bridging\annotation\stage3_paperpack72_labelme\paperpack81_final_assets
```

Qwen2.5-VL-PLT primary smoke 结果：

```text
E:\Bridging\doc\experiments\stage3\cross_model\stage3_paperpack_asset_preflight_primary_smoke.json
E:\Bridging\doc\experiments\stage3\cross_model\stage3_qwen2p5vl_plt_feature_union_primary_smoke.csv
E:\Bridging\doc\experiments\stage3\cross_model\stage3_qwen2p5vl_plt_source_control_primary_smoke.csv
```

新增脚本：

```text
E:\Bridging\scripts\local\build_stage3_paperpack_final_assets.py
E:\Bridging\scripts\local\run_stage3_paperpack_plt_remote.py
```

## 方法

1. 从原始 72 中剔除 3 个用户明确指出的非强图像依赖样本。
2. 将 5 个 `manual_review_image_dependence` 样本保留为 `moderate`，但不放进最强切片。
3. 将 replacement 12 个样本加入 annotated pool。
4. 构造 `paperpack81_annotated_pool`、`paperpack72_primary`、`paperpack72_strict_sensitivity`。
5. 从 LabelMe JSON 导出 `answer.png`、`relate.png`、`union.png`、`shifted.png`、`shuffled.png` 和 `random16` 控制 masks。
6. 远端只上传 smoke 所需的前 3 个样本资产，跑 Qwen2.5-VL-PLT feature bridge 和 approximate source-control probe。

## 结果

Manifest 结果：

| item | count |
|---|---:|
| annotated pool | 81 |
| primary manifest | 72 |
| strict sensitivity manifest | 72 |
| final exclusions | 3 |
| moderate samples in pool | 5 |
| replacements in pool | 12 |
| replacements in primary | 3 |
| replacements in strict sensitivity | 8 |

Mask 导出结果：

| item | count |
|---|---:|
| mask rows | 81 |
| mask status ok | 81 |
| answer empty | 0 |
| relate empty | 0 |
| union empty | 0 |
| random16 files | 1296 |

Geometry warning：

| mask_geometry_tier | count |
|---|---:|
| compact_union | 1 |
| medium_union | 39 |
| broad_union | 30 |
| very_broad_union | 11 |

Qwen2.5-VL-PLT smoke：

| artifact | result |
|---|---:|
| uploaded smoke samples | 3 |
| prompt-runs requested | 6 |
| feature bridge rows | 180 |
| source-control rows | 42 |
| feature bridge usable runs | 6 |
| source-control usable pairs | 7 |
| remote exit status | 0 |

Smoke aggregate, only for debugging:

| metric | evidence/source mean | control mean | gap |
|---|---:|---:|---:|
| feature restore logit | +0.0382 | -0.0347 | +0.0729 |
| feature corrupt logit damage | +0.0208 | -0.0833 | +0.1041 |
| source-control restore logit | +0.0000 | +0.0179 | -0.0179 |
| source-control zeroing logit damage | +0.1250 | -0.0179 | +0.1429 |

## 预期与实际偏差

预期：所有样本可顺利导出 masks，并且 random16 能提供低重叠随机区域控制。

实际：mask 均可导出且非空，但许多 `union_mask` 面积偏大，导致低 IoU random16 控制较难生成。这个问题不阻塞 primary smoke，但必须作为 geometry sensitivity 记录。后续 full paperpack 结果需要同时报告：

```text
primary72 full result
strict72 sensitivity result
mask geometry slice result
moderate_image_dependence slice result
```

## 结论

paperpack 已经从标注包推进为可执行实验资产。Qwen2.5-VL-PLT primary smoke 跑通，并出现初步正向 feature/zeroing 信号，但样本太少，不能写成正式结论。

下一步按 PLT-first 顺序推进：

```text
1. 跑 Qwen2.5-VL-PLT primary72 full。
2. 跑 strict72 sensitivity 或复用 full 结果抽 slice。
3. 单独补 Gemma3-PLT paperpack source-tracing runner。
4. 完成 PLT-only verdict 后，再进入 Qwen-CLT/LLaVA-CLT。
```
