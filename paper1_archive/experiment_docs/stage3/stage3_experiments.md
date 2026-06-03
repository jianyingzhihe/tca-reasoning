# Stage3 实验总记录

## 总目标

Stage3 采用论文级 `PLT-first / CLT-second` 顺序组织跨模型证据。核心问题不是“所有模型是否完全同构”，而是：

```text
在 localized、strong image-dependence、证据区域可定位的 VQA 样本中，
evidence-region-sensitive answer-support mechanism
是否能在独立标注样本包上跨模型、跨 transcoder 类型稳定出现。
```

失败时必须区分模型差异、transcoder 类型差异和 adapter 工程限制。

## 当前有效执行顺序

```text
1. paperpack72 independent heldout pack
2. PLT-first: Gemma3-PLT / Qwen2.5-VL-PLT / Qwen35-PLT
3. PLT-only verdict
4. CLT-second: Qwen2.5-VL-CLT / LLaVA-CLT auxiliary completion
5. final paper-grade verdict
```

CLT 线在 PLT-only verdict 完成前保持 blocked，不提前启动。

## 进度表

| 编号 | 文档 | 状态 | 核心目的 |
|---|---|---|---|
| 016 | `016_stage3_plt_first_paper_grade_run_plan.md` | current_plan | 论文级 PLT-first / CLT-second run plan |
| 017 | `017_paperpack72_annotation_protocol.md` | done | 单人标注 + quality audit + sensitivity analysis 协议 |
| 018 | `018_paperpack72_manifest_report.md` | seed_ready | paperpack 候选池与 manifest 报告 |
| 019 | `019_gemma3_plt_paperpack_results.md` | primary_passed_strict_graph_compare_passed | Gemma3-PLT paperpack source-tracing 复跑 |
| 020 | `020_qwen2p5vl_plt_paperpack_results.md` | primary_and_strict_done | Qwen2.5-VL-PLT paperpack 主结果 |
| 021 | `021_qwen35_plt_feasibility_verdict.md` | pending_recheck | Qwen35-PLT VLM 可行性与 loader 判定 |
| 022 | `022_plt_only_verdict.md` | plt_only_verdict_ready | PLT-only verdict，Gemma/Qwen PLT 结果已合并 |
| 023 | `023_qwen_clt_paperpack_results.md` | blocked_until_plt | Qwen-CLT paperpack 辅助线 |
| 024 | `024_llava_clt_paperpack_results.md` | blocked_until_plt | LLaVA-CLT diagnostic 辅助线 |
| 025 | `025_stage3_paper_grade_final_verdict.md` | not_ready | Stage3 最终结论 |
| 026 | `026_paperpack72_pass1_audit_and_replacements.md` | done | 标注审计、非图像依赖剔除、replacement |
| 027 | `027_paperpack72_final_assets_and_qwen_plt_smoke.md` | done | paperpack final assets 与 Qwen smoke |
| 028 | `028_qwen2p5vl_plt_primary72_full_results.md` | done | Qwen2.5-VL-PLT primary72 full 结果 |
| 029 | `029_qwen2p5vl_plt_strict72_sensitivity_results.md` | done | Qwen2.5-VL-PLT strict72 sensitivity 结果 |
| 030 | `030_qwen2p5vl_plt_shifted_shuffled_controls.md` | done | Qwen2.5-VL-PLT shifted/shuffled spatial controls |
| 031 | `031_stage3_gemma_paperpack_overlap_report.md` | done_limited_overlap | Gemma paperpack 与历史 rows 的 overlap audit |
| 032 | `032_stage3_gemma_source_tracing_smoke.md` | smoke_passed | Gemma paperpack source-tracing smoke |
| 033 | `033_stage3_gemma_primary72_source_tracing_full.md` | primary_full_passed | Gemma3-PLT primary72 full source tracing |
| 034 | `034_stage3_gemma_strict72_source_tracing_sensitivity.md` | graph_compare_passed_intervention_blocked | Gemma3-PLT strict72 sensitivity full source tracing |

## 已完成 Artifact

Paperpack:

```text
doc/experiments/stage3/paperpack72/paperpack81_annotated_pool.csv
doc/experiments/stage3/paperpack72/paperpack72_primary_manifest.csv
doc/experiments/stage3/paperpack72/paperpack72_primary_prompt_runs.csv
doc/experiments/stage3/paperpack72/paperpack72_strict_sensitivity_manifest.csv
doc/experiments/stage3/paperpack72/paperpack72_strict_sensitivity_prompt_runs.csv
doc/experiments/stage3/paperpack72/paperpack72_final_exclusion_manifest.csv
doc/experiments/stage3/paperpack72/paperpack81_mask_export_summary.csv
doc/experiments/stage3/paperpack72/paperpack81_control_mask_summary.csv
doc/experiments/stage3/paperpack72/paperpack81_mask_geometry_warnings.csv
```

Qwen2.5-VL-PLT primary72 / strict72 / controls:

```text
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_plt_feature_union_primary_full.csv/json
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_plt_source_control_primary_full.csv/json
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_plt_primary_full_*summary.csv
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_plt_primary_full_case_table.csv
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_plt_primary_full_decision.json
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_plt_feature_union_strict_full.csv/json
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_plt_source_control_strict_full.csv/json
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_plt_strict_full_*summary.csv
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_plt_strict_full_case_table.csv
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_plt_strict_full_decision.json
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_plt_source_control_primary_controls_full.csv/json
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_plt_source_control_strict_controls_full.csv/json
doc/experiments/stage3/cross_model/stage3_qwen2p5vl_plt_controls_full_specificity_summary.csv
```

Gemma3-PLT:

```text
doc/experiments/stage3/cross_model/stage3_gemma_eval_primary_B_direct.csv
doc/experiments/stage3/cross_model/stage3_gemma_eval_primary_D_visual_only.csv
doc/experiments/stage3/cross_model/stage3_gemma_eval_strict_B_direct.csv
doc/experiments/stage3/cross_model/stage3_gemma_eval_strict_D_visual_only.csv
doc/experiments/stage3/cross_model/stage3_gemma_paperpack_overlap_report.csv/json
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_smoke_decision.json
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_smoke_*controlled.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_smoke_intervention.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_primary_full_decision.json
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_primary_full_analysis.json
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_primary_full_sample_compare_controlled.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_primary_full_nodes_detailed_controlled.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_primary_full_edges_detailed_controlled.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_primary_full_intervention.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_strict_full_decision.json
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_strict_full_analysis.json
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_strict_full_sample_compare_controlled.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_strict_full_nodes_detailed_controlled.csv
doc/experiments/stage3/cross_model/stage3_gemma_source_tracing_strict_full_edges_detailed_controlled.csv
```

## 当前最重要观察

Qwen2.5-VL-PLT 在 `paperpack72_primary` 和 `paperpack72_strict_sensitivity` 上完成 full run，feature/source-control 信号方向稳定。保守口径是：

```text
Qwen2.5-VL-PLT provides approximate feature/source-control support on paperpack72.
```

不能写：

```text
Qwen fully replicates Gemma-style source tracing.
Qwen has decoded generation-level causal bridge.
```

Gemma3-PLT paperpack overlap audit 显示历史重录不足：

```text
historical_available_samples: 4
historical_available_primary_prompt_runs: 8
```

因此新增 Gemma source-tracing rerun。当前 smoke、primary72 full 与 strict72 graph/compare sensitivity 均已完成：

```text
smoke valid rows: 2
primary valid samples: 71 / 72
primary graph success vs valid: 1.0
primary sample compare rows: 71
primary nodes detailed rows: 4126
primary intervention rows: 127
strict valid samples: 71 / 72
strict graph success vs valid: 1.0
strict sample compare rows: 71
strict nodes detailed rows: 4095
strict intervention rows: 0 (model-load SIGKILL / engineering blocked)
```

## 下一步

```text
1. PLT-only verdict 已可作为当前 PLT-first 收口口径。
2. 后续如继续加固 Gemma strict intervention，需要单独修复当前远端 model-load SIGKILL。
3. PLT-only verdict 完成后，再进入 Qwen-CLT / LLaVA-CLT paperpack 辅助线。
```
<!-- 2026-05-23 update: CLT-second completed. Qwen2.5-VL-CLT primary/strict paperpack runs support auxiliary robustness; Qwen PLT-vs-CLT paired comparison is partial robustness/representation-dependent; LLaVA-CLT primary/strict runs complete but feature/source route specificity is not supported. Gemma strict lightweight B4 intervention repair passed; full strict intervention remains incomplete. -->
