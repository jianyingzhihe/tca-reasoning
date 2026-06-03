# Stage3：PLT-first / CLT-second 跨模型实验

## 当前目标

Stage3 现在采用论文级 `PLT-first / CLT-second` 顺序：

```text
1. 构造独立 paperpack72 标注包
2. 先完成 Gemma3 / Qwen2.5-VL / Qwen35 的 PLT 主线
3. 写 PLT-only verdict
4. 再做 Qwen-CLT / LLaVA-CLT 辅助补强
5. 最后写 paper-grade final verdict
```

主 claim 仍然限定为：

```text
localized、strong image-dependence、证据区域可定位的 VQA 样本中，
存在 evidence-region-sensitive answer-support mechanisms。
```

## 证据轨道

PLT 主线：

```text
tianhux2/gemma3-4b-it-plt
KokosDev/qwen2p5vl-7b-plt
KokosDev/qwen35-4b-plt
```

CLT 辅助线：

```text
KokosDev/qwen2p5vl-7b-clt
KokosDev/llava15-7b-clt
```

Qwen2.5-VL 最关键，因为它同时有 PLT 和 CLT，可以直接审计 transcoder 类型对结果的影响。LLaVA 没有公开 PLT，所以只作为 CLT auxiliary / heterogeneity diagnostic。

## 当前状态

```text
paperpack81_annotated_pool: done
paperpack72_primary: done
paperpack72_strict_sensitivity: done
Qwen2.5-VL-PLT primary72 full: done
Qwen2.5-VL-PLT strict72 sensitivity full: done
Qwen2.5-VL-PLT shifted/shuffled source-control controls: done
Gemma3-PLT paperpack: pending
Qwen35-PLT feasibility: pending recheck
CLT paperpack: blocked until PLT-only verdict
```

## 关键文档

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
026_paperpack72_pass1_audit_and_replacements.md
027_paperpack72_final_assets_and_qwen_plt_smoke.md
028_qwen2p5vl_plt_primary72_full_results.md
029_qwen2p5vl_plt_strict72_sensitivity_results.md
030_qwen2p5vl_plt_shifted_shuffled_controls.md
stage3_experiments.md
```

## 当前边界

可以写：

```text
Qwen2.5-VL-PLT 在 independently annotated paperpack72 primary 与 strict sensitivity set 上有 approximate feature/source-control support。
```

暂时不能写：

```text
Qwen2.5-VL 完整复现 Gemma-style source tracing。
PLT-only paper-grade verdict 已完成。
decoded generation-level causal bridge 已成立。
CLT 结果可替代 PLT 主线。
```
