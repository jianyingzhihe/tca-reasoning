# 031 Stage3 Gemma Paperpack Overlap Report

## 目的

检查 `paperpack72_primary` / `paperpack72_strict_sensitivity` 与历史 Gemma3-PLT source-tracing / controlled compare 结果的重叠程度，判断历史 Gemma 结果能否直接作为 paperpack confirmatory baseline。

这个实验只做 overlap audit，不产生新的机制结果。

## 输入

```text
paperpack primary:
  doc/experiments/stage3/paperpack72/paperpack72_primary_prompt_runs.csv

paperpack strict:
  doc/experiments/stage3/paperpack72/paperpack72_strict_sensitivity_prompt_runs.csv

historical Gemma rows:
  remote_sync/**/sample_compare_controlled.csv
  remote_sync/**/nodes_detailed_controlled.csv
  remote_sync/**/intervention_smoke*.csv
```

## 输出

```text
doc/experiments/stage3/cross_model/stage3_gemma_paperpack_overlap_report.csv
doc/experiments/stage3/cross_model/stage3_gemma_paperpack_overlap_report.json
doc/experiments/stage3/cross_model/stage3_gemma_paperpack_overlap_raw_compare_rows.csv
doc/experiments/stage3/cross_model/stage3_gemma_paperpack_overlap_raw_node_rows.csv
doc/experiments/stage3/cross_model/stage3_gemma_paperpack_overlap_raw_intervention_rows.csv
```

## 方法

扫描历史 `remote_sync` 中可用的 Gemma controlled compare、node detail 和 intervention smoke 文件，然后按 `sample_id` 与 paperpack primary/strict 样本做匹配。

判据：

```text
historical_available_primary_prompt_runs >= 20:
  可作为 confirmatory overlap baseline 的候选

historical_available_primary_prompt_runs < 20:
  只能作为 limited calibration，不能替代 Gemma paperpack rerun
```

## 结果

```text
status: limited_calibration_only
primary_samples: 72
strict_samples: 72
union_samples: 77
historical_available_samples: 4
historical_available_primary_prompt_runs: 8
historical_compare_nodes_overlap: 4
no_historical_gemma_overlap: 73
```

历史 overlap 低于 20 prompt-runs 门槛，因此不能把历史 Gemma rows 当作 paperpack confirmatory 结果。

## 预期与实际偏差

预期中可能存在足够历史重叠，从而减少 Gemma full rerun 成本。实际只有 4 个样本、8 个 primary prompt-runs 有 compare + node overlap，因此只能保留为 sanity-check / limited calibration。

## 结论

Gemma3-PLT paperpack 侧必须新增 source-tracing rerun。历史 overlap 不能支撑 PLT-only verdict 的 Gemma confirmatory side。

