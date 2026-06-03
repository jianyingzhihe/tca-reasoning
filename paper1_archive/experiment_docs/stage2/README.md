# Stage 2 实验档案夹

本目录用于保存 Stage 2 的细分实验记录。总日志在：

```text
doc/experiments/stage2_expetiments.md
```

使用规则：

1. 一个完整实验或工程 part 对应一个 md；
2. 文件名使用三位编号开头，例如 `001_stage2a_candidate_selection.md`；
3. 如果有图片、表格、日志或中间 CSV，放入对应子目录；
4. 每个 md 必须写清楚目的、输入、输出、方法、结果、预期与实际偏差、结论；
5. 工程失败必须单独记录，不等同于机制失败；
6. 所有结论都要写明它支持、削弱或不影响哪个 claim。

推荐目录结构：

```text
stage2/
  README.md
  EXPERIMENT_TEMPLATE.md
  000_cross_model_asset_survey.md
  001_stage2a_candidate_selection.md
  002_stage2a_annotation_manifest.md
  003_stage2a_region_replication_readout.md
  cross_model/
  figures/
  logs/
```

当前优先顺序：

```text
1. Stage 2A targeted replication pack
2. Stage 2B node-to-generation bridge
3. Stage 2C semantic feature / region cases
4. Stage 2F cross-model feasibility smoke
5. Stage 2D suppressor deep case
```

注意：Stage 2F 已进入计划，但不抢 Stage 2A 的主优先级。

