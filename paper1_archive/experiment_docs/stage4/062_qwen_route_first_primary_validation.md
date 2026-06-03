# Stage4-062 Qwen Route-First Primary Validation

## 目的

在 primary pack 上完整遍历 L10-L17 route-first candidates，发现满足 `2+3+4` 的答案传输线候选。

## 指标

```text
gate2: clean_source_minus_controls > 0
gate3: restore_source_minus_controls > 0
gate4: real_minus_shifted > 0 and real_minus_shuffled > 0
gate1 retrospective: evidence_specificity > 0
gate5 retrospective: clean_correct_minus_wrong > 0 or restore_correct_minus_wrong > 0
```

## 输出

```text
stage4_qwen_route_first_routefirst_v1_summary.csv
stage4_qwen_route_first_routefirst_v1_specificity.csv
stage4_qwen_route_first_routefirst_v1_route_candidates.csv
stage4_qwen_route_first_routefirst_v1_concentration.csv
stage4_qwen_route_first_routefirst_v1_decision.json
```

## 当前执行记录

本地脚本已通过 `py_compile`：

```text
build_stage4_qwen_route_first_manifest.py
run_stage4_qwen_route_first_remote.py
analyze_stage4_qwen_route_first_validation.py
run_stage4_qwen_causal_cutter_validation.py
```

`primary smoke` 已完成：

```text
tag = routefirst_v1
layers = 10,13,15,16,17
smoke candidates = 12
fetched raw/run artifacts = 15/15
analysis status = smoke_ok
```

Smoke 检查确认：

```text
raw CSV 非空
run JSON status ok
新增 clean_feature_activation / mask_feature_activation / activation_drop 字段已落盘
```

`primary full` 已于 `2026-05-28 20:26 CST` 重新启动为 detached/nohup：

```text
remote pid = 815105
remote log = /root/autodl-tmp/tca-reasoning/stage4_qwen_route_first/logs/route_first_primary_full_20260528_202643.log
layers = 10,11,12,13,14,15,16,17
candidate rows = 23040
```

为避免夜间中断导致一层结果完全丢失，远端 validation 脚本已增加 checkpoint：

```text
--checkpoint-every 30
每 30 个 usable candidates 写一次 raw CSV 和 run JSON
checkpoint status = running_checkpoint
```

该改动只影响落盘安全性，不改变候选、干预、controls 或 gate 定义。
