# Stage4-063 Qwen Route-First Strict Confirmation

## 目的

Strict 只复验 primary 中满足 `2+3+4` 的 exact candidates，不重新挑选节点。

## 冻结规则

冻结字段：

```text
candidate_id
sample_id
prompt_name
layer
source_pos
source_feature_id
source_zeroing_mode
```

如果 strict pack 中没有同 sample/prompt，则标记 missing，不做 fallback。

## 判定

Strict n `<20` 时只写 case-clustered / primary-only diagnostic，不升级 paperpack-level claim。

## 执行状态

`2026-05-29 17:17 CST` 已冻结 primary route-first candidates：

```text
primary route_first_234 candidates = 2029
strict frozen exact candidates = 1930
strict frozen unique samples = 48
missing in strict = 99
```

strict frozen layer distribution：

```text
L10 = 1062
L11 = 138
L12 = 129
L13 = 117
L14 = 132
L15 = 115
L16 = 114
L17 = 123
```

`2026-05-29 17:20 CST` 已启动 strict full detached：

```text
remote pid = 855441
remote log = /root/autodl-tmp/tca-reasoning/stage4_qwen_route_first/logs/route_first_strict_full_20260529_172045.log
remote script = /root/autodl-tmp/tca-reasoning/stage4_qwen_route_first/run_stage4_qwen_route_first_strict_full_routefirst_v1.sh
```

首个 L10 checkpoint 已确认：

```text
requested candidates = 1062
usable checkpoint candidates = 30
raw rows = 1200
skipped = 0
status = running_checkpoint
```

启动条件：

```text
primary full 完成或得到足够稳定的 primary route_first_234 candidate table
stage4_qwen_route_first_routefirst_v1_route_candidates.csv 中 route_first_234 == 1
```

冻结命令：

```powershell
python scripts/local/build_stage4_qwen_route_first_manifest.py --strict-from-primary routefirst_v1
```

严格确认命令：

```powershell
python scripts/local/run_stage4_qwen_route_first_remote.py --pack strict --mode full --layers selected --selection frozen --tag routefirst_v1 --detach
```

注意：strict 不允许从 strict pack 重新搜索更强节点；如果 exact candidate 在 strict 中不存在，只记为 `strict_unavailable` / `missing_in_strict`，不做 same-feature 或 same-layer fallback。
