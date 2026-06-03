# 实验 XXX：实验名

日期：YYYY-MM-DD  
状态：未开始 / 运行中 / 已完成 / 工程失败 / 机制失败 / partial  
负责人：Codex + user

---

## 1. 目的

本实验要回答什么问题？

它对应 Stage 2 哪一部分？

```text
Stage 2A / 2B / 2C / 2D / 2F
```

它想支持、削弱或排除哪个 claim？

---

## 2. 输入

数据输入：

```text
sample manifest
image paths
questions
gold answers
prompts
node manifest
mask assets
```

模型输入：

```text
model_name
transcoder_set
dtype
device
```

脚本输入：

```text
script path
command
important args
```

---

## 3. 输出

预期输出文件：

```text
csv / json / md / figures / logs
```

实际输出文件：

```text
to be filled
```

---

## 4. 方法

实验步骤：

1. Step 1
2. Step 2
3. Step 3

关键指标：

```text
route weakening
source-minus-nearest
answer/union minus random16
target rank damage
margin drop
decoded answer change
format_prefix_ok
empty_or_error
```

统计方法：

```text
bootstrap CI
per-sample / per-source unit
descriptive case table
```

---

## 5. 预期

成功预期：

```text
what should happen if claim is supported
```

partial 预期：

```text
what would count as partial support
```

失败预期：

```text
what would count as not supported
```

---

## 6. 实际结果

填入定量结果：

```text
mean
CI
counts
case ids
failure counts
```

填入定性观察：

```text
notable cases
failure modes
unexpected behavior
```

---

## 7. 预期与实际偏差

哪些地方符合预期？

哪些地方偏离预期？

偏离是工程原因、数据原因、统计原因，还是机制原因？

---

## 8. 结论

本实验结论：

```text
success / partial / not supported / engineering blocked
```

对主 claim 的影响：

```text
strengthens / weakens / does not affect / only supports secondary claim
```

---

## 9. 后续动作

下一步要做什么？

是否需要补跑、修脚本、重新标注、扩大样本或降级 claim？

