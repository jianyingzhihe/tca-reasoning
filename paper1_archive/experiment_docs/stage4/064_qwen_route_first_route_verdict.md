# Stage4-064 Qwen Route-First Route Verdict

## 当前状态

`2026-05-29 16:41 CST` primary full 最终结果：

```text
primary full candidates = 23040
route_first_234 = 2029
route_first_gold = 1491
route_first_evidence_gold = 1334
unique samples = 50
```

`2026-05-29 17:17 CST` strict frozen manifest 已生成：

```text
strict frozen exact candidates = 1930
strict frozen unique samples = 48
missing in strict = 99
```

`2026-05-29 19:38 CST` strict full 已完成并拉回分析：

```text
strict full candidates = 1930
strict route_first_234 = 1930
strict route_first_gold = 1447
strict route_first_evidence_gold = 1295
unique strict samples = 48
```

最终 decision：

```text
qwen_route_first_full_supported
```

## 允许结论

```text
qwen_route_first_candidates_exist
qwen_route_first_evidence_linked_supported
qwen_route_first_gold_answer_supported
qwen_route_first_full_supported
qwen_answer_route_not_evidence_linked
qwen_evidence_route_not_gold_specific
qwen_route_first_not_supported
blocked
```

## 禁止过度结论

不把本轮 route-first candidates 写成 Gemma-style automatic source tracing。  
不把 strict missing 或 small-n 写成 Qwen 没有机制。  
不把少数 case-clustered 结果升级为 paperpack-level full claim。

## 当前边界

本轮现在可以写 paperpack-level route-first node support，因为 strict frozen confirmation 在足够候选数和样本数上同时通过 `1+2+3+4+5`。

但它仍然是 node-level route-first support，不等于 automatic graph-level Gemma-style source tracing。

## 当前判断

primary + strict 已经支持：

```text
Qwen route-first causal candidates exist.
Qwen route-first gold-specific candidates exist.
Qwen route-first evidence-gold candidates exist.
Qwen route-first full node-level support is established under frozen strict confirmation.
```

下一步 Stage4-066 已经把这些 node-level candidates 组成 sample/prompt-level feature route bundles 做 grouped route validation。该 route-level 实验结果单独见 `069_qwen_feature_route_replication_verdict.md`。
