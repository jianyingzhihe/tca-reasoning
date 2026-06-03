# Stage4-056 Qwen Evidence-Specific Route Verdict

## 当前状态

Primary smoke 已完成并显示 targeted near-pass；但 main full + strict 修正分析后未通过 source/control 与 correct/wrong causal gates。

当前只能写：

```text
Qwen has a large pool of strong evidence-specific PLT candidates. In the 160-candidate main targeted validation, current feature zeroing/restore does not yet establish stable causal answer-support or evidence-linked route gates. Full-pool primary validation is running to rule out missed stable candidates.
```

## 允许结论

- `qwen_evidence_specific_causal_nodes_supported`
- `qwen_evidence_linked_route_supported`
- `qwen_evidence_sensitive_but_not_causal`
- `qwen_answer_support_not_evidence_linked`
- `qwen_plt_localization_unresolved`

## 禁止过度结论

- 不写 Qwen fully replicates Gemma-style source tracing。
- 不把 targeted validation 失败写成 Qwen 没有跨模态机制。
- 不把少数强单例直接写成 paperpack-level route，除非 primary/strict gates 同时成立。
