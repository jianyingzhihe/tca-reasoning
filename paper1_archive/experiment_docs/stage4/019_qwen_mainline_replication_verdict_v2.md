# Stage4-019 Qwen Mainline Replication Verdict V2

## 目的

汇总 Stage4-016/017/018，回答 Qwen2.5-VL-PLT 是否能推进到类似 Gemma 主线的结论。

## 当前可判定状态

Before overnight V2:

- Supported: `Qwen causal-screened PLT cutter nodes exist`
- Not yet supported: `Qwen causal-screened evidence-linked cutter support`
- Not supported under current hook-aligned adapter: `Qwen fully replicates Gemma-style automatic source tracing`

## 判据

- `qwen_gemma_style_supported`: automatic source tracing and all controls pass.
- `qwen_native_evidence_linked_supported`: Qwen-native cutter candidates pass evidence-link controls but automatic route remains incomplete.
- `qwen_cutter_only_supported`: clean zeroing source-control remains positive but evidence-link fails.
- `qwen_not_gemma_style_under_adapter`: expanded automatic source tracing fails source/evidence controls after running.
- `blocked`: engineering failure.

## 结果

Automatic source tracing V2:

- L26 primary top32: `qwen_source_tracing_not_supported`
- L26 strict top32: `qwen_source_tracing_not_supported`
- L22/L24 sensitivity: not supported
- visual-only / answer-adjacent-only sensitivity: not supported

Expanded cutter:

- main candidates increased from `12` to `31`
- main candidates include `16` numeric and `15` non-numeric
- `26/31` main candidates have strict confirmation
- clean source-control remains supported

Evidence-link V2:

- grouped top1/top4/top8/top16 restore completed
- final decision: `qwen_cutter_only_supported`
- activation-drop has partial union real-vs-shifted signal, but restore and shuffled controls do not pass

## 结论

Final Stage4-016/017/018 verdict:

`Qwen2.5-VL-PLT supports Qwen-native causal-screened PLT cutter nodes on an expanded candidate set, including a strong numeric slice and a smaller positive non-numeric slice. However, expanded automatic source tracing still does not replicate Gemma-style route selection, and grouped evidence-mask restore does not establish evidence-linked cutter support under shifted/shuffled controls.`

Decision label:

`qwen_cutter_only_supported`

This is a real step forward over Stage4-014 because the candidate set is larger and more balanced, but it is still not the same claim as Gemma's full mainline.

Forbidden conclusions remain:

- `Qwen has no evidence-sensitive mechanisms`
- `Qwen fully replicates Gemma-style source tracing` without automatic source-tracing gates
- `Qwen feature ids are semantic object nodes` without separate top-activation/heatmap evidence
