# 目的

Stage5 用 CLT 做机制异质性图谱：判断 evidence-to-answer 因果流在 CLT 表示中是否可定位、在哪些层/形态可见。CLT 不覆盖 PLT 主线，也不强行复现 Gemma-style sparse source tracing。

# 输入

- paperpack72 primary/strict prompt-runs 与 masks。
- Qwen2.5-VL-CLT: `KokosDev/qwen2p5vl-7b-clt`。
- LLaVA-1.5-CLT: `KokosDev/llava15-7b-clt`。
- Stage3/4 已有 Qwen/LLaVA CLT artifacts 与 Stage4 Qwen PLT hidden/error results。

# 输出

- `stage5_clt_heterogeneity_*` raw CSV/JSON。
- Stage5 summary、specificity、case table、decision JSON。
- Qwen-CLT、LLaVA-CLT、CLT-vs-PLT 表示对比文档。

# 方法

1. 先审计已有 CLT 结果：Stage3 Qwen-CLT primary/strict 是辅助支持；Stage4 Qwen topK sweep 不完整，因为 topK4/8/16/32 产物为 0 字节；Stage3 LLaVA-CLT 是 diagnostic partial。
2. 先 smoke，不写科学结论。
3. 再做 primary bounded screen：Qwen 不固定 L26，LLaVA 不固定 L15。
4. 只把 primary near-pass 的 layer/topK/operator 送入 full validation。
5. strict 只确认 primary-selected configs，不重新挑层或 topK。

# 结果

待运行。当前策略是等 Stage4 Qwen all-layer PLT full 完成后再启动 CLT full，避免 GPU 并行污染。

# 预期与实际偏差

预期 Qwen-CLT 可能比 Qwen-PLT sparse feature route 更可见；LLaVA-CLT 可能仍然弱或层依赖。失败只说明当前 CLT asset/层/topK/干预设置下 route 未建立，不说明模型没有跨模态机制。

# 结论

待 `007_stage5_clt_final_verdict.md` 汇总。
