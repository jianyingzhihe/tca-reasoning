# Stage4-006 CLT Finalization Run Plan

## 目的

CLT 线用于 robustness / heterogeneity / negative diagnostic，不覆盖 PLT 主线。Stage4 需要把 Qwen-CLT 和 LLaVA-CLT 收口到最终辅助结论。

## Qwen-CLT

- 复用 paperpack72 primary/strict。
- 跑 topK sensitivity: `1,4,8,16,32`。
- 补 sequence-score bridge 与 passing-case decoded smoke。
- 与 Qwen-PLT 做同 sample/prompt/mask/control paired summary。
- 成功时写：`Qwen CLT robustness final, with representation-dependent effect size`。

## LLaVA-CLT

- 复用 paperpack72 primary/strict。
- 跑 layer sweep: `12,15,18,21`。
- 每层跑 topK sensitivity、source-control、real-vs-shuffled、correct-vs-wrong。
- 若仍失败，写：`LLaVA-CLT feature/source route not established under tested layers/controls`，不写“没有机制”。

## 输出

所有输出放在 `doc/experiments/stage4/cross_model/`，并由 `analyze_stage4_clt_finalization.py` 写 decision JSON。

## 当前执行状态

- Qwen-CLT primary topK1 已完成并拉回。
- Qwen-CLT primary topK4/8/16/32 已切换到 detached/resumable 后台任务。
- Qwen-CLT strict 尚未启动。
- LLaVA-CLT layer/topK diagnostic 尚未启动。

执行顺序保持：Qwen-CLT primary topK sweep → Qwen-CLT strict → LLaVA-CLT diagnostic。
