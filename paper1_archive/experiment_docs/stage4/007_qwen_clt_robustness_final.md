# Stage4-007 Qwen-CLT Robustness Final

## 目的

检验 Qwen 的 evidence-region-sensitive route 是否对 transcoder 类型稳健，而不是只在 PLT 中出现。

## 输入

Stage3 已完成 Qwen-CLT primary/strict。Stage4 补 topK sensitivity、sequence-score 和 decoded smoke。

## 结果

Stage4 Qwen-CLT primary topK sweep 已启动。

已完成并拉回：

- `topK1 feature_union`: usable runs 144, rows 4320。
- `topK1 source_control`: usable pairs 231, rows 1848。

Interim analyzer on `topK1`:

- status: `qwen_clt_robustness_final`。
- best feature mean source-control gap: `+0.127253`。
- best source-control mean gap: `+0.195985`。

仍在后台运行：

- detached PID: `743301`。
- remote log: `/root/autodl-tmp/tca-reasoning/stage4_clt_finalization/detached_stage4_clt_qwen_clt_primary_full_4_8_16_32.log`。
- remaining topK: `4,8,16,32`。

该后台 runner 是 resumable/skip-existing：已有 topK1 不会重跑。

## 结论

当前只能写 interim：Qwen-CLT topK1 已支持 robustness direction，但 final verdict 需等待 topK4/8/16/32 与 strict pack。

CLT 成功只写 robustness support；CLT 失败也不推翻 PLT 主线。
