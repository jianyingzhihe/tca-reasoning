# Stage4-043 Qwen Native Route Verdict

## 目的

汇总 Stage4-038 到 042，给出 Qwen-native route 的最终限定结论。

## 输入

- V4 failure decomposition
- intervention operator sweep
- exact-position PLT route
- hidden-to-PLT mediation

## 输出

最终 verdict JSON 与中文结论。

## 方法

Decision statuses：

- `qwen_native_route_supported`
- `qwen_plt_mediated_distributed_route_supported`
- `qwen_hidden_route_plt_localization_failed`
- `qwen_sparse_plt_route_not_supported`
- `blocked`

## 结果

Stage4-038/042 已完成 primary 与 strict hidden-to-PLT mediation。

主要结果：

- V4 failure decomposition: `operator_or_feature_not_sufficient`
- exact-position candidate count 偏少：primary exact main `14`, strict exact main `10`
- hidden-to-PLT mediation primary: `qwen_route_may_live_in_plt_error`
- hidden-to-PLT mediation strict: `qwen_route_may_live_in_plt_error`

Primary/strict 均显示：

- full hidden residual restore 成立；
- PLT reconstruction error restore 几乎复现 hidden residual effect；
- sparse PLT topK reconstruction 未通过主 analyzer gate。

## 结论

当前 Stage4 Qwen-native verdict：

`qwen_route_may_live_in_plt_error`

可以写：

`Qwen2.5-VL shows a replicated hidden-level evidence-to-answer route at layer 14. Under the current public Qwen-PLT, this route is not mediated by sparse topK PLT features; the causal effect is largely preserved in PLT reconstruction error / non-feature residual.`

不能写：

`Qwen fully replicates Gemma-style sparse source tracing.`

也不能写：

`Qwen has no evidence-sensitive route.`
