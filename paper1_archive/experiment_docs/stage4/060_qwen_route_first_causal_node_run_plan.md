# Stage4-060 Qwen Route-First Causal Node Run Plan

## 目的

本轮把 Qwen2.5-VL-PLT 的搜索顺序反过来：先找更像“答案传输线”的节点，再回头看它们是否和真实图像证据、正确答案特异性对齐。

固定 gate：

```text
1 = evidence-sensitive
2 = clean source > controls
3 = restore source > controls
4 = real restore > shifted/shuffled
5 = correct > wrong
```

主发现规则是 `2+3+4`。`1` 和 `5` 只做回看与升级，不再作为发现前置门槛。

## 输入

- Qwen2.5-VL-PLT primary broad candidate artifacts。
- `stage4_qwen_all_layer_bounded_exhaustive_primary_full_L10-L17_candidates.csv`
- `stage4_qwen_middle_dense_primary_full_L10_*_candidates.csv`
- paperpack72 primary / strict prompt-runs 和 masks。

## 方法

候选只做最小工程过滤：身份字段齐全、image/mask 路径存在、`clean_target_rank <= 10`、target token 存在。保留候选集中性，不做 sample cap。

Primary full 跑 L10-L17 全候选；分析器先筛 `2+3+4`，再回看 `1/5`。Strict 只复验 primary-selected exact candidates，不重新挑层、节点或阈值。

## 预期

如果存在一批 `2+3+4` 候选，即使集中在少数样本，也记录为 Qwen route-first candidates exist；只有 primary + strict 同时满足 `1+2+3+4+5`，才升级为 Qwen-native route-first full support。

## 结论边界

本轮即使成功，也只支持 Qwen-native route-first causal candidates；不写 Qwen fully replicates Gemma-style source tracing，除非后续 automatic source-tracing graph 也闭合。

