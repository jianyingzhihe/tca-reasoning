# Stage4-066 Qwen Feature Route-Level Replication Run Plan

## 目的

本轮要把 Stage4-060 的结论从“单个 Qwen PLT feature node 可以满足 route-first causal gates”推进到“同一个 sample/prompt 内的一组 feature nodes 能不能作为一条 route bundle 被整体干预”。

这里的 route 不是 Gemma 的固定路线图，也不读取 Gemma node id。它的定义是：

```text
同一个 sample_id + prompt_name 下，由 Qwen route-first 方法发现的一组 PLT feature nodes。
```

本轮的核心顺序保持和 Stage4-060 一样：

```text
先冻结 2+3+4，再回看 1+5。
```

五个 gate 的含义固定为：

```text
1 = evidence-sensitive：遮真实证据区域时，route/node activation 明显变化，且强于 shifted/shuffled。
2 = clean source > controls：正常图像下剪 source route，比剪 matched/random controls 更伤答案。
3 = restore source > controls：遮证据后补 source route，比补 controls 更能救答案。
4 = real restore > shifted/shuffled：真实证据 mask 下的 restore 效果强于 shifted/shuffled mask。
5 = correct > wrong：对 gold/correct target 的影响强于 wrong target。
```

## 为什么要做这一轮

Stage4-060 已经支持：

```text
qwen_route_first_full_supported
```

也就是说，在单节点层面，Qwen 中层 PLT 里确实存在一批节点满足 1+2+3+4+5。但这还不是 route-level replication，因为它没有证明“同一个 sample/prompt 内的一组节点被当作整体 route 干预时也成立”。

本轮补的就是这个缺口：如果 grouped route 通过，那么可以写：

```text
Qwen supports a Qwen-native feature-level evidence-to-answer route under route-first discovery.
```

不能写：

```text
Qwen fully replicates Gemma-style automatic source tracing.
```

因为本轮仍然不是 Gemma 那种自动 source-tracing graph closure，而是 Qwen-native route-first discovery + grouped intervention。

## 输入

主输入来自已有 Stage4-060 artifact：

```text
doc/experiments/stage4/cross_model/stage4_qwen_route_first_routefirst_v1_route_candidates.csv
```

只使用其中：

```text
pack = primary
mode = full
route_first_234 = 1
```

然后按同一 `sample_id + prompt_name` 聚合为 route bundle。

## 输出

新增 artifact prefix：

```text
stage4_qwen_feature_route_*
```

主要输出：

```text
stage4_qwen_feature_route_primary_featureroute_v1_manifest.csv
stage4_qwen_feature_route_strict_featureroute_v1_manifest.csv
stage4_qwen_feature_route_featureroute_v1_manifest_summary.json
stage4_qwen_feature_route_primary_smoke_featureroute_v1_raw.csv
stage4_qwen_feature_route_primary_smoke_featureroute_v1_run.json
stage4_qwen_feature_route_primary_full_featureroute_v1_raw.csv
stage4_qwen_feature_route_primary_full_featureroute_v1_run.json
stage4_qwen_feature_route_strict_full_featureroute_v1_raw.csv
stage4_qwen_feature_route_strict_full_featureroute_v1_run.json
stage4_qwen_feature_route_featureroute_v1_summary.csv
stage4_qwen_feature_route_featureroute_v1_specificity.csv
stage4_qwen_feature_route_featureroute_v1_route_metrics.csv
stage4_qwen_feature_route_featureroute_v1_decision.json
```

## 方法

每个 route bundle 按 frozen route score 排序，取：

```text
topK = 4,8,16,32,64
```

对整组 nodes 做真实 grouped intervention：

```text
clean_route_zeroing：
  clean image 下同时剪掉 route 中所有 source feature directions。

mask_route_restore：
  answer/union/shifted/shuffled mask 下，同时补回 route 中所有 source feature clean-minus-mask deltas。

route_corrupt：
  clean image 下把 route 方向推向 masked state，作为方向性补充，不作为主 gate。
```

对照 route：

```text
same-size matched feature route
same-feature random-position route
random-active route
shifted/shuffled mask restore
wrong target
```

## 严谨性约束

Primary 负责：

```text
构造 route bundle
冻结 topK
冻结 route node 顺序
冻结 primary-selected exact nodes
```

Strict 只负责：

```text
复验 primary frozen route
不重新选 sample
不重新选 layer
不重新选 feature
不重新选 position
不重新调 threshold
```

Strict exact match 字段固定为：

```text
sample_id
prompt_name
layer
source_pos
source_feature_id
```

当前实现优先使用 route-first candidate_id，因为 Stage4-060 strict frozen manifest 已经保证这些 id 是 primary exact candidates 的复验版本；如果后续发现 id 不稳定，再切换为显式五元组 exact key。

## 预期结果

可能结果分四档：

```text
qwen_feature_route_supported：
  grouped feature route 在 primary/strict 都通过 1+2+3+4+5。

qwen_distributed_feature_route_supported：
  小 topK 或单节点较弱，但 grouped topK route 通过，说明 Qwen feature route 是分布式的。

qwen_feature_route_case_clustered：
  方向成立但 route 数或样本覆盖不足，只写 case-clustered support。

qwen_route_first_nodes_supported_route_unresolved：
  单节点 1+2+3+4+5 已成立，但 grouped route 没闭合。
```

如果失败，不推翻 Stage4-060 node-level support，也不推翻 hidden-level route；只说明当前 grouped feature route / PLT localization 仍未闭合。

