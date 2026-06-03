# Run Plan Stage 2：从强 pilot 推进到可投稿机制论文

日期：2026-05-19

本文基于当前结果和批判性判断，规划 Stage 2 应该补什么实验、做什么工作、每一步的成功标准是什么。

核心判断：

> 当前证据已经足够支撑一个收窄后的机制结论：在 localized、strong image-dependence 的 VQA 样本中，VLM 答案附近存在 evidence-region-sensitive support routes。  
> 但如果要冲更强论文，尤其 ICLR 正会级别，还需要补：独立复现、node-to-generation 直接桥、少量高质量语义解释 case。

Stage 2 的目标不是扩大 claim，而是把当前 claim 打硬。

---

## 0. 当前已完成的主结论

当前 strongest claim：

> 在强视觉依赖、证据区域可定位的视觉问答样本中，VLM 答案生成附近存在带方向的内部因果路径。其中 support source routes 会被 wrong image 和 answer / union evidence-region mask 削弱；这种削弱强于 nearest matched non-source controls，也强于 same-area random controls，包括 random16；区域遮挡还会更频繁改变 decoded answers。

当前 strongest evidence：

```text
support source answer-mask weakening:
mean = +0.601
95% CI = [+0.313, +0.914]

support source union-mask weakening:
mean = +0.649
95% CI = [+0.351, +0.962]

source-minus-nearest answer weakening:
mean = +0.861
95% CI = [+0.250, +1.597]

source-minus-nearest union weakening:
mean = +0.924
95% CI = [+0.257, +1.729]

answer minus random16:
mean = +0.423
95% CI = [+0.058, +0.828]

union minus random16:
mean = +0.464
95% CI = [+0.112, +0.868]
```

Behavior side：

```text
B_direct answer_mask changed = 6/10
B_direct union_mask changed = 8/10
B_direct random4_valid changed = 4/22

D_visual_only answer_mask changed = 7/9
D_visual_only union_mask changed = 6/9
D_visual_only random4_valid changed = 6/21

format_prefix_ok = 100%
empty/error = 0
```

当前不能 claim：

1. `D_visual_only` 比 `B_direct` 更好；
2. source node 是明确 object-level semantic node；
3. node intervention 已经直接改变 decoded generation；
4. restoration 是主证据；
5. 机制适用于所有 VQA 类型。

---

## 1. Stage 2 总目标

Stage 2 的目标是把当前工作从：

```text
focused mechanism pilot
```

推进到：

```text
可复现、有因果行为桥、有解释性 case 的机制论文雏形
```

最重要的三个补强方向：

1. **Targeted replication pack**  
   独立复现当前 evidence-region-sensitive support route 结果。

2. **Node-to-generation direct bridge**  
   证明 source node intervention 不只改变 target logit，也能影响 decoded generation 或至少答案第一个 token 的生成分布。

3. **Feature / region semantic case**  
   对 2-3 个高质量 case 展示 source feature 与答案证据区域的关系，让工作更像解释性机制研究，而不是纯统计干预。

次要方向：

4. **Suppressor deep case**  
   解释 suppressor 到底是压制正确答案，还是支持竞争答案。

5. **Optional restoration / broader type extension**  
   作为后续，不进入 Stage 2 第一优先级。

6. **Cross-model feasibility and mini replication**  
   先做现成 VLM transcoder / CLT 资产的可用性验证；如果 smoke 成功，再做极小规模跨模型复现。它进入 Stage 2 计划，但不替代 Stage 2A targeted replication。

---

## 2. Stage 2 工作顺序

建议严格按以下顺序推进：

```text
Stage 2A: targeted replication pack
Stage 2B: node-to-generation bridge
Stage 2C: semantic interpretation cases
Stage 2D: suppressor deep case
Stage 2E: optional extensions
Stage 2F: cross-model feasibility and mini replication
```

原因：

1. replication 是审稿人最可能首先质疑的点；
2. node-to-generation 是因果闭环的最大缺口；
3. semantic case 提升解释性和图示质量；
4. suppressor 有价值但复杂，不能抢主线；
5. restoration 和 broader type 很重，暂时不是最小补强路径；
6. cross-model 很有论文价值，但必须先做 loader / hook / transcoder compatibility smoke，不能一上来承诺完整复现。

---

## 3. Stage 2A：Targeted Replication Pack

### 3.1 目的

验证当前 core24 结果不是少数样本或选择偏差造成的。

要回答：

> 在一批新的 localized、strong image-dependence 样本上，support source routes 是否仍然 answer/union evidence-region-sensitive，并且强于 nearest controls 和 random16 controls？

### 3.2 为什么这是第一优先级

当前 core24 有：

```text
support source rows = 13
support nearest rows = 9
source samples ≈ 8-10
```

结果很强，但样本量偏小。ICLR 正会审稿人很可能问：

1. 是否只在 easy localized cases 上成立；
2. 是否选择了结果好的样本；
3. 换一批样本是否复现；
4. random16 是否仍然成立；
5. behavior linkage 是否仍然成立。

所以 Stage 2A 必须做。

### 3.3 样本选择规则

目标样本数：

```text
新增 10-15 个 samples
目标 support source rows 至少 15-25 条
目标 support nearest rows 至少 10-15 条
```

样本必须满足：

1. 不与 core24 主分析样本重叠；
2. `image_dependence = strong`；
3. answer evidence localized；
4. answer region compact；
5. support source node 可用；
6. nearest control 可用；
7. target token metadata complete；
8. clean answer rank 不应极差；
9. B/D 至少一个 prompt 的 clean answer 可解释；
10. 不优先 diffuse / global / ambiguous cases。

类型优先级：

```text
第一优先级：symbol_text_reading
第二优先级：visual_readout
第三优先级：scene_inference 中证据仍然 compact 的样本
暂缓：entity_linking、diffuse scene gist、global tint、terrain/global context
```

### 3.4 样本发现流程

由于已有 extra annotation packs 与 current source/control backbone overlap 不足，不能直接复用。需要重新构造 replication pack。

流程：

1. 从当前或新 held-out visual-positive pool 中选候选样本；
2. 先跑 clean generation / target metadata；
3. 过滤 strong image-dependence；
4. 跑 answer-aligned tracing；
5. 选 top non-generic support source nodes；
6. 构造 nearest matched non-source controls；
7. 只把 support source + nearest 可用的样本送去标注；
8. 标注 answer / relate；
9. 跑 prefix-fixed region pipeline。

### 3.5 标注要求

标注类型仍然只用两类：

```text
answer: 最小答案证据区域
relate: 相关支持区域 / 上下文 / 第二证据区域
```

如果一个图有多个 answer 区域：

> 多个 answer shapes 都保留，分析时取 union。

标注注意：

1. 不标 diffuse global tint；
2. 不标全图场景 gist；
3. 如果 answer 几乎占满全屏，标记为 `large_answer_region`，不要作为主样本；
4. 如果问题本身多答案或模糊，标记为 `ambiguous`；
5. 优先能清晰说“遮这里答案会变”的样本。

### 3.6 实验条件

每个 `sample_id × prompt × node` 跑：

```text
clean
answer_mask
relate_mask
union_mask
random_control_1 ... random_control_16
```

prompt：

```text
B_direct
D_visual_only
```

node：

```text
support source
support nearest_control
secondary: suppressor source / nearest_control
```

random region：

```text
same area as answer mask
avoid answer ∪ relate
IoU threshold <= 0.06
filter invalid / high-IoU controls
```

### 3.7 指标

Primary route metrics：

```text
support_source_answer_mask_weakening
support_source_union_mask_weakening
support_source_answer_minus_random16
support_source_union_minus_random16
source_minus_nearest_answer_weakening
source_minus_nearest_union_weakening
```

Behavior metrics：

```text
target_rank_damage
margin_drop
decoded_answer_changed_from_clean
format_prefix_ok
empty_or_error
```

Typed metrics：

```text
symbol_text_reading
visual_readout
scene_inference
```

### 3.8 成功标准

Stage 2A 成功：

```text
support source answer/union weakening > 0，bootstrap CI 不跨 0
answer/union minus random16 > 0，bootstrap CI 不跨 0
source-minus-nearest answer/union weakening > 0，至少一个 CI 不跨 0
decoded answer change rate under answer/union mask > valid random controls
format_prefix_ok 接近 100%
empty/error 不显著增加
至少一个 type slice 复现
```

Stage 2A partial：

```text
support source weakening 复现
但 source-nearest 或 random16 只有方向一致，CI 跨 0
behavior answer change 弱但 margin/rank 方向一致
```

Stage 2A fail：

```text
answer/union mask 与 random16 无差异
source 与 nearest 无差异
decoded answer change 不高于 random controls
```

### 3.9 产物

输出目录建议：

```text
doc/experiments/stage2_replication_YYYY-MM-DD
```

必须产出：

1. `STAGE2_REPLICATION_RUN_PLAN.md`
2. `stage2_replication_manifest.csv`
3. `stage2_region_route_weakening.csv`
4. `stage2_random16_summary.csv`
5. `stage2_source_nearest_summary.csv`
6. `stage2_generation_summary.csv`
7. `stage2_typed_summary.csv`
8. `STAGE2_REPLICATION_READOUT.md`

---

## 4. Stage 2B：Node-to-Generation Direct Bridge

### 4.1 目的

补当前最大因果缺口。

当前已有：

```text
region mask -> route weakening
region mask -> decoded answer changes
```

缺少：

```text
node intervention -> decoded answer changes
```

Stage 2B 要回答：

> 清零 support source node 是否能改变答案生成，或者显著降低目标答案第一个 token 的生成倾向，并且强于 nearest control？

### 4.2 为什么重要

如果只证明区域遮挡同时影响 route 和 answer，审稿人可能说：

> route 变化和 answer 变化相关，但你没有证明这些节点本身导致答案变化。

如果能证明 node zeroing 改变 decoded generation 或 first answer token distribution，机制闭环会强很多。

### 4.3 工程现状

已尝试：

```text
run_region_mask_node_generation_smoke.py
```

结果：

- import 问题已通过 `PYTHONPATH` 修复；
- 模型加载成功；
- rows 选中；
- `feature_intervention_generate` 在 multimodal batch path 下触发 `AssertionError`。

判断：

> 当前 API 不支持我们需要的 multimodal batch generation path，需要自定义 generation loop 或改 hook 接口。

### 4.4 最小可行方案

不要一开始做完整自然语言生成。先做 first-token bridge。

方案 B1：first answer token distribution

对每个 case：

```text
clean forward
source zeroing forward
nearest zeroing forward
answer mask forward
```

比较：

```text
target_token_logit
target_token_prob
target_rank
margin
top-k token set
```

这其实我们已经部分有了，但 Stage 2B 要把它包装成 generation bridge：

> source zeroing 对第一答案 token 的影响是否和 answer mask 方向一致？

方案 B2：custom greedy one-step / multi-step generation

写自定义 loop：

1. 构造 multimodal batch；
2. 在第一个答案 token 位置施加 feature zeroing；
3. 取 argmax token；
4. append token；
5. 对后续 token 可以不再干预，或使用 open-ended intervention；
6. decode short answer；
7. 比较 baseline / source zeroing / nearest zeroing。

### 4.5 实验样本

先选 3-5 个强 case：

```text
okvqa_val_02444
okvqa_val_4739195
okvqa_val_1729795
可选：okvqa_val_3658865
可选：Stage 2A 新 replication 中最强 case
```

优先条件：

1. clean answer rank 好；
2. support source weakening 大；
3. source > nearest；
4. answer mask changes decoded answer；
5. answer region compact。

### 4.6 实验设计

每个 case 跑：

```text
clean generation
source support zeroing generation
nearest support zeroing generation
answer_mask generation
union_mask generation
```

如果完整 generation 工程困难，则先跑：

```text
first answer token distribution
```

### 4.7 成功标准

强成功：

```text
source support zeroing changes decoded answer
nearest control zeroing does not or weaker
source zeroing lowers target rank / margin
direction matches answer mask damage
```

中等成功：

```text
decoded answer 不变
但 source zeroing 显著降低 target token prob / rank / margin
且强于 nearest
```

失败：

```text
source zeroing 和 nearest zeroing 无差异
source zeroing 不影响 first token distribution
```

工程失败：

```text
multimodal generation hook 仍不可用
```

工程失败不等于机制失败，但要写进 limitation。

### 4.8 产物

建议输出：

```text
NODE_TO_GENERATION_ENGINEERING_NOTES.md
node_generation_smoke_results.csv
node_generation_case_table.csv
NODE_TO_GENERATION_READOUT.md
```

---

## 5. Stage 2C：Feature / Region Semantic Case

### 5.1 目的

补当前语义解释不足的问题。

当前能说：

> source route 对 evidence region 敏感。

还不能说：

> source feature 表示某个对象、文字、颜色或局部视觉概念。

Stage 2C 的目标不是解释所有 feature，而是做 2-3 个高质量 case，让论文更像机制解释。

### 5.2 选哪些 case

首选：

```text
okvqa_val_02444
```

理由：

- visual_readout；
- answer region 是孩子本体；
- decoded answer 从年龄变成其他答案；
- support route strong；
- 图好看，适合主 figure。

第二：

```text
okvqa_val_4739195
```

理由：

- symbol_text_reading；
- answer region 是牌子文字；
- D answer mask 从 Spanish 变 English；
- 可和 OCR 区域对齐。

第三：

```text
okvqa_val_1729795
```

理由：

- symbol/sign reading；
- answer mask 后 decoded answer 从 sign category 变 number；
- behavior 变化强。

### 5.3 要做什么分析

每个 case 尝试：

1. 展示原图、answer mask、relate mask、union mask；
2. 展示 source support node 的 clean effect；
3. 展示 answer/union mask 后 source effect weakening；
4. 展示 nearest control 对照；
5. 展示 random16 对照；
6. 展示 decoded answer change；
7. 分析 source feature activation 是否在 answer mask 后下降；
8. 如果能做到 patch-level，比较 answer region 内外 activation；
9. 如果是 OCR / text case，比较 answer mask 与 OCR box；
10. 收集 top activating examples 或 top activating patches。

### 5.4 最小成功标准

Stage 2C 不要求证明全部 feature semantics。

成功标准：

```text
在 2-3 个 case 中，source feature / route 的 effect 与 answer evidence region 有一致关系
遮挡 answer region 后 source activation 或 source effect 下降
nearest / random controls 无同等现象
decoded answer 同方向变化
```

可以写：

> These case studies suggest that the traced route is aligned with localized visual evidence.

不能写：

> This feature is a semantic detector for X.

除非有 top activating examples / activation maps 支持。

### 5.5 产物

```text
SEMANTIC_CASE_02444.md
SEMANTIC_CASE_4739195.md
SEMANTIC_CASE_1729795.md
semantic_case_summary.md
figure_assets_stage2/
```

---

## 6. Stage 2D：Suppressor Deep Case

### 6.1 目的

当前 suppressor 有信号，但解释不清。

Stage 2D 要回答：

> suppressor route 是压制正确答案，还是支持竞争答案？

### 6.2 样本选择

选条件：

1. suppressor source answer/union weakening 大；
2. source-minus-nearest suppressor gap 大；
3. 有明确 competitor token；
4. decoded answer 或 margin 有变化；
5. clean answer rank 可解释。

### 6.3 指标

记录：

```text
target logit
competitor logit
target-vs-competitor margin
top-k tokens
decoded answer
suppressor zeroing effect
nearest suppressor zeroing effect
answer/union mask effect
```

### 6.4 关键问题

1. 清零 suppressor 后 target logit 是否上升？
2. competitor logit 是否下降？
3. decoded answer 是否更接近 target？
4. suppressor 对 wrong answer token 是否有支持作用？
5. nearest suppressor control 是否没有类似效果？

### 6.5 成功标准

成功：

```text
suppressor source zeroing reduces competitor advantage
or improves target margin
or changes decoded answer toward target
and stronger than nearest control
```

partial：

```text
logit/margin 有方向，但 decoded answer 不变
```

失败：

```text
suppressor source 和 nearest 无差异
competitor relation 不清楚
```

### 6.6 写法

即使成功，也建议写成 secondary：

> Suppressor routes also appear evidence-sensitive, but their function is more heterogeneous and may correspond to answer competition rather than direct visual support.

---

## 7. Stage 2E：可选扩展

这些不是 Stage 2 必须项。

### 7.1 Restoration case refinement

只做 case study，不做主线。

可做：

- `okvqa_val_00558` multi-node curve；
- top-1/top-2/top-3/top-4 restoration；
- matched random group restoration；
- generation-side check。

不建议：

- 大规模 single-node restoration；
- 把 suppressor restoration 当主线；
- 在 position alignment 没解决前扩样本。

### 7.2 Broader type extension

如果 Stage 2A 复现成功，再尝试：

- scene_inference；
- OCR-heavy cases；
- entity_linking；
- counting；
- relation reasoning。

但这些会引入更复杂机制，不应抢当前 claim。

### 7.3 Restoration / broader type extension 的边界

这些扩展可以补充论文完整性，但不应该替代 replication / node-to-generation / semantic case。

建议只在 Stage 2A-C 有稳定进展后再做：

- restoration 只保留 case-level distributed-route evidence；
- broader type extension 只在主类型复现后扩展到 scene inference、counting、entity linking；
- 不要为了覆盖更多类型而牺牲 localized evidence 的主 claim 清晰度。

---

## 8. Stage 2F：Cross-model Feasibility and Mini Replication

### 8.1 目的

跨模型验证要回答的问题不是：

> 另一个模型是否完全复现 Gemma3-4B-IT 的所有节点和所有数值。

而是：

> 在另一个 VLM 中，是否也能观察到 answer-adjacent signed routes、support route evidence sensitivity、source-over-control specificity 这些同构现象？

这一步的论文价值很高，因为它能回应一个重要质疑：

> 当前结果是否只是 Gemma3-4B-IT 和 `tianhux2/gemma3-4b-it-plt` 这套 transcoder 的特例？

但它也是高风险工程项。跨模型验证必须先做 feasibility smoke，再决定是否升级成 mini replication。

### 8.2 当前已发现的现成资产

截至 2026-05-19，已有几个可能可用的公开 VLM transcoder / CLT 资产：

```text
Gemma3-4B-IT:
  transcoder_set = tianhux2/gemma3-4b-it-plt
  当前主实验已使用

Qwen2.5-VL-7B:
  candidate = KokosDev/qwen2p5vl-7b-clt
  model_name = Qwen/Qwen2.5-VL-7B-Instruct
  format = layer_*.safetensors + config.yaml
  priority = highest for cross-model smoke

Qwen2.5-VL-7B PLT:
  candidate = KokosDev/qwen2p5vl-7b-plt
  format = layer_*.safetensors + config.yaml
  priority = second, because model/card naming needs smoke verification

LLaVA-1.5-7B:
  candidate = KokosDev/llava15-7b-clt
  model_name = llava-hf/llava-1.5-7b-hf
  format = transcoder_L*.pt / mapping_L*.pt
  priority = lower, because current loader format likely needs more adapter work
```

注意：这些资产“存在”不等于能直接跑我们当前 pipeline。当前本地 `ReplacementModel` 仍主要绑定 `Gemma3ForConditionalGeneration` 和 `HookedVLTransformer` 的 Gemma3 路径；Qwen2.5-VL / LLaVA 需要 loader、processor、hook point、transcoder format 的兼容性验证。

### 8.3 与语言模型 circuit 资产的区别

已有很多小语言模型 circuit / transcoder 资产，例如：

```text
Gemma-2 2B
Llama-3.2 1B
Qwen-3 0.6B / 1.7B / 4B / 8B / 14B
```

这些可以用于方法 sanity check，但不能作为 VLM 视觉证据路径的主复现。原因是它们没有图像输入，也没有 image-region mask，因此最多支持：

> attribution / intervention pipeline 在另一个 LM 上可运行。

不能支持：

> VLM support routes are evidence-region-sensitive.

所以 Stage 2F 的优先级应该是 VLM 资产，而不是 language-only assets。

### 8.4 Stage 2F 分阶段设计

#### Stage 2F-0：公开资产核验

目的：

确认候选模型、transcoder 文件、config、hook point、模型名字是否真实可用。

输入：

```text
Hugging Face repo pages
config.yaml
model card
repo file list
local circuit_tracer_vlm loader
```

输出：

```text
cross_model_asset_survey.md
cross_model_candidate_table.csv
```

成功标准：

```text
至少找到 1 个 VLM candidate 满足：
1. 有明确 base VLM model_name
2. 有每层 transcoder / CLT 权重
3. 有 hook point 描述或能从 config 推断
4. 文件格式可被当前 loader 直接读取，或能写小 adapter 读取
```

当前初步判断：

```text
KokosDev/qwen2p5vl-7b-clt 是最高优先级。
KokosDev/qwen2p5vl-7b-plt 作为备用。
KokosDev/llava15-7b-clt 可能有价值，但 adapter 成本更高。
```

#### Stage 2F-1：Loader / hook smoke

目的：

只验证一件事：

> 另一个 VLM 能否被加载为 replacement / intervention-capable model，并读出 transcoder feature activations？

输入：

```text
1 个简单 image-question prompt
1 张本地 OK-VQA image
candidate transcoder_set
candidate base model
```

方法：

1. 尝试直接用 `ReplacementModel.from_pretrained(model_name, transcoder_set)`；
2. 如果失败，记录失败点：model class、processor、hook name、transcoder format、dimension mismatch；
3. 最小化 patch loader，不动主 pipeline；
4. 只跑一条 forward；
5. 读出指定层的 feature activation shape；
6. 检查是否能施加 feature zeroing 并得到 logits。

输出：

```text
cross_model_loader_smoke.md
cross_model_loader_smoke_log.txt
cross_model_loader_smoke_results.json
```

成功标准：

```text
model loads
image prompt works
transcoder activations load
feature activation shape matches config
single feature intervention changes logits without crash
```

失败判据：

```text
模型无法进入 HookedVLTransformer / equivalent wrapper
hook points 不存在或位置不一致
transcoder dimension 与 hidden state 不匹配
processor 无法和当前 prompt/image path 对齐
```

如果失败，要把它写成 engineering limitation，不把它解释为机制失败。

#### Stage 2F-2：One-case attribution / intervention smoke

目的：

在另一个 VLM 上跑通最小机制链：

```text
answer-aligned trace -> candidate support node -> source zeroing -> target logit effect
```

输入：

```text
1 个 localized visual_readout 或 symbol_text_reading sample
B_direct prompt
gold target token
answer mask if available
```

输出：

```text
cross_model_one_case_trace.pt
cross_model_one_case_nodes.csv
cross_model_one_case_intervention.csv
cross_model_one_case_readout.md
```

成功标准：

```text
能追踪出 answer-adjacent feature nodes
至少一个 support source node 清零后 target logit 降低
至少一个 matched nearest control effect 较弱
```

注意：这一步不要求证明 evidence-region sensitivity，只证明“另一个模型上 pipeline 可进入机制层”。

#### Stage 2F-3：Mini region replication

目的：

在另一个 VLM 上用极小样本验证主现象是否同构。

样本数：

```text
3-5 个 localized strong-image-dependence samples
优先复用已经有 answer / relate mask 的样本
```

条件：

```text
clean
answer_mask
union_mask
random16
```

节点：

```text
support source
nearest matched non-source control
```

指标：

```text
support source answer/union weakening
answer/union minus random16
source-minus-nearest weakening
target rank damage
decoded answer change if generation works
```

成功标准：

```text
方向复现即可，不要求 CI 稳定：
answer/union weakening > random16
source weakening > nearest weakening
rank/margin damage 与 route weakening 同方向
```

写法：

> Cross-model mini replication suggests that evidence-sensitive support routes are not unique to Gemma3-4B-IT, but the evidence remains preliminary due to small sample size and model-specific adapter constraints.

### 8.5 跨模型实验的优先级

跨模型进入 Stage 2，但执行顺序建议如下：

```text
必须先做：Stage 2A targeted replication
可以并行轻量做：Stage 2F-0 asset survey
Stage 2A 有稳定结果后：Stage 2F-1 loader smoke
Stage 2B 或 2C 至少一个完成后：Stage 2F-2 / 2F-3
```

如果资源有限，不要让跨模型挤掉 Stage 2A。原因是：

1. 独立复现是当前主 claim 最大缺口；
2. 跨模型首先是工程风险，不一定短期产出机制结果；
3. 即使跨模型 smoke 成功，也需要重新 trace、重新 source/control、重新分析；
4. 当前主线已经足够成为 focused mechanism claim，不依赖跨模型才能成立。

### 8.6 如果没有现成资产，是否手动训练？

可以，但不建议作为 Stage 2 主路。

手动训练 transcoder / CLT 的最低要求：

```text
1. 选一个 hook 友好的小 VLM
2. 准备 image-text activation dataset
3. 决定训练 PLT 还是 CLT
4. 缓存每层 MLP / residual activations
5. 训练 sparse transcoder
6. 验证 reconstruction loss、L0、dead features
7. 接入 ReplacementModel 或等价 wrapper
8. 跑 attribution / intervention smoke
9. 再进入 region-mask replication
```

候选模型优先级：

```text
优先：当前框架能 hook 的 Gemma3 family 小模型，如果有可用 vision variant
次优先：Qwen2.5-VL / Qwen2-VL 小模型，但需要 Qwen-VL wrapper
再次：SmolVLM / LLaVA small variants，但需要更多模型适配
```

手动训练风险：

1. 训练质量不稳定会污染机制结论；
2. 训练数据分布会影响 feature space；
3. VLM activation 缓存成本高；
4. 每层 transcoder 训练和验证都要额外时间；
5. 如果 wrapper 不稳定，训练好也未必能跑 attribution graph。

结论：

> Stage 2 先用现成资产做 cross-model feasibility。只有在 Qwen / LLaVA 现成资产无法适配、且主线 replication 已完成后，才考虑手动训练。

### 8.7 Stage 2F 产物

推荐输出目录：

```text
doc/experiments/stage2/cross_model/
```

必须产物：

```text
cross_model_asset_survey.md
cross_model_candidate_table.csv
cross_model_loader_smoke.md
cross_model_one_case_readout.md
```

如果 mini replication 成功，再产出：

```text
cross_model_region_replication_manifest.csv
cross_model_region_replication_results.csv
cross_model_region_replication_readout.md
```

---

## 9. Stage 2 不应该做什么

明确不要做：

1. 不要继续证明 `D_visual_only > B_direct`；
2. 不要把 `no_image` 纳入 strongest evidence；
3. 不要继续大规模 raw graph compare；
4. 不要在没有 replication 前扩大 claim；
5. 不要把 restoration 写成主证据；
6. 不要把 source node 直接命名成 object detector；
7. 不要把 diffuse/global mask 样本混入主 aggregate；
8. 不要在 invalid random masks 上做 specificity claim；
9. 不要把 cross-model loader smoke 写成跨模型机制复现；
10. 不要在 Stage 2A 复现完成前转向手动训练 transcoder。

---

## 10. Stage 2 时间与优先级建议

### 9.1 最小补强版本

如果时间有限，只做：

```text
Stage 2A targeted replication pack
```

这是最关键的。

完成后论文定位：

> strong workshop / weak-to-medium conference submission。

### 9.2 标准补强版本

建议做：

```text
Stage 2A targeted replication
Stage 2B first-token node-to-generation bridge
Stage 2C two semantic cases
```

完成后论文定位：

> 可认真冲 ICLR / NeurIPS / ICML 主会，但仍取决于写法和审稿口味。

### 9.3 强补强版本

如果资源充足：

```text
Stage 2A replication
Stage 2B decoded node-to-generation
Stage 2C semantic cases
Stage 2D suppressor deep case
Stage 2E restoration case curve
Stage 2F cross-model loader smoke or mini replication
```

完成后论文定位：

> 更完整的 mechanistic interpretability paper。

---

## 11. Stage 2 论文定位

建议标题方向：

```text
Evidence-Region-Sensitive Answer Routes in Vision-Language Models
```

中文理解：

```text
视觉语言模型中对关键证据区域敏感的答案路径
```

不要用：

```text
Visual Evidence Prompting Makes VLMs More Grounded
```

也不要用：

```text
We Discover Object-Level Visual Circuits in VLMs
```

推荐贡献：

1. 提出问题：VLM answer generation 是否存在 evidence-region-sensitive internal routes；
2. 提出验证链：answer-aligned tracing + node zeroing + wrong image + evidence mask + matched controls + random controls + decoded generation；
3. 发现机制：localized strong-image-dependence cases 中 support routes 对 evidence regions 敏感，强于 controls，并关联最终答案变化；
4. secondary：suppressor routes also recur, but heterogeneous。

---

## 12. Stage 2 成功后的最终 claim

如果 Stage 2A-C 成功，最终 claim 可以升级为：

> Across an original core24 pack and an independent targeted replication pack, localized visual-evidence VQA cases contain answer-adjacent support routes that are causally identifiable, evidence-region-sensitive, stronger than matched controls, stronger than same-area random region controls, and behaviorally relevant to answer generation. Case-level analyses further show that these routes align with localized visual evidence regions.

中文：

> 在原始 core24 和独立 targeted replication pack 中，我们发现 localized 视觉证据问答样本的答案附近存在可因果识别的 support routes。这些 routes 对 answer / union evidence regions 敏感，强于 matched controls 和 random region controls，并与最终答案生成相关。高质量 case 进一步显示，这些 routes 与局部视觉证据区域对齐。

仍然不建议升级为：

> 我们解释了所有 VLM 视觉推理。

---

## 13. 立即下一步行动清单

### Step 1：冻结 Stage 2A candidate selection rule

写一个：

```text
STAGE2_REPLICATION_SAMPLE_SELECTION.md
```

内容包括：

- inclusion criteria；
- exclusion criteria；
- type priority；
- required source/control metadata；
- annotation instructions。

### Step 2：生成 candidate table

从 visual-positive / held-out pool 生成：

```text
stage2_candidate_samples.csv
```

字段：

```text
sample_id
image_path
question
gold_answer
prompt
clean_answer
target_rank
reasoning_operation
image_dependence
visual_structure
support_source_available
nearest_control_available
answer_region_likely_compact
exclude_reason
priority_score
```

### Step 3：人工筛选 20-30 个候选

从中挑 10-15 个送标。

### Step 4：标注 answer / relate

继续使用 LabelMe。

### Step 5：跑 prefix-fixed region pipeline

包括：

- route intervention；
- random16；
- behavior rank/margin；
- decoded generation。

### Step 6：写 replication readout

明确：

```text
success / partial / fail
```

不要事后改 selection rule。

---

## 14. 最终建议

Stage 2 不要再换主故事。

主故事已经对了：

> localized VQA 中存在 evidence-region-sensitive answer support routes。

现在要做的是：

1. 复现它；
2. 把它连接到生成行为；
3. 用少量 case 解释它；
4. 保持 claim 边界。

如果只做一件事：

> targeted replication pack。

如果做两件事：

> targeted replication pack + node-to-generation first-token bridge。

如果做三件事：

> targeted replication pack + node-to-generation + semantic case studies。
