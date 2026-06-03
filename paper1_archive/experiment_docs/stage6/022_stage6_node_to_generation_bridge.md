# Stage6-022 Node-to-Generation Bridge

Updated: 2026-06-02

## Question

Most route evidence is logit/rank/margin based. A small decoded bridge can make the story easier to understand:

```text
Do strong node/route interventions sometimes change the generated answer?
```

This is optional and case-level. It should not become a new main requirement.

## Gemma Case Design

Choose 1 smoke case and up to 3-5 full cases from strong Gemma source-tracing examples.

Conditions:

```text
clean generation
hidden/source bridge corruption generation
matched control corruption generation
masked image generation
hidden/source bridge restore generation
```

The initial smoke uses hidden-residual bridge interventions because graph-level source-route zeroing is not yet a generation-time operator in the current runner.

Expected strongest case:

```text
source/hidden bridge corruption or evidence mask changes the answer;
matched control corruption does not;
source/hidden bridge restore improves target rank/margin or decoded answer.
```

## Qwen Case Design

Choose strong Qwen hidden or feature-node cases.

Conditions:

```text
clean generation
hidden restore generation
strong feature node zeroing/restoration generation
matched control intervention generation
```

Qwen decoded bridge is allowed to be weaker because Qwen's strongest support is currently hidden-level and individual feature-node level, not closed grouped sparse route.

## Metrics

```text
decoded answer
target token logit
target rank
target-vs-wrong margin
format failure
empty answer
```

Decoded answer change is useful but not required. Rank/margin movement is accepted as the fallback bridge.

## Decision Labels

```text
decoded_bridge_case_supported
decoded_bridge_rank_margin_only
decoded_bridge_format_blocked
decoded_bridge_not_attempted
```

## Boundary

Do not run a large decoded generation sweep unless the first cases are clean. The paper can use this as an illustrative figure/table, not a core proof.

## Smoke Implementation

Stage6-022 smoke has been implemented with tag:

```text
defensive_v1
```

Outputs:

```text
stage6_node_generation_bridge_smoke_defensive_v1_gemma.csv
stage6_node_generation_bridge_smoke_defensive_v1_qwen2p5vl_plt.csv
stage6_node_generation_bridge_smoke_defensive_v1_case_table.csv
stage6_node_generation_bridge_smoke_defensive_v1_summary.csv
stage6_node_generation_bridge_smoke_defensive_v1_decision.json
```

Implementation boundary:

```text
Gemma v1 smoke uses a hidden-residual generation bridge at the confirmed strong case,
not graph-level source-route zeroing and not PLT generation-time reconstruction.

Qwen v1 smoke reuses the existing multifeature sequence bridge on a strong
Qwen2.5-VL-PLT case, not hidden-residual generation-time restore/corrupt.
```

This boundary is important. The smoke should be read as:

```text
Can already-confirmed internal evidence-flow objects move generation-side
rank / margin / sequence / short decoded outputs in strong cases?
```

It should not be read as:

```text
Gemma graph source-tracing route zeroing has now been run during greedy generation.
```

## Smoke Result

Decision:

```text
status = decoded_bridge_smoke_ready

Gemma:
asset = gemma3_hidden
status = decoded_bridge_rank_margin_only

Qwen:
asset = qwen2p5vl_plt
status = decoded_bridge_case_supported
```

Gemma smoke case:

```text
sample_id = okvqa_val_2100995
prompt = B_direct
target = wicker
clean decoded answer = rattan
masked decoded answer = 3d
```

Gemma hidden restore result:

```text
oriented_source_minus_control_sequence = +24.5891
oriented_source_minus_control_first_token = +30.5625
oriented_source_minus_control_rank = +4191.0
oriented_source_minus_control_margin = +2.3125
decoded restore to clean = 0
```

Interpretation:

```text
Gemma hidden restore strongly moves sequence score, first-token logit/rank,
and target-vs-wrong margin in the correct direction, but does not restore the
short greedy decoded answer in this case.
```

Qwen smoke case:

```text
sample_id = okvqa_val_3959785
prompt = B_direct
target = kuwait airway
clean decoded answer = kuwait airways
masked decoded answer = 0
```

Qwen PLT summary:

```text
restore:
mean_oriented_sequence_gap = +0.0176
mean_oriented_first_token_gap = +0.1547
mean_oriented_rank_gap = +59.4

corrupt:
mean_oriented_sequence_gap = +0.0798
source decoded changed vs reference = 2 / 5 topK rows
```

Interpretation:

```text
Qwen PLT feature bridge gives a small positive rank/sequence bridge and
some decoded-answer change under corrupt interventions. Restore still does
not stably bring the greedy decoded answer back to the clean answer.
```

## V1 Mini-Full Result: Historical, Not Final Symmetric Comparison

After the smoke passed, Stage6-022 was expanded to a small full case panel:

```text
mode = full
tag = defensive_v1
status = decoded_bridge_full_ready
Gemma rows = 18
Qwen PLT rows = 156
Qwen CLT rows = 156
```

Gemma hidden bridge:

```text
decision = decoded_bridge_case_supported

restore:
case_count = 3
mean_oriented_sequence_gap = +18.0716
mean_oriented_first_token_gap = +27.6875
mean_oriented_rank_gap = +1601.3333
mean_oriented_margin_gap = +11.2708
decoded restore to clean = 0

corrupt:
case_count = 3
mean_oriented_first_token_gap = +1.5833
mean_oriented_rank_gap = +1.3333
mean_oriented_margin_gap = +1.1667
decoded changed vs reference = 1 / 3
```

Qwen PLT bridge:

```text
decision = decoded_bridge_case_supported

restore:
case_count = 15
mean_oriented_sequence_gap = -0.0318
mean_oriented_first_token_gap = +0.0938
mean_oriented_rank_gap = +57.6667
decoded restore to clean = 0

corrupt:
case_count = 15
mean_oriented_sequence_gap = +0.2049
decoded changed vs reference = 2 / 15
```

Qwen CLT bridge:

```text
decision = decoded_bridge_rank_margin_only

restore:
case_count = 15
mean_oriented_sequence_gap = +0.1269
mean_oriented_first_token_gap = +0.2896
mean_oriented_rank_gap = +500.85
decoded restore to clean = 0

corrupt:
case_count = 15
mean_oriented_sequence_gap = +0.0385
mean_oriented_first_token_gap = +0.0146
decoded changed vs reference = 0 / 15
```

Full interpretation:

```text
The mini-full confirms the smoke-level boundary. Internal interventions move
generation-side scores, especially Gemma hidden restore and Qwen CLT restore.
Stable greedy decoded restoration is still not established. Decoded changes
appear only in limited corrupt cases for Gemma hidden and Qwen PLT.
```

This v1 result is useful as a first decoded bridge, but it is not the final
cross-model comparison because the two model arms used different stores:

```text
Gemma v1 = hidden-residual generation bridge only.
Qwen v1 = PLT/CLT multifeature generation bridge only.
```

The final Stage6-022 interpretation should use the symmetric v2 run below.

## Symmetric V2 Mini-Full Result

The symmetry objection was correct: Qwen should also get a hidden-residual
generation bridge, and Gemma should also get PLT/topK generation-time operators.
This was implemented under:

```text
mode = full
tag = defensive_v2_symmetric
status = decoded_bridge_full_ready

Gemma rows = 90
Qwen hidden rows = 16
Qwen PLT rows = 156
Qwen CLT rows = 156
```

The symmetric v2 lenses are:

```text
Gemma hidden_residual
Gemma plt_topk_reconstruction
Gemma plt_reconstruction_error

Qwen hidden_residual
Qwen PLT multifeature
Qwen CLT multifeature
```

Gemma hidden-residual bridge:

```text
decision = decoded_bridge_case_supported

restore:
case_count = 3
mean_oriented_sequence_gap = +18.0716
mean_oriented_first_token_gap = +27.6875
mean_oriented_rank_gap = +1601.3333
mean_oriented_margin_gap = +11.2708
decoded restore to clean = 0

corrupt:
case_count = 3
mean_oriented_first_token_gap = +1.5833
mean_oriented_rank_gap = +1.3333
mean_oriented_margin_gap = +1.1667
decoded changed vs reference = 1 / 3
```

Gemma PLT reconstruction-error bridge:

```text
decision = decoded_bridge_case_supported

restore:
case_count = 9
mean_oriented_sequence_gap = +19.1819
mean_oriented_first_token_gap = +26.8681
mean_oriented_rank_gap = +1362.8889
mean_oriented_margin_gap = +9.7014
decoded restore to clean = 2 / 9

corrupt:
case_count = 9
mean_oriented_first_token_gap = +1.5556
mean_oriented_rank_gap = +0.7778
mean_oriented_margin_gap = +0.6389
decoded changed vs reference = 3 / 9
```

Gemma sparse PLT topK reconstruction bridge:

```text
decision = decoded_bridge_rank_margin_only

restore:
case_count = 9
mean_oriented_sequence_gap = -0.5111
mean_oriented_first_token_gap = -0.4462
mean_oriented_rank_gap = -585.3333
positive_rank_frac = 0.4444
decoded restore to clean = 0

corrupt:
case_count = 9
mean_oriented_sequence_gap = +0.3811
positive_sequence_frac = 0.6667
decoded changed vs reference = 0
```

Qwen hidden-residual bridge:

```text
decision = decoded_bridge_case_supported

restore:
case_count = 4
mean_oriented_sequence_gap = +8.6032
mean_oriented_first_token_gap = +7.0859
mean_oriented_rank_gap = +2568.75
mean_oriented_margin_gap = +2.25
decoded changed vs reference = 3 / 4

corrupt:
case_count = 4
mean_oriented_sequence_gap = +5.8258
mean_oriented_first_token_gap = +7.0313
mean_oriented_rank_gap = +703.5
mean_oriented_margin_gap = +7.1406
decoded changed vs reference = 3 / 4
```

Qwen PLT and CLT multifeature bridges:

```text
Qwen PLT restore:
case_count = 15
mean_oriented_sequence_gap = -0.0318
mean_oriented_first_token_gap = +0.0938
mean_oriented_rank_gap = +57.6667
decoded restore to clean = 0

Qwen PLT corrupt:
case_count = 15
mean_oriented_sequence_gap = +0.2049
decoded changed vs reference = 2 / 15

Qwen CLT restore:
case_count = 15
mean_oriented_sequence_gap = +0.1269
mean_oriented_first_token_gap = +0.2896
mean_oriented_rank_gap = +500.85
decoded restore to clean = 0

Qwen CLT corrupt:
case_count = 15
mean_oriented_sequence_gap = +0.0385
decoded changed vs reference = 0 / 15
```

Symmetric v2 interpretation:

```text
Both Gemma and Qwen show generation-side score movement under hidden-residual
restore/corrupt interventions.

Gemma's generation bridge mirrors the hidden-to-PLT decomposition: sparse PLT
topK reconstruction is weak or mixed, while PLT reconstruction error carries
the strong generation-side effect.

Qwen hidden-residual bridge is stronger than Qwen PLT/CLT multifeature bridge
in this small panel, consistent with the broader claim that Qwen's evidence flow
is most visible in hidden/residual space and less cleanly closed as a sparse
feature bundle.
```

## Core Conclusion

The complete Stage6-022 conclusion is:

```text
This experiment is a generation-side score bridge, not a stable decoded-control
claim. Under a symmetric comparison, hidden-residual interventions in both Gemma
and Qwen move answer-side sequence score, rank, margin, and first-token score.

However, sparse feature-store interventions only partially capture that bridge.
For Gemma, PLT reconstruction error carries the strong generation-side effect,
while sparse PLT topK reconstruction is weak or mixed. For Qwen, the hidden
bridge is stronger than the tested PLT/CLT feature-store bridge.

Therefore, the most defensible interpretation is that both models contain
evidence-to-answer flow that can affect generation-side scoring, but the flow is
most directly visible in hidden/residual space. Sparse feature bundles are useful
diagnostic objects, but they do not fully mediate the generation-side effect in
this small panel.
```

中文论文表述可以写成：

```text
Stage6-022 的对称 node-to-generation bridge 显示，Gemma 和 Qwen 的 hidden-residual
干预都会显著移动答案生成侧的 sequence score、rank、margin 和 first-token score。
但这种生成侧桥接并没有被 sparse topK feature 完整承载：Gemma 中更强的是
PLT reconstruction error，而不是 PLT topK reconstruction；Qwen 中 hidden-residual
bridge 也强于当前测试的 PLT/CLT feature-store bridge。因此，这个实验支持
“两类模型都有 evidence-to-answer flow，但该 flow 最直接可见于 hidden/residual
空间，稀疏 feature bundle 只部分捕捉它”的解释。
```

## Current Wording

Safe paper wording:

```text
In a small symmetric Stage6 decoded-bridge case panel, hidden-residual
interventions in both Gemma and Qwen move generation-side rank, margin, and
sequence scores. For Gemma, PLT reconstruction error, not sparse PLT topK
reconstruction, carries the stronger generation-side effect. For Qwen,
hidden-residual interventions are stronger than the tested PLT/CLT
multifeature bridges. Greedy decoded answer control remains case-level and
partial, so this result is illustrative defensive evidence rather than a
required generation-level causal proof.
```

Do not write:

```text
Stage6-022 proves stable decoded generation control.
Gemma graph source-tracing route zeroing has been tested during generation.
Qwen restore reliably brings masked generations back to the clean answer.
```
