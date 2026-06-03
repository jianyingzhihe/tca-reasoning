# Stage4-009 Final Verdict

## 当前状态

Qwen2.5-VL-PLT true source-tracing adapter 已完成 primary72 与 strict72 full。

- primary72: `qwen_source_tracing_not_supported`。
- strict72: `qwen_source_tracing_not_supported`。
- 原因：adapter graph/compare 成功，但 source-traced feature node zeroing 没有稳定损伤 target。
- 这不推翻 Stage3 Qwen-PLT approximate feature/source-control support；它只说明当前 Qwen true source-tracing adapter 未复现 Gemma-style source tracing。

CLT finalization 正在进行。Qwen-CLT primary topK1 已给出正向 interim robustness signal，topK4/8/16/32 正在 detached 后台任务中继续。

## 可用结论模板

- `qwen_full_source_tracing_supported`：Qwen adapter 跑通，primary/strict graph compare、node intervention、source/control、negative controls 与 behavior bridge 达到预注册判据。
- `qwen_approx_only`：Qwen 仍只有 Stage3 approximate source-control support，full source tracing 未达标。
- `qwen_adapter_blocked`：adapter 或远端资源工程阻塞，不能写 Qwen 负结果。
- `qwen_source_tracing_not_supported`：adapter 跑通，但 source-control route 或 negative controls 失败。

## Qwen-PLT 当前结论

可写：

`Qwen2.5-VL shows PLT approximate feature/source-control support on paperpack72, but current Qwen answer-aligned source-tracing adapter does not replicate Gemma-style source tracing on primary or strict paperpack.`

不可写：

`Qwen fully replicates Gemma-style source tracing.`

也不可写：

`Qwen lacks evidence-region-sensitive mechanisms.`

## 禁止口径

- 不从 adapter blocked 推出机制负结论。
- 不把 LLaVA-CLT failure 写成没有跨模态机制。
- 不让 CLT 结果覆盖 PLT 主线。
- 不写 `D_visual_only` 更好。

## 2026-05-24 Critical Audit Update

Stage4-011 found that the first Qwen source-tracing adapter result should be treated as **provisional**, not as a final negative result.

Reason:
- `run_qwen_answer_aligned_attribute.py` selected features from `outputs.hidden_states[args.layer]`, while intervention was applied to `language_model.layers[args.layer]`.
- In common HuggingFace decoder outputs, `hidden_states[0]` is the embedding state and `hidden_states[layer + 1]` is the layer output, so this can create layer-output mismatch.
- The adapter also only tested layer 26, visual/answer-adjacent positions, support features, top3 compare nodes, and top2 intervention nodes per sample/prompt.

Fix applied:
- Feature selection now uses a forward hook on `language_model.layers[args.layer]` and records `hidden_capture_source`.
- The script compiles successfully after the fix.

Updated claim boundary:
- Current Qwen-PLT approximate feature/source-control support remains valid.
- The old `qwen_source_tracing_not_supported` result is not enough for a final negative claim.
- Before finalizing Qwen source-tracing status, rerun hook-aligned smoke, primary72, strict72, and ideally layer/topK sensitivity.

## 2026-05-24 Hook-Aligned Rerun Result

The hook-aligned Qwen rerun has now completed. See `012_qwen_source_tracing_hookfix_rerun.md`.

Completed runs:
- layer 26 primary72 full, top2 nodes: `qwen_source_tracing_not_supported`
- layer 26 strict72 full, top2 nodes: `qwen_source_tracing_not_supported`
- layer 26 primary72 full, top8 nodes: `qwen_source_tracing_not_supported`
- layer 26 strict72 full, top8 nodes: `qwen_source_tracing_not_supported`
- layer 22 smoke: artifacts passed, direction not supported
- layer 24 smoke: artifacts passed, direction not supported
- layer 27 smoke: blocked by `index 27 is out of range` in the current PLT asset/loader

Updated Qwen-PLT conclusion:

`Qwen2.5-VL-PLT retains approximate feature/source-control and hidden/first-token bridge evidence, but the stricter hook-aligned Gemma-style source-tracing adapter is not supported on paperpack72.`

This is now stronger than the earlier provisional result because it resolves the layer-output mismatch and adds topK/layer sensitivity. It is still not a claim that Qwen lacks evidence-sensitive mechanisms.

## 2026-05-24 Causal Cutter Map Update

Stage4-013 mined the completed hookfix top8 intervention rows for rank-aware cuttable nodes.

Result:
- Sparse cuttable Qwen-PLT nodes exist, especially among rank-aware numeric-answer cases.
- Primary top8 rank-aware cutter rows: 157 rows across 35 samples.
- Strict top8 rank-aware cutter rows: 142 rows across 35 samples.
- Recurring candidate features include `1091`, `1215`, `2508`, `3353`, and `4383`.

Interpretation:

`Qwen has sparse causal-screened PLT cutter candidates, but automatic Gemma-style source tracing still does not reliably select a full causal route.`

Next required step:

Run targeted validation for these cutter candidates with wrong-target, answer/union-mask, shifted/shuffled, same-position matched-feature, and same-feature random-position controls.

## 2026-05-24 Stage4-014 Native Cutter Validation Added

Stage4-014 now separates two claims that were easy to blur:

- Gemma-style automatic source tracing: the adapter automatically selects a route and that route passes intervention/control tests.
- Qwen-native causal cutter validation: Qwen candidates are first found from Qwen's own hook-aligned top8 intervention rows, then rerun with stricter controls.

The new candidate manifest is:

`cross_model/stage4_qwen_causal_cutter_candidate_manifest.csv`

Manifest audit:

- Discovery source: primary Qwen top8 intervention only.
- Strict top8 is confirmation/sensitivity only, not candidate selection.
- Rank<=10 prompt-run candidates: `17`.
- Main candidates: `12`, with `7` numeric and `5` non-numeric.
- Strict-confirmed main candidates: `10`.

If this validation succeeds, the upgraded wording is:

`Qwen causal-screened PLT cutter nodes exist.`

If evidence-mask restore and wrong-target controls also pass:

`Qwen causal-screened evidence-linked cutter support.`

Even then, this still does not become:

`Qwen fully replicates Gemma-style source tracing.`

## 2026-05-24 Stage4-014 Validation Result

Stage4-014 completed the Qwen-native causal cutter validation.

Main run:

- 12/12 candidates completed.
- 480 raw intervention/control rows.
- Decision: `causal_cutter_supported`.
- `clean_source_minus_controls`: mean `0.7951`, CI `[0.3819, 1.2743]`.
- `clean_correct_minus_wrong`: mean `0.1094`, CI `[0.0625, 0.1615]`.
- Numeric slice is strong; non-numeric slice is positive but much smaller.

Rank<=10 sensitivity:

- 17/17 candidates completed.
- 680 raw intervention/control rows.
- Decision remains `causal_cutter_supported`.
- `clean_source_minus_controls`: mean `0.5748`, CI `[0.2672, 0.9424]`.

Mask/evidence-link controls:

- Answer/union restore did not reliably beat shifted/shuffled controls.
- Therefore Stage4-014 does not support `evidence_linked_cutter_supported`.

Updated Qwen-PLT wording:

`Qwen2.5-VL-PLT has Qwen-native causal-screened cutter nodes, especially strong for numeric-answer cases and smaller but positive for non-numeric cases. However, these nodes are not yet evidence-mask-linked under shifted/shuffled controls, and the hook-aligned automatic source-tracing adapter still does not replicate Gemma-style source tracing.`

Forbidden wording remains:

- `Qwen fully replicates Gemma-style source tracing.`
- `Qwen has no evidence-sensitive mechanisms.`
- `Qwen cutter feature ids are semantic object nodes.`

## 2026-05-24 Stage4-016/017/018 Expanded Qwen V2 Result

We ran the overnight-style Qwen V2 plan to test whether the Stage4-014 result was too small or too post-hoc.

Expanded automatic source tracing:

- L26 primary top32: `qwen_source_tracing_not_supported`
- L26 strict top32: `qwen_source_tracing_not_supported`
- L22/L24 sensitivity: not supported
- visual-only and answer-adjacent-only sensitivity: not supported
- The expanded runner completed with `140/144` prompt-runs usable for primary and strict.

Expanded cutter discovery:

- main candidates increased from `12` to `31`
- pool candidates: `43`
- main numeric: `16`
- main non-numeric: `15`
- strict-confirmed main: `26`

Expanded clean source/control validation:

- decision: `causal_cutter_supported`
- `clean_source_minus_controls`: mean `0.4872`, CI `[0.2870, 0.7030]`
- numeric slice remains strong; non-numeric slice remains positive but small

Evidence-link V2:

- grouped top1/top4/top8/top16 restore completed
- final decision: `qwen_cutter_only_supported`
- union activation-drop has partial real-vs-shifted signal, but grouped restore does not beat source controls or shifted/shuffled controls

Updated Qwen-PLT verdict:

`Qwen2.5-VL-PLT has robust Qwen-native causal-screened PLT cutter nodes on an expanded candidate set, but current automatic source tracing still does not replicate Gemma-style route selection, and evidence-mask-linked cutter support remains unestablished under shifted/shuffled controls.`

## 2026-05-24 Stage4-020 Evidence-First Follow-Up

Stage4-020 is added as the next Qwen-only PLT test. The key correction is methodological: instead of starting from nodes that hurt the target when cut, it starts from Qwen PLT feature/position pairs that are genuinely more sensitive to `answer/union` evidence masks than to `shifted/shuffled` controls, then asks whether those nodes support the answer.

New files:

- `020_qwen_evidence_first_route_run_plan.md`
- `021_qwen_evidence_first_route_results.md`
- `022_qwen_adapter_v3_evidence_biased_route.md`
- `023_qwen_evidence_first_final_verdict.md`

Current status: completed for primary + strict full.

Stage4-020 result:

- Evidence-first primary: `384` candidates, `192` main, covering `96` prompt-runs.
- Evidence-first strict: `404` candidates, `202` main, covering `101` prompt-runs.
- Evidence-first clean zeroing source-minus-controls is near zero and CI crosses 0 in both primary and strict.
- Evidence-first restore does not pass source/control, real-vs-shifted/shuffled, correct-vs-wrong, or rank gates.
- Adapter V3 primary/strict also fails source/control, correct-vs-wrong, and rank gates.

Therefore the attempted upgrade is not supported:

`Qwen-native evidence-linked route support.`

Adapter V3 also does not support:

`Qwen evidence-biased adapter route support.`

Current Stage4-020 verdict:

`qwen_not_gemma_style_under_adapter`

This means the negative result is about the current Qwen adapter / evidence-first route validation setup, not about all possible Qwen mechanisms. The retained positive Qwen conclusion remains:

`Qwen2.5-VL-PLT has robust Qwen-native causal-screened PLT cutter nodes.`

Still not allowed:

- `Qwen fully replicates Gemma-style source tracing.`
- `Qwen has no evidence-sensitive or cross-modal mechanism.`

## 2026-05-26 Stage4-033 All-Layer Hidden Retest Update

Stage4-033 completed a full Qwen2.5-VL all-layer hidden residual sweep over language layers `0..27`.

Key result:

- Primary-selected gate: layer `14`, `restore`, `top_hidden_delta`, `answer_mask`.
- Primary usable prompt-runs: `96`.
- Strict usable prompt-runs: `101`.
- Strict confirms the same primary-selected gate.
- Decision: `qwen_hidden_route_supported`.

Strict confirmation metrics:

- target logit effect mean `1.7242`, CI low `1.2684`
- real minus shifted mean `1.1144`, CI low `0.7042`
- real minus shuffled mean `1.2183`, CI low `0.8010`
- correct minus wrong mean `0.3311`, CI low `0.0578`
- target rank effect mean `766.53`, CI low `151.23`

Interpretation:

`Qwen2.5-VL shows a replicated hidden-level evidence-region-sensitive answer-support route at layer 14.`

This changes the main failure diagnosis: prior layer-26-heavy PLT/source-tracing failures are no longer enough to argue route absence. The stronger current hypothesis is:

`Qwen has hidden-level route support, while current PLT/source-tracing localization has not yet captured a Gemma-style sparse feature route.`

Still not allowed:

- `Qwen fully replicates Gemma-style source tracing.`
- `Qwen has no evidence-sensitive or cross-modal mechanism.`

Next step:

Run PLT dense sweep and Adapter V4 around layer 14 and neighboring layers, rather than continuing to privilege layer 26.

## 2026-05-26 Stage4-034 Adapter V4 Plan Added

Stage4-034 is now the next Qwen-only PLT mainline test.

Goal:

`Test whether the confirmed Qwen layer-14 hidden route can be captured by an automatic Qwen-native PLT feature/source route.`

Key methodological guardrail:

- V4 uses Qwen layer-14 source-tracing, evidence sensitivity, target attribution, and zeroing damage.
- V4 does not use Gemma node ids or Gemma route maps.
- Strict confirmation uses the same primary-frozen scoring policy.

Possible verdicts:

- `qwen_adapter_v4_route_supported`
- `qwen_hidden_route_supported_plt_unresolved`
- `qwen_hidden_only_plt_failed`
- `qwen_not_gemma_style_under_v4_adapter`

Still not allowed:

- `Qwen fully replicates Gemma-style source tracing.`
- `Qwen has no evidence-sensitive or cross-modal mechanism.`

## 2026-05-26 Stage4-038 Qwen Native Route Update

Stage4-038 hidden-to-PLT mediation completed primary and strict full runs.

Result:

- `hidden_residual` reproduces the layer-14 evidence-to-answer route on primary and strict.
- `plt_reconstruction_error` almost reproduces the hidden residual target/rank/mask-specific effects on primary and strict.
- `plt_topk_reconstruction` does not pass the sparse feature mediation gate.

Current Qwen-native verdict:

`qwen_route_may_live_in_plt_error`

Paper wording:

`Qwen supports a Qwen-native hidden-level evidence-to-answer route, but current public Qwen-PLT does not localize it as a sparse topK feature route; the causal effect is largely preserved in PLT reconstruction error / non-feature residual.`

This is a mechanism-heterogeneity result. It is not equivalent to Gemma-style sparse source tracing replication, and it is not a negative result about Qwen having no evidence-sensitive mechanism.

## 2026-05-29 Stage4-060 Route-First Node-Level Update

Stage4-060 reversed the search order for Qwen-PLT:

- Instead of first selecting evidence-sensitive features, it first selected features satisfying route-first causal gates `2+3+4`.
- It then checked evidence sensitivity and correct-vs-wrong specificity as gates `1+5`.

Final decision:

```text
qwen_route_first_full_supported
```

Key counts:

- primary candidates: `23040`
- primary `route_first_234`: `2029`
- primary `route_first_gold`: `1491`
- primary `route_first_evidence_gold`: `1334`
- strict candidates: `1930`
- strict `route_first_234`: `1930`
- strict `route_first_gold`: `1447`
- strict `route_first_evidence_gold`: `1295`

Interpretation:

`Qwen-PLT contains strict-confirmed route-first feature nodes that satisfy answer-support, restore, real-mask specificity, evidence sensitivity, and correct-target specificity.`

Boundary:

This is node-level Qwen-native route-first support. It still is not Gemma-style automatic source-tracing graph replication.

## 2026-05-29 Stage4-066 Feature Route-Level Update

Stage4-066 grouped the Stage4-060 route-first nodes by `sample_id + prompt_name` and tested whole feature-route bundles with topK `4,8,16,32,64`.

Final decision:

```text
qwen_route_first_nodes_supported_route_unresolved
```

Key counts:

- primary grouped routes: `455`, raw rows: `18200`, unique samples: `50`
- strict grouped routes: `435`, raw rows: `17400`, unique samples: `48`
- strict missing fraction mean: `0.0`

Stable positives:

- route evidence specificity passes primary/strict for all topK.
- route correct > wrong passes primary/strict for all topK.
- clean route zeroing source > controls passes primary/strict for topK `32/64`.
- real restore > shifted/shuffled passes primary/strict for topK `8`.

Failed route-level closure:

- no single frozen topK passes all `1+2+3+4+5` aggregate gates.
- grouped `restore source > controls` remains weak.
- rank effects have positive means but positive fractions below the pre-registered `0.5` threshold.

Updated Qwen wording:

`Qwen2.5-VL-PLT has strict-confirmed Qwen-native route-first feature nodes, but grouped feature-route closure remains unresolved under the current public Qwen-PLT basis and grouped patch operator. The strongest Qwen route claim remains hidden-level / reconstruction-error-level support plus feature node-level support, not Gemma-style automatic sparse route replication.`
