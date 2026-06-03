# 023 Qwen2.5-VL-CLT Paperpack Results

## Purpose

After the PLT-only verdict, run Qwen2.5-VL-CLT on the same paperpack72 primary/strict manifests to test whether the Qwen evidence-region-sensitive feature/source-control signal is robust to transcoder type.

This is a CLT auxiliary robustness line. It does not replace the PLT-first claim and does not establish Gemma-style source tracing.

## Inputs

```text
base: Qwen/Qwen2.5-VL-7B-Instruct
CLT: KokosDev/qwen2p5vl-7b-clt
layer: 26
packs: paperpack72_primary, paperpack72_strict_sensitivity
prompts: B_direct, D_visual_only
conditions: union_mask feature bridge; answer_mask/union_mask source-control
controls: activation_matched, drop_matched, mask_insensitive, random_active, mask_shuffled, wrong_target
```

## Outputs

```text
cross_model/stage3_qwen2p5vl_clt_feature_union_primary_full.csv/json
cross_model/stage3_qwen2p5vl_clt_source_control_primary_full.csv/json
cross_model/stage3_qwen2p5vl_clt_primary_full_*summary.csv
cross_model/stage3_qwen2p5vl_clt_primary_full_decision.json
cross_model/stage3_qwen2p5vl_clt_feature_union_strict_full.csv/json
cross_model/stage3_qwen2p5vl_clt_source_control_strict_full.csv/json
cross_model/stage3_qwen2p5vl_clt_strict_full_*summary.csv
cross_model/stage3_qwen2p5vl_clt_strict_full_decision.json
cross_model/stage3_qwen_plt_vs_clt_primary_*.csv/json
cross_model/stage3_qwen_plt_vs_clt_strict_*.csv/json
```

## Method

The runner reuses the exact paperpack sample/prompt/mask schema used by Qwen2.5-VL-PLT. It runs the Stage 2O attribution-weighted feature bridge and approximate source-control probe with the CLT asset.

Feature bridge compares `evidence_attribution_topk` against matched feature controls. Source-control compares source-like features against matched controls, and also evaluates real mask vs shuffled mask and correct target vs wrong target.

## Results

Primary72:

```text
status: qwen_clt_robustness_support_primary
feature prompt-runs: 144
feature rows: 4320
source-control prompt-runs: 144
source-control rows: 2288
source usable pairs: 286
primary feature specificity positive: 8 / 8
source-control positive: 4 / 4
real-vs-shuffled positive: 4 / 8
```

Strict72 sensitivity:

```text
status: qwen_clt_robustness_support_strict
feature prompt-runs: 144
feature rows: 4320
source-control prompt-runs: 144
source-control rows: 2288
source usable pairs: 286
primary feature specificity positive: 8 / 8
source-control positive: 4 / 4
real-vs-shuffled positive: 4 / 8
```

PLT-vs-CLT paired comparison:

```text
primary status: qwen_clt_partial_robustness
strict status: qwen_clt_partial_robustness
primary paired feature rows: 4320
strict paired feature rows: 4320
primary paired source rows: 726
strict paired source rows: 723
primary sign agreement mean: 0.5707
strict sign agreement mean: 0.5766
```

## Expected vs Actual

Expected: Qwen-CLT may be weaker than Qwen-PLT because CLT and PLT target different residual/transcoder alignments.

Actual: Qwen-CLT is not blocked and is positive on primary and strict packs. The PLT-vs-CLT comparison is same-direction enough to support robustness, but not identical enough to claim representation invariance.

## Conclusion

```text
Qwen2.5-VL-CLT provides auxiliary robustness support for the Qwen evidence-region-sensitive feature/source-control signal on paperpack72.
```

Allowed wording:

```text
Qwen evidence is robust across PLT and CLT at the auxiliary feature/source-control level, with representation-dependent effect size and imperfect paired agreement.
```

Not allowed:

```text
Qwen-CLT fully replicates Gemma-style source tracing.
CLT success upgrades the PLT-only verdict by itself.
Decoded generation-level causal bridge is established here.
```
