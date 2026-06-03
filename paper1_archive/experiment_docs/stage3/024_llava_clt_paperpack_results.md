# 024 LLaVA-CLT Paperpack Diagnostic Results

## Purpose

Run LLaVA-1.5-CLT on paperpack72 as an auxiliary diagnostic line. The goal is to test whether the CLT feature/source-control localization seen in Qwen is also stable in a LLaVA-family VLM, and to diagnose heterogeneity without treating weak results as proof that no mechanism exists.

## Inputs

```text
base: llava-hf/llava-1.5-7b-hf
local base path used: /root/autodl-tmp/tca-reasoning/data/modelscope_cache/swift/llava-1___5-7b-hf
CLT: KokosDev/llava15-7b-clt
layer: 15
packs: paperpack72_primary, paperpack72_strict_sensitivity
prompts: B_direct, D_visual_only
conditions: union_mask feature bridge; answer_mask/union_mask source-control
controls: activation_matched, drop_matched, mask_insensitive, random_active, mask_shuffled, wrong_target
```

## Outputs

```text
cross_model/stage3_llava15_clt_feature_union_primary_full.csv/json
cross_model/stage3_llava15_clt_source_control_primary_full.csv/json
cross_model/stage3_llava15_clt_primary_full_*summary.csv
cross_model/stage3_llava15_clt_primary_full_decision.json
cross_model/stage3_llava15_clt_feature_union_strict_full.csv/json
cross_model/stage3_llava15_clt_source_control_strict_full.csv/json
cross_model/stage3_llava15_clt_strict_full_*summary.csv
cross_model/stage3_llava15_clt_strict_full_decision.json
```

## Method

The same Stage 2O feature bridge and approximate source-control probe were run with the LLaVA CLT asset. LLaVA is not part of the PLT-aligned mainline because no matching public VLM PLT is available here.

## Results

Primary72:

```text
status: not_supported
feature prompt-runs: 144
feature rows: 4320
source-control prompt-runs: 139
source-control rows: 2112
source usable pairs: 264
primary feature specificity positive: 3 / 8
source-control positive: 4 / 4
real-vs-shuffled positive: 2 / 8
```

Strict72 sensitivity:

```text
status: not_supported
feature prompt-runs: 144
feature rows: 4320
source-control prompt-runs: 138
source-control rows: 2096
source usable pairs: 262
primary feature specificity positive: 3 / 8
source-control positive: 4 / 4
real-vs-shuffled positive: 3 / 8
```

## Expected vs Actual

Expected: LLaVA could show smaller or less stable CLT feature localization than Qwen because the asset format, model architecture, and prior Stage 2 hidden-state results were more heterogeneous.

Actual: The LLaVA runner is not blocked, and source-control aggregate direction is positive. However, the feature specificity and real-vs-shuffled controls are not strong enough on either primary or strict packs. The conservative decision is therefore `not_supported` for feature/source route replication.

## Conclusion

Allowed wording:

```text
LLaVA-CLT runs on paperpack72 and shows some source-control directionality, but current CLT feature/source localization is not specific enough under matched and shuffled controls.
```

Also allowed:

```text
LLaVA remains an auxiliary heterogeneity/diagnostic line: prior hidden-state bridge evidence may coexist with weak CLT feature localization.
```

Not allowed:

```text
LLaVA has no cross-modal mechanism.
LLaVA fully replicates the Gemma/Qwen feature/source-control route.
The negative CLT diagnostic overturns the PLT-aligned mainline.
```
