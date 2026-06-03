# 025 Stage3 Paper-Grade Final Verdict

## Purpose

Summarize the Stage3 PLT-first / CLT-second evidence after completing the paperpack72 experiments.

## Current Status

```text
paperpack72 annotated pool: complete
Gemma3-PLT primary full: passed
Gemma3-PLT strict graph/compare: passed
Gemma3-PLT strict lightweight intervention repair: passed on B4
Qwen2.5-VL-PLT primary/strict: passed approximate feature/source-control
Qwen2.5-VL-CLT primary/strict: passed auxiliary robustness
LLaVA-CLT primary/strict: diagnostic not_supported for feature/source route specificity
Qwen35-PLT: feasibility remains high-risk/blocked for VLM mainline
```

## Main Verdict

The strongest Stage3 claim remains PLT-first:

```text
Gemma3 and Qwen2.5-VL show PLT-aligned evidence-region-sensitive answer-support evidence on an independently annotated paperpack72.
```

The Qwen CLT line strengthens robustness:

```text
Qwen2.5-VL-CLT reproduces the auxiliary feature/source-control signal on primary and strict paperpack72, but PLT-vs-CLT paired agreement is partial rather than identical.
```

The LLaVA CLT line is a heterogeneity diagnostic:

```text
LLaVA-CLT runs successfully and has positive aggregate source-control direction, but feature specificity and real-vs-shuffled controls are too weak for feature/source route support.
```

## Evidence Matrix

| Model | Asset | Paperpack Status | Claim Level |
|---|---|---|---|
| Gemma3 | PLT | primary full passed; strict graph/compare passed; strict B4 intervention repair passed | full Gemma mainline support on primary; strict sensitivity for graph/compare |
| Qwen2.5-VL | PLT | primary/strict passed | PLT-aligned approximate feature/source-control support |
| Qwen2.5-VL | CLT | primary/strict passed | CLT auxiliary robustness with representation-dependent effect size |
| LLaVA-1.5 | CLT | primary/strict diagnostic not_supported | weak CLT feature/source localization; not a mechanism absence claim |
| Qwen35 | PLT | high-risk / not VLM-mainline-ready | blocked/feasibility only |

## Claim Boundaries

Allowed:

```text
The phenomenon is not Gemma-only: Gemma3 and Qwen2.5-VL show PLT-aligned paperpack support, and Qwen-CLT provides auxiliary robustness evidence.
```

Allowed:

```text
CLT results indicate representation dependence: Qwen remains positive across PLT/CLT, while LLaVA-CLT does not establish specific feature/source localization.
```

Not allowed:

```text
All models fully replicate Gemma-style source tracing.
Qwen fully replicates Gemma source-control route without a ReplacementModel/source-tracing adapter.
LLaVA has no cross-modal mechanism.
D_visual_only is better than B_direct.
Decoded generation-level causal bridge is established by Stage3.
```

## Remaining Gaps

```text
1. Qwen still lacks a true Gemma-style ReplacementModel/source-tracing adapter.
2. Gemma strict full intervention is not complete, although lightweight B4 repair passed.
3. LLaVA-CLT feature/source localization is weak under current layer/CLT controls.
4. Decoded generation bridge remains secondary and not established as a full causal generation claim.
```

## Paper Wording

Recommended conservative wording:

```text
On an independently annotated localized VQA paperpack, Gemma3-PLT and Qwen2.5-VL-PLT show evidence-region-sensitive answer-support evidence. Qwen2.5-VL-CLT provides auxiliary robustness across transcoder type, whereas LLaVA-CLT shows weaker, non-specific feature localization despite runnable source-control diagnostics. These results support cross-model evidence for the phenomenon, but do not establish full Gemma-style source-route replication in every model.
```
