# Stage6-019 Gemma Error-Heavy vs Source-Route Case Panel

Updated: 2026-06-02

## Question

Stage6-016 found that Gemma is `error-heavy` under hidden-to-PLT decomposition:

```text
hidden_residual is supported;
PLT topK reconstruction does not pass;
PLT reconstruction error nearly retains the hidden effect.
```

The reviewer-facing question is:

```text
If Gemma hidden effect is not captured by sparse PLT topK reconstruction,
why can we still say Gemma has a sparse source-tracing route?
```

## Key Distinction

These are two different lenses:

```text
local hidden-to-PLT topK reconstruction:
  asks whether a local topK decoder reconstruction of hidden delta carries the hidden patch effect.

graph-level source-tracing route:
  asks whether the traced route object is causally useful under evidence masks, controls, and correct-vs-wrong tests.
```

Therefore, topK reconstruction failing does not imply source-tracing graph failure. It means the full hidden residual effect is not locally explained by the selected topK reconstruction operator.

## Existing Numeric Baseline

Gemma Stage6-016:

```text
primary hidden residual:
n = 37
target_effect mean = 9.2736
CI low = 5.0034
positive_frac = 0.7297

strict hidden residual:
n = 33
target_effect mean = 10.4735
CI low = 5.7841
positive_frac = 0.7576

PLT topK reconstruction:
primary_topk_gate = null
strict_topk_gate = null
topk_retention_mean = -0.0040

PLT reconstruction error:
primary target_effect mean = 9.2196
primary CI low = 4.9628
strict target_effect mean = 10.6705
strict CI low = 6.0947
error_retention_mean = 0.9949
```

Gemma source-tracing baseline:

```text
Gemma source-tracing route passes graph-level gates:
evidence sensitivity
clean source > controls
masked restore source > controls
real mask restore > shifted/shuffled
correct > wrong
```

## Case Panel Design

Select 2-3 strong Gemma cases from Stage3 source-tracing and Stage6-016 decomposition artifacts.

For each case, report:

```text
sample_id
prompt_name
answer
mask_condition
hidden_residual target_effect
PLT topK reconstruction target_effect
PLT reconstruction error target_effect
source-tracing route source > controls
source-tracing route real > shifted/shuffled
source-tracing route correct > wrong
decoded behavior if available
```

## Case Panel Result

Generated artifacts:

```text
stage6_defensive_gemma_case_panel_full_defensive_v1.csv
stage6_defensive_gemma_case_panel_full_defensive_v1_decision.json
```

Analyzer status:

```text
status = case_panel_ready
raw_candidate_rows = 70
selected_rows = 6
```

Representative rows:

```text
sample = okvqa_val_2100995
prompt = B_direct
answer = wicker
hidden_effect = 44.375
PLT topK reconstruction, K=32 = -0.34375
PLT reconstruction error, K=32 = 45.625
error_over_hidden = 1.0282
topk_over_hidden = -0.0077
source node overlap Jaccard = 0.45
source edge overlap Jaccard = 0.371

sample = okvqa_val_2100995
prompt = D_visual_only
answer = wicker
hidden_effect = 37.75
PLT topK reconstruction, K=32 = 1.1875
PLT reconstruction error, K=32 = 37.75
error_over_hidden = 1.0
topk_over_hidden = 0.0315
source node overlap Jaccard = 0.45
source edge overlap Jaccard = 0.371

sample = okvqa_val_3955315
prompt = D_visual_only
answer = buddhism
hidden_effect = 26.0
PLT topK reconstruction, K=32 = 0.125
PLT reconstruction error, K=32 = 25.5
error_over_hidden = 0.9808
topk_over_hidden = 0.0048
```

These rows show the defensive pattern directly: local sparse topK reconstruction is near zero or even negative, while reconstruction error almost exactly recovers the hidden residual effect. This does not remove the source-tracing route object; it says that the hidden residual flow is not locally mediated by the tested topK reconstruction operator.

## Interpretation Rules

Supported explanation:

```text
hidden effect is error-heavy locally, but graph-level source route still passes route controls.
```

Mixed explanation:

```text
case-level source route is strong for some cases and weak for others;
write this as route visibility heterogeneity, not as route absence.
```

Blocked explanation:

```text
artifacts cannot be aligned by sample/prompt/token;
write as blocked_by_artifact_alignment.
```

## Paper-Ready Wording

Safe wording:

```text
The hidden-to-PLT decomposition and source-tracing graph answer different questions. In Gemma, the local topK reconstruction does not mediate most of the hidden residual effect, but the graph-level source-tracing route remains causally supported under evidence masks and matched controls. Thus Gemma's sparse route claim is a route-object claim, not a claim that sparse topK reconstruction explains the entire hidden residual flow.
```

Short Chinese wording:

```text
Gemma 的 hidden flow 不是被 local PLT topK reconstruction 解释掉的；它主要落在 reconstruction error / non-topK residual 里。但 Gemma 的 sparse source-tracing route 仍然是一个 graph-level route object：它在证据 mask、controls、correct-vs-wrong 这些 route-level 检验下成立。两者不是矛盾，而是两个不同镜头。
```
