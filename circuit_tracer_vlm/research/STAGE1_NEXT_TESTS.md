# Stage 1 Next Tests

## 0. Update (2026-05-06)

New progress since the earlier version of this file:

- coarse image-side controls have already been run:
  - `clean`
  - `no_image`
  - `wrong_image`
  - center-style `masked_image`
- matched random controls have already been run
- restoration pilots for support and suppressor nodes have already been run
- a new manual evidence-mask pilot has now been completed on 12 hand-annotated samples:
  - `answer_mask`
  - `relate_mask`
  - `union_mask`
  - `auto_control_mask`

Current best read:

- some traced signed routes are genuinely image-sensitive
- manual answer-region masking often weakens those routes more than a heuristic control mask
- this is promising, but still pilot-scale and not yet a final prompt-grounding claim

So the roadmap below should be read as historical context plus a queue for the next validation step.

## 1. Target-Mode Comparison

After the gold-target run finishes, compare it to the predicted-target run:

```bash
python scripts/research/compare_stage1_target_modes.py \
  --predicted-summary-dir outputs/phase_ab/ab_answer_aligned/stage1_random20_1024_seed42501_20260425_235607_stage1_summary_v2 \
  --gold-summary-dir outputs/phase_ab/ab_answer_aligned/<gold_run_tag>_stage1_summary \
  --out-dir outputs/phase_ab/ab_answer_aligned/<gold_run_tag>_vs_predicted_comparison
```

Key outputs:

- `target_mode_metric_comparison.csv`
- `target_mode_same_target_comparison.csv`
- `target_mode_filtered_overlap_comparison.csv`
- `target_mode_comparison.md`

## 2. Intervention Smoke

Once a Stage 1 run has completed and compare CSVs exist, run lightweight feature-zeroing interventions:

```bash
RUN_TAG_BASE=stage1_random20_1024_seed42501_20260425_235607 \
GENERIC_NODES_CSV=outputs/phase_ab/ab_answer_aligned/stage1_random20_1024_seed42501_20260425_235607_stage1_summary_v2/stage1_generic_nodes.csv \
TOP_FEATURES_PER_SAMPLE=2 \
MAX_SAMPLES=4 \
bash scripts/server/run_stage1_intervention_smoke.sh
```

The wrapper now also writes an aggregated summary directory:

- `outputs/phase_ab/ab_answer_aligned/${RUN_TAG_BASE}_intervention_smoke_summary/`
- `intervention_smoke_summary.csv`
- `intervention_smoke_strongest_rows.csv`
- `intervention_smoke_summary.md`

What this does:

- ranks samples by low overlap by default;
- picks top answer-aligned feature nodes per sample/run;
- skips generic scaffold features if `GENERIC_NODES_CSV` is provided;
- zeroes those features one by one;
- measures target logit / target probability change.

This is a smoke test, not the final causal claim. It helps answer:

- are the traced feature nodes actually connected to answer-token logit changes;
- do some candidate nodes produce larger target-logit drops than others;
- which samples are worth promoting into a more expensive patching experiment.

If you already have per-bucket smoke CSVs and just want to rebuild the summary:

```bash
python scripts/research/summarize_intervention_smoke.py \
  --inputs \
    outputs/phase_ab/ab_answer_aligned/<run_tag>_A0_B0/intervention_smoke_A0_B0.csv \
    outputs/phase_ab/ab_answer_aligned/<run_tag>_A0_B1/intervention_smoke_A0_B1.csv \
    outputs/phase_ab/ab_answer_aligned/<run_tag>_A1_B0/intervention_smoke_A1_B0.csv \
    outputs/phase_ab/ab_answer_aligned/<run_tag>_A1_B1/intervention_smoke_A1_B1.csv \
  --out-dir outputs/phase_ab/ab_answer_aligned/<run_tag>_intervention_smoke_summary
```

## 3. Focused Intervention Planning

After the gold-target run, build a focused shortlist of same-target, low-overlap, feature-down /
token-up samples and export exact per-bucket sample lists for the smoke run:

```bash
python scripts/research/plan_stage1_interventions.py \
  --run-tag-base stage1_goldtarget20_1024_seed42501_<timestamp> \
  --summary-dir outputs/phase_ab/ab_answer_aligned/stage1_goldtarget20_1024_seed42501_<timestamp>_stage1_summary \
  --generic-nodes-csv outputs/phase_ab/ab_answer_aligned/stage1_goldtarget20_1024_seed42501_<timestamp>_stage1_summary/stage1_generic_nodes.csv \
  --out-dir outputs/phase_ab/ab_answer_aligned/stage1_goldtarget20_1024_seed42501_<timestamp>_intervention_plan \
  --same-target-only \
  --max-samples-per-bucket 2 \
  --top-features-per-run 2
```

This writes:

- `candidate_samples.csv`
- `candidate_features.csv`
- `bucket_sample_ids/<bucket>.csv`
- `intervention_plan.md`

Then run the smoke test only on those selected samples:

```bash
RUN_TAG_BASE=stage1_goldtarget20_1024_seed42501_<timestamp> \
SAMPLE_IDS_DIR=outputs/phase_ab/ab_answer_aligned/stage1_goldtarget20_1024_seed42501_<timestamp>_intervention_plan/bucket_sample_ids \
GENERIC_NODES_CSV=outputs/phase_ab/ab_answer_aligned/stage1_goldtarget20_1024_seed42501_<timestamp>_stage1_summary/stage1_generic_nodes.csv \
REQUIRE_SAME_TARGET=1 \
TOP_FEATURES_PER_SAMPLE=2 \
MAX_SAMPLES=999 \
bash scripts/server/run_stage1_intervention_smoke.sh
```

Notes:

- when `SAMPLE_IDS_DIR` is provided, `MAX_SAMPLES` should be set high enough that it does not
  silently truncate the per-bucket CSV lists;
- the planner now filters out samples whose `answer_aligned_meta_a.csv` or
  `answer_aligned_meta_b.csv` rows are missing `target_token_id`, so the resulting
  shortlist should not waste overnight slots on known-invalid samples.

## 4. Current Status (2026-05-05)

Completed stages so far:

1. predicted-target random20 summary:
   - stable feature-down / token-up composition shift
   - only `31/80` same-target cases
2. gold-target control summary:
   - same-target rate improved to `73/80`
   - overlap confound reduced substantially
   - feature-down / token-up shift survived control
3. focused + overnight intervention smoke:
   - pipeline runs end to end after multimodal batch-forwarding fix
   - `88` valid single-feature interventions completed in the overnight run
   - `64.8%` of interventions produced `delta_target_logit < 0`
   - overall mean `delta_target_logit` stayed slightly positive (`+0.073`)

Interpretation:

- traced feature nodes are not all equivalent;
- many high-path-mass feature nodes do causally support the answer token;
- a substantial minority appear suppressive or competitive, so path mass alone is not a signed
  causal score.

## 5. Recommended Next Step

The next best experiment is not another large Stage 1 attribution sweep.

The next best step is a held-out manual evidence-mask validation pass, not another broad discovery sweep.

That should answer three narrower questions:

1. whether manually marked answer evidence weakens traced signed routes more than matched control masks on held-out samples;
2. whether this pattern differs across `A0_B1` / `A1_B0` / `A1_B1`;
3. whether support and suppressor routes show different sensitivity to `answer_mask` vs `relate_mask`.

## 6. Refocused Roadmap

Based on the current results, the Stage 1 roadmap should now narrow from
"show that A/B graphs differ" to "explain what signed causal mechanism produces the
difference."

### Phase A: Clean Signed-Intervention Pass

Goal:

- estimate the balance of supportive vs suppressive traced features under clean
  same-target conditions

Required settings:

- same-target only
- generic-feature filtered
- meta-complete samples only
- moderate breadth rather than another full Stage 1 sweep

Recommended scale:

- `8-12` samples per bucket
- `4` features per sample per run
- `run=both`

Main outputs to track:

- fraction with `delta_target_logit < 0`
- strongest negative-`delta_target_logit` nodes
- per-bucket supportive vs suppressive mix

### Phase B: Required Controls

Before making stronger mechanistic claims, add these controls:

1. random matched-node control:
   - sample features matched on layer / position and rough scale, then compare their
     ablation effect against traced features
2. non-target token control:
   - test whether ablations are target-specific or just produce broad logit movement
3. prompt paraphrase control:
   - use several Prompt A and Prompt B paraphrases to separate prompt style from
     template-specific artifacts
4. prompt length / formatting control:
   - distinguish style effects from answer-format or prompt-length effects
5. image-side control:
   - no-image, mismatched-image, or degraded-image variants to test whether candidate
     support nodes depend on visual evidence
6. evidence-aware region masking:
   - held-out human-marked answer regions
   - either human-marked irrelevant regions or stronger automatic control masks
   - compare `answer_mask`, `relate_mask`, `union_mask`, and control conditions

### Phase C: From Zeroing To Patching

Single-feature zeroing is a good smoke test, but it is not yet a full circuit claim.

The next causal upgrade should include:

- A->B activation patching
- B->A activation patching
- corrupt-and-restore style tests
- multi-node ablations
- small-circuit sufficiency tests

Goal:

- move from "this node matters" to "this small set of nodes can reproduce or explain
  the A/B behavior difference"
- especially under evidence-aware masked conditions rather than only coarse corruptions

### Phase D: Feature Semantics

The current analysis is strong at the graph/composition level, but still weak at the
semantic level.

We should assign human-interpretable labels to strong support and suppressor features
using:

- top activating examples
- token-position analysis
- image occlusion / corruption responses
- question perturbations
- limited LLM-assisted labeling followed by manual checks

The target taxonomy should at least distinguish:

- visual evidence features
- text / question-type features
- answer-prior features
- prompt-format / instruction-following features
- suppressor / competitor features

## 7. Updated Main Question

At the beginning of Stage 1, the key question was:

- do Prompt A and Prompt B produce different answer-target circuits?

That is now mostly established.

The updated question is:

- which traced nodes are supportive vs suppressive,
- whether those signed node groups differ across buckets,
- and whether they correspond to visual evidence, answer priors, or prompt-format effects.

## 8. Multimodal Refocus

To avoid making this look like a text-only circuit workflow copied onto a VLM,
the next phase should explicitly center a multimodal question:

- does prompt style change **modality routing** inside the VLM?

That means moving from:

- prompt A/B changes attribution graphs

to:

- prompt A/B changes the causal allocation of answer generation across
  visual evidence, language prior, and cross-modal binding routes.

### Core Reframing

The main Stage 1 follow-up should now be framed as:

- **Prompt-Induced Modality Routing in Vision-Language Models**

Concretely:

- under the same image, question, and answer target,
- does Prompt A increase or decrease reliance on visual evidence nodes,
- language-prior nodes,
- or image-question binding nodes?

### Modality-Counterfactual Conditions

The next intervention stage should include more than clean-image ablations.

For a small, carefully selected set of samples, add:

1. clean image + original question
2. no-image / blank-image + original question
3. wrong-image + original question
4. object-masked image + original question

Optional later extension:

5. image-question mismatch or object-swapped question

These conditions should be used for both tracing and intervention analysis.

### Node Taxonomy We Need

We should no longer stop at feature/token/error categories.

Strong candidate nodes should be classified into:

- visual-evidence features
- text-question features
- answer-prior features
- cross-modal binding features
- prompt-format / instruction-following features
- suppressor / competitor features

### Multimodal-Specific Metrics

The current `delta_target_logit` metric remains useful, but it should be paired with
modality-sensitive metrics such as:

1. visual dependence score:
   - target-logit drop from clean image to no-image / wrong-image / masked-image
2. visual restoration effect:
   - effect of restoring clean visual nodes under a corrupted-image condition
3. language prior leakage:
   - target strength under no-image or mismatched-image settings
4. cross-modal binding sensitivity:
   - difference between matched and mismatched image-question conditions

### Bucket-Level Multimodal Questions

The bucket split is especially useful once the project is framed as modality routing:

- `A1_B1`:
  - both prompts succeed; do they rely on different visual/prior/binding routes?
- `A1_B0`:
  - does Prompt A succeed by strengthening visual evidence or by suppressing a wrong prior?
- `A0_B1`:
  - does Prompt B succeed by preserving a more direct visual route?
- `A0_B0`:
  - when both fail, do they collapse toward answer-prior or format-driven routes?

### Immediate Experimental Implication

The next small, focused study should ideally be:

- same-target only
- selected buckets: `A1_B1`, `A1_B0`, `A0_B1`
- modality counterfactual conditions
- signed interventions on top support and suppressor candidates

This is the cleanest way to make the project specifically about VLM mechanism,
not just prompt-conditioned graph overlap.

## 9. Modality Counterfactual Pilot Status

We now have a first small pilot for the multimodal refocus.

Remote result CSV:

- `outputs/phase_ab/ab_answer_aligned/stage1_goldtarget20_1024_seed42501_20260426_212432_modality_pilot/pilot_clean_noimage_wrongimage.csv`

Conditions used:

1. `clean`
2. `no_image`
3. `wrong_image`

Selection rule:

- reuse completed clean smoke CSVs
- choose strongest clean support node and strongest clean suppressor node per selected sample/run

Current pilot scope:

- buckets: `A0_B1`, `A1_B0`, `A1_B1`
- `45` condition rows
- `11` support-node condition triplets
- `4` suppressor-node condition triplets

Current pilot result:

- support clean mean `delta_target_logit = -0.9865`
- support `no_image` mean `delta_target_logit = -0.6222`
- support `wrong_image` mean `delta_target_logit = -0.6790`
- suppressor clean mean `delta_target_logit = +2.3281`
- suppressor `no_image` mean `delta_target_logit = +2.3125`
- suppressor `wrong_image` mean `delta_target_logit = +1.8906`

Interpretation:

- support-node effects weaken noticeably when image evidence is removed or replaced
- suppressor-node effects remain positive overall
- this is the first direct evidence that some traced support nodes are image-sensitive,
  while at least part of the suppressor set may be less tied to the clean image
- the expanded `A0_B1` pass also produced a clean support-to-nonsupport sign flip, which makes
  the modality-routing story less likely to be only an `A1_B1` artifact

### First Matched-Control Check

We also now have a first matched-control pass on the expanded pilot selection.

Remote result CSV:

- `outputs/phase_ab/ab_answer_aligned/stage1_goldtarget20_1024_seed42501_20260426_212432_modality_pilot/pilot_A0B1_A1B0_A1B1_matched_control_v2.csv`

Summary artifact:

- `modality_matched_control_summary.md`

What this control does:

- same sample, run, and condition
- exclude generic features
- match to a nearby alternative feature, preferring same layer and same position
- do not reuse the same control node within the same sample/run

Current paired summary:

- clean support: source mean `-0.9477`, control mean `-0.2688`
- no-image support: source mean `-0.5156`, control mean `-0.2719`
- wrong-image support: source mean `-0.6219`, control mean `-0.1328`
- clean suppressor: source mean `+2.3958`, control mean `+0.0208`

Interpretation:

- selected source nodes are usually stronger and more sign-consistent than the current matched controls
- that is good news for the causal story
- but this is still an initial nearest-match control, not a repeated matched-random benchmark

So the next control upgrade should be:

- repeated matched-random controls per source node
- with stricter matching strata where feasible

### Repeated Matched-Random Control Status

We now also have a repeated matched-random control pass.

Remote result CSV:

- `outputs/phase_ab/ab_answer_aligned/stage1_goldtarget20_1024_seed42501_20260426_212432_modality_pilot/pilot_A0B1_A1B0_A1B1_matched_random4.csv`

Settings:

- same expanded pilot source set
- `match_mode=random`
- up to `4` controls per source node

Per-source summary:

- clean support: source mean `-0.9477`, control mean `-0.3516`
- clean suppressor: source mean `+2.3281`, control mean `-0.0742`
- wrong-image support: source mean `-0.6219`, control mean `-0.1828`
- wrong-image suppressor: source mean `+1.8906`, control mean `-0.0898`

Interpretation:

- the selected source nodes still look stronger and more sign-stable than a repeated matched-random control set
- that materially strengthens the causal-selection story
- the next control improvement is no longer "add any control at all"
- it is "tighten the control pool further," for example by activation- or path-mass-stratified random sampling

This does **not** yet license a strong claim that Prompt A is less visually grounded than Prompt B.
But it does justify the current direction:

- keep Stage 1 centered on signed nodes plus modality counterfactuals
- expand the pilot before spending more budget on larger tracing sweeps

## 10. Masked-Image Update

We now also have a broader `masked_image` pass on top of the broad modality pilot.

Remote summary:

- `outputs/phase_ab/ab_answer_aligned/stage1_goldtarget20_1024_seed42501_20260426_212432_modality_pilot_summary_broad_masked/modality_pilot_summary.md`

Main aggregate values:

- support clean mean: `-0.8166`
- support masked-image mean: `-0.6650`
- support no-image mean: `-0.5675`
- support wrong-image mean: `-0.4306`
- suppressor clean mean: `+2.0759`
- suppressor masked-image mean: `+1.6161`
- suppressor no-image mean: `+1.1987`
- suppressor wrong-image mean: `+0.6853`

Interpretation:

- `masked_image` sits between `clean` and the stronger corruption settings
- local occlusion weakens traced effects without collapsing them as much as a fully wrong image
- this makes the Stage 1 story more specifically about graded visual evidence, not only
  image-present versus image-absent differences

## 11. Immediate Next Experiment: Restoration Smoke

The next best test is a small restoration experiment rather than another bigger ablation sweep.

Goal:

- take support nodes whose effect weakened under `masked_image`
- restore the clean feature value into the masked-image condition
- measure whether the target logit rises again

This is the cleanest next causal upgrade because it tests:

- not only whether a node matters when removed
- but whether reintroducing its clean state under degraded visual input recovers target support

Suggested script:

```bash
python scripts/research/run_masked_restoration_smoke.py \
  --run-tag-base stage1_goldtarget20_1024_seed42501_20260426_212432 \
  --pilot-csv outputs/phase_ab/ab_answer_aligned/stage1_goldtarget20_1024_seed42501_20260426_212432_modality_pilot/pilot_broad_A0B1_A1B0_A1B1_clean_noimage_wrongimage_masked.csv \
  --corrupt-condition masked_image \
  --node-role support \
  --max-nodes 8 \
  --max-nodes-per-sample-run 1 \
  --out-csv outputs/phase_ab/ab_answer_aligned/stage1_goldtarget20_1024_seed42501_20260426_212432_restoration_smoke_masked.csv
```

Primary readout:

- `restore_minus_corrupt_target_logit`

Helpful secondary readouts:

- `restore_minus_zero_target_logit`
- `restore_fraction_of_clean_gap`
- change in top-1 token after restoration

## 12. First Restoration Smoke Status

We now have a first masked-image restoration smoke result.

Remote outputs:

- exploratory:
  - `outputs/phase_ab/ab_answer_aligned/stage1_goldtarget20_1024_seed42501_20260426_212432_restoration_smoke_masked_top8.csv`
- gap-filtered:
  - `outputs/phase_ab/ab_answer_aligned/stage1_goldtarget20_1024_seed42501_20260426_212432_restoration_smoke_masked_gapfiltered_top6.csv`

Key result:

- exploratory top-8 mean `restore_minus_corrupt_target_logit = +0.5234`
- gap-filtered top-6 mean `restore_minus_corrupt_target_logit = +0.6563`
- gap-filtered top-6 had positive restoration in `5/6` cases

Most important methodological lesson:

- selecting nodes only by "masked ablation weakened relative to clean" is not enough
- restoration works much better when support nodes also satisfy
  `clean_feature_value > masked_feature_value`

So the next restoration pass should use:

- weakened-under-masking support nodes
- positive clean-minus-masked activation gap
- one node per sample/run

### Pilot Summarizer

To summarize a pilot CSV into aggregate tables and per-node clean-vs-corrupt comparisons:

```bash
python scripts/research/summarize_modality_counterfactual_pilot.py \
  --input outputs/phase_ab/ab_answer_aligned/<run_tag>_modality_pilot/<pilot_csv>.csv \
  --out-dir outputs/phase_ab/ab_answer_aligned/<run_tag>_modality_pilot_summary
```

Outputs:

- `modality_pilot_summary.csv`
- `modality_pilot_per_node.csv`
- `modality_pilot_sign_flips.csv`
- `modality_pilot_summary.md`

### Matched-Control Summarizer

To compare source pilot nodes against matched controls:

```bash
python scripts/research/summarize_modality_matched_control.py \
  --pilot-csv outputs/phase_ab/ab_answer_aligned/<run_tag>_modality_pilot/<pilot_csv>.csv \
  --control-csv outputs/phase_ab/ab_answer_aligned/<run_tag>_modality_pilot/<matched_control_csv>.csv \
  --out-dir outputs/phase_ab/ab_answer_aligned/<run_tag>_modality_pilot_control_summary
```

Outputs:

- `modality_matched_control_pairs.csv`
- `modality_matched_control_summary.csv`
- `modality_matched_control_per_source.csv`
- `modality_matched_control_per_source_summary.csv`
- `modality_matched_control_summary.md`
