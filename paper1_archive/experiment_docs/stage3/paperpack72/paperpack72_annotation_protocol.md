# Paperpack72 Annotation Protocol

This file mirrors `017_paperpack72_annotation_protocol.md` and is colocated with the annotation CSV artifacts.

Use `single_pass1_localized` for the first complete annotation pass. Use `single_pass2_localized_reliability` later, after a delay, for self-consistency / reliability relabeling.

Required labels per sample:

```text
answer_mask
relate_mask
final_reasoning_operation
image_dependence_final
ambiguity / exclusion notes
```

Adjudication outputs:

```text
adjudicated_answer_mask
adjudicated_relate_mask
adjudicated_union_mask
shifted_mask
shuffled_mask
random16 masks
paperpack72_manifest.csv
paperpack72_exclusion_manifest.csv
```

Formal experiments must not start until `paperpack72_manifest.csv` exists and every included row has adjudicated masks and agreement metrics.
