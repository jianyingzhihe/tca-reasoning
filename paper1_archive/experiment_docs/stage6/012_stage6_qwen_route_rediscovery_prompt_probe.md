# Stage6-012: Qwen Route Rediscovery Prompt Probe

## Goal

Give Qwen a Gemma-like route identity question without pretending Qwen has Gemma edge graphs: for each prompt/text condition, revalidate the broad visual feature pool, re-rank route-first candidates, and compare the top route set with `B_direct + original`.

## Candidate Pool

- Source: `stage4_qwen_route_first_primary_manifest.csv`.
- Samples: the 12 Stage6 prompt/text samples.
- Layers: `L10-L17`.
- Candidate restriction: visual source positions only.
- Each candidate is expanded across 3 question variants and 4 prompt families.

## Main Metrics

- `route_identity_score`: positive sum of route-first causal components.
- `topK_node_overlap@4/8/16/32`: overlap of baseline candidate ids.
- `feature_id_overlap`: overlap by feature id only.
- `layer_pos_feature_overlap`: stricter overlap by layer, position, and feature id.
- `pos234_frac`: fraction satisfying restore source > controls and real restore > shifted/shuffled.

## Interpretation

Low overlap means Qwen route identity changes across prompt/text under this Qwen-native feature route proxy. It is not a Gemma edge-graph failure.
