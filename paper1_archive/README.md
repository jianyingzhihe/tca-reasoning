# Paper1 Archive

This directory archives the local orchestration code and paper-level experiment
records used for the evidence-to-answer route project.

## Contents

- `local_scripts/`: Windows-side runners, analyzers, manifest builders, and
  remote launch/fetch helpers used during Stages 2-6.
- `experiment_docs/`: compact Markdown reports plus decision/summary artifacts
  for Stages 2-6.

Large raw CSVs, graph `.pt` files, model caches, datasets, SSH connection
documents, and local environment files are intentionally excluded. They should
be managed through external storage, GitHub releases, or regenerated from the
archived scripts when needed.

## Paper1 Scope

Paper1 is the main evidence-to-answer route study. Its current high-level claim
is that multimodal models contain internal causal flow from visual evidence
regions to answer tokens, with Gemma exposing a clearer sparse PLT/source-route
view and Qwen exposing a more distributed hidden/local-feature view.
