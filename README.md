# Vision-Language Circuit Tracing Code

This repository contains source code for attribution graphs and internal
interventions in vision-language models. It does not include manuscripts,
model weights, or experiment outputs.

## Repository layout

- `circuit_tracer_vlm/`: Python package, command-line interface, demos, tests,
  and research scripts.
- `third_party/TransformerLens/`: the TransformerLens code used by this project.

## Quick start

Python 3.10 or newer is required. From the repository root:

```sh
cd circuit_tracer_vlm
pip install -e .
circuit-tracer --help
```

See `circuit_tracer_vlm/README.md` for usage details and demos. Licenses and
upstream attribution are retained with the corresponding code.
