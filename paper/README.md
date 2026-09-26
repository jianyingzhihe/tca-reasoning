# Manuscript

`main_iclr2026.pdf` is the compiled manuscript. The LaTeX source, bibliography,
generated tables, and figures used by that PDF are included in this directory.

To rebuild from the `paper` directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main_iclr2026.tex
```

This directory contains the manuscript package, not the raw model checkpoints or
intervention tensors.
