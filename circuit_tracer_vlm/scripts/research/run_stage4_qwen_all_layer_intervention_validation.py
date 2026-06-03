#!/usr/bin/env python3
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--validation-kind", choices=["zeroing", "grouped_restore"], default="zeroing")
    known, rest = parser.parse_known_args()
    script = "run_stage4_qwen_causal_cutter_validation.py" if known.validation_kind == "zeroing" else "run_stage4_qwen_evidence_first_intervention.py"
    sys.argv = [script, *rest]
    runpy.run_path(str(Path(__file__).with_name(script)), run_name="__main__")


if __name__ == "__main__":
    main()
