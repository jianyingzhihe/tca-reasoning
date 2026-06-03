#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(r"E:\Bridging")
SCRIPT = ROOT / "vlm-circuit-tracing" / "circuit_tracer_vlm" / "scripts" / "research" / "analyze_stage4_qwen_source_tracing.py"


if __name__ == "__main__":
    if "--cross-dir" not in sys.argv:
        sys.argv.extend(["--cross-dir", str(ROOT / "doc" / "experiments" / "stage4" / "cross_model")])
    runpy.run_path(str(SCRIPT), run_name="__main__")

