#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).with_name("run_stage4_qwen_evidence_first_feature_discovery.py")
    runpy.run_path(str(target), run_name="__main__")
