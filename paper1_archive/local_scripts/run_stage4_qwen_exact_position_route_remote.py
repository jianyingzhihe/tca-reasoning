#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"E:\Bridging")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage4-038 exact-position route analysis entrypoint.")
    parser.add_argument("--min-exact-candidates", type=int, default=20)
    args = parser.parse_args()
    return subprocess.call(
        [
            sys.executable,
            str(ROOT / "scripts" / "local" / "analyze_stage4_qwen_native_route.py"),
            "--min-exact-candidates",
            str(args.min_exact_candidates),
        ],
        cwd=str(ROOT),
    )


if __name__ == "__main__":
    raise SystemExit(main())
