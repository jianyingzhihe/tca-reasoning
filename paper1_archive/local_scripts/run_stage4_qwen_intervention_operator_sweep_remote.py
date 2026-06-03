#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"E:\Bridging")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage4-038 intervention operator sweep remote entrypoint.")
    parser.add_argument("--pack", choices=["primary", "strict"], default="primary")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--tag", default="operator_sweep")
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "local" / "run_stage4_qwen_hidden_to_plt_mediation_remote.py"),
        "--pack",
        args.pack,
        "--mode",
        args.mode,
        "--tag",
        args.tag,
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    if args.detach:
        cmd.append("--detach")
    if args.fetch_only:
        cmd.append("--fetch-only")
    if args.status:
        cmd.append("--status")
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
