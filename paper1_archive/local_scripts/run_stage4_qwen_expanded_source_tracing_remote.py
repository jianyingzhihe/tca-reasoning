#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"E:\Bridging")
BASE_RUNNER = ROOT / "scripts" / "local" / "run_stage4_qwen_source_tracing_remote.py"


def _run(args: list[str]) -> None:
    command = [sys.executable, str(BASE_RUNNER), *args]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage4-016 expanded Qwen source-tracing suite.")
    parser.add_argument("--suite", choices=["smoke", "overnight"], default="smoke")
    parser.add_argument("--timeout-seconds", type=int, default=86400)
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Run only L26 primary/strict top32, skipping layer/position sensitivity.",
    )
    args = parser.parse_args()

    common = [
        "--candidate-pool-size",
        "8192",
        "--compare-topk-per-node",
        "8",
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    if args.suite == "smoke":
        _run(
            [
                "--pack",
                "primary",
                "--mode",
                "smoke",
                "--tag",
                "expanded_v2_L26_top32",
                "--layer",
                "26",
                "--max-feature-nodes",
                "128",
                "--top-features-per-sample",
                "32",
                "--position-filter",
                "visual_answer",
                "--cleanup-run-root",
                *common,
            ]
        )
        return 0

    # Confirmatory route: full primary discovery and strict confirmation use identical settings.
    for pack in ["primary", "strict"]:
        _run(
            [
                "--pack",
                pack,
                "--mode",
                "full",
                "--tag",
                "expanded_v2_L26_top32",
                "--layer",
                "26",
                "--max-feature-nodes",
                "128",
                "--top-features-per-sample",
                "32",
                "--position-filter",
                "visual_answer",
                "--cleanup-run-root",
                *common,
            ]
        )

    if args.skip_sensitivity:
        return 0

    # Sensitivity jobs are diagnostic only; they must not be used for main candidate selection.
    for layer in ["22", "24"]:
        _run(
            [
                "--pack",
                "primary",
                "--mode",
                "full",
                "--tag",
                f"expanded_v2_L{layer}_top16",
                "--layer",
                layer,
                "--max-feature-nodes",
                "128",
                "--top-features-per-sample",
                "16",
                "--position-filter",
                "visual_answer",
                "--cleanup-run-root",
                *common,
            ]
        )

    for position_filter in ["visual_only", "answer_adjacent_only"]:
        _run(
            [
                "--pack",
                "primary",
                "--mode",
                "full",
                "--tag",
                f"expanded_v2_L26_top16_{position_filter}",
                "--layer",
                "26",
                "--max-feature-nodes",
                "128",
                "--top-features-per-sample",
                "16",
                "--position-filter",
                position_filter,
                "--cleanup-run-root",
                *common,
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
